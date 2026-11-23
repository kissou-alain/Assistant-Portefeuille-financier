"""
Logique de l'API : chargement des artefacts produits par le notebook et appels aux agents.
Aucun modèle n'est entraîné ici : l'API ne fait que charger et appliquer.
"""
from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (ASSET_CLASS, ASSET_NAME, CURRENCIES, DATA_PROC, FIG_DIR, LOOKBACK_COV,   # noqa: E402
                        MODELS_DIR, PROFILE_BENCHMARK, PROFILE_CLASSES, PROFILE_CONSTRAINTS, RESULTS_DIR,
                        SCF_LABELS, UNIVERSE)
from src.agents import Blackboard, InvestorProfileAgent, PortfolioOrchestrator   # noqa: E402
from src.allocation import ledoit_wolf_cov   # noqa: E402
from src.anomalies import detect   # noqa: E402
from src.explain import local_explanation, make_explainer, shap_values   # noqa: E402
from src.questionnaire import QUESTIONS, score_questionnaire   # noqa: E402
from src import data_update   # noqa: E402

DISCLAIMER = "Outil pédagogique : les recommandations ne constituent pas un conseil en investissement."


# ---------------------------------------------------------------------------
# Conversion en JSON (NaN → null, numpy → types Python)
# ---------------------------------------------------------------------------
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def records(df: pd.DataFrame, index_name: str = "nom") -> list:
    d = df.copy()
    if isinstance(d.index, pd.MultiIndex):
        d.index = [" / ".join(map(str, i)) for i in d.index]
    d.index.name = index_name
    return clean(d.reset_index().to_dict(orient="records"))


# ---------------------------------------------------------------------------
# État de l'application (chargé au démarrage, rechargé après mise à jour des données)
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.load_models()
        self.load_data()

    def load_models(self):
        self.profile = joblib.load(MODELS_DIR / "agent_profil.joblib")
        self.regime = joblib.load(MODELS_DIR / "agent_regime.joblib")
        self.forecast = joblib.load(MODELS_DIR / "agent_prevision.joblib")
        self.alert = joblib.load(MODELS_DIR / "agent_alerte.joblib")
        meta_file = MODELS_DIR / "metadata.json"
        self.meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        self.explainer = make_explainer(self.profile["pipeline"], self.profile["background"], n_background=100)

    def load_data(self):
        self.prices = pd.read_csv(DATA_PROC / "prices_clean.csv", index_col=0, parse_dates=True)
        self.returns = self.prices.pct_change(fill_method=None)
        self.panel = pd.read_csv(DATA_PROC / "panel_recent.csv", parse_dates=["date"]).set_index(["date", "ticker"])
        self.mkt = pd.read_csv(DATA_PROC / "market_features.csv", index_col=0, parse_dates=True)
        self.last_date = self.panel.index.get_level_values("date").max()
        # probabilités de régime recalculées à partir des variables de marché (reflète les mises à jour)
        rb = self.regime
        M = self.mkt[rb["features"]].ffill().fillna(pd.Series(rb["fill"]))
        P = rb["model"].predict_proba(rb["scaler"].transform(M.dropna()))
        proba = pd.DataFrame(P, index=M.dropna().index).rename(columns=rb["label_map"])
        self.regime_proba = proba.T.groupby(level=0).sum().T.reindex(columns=["Calme", "Neutre", "Stress"])
        self.orchestrator = PortfolioOrchestrator(self.profile, self.regime, self.forecast, alert_bundle=self.alert)


STATE: State | None = None


def state() -> State:
    global STATE
    if STATE is None:
        STATE = State()
    return STATE


# ---------------------------------------------------------------------------
# Client : formulaire → variables du SCF (montants convertis en dollars US)
# ---------------------------------------------------------------------------
SITUATIONS = {"Salarié": 1, "Indépendant / chef d'entreprise": 2, "Retraité": 3, "Sans activité": 4}
ETUDES = ["Sans diplôme du secondaire", "Diplôme du secondaire", "Études supérieures courtes", "Diplôme universitaire"]


def build_client(f: dict) -> dict:
    per_usd = float(f.get("taux_change") or CURRENCIES.get(f.get("devise", "USD"), {}).get("per_usd", 1.0))
    income, assets, debt = (float(f.get(k, 0)) / per_usd for k in ("revenu", "actifs", "dettes"))
    age, kids = int(f.get("age", 40)), int(f.get("enfants", 0))
    married = bool(f.get("en_couple", True))
    situation = f.get("situation", "Salarié")
    lifecl = (1 if age < 55 and not married and kids == 0 else 2 if age < 55 and married and kids > 0
              else 3 if age < 55 and married else 4 if age < 55 else 5 if situation != "Retraité" else 6)
    client = {
        "AGE": age, "KIDS": kids, "INCOME": income, "NORMINC": income, "ASSET": assets, "DEBT": debt,
        "NETWORTH": assets - debt, "LEVRATIO": debt / assets if assets > 0 else 0,
        "DEBT2INC": debt / income if income > 0 else 0,
        "WAGEINC": income if situation == "Salarié" else 0,
        "KNOWL": int(f.get("connaissance", 5)), "FINLIT": int(f.get("quiz", 2)),
        "HHSEX": 1 if f.get("sexe", "Homme") == "Homme" else 2,
        "EDCL": ETUDES.index(f.get("etudes", ETUDES[-1])) + 1 if f.get("etudes") in ETUDES else 4,
        "MARRIED": 1 if married else 2, "OCCAT1": SITUATIONS.get(situation, 1), "LIFECL": lifecl,
        "HOUSECL": 1 if f.get("proprietaire", True) else 2,
        "SAVED": 1 if f.get("a_epargne", True) else 0, "WSAVED": 3 if f.get("a_epargne", True) else 1,
    }
    for k, v in (f.get("reponses") or {}).items():
        if k in QUESTIONS:
            client[k] = int(v)
    return client


def _profile_row(client: dict) -> pd.DataFrame:
    b = state().profile
    X = pd.DataFrame([{c: client.get(c, np.nan) for c in b["num"] + b["cat"]}])
    for c in b["cat"]:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    return X


def questionnaire() -> dict:
    return {"questions": [{"cle": k, "intitule": q, "options": opts} for k, (q, opts) in QUESTIONS.items()],
            "devises": [{"code": c, **v} for c, v in CURRENCIES.items()],
            "situations": list(SITUATIONS), "etudes": ETUDES, "avertissement": DISCLAIMER}


# ---------------------------------------------------------------------------
# Profil
# ---------------------------------------------------------------------------
def profile(form: dict) -> dict:
    s = state()
    client = build_client(form)
    bb = Blackboard()
    prof, proba = InvestorProfileAgent(s.profile, bb).run(client)
    det = bb.get("profile_detail")
    X = _profile_row(client)
    sv = shap_values(s.profile["pipeline"], s.explainer, X, class_idx=0).iloc[0]
    expl = local_explanation(sv, X.iloc[0], labels=SCF_LABELS, top=8)
    q = det.get("questionnaire") or score_questionnaire(client)
    return clean({
        "profil": prof, "declare": det["declare"], "capacite": det["capacite"], "alerte": det["alerte"],
        "proba_prudent": proba.get("Prudent"), "seuil": s.profile["threshold"],
        "questionnaire": {"score": q["score"], "score_max": q["score_max"], "plafonds": q["plafonds"],
                          "detail": q["detail"]},
        "explication": {"valeur_de_base": sv.attrs.get("base_value"),
                        "contributions": expl[["libellé", "contribution", "effet"]].to_dict(orient="records"),
                        "unite": "log-odds de « Prudent » (positif = rend plus prudent)"},
        "contraintes": PROFILE_CONSTRAINTS[prof], "modele": s.profile.get("model_name"),
    })


# ---------------------------------------------------------------------------
# Recommandation
# ---------------------------------------------------------------------------
def _frontier(cov: pd.DataFrame, mu: pd.Series, n: int = 14) -> list:
    """Frontière efficiente long-only (Σw = 1, 0 ≤ w ≤ 25 %) pour les rendements d'équilibre μ."""
    S, m = cov.values, mu.reindex(cov.index).values
    k = len(m)
    bounds = [(0, 0.25)] * k
    base = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.full(k, 1 / k)
    w_min = minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, bounds=bounds, constraints=base,
                     method="SLSQP").x
    r_lo, r_hi = float(w_min @ m), float(np.sort(m)[-4:].mean())
    pts = []
    for r in np.linspace(r_lo, r_hi, n):
        cons = base + [{"type": "eq", "fun": lambda w, r=r: w @ m - r}]
        res = minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 200})
        if res.success:
            pts.append({"volatilite": float(np.sqrt(res.x @ S @ res.x)), "rendement": float(r)})
    return pts


def _projection(mean: float, vol: float, montant: float, annees: int = 10) -> list:
    """Projection illustrative de la valeur (loi log-normale) : quantiles 5 %, 25 %, 50 %, 75 %, 95 %."""
    out = []
    for t in range(annees + 1):
        row = {"annee": t}
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            row[f"q{int(q * 100)}"] = montant * math.exp((mean - vol ** 2 / 2) * t + norm.ppf(q) * vol * math.sqrt(t))
        out.append(row)
    return out


def recommendation(form: dict, forced_profile: str | None = None, montant: float = 10_000) -> dict:
    s = state()
    client = build_client(form)
    with s.lock:
        res = s.orchestrator.recommend(s.last_date, s.prices, s.panel, s.mkt, client=client,
                                       profile=forced_profile if forced_profile in PROFILE_CLASSES else None)
    w = res["weights"]
    w = w[w > 0.001].sort_values(ascending=False)
    fc = res.get("forecast")
    lignes = []
    for t, wt in w.items():
        lignes.append({"ticker": t, "nom": ASSET_NAME.get(t, t), "classe": ASSET_CLASS.get(t), "poids": wt,
                       "prevision": fc.loc[t, "prevision"] if fc is not None and "prevision" in fc and t in fc.index else None,
                       "rang": int(fc["score"].rank(ascending=False)[t]) if fc is not None and t in fc.index else None,
                       "rendement_1an": s.prices[t].iloc[-1] / s.prices[t].iloc[-253] - 1})
    # risque, frontière et projection
    R = s.returns.loc[:s.last_date].tail(LOOKBACK_COV)
    cov = ledoit_wolf_cov(R)
    prof = res["profile"]
    # Rendements excédentaires attendus, hypothèse de marché COMMUNE à tous les profils (Black-Litterman) :
    # π = δ · Σ · w_marché, avec δ = 2,5 (aversion au risque du marché) et un portefeuille de marché
    # approché par le 1/N. (L'optimiseur utilise, lui, les rendements implicites propres au profil.)
    w_mkt = pd.Series(1 / len(cov), index=cov.index)
    pi = pd.Series(2.5 * cov.values @ w_mkt.values, index=cov.index)
    wv = res["weights"].reindex(cov.index).fillna(0)
    rf = float(s.mkt.get("DTB3", pd.Series([4.0])).dropna().iloc[-1]) / 100 if "DTB3" in s.mkt else 0.04
    exp_ret = rf + float(wv @ pi)
    met = res["metrics"]
    actifs = [{"ticker": t, "volatilite": float(np.sqrt(cov.loc[t, t])), "rendement": float(pi[t]),
               "classe": ASSET_CLASS.get(t)} for t in cov.index]
    return clean({
        "date": s.last_date, "profil": prof, "profil_detail": {k: v for k, v in (res["profile_detail"] or {}).items()
                                                                  if k in ("declare", "capacite", "final", "alerte")},
        "regime": res["regime"], "regime_proba": res["regime_proba"], "alerte_marche": res.get("alert"),
        "lignes": lignes,
        "classes": {k: v for k, v in met["expositions"].items() if v > 0.001},
        "risque": {"volatilite": met["vol_ex_ante"], "var95_1mois": met["VaR95_1mois_param"],
                   "cvar95_jour": met["CVaR95_jour_hist"], "perte_max_fenetre": met["max_drawdown_fenetre"],
                   "cible_volatilite": PROFILE_CONSTRAINTS[prof]["target_vol"],
                   "nb_lignes": met["nb_actifs_>1%"]},
        "frontiere": {"points": _frontier(cov, pi), "actifs": actifs,
                      "portefeuille": {"volatilite": met["vol_ex_ante"], "rendement": float(wv @ pi)},
                      "note": "Rendements d'équilibre du marché (Black-Litterman, δ = 2,5), excédentaires au taux sans risque"},
        "projection": {"montant": montant, "rendement_espere": exp_ret, "volatilite": met["vol_ex_ante"],
                       "points": _projection(exp_ret, met["vol_ex_ante"], montant)},
        "prevision_active": bool(s.forecast.get("active", False)),
        "explication": res["explanation"],
        "journal": clean(res["log"].to_dict(orient="records")),
        "devise_note": "Actifs cotés en dollars US : performances et risques en USD, hors effet de change.",
        "avertissement": DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# Marché
# ---------------------------------------------------------------------------
def market(annees: int = 3) -> dict:
    s = state()
    start = s.last_date - pd.DateOffset(years=annees)
    rp = s.regime_proba.loc[start:]
    reg = rp.idxmax(axis=1)
    spy = s.prices["SPY"].reindex(rp.index)
    hist = pd.DataFrame({"spy": spy, "regime": reg}).iloc[::2]       # un point sur deux (allège l'envoi)
    feats = s.alert["features"]
    M = s.mkt[feats].loc[start:].dropna()
    fl = detect(s.alert["detectors"], M)
    alertes = fl[fl["nb détecteurs"] >= s.alert.get("min_votes", 3)]
    last = s.mkt.ffill().iloc[-1]
    ind = {"VIX": last.get("VIX"), "Pente 10 ans – 2 ans (%)": last.get("T10Y2Y"),
           "Spread de crédit Baa (%)": last.get("BAA10Y"), "S&P 500 sur 1 mois": last.get("mkt_ret_21"),
           "Volatilité S&P 500 (1 mois)": last.get("mkt_vol_21"), "Écart au plus haut sur 1 an": last.get("mkt_drawdown")}
    return clean({
        "date": s.last_date, "regime": reg.iloc[-1], "proba": rp.iloc[-1].to_dict(),
        "historique": [{"date": d, "spy": r["spy"], "regime": r["regime"]} for d, r in hist.iterrows()],
        "alerte_du_jour": bool(fl["nb détecteurs"].iloc[-1] >= s.alert.get("min_votes", 3)),
        "votes_du_jour": int(fl["nb détecteurs"].iloc[-1]),
        "alertes": [{"date": d, "votes": int(v)} for d, v in alertes["nb détecteurs"].tail(30).items()],
        "indicateurs": ind,
        "actifs": [{"ticker": t, "nom": n, "classe": c,
                    "rendement_1m": s.prices[t].iloc[-1] / s.prices[t].iloc[-22] - 1,
                    "rendement_1an": s.prices[t].iloc[-1] / s.prices[t].iloc[-253] - 1,
                    "volatilite_3m": s.returns[t].tail(63).std() * np.sqrt(252)} for t, (n, c) in UNIVERSE.items()],
    })


def prices_series(ticker: str, annees: int = 3) -> dict:
    s = state()
    p = s.prices[ticker].dropna()
    p = p.loc[s.last_date - pd.DateOffset(years=annees):]
    return clean({"ticker": ticker, "nom": ASSET_NAME.get(ticker, ticker),
                  "points": [{"date": d, "prix": v} for d, v in p.items()]})


# ---------------------------------------------------------------------------
# Backtest et laboratoire des modèles (résultats du notebook)
# ---------------------------------------------------------------------------
def _csv(name, index_col=0):
    f = RESULTS_DIR / name
    return pd.read_csv(f, index_col=index_col) if f.exists() else None


def backtest() -> dict:
    r = pd.read_csv(RESULTS_DIR / "backtest_rendements_quotidiens.csv", index_col=0, parse_dates=True)
    wealth = (1 + r.fillna(0)).cumprod().iloc[::5]
    dd = (wealth / wealth.cummax() - 1)
    out = {"series": [{"date": d, **row.to_dict()} for d, row in wealth.iterrows()],
           "drawdowns": [{"date": d, **row.to_dict()} for d, row in dd.iterrows()],
           "strategies": list(r.columns)}
    for key, f, idx in (("performances", "backtest_performances_test.csv", 0),
                        ("risque_comparable", "backtest_risque_comparable.csv", [0, 1]),
                        ("significativite", "backtest_significativite.csv", 0),
                        ("renforcement", "agent_rl_comparaison_test.csv", 0),
                        ("renforcement_significativite", "agent_rl_significativite.csv", 0),
                        ("reglages", "choix_reglages_validation.csv", [0, 1, 2])):
        d = _csv(f, idx)
        if d is not None:
            out[key] = records(d)
    return clean(out)


FIGURES = {
    "07_profil_confusion_roc": "Profil – matrice de confusion et courbe ROC (test)",
    "07c_profil_courbe_apprentissage": "Profil – courbe d'apprentissage (biais / variance)",
    "08a_profil_coefficients": "Profil – coefficients de la régression logistique",
    "09_profil_shap_global": "Profil – SHAP global",
    "10b_acp": "ACP – éboulis des valeurs propres et cercle des corrélations",
    "10c_lasso_chemin": "Prévision – chemin de régularisation du Lasso",
    "10d_prevision_courbe_apprentissage": "Prévision – courbe d'apprentissage",
    "11_prevision_test": "Prévision – coefficient d'information et spread sur le test",
    "13_prevision_shap": "Prévision – SHAP global",
    "15_regime_choix_k": "Régime – choix du nombre de groupes (coude, BIC, DBSCAN)",
    "15b_regime_dendrogramme": "Régime – dendrogramme de la CAH",
    "16_regimes": "Régime – régimes détectés et S&P 500",
    "16b_anomalies": "Alerte – journées anormales détectées",
    "18b_backtest_risque_comparable": "Backtest – comparaison à risque comparable",
    "20_rl_ppo": "Renforcement – agent PPO sur le test",
}


def models_lab() -> dict:
    s = state()
    tables = {}
    for key, f, idx in (("profil_validation", "agent_profil_comparaison_validation.csv", 0),
                        ("profil_desequilibre", "agent_profil_desequilibre.csv", [0, 1]),
                        ("profil_test", "agent_profil_test.csv", 0),
                        ("profil_externe", "agent_profil_validation_externe.csv", 0),
                        ("prevision_validation", "agent_prevision_comparaison_validation.csv", 0),
                        ("prevision_trois_evaluations", "agent_prevision_trois_evaluations.csv", [0, 1]),
                        ("prevision_diebold_mariano", "agent_prevision_diebold_mariano.csv", 0),
                        ("prevision_test", "agent_prevision_test.csv", 0),
                        ("prevision_apport_gamma", "agent_prevision_apport_gamma.csv", 0),
                        ("regime_clustering", "agent_regime_comparaison_clustering.csv", 0),
                        ("regime_validation", "agent_regime_validation.csv", [0, 1])):
        d = _csv(f, idx)
        if d is not None:
            tables[key] = records(d.drop(columns=[c for c in d.columns if c == "Hyperparamètres"], errors="ignore"))
    figures = [{"id": k, "titre": v, "url": f"/figures/{k}.png"} for k, v in FIGURES.items()
               if (FIG_DIR / f"{k}.png").exists()]
    fiches = ROOT / "docs" / "fiches_modeles.md"
    return clean({"meta": s.meta, "tables": tables, "figures": figures,
                  "fiches": fiches.read_text(encoding="utf-8") if fiches.exists() else None})


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def data_status() -> dict:
    s = state()
    return clean({"date_donnees": s.last_date, "date_prix": s.prices.index.max(),
                  "sauvegarde_disponible": data_update.has_backup(), "modeles_entraines_le": s.meta.get("date_donnees")})


def refresh() -> dict:
    s = state()
    new_last = data_update.refresh_market_data()
    with s.lock:
        s.load_data()
    return {"ok": True, "message": f"Données mises à jour jusqu'au {new_last:%d/%m/%Y}.", **data_status()}


def restore() -> dict:
    s = state()
    data_update.restore_data()
    with s.lock:
        s.load_data()
    return {"ok": True, "message": "Données précédentes restaurées.", **data_status()}
