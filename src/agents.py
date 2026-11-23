"""
Système multi-agent de sélection de portefeuille.

Architecture « tableau noir » (blackboard) : chaque agent spécialisé lit les
informations dont il a besoin sur un espace partagé, y publie ses conclusions,
et l'orchestrateur coordonne l'ordre d'intervention ainsi que la boucle de
négociation entre l'agent Allocation et l'agent Risque.

    Client ──► [Agent Profil] ─┐
    Marché ──► [Agent Régime] ─┤
    Panel  ──► [Agent Prévision]┼──► [Agent Allocation] ⇄ [Agent Risque] ──► [Agent Explication]
    Prix   ──► [Agent Risque]  ─┘            (boucle : si contrainte violée, on réduit la cible de risque)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .questionnaire import PROFILE_ORDER, has_answers, most_prudent, score_questionnaire
from .config import (ASSET_CLASS, ASSET_NAME, DYNAMIC_RULE, LONG_BOND_TICKERS, LOOKBACK_COV,
                     PROFILE_BENCHMARK, PROFILE_CLASSES, PROFILE_CONSTRAINTS, REGIME_VOL_MULTIPLIER,
                     TURNOVER_PENALTY)


# ---------------------------------------------------------------------------
# Infrastructure de communication
# ---------------------------------------------------------------------------
@dataclass
class Message:
    sender: str
    key: str
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class Blackboard:
    """Espace de travail partagé entre les agents."""

    def __init__(self):
        self.data: dict = {}
        self.messages: list[Message] = []

    def post(self, sender: str, key: str, value, summary: str = ""):
        self.data[key] = value
        self.messages.append(Message(sender, key, summary or str(key)))

    def get(self, key, default=None):
        return self.data.get(key, default)

    def log(self) -> pd.DataFrame:
        return pd.DataFrame([m.__dict__ for m in self.messages])


class BaseAgent:
    name = "Agent"

    def __init__(self, blackboard: Blackboard | None = None):
        self.bb = blackboard

    def post(self, key, value, summary=""):
        if self.bb is not None:
            self.bb.post(self.name, key, value, summary)


# ---------------------------------------------------------------------------
# 1. Agent Profil investisseur (classification supervisée, données SCF)
# ---------------------------------------------------------------------------
class InvestorProfileAgent(BaseAgent):
    name = "Agent Profil"

    def __init__(self, bundle: dict, blackboard=None):
        super().__init__(blackboard)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["num"] + bundle["cat"]
        self.cat = bundle["cat"]
        self.classes = list(bundle.get("classes", PROFILE_CLASSES))
        self.mode = bundle.get("mode", "3classes")
        self.threshold = float(bundle.get("threshold", 0.5))

    def run(self, client: dict | None = None, forced_profile: str | None = None):
        if forced_profile is not None:
            proba = {c: float(c == forced_profile) for c in PROFILE_CLASSES}
            self.post("profile", forced_profile, f"Profil imposé : {forced_profile}")
            self.post("profile_proba", proba)
            return forced_profile, proba
        client = client or {}
        row = {f: client.get(f, np.nan) for f in self.features}
        X = pd.DataFrame([row])
        for c in self.cat:
            X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(int(v)))
        p = self.pipeline.predict_proba(X)[0]
        labels = [self.classes[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                  for c in self.pipeline.classes_]
        proba = {labels[i]: float(p[i]) for i in range(len(labels))}

        # --- Capacité à supporter des pertes : estimée par le modèle ML -------------
        if self.mode == "binaire":
            p_prud = proba.get("Prudent", 0.0)
            capacity = "Prudent" if p_prud >= self.threshold else None   # None = pas de plafond
            cap_txt = (f"profil financier proche des ménages prudents (p={p_prud:.2f} ≥ seuil "
                       f"{self.threshold:.2f})" if capacity else
                       f"pas de contrainte détectée (p(prudent)={p_prud:.2f})")
        else:
            capacity = max(proba, key=proba.get)
            cap_txt = f"profil prédit {capacity} (p={proba[capacity]:.2f})"

        # --- Tolérance déclarée : questionnaire noté ---------------------------
        if has_answers(client):
            q = score_questionnaire(client)
            declared = q["profile"]
            decl_txt = f"questionnaire {q['score']}/{q['score_max']} → {declared}"
        else:   # repli : deux questions simples (horizon, perte acceptable)
            q = None
            horizon = client.get("HORIZON_ANS", np.nan)
            perte = client.get("PERTE_ACCEPTABLE_PCT", np.nan)
            dyn = (pd.notna(horizon) and horizon >= DYNAMIC_RULE["horizon_min_ans"]
                   and pd.notna(perte) and perte >= DYNAMIC_RULE["perte_acceptable_min_pct"])
            declared = "Dynamique" if dyn else "Équilibré"
            decl_txt = f"horizon {horizon} ans, perte acceptable {perte} % → {declared}"

        # --- Décision : le plus prudent des deux + alerte en cas de divergence ---
        profile = most_prudent(declared, capacity) if capacity else declared
        alert = None
        if capacity and PROFILE_ORDER.index(declared) > PROFILE_ORDER.index(capacity):
            alert = (f"Incohérence : le client se déclare {declared} mais sa situation financière "
                     f"ressemble à celle des ménages prudents. Profil ramené à {profile} – "
                     f"à valider par le conseiller.")
        detail = {"declare": declared, "capacite": capacity or "aucun plafond", "final": profile,
                  "questionnaire": q, "proba_ml": proba, "alerte": alert}
        self.post("profile_detail", detail, f"Tolérance déclarée : {decl_txt} | Capacité (ML) : {cap_txt}")
        if alert:
            self.post("profile_alert", alert, alert)
        self.post("profile", profile, f"Profil retenu : {profile}")
        self.post("profile_proba", proba)
        return profile, proba


# ---------------------------------------------------------------------------
# 2. Agent Régime de marché (apprentissage non supervisé – mélange gaussien)
# ---------------------------------------------------------------------------
class MarketRegimeAgent(BaseAgent):
    name = "Agent Régime"

    def __init__(self, bundle: dict, blackboard=None):
        super().__init__(blackboard)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]
        self.fill = bundle["fill"]
        self.label_map = bundle["label_map"]

    def run(self, market_row: pd.Series):
        x = market_row.reindex(self.features).astype(float).fillna(pd.Series(self.fill))
        z = self.scaler.transform(pd.DataFrame([x.values], columns=self.features))
        p = self.model.predict_proba(z)[0]
        proba = {}
        for k, v in enumerate(p):
            lab = self.label_map[k]
            proba[lab] = proba.get(lab, 0.0) + float(v)
        regime = max(proba, key=proba.get)
        self.post("regime", regime, f"Régime détecté : {regime} (p={proba[regime]:.2f})")
        self.post("regime_proba", proba)
        return regime, proba


# ---------------------------------------------------------------------------
# 3. Agent Prévision (classification supervisée de la surperformance)
# ---------------------------------------------------------------------------
class ForecastAgent(BaseAgent):
    """Transforme les probabilités de surperformance en rendements espérés
    excédentaires via la « loi fondamentale » de Grinold & Kahn :
        alpha_i = IC × σ_i × z_i
    où z_i est le score standardisé transversalement, σ_i la volatilité de
    l'actif et IC le coefficient d'information mesuré en validation. Si le
    modèle n'a pas de pouvoir prédictif (IC ≤ 0), alpha = 0 et l'allocation
    devient purement fondée sur le risque : garde-fou contre le sur-apprentissage.
    """
    name = "Agent Prévision"

    def __init__(self, bundle: dict, blackboard=None):
        super().__init__(blackboard)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]
        self.ic = max(float(bundle.get("ic", 0.0)), 0.0)

    def run(self, panel_t: pd.DataFrame) -> pd.DataFrame:
        X = panel_t[self.features]
        out = pd.DataFrame(index=panel_t.index)
        if hasattr(self.pipeline, "predict_proba"):          # classifieur : probabilité de surperformance
            score = self.pipeline.predict_proba(X)[:, 1]
        else:                                                 # régresseur : rendement excédentaire prédit à 21 j
            score = self.pipeline.predict(X)
            out["prevision"] = score
        out["score"] = score
        out["rang"] = out["score"].rank(pct=True)             # position parmi les actifs (1 = le mieux noté)
        out["proba"] = out["rang"] if "prevision" in out else out["score"]
        sd = out["score"].std()
        out["z"] = (out["score"] - out["score"].mean()) / (sd if sd > 0 else 1.0)
        out["vol"] = panel_t["vol_63"].clip(lower=0.01)
        out["alpha"] = self.ic * out["vol"] * out["z"]
        top = out["score"].sort_values(ascending=False).head(3).index.tolist()
        etat = "signal utilisé" if self.ic > 0 else "signal non utilisé (α = 0, garde-fou)"
        self.post("forecast", out, f"Prévisions pour {len(out)} actifs – favoris : {', '.join(top)} – {etat}")
        return out


# ---------------------------------------------------------------------------
# 4. Agent Risque (estimation et contrôle)
# ---------------------------------------------------------------------------
class RiskAgent(BaseAgent):
    name = "Agent Risque"

    def __init__(self, lookback: int = LOOKBACK_COV, blackboard=None):
        super().__init__(blackboard)
        self.lookback = lookback

    def estimate(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        """Covariance annualisée par rétrécissement de Ledoit-Wolf (plus stable
        que la covariance empirique quand le nombre d'actifs est élevé)."""
        R = returns_window.dropna(how="any")
        lw = LedoitWolf().fit(R.values)
        cov = pd.DataFrame(lw.covariance_ * 252, index=R.columns, columns=R.columns)
        self.post("cov", cov, f"Covariance Ledoit-Wolf estimée sur {len(R)} jours "
                              f"(rétrécissement={lw.shrinkage_:.2f})")
        return cov

    def evaluate(self, w: pd.Series, cov: pd.DataFrame, returns_window: pd.DataFrame) -> dict:
        w = w.reindex(cov.index).fillna(0)
        vol = float(np.sqrt(w @ cov @ w))
        port = returns_window[cov.index].fillna(0) @ w
        q = port.quantile(0.05)
        wealth = (1 + port).cumprod()
        exposures = w.groupby(w.index.map(ASSET_CLASS)).sum().to_dict()
        return {
            "vol_ex_ante": vol,
            "VaR95_1mois_param": 1.645 * vol * np.sqrt(21 / 252),
            "CVaR95_jour_hist": float(-port[port <= q].mean()),
            "max_drawdown_fenetre": float((wealth / wealth.cummax() - 1).min()),
            "HHI_concentration": float((w ** 2).sum()),
            "nb_actifs_>1%": int((w > 0.01).sum()),
            "expositions": exposures,
        }

    def check(self, metrics: dict, constraints: dict, target_vol: float):
        violations = []
        if metrics["vol_ex_ante"] > target_vol * 1.05:
            violations.append(f"volatilité {metrics['vol_ex_ante']:.1%} > cible {target_vol:.1%}")
        exp = metrics["expositions"]
        eq = exp.get("Actions", 0) + exp.get("Immobilier", 0)
        if eq > constraints["equity_max"] + 1e-3:
            violations.append(f"actions {eq:.1%} > {constraints['equity_max']:.0%}")
        defensive = exp.get("Obligations", 0) + exp.get("Monétaire", 0)
        if defensive < constraints["defensive_min"] - 1e-3:
            violations.append(f"obligations + monétaire {defensive:.1%} < {constraints['defensive_min']:.0%}")
        if exp.get("Crypto", 0) > constraints["crypto_max"] + 1e-3:
            violations.append("exposition crypto excessive")
        ok = not violations
        self.post("risk_check", {"ok": ok, "violations": violations},
                  "Contrôle des risques : OK" if ok else "Violations : " + "; ".join(violations))
        return ok, violations


# ---------------------------------------------------------------------------
# 5. Agent Allocation (optimisation moyenne-variance sous contraintes)
# ---------------------------------------------------------------------------
class AllocationAgent(BaseAgent):
    """Résout  max_w  wᵀα − (λ/2)·wᵀΣw − (γ/2)·‖w − w_prec‖²
       s.c.  Σw = 1, 0 ≤ w_i ≤ w_max, contraintes par classe d'actifs,
             wᵀΣw ≤ σ_cible²  (budget de risque du profil)."""
    name = "Agent Allocation"

    def __init__(self, turnover_penalty: float = TURNOVER_PENALTY, blackboard=None):
        super().__init__(blackboard)
        self.gamma = turnover_penalty

    @staticmethod
    def _masks(tickers):
        cls = np.array([ASSET_CLASS[t] for t in tickers])
        return {"equity": np.isin(cls, ["Actions", "Immobilier"]),
                "defensive": np.isin(cls, ["Obligations", "Monétaire"]),
                "cash": cls == "Monétaire", "long_bond": np.isin(tickers, LONG_BOND_TICKERS),
                "commo": cls == "Matières premières", "crypto": cls == "Crypto"}

    def optimize(self, alpha: pd.Series, cov: pd.DataFrame, constraints: dict,
                 target_vol: float, risk_aversion: float, prev_w: pd.Series | None = None) -> pd.Series:
        tickers = list(cov.index)
        S = cov.values
        a = alpha.reindex(tickers).fillna(0).values
        n = len(tickers)
        m = self._masks(tickers)
        # Pas de portefeuille précédent (première allocation) : aucune pénalité de rotation
        gamma = self.gamma if prev_w is not None else 0.0
        w_prev = prev_w.reindex(tickers).fillna(0).values if prev_w is not None else np.zeros(n)

        bounds = []
        for i, t in enumerate(tickers):
            if m["crypto"][i]:
                ub = constraints["crypto_max"]
            elif m["cash"][i]:
                ub = constraints["cash_max"]
            elif m["long_bond"][i]:
                ub = min(constraints["w_max"], constraints["long_bond_max"])
            else:
                ub = constraints["w_max"]
            bounds.append((0.0, ub))
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: constraints["equity_max"] - w[m["equity"]].sum()},
            {"type": "ineq", "fun": lambda w: w[m["defensive"]].sum() - constraints["defensive_min"]},
            {"type": "ineq", "fun": lambda w: constraints["commodity_max"] - w[m["commo"]].sum()},
            {"type": "ineq", "fun": lambda w: target_vol ** 2 - w @ S @ w},
        ]

        def obj(w):
            d = w - w_prev
            return -(w @ a) + 0.5 * risk_aversion * (w @ S @ w) + 0.5 * gamma * (d @ d)

        def grad(w):
            return -a + risk_aversion * (S @ w) + gamma * (w - w_prev)

        x0 = np.array([b[1] for b in bounds]); x0 = x0 / x0.sum()
        res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-10})
        status = "optimisation moyenne-variance"
        if not res.success:
            # Repli : portefeuille de variance minimale sous les mêmes contraintes (hors cible de vol)
            res = minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, bounds=bounds,
                           constraints=cons[:4], method="SLSQP", options={"maxiter": 500})
            status = "repli variance minimale"
        w = np.clip(res.x, 0, None)
        w = pd.Series(w / w.sum(), index=tickers)
        w[w < 1e-4] = 0
        w = w / w.sum()
        self.post("weights", w, f"Allocation calculée ({status}) – {int((w > 0.01).sum())} lignes")
        return w


# ---------------------------------------------------------------------------
# 6. Agent Explication (rapport en langage naturel)
# ---------------------------------------------------------------------------
class ExplanationAgent(BaseAgent):
    name = "Agent Explication"

    def __init__(self, top_features: list[str] | None = None, blackboard=None):
        super().__init__(blackboard)
        self.top_features = top_features or []

    def run(self, date) -> str:
        bb = self.bb
        w = bb.get("weights")
        fc = bb.get("forecast")
        met = bb.get("risk_metrics")
        profile, regime = bb.get("profile"), bb.get("regime")
        pp, rp = bb.get("profile_proba", {}), bb.get("regime_proba", {})
        lines = [f"### Recommandation au {pd.Timestamp(date):%d/%m/%Y}", ""]
        det = bb.get("profile_detail")
        if det:
            q = det.get("questionnaire")
            decl = (f"questionnaire {q['score']}/{q['score_max']} → {det['declare']}" if q
                    else det["declare"])
            lines.append(f"**Profil investisseur retenu** : {profile} – tolérance déclarée : {decl} ; "
                         f"capacité estimée par le modèle : {det['capacite']}"
                         + (f" (probabilité de profil prudent {pp['Prudent']:.0%})" if "Prudent" in pp else "")
                         + ".")
            if det.get("alerte"):
                lines.append(f"⚠️ **{det['alerte']}**")
        elif "Prudent" in pp and "Non prudent" in pp:
            lines.append(f"**Profil investisseur** : {profile} (probabilité d'aversion au risque "
                         f"estimée par le modèle : {pp['Prudent']:.0%}).")
        else:
            lines.append(f"**Profil investisseur** : {profile} (confiance {pp.get(profile, 1):.0%}).")
        lines.append(f"**Régime de marché** : {regime} (probabilité {rp.get(regime, 1):.0%}) – "
                     f"la cible de volatilité est ajustée d'un facteur "
                     f"{bb.get('vol_multiplier', REGIME_VOL_MULTIPLIER.get(regime, 1)):.2f}.")
        exp = met["expositions"]
        lines.append("**Répartition par classe** : " + ", ".join(
            f"{k} {v:.0%}" for k, v in sorted(exp.items(), key=lambda x: -x[1]) if v > 0.005) + ".")
        lines.append(f"**Risque ex ante** : volatilité annuelle {met['vol_ex_ante']:.1%}, "
                     f"VaR 95 % à 1 mois {met['VaR95_1mois_param']:.1%}, "
                     f"pire baisse sur la fenêtre d'estimation {met['max_drawdown_fenetre']:.1%}.")
        lines.append("")
        lines.append("**Principales positions :**")
        for t, wt in w.sort_values(ascending=False).head(6).items():
            if wt < 0.01:
                break
            if fc is not None and t in fc.index and "prevision" in fc:
                rg, pv = fc.loc[t, "rang"], fc.loc[t, "prevision"]
                avis = "signal favorable" if rg > 0.7 else ("signal défavorable" if rg < 0.3 else "signal neutre")
                lines.append(f"- {t} ({ASSET_NAME.get(t, t)}) : {wt:.1%} – rendement excédentaire prévu "
                             f"{pv:+.1%} à 1 mois ({avis}).")
            else:
                p = fc.loc[t, "proba"] if fc is not None and t in fc.index else np.nan
                avis = "signal favorable" if p > 0.55 else ("signal défavorable" if p < 0.45 else "signal neutre")
                lines.append(f"- {t} ({ASSET_NAME.get(t, t)}) : {wt:.1%} – probabilité de "
                             f"surperformance {p:.0%} ({avis}).")
        if self.top_features:
            lines.append("")
            lines.append("**Variables les plus influentes du modèle de prévision** : "
                         + ", ".join(self.top_features[:5]) + ".")
        lines.append("")
        lines.append("_Outil d'aide à la décision à visée pédagogique – ne constitue pas un conseil "
                     "en investissement._")
        text = "\n".join(lines)
        self.post("explanation", text, "Rapport rédigé")
        return text


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------
class PortfolioOrchestrator:
    def __init__(self, profile_bundle=None, regime_bundle=None, forecast_bundle=None,
                 lookback: int = LOOKBACK_COV, max_negotiation_rounds: int = 3,
                 turnover_penalty: float = TURNOVER_PENALTY):
        self.profile_bundle = profile_bundle
        self.regime_bundle = regime_bundle
        self.forecast_bundle = forecast_bundle
        self.lookback = lookback
        self.max_rounds = max_negotiation_rounds
        self.turnover_penalty = (forecast_bundle or {}).get("turnover_penalty", turnover_penalty)
        # Réduction du risque par régime : valeur choisie en validation si elle est enregistrée
        # avec le modèle de régime, sinon valeur par défaut de config.py
        self.regime_mult = (regime_bundle or {}).get("vol_multiplier", REGIME_VOL_MULTIPLIER)

    @staticmethod
    def benchmark_weights(profile: str, tickers: list) -> pd.Series:
        """Poids de référence par actif : poids de la classe / nb d'actifs disponibles de la classe."""
        cls = pd.Series({t: ASSET_CLASS[t] for t in tickers})
        bm = PROFILE_BENCHMARK[profile]
        w = pd.Series(0.0, index=tickers)
        for c, wc in bm.items():
            members = cls[cls == c].index
            if len(members):
                w[members] = wc / len(members)
        return w / w.sum()

    def recommend(self, date, prices: pd.DataFrame, panel: pd.DataFrame,
                  market_features: pd.DataFrame, client: dict | None = None,
                  profile: str | None = None, prev_w: pd.Series | None = None,
                  explain: bool = True) -> dict:
        date = pd.Timestamp(date)
        bb = Blackboard()

        # 1) Profil
        if self.profile_bundle is not None and profile is None:
            prof, _ = InvestorProfileAgent(self.profile_bundle, bb).run(client or {})
        else:
            prof, _ = InvestorProfileAgent({"pipeline": None, "num": [], "cat": []}, bb) \
                .run(forced_profile=profile or "Équilibré")
        cons = PROFILE_CONSTRAINTS[prof]

        # 2) Régime
        if self.regime_bundle is not None:
            mrow = market_features.loc[:date].iloc[-1]
            regime, _ = MarketRegimeAgent(self.regime_bundle, bb).run(mrow)
        else:
            regime = "Neutre"
            bb.post("Agent Régime", "regime", regime, "Régime par défaut")
        mult = self.regime_mult.get(regime, 1.0)
        bb.post("Orchestrateur", "vol_multiplier", mult, f"Facteur de risque du régime {regime} : {mult:.2f}")
        target_vol = cons["target_vol"] * mult
        lam = cons["risk_aversion"] / mult

        # 3) Univers disponible + prévisions
        panel_t = panel.xs(date, level="date") if date in panel.index.get_level_values("date") \
            else panel.loc[panel.index.get_level_values("date") <= date].groupby(level="ticker").tail(1) \
            .droplevel("date")
        rets = prices.pct_change(fill_method=None).loc[:date].tail(self.lookback)
        enough = rets.notna().mean() >= 0.9
        tickers = [t for t in panel_t.index if enough.get(t, False)]
        panel_t = panel_t.loc[tickers]
        if self.forecast_bundle is not None:
            fc = ForecastAgent(self.forecast_bundle, bb).run(panel_t)
            alpha = fc["alpha"]
        else:
            alpha = pd.Series(0.0, index=tickers)

        # 4) Risque + allocation, avec boucle de négociation
        risk = RiskAgent(self.lookback, bb)
        cov = risk.estimate(rets[tickers])
        # Rendements d'équilibre implicites de l'allocation stratégique du profil (Black-Litterman) :
        # π = λ·Σ·w_ref. Sans signal ML (α = 0), avec l'aversion de base (régime Calme), sans
        # contrainte de volatilité active ni pénalité de rotation, l'optimum est exactement w_ref
        # (vérifié dans le notebook, section 7). En régime Neutre/Stress, l'aversion augmente et la
        # cible de volatilité baisse : le portefeuille s'écarte volontairement de w_ref vers moins
        # de risque. L'agent Prévision incline ensuite le portefeuille autour de ce point.
        w_ref = self.benchmark_weights(prof, list(cov.index))
        pi = cons["risk_aversion"] * cov.values @ w_ref.values
        mu = pd.Series(pi, index=cov.index) + alpha.reindex(cov.index).fillna(0)
        bb.post("Orchestrateur", "benchmark", w_ref,
                "Allocation stratégique de référence : " + ", ".join(
                    f"{k} {v:.0%}" for k, v in PROFILE_BENCHMARK[prof].items()))
        alloc = AllocationAgent(turnover_penalty=self.turnover_penalty, blackboard=bb)
        for rnd in range(self.max_rounds):
            w = alloc.optimize(mu, cov, cons, target_vol, lam, prev_w)
            metrics = risk.evaluate(w, cov, rets[cov.index])
            bb.post("Agent Risque", "risk_metrics", metrics,
                    f"Vol ex ante {metrics['vol_ex_ante']:.1%} (tour {rnd + 1})")
            ok, _ = risk.check(metrics, cons, target_vol)
            if ok:
                break
            target_vol *= 0.9
            bb.post("Orchestrateur", "target_vol", target_vol,
                    f"Renégociation : nouvelle cible de volatilité {target_vol:.1%}")

        text = ExplanationAgent(self.forecast_bundle.get("top_features") if self.forecast_bundle else None,
                                bb).run(date) if explain else ""
        return {"date": date, "profile": prof, "regime": regime, "weights": w,
                "profile_proba": bb.get("profile_proba"), "regime_proba": bb.get("regime_proba"),
                "profile_detail": bb.get("profile_detail"),
                "metrics": metrics, "explanation": text, "log": bb.log(),
                "forecast": bb.get("forecast")}
