"""
Application Streamlit – Assistant multi-agent de sélection de portefeuille.

Lancement (depuis la racine du projet, APRÈS exécution complète du notebook
qui crée les modèles dans models/ et les données dans data/processed/) :

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (ASSET_CLASS, ASSET_NAME, CURRENCIES, DATA_PROC, MODELS_DIR, PROFILE_CLASSES,
                        PROFILE_CONSTRAINTS, RESULTS_DIR)
from src.agents import PortfolioOrchestrator
from src.questionnaire import QUESTIONS, score_questionnaire

st.set_page_config(page_title="Assistant multi-agent de portefeuille", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    needed = ["agent_profil.joblib", "agent_regime.joblib", "agent_prevision.joblib"]
    missing = [f for f in needed if not (MODELS_DIR / f).exists()]
    if missing:
        return None
    return {
        "profil": joblib.load(MODELS_DIR / "agent_profil.joblib"),
        "regime": joblib.load(MODELS_DIR / "agent_regime.joblib"),
        "prevision": joblib.load(MODELS_DIR / "agent_prevision.joblib"),
        "meta": json.loads((MODELS_DIR / "metadata.json").read_text()) if (MODELS_DIR / "metadata.json").exists() else {},
    }


@st.cache_data
def load_data():
    prices = pd.read_csv(DATA_PROC / "prices_clean.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(DATA_PROC / "panel_recent.csv", parse_dates=["date"]).set_index(["date", "ticker"])
    mkt = pd.read_csv(DATA_PROC / "market_features.csv", index_col=0, parse_dates=True)
    bt = None
    if (RESULTS_DIR / "backtest_rendements_quotidiens.csv").exists():
        bt = pd.read_csv(RESULTS_DIR / "backtest_rendements_quotidiens.csv", index_col=0, parse_dates=True)
    perf = None
    if (RESULTS_DIR / "backtest_performances_test.csv").exists():
        perf = pd.read_csv(RESULTS_DIR / "backtest_performances_test.csv", index_col=0)
    return prices, panel, mkt, bt, perf


BACKUP_DIR = ROOT / "data" / "backup"


def _data_files():
    from src.data_download import YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE, FRED_FILE
    return [YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE, FRED_FILE, DATA_PROC / "prices_clean.csv",
            DATA_PROC / "panel_recent.csv", DATA_PROC / "market_features.csv"]


def backup_data() -> None:
    """Copie les données actuelles dans data/backup/ avant toute mise à jour."""
    import shutil
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in _data_files():
        if f.exists() and f.stat().st_size > 1000:          # on ne sauvegarde pas un fichier vide
            shutil.copy2(f, BACKUP_DIR / f.name)


def has_backup() -> bool:
    return all((BACKUP_DIR / f.name).exists() for f in _data_files())


def restore_data() -> None:
    """Remet en place les données sauvegardées avant la dernière mise à jour."""
    import os, shutil
    for f in _data_files():
        tmp = f.with_name(f.name + ".tmp")
        shutil.copy2(BACKUP_DIR / f.name, tmp)
        os.replace(tmp, f)


def refresh_market_data():
    """Mise à jour RAPIDE et sécurisée des données de marché (objectif : moins de 30 s).

    - Seuls les derniers jours sont téléchargés (une requête Yahoo groupée, séries FRED en
      parallèle), avec une limite de temps stricte ; puis raccordement à l'historique en cache.
    - Tout est recalculé en mémoire et contrôlé ; les fichiers ne sont remplacés que si tout est
      complet et au moins aussi récent. Sinon une exception est levée, rien n'est modifié.
    - Les modèles ne sont pas réentraînés (relancer le notebook pour cela).
    """
    from src.data_download import (update_yahoo_incremental, update_fred_incremental, _atomic_to_csv,
                                   YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE, FRED_FILE)
    from src.preprocessing import clean_prices, build_macro_features
    from src.features import compute_asset_features, compute_market_features

    close, volume = update_yahoo_incremental(timeout=20)     # lève une erreur si incomplet ou trop lent
    try:
        fred = update_fred_incremental(timeout=10)
    except Exception:
        fred = pd.read_csv(FRED_FILE, index_col=0, parse_dates=True)   # macro non bloquante
    prices_new, _ = clean_prices(close)
    macro = build_macro_features(fred, prices_new.index)
    recent_prices = prices_new.tail(700)                       # 252 j d'historique + 260 dates utiles
    panel = compute_asset_features(recent_prices, macro.loc[recent_prices.index])
    mkt_new = compute_market_features(prices_new, macro)
    if panel.empty or mkt_new.dropna(how="all").empty:
        raise RuntimeError("les variables recalculées sont vides")
    old_last = pd.read_csv(DATA_PROC / "prices_clean.csv", index_col=0, parse_dates=True).index.max() \
        if (DATA_PROC / "prices_clean.csv").exists() else None
    if old_last is not None and pd.notna(old_last) and prices_new.index.max() < old_last:
        raise RuntimeError(f"les nouvelles données s'arrêtent au {prices_new.index.max():%d/%m/%Y}, "
                           f"avant les données actuelles ({old_last:%d/%m/%Y})")

    # Tout est valide : sauvegarde des données actuelles, puis écriture (fichier temporaire + renommage)
    backup_data()
    dates = panel.index.get_level_values("date").unique().sort_values()[-260:]
    _atomic_to_csv(close, YAHOO_CLOSE_FILE)
    _atomic_to_csv(volume, YAHOO_VOLUME_FILE)
    _atomic_to_csv(fred, FRED_FILE)
    _atomic_to_csv(prices_new, DATA_PROC / "prices_clean.csv")
    _atomic_to_csv(panel[panel.index.get_level_values("date").isin(dates)], DATA_PROC / "panel_recent.csv")
    _atomic_to_csv(mkt_new, DATA_PROC / "market_features.csv")
    return prices_new.index.max()


models = load_models()
if models is None:
    st.error("Modèles introuvables. Exécutez d'abord entièrement le notebook "
             "`notebooks/Projet_MultiAgent_Portefeuille.ipynb` (section 9 : sauvegarde des artefacts).")
    st.stop()
prices, panel, mkt, bt, perf = load_data()
if prices.empty or panel.empty or mkt.dropna(how="all").empty:
    st.error("Les données de marché préparées sont vides ou incomplètes (téléchargement interrompu ?). "
             "Relancez la section 2 du notebook, ou cliquez sur « Mettre à jour les données de marché » "
             "quand Yahoo Finance répond de nouveau.")
    col_a, col_b = st.columns(2)
    if has_backup() and col_a.button("↩️ Revenir aux données précédentes"):
        restore_data()
        st.cache_data.clear()
        st.rerun()
    if col_b.button("🔄 Réessayer la mise à jour des données"):
        done = False
        try:
            with st.spinner("Mise à jour des derniers cours (30 s maximum)…"):
                refresh_market_data()
            done = True
        except Exception as exc:
            st.error(f"Mise à jour impossible : {exc}")
        if done:                                             # st.rerun() hors du try
            st.cache_data.clear()
            st.rerun()
    st.stop()
last_date = panel.index.get_level_values("date").max()

# ---------------------------------------------------------------------------
# Barre latérale : questionnaire client
# ---------------------------------------------------------------------------
st.sidebar.title("👤 Profil du client")
cur_code = st.sidebar.selectbox("Devise des montants saisis", list(CURRENCIES),
                                format_func=lambda c: CURRENCIES[c]["label"])
cur = CURRENCIES[cur_code]
if cur_code == "USD":
    per_usd = 1.0
else:
    per_usd = st.sidebar.number_input(f"Taux de change : 1 USD = … {cur_code}", min_value=0.01,
                                      value=float(cur["per_usd"]), step=0.01, format="%.4f",
                                      help="Taux indicatif de novembre 2026, à ajuster au cours du jour.")
st.sidebar.caption(f"Montants en {cur['label'].lower()}. Ils sont convertis en dollars US pour le modèle, "
                   "entraîné sur des ménages américains (enquête de la Réserve fédérale, montants en USD).")
sym = cur["symbol"]
age = st.sidebar.slider("Âge", 18, 95, 40)
kids = st.sidebar.number_input("Enfants à charge", 0, 10, 1)
income_loc = st.sidebar.number_input(f"Revenu annuel du foyer ({sym})", 0, int(5_000_000 * per_usd),
                                     int(round(70_000 * per_usd, -3)), step=int(round(5_000 * per_usd, -3)) or 1000)
assets_loc = st.sidebar.number_input(f"Total des actifs, immobilier compris ({sym})", 0, int(50_000_000 * per_usd),
                                     int(round(300_000 * per_usd, -3)), step=int(round(10_000 * per_usd, -3)) or 1000)
debt_loc = st.sidebar.number_input(f"Total des dettes ({sym})", 0, int(50_000_000 * per_usd),
                                   int(round(150_000 * per_usd, -3)), step=int(round(10_000 * per_usd, -3)) or 1000)
# Conversion en dollars US pour le modèle
income, assets, debt = income_loc / per_usd, assets_loc / per_usd, debt_loc / per_usd
knowl = st.sidebar.slider("Connaissance des questions financières (0 = aucune, 10 = excellente)", 0, 10, 6)
finlit = st.sidebar.slider("Bonnes réponses au quiz de culture financière (0-3)", 0, 3, 2)
sex = st.sidebar.radio("Sexe de la personne de référence", ["Homme", "Femme"], horizontal=True)
edu = st.sidebar.selectbox("Niveau d'études", ["Sans diplôme du secondaire", "Diplôme du secondaire",
                                               "Études supérieures courtes", "Diplôme universitaire"], index=3)
married = st.sidebar.radio("En couple", ["Oui", "Non"], horizontal=True)
occ = st.sidebar.selectbox("Situation professionnelle", ["Salarié", "Indépendant / chef d'entreprise",
                                                         "Retraité", "Sans activité"])
owner = st.sidebar.radio("Propriétaire de sa résidence", ["Oui", "Non"], horizontal=True)
saved = st.sidebar.radio("A épargné l'an dernier", ["Oui", "Non"], horizontal=True)
st.sidebar.divider()
st.sidebar.title("📝 Questionnaire de tolérance au risque")
answers = {}
for key, (question, options) in QUESTIONS.items():
    answers[key] = options.index(st.sidebar.radio(question, options, index=1, key=key))

lifecl = 1 if age < 55 and married == "Non" and kids == 0 else 2 if age < 55 and married == "Oui" and kids > 0 \
    else 3 if age < 55 and married == "Oui" else 4 if age < 55 else 5 if occ != "Retraité" else 6
client = {
    "AGE": age, "KIDS": kids, "INCOME": income, "NETWORTH": assets - debt, "ASSET": assets, "DEBT": debt,
    "LEVRATIO": debt / assets if assets > 0 else 0, "DEBT2INC": debt / income if income > 0 else 0,
    "KNOWL": knowl, "FINLIT": finlit, "HHSEX": 1 if sex == "Homme" else 2,
    "EDCL": ["Sans diplôme du secondaire", "Diplôme du secondaire", "Études supérieures courtes",
             "Diplôme universitaire"].index(edu) + 1,
    "MARRIED": 1 if married == "Oui" else 2,
    "OCCAT1": {"Salarié": 1, "Indépendant / chef d'entreprise": 2, "Retraité": 3, "Sans activité": 4}[occ],
    "LIFECL": lifecl, "HOUSECL": 1 if owner == "Oui" else 2, "SAVED": 1 if saved == "Oui" else 0,
    "WSAVED": 3 if saved == "Oui" else 1, "WAGEINC": income if occ == "Salarié" else 0, "NORMINC": income,
    **answers,
}

st.sidebar.divider()
override = st.sidebar.selectbox("Profil validé par le conseiller",
                                ["Utiliser la prédiction de l'agent"] + PROFILE_CLASSES)
if st.sidebar.button("🔄 Mettre à jour les données de marché"):
    new_last = None
    try:
        with st.spinner("Mise à jour des derniers cours (Yahoo Finance + FRED, 30 s maximum)…"):
            new_last = refresh_market_data()
    except Exception as exc:
        st.session_state["refresh_msg"] = ("error", "Mise à jour annulée, les données actuelles sont "
                                                    f"conservées. Motif : {exc}")
    if new_last is not None:                                 # st.rerun() hors du try
        st.session_state["refresh_msg"] = ("success", f"Données mises à jour jusqu'au {new_last:%d/%m/%Y}.")
        st.cache_data.clear()
        st.rerun()
if has_backup() and st.sidebar.button("↩️ Revenir aux données précédentes"):
    restore_data()
    st.session_state["refresh_msg"] = ("success", "Données précédentes restaurées.")
    st.cache_data.clear()
    st.rerun()
if "refresh_msg" in st.session_state:
    kind, text = st.session_state.pop("refresh_msg")
    (st.sidebar.success if kind == "success" else st.sidebar.error)(text)

# ---------------------------------------------------------------------------
# Corps de page
# ---------------------------------------------------------------------------
st.title("📈 Assistant multi-agent de sélection de portefeuille")
st.info("Les actifs proposés sont des ETF cotés en **dollars US** : les performances et les risques affichés sont "
        "exprimés en USD, **hors effet de change**."
        + ("" if cur_code == "USD" else
           f" Pour un investisseur qui raisonne en {cur['label'].split(' (')[0].lower()}, la valeur du "
           "portefeuille varie aussi avec le cours du dollar."))
st.caption(f"Données de marché au {last_date:%d/%m/%Y} · modèle Profil : "
           f"{models['meta'].get('modele_profil', '?')} · modèle Prévision : "
           f"{models['meta'].get('modele_prevision', '?')}")

tab_reco, tab_bt, tab_about = st.tabs(["Recommandation", "Backtest", "Architecture"])

with tab_reco:
    orch = PortfolioOrchestrator(models["profil"], models["regime"], models["prevision"])
    forced = None if override.startswith("Utiliser") else override
    res = None
    try:
        with st.spinner("Les agents délibèrent…"):
            res = orch.recommend(last_date, prices, panel, mkt, client=client, profile=forced)
    except Exception as exc:
        st.error(f"Les agents n'ont pas pu produire de recommandation : {exc}. "
                 "Si cela suit une mise à jour des données, utilisez « Revenir aux données précédentes » "
                 "dans la barre latérale.")
    if res is not None:

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profil retenu", res["profile"])
        c2.metric("Régime de marché", res["regime"])
        c3.metric("Volatilité ex ante", f"{res['metrics']['vol_ex_ante']:.1%}",
                  help=f"Cible du profil : {PROFILE_CONSTRAINTS[res['profile']]['target_vol']:.0%}")
        c4.metric("VaR 95 % à 1 mois", f"{res['metrics']['VaR95_1mois_param']:.1%}")

        det = res.get("profile_detail")
        if forced is None and det:
            q = det["questionnaire"]
            p1, p2, p3 = st.columns(3)
            p1.metric("Tolérance déclarée (questionnaire)", det["declare"], f"score {q['score']}/{q['score_max']}",
                      delta_color="off")
            p2.metric("Capacité estimée (modèle ML)", det["capacite"],
                      f"P(profil prudent) = {res['profile_proba'].get('Prudent', 0):.0%}", delta_color="off")
            p3.metric("Profil retenu (le plus prudent)", det["final"])
            for cap in q["plafonds"]:
                st.info(cap)
            if det["alerte"]:
                st.warning(det["alerte"])
        w = res["weights"][res["weights"] > 0.005].sort_values(ascending=False)
        df_w = pd.DataFrame({"Ticker": w.index, "Actif": w.index.map(ASSET_NAME),
                             "Classe": w.index.map(ASSET_CLASS), "Poids": w.values})
        g1, g2 = st.columns([1, 1])
        with g1:
            fig = px.pie(df_w, names="Classe", values="Poids", hole=.45, title="Répartition par classe d'actifs")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig = px.bar(df_w, x="Poids", y="Ticker", color="Classe", orientation="h",
                         title="Poids par actif", hover_data=["Actif"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(res["explanation"])
        with st.expander("Détail des positions"):
            st.dataframe(df_w.style.format({"Poids": "{:.2%}"}), use_container_width=True, hide_index=True)
        if res["forecast"] is not None:
            with st.expander("Scores de l'agent Prévision"):
                st.dataframe(res["forecast"].sort_values("proba", ascending=False)
                             .style.format({"proba": "{:.1%}", "z": "{:.2f}", "vol": "{:.1%}", "alpha": "{:.2%}"}),
                             use_container_width=True)
        with st.expander("Journal des échanges entre agents"):
            st.dataframe(res["log"], use_container_width=True, hide_index=True)

with tab_bt:
    if bt is None:
        st.info("Résultats de backtest absents : exécutez la section 8 du notebook.")
    else:
        st.subheader("Performance hors échantillon (période de test)")
        choix = st.multiselect("Stratégies", list(bt.columns),
                               default=[c for c in bt.columns if c.startswith("Multi-agent –")]
                               + ["Équipondérée 1/N", "60/40"])
        if choix:
            st.line_chart((1 + bt[choix].fillna(0)).cumprod())
        if perf is not None:
            st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)

with tab_about:
    st.markdown("""
**Cycle de décision**

1. **Agent Profil** – classifieur entraîné sur le Survey of Consumer Finances : estime la tolérance au risque.
2. **Agent Régime** – mélange gaussien sur la volatilité, le VIX, la pente de courbe et le spread de crédit.
3. **Agent Prévision** – classifieur de surperformance à 1 mois de 25 ETF ; probabilités → rendements espérés
   (α = IC × σ × z).
4. **Agent Risque** – covariance de Ledoit-Wolf, volatilité, VaR, CVaR, contrôle des contraintes.
5. **Agent Allocation** – optimisation moyenne-variance sous contraintes du profil ; renégocie avec l'agent
   Risque si une limite est dépassée.
6. **Agent Explication** – rédige la justification.

*Outil pédagogique – ne constitue pas un conseil en investissement.*
""")
