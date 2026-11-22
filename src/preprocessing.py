"""
Prétraitement des données brutes : nettoyage des cours, alignement des
calendriers, gestion des valeurs manquantes et aberrantes, construction des
variables macroéconomiques SANS fuite temporelle, préparation du SCF.

Règle d'or : toute transformation qui « apprend » quelque chose des données
(quantiles, moyennes, écarts-types, imputations) est encapsulée dans un
transformateur scikit-learn et n'est ajustée QUE sur l'ensemble d'entraînement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (DAILY_RELEASE_LAG, FRED_DAILY, FRED_MONTHLY, MARKET_TICKER,
                     MONTHLY_RELEASE_LAG_DAYS, PROFILE_CLASSES, PROFILE_ML_CLASSES,
                     PROFILE_TARGET_MODE, SCF_CAT_FEATURES, SCF_HOUSEHOLD_ID,
                     SCF_NUM_FEATURES, SCF_WEIGHT)


# ---------------------------------------------------------------------------
# Cours des actifs
# ---------------------------------------------------------------------------
def clean_prices(close: pd.DataFrame, market: str = MARKET_TICKER,
                 max_ffill: int = 5, jump_threshold: float = 0.40):
    """Nettoie la matrice des cours ajustés.

    Étapes :
    1. tri chronologique et suppression des dates dupliquées ;
    2. alignement sur le calendrier boursier NYSE (dates où le marché de
       référence cote) – le Bitcoin, coté 7j/7, est ainsi ré-échantillonné ;
    3. suppression des prix non positifs (erreurs de saisie) ;
    4. détection des sauts aberrants (|rendement| > seuil suivi d'un retour
       immédiat = « spike ») hors crypto, remplacés par NaN ;
    5. propagation de la dernière valeur connue sur au plus `max_ffill` jours
       (jours fériés, suspensions) – jamais de remplissage vers l'arrière
       (ce serait utiliser une information future).

    Retourne (prix_nettoyés, rapport).
    """
    if market not in close.columns or close[market].notna().sum() == 0:
        raise ValueError(f"Le marché de référence {market} est absent ou vide : impossible de construire "
                         "le calendrier boursier. Les données téléchargées sont incomplètes.")
    report = {}
    df = close.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    report["dates_brutes"] = len(df)

    calendar = df.index[df[market].notna()]
    df = df.reindex(calendar)
    report["dates_calendrier_NYSE"] = len(df)

    n_nonpos = int((df <= 0).sum().sum())
    df = df.where(df > 0)
    report["prix_non_positifs"] = n_nonpos

    rets = df.pct_change(fill_method=None)
    spikes = (rets.abs() > jump_threshold) & (rets.shift(-1).abs() > jump_threshold) \
        & (np.sign(rets) != np.sign(rets.shift(-1)))
    if "BTC-USD" in spikes.columns:
        spikes["BTC-USD"] = False
    report["pics_aberrants"] = int(spikes.sum().sum())
    df = df.mask(spikes)

    report["valeurs_manquantes_avant_ffill"] = df.isna().sum().to_dict()
    df = df.ffill(limit=max_ffill)
    report["valeurs_manquantes_apres_ffill"] = df.isna().sum().to_dict()
    report["premiere_date_par_actif"] = {c: str(df[c].first_valid_index())[:10] for c in df}
    return df, report


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements arithmétiques quotidiens (NaN tant que l'actif n'existe pas)."""
    return prices.pct_change(fill_method=None)


# ---------------------------------------------------------------------------
# Variables macroéconomiques
# ---------------------------------------------------------------------------
def build_macro_features(fred: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Construit des variables macro alignées sur le calendrier boursier,
    en ne retenant que l'information réellement disponible à chaque date.

    * Séries quotidiennes : décalées de DAILY_RELEASE_LAG jour(s) ouvré(s).
    * Séries mensuelles : la valeur du mois M devient disponible le
      (fin du mois M + MONTHLY_RELEASE_LAG_DAYS jours).
    """
    out = pd.DataFrame(index=calendar)

    for sid in FRED_DAILY:
        if sid not in fred:
            continue
        s = fred[sid].dropna()
        s = s.reindex(s.index.union(calendar)).ffill().reindex(calendar)
        out[sid] = s.shift(DAILY_RELEASE_LAG)

    monthly = {}
    for sid in FRED_MONTHLY:
        if sid not in fred:
            continue
        s = fred[sid].dropna()
        s = s[~s.index.duplicated()]
        if sid in ("CPIAUCSL", "INDPRO"):
            monthly[f"{sid}_yoy"] = s.pct_change(12) * 100
        if sid == "UNRATE":
            monthly["UNRATE"] = s
            monthly["UNRATE_chg3m"] = s.diff(3)
    for name, s in monthly.items():
        avail = s.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=MONTHLY_RELEASE_LAG_DAYS)
        s_av = pd.Series(s.values, index=avail).dropna()
        s_av = s_av[~s_av.index.duplicated(keep="last")]
        out[name] = s_av.reindex(s_av.index.union(calendar)).ffill().reindex(calendar)

    # Variables dérivées
    if "VIXCLS" in out:
        out["VIX_chg21"] = out["VIXCLS"] - out["VIXCLS"].shift(21)
    if "DGS10" in out:
        out["DGS10_chg63"] = out["DGS10"] - out["DGS10"].shift(63)
    if "BAA10Y" in out:
        out["BAA10Y_chg63"] = out["BAA10Y"] - out["BAA10Y"].shift(63)
    out = out.rename(columns={"CPIAUCSL_yoy": "CPI_yoy", "INDPRO_yoy": "INDPRO_yoy",
                              "VIXCLS": "VIX"})
    return out


def risk_free_daily(macro: pd.DataFrame) -> pd.Series:
    """Taux sans risque quotidien (T-Bill 3 mois, convention base 252)."""
    rf = macro.get("DTB3", pd.Series(0.0, index=macro.index)).ffill().fillna(0) / 100
    return rf / 252


# ---------------------------------------------------------------------------
# Transformateurs scikit-learn réutilisables (importables par l'application)
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Écrête chaque variable entre ses quantiles [lower, upper] appris
    sur l'ensemble d'ENTRAÎNEMENT uniquement (robustesse aux valeurs extrêmes)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


class SignedLog(BaseEstimator, TransformerMixin):
    """Transformation sign(x)·log(1+|x|) pour les montants très asymétriques
    (revenu, patrimoine, dettes). Sans paramètre appris -> aucune fuite."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


# ---------------------------------------------------------------------------
# Survey of Consumer Finances
# ---------------------------------------------------------------------------
MONEY_FEATURES = ["INCOME", "NORMINC", "WAGEINC", "NETWORTH", "ASSET", "DEBT"]


def build_risk_target(scf: pd.DataFrame, mode: str = PROFILE_TARGET_MODE) -> pd.Series:
    """Cible à 3 classes à partir de la question SCF sur la tolérance au risque :
    - NOFINRISK = 1  (« pas prêt à prendre de risque »)          -> Prudent
    - YESFINRISK = 1 (« prêt à prendre un risque élevé/substantiel ») -> Dynamique
    - sinon (risque « moyen »)                                      -> Équilibré
    En mode "binaire" : NOFINRISK = 1 -> Prudent, sinon -> Non prudent.
    """
    if mode == "binaire":
        y = pd.Series("Non prudent", index=scf.index)
        y[scf["NOFINRISK"] == 1] = "Prudent"
        return pd.Categorical(y, categories=["Prudent", "Non prudent"], ordered=True)
    y = pd.Series("Équilibré", index=scf.index)
    y[scf["NOFINRISK"] == 1] = "Prudent"
    y[scf["YESFINRISK"] == 1] = "Dynamique"
    return pd.Categorical(y, categories=PROFILE_CLASSES, ordered=True)


def prepare_scf(scf: pd.DataFrame):
    """Sélectionne les variables, crée la cible, contrôle la qualité.

    Retourne (X, y, groups, weights, rapport). `groups` = identifiant du ménage :
    les 5 « implicates » d'un même ménage (imputation multiple) doivent rester
    dans le MÊME sous-ensemble, sinon le modèle verrait des quasi-doublons du
    jeu de test pendant l'entraînement (fuite de données).
    """
    scf = scf.copy()
    scf.columns = scf.columns.str.upper()
    report = {"lignes_brutes": len(scf)}
    num = [c for c in SCF_NUM_FEATURES if c in scf]
    cat = [c for c in SCF_CAT_FEATURES if c in scf]
    report["variables_numeriques"], report["variables_categorielles"] = num, cat
    report["variables_absentes"] = [c for c in SCF_NUM_FEATURES + SCF_CAT_FEATURES
                                    if c not in scf]

    scf = scf.dropna(subset=["NOFINRISK", "YESFINRISK"])
    y = pd.Series(build_risk_target(scf), index=scf.index, name="profil_risque")
    X = scf[num + cat].copy()
    # Codes négatifs de non-réponse éventuels -> valeurs manquantes
    for c in ["KNOWL", "FINLIT", "KIDS", "AGE"]:
        if c in X:
            X.loc[X[c] < 0, c] = np.nan
    # Bornes de cohérence
    if "LEVRATIO" in X:
        X.loc[X["LEVRATIO"] < 0, "LEVRATIO"] = np.nan
    for c in cat:
        X[c] = X[c].astype("Int64").astype(str).replace("<NA>", np.nan)

    groups = scf[SCF_HOUSEHOLD_ID] if SCF_HOUSEHOLD_ID in scf else pd.Series(scf.index, index=scf.index)
    weights = scf[SCF_WEIGHT] if SCF_WEIGHT in scf else pd.Series(1.0, index=scf.index)
    report["lignes_finales"] = len(X)
    report["menages_uniques"] = int(groups.nunique())
    report["valeurs_manquantes"] = X.isna().sum().to_dict()
    report["repartition_cible"] = y.value_counts(normalize=True).round(3).to_dict()
    return X, y, groups, weights, report
