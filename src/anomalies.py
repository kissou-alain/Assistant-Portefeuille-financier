"""
Agent Alerte : détection d'anomalies sur les journées de marché (cours 1 p. 22-24 ;
cours 2, « La détection d'anomalies »).

Méthodes statistiques : z-score, test de Grubbs.
Méthodes d'apprentissage non supervisé : distribution gaussienne multivariée (seuil ε sur
la densité), Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.

Il n'existe pas d'étiquette « anomalie » pour les marchés : la validation est QUALITATIVE
(les épisodes de crise connus sont-ils détectés ?) et repose sur l'accord entre méthodes.
Tous les seuils et paramètres sont appris sur l'entraînement ; les autres périodes sont
seulement scorées.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import RANDOM_STATE

# Épisodes de crise connus, pour la validation qualitative
KNOWN_EVENTS = {
    "Crise de la dette US / euro (août 2011)": ("2011-08-01", "2011-08-31"),
    "Dévaluation du yuan (août 2015)": ("2015-08-20", "2015-08-31"),
    "Volmageddon (février 2018)": ("2018-02-02", "2018-02-09"),
    "Krach COVID (mars 2020)": ("2020-02-24", "2020-03-31"),
    "Choc inflation / taux (juin 2022)": ("2022-06-09", "2022-06-17"),
    "Choc de liquidité (août 2024)": ("2024-08-01", "2024-08-07"),
}


def zscore_flags(r: pd.Series, train_end, threshold: float = 4.0) -> pd.Series:
    """|z| = |r − moyenne| / écart-type > seuil, moyenne et écart-type de l'entraînement."""
    mu, sd = r.loc[:train_end].mean(), r.loc[:train_end].std()
    return ((r - mu) / sd).abs() > threshold


def grubbs_test(x: np.ndarray, alpha: float = 0.01) -> dict:
    """Test de Grubbs (une valeur aberrante dans un échantillon supposé gaussien).

    G = max |x_i − x̄| / s ;   valeur critique
    G_c = (n − 1)/√n · √( t² / (n − 2 + t²) ),  t = quantile de Student t_{α/(2n), n−2}
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    i = int(np.argmax(np.abs(x - x.mean())))
    G = float(abs(x[i] - x.mean()) / x.std(ddof=1))
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    Gc = float((n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2)))
    return {"G": G, "G critique": Gc, "aberrante": G > Gc, "indice": i, "valeur": float(x[i])}


def grubbs_iterative(r: pd.Series, alpha: float = 0.01, max_out: int = 50) -> pd.Series:
    """Applique Grubbs de façon répétée (on retire la valeur aberrante et on recommence)."""
    s = r.dropna().copy()
    flags = pd.Series(False, index=r.index)
    for _ in range(max_out):
        res = grubbs_test(s.values, alpha)
        if not res["aberrante"]:
            break
        idx = s.index[res["indice"]]
        flags[idx] = True
        s = s.drop(idx)
    return flags


class GaussianAnomaly:
    """Distribution gaussienne multivariée (cours 2) : p(x; μ, Σ) estimée sur l'entraînement ;
    anomalie si p(x) ≤ ε. Le seuil ε est le quantile `contamination` des densités du train."""

    def __init__(self, contamination: float = 0.02):
        self.contamination = contamination

    def fit(self, Z):
        self.dist_ = stats.multivariate_normal(mean=Z.mean(0), cov=np.cov(Z, rowvar=False),
                                               allow_singular=True)
        self.eps_ = np.quantile(self.dist_.logpdf(Z), self.contamination)
        return self

    def score(self, Z):
        return -np.atleast_1d(self.dist_.logpdf(Z))      # plus grand = plus anormal

    def predict(self, Z):
        # atleast_1d : scipy renvoie un scalaire quand on lui passe une seule observation
        return np.atleast_1d(self.dist_.logpdf(Z)) <= self.eps_


def fit_detectors(M_train: pd.DataFrame, contamination: float = 0.02) -> dict:
    """Ajuste le standardiseur et les 4 détecteurs ML sur l'entraînement."""
    sc = StandardScaler().fit(M_train)
    Z = sc.transform(M_train)
    return {
        "scaler": sc,
        "Gaussienne multivariée": GaussianAnomaly(contamination).fit(Z),
        "Isolation Forest": IsolationForest(n_estimators=300, contamination=contamination,
                                            random_state=RANDOM_STATE).fit(Z),
        "LOF": LocalOutlierFactor(n_neighbors=35, contamination=contamination, novelty=True).fit(Z),
        "One-Class SVM": OneClassSVM(nu=contamination, gamma="scale").fit(Z),
    }


def detect(detectors: dict, M: pd.DataFrame) -> pd.DataFrame:
    """Drapeau d'anomalie de chaque détecteur ML, et nombre de détecteurs d'accord."""
    Z = detectors["scaler"].transform(M)
    out = pd.DataFrame(index=M.index)
    for name, det in detectors.items():
        if name == "scaler":
            continue
        pred = det.predict(Z)
        out[name] = pred.astype(bool) if name == "Gaussienne multivariée" else (pred == -1)
    out["nb détecteurs"] = out.sum(1)
    return out


def event_recall(flags: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque épisode connu : chaque méthode a-t-elle signalé au moins un jour ?"""
    rows = {}
    for ev, (a, b) in KNOWN_EVENTS.items():
        win = flags.loc[a:b]
        if len(win):
            rows[ev] = win.drop(columns=["nb détecteurs"], errors="ignore").any()
    return pd.DataFrame(rows).T
