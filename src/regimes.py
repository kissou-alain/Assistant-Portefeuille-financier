"""
Agent Régime : clustering non supervisé des états du marché (cours 2, « Techniques de
clustering »).

Les quatre familles du cours sont comparées sur les mêmes variables d'état (stationnaires,
standardisées sur l'entraînement) :
1. partition       : K-means (coude + silhouette) ;
2. hiérarchique    : CAH avec le critère de Ward (dendrogramme, plus grand saut) ;
3. densité         : DBSCAN (ε choisi par le graphe des k-distances) ;
4. probabiliste    : mélange gaussien (GMM) estimé par l'algorithme EM.

Indices d'évaluation du cours : inertie intra-classe J_w, inertie inter-classes J_b,
silhouette (↑), Davies-Bouldin (↓), Calinski-Harabasz (↑).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Indices d'évaluation
# ---------------------------------------------------------------------------
def inertias(Z: np.ndarray, labels: np.ndarray) -> dict:
    """J_w = Σ_k Σ_{x∈G_k} ||x − μ_k||²   et   J_b = Σ_k n_k ||μ_k − μ||²  (J_w + J_b = inertie totale)."""
    mask = labels >= 0                       # DBSCAN : le bruit (−1) est exclu
    Z, labels = Z[mask], labels[mask]
    mu = Z.mean(0)
    jw = jb = 0.0
    for k in np.unique(labels):
        Zk = Z[labels == k]
        mk = Zk.mean(0)
        jw += ((Zk - mk) ** 2).sum()
        jb += len(Zk) * ((mk - mu) ** 2).sum()
    return {"inertie intra J_w": jw, "inertie inter J_b": jb, "% inertie expliquée": jb / (jw + jb) * 100}


def cluster_scores(Z: np.ndarray, labels: np.ndarray, sample: int = 3000) -> dict:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz et inerties (bruit DBSCAN exclu).
    La silhouette est calculée sur un échantillon (coût en O(n²))."""
    mask = labels >= 0
    Zm, lm = Z[mask], labels[mask]
    k = len(np.unique(lm))
    out = {"nb clusters": k, "% bruit": float((~mask).mean() * 100)}
    if k < 2:
        return {**out, "silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    out.update({
        "silhouette": float(silhouette_score(Zm, lm, sample_size=min(sample, len(Zm)),
                                             random_state=RANDOM_STATE)),
        "Davies-Bouldin": float(davies_bouldin_score(Zm, lm)),
        "Calinski-Harabasz": float(calinski_harabasz_score(Zm, lm)),
    })
    out.update(inertias(Z, labels))
    return out


# ---------------------------------------------------------------------------
# Choix du nombre de groupes
# ---------------------------------------------------------------------------
def kmeans_elbow(Z: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    """Méthode du coude (inertie intra J_w) et silhouette pour K-means."""
    rows = {}
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        rows[k] = {"inertie intra J_w": km.inertia_,
                   "silhouette": silhouette_score(Z, km.labels_, sample_size=min(3000, len(Z)),
                                                  random_state=RANDOM_STATE)}
    return pd.DataFrame(rows).T.rename_axis("K")


def gmm_bic(Z: np.ndarray, k_range=range(2, 8)) -> pd.Series:
    """Critère BIC du mélange gaussien (plus bas = meilleur compromis ajustement / complexité)."""
    return pd.Series({k: GaussianMixture(k, covariance_type="full", n_init=5,
                                         random_state=RANDOM_STATE).fit(Z).bic(Z) for k in k_range},
                     name="BIC").rename_axis("K")


def k_distance(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Distance au k-ième plus proche voisin, triée : le « coude » de cette courbe donne ε pour DBSCAN."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return np.sort(d[:, -1])


# ---------------------------------------------------------------------------
# Les quatre familles
# ---------------------------------------------------------------------------
def fit_four_families(Z: np.ndarray, k: int = 3, eps: float = 0.6, min_samples: int = 10) -> dict:
    """Ajuste les 4 méthodes sur Z (données d'entraînement standardisées).
    Retourne {nom: (modèle, étiquettes)}. Le modèle CAH est la matrice de liaison."""
    out = {}
    km = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(Z)
    out["K-means"] = (km, km.labels_)
    link = linkage(Z, method="ward")
    out["CAH (Ward)"] = (link, fcluster(link, k, criterion="maxclust") - 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    out["DBSCAN"] = (db, db.labels_)
    gmm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=RANDOM_STATE).fit(Z)
    out["GMM (EM)"] = (gmm, gmm.predict(Z))
    return out


def compare_families(Z: np.ndarray, fitted: dict) -> pd.DataFrame:
    return pd.DataFrame({name: cluster_scores(Z, lab) for name, (_, lab) in fitted.items()}).T


def describe_clusters(M: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Profil statistique des clusters (cours 2, « Décrire les clusters obtenus ») :
    effectif, poids relatif, moyennes, et p-valeur de Kruskal-Wallis par variable
    (les groupes diffèrent-ils significativement sur cette variable ?)."""
    df = M.copy()
    df["cluster"] = labels
    df = df[df["cluster"] >= 0]
    prof = df.groupby("cluster").mean()
    prof.insert(0, "% période", df["cluster"].value_counts(normalize=True).sort_index() * 100)
    prof.insert(0, "effectif", df["cluster"].value_counts().sort_index())
    kw = {c: stats.kruskal(*[g[c].values for _, g in df.groupby("cluster")]).pvalue for c in M.columns}
    prof.loc["p-valeur Kruskal-Wallis"] = [np.nan, np.nan] + [kw[c] for c in M.columns]
    return prof


def label_by_volatility(centers: pd.DataFrame, vol_col: str) -> dict:
    """Nomme les régimes par volatilité croissante : Calme < Neutre < Stress."""
    order = centers[vol_col].argsort().values
    names = ["Calme", "Neutre", "Stress"] if len(order) == 3 else [f"Régime {i + 1}" for i in range(len(order))]
    return {int(o): names[i] for i, o in enumerate(order)}
