"""
Stratégies d'allocation de référence, fondées uniquement sur le risque (aucune prévision
de rendement). Elles servent de points de comparaison à l'agent Allocation
(Markowitz / Black-Litterman) et à l'agent d'apprentissage par renforcement.

- Parité de risque (Equal Risk Contribution) : chaque actif contribue autant au risque.
- HRP (Hierarchical Risk Parity, López de Prado 2016) : regroupe les actifs par
  CLASSIFICATION HIÉRARCHIQUE (lien avec la CAH du cours), puis répartit le risque de façon
  récursive entre les groupes. Robuste car il n'inverse pas la matrice de covariance.
- Mélange avec la poche monétaire pour ramener un portefeuille à la volatilité cible d'un
  profil (comparaison « à risque égal »).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns_window: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    """Covariance annualisée (rétrécissement de Ledoit-Wolf) sur les actifs assez cotés."""
    R = returns_window.dropna(axis=1, thresh=int(min_coverage * len(returns_window))).dropna()
    return pd.DataFrame(LedoitWolf().fit(R.values).covariance_ * 252, index=R.columns, columns=R.columns)


def risk_contributions(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    """RC_i = w_i (Σw)_i / σ_p  (les contributions s'additionnent à la volatilité σ_p)."""
    sig = np.sqrt(w @ S @ w)
    return w * (S @ w) / sig


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Poids tels que toutes les contributions au risque soient égales (sans vente à découvert).
    Formulation convexe de Spinu (2013) : min ½ yᵀΣy − (1/n) Σ ln y_i, puis w = y / Σy."""
    S = cov.values
    n = len(S)
    res = minimize(lambda y: 0.5 * y @ S @ y - np.log(y).sum() / n, np.full(n, 1 / n),
                   jac=lambda y: S @ y - 1 / (n * y), bounds=[(1e-8, None)] * n, method="L-BFGS-B")
    w = res.x / res.x.sum()
    return pd.Series(w, index=cov.index)


def hrp(cov: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity.
    1. distance entre actifs d_ij = √(½(1 − ρ_ij)) ;
    2. classification hiérarchique (liaison simple) → ordre des feuilles du dendrogramme ;
    3. bissection récursive : à chaque coupe, le risque est réparti entre les deux moitiés
       en proportion inverse de leur variance (variance de chaque moitié en inverse-variance)."""
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    tickers = cov.index[order]
    w = pd.Series(1.0, index=tickers)

    def cluster_var(items):
        c = cov.loc[items, items].values
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return float(ivp @ c @ ivp)

    stack = [list(tickers)]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        left, right = items[:len(items) // 2], items[len(items) // 2:]
        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [left, right]
    return (w / w.sum()).reindex(cov.index)


def to_target_vol(w: pd.Series, cov: pd.DataFrame, target_vol: float, cash: str = "SHY") -> pd.Series:
    """Ramène un portefeuille à une volatilité cible SANS levier : w' = k·w + (1 − k)·cash,
    avec k = min(1, σ_cible / σ_w). Permet de comparer des stratégies à risque égal."""
    w = w.reindex(cov.index).fillna(0)
    vol = float(np.sqrt(w @ cov.values @ w))
    k = min(1.0, target_vol / vol) if vol > 0 else 1.0
    out = k * w
    if cash in out.index:
        out[cash] += 1 - k
    return out / out.sum()
