"""
Backtest « walk-forward » des stratégies d'allocation et mesures de performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .config import BOND_PROXY, MARKET_TICKER, TRANSACTION_COST_BPS


# ---------------------------------------------------------------------------
# Mesures de performance
# ---------------------------------------------------------------------------
def perf_metrics(r: pd.Series, rf_daily: pd.Series | float = 0.0, turnover: float | None = None) -> dict:
    r = r.dropna()
    rf = rf_daily.reindex(r.index).fillna(0) if isinstance(rf_daily, pd.Series) else rf_daily
    ex = r - rf
    n = len(r)
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    downside = ex[ex < 0].std() * np.sqrt(252)
    mdd = (wealth / wealth.cummax() - 1).min()
    out = {
        "Rendement annualisé": cagr,
        "Volatilité annualisée": vol,
        "Sharpe": ex.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "Sortino": ex.mean() * 252 / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "VaR 95% (jour)": -np.quantile(r, 0.05),
        "% jours positifs": (r > 0).mean(),
    }
    if turnover is not None:
        out["Rotation annuelle"] = turnover
    return out


# ---------------------------------------------------------------------------
# Simulation d'un portefeuille rebalancé périodiquement
# ---------------------------------------------------------------------------
def simulate(weights: dict, returns: pd.DataFrame, tc_bps: float = TRANSACTION_COST_BPS):
    """weights : {date_de_rebalancement: pd.Series de poids cibles}.
    Les poids dérivent entre deux rebalancements avec les prix ; le coût de
    transaction est prélevé le jour suivant le rebalancement.
    Retourne (rendements quotidiens nets, rotation annualisée)."""
    dates = sorted(weights)
    rets = returns.loc[dates[0]:].fillna(0)
    out = pd.Series(0.0, index=rets.index[1:])
    w_cur = pd.Series(0.0, index=returns.columns)
    total_turnover = 0.0
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else rets.index[-1]
        target = weights[d].reindex(returns.columns).fillna(0)
        trade = (target - w_cur).abs().sum()
        total_turnover += trade
        w = target.copy()
        period = rets.loc[d:end].iloc[1:]
        first = True
        for day, r in period.iterrows():
            rp = float((w * r).sum())
            if first:
                rp -= trade * tc_bps / 1e4
                first = False
            out[day] = rp
            w = w * (1 + r) / (1 + float((w * r).sum()))
        w_cur = w
    years = len(out) / 252
    return out, total_turnover / years if years > 0 else np.nan


# ---------------------------------------------------------------------------
# Stratégies de référence
# ---------------------------------------------------------------------------
def equal_weight(available: list[str]) -> pd.Series:
    return pd.Series(1 / len(available), index=available)


def sixty_forty(available: list[str]) -> pd.Series:
    return pd.Series({MARKET_TICKER: 0.6, BOND_PROXY: 0.4})


def min_variance(returns_window: pd.DataFrame, w_max: float = 0.25) -> pd.Series:
    R = returns_window.dropna(axis=1, thresh=int(0.9 * len(returns_window))).dropna()
    S = LedoitWolf().fit(R.values).covariance_ * 252
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w,
                   bounds=[(0, w_max)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP")
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=R.columns)


def markowitz_historical(returns_window: pd.DataFrame, risk_aversion: float = 5.0,
                         w_max: float = 0.25) -> pd.Series:
    """Markowitz classique avec rendements espérés = moyenne historique sur 1 an
    (référence connue pour son instabilité)."""
    R = returns_window.dropna(axis=1, thresh=int(0.9 * len(returns_window))).dropna()
    mu = R.mean().values * 252
    S = LedoitWolf().fit(R.values).covariance_ * 252
    n = len(mu)
    res = minimize(lambda w: -(w @ mu) + 0.5 * risk_aversion * w @ S @ w, np.full(n, 1 / n),
                   jac=lambda w: -mu + risk_aversion * S @ w, bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=R.columns)


def rebalance_dates(index: pd.DatetimeIndex, start, end=None) -> list:
    """Dernier jour de bourse de chaque mois entre start et end."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end or index[-1]))]
    s = pd.Series(idx, index=idx)
    return list(pd.to_datetime(s.groupby(idx.to_period("M")).last().values))


def vol_targeted_equal_weight(returns_window: pd.DataFrame, target_vol: float,
                              cash: str = "SHY") -> pd.Series:
    """Benchmark 1/N ramené à une volatilité cible (sans levier).

    On combine le portefeuille équipondéré et un actif quasi monétaire (bons du
    Trésor 1-3 ans) : w = k·(1/N) + (1 − k)·cash, avec k = min(1, σ_cible / σ_1/N),
    σ_1/N estimée par Ledoit-Wolf sur la fenêtre. Permet de comparer chaque profil
    à un 1/N de MÊME niveau de risque.
    """
    R = returns_window.dropna(axis=1, thresh=int(0.9 * len(returns_window))).dropna()
    S = LedoitWolf().fit(R.values).covariance_ * 252
    n = R.shape[1]
    ew = np.full(n, 1 / n)
    vol_ew = float(np.sqrt(ew @ S @ ew))
    k = min(1.0, target_vol / vol_ew) if vol_ew > 0 else 1.0
    w = pd.Series(k * ew, index=R.columns)
    w[cash] = w.get(cash, 0.0) + (1 - k)
    return w / w.sum()
