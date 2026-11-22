"""
Ingénierie des variables pour l'agent Prévision et l'agent Régime, découpage
temporel avec purge/embargo et validation croisée temporelle purgée.

Toutes les variables à la date t sont calculées avec des informations
disponibles au plus tard à la clôture de t. La cible utilise t+1..t+HORIZON.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ASSET_CLASS, EMBARGO_DAYS, HORIZON, MARKET_TICKER,
                     SAMPLE_EVERY, TRAIN_END, VAL_END)

ASSET_NUM_FEATURES = [
    "ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1",
    "vol_21", "vol_63", "vol_ratio", "drawdown_252", "ma50_gap", "ma200_gap",
    "rsi_14", "skew_63", "beta_63", "corr_63",
    "ret_21_rank", "ret_63_rank", "mom_12_1_rank", "vol_63_rank",
]
MACRO_FEATURES = [
    "VIX", "VIX_chg21", "T10Y2Y", "BAA10Y", "BAA10Y_chg63", "DGS10_chg63",
    "CPI_yoy", "UNRATE", "UNRATE_chg3m", "INDPRO_yoy",
]
CAT_FEATURES = ["asset_class"]
# Variables STATIONNAIRES uniquement : les niveaux de la pente de courbe et du spread de crédit
# ont dérivé hors de leur plage d'entraînement après 2022 (courbe inversée), ce qui faisait
# classer 91 % de la période de test en « Stress ». On utilise des logs et des variations.
REGIME_FEATURES = ["mkt_ret_21", "mkt_logvol_21", "mkt_drawdown", "logVIX", "credit_chg63"]


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index de Wilder."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _stack(df: pd.DataFrame) -> pd.Series:
    return df.rename_axis(index="date", columns="ticker").stack()


def compute_asset_features(prices: pd.DataFrame, macro: pd.DataFrame,
                           market: str = MARKET_TICKER, horizon: int = HORIZON) -> pd.DataFrame:
    """Panel long (date, ticker) : variables techniques + macro + cible.

    Cible : `target` = 1 si le rendement de l'actif sur les `horizon` prochains
    jours dépasse la médiane transversale des actifs disponibles (surperformance
    relative), 0 sinon. `fwd_ret` conserve le rendement futur pour l'évaluation
    financière (corrélation de rang, spread long-short).
    """
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)
    f = {}
    for w in (5, 21, 63, 126, 252):
        f[f"ret_{w}"] = prices / prices.shift(w) - 1
    f["mom_12_1"] = prices.shift(21) / prices.shift(252) - 1      # momentum 12-1 mois
    f["vol_21"] = rets.rolling(21, min_periods=15).std() * np.sqrt(252)
    f["vol_63"] = rets.rolling(63, min_periods=45).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["drawdown_252"] = prices / prices.rolling(252, min_periods=126).max() - 1
    f["ma50_gap"] = prices / prices.rolling(50, min_periods=40).mean() - 1
    f["ma200_gap"] = prices / prices.rolling(200, min_periods=150).mean() - 1
    f["rsi_14"] = rsi(prices, 14)
    f["skew_63"] = rets.rolling(63, min_periods=45).skew()
    mkt = rets[market]
    mkt_var = mkt.rolling(63, min_periods=45).var()
    f["beta_63"] = pd.DataFrame({c: rets[c].rolling(63, min_periods=45).cov(mkt) / mkt_var
                                 for c in rets.columns})
    f["corr_63"] = pd.DataFrame({c: rets[c].rolling(63, min_periods=45).corr(mkt)
                                 for c in rets.columns})
    # Rangs transversaux (position relative de l'actif parmi l'univers à la date t)
    for k in ("ret_21", "ret_63", "mom_12_1", "vol_63"):
        f[f"{k}_rank"] = f[k].rank(axis=1, pct=True)

    panel = pd.concat({k: _stack(v) for k, v in f.items()}, axis=1)
    fwd = prices.shift(-horizon) / prices - 1
    panel["fwd_ret"] = _stack(fwd)
    panel = panel.reset_index()
    panel["asset_class"] = panel["ticker"].map(ASSET_CLASS)

    # Cible principale (RÉGRESSION) : rendement excédentaire relatif à la moyenne transversale
    #   y_{i,t} = r_{i,t→t+H} − moyenne_j r_{j,t→t+H}
    # On retire l'effet « marché » commun à tous les actifs : seul compte le classement
    # relatif, qui est l'information utile pour choisir ENTRE les actifs.
    mean_cs = panel.groupby("date")["fwd_ret"].transform("mean")
    panel["target_reg"] = panel["fwd_ret"] - mean_cs
    # Cible secondaire (CLASSIFICATION) : surperformance vs médiane transversale
    med = panel.groupby("date")["fwd_ret"].transform("median")
    panel["target"] = np.where(panel["fwd_ret"].isna(), np.nan,
                               (panel["fwd_ret"] > med).astype(float))

    macro_cols = [c for c in MACRO_FEATURES if c in macro.columns]
    panel = panel.merge(macro[macro_cols], left_on="date", right_index=True, how="left")
    # On ne garde que les lignes où l'actif cote et où les variables principales existent
    panel = panel.dropna(subset=["ret_252", "vol_63", "ma200_gap"])
    return panel.set_index(["date", "ticker"]).sort_index()


def compute_market_features(prices: pd.DataFrame, macro: pd.DataFrame,
                            market: str = MARKET_TICKER) -> pd.DataFrame:
    """Variables de l'état du marché (agent Régime), une ligne par date."""
    p = prices[market]
    r = p.pct_change(fill_method=None)
    out = pd.DataFrame(index=prices.index)
    out["mkt_ret_21"] = p / p.shift(21) - 1
    out["mkt_vol_21"] = r.rolling(21, min_periods=15).std() * np.sqrt(252)
    out["mkt_drawdown"] = p / p.rolling(252, min_periods=126).max() - 1
    out["mkt_logvol_21"] = np.log(out["mkt_vol_21"].clip(lower=1e-4))
    if "VIX" in macro:
        out["VIX"] = macro["VIX"]
        out["logVIX"] = np.log(macro["VIX"].clip(lower=1))
    if "BAA10Y_chg63" in macro:
        out["credit_chg63"] = macro["BAA10Y_chg63"]
    for c in ("T10Y2Y", "BAA10Y"):
        if c in macro:
            out[c] = macro[c]
    return out


def sample_dates(panel: pd.DataFrame, every: int = SAMPLE_EVERY) -> pd.DataFrame:
    """Sous-échantillonnage hebdomadaire pour limiter la redondance des
    étiquettes qui se chevauchent (horizon 21 jours)."""
    dates = panel.index.get_level_values("date").unique().sort_values()
    keep = dates[::every]
    return panel[panel.index.get_level_values("date").isin(keep)]


def temporal_split(panel: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END,
                   embargo_days: int = EMBARGO_DAYS):
    """Découpage chronologique train / validation / test AVEC purge.

    Les dernières observations de chaque bloc dont la cible (fenêtre future de
    21 jours) déborderait sur le bloc suivant sont supprimées (embargo). Sans
    cela, la cible d'entraînement contiendrait des rendements de la période
    de validation : fuite d'information.
    """
    d = panel.index.get_level_values("date")
    t_end, v_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    emb = pd.Timedelta(days=embargo_days)
    train = panel[d <= t_end - emb]
    val = panel[(d > t_end) & (d <= v_end - emb)]
    test = panel[d > v_end]
    return train, val, test


class PurgedTimeSeriesSplit:
    """Validation croisée temporelle à fenêtre croissante avec embargo.

    Les plis sont définis sur les DATES (et non les lignes) : toutes les
    observations d'une même date restent dans le même pli. Entre la fin du
    bloc d'apprentissage et le début du bloc de validation, `embargo` dates
    échantillonnées sont retirées (purge des étiquettes chevauchantes).
    S'utilise avec GridSearchCV(..., cv=PurgedTimeSeriesSplit()) et
    .fit(X, y, groups=dates).
    """

    def __init__(self, n_splits: int = 5, embargo: int = int(np.ceil(HORIZON / SAMPLE_EVERY))):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Passer groups=dates à la méthode fit().")
        dates = np.asarray(groups)
        udates = np.unique(dates)
        folds = np.array_split(udates, self.n_splits + 1)
        for k in range(1, self.n_splits + 1):
            train_dates = np.concatenate(folds[:k])
            if self.embargo:
                train_dates = train_dates[:-self.embargo]
            test_dates = folds[k]
            yield (np.where(np.isin(dates, train_dates))[0],
                   np.where(np.isin(dates, test_dates))[0])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ---------------------------------------------------------------------------
# Métriques financières des prévisions
# ---------------------------------------------------------------------------
def information_coefficient(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """IC = corrélation de rang (Spearman) transversale, date par date,
    entre le score du modèle et le rendement futur réalisé."""
    df = pd.DataFrame({"s": scores, "r": fwd_ret}).dropna()
    return df.groupby(level="date").apply(
        lambda g: g["s"].rank().corr(g["r"].rank()) if len(g) > 4 else np.nan).dropna()


def newey_west_tstat(x: pd.Series, lags: int = int(np.ceil(HORIZON / SAMPLE_EVERY))) -> float:
    """t-statistique de la moyenne corrigée de l'autocorrélation (Newey-West, noyau de Bartlett).
    Les IC hebdomadaires portent sur des rendements à 21 jours qui se chevauchent : les
    observations successives sont corrélées et le t-stat naïf est surestimé (~×2)."""
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < lags + 2:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def long_short_spread(scores: pd.Series, fwd_ret: pd.Series, q: float = 0.2) -> pd.Series:
    """Rendement futur moyen du quintile supérieur moins quintile inférieur."""
    df = pd.DataFrame({"s": scores, "r": fwd_ret}).dropna()

    def _spread(g):
        hi = g["s"] >= g["s"].quantile(1 - q)
        lo = g["s"] <= g["s"].quantile(q)
        return g.loc[hi, "r"].mean() - g.loc[lo, "r"].mean()

    return df.groupby(level="date").apply(_spread).dropna()
