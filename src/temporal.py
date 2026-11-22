"""
Apprentissage supervisé sur données temporelles (cours 2, « Apprentissage supervisé :
cas des données temporelles »).

Ce module regroupe :
1. la description de la préparation par fenêtre glissante (lookback L, horizon H,
   stride S, input size) ;
2. les trois façons d'évaluer un modèle de prévision vues en cours :
   - hold-out temporel (train / validation / test chronologiques),
   - validation croisée à fenêtre croissante (expanding window),
   - validation croisée à fenêtre glissante (rolling / sliding window),
   toutes avec purge (embargo) car les cibles à 21 jours se chevauchent ;
3. les mesures de performance : MSE, RMSE, MAE (cours), coefficient d'information
   avec t-stat de Newey-West, spread Q5 − Q1 net de coûts ;
4. le test de Diebold-Mariano (1995) pour comparer la capacité prédictive de deux
   modèles.

Toutes les découpes se font sur les DATES et jamais sur les lignes : les 25 actifs
d'une même date restent toujours dans le même ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import HORIZON, SAMPLE_EVERY

# Nombre de pas d'échantillonnage couverts par une cible : ⌈H / S⌉ = ⌈21 / 5⌉ = 5.
# Deux cibles séparées de moins de 5 pas se chevauchent ; c'est l'« horizon » h utilisé
# dans Newey-West et Diebold-Mariano (h − 1 = 4 autocovariances).
H_STEPS = int(np.ceil(HORIZON / SAMPLE_EVERY))


# ---------------------------------------------------------------------------
# 1. Fenêtre glissante : vocabulaire du cours
# ---------------------------------------------------------------------------
def window_table(lookback: int, horizon: int, stride: int, input_size: int) -> pd.DataFrame:
    """Tableau « Terme / Définition / Valeur » du cours (cours 2, diapo 7)."""
    rows = [
        ("Lookback (L)", "Nombre de pas de temps passés utilisés en entrée",
         f"{lookback} jours de bourse (variable la plus longue : momentum 12 mois)"),
        ("Forecast horizon (H)", "Nombre de pas de temps futurs à prédire",
         f"{horizon} jours de bourse, méthode directe à H (on prédit seulement ŷ(t+H))"),
        ("Stride (S)", "Décalage de la fenêtre entre deux échantillons",
         f"{stride} jours (un échantillon par semaine)"),
        ("Input size", "Nombre de variables (features) par pas de temps", f"{input_size}"),
        ("Chevauchement", "Nombre d'échantillons dont les cibles se recouvrent",
         f"⌈H/S⌉ − 1 = {int(np.ceil(horizon / stride)) - 1} (corrigé par Newey-West et Diebold-Mariano)"),
    ]
    return pd.DataFrame(rows, columns=["Terme", "Définition", "Valeur dans le projet"]).set_index("Terme")


# ---------------------------------------------------------------------------
# 2. Découpages temporels purgés
# ---------------------------------------------------------------------------
class TemporalCV:
    """Validation croisée temporelle purgée, sur les dates.

    mode = "expanding" : la fenêtre d'apprentissage grandit à chaque pli
                          (toutes les dates passées sont utilisées) ;
    mode = "rolling"   : la fenêtre d'apprentissage garde une taille fixe
                          (`train_size` dates) et glisse dans le temps.

    Entre la fin du bloc d'apprentissage et le début du bloc de validation, `embargo`
    dates échantillonnées sont supprimées : leurs cibles (21 jours futurs) déborderaient
    sur le bloc de validation (fuite d'information).

    Compatible scikit-learn : GridSearchCV(..., cv=TemporalCV(...)).fit(X, y, groups=dates).
    """

    def __init__(self, n_splits: int = 5, mode: str = "expanding", train_size: int | None = None,
                 embargo: int = H_STEPS):
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode doit valoir 'expanding' ou 'rolling'")
        self.n_splits = n_splits
        self.mode = mode
        self.train_size = train_size
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Passer groups=dates à la méthode fit() / split().")
        dates = np.asarray(groups)
        udates = np.unique(dates)
        # n_splits + 1 blocs de même taille : le 1er sert seulement à l'apprentissage
        blocks = np.array_split(udates, self.n_splits + 1)
        for k in range(1, self.n_splits + 1):
            train_dates = np.concatenate(blocks[:k])
            if self.embargo:
                train_dates = train_dates[:-self.embargo]
            if self.mode == "rolling":
                size = self.train_size or len(blocks[0])
                train_dates = train_dates[-size:]
            test_dates = blocks[k]
            yield (np.flatnonzero(np.isin(dates, train_dates)),
                   np.flatnonzero(np.isin(dates, test_dates)))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def describe_splits(cv: TemporalCV, dates) -> pd.DataFrame:
    """Période d'apprentissage et de validation de chaque pli (pour le notebook)."""
    d = pd.DatetimeIndex(np.asarray(dates))
    rows = []
    for k, (tr, te) in enumerate(cv.split(None, groups=d), start=1):
        rows.append({"pli": k, "début train": d[tr].min().date(), "fin train": d[tr].max().date(),
                     "nb dates train": d[tr].nunique(), "début val": d[te].min().date(),
                     "fin val": d[te].max().date(), "nb dates val": d[te].nunique()})
    return pd.DataFrame(rows).set_index("pli")


# ---------------------------------------------------------------------------
# 3. Mesures de performance
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """MSE, RMSE, MAE du cours (cours 2, diapo 9).

    Le MAPE n'est pas calculé : la cible (rendement excédentaire) vaut souvent presque 0,
    et diviser par y_t ferait exploser la mesure.
    R² hors échantillon : 1 − MSE(modèle) / MSE(prévision nulle). Il est négatif si le
    modèle fait moins bien que prédire 0.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    e = y - p
    mse = float(np.mean(e ** 2))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(e))),
            "R² hors échantillon": float(1 - mse / np.mean(y ** 2))}


def information_coefficient(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """IC = corrélation de rang (Spearman) transversale, date par date, entre la
    prévision et le rendement réalisé."""
    df = pd.DataFrame({"s": scores, "r": fwd_ret}).dropna()
    return df.groupby(level="date").apply(
        lambda g: g["s"].rank().corr(g["r"].rank()) if len(g) > 4 else np.nan).dropna()


def newey_west_tstat(x, lags: int = H_STEPS - 1) -> float:
    """t-statistique de la moyenne, avec variance corrigée de l'autocorrélation
    (Newey-West, noyau de Bartlett). Nécessaire car les cibles se chevauchent."""
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < lags + 2:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def quintile_spread(scores: pd.Series, fwd_ret: pd.Series, q: float = 0.2,
                    cost_bps: float = 10.0, step: int = H_STEPS) -> pd.DataFrame:
    """Portefeuille long-short « quintile supérieur − quintile inférieur », net de coûts.

    On ne garde qu'une date sur `step` (≈ tous les 21 jours de bourse) pour que les
    périodes de détention ne se chevauchent pas. À chaque date : achat à poids égaux des
    actifs du quintile supérieur, vente du quintile inférieur. Le coût vaut
    rotation × cost_bps, où la rotation est la somme des |variations de poids| des deux
    jambes.
    Retourne un DataFrame (date → brut, rotation, net).
    """
    df = pd.DataFrame({"s": scores, "r": fwd_ret}).dropna()
    dates = df.index.get_level_values("date").unique().sort_values()[::step]
    prev = pd.Series(dtype=float)
    rows = []
    for d in dates:
        g = df.xs(d, level="date")
        if g["s"].nunique() <= 1:   # prévision constante : aucun classement possible
            rows.append({"date": d, "brut": np.nan, "rotation": np.nan, "net": np.nan})
            continue
        hi = g["s"] >= g["s"].quantile(1 - q)
        lo = g["s"] <= g["s"].quantile(q)
        w = pd.Series(0.0, index=g.index)
        w[hi] = 1 / hi.sum()
        w[lo] = -1 / lo.sum()
        turnover = float(w.sub(prev, fill_value=0).abs().sum()) if len(prev) else float(w.abs().sum())
        gross = float((w * g["r"]).sum())
        rows.append({"date": d, "brut": gross, "rotation": turnover,
                     "net": gross - turnover * cost_bps / 1e4})
        prev = w
    return pd.DataFrame(rows).set_index("date")


def forecast_report(pred: pd.Series, data: pd.DataFrame, target: str = "target_reg",
                    fwd_col: str = "fwd_ret") -> dict:
    """Toutes les mesures d'une prévision de rendement sur un ensemble (date, ticker)."""
    pred = pd.Series(np.asarray(pred, float), index=data.index)
    out = regression_metrics(data[target], pred)
    ic = information_coefficient(pred, data[fwd_col])
    ls = quintile_spread(pred, data[fwd_col])
    periods_per_year = 252 / (SAMPLE_EVERY * H_STEPS)
    out.update({
        "IC moyen": float(ic.mean()),
        "IC t-stat (Newey-West)": newey_west_tstat(ic),
        "% dates IC>0": float((ic > 0).mean()),
        "Spread Q5-Q1 net (annualisé)": float(ls["net"].mean() * periods_per_year),
        "Spread Q5-Q1 brut (annualisé)": float(ls["brut"].mean() * periods_per_year),
    })
    return out


def evaluate_three_ways(pipe, train: pd.DataFrame, valid: pd.DataFrame, feats: list,
                        target: str = "target_reg", n_splits: int = 5,
                        rolling_size: int | None = None) -> pd.DataFrame:
    """Évalue UN modèle (hyperparamètres fixés) selon les trois approches du cours 2 :

    1. hold-out temporel : apprentissage sur `train`, évaluation sur `valid` ;
    2. CV expanding     : n_splits plis sur train + valid, fenêtre d'apprentissage croissante ;
    3. CV rolling       : mêmes blocs de validation, fenêtre d'apprentissage de taille fixe.

    Les blocs de validation des deux CV sont identiques : seule la mémoire du modèle change.
    La période de test n'est jamais utilisée ici.
    Retourne une ligne par approche (moyenne et écart-type des plis pour les CV).
    """
    from sklearn.base import clone

    def _score(fitted, data):
        p = fitted.predict(data[feats])
        m = regression_metrics(data[target], p)
        m["IC moyen"] = float(information_coefficient(pd.Series(p, index=data.index),
                                                      data["fwd_ret"]).mean())
        return m

    rows = {"Hold-out temporel": {**_score(clone(pipe).fit(train[feats], train[target]), valid),
                                  "nb plis": 1}}
    full = pd.concat([train, valid])
    dates = full.index.get_level_values("date")
    for label, mode in (("CV expanding window", "expanding"), ("CV rolling window", "rolling")):
        cv = TemporalCV(n_splits, mode, train_size=rolling_size)
        folds = [_score(clone(pipe).fit(full.iloc[a][feats], full.iloc[a][target]), full.iloc[b])
                 for a, b in cv.split(None, groups=dates)]
        f = pd.DataFrame(folds)
        rows[label] = {**f.mean().to_dict(), "écart-type MSE": f["MSE"].std(),
                       "écart-type IC": f["IC moyen"].std(), "nb plis": len(folds)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 4. Test de Diebold-Mariano (cours 2, diapos 15-16)
# ---------------------------------------------------------------------------
def loss_by_date(y_true: pd.Series, y_pred, loss: str = "mse") -> pd.Series:
    """Perte moyenne transversale à chaque date : une observation par date.

    Le test DM porte sur une SÉRIE TEMPORELLE de pertes. Avec un panel (date, actif),
    on agrège donc les 25 erreurs d'une même date en une seule perte.
    """
    e = pd.Series(np.asarray(y_true, float) - np.asarray(y_pred, float), index=y_true.index)
    f = (e ** 2) if loss == "mse" else e.abs()
    return f.groupby(level="date").mean()


def diebold_mariano(loss_1: pd.Series, loss_2: pd.Series, h: int = H_STEPS,
                    small_sample: bool = True) -> dict:
    """Test de Diebold-Mariano.

    d_t = L(e_1,t) − L(e_2,t)        (différentiel de perte)
    H0 : E[d_t] = 0  (même précision)   H1 : E[d_t] ≠ 0
    DM = d̄ / sqrt( Var(d̄) ),   Var(d̄) = (γ̂_0 + 2 Σ_{k=1}^{h−1} γ̂_k) / T

    Lecture (cours) : DM > 0 et p < α → le modèle 2 est meilleur ;
                      DM < 0 et p < α → le modèle 1 est meilleur ;
                      p > α → les deux modèles sont statistiquement équivalents.

    small_sample=True applique la correction de Harvey, Leybourne et Newbold (1997),
    qui compare la statistique à une loi de Student à T − 1 degrés de liberté.
    """
    d = (loss_1 - loss_2).dropna()
    T = len(d)
    if np.allclose(d.values, 0):   # prévisions identiques : rien à tester
        return {"DM": np.nan, "p-valeur": np.nan, "d moyen": 0.0, "T": T}
    dc = d - d.mean()
    gamma = [float(dc @ dc) / T] + [float(dc.values[k:] @ dc.values[:-k]) / T for k in range(1, h)]
    var_dbar = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var_dbar <= 0:   # la formule du cours peut donner une variance négative : repli sur Bartlett
        var_dbar = (gamma[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gamma[1:], start=1))) / T
    dm = float(d.mean() / np.sqrt(var_dbar))
    if small_sample:
        dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        p = float(2 * stats.t.sf(abs(dm), df=T - 1))
    else:
        p = float(2 * stats.norm.sf(abs(dm)))
    return {"DM": dm, "p-valeur": p, "d moyen": float(d.mean()), "T": T}


def dm_table(losses: dict[str, pd.Series], reference: str, alpha: float = 0.05) -> pd.DataFrame:
    """Compare chaque modèle à une référence (modèle 1 = référence, modèle 2 = candidat).
    DM > 0 et p < α : le candidat bat la référence de façon significative."""
    rows = {}
    for name, l in losses.items():
        if name == reference:
            continue
        r = diebold_mariano(losses[reference], l)
        verdict = ("prévisions identiques" if np.isnan(r["DM"]) else
                   "meilleur que la référence" if r["DM"] > 0 and r["p-valeur"] < alpha else
                   "moins bon que la référence" if r["DM"] < 0 and r["p-valeur"] < alpha else
                   "équivalent (H0 non rejetée)")
        rows[name] = {**r, "verdict": verdict}
    return pd.DataFrame(rows).T.sort_values("DM", ascending=False)
