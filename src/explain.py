"""
Interprétabilité et explicabilité (cours 1, « Interprétabilité et explicabilité »).

- Explicabilité GLOBALE : comportement moyen du modèle sur l'ensemble des données
  (moyenne des |valeurs SHAP| par variable).
- Explicabilité LOCALE : explication d'UNE prédiction (un client, un actif à une date) :
      prédiction_i = valeur de base + Σ_j contribution_j      (valeurs de Shapley)

Les modèles sont des Pipeline (prétraitement → modèle). SHAP est calculé dans l'espace
transformé, puis les colonnes issues d'une même variable (one-hot) sont regroupées pour que
l'explication porte sur les variables d'origine, compréhensibles par un client.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:      # SHAP facultatif : l'application fonctionne sans
    shap = None

TREE_TYPES = ("LGBM", "XGB", "CatBoost", "RandomForest", "GradientBoosting", "DecisionTree", "ExtraTrees")


def _split_pipeline(pipe):
    """Sépare la partie « transformations » et le modèle final d'un Pipeline."""
    steps = pipe.steps
    return steps[:-1], steps[-1][1]


def _transform(pre_steps, X: pd.DataFrame) -> pd.DataFrame:
    Xt, names = X, list(X.columns)
    for _, step in pre_steps:
        if not hasattr(step, "transform"):      # ex. SMOTE : n'intervient qu'à l'entraînement
            continue
        Xt = step.transform(Xt)
        names = list(step.get_feature_names_out()) if hasattr(step, "get_feature_names_out") else \
            [f"c{i}" for i in range(np.asarray(Xt).shape[1])]
    return pd.DataFrame(np.asarray(Xt, dtype=float), columns=names, index=X.index)


def original_name(transformed: str, originals: list[str]) -> str:
    """'cat__EDCL_4' → 'EDCL' ; 'num__AGE' → 'AGE' (on garde la variable d'origine la plus longue
    dont le nom est contenu dans la colonne transformée)."""
    base = transformed.split("__", 1)[-1]
    cands = [o for o in originals if base == o or base.startswith(o + "_")]
    return max(cands, key=len) if cands else base


def make_explainer(pipe, X_background: pd.DataFrame, n_background: int = 200, seed: int = 0):
    """Crée l'explicateur SHAP adapté : TreeExplainer (arbres), LinearExplainer (linéaire),
    sinon méthode générique (plus lente) sur un échantillon de fond."""
    if shap is None:
        return None
    pre_steps, model = _split_pipeline(pipe)
    bg = _transform(pre_steps, X_background.sample(min(n_background, len(X_background)), random_state=seed))
    kind = type(model).__name__
    if any(t in kind for t in TREE_TYPES):
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, bg)
    f = (lambda a: model.predict_proba(a)[:, 1]) if hasattr(model, "predict_proba") else model.predict
    return shap.Explainer(f, bg)


def shap_values(pipe, explainer, X: pd.DataFrame, class_idx: int = 1, group: bool = True) -> pd.DataFrame:
    """Valeurs SHAP (une ligne par observation), regroupées par variable d'origine si group=True.
    Pour un classifieur, on explique la probabilité (ou le log-odds) de la classe `class_idx`."""
    pre_steps, _ = _split_pipeline(pipe)
    Xt = _transform(pre_steps, X)
    sv = explainer(Xt)
    vals = sv.values
    if vals.ndim == 3:
        vals = vals[:, :, class_idx]
    df = pd.DataFrame(vals, columns=Xt.columns, index=X.index)
    if group:
        df = df.T.groupby([original_name(c, list(X.columns)) for c in df.columns]).sum().T
    base = sv.base_values
    base = np.asarray(base)[..., class_idx] if np.ndim(base) > 1 else np.asarray(base)
    df.attrs["base_value"] = float(np.mean(base))
    return df


def global_importance(sv: pd.DataFrame) -> pd.Series:
    """Explicabilité globale : moyenne des |valeurs SHAP| par variable (décroissante)."""
    return sv.abs().mean().sort_values(ascending=False)


def local_explanation(sv_row: pd.Series, X_row: pd.Series, labels: dict | None = None,
                      top: int = 8) -> pd.DataFrame:
    """Explicabilité locale d'une observation : les `top` contributions les plus fortes,
    avec la valeur de la variable et le sens de l'effet (pour la cascade / waterfall)."""
    s = sv_row.reindex(sv_row.abs().sort_values(ascending=False).index)
    rows = []
    for var, contrib in s.head(top).items():
        rows.append({"variable": var, "libellé": (labels or {}).get(var, var),
                     "valeur": X_row.get(var, np.nan), "contribution": float(contrib),
                     "effet": "augmente" if contrib > 0 else "diminue"})
    reste = float(s.iloc[top:].sum())
    if len(s) > top:
        rows.append({"variable": "autres", "libellé": f"{len(s) - top} autres variables", "valeur": np.nan,
                     "contribution": reste, "effet": "augmente" if reste > 0 else "diminue"})
    return pd.DataFrame(rows)
