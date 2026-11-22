"""
Catalogue des modèles de l'agent Prévision (tâche de régression) et de leurs grilles
d'hyperparamètres.

Chaque modèle est placé dans un `Pipeline` scikit-learn :
    prétraitement (imputation → écrêtage → standardisation | one-hot)
    → [ACP optionnelle] → modèle
Le pipeline entier est ajusté sur l'entraînement uniquement : aucune statistique de la
validation ou du test n'est utilisée pour le prétraitement (prévention des fuites).

Les grilles sont volontairement petites et centrées sur la régularisation : avec un
signal faible (rendements financiers), le risque principal est le SUR-APPRENTISSAGE
(forte variance), et le cours recommande alors d'augmenter la pénalisation, de simplifier
le modèle ou de réduire la dimension.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE
from .preprocessing import Winsorizer

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import catboost as cb
except ImportError:
    cb = None


def preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """Prétraitement commun (cours : imputation, standardisation, one-hot encoding)."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("win", Winsorizer(0.005, 0.995)),
                          ("sc", StandardScaler())]), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])


def make_pipeline(model, num_features: list, cat_features: list, n_pca: int | None = None) -> Pipeline:
    """Pipeline complet du cours (cours 2, diapo 37), avec réduction de dimension optionnelle."""
    steps = [("pre", preprocessor(num_features, cat_features))]
    if n_pca:
        steps.append(("acp", PCA(n_components=n_pca, random_state=RANDOM_STATE)))
    steps.append(("m", model))
    return Pipeline(steps)


def regression_models(quick: bool = False) -> dict:
    """{nom: (famille, estimateur, grille)}. Les clés de grille portent le préfixe « m__ »."""
    rs = RANDOM_STATE
    m = {
        # --- Modèles linéaires (cours 1 : régression et régularisation) ---
        "Régression linéaire (MCO)": ("Linéaire", LinearRegression(), {}),
        "Ridge": ("Linéaire régularisé", Ridge(),
                  {"m__alpha": np.logspace(1, 5, 9)}),
        "Lasso": ("Linéaire régularisé", Lasso(max_iter=20000),
                  {"m__alpha": np.logspace(-5, -2, 7)}),
        "Elastic Net": ("Linéaire régularisé", ElasticNet(max_iter=20000),
                        {"m__alpha": np.logspace(-5, -2, 7), "m__l1_ratio": [0.2, 0.5, 0.8]}),
        # --- Voisinage et noyaux ---
        "KNN": ("Voisinage", KNeighborsRegressor(),
                {"m__n_neighbors": [50, 200, 500], "m__weights": ["uniform"]}),
        "SVR (noyau RBF)": ("Noyau", SVR(kernel="rbf", cache_size=1000),
                            {"m__C": [0.01, 0.1], "m__epsilon": [0.02, 0.05], "m__gamma": ["scale"]}),
        # --- Arbres et ensembles ---
        "Arbre de décision": ("Arbre", DecisionTreeRegressor(random_state=rs),
                              {"m__max_depth": [2, 3, 5], "m__min_samples_leaf": [200, 500]}),
        "Forêt aléatoire": ("Ensemble (bagging)",
                            RandomForestRegressor(n_estimators=200 if not quick else 50, n_jobs=1, random_state=rs),
                            {"m__max_depth": [3, 6], "m__min_samples_leaf": [200, 500],
                             "m__max_features": [0.3]}),
        "Gradient Boosting": ("Ensemble (boosting)", HistGradientBoostingRegressor(random_state=rs),
                              {"m__learning_rate": [0.02, 0.05], "m__max_leaf_nodes": [7, 15],
                               "m__min_samples_leaf": [200, 500], "m__max_iter": [200],
                               "m__l2_regularization": [1.0]}),
        # --- Réseau de neurones (cours 1 : descente de gradient, Adam) ---
        "MLP (Adam)": ("Réseau de neurones",
                       MLPRegressor(solver="adam", early_stopping=True, max_iter=300, random_state=rs),
                       {"m__hidden_layer_sizes": [(32,), (64, 32)], "m__alpha": [1e-2, 1e-1, 1.0],
                        "m__learning_rate_init": [1e-3]}),
    }
    if lgb is not None:
        m["LightGBM"] = ("Ensemble (boosting)",
                         lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=rs, subsample_freq=1,
                                           n_estimators=300, subsample=0.8, colsample_bytree=0.7),
                         {"m__learning_rate": [0.01, 0.03], "m__num_leaves": [7, 15],
                          "m__min_child_samples": [200, 500], "m__reg_lambda": [1, 10]})
    if xgb is not None:
        m["XGBoost"] = ("Ensemble (boosting)",
                        xgb.XGBRegressor(tree_method="hist", n_jobs=1, random_state=rs, n_estimators=300,
                                         subsample=0.8, colsample_bytree=0.7),
                        {"m__learning_rate": [0.01, 0.03], "m__max_depth": [2, 3],
                         "m__min_child_weight": [50, 200], "m__reg_lambda": [1, 10]})
    if cb is not None:
        m["CatBoost"] = ("Ensemble (boosting)",
                         cb.CatBoostRegressor(verbose=0, random_seed=rs, iterations=300, thread_count=1,
                                              allow_writing_files=False),
                         {"m__learning_rate": [0.02, 0.05], "m__depth": [3, 5], "m__l2_leaf_reg": [3, 30]})
    if quick:   # test rapide : on garde seulement le 1er point de chaque grille
        m = {k: (f, e, {p: v[:1] for p, v in g.items()}) for k, (f, e, g) in m.items()}
    return m
