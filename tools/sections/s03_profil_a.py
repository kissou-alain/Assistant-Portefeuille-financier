from cellules import code, md

CELLS = [
md(r"""
## 3. Agent Profil – classification supervisée de l'aversion au risque

### 3.1 Problématique et formulation
À partir de la situation d'un ménage (âge, revenus, patrimoine, dettes, éducation, connaissances financières, comportements d'épargne), on cherche à estimer **sa capacité à supporter des pertes**. La cible vient d'une question réelle du SCF : le ménage est-il prêt à prendre des risques financiers ?

- `NOFINRISK = 1` (« aucun risque ») → classe **Prudent** (≈ 32 % des ménages) ;
- sinon → classe **Non prudent**.

C'est une **classification binaire** (cours 1). La version à 3 classes (Prudent / Équilibré / Dynamique) a été testée dans la version précédente du projet : la classe « Dynamique » ne représente que 5 % des ménages et n'est pas séparable avec ces variables (F1 macro plafonné à 0,50 quel que soit le modèle). Le passage Équilibré → Dynamique est donc décidé par un **questionnaire noté de 7 questions** (horizon, réaction à une baisse, perte acceptable…), comme dans les questionnaires réglementaires MiFID II.

**Décision finale** : profil retenu = le plus prudent entre la *tolérance déclarée* (questionnaire) et la *capacité estimée* (modèle ML). Toute divergence déclenche une alerte pour le conseiller.

**Piège de fuite propre au SCF** : chaque ménage apparaît sur **5 lignes quasi identiques** (imputation multiple des valeurs manquantes). Un découpage aléatoire ligne par ligne placerait des copies d'un même ménage dans l'entraînement et dans le test. Le découpage se fait donc **par ménage**.
"""),
code(r"""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV, learning_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score,
                             accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from src.preprocessing import prepare_scf, Winsorizer, SignedLog, MONEY_FEATURES

X_p, y_p, groups_p, w_p, scf_report = prepare_scf(scf_raw)
num_p, cat_p = scf_report["variables_numeriques"], scf_report["variables_categorielles"]
print(f"{scf_report['lignes_finales']} lignes | {scf_report['menages_uniques']} ménages | "
      f"{len(num_p)} variables numériques + {len(cat_p)} catégorielles")
print("Répartition de la cible :", scf_report["repartition_cible"])
"""),
md(r"""
### 3.2 Découpage entraînement / validation / test (70 / 15 / 15) **avant** tout prétraitement
Méthode « *three-way holdout* » du cours : l'entraînement sert à ajuster les modèles, la validation à **choisir** un modèle et un seuil, et le test à mesurer une seule fois la performance du modèle choisi. Le découpage est stratifié sur la classe et fait **par ménage**.
"""),
code(r"""
hh = pd.DataFrame({"g": groups_p.values, "y": y_p.astype(str).values}).groupby("g")["y"].agg(lambda s: s.mode().iloc[0])
hh_train, hh_tmp = train_test_split(hh.index, test_size=0.30, stratify=hh.values, random_state=RANDOM_STATE)
hh_val, hh_test = train_test_split(hh_tmp, test_size=0.50, stratify=hh.loc[hh_tmp].values, random_state=RANDOM_STATE)
m_tr, m_va, m_te = (groups_p.isin(h).values for h in (hh_train, hh_val, hh_test))
assert not (set(hh_train) & set(hh_val) or set(hh_train) & set(hh_test) or set(hh_val) & set(hh_test))

# Codage : 0 = Prudent, 1 = Non prudent (ordre de PROFILE_ML_CLASSES, utilisé par l'agent)
CLS_TO_INT = {c: i for i, c in enumerate(PROFILE_ML_CLASSES)}
y_int = y_p.astype(str).map(CLS_TO_INT).astype(int)
Xp_tr, Xp_va, Xp_te = X_p[m_tr], X_p[m_va], X_p[m_te]
yp_tr, yp_va, yp_te = y_int[m_tr], y_int[m_va], y_int[m_te]
gp_tr = groups_p[m_tr]
pd.DataFrame({"lignes": [m_tr.sum(), m_va.sum(), m_te.sum()],
              "ménages": [len(hh_train), len(hh_val), len(hh_test)],
              "% Prudent": [(y == 0).mean() * 100 for y in (yp_tr, yp_va, yp_te)]},
             index=["entraînement", "validation", "test"])
"""),
md(r"""
### 3.3 Analyse exploratoire (entraînement uniquement)
"""),
code(r"""
eda = Xp_tr.copy()
eda["profil"] = yp_tr.map({v: k for k, v in CLS_TO_INT.items()})
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
eda["profil"].value_counts().reindex(PROFILE_ML_CLASSES).plot.bar(ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Classes déséquilibrées (≈ 1/3 de Prudents)")
sns.boxplot(data=eda, x="profil", y="AGE", order=PROFILE_ML_CLASSES, ax=axes[1])
axes[1].set_title("Âge par profil")
eda["log_patrimoine"] = np.sign(eda["NETWORTH"]) * np.log1p(eda["NETWORTH"].abs())
sns.boxplot(data=eda, x="profil", y="log_patrimoine", order=PROFILE_ML_CLASSES, ax=axes[2])
axes[2].set_title("log du patrimoine net")
pd.crosstab(eda["KNOWL"], eda["profil"], normalize="index")[PROFILE_ML_CLASSES].plot.bar(stacked=True, ax=axes[3], legend=False)
axes[3].set_title("Profil selon la connaissance financière")
plt.tight_layout()
savefig("06_eda_profils")
plt.show()
"""),
md(r"""
**Lecture** : les Prudents sont plus âgés, nettement moins aisés et déclarent moins de connaissances financières. Les montants (revenu, patrimoine, dettes) sont **très asymétriques**, d'où une transformation logarithmique signée $\text{sign}(x)\ln(1+|x|)$. Les classes sont **déséquilibrées** : l'*accuracy* serait trompeuse (prédire toujours « Non prudent » donne déjà 68 %). On utilise l'**AUC ROC**, le **rappel**, la **précision** et le **F1**, comme le recommande le cours.

### 3.4 Pipeline de prétraitement (cours : imputation, standardisation, one-hot encoding)
- **Montants** : imputation par la médiane → log signé → standardisation.
- **Autres variables numériques** : imputation par la médiane → écrêtage aux quantiles 1 % / 99 % → standardisation.
- **Variables catégorielles** : imputation par la modalité la plus fréquente → encodage one-hot (modalités rares regroupées).

Toutes ces étapes sont **apprises sur l'entraînement** (médianes, quantiles, moyennes, écarts-types).
"""),
code(r"""
money_p = [c for c in MONEY_FEATURES if c in num_p]
other_p = [c for c in num_p if c not in money_p]
pre_profile = ColumnTransformer([
    ("montants", SkPipeline([("imp", SimpleImputer(strategy="median")), ("log", SignedLog()),
                             ("sc", StandardScaler())]), money_p),
    ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizer(0.01, 0.99)),
                        ("sc", StandardScaler())]), other_p),
    ("cat", SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20))]), cat_p),
])
"""),
md(r"""
### 3.5 Modèles comparés

| Modèle | Idée | Fonction de coût / principe |
|---|---|---|
| **Baseline** classe majoritaire | Niveau « sans apprentissage » | — |
| **Régression logistique** | $p(y=1\mid x)=\sigma(\beta_0+\sum_j\beta_j x_j)$ ; coefficients interprétables | Entropie croisée binaire (BCE) + pénalité $\lambda\lVert\beta\rVert_2^2$ (C = 1/λ) |
| **KNN** | Vote des k plus proches voisins | Distance euclidienne ; pas d'apprentissage |
| **SVM** (noyau RBF) | Frontière de marge maximale dans un espace transformé | Perte charnière + régularisation (C) |
| **Arbre de décision** | Questions successives sur les variables | Impureté de Gini |
| **Forêt aléatoire** | Moyenne d'arbres décorrélés (*bagging*) | Réduit la variance |
| **LightGBM, XGBoost, CatBoost** | Arbres ajoutés un à un pour corriger les erreurs (*boosting*) | Descente de gradient sur la BCE |
| **MLP** | Réseau de neurones à 2 couches cachées | BCE minimisée par **Adam** |

**Réglage des hyperparamètres** : recherche aléatoire (au plus `N_ITER` combinaisons) avec **validation croisée stratifiée par groupes** (ménages) sur l'entraînement, critère AUC ROC. Les modèles qui l'acceptent utilisent des **poids de classes équilibrés** ; KNN et MLP n'en acceptent pas et utilisent **SMOTE** (voir 3.6). La SVM, coûteuse (complexité au moins quadratique en n), est entraînée sur un sous-échantillon de 4 000 lignes.

**Règle de choix fixée à l'avance** : on retient le modèle de meilleure AUC de validation. Si la régression logistique est à moins de 0,005 de ce meilleur score, on la préfère (principe de parcimonie : c'est le modèle le plus simple et le plus explicable).
"""),
code(r"""
spw = float((yp_tr == 1).sum() / (yp_tr == 0).sum())   # poids de la classe 0 (Prudent) pour XGBoost


def profile_models():
    rs = RANDOM_STATE
    m = {
        "Baseline – classe majoritaire": ([("m", DummyClassifier(strategy="most_frequent"))], {}),
        "Régression logistique": ([("m", LogisticRegression(max_iter=3000, class_weight="balanced"))],
                                  {"m__C": [0.003, 0.01, 0.03, 0.1, 1]}),
        "KNN (+SMOTE)": ([("smote", SMOTE(random_state=rs)), ("m", KNeighborsClassifier())],
                         {"m__n_neighbors": [50, 100, 150, 250]}),
        "SVM (noyau RBF)": ([("m", SVC(probability=True, class_weight="balanced", random_state=rs))],
                            {"m__C": [0.1, 0.3, 1]}),
        "Arbre de décision": ([("m", DecisionTreeClassifier(class_weight="balanced", random_state=rs))],
                              {"m__max_depth": [3, 4, 6, 8], "m__min_samples_leaf": [20, 50, 100]}),
        "Forêt aléatoire": ([("m", RandomForestClassifier(200, class_weight="balanced_subsample", n_jobs=1, random_state=rs))],
                            {"m__max_depth": [8, 12, 16], "m__min_samples_leaf": [10, 20, 50]}),
        "LightGBM": ([("m", lgb.LGBMClassifier(n_estimators=300, class_weight="balanced", subsample=0.8, subsample_freq=1,
                                               colsample_bytree=0.8, verbose=-1, n_jobs=1, random_state=rs))],
                     {"m__learning_rate": [0.02, 0.05, 0.1], "m__num_leaves": [7, 15, 31], "m__min_child_samples": [20, 50]}),
        "XGBoost": ([("m", xgb.XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1 / spw,
                                             tree_method="hist", n_jobs=1, random_state=rs))],
                    {"m__learning_rate": [0.02, 0.05, 0.1], "m__max_depth": [2, 3, 5], "m__min_child_weight": [1, 10]}),
        "CatBoost": ([("m", cb.CatBoostClassifier(iterations=300, auto_class_weights="Balanced", verbose=0, thread_count=1,
                                                  allow_writing_files=False, random_seed=rs))],
                     {"m__learning_rate": [0.03, 0.1], "m__depth": [4, 6]}),
        "MLP (Adam)": ([("smote", SMOTE(random_state=rs)),
                        ("m", MLPClassifier((64, 32), solver="adam", early_stopping=True, max_iter=500, random_state=rs))],
                       {"m__alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }
    return m


def eval_profile(model, X, y, threshold=0.5, sample_weight=None) -> dict:
    # Classe d'intérêt = Prudent (codée 0) : on prédit « Prudent » si P(Prudent) ≥ seuil
    p_prud = model.predict_proba(X)[:, 0]
    pred_prud = (p_prud >= threshold).astype(int)
    y_prud = (np.asarray(y) == 0).astype(int)
    sw = sample_weight
    return {"ROC-AUC": roc_auc_score(y_prud, p_prud, sample_weight=sw),
            "F1 macro": f1_score(y_prud, pred_prud, average="macro", sample_weight=sw),
            "Balanced acc.": balanced_accuracy_score(y_prud, pred_prud, sample_weight=sw),
            "Précision Prudent": precision_score(y_prud, pred_prud, sample_weight=sw, zero_division=0),
            "Rappel Prudent": recall_score(y_prud, pred_prud, sample_weight=sw),
            "Accuracy": accuracy_score(y_prud, pred_prud, sample_weight=sw)}


cv_p = StratifiedGroupKFold(n_splits=3 if QUICK_RUN else 5, shuffle=True, random_state=RANDOM_STATE)
sub = Xp_tr.sample(4000, random_state=RANDOM_STATE).index      # sous-échantillon pour la SVM
profile_fitted, rows = {}, {}
for name, (steps, grid) in profile_models().items():
    t0 = time.time()
    Xf, yf, gf = (Xp_tr.loc[sub], yp_tr.loc[sub], gp_tr.loc[sub]) if name.startswith("SVM") else (Xp_tr, yp_tr, gp_tr)
    pipe = Pipeline([("pre", pre_profile)] + steps)
    if grid:
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(pipe, grid, n_iter=min(N_ITER, n_comb), cv=cv_p, scoring="roc_auc",
                                    n_jobs=N_JOBS, random_state=RANDOM_STATE).fit(Xf, yf, groups=gf)
        best, cv_auc, params = search.best_estimator_, search.best_score_, search.best_params_
    else:
        best, cv_auc, params = pipe.fit(Xf, yf), 0.5, {}
    profile_fitted[name] = best
    rows[name] = {"AUC CV (train)": cv_auc, **eval_profile(best, Xp_va, yp_va), "Durée (s)": time.time() - t0,
                  "Hyperparamètres": {k.replace("m__", ""): v for k, v in params.items()}}
    print(f"✓ {name:<32} AUC validation = {rows[name]['ROC-AUC']:.4f}  ({time.time() - t0:.0f} s)")

profile_comparison = pd.DataFrame(rows).T.sort_values("ROC-AUC", ascending=False)
profile_comparison.to_csv(RESULTS_DIR / "agent_profil_comparaison_validation.csv")
RESULTS["profil_validation"] = profile_comparison.drop(columns="Hyperparamètres")
profile_comparison
"""),
md(r"""
**Lecture** : tous les modèles appris battent largement la baseline (AUC 0,5). Les écarts entre les meilleurs modèles sont faibles (quelques millièmes d'AUC) : la relation entre situation financière et aversion au risque est **essentiellement monotone**, et un modèle linéaire la capte presque aussi bien que des ensembles d'arbres. Le MLP, plus flexible, est le moins bon : avec ~16 000 lignes bruitées, il souffre de **variance** (surapprentissage).
"""),
]
