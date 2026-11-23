from cellules import code, md

CELLS = [
md(r"""
### 4.5 Baselines et modèles

**Baselines** (aucun apprentissage, ou un seul paramètre) : toute méthode plus complexe doit faire mieux qu'elles.
- **Prévision nulle** : $\hat y = 0$ (« aucun actif ne fera mieux que la moyenne »).
- **Moyenne historique par actif** : rendement excédentaire moyen de chaque actif sur l'entraînement.
- **Momentum 12-1** (Jegadeesh & Titman, 1993) : « les gagnants des 12 derniers mois continuent de gagner ». Rang transversal du momentum, mis à l'échelle de la cible par une régression à une variable.

**Modèles** (tous dans un `Pipeline` : prétraitement → modèle) :

| Famille | Modèle | Fonction de coût (cours 1) | Hyperparamètres réglés |
|---|---|---|---|
| Linéaire | Régression linéaire (MCO) | $J(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert_2^2$ (MSE) | — |
| Linéaire régularisé | **Ridge** | MSE $+\lambda\lVert\beta\rVert_2^2$ | λ |
| | **Lasso** | MSE $+\lambda\lVert\beta\rVert_1$ (met des coefficients à 0) | λ |
| | **Elastic Net** | MSE $+\lambda[\alpha\lVert\beta\rVert_1+(1-\alpha)\lVert\beta\rVert_2^2]$ | λ, α |
| Voisinage | KNN | moyenne des k voisins | k |
| Noyau | SVR (RBF) | perte ε-insensible + régularisation | C, ε |
| Arbre | Arbre de décision | MSE dans chaque feuille | profondeur, taille des feuilles |
| Bagging | Forêt aléatoire | moyenne d'arbres décorrélés | profondeur, taille des feuilles |
| Boosting | Gradient Boosting, LightGBM, XGBoost, CatBoost | descente de gradient sur la MSE | taux d'apprentissage, profondeur, régularisation |
| Réseau | MLP | MSE minimisée par **Adam** | taille, pénalité L2 α |

**Réglage des hyperparamètres** : recherche aléatoire (au plus `N_ITER` combinaisons par modèle), en **validation croisée temporelle à fenêtre croissante purgée** sur l'entraînement, critère MSE. Jamais de validation croisée mélangée (KFold aléatoire) : elle mettrait des dates futures dans l'apprentissage. Les réglages trouvés sont **réutilisés tels quels** pour toutes les comparaisons qui suivent. La SVR, dont le coût croît au moins comme le carré du nombre d'observations, est entraînée sur un sous-échantillon de 3 000 lignes.
"""),
code(r"""
from sklearn.base import clone
from sklearn.model_selection import learning_curve
from src.model_zoo import regression_models, make_pipeline

# --- Baselines ---
asset_mean = tr.groupby(level="ticker")["target_reg"].mean()
m_tr = tr["mom_12_1_rank"] - 0.5
beta_mom = float((m_tr * tr["target_reg"]).sum() / (m_tr ** 2).sum())


def baseline_preds(data):
    return {"Baseline – prévision nulle": np.zeros(len(data)),
            "Baseline – moyenne par actif": data.index.get_level_values("ticker").map(asset_mean).fillna(0).values,
            "Baseline – momentum 12-1": beta_mom * (data["mom_12_1_rank"].fillna(0.5) - 0.5).values}


# --- Modèles appris : un seul réglage des hyperparamètres, en CV expanding purgée ---
cv_exp = TemporalCV(n_splits=3 if QUICK_RUN else 5, mode="expanding")
sub_svr = tr.sample(3000, random_state=RANDOM_STATE).sort_index()
fc_fitted, fc_info = {}, {}
for name, (famille, est, grid) in regression_models(quick=QUICK_RUN).items():
    t0 = time.time()
    data = sub_svr if name.startswith("SVR") else tr
    pipe = make_pipeline(est, NUM_F, CAT_FEATURES)
    if grid:
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(pipe, grid, n_iter=min(N_ITER, n_comb), cv=cv_exp, scoring="neg_mean_squared_error",
                                    n_jobs=N_JOBS, random_state=RANDOM_STATE)
        search.fit(data[FEATS], data["target_reg"], groups=data.index.get_level_values("date"))
        best, params = search.best_estimator_, {k.replace("m__", ""): v for k, v in search.best_params_.items()}
    else:
        best, params = pipe.fit(data[FEATS], data["target_reg"]), {}
    fc_fitted[name] = best
    fc_info[name] = {"famille": famille, "hyperparamètres": params, "durée (s)": round(time.time() - t0, 1)}
    print(f"✓ {name:<26} {time.time() - t0:5.0f} s  {params}")
"""),
code(r"""
val_preds = {**baseline_preds(va), **{n: m.predict(va[FEATS]) for n, m in fc_fitted.items()}}
fc_val = pd.DataFrame({n: forecast_report(p, va) for n, p in val_preds.items()}).T
fc_val.insert(0, "famille", [fc_info.get(n, {}).get("famille", "Baseline") for n in fc_val.index])
fc_val = fc_val.sort_values("MSE")
fc_val.to_csv(RESULTS_DIR / "agent_prevision_comparaison_validation.csv")
RESULTS["prevision_validation"] = fc_val
pd.DataFrame(fc_info).T.to_csv(RESULTS_DIR / "agent_prevision_hyperparametres.csv")
fc_val
"""),
md(r"""
**Comment lire ce tableau** : un **R² hors échantillon** positif signifie que le modèle fait mieux que la prévision nulle en MSE. L'**IC** mesure la qualité du classement des actifs, qui est ce qui compte pour l'allocation. Le **spread Q5 − Q1 net** est le rendement annualisé d'un portefeuille qui achète les 20 % d'actifs les mieux notés et vend les 20 % les moins bien notés, après 10 points de base de coûts par transaction. Tous ces chiffres portent sur **une seule** période de validation de 2 ans (96 dates) : ils sont bruités, et la section 4.7 vérifiera leur stabilité.

**Lecture** : sur la validation 2020-2021, le meilleur « modèle » est une **baseline**, la moyenne historique par actif (R² ≈ 8 %, IC ≈ 0,20). Les meilleurs modèles appris (Elastic Net, Ridge, XGBoost, forêt aléatoire) obtiennent un R² de 2 à 4 % et un IC de 0,13 à 0,16. Les modèles les plus flexibles (MLP, SVR, KNN, CatBoost) font moins bien que la prévision nulle : avec un signal aussi faible, la **variance** domine. La recherche d'hyperparamètres choisit d'ailleurs presque toujours la régularisation la plus forte.

### 4.6 Régularisation : chemin de λ et réduction de dimension (cours 1 et 2)
**Chemin de régularisation du Lasso** : quand λ augmente, la pénalité $\lambda\lVert\beta\rVert_1$ prend le pas sur l'erreur et les coefficients sont progressivement ramenés **exactement à zéro**. Le Lasso fait ainsi une **sélection de variables**. La droite verticale indique le λ choisi par validation croisée.
"""),
code(r"""
from sklearn.linear_model import lasso_path

pre_fit = make_pipeline(None, NUM_F, CAT_FEATURES).named_steps["pre"].fit(tr[FEATS])
Xt_tr = pre_fit.transform(tr[FEATS])
alphas, coefs, _ = lasso_path(Xt_tr, tr["target_reg"].values, alphas=np.logspace(-5, -2, 40))
names_t = [n.split("__", 1)[-1] for n in pre_fit.get_feature_names_out()]
top_idx = np.argsort(-np.abs(coefs).max(1))[:8]
plt.figure(figsize=(10, 5))
for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i], color="lightgrey", lw=.8)
for i in top_idx:
    plt.plot(alphas, coefs[i], lw=1.8, label=names_t[i])
lasso_alpha = fc_info["Lasso"]["hyperparamètres"].get("alpha")
if lasso_alpha is not None:
    plt.axvline(lasso_alpha, color="r", ls="--", label=f"λ choisi = {lasso_alpha:.1e}")
plt.xscale("log")
plt.xlabel("λ (échelle logarithmique)")
plt.ylabel("Coefficient")
plt.title("Chemin de régularisation du Lasso")
plt.legend(fontsize=8, ncol=2)
savefig("10c_lasso_chemin")
plt.show()
n_nonzero = int((np.abs(fc_fitted["Lasso"].named_steps["m"].coef_) > 1e-10).sum())
print(f"Lasso retenu : {n_nonzero} coefficients non nuls sur {len(names_t)}")
"""),
md(r"""
**Lecture** : la validation croisée retient un λ pour lequel **tous les coefficients du Lasso sont nuls** : le meilleur Lasso prédit 0 pour tous les actifs, c'est-à-dire la prévision nulle. Autrement dit, d'après la validation croisée, aucune variable n'apporte assez d'information pour justifier un coefficient non nul. C'est une illustration directe de la faiblesse du signal.

**Modèles avec ACP dans le pipeline** : on reprend les deux meilleurs modèles appris de la validation, avec leurs hyperparamètres, en ajoutant une ACP qui conserve **90 % de la variance** (critère du cours). L'ACP est ajustée sur l'entraînement uniquement.
"""),
code(r"""
learned_val = fc_val.drop(index=[i for i in fc_val.index if i.startswith("Baseline")])
top2 = learned_val.index[:2].tolist()
for name in top2:
    m = clone(fc_fitted[name].named_steps["m"])
    data = sub_svr if name.startswith("SVR") else tr
    pipe_acp = make_pipeline(m, NUM_F, CAT_FEATURES, n_pca=0.90).fit(data[FEATS], data["target_reg"])
    k = pipe_acp.named_steps["acp"].n_components_
    fc_fitted[f"{name} + ACP"] = pipe_acp
    fc_info[f"{name} + ACP"] = {**fc_info[name], "famille": fc_info[name]["famille"] + " + ACP"}
    val_preds[f"{name} + ACP"] = pipe_acp.predict(va[FEATS])
    print(f"{name} + ACP : {k} composantes conservées")
acp_tab = pd.DataFrame({n: forecast_report(val_preds[n], va) for n in top2 + [f"{n} + ACP" for n in top2]}).T
acp_tab[["MSE", "R² hors échantillon", "IC moyen", "IC t-stat (Newey-West)", "Spread Q5-Q1 net (annualisé)"]]
"""),
md(r"""
### 4.7 Diagnostic biais / variance : courbe d'apprentissage (cours 1)
Pour le meilleur modèle appris, on trace l'erreur (MSE) sur l'entraînement ($J_{train}$) et en validation croisée ($J_{cv}$) selon la taille de l'entraînement :
- $J_{train}$ et $J_{cv}$ élevés et proches → **biais élevé** (sous-apprentissage) ;
- $J_{train}$ faible et $J_{cv}$ élevé → **variance élevée** (surapprentissage).
"""),
code(r"""
best_learned = learned_val.index[0]
data_lc = sub_svr if best_learned.startswith("SVR") else tr
sizes, sc_tr, sc_cv = learning_curve(clone(fc_fitted[best_learned]), data_lc[FEATS], data_lc["target_reg"],
                                     groups=data_lc.index.get_level_values("date"), cv=cv_exp,
                                     train_sizes=np.linspace(0.3, 1.0, 3 if QUICK_RUN else 5),
                                     scoring="neg_mean_squared_error", n_jobs=N_JOBS)
J_tr, J_cv = -sc_tr.mean(1), -sc_cv.mean(1)
J_null = float((tr["target_reg"] ** 2).mean())
plt.figure(figsize=(7, 4))
plt.plot(sizes, J_tr, "o-", label="$J_{train}$")
plt.plot(sizes, J_cv, "o-", label="$J_{cv}$")
plt.axhline(J_null, color="grey", ls=":", label="MSE de la prévision nulle")
plt.xlabel("Nombre d'observations d'entraînement")
plt.ylabel("MSE")
plt.legend()
plt.title(f"Courbe d'apprentissage – {best_learned}")
savefig("10d_prevision_courbe_apprentissage")
plt.show()
print(f"J_train = {J_tr[-1]:.5f} | J_cv = {J_cv[-1]:.5f} | prévision nulle ≈ {J_null:.5f}")
chrono("4b. Prévision : modèles, régularisation, ACP")
"""),
]
