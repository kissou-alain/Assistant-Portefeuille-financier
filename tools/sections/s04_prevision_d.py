from cellules import code, md

CELLS = [
md(r"""
### 4.11 Évaluation unique sur le test (2022 → fin)
Le modèle retenu et les baselines sont ré-entraînés sur **entraînement + validation** (étape 4 de la méthode du cours), puis évalués **une seule fois** sur le test. Aucun choix n'est modifié après cette étape.
"""),
code(r"""
best_fc = build_forecast_model(best_fc_name).fit(trva[FEATS], trva["target_reg"])
test_models = {best_fc_name: best_fc, **{n: clone(b).fit(trva[FEATS], trva["target_reg"]) for n, b in BASELINES.items()}}
test_preds = {n: m.predict(te[FEATS]) for n, m in test_models.items()}
fc_test = pd.DataFrame({n: forecast_report(p, te) for n, p in test_preds.items()}).T
losses_te = {n: loss_by_date(te["target_reg"], p) for n, p in test_preds.items()}
for ref in ("Baseline – prévision nulle", "Baseline – moyenne par actif"):
    d = dm_table(losses_te, ref)
    fc_test[f"DM vs {ref.split('– ')[1]}"] = d["DM"]
    fc_test[f"p vs {ref.split('– ')[1]}"] = d["p-valeur"]
fc_test.to_csv(RESULTS_DIR / "agent_prevision_test.csv")
RESULTS["prevision_test"] = fc_test
fc_test
"""),
code(r"""
s_te = pd.Series(test_preds[best_fc_name], index=te.index)
ic_te = information_coefficient(s_te, te["fwd_ret"])
ls_te = quintile_spread(s_te, te["fwd_ret"])
ic_base = information_coefficient(pd.Series(test_preds["Baseline – moyenne par actif"], index=te.index), te["fwd_ret"])
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
ic_te.rolling(13).mean().plot(ax=axes[0], color="k", label=f"{best_fc_name} (moyenne mobile 3 mois)")
ic_base.rolling(13).mean().plot(ax=axes[0], color="darkorange", ls="--", label="Baseline moyenne par actif")
axes[0].axhline(0, color="r", lw=.8)
axes[0].legend()
axes[0].set_title("Coefficient d'information sur le test")
(1 + ls_te["net"].fillna(0)).cumprod().plot(ax=axes[1], color="seagreen", label="net de coûts")
(1 + ls_te["brut"].fillna(0)).cumprod().plot(ax=axes[1], color="grey", ls=":", label="brut")
axes[1].axhline(1, color="k", lw=.8)
axes[1].legend()
axes[1].set_title(f"Portefeuille long-short Q5 − Q1 ({best_fc_name}, test)")
plt.tight_layout()
savefig("11_prevision_test")
plt.show()
"""),
md(r"""
### 4.12 Interprétabilité de l'agent Prévision (cours 1)
1. **Interprétabilité intrinsèque** : coefficients du modèle Ridge (variables standardisées, donc comparables entre elles).
2. **Explicabilité globale** : importance par permutation sur le test (hausse de la MSE quand on mélange une variable) et valeurs SHAP moyennes.
3. **Explicabilité locale** : contribution de chaque variable à la prévision d'un actif à la dernière date disponible.
4. **Analyse des erreurs** : par classe d'actifs et par année.
"""),
code(r"""
from sklearn.inspection import permutation_importance
from src.explain import make_explainer, shap_values, global_importance, local_explanation

ridge_pipe = fc_fitted["Ridge"]
names_r = [n.split("__", 1)[-1] for n in ridge_pipe.named_steps["pre"].get_feature_names_out()]
coef_r = pd.Series(ridge_pipe.named_steps["m"].coef_, index=names_r)
coef_r = coef_r.reindex(coef_r.abs().sort_values(ascending=False).index).head(15).sort_values()
coef_r.plot.barh(figsize=(8, 6), color=np.where(coef_r > 0, "seagreen", "indianred"))
plt.title("Ridge : coefficients (effet sur le rendement excédentaire prévu)")
savefig("12a_prevision_coefficients_ridge")
plt.show()

pi_fc = permutation_importance(best_fc, te[FEATS], te["target_reg"], scoring="neg_mean_squared_error",
                               n_repeats=3 if QUICK_RUN else 5, random_state=RANDOM_STATE, n_jobs=N_JOBS)
pi_fc = pd.DataFrame({"importance": pi_fc.importances_mean, "écart-type": pi_fc.importances_std},
                     index=FEATS).sort_values("importance", ascending=False)
pi_fc.to_csv(RESULTS_DIR / "agent_prevision_importance_permutation.csv")
TOP_FEATURES = pi_fc.index[:10].tolist()
pi_fc.head(15).iloc[::-1].plot.barh(y="importance", xerr="écart-type", legend=False, figsize=(8, 6), color="slateblue")
plt.title(f"Importance par permutation – {best_fc_name} (hausse de la MSE, test)")
savefig("12_prevision_permutation")
plt.show()
"""),
code(r"""
# SHAP : sur le modèle retenu s'il s'agit d'un pipeline unique, sinon sur le premier modèle de l'ensemble
shap_name = best_fc_name if not best_fc_name.startswith("Ensemble") else top3_oof[0]
shap_pipe = best_fc if not best_fc_name.startswith("Ensemble") else clone(fc_fitted[shap_name]).fit(trva[FEATS], trva["target_reg"])
explainer_fc = make_explainer(shap_pipe, trva[FEATS])
X_sh = te[FEATS].sample(400 if QUICK_RUN else 2000, random_state=RANDOM_STATE)
sv_fc = shap_values(shap_pipe, explainer_fc, X_sh)
imp_fc = global_importance(sv_fc).head(15)
imp_fc.iloc[::-1].plot.barh(figsize=(8, 6), color="teal")
plt.title(f"SHAP global – {shap_name} (moyenne des |contributions|, test)")
savefig("13_prevision_shap")
plt.show()
print("Variables les plus influentes (SHAP global) :", imp_fc.head(6).round(5).to_dict())

last_d = te.index.get_level_values("date").max()
te_last = te.xs(last_d, level="date", drop_level=False)
top_asset = pd.Series(shap_pipe.predict(te_last[FEATS]), index=te_last.index).idxmax()
row = te_last.loc[[top_asset]][FEATS]
s_loc = shap_values(shap_pipe, explainer_fc, row).iloc[0]
loc = local_explanation(s_loc, row.iloc[0], top=8).iloc[::-1]
plt.figure(figsize=(8, 5))
plt.barh(loc["libellé"], loc["contribution"], color=np.where(loc["contribution"] > 0, "seagreen", "indianred"))
plt.axvline(0, color="k", lw=.8)
plt.title(f"Explication locale : {top_asset[1]} au {last_d.date()} "
          f"(prévision {shap_pipe.predict(row)[0]:+.2%} à 1 mois)")
plt.xlabel("Contribution SHAP au rendement excédentaire prévu")
savefig("14_prevision_shap_local")
plt.show()
"""),
code(r"""
err_fc = te[["fwd_ret", "target_reg", "asset_class"]].copy()
err_fc["prévision"] = test_preds[best_fc_name]
err_fc["année"] = err_fc.index.get_level_values("date").year
by_class = err_fc.groupby("asset_class").apply(lambda g: pd.Series({
    "n": len(g), "MSE modèle": ((g["target_reg"] - g["prévision"]) ** 2).mean(),
    "MSE prévision nulle": (g["target_reg"] ** 2).mean(),
    "rendement excédentaire moyen réalisé": g["target_reg"].mean(), "prévision moyenne": g["prévision"].mean()}))
by_year = err_fc.groupby("année").apply(lambda g: pd.Series({
    "n": len(g), "IC": information_coefficient(g["prévision"], g["fwd_ret"]).mean(),
    "R² hors échantillon": 1 - ((g["target_reg"] - g["prévision"]) ** 2).mean() / (g["target_reg"] ** 2).mean()}))
display(by_class)
display(by_year)
by_class.to_csv(RESULTS_DIR / "agent_prevision_erreurs_par_classe.csv")
by_year.to_csv(RESULTS_DIR / "agent_prevision_erreurs_par_annee.csv")

forecast_bundle = {"pipeline": best_fc, "features": FEATS, "ic": IC_USED, "active": FC_ACTIVE, "task": "régression",
                   "top_features": TOP_FEATURES, "model_name": best_fc_name, "params": fc_info.get(best_fc_name, {}),
                   "ic_hors_pli": IC_VAL}
chrono("4d. Prévision : test et interprétabilité")
"""),
]
