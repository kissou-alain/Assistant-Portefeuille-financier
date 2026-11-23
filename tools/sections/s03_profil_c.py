from cellules import code, md

CELLS = [
md(r"""
### 3.12 Interprétabilité de l'agent Profil (cours 1)
1. **Interprétabilité intrinsèque** : les coefficients de la régression logistique. Un coefficient $\beta_j > 0$ augmente le log-odds de la classe modélisée.
2. **Explicabilité globale** : importance par permutation (baisse de l'AUC quand on mélange une variable) et valeurs SHAP moyennes.
3. **Explicabilité locale** : décomposition SHAP d'une prédiction individuelle, $\text{prédiction}_i = \text{valeur de base} + \sum_j \text{contribution}_j$. C'est elle qui est affichée au client dans l'application (« pourquoi ce profil ? »).
4. **Analyse des erreurs** par tranche d'âge.
"""),
code(r"""
from src.explain import make_explainer, shap_values, global_importance, local_explanation, original_name

logit_pipe = profile_fitted["Régression logistique"]
names = logit_pipe.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(-logit_pipe.named_steps["m"].coef_[0], index=names)   # signe inversé : effet sur P(Prudent)
coefs.index = [f"{SCF_LABELS.get(original_name(n, list(X_p.columns)), n)} [{n.split('__', 1)[-1]}]" for n in coefs.index]
top_coef = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)
top_coef.sort_values().plot.barh(figsize=(8, 6), color=np.where(top_coef.sort_values() > 0, "indianred", "seagreen"))
plt.title("Régression logistique : effet sur le log-odds de « Prudent »\n(variables standardisées ; rouge = rend plus prudent)")
savefig("08a_profil_coefficients")
plt.show()
"""),
code(r"""
pi = permutation_importance(best_profile, Xp_te, yp_te, scoring="roc_auc", n_repeats=3 if QUICK_RUN else 10,
                            random_state=RANDOM_STATE, n_jobs=N_JOBS)
pi_profile = pd.DataFrame({"importance": pi.importances_mean, "écart-type": pi.importances_std},
                          index=[SCF_LABELS.get(c, c) for c in Xp_te.columns]).sort_values("importance")
pi_profile.tail(15).plot.barh(y="importance", xerr="écart-type", legend=False, figsize=(8, 6), color="teal")
plt.title("Importance par permutation – baisse de l'AUC sur le test")
savefig("08_profil_permutation")
plt.show()
pi_profile.sort_values("importance", ascending=False).to_csv(RESULTS_DIR / "agent_profil_importance_permutation.csv")
"""),
code(r"""
explainer_p = make_explainer(best_profile, Xp_tr)
X_shap = Xp_te.sample(300 if QUICK_RUN else 1500, random_state=RANDOM_STATE)
sv_p = shap_values(best_profile, explainer_p, X_shap, class_idx=0)    # contributions au log-odds de « Prudent »
imp = global_importance(sv_p).head(15).rename(index=lambda c: SCF_LABELS.get(c, c))
imp.sort_values().plot.barh(figsize=(8, 6), color="slateblue")
plt.title("SHAP global : moyenne des |contributions| (variables d'origine)")
savefig("09_profil_shap_global")
plt.show()
print("Variables les plus influentes (SHAP global) :")
print(imp.head(8).round(3).to_string())
"""),
md(r"""
**Explication locale de deux ménages du test** : un ménage classé Prudent et un ménage classé Non prudent. Barres rouges : la variable pousse vers « Prudent » ; barres vertes : elle en éloigne.
"""),
code(r"""
p_all = best_profile.predict_proba(Xp_te)[:, 0]
exemples = {"Ménage classé Prudent": Xp_te.index[np.argmax(p_all)],
            "Ménage classé Non prudent": Xp_te.index[np.argmin(p_all)]}
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, (titre, idx) in zip(axes, exemples.items()):
    row = Xp_te.loc[[idx]]
    s = shap_values(best_profile, explainer_p, row, class_idx=0).iloc[0]
    loc = local_explanation(s, row.iloc[0], labels=SCF_LABELS, top=8)
    loc = loc.iloc[::-1]
    ax.barh(loc["libellé"], loc["contribution"], color=np.where(loc["contribution"] > 0, "indianred", "seagreen"))
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"{titre} – P(Prudent) = {best_profile.predict_proba(row)[0, 0]:.0%}")
    ax.set_xlabel("Contribution SHAP (log-odds de « Prudent »)")
plt.tight_layout()
savefig("09b_profil_shap_local")
plt.show()
"""),
code(r"""
err = pd.DataFrame({"age": Xp_te["AGE"], "vrai": yp_te, "prédit": pred_te_p})
err["tranche d'âge"] = pd.cut(err["age"], [0, 35, 50, 65, 120], labels=["<35", "35-50", "50-65", "65+"])
err_tab = err.groupby("tranche d'âge", observed=True).apply(lambda g: pd.Series({
    "n": len(g), "% Prudent réel": (g["vrai"] == 0).mean(),
    "rappel Prudent": recall_score(g["vrai"] == 0, g["prédit"] == 0),
    "F1 macro": f1_score(g["vrai"], g["prédit"], average="macro")}))
display(err_tab)

# Bundle de l'agent Profil, utilisé par l'orchestrateur (section 6) et l'application
profile_bundle = {"pipeline": best_profile, "num": num_p, "cat": cat_p, "classes": PROFILE_ML_CLASSES,
                  "mode": PROFILE_TARGET_MODE, "threshold": THRESH,
                  "defaults": {**Xp_tr[num_p].median().to_dict(), **{c: Xp_tr[c].mode().iloc[0] for c in cat_p}},
                  "model_name": best_profile_name, "background": Xp_tr.sample(200, random_state=RANDOM_STATE)}
chrono("3. Agent Profil (interprétabilité)")
"""),
md(r"""
**Lecture** : les variables les plus influentes sont le **patrimoine** (actifs, patrimoine net), le **revenu**, le **niveau d'études** et les **habitudes de décision financière** (comparer les placements, s'informer). Un patrimoine et un revenu faibles rendent plus prudent. C'est cohérent avec la notion de **capacité à supporter des pertes** : un ménage peu aisé n'a pas de marge pour absorber une baisse de ses placements. **Analyse des erreurs** : le rappel des Prudents est nettement plus faible chez les **moins de 35 ans** : les jeunes prudents ont souvent une situation financière modeste mais « normale » pour leur âge, et le modèle les distingue mal des jeunes non prudents. C'est une raison supplémentaire de croiser le modèle avec le questionnaire déclaratif. La tolérance au risque reste en partie psychologique et n'est qu'en partie expliquée par ces variables : dans l'application, la prédiction est donc **proposée** au conseiller, qui peut la corriger (principe de l'humain dans la boucle).
"""),
]
