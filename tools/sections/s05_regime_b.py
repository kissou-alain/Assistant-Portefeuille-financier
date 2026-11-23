from cellules import code, md

CELLS = [
md(r"""
### 5.5 Le GMM comme extraction de variables (cours 2, remarque 2)
Les probabilités a posteriori $\gamma_{ik}$ (« responsabilités ») mesurent le degré d'appartenance de la journée *i* à chaque régime *k*. On les ajoute comme **3 nouvelles variables** de l'agent Prévision : un modèle peut ainsi apprendre que certains actifs réagissent différemment selon le régime de marché.

**Protocole** : pour le modèle Ridge et pour le meilleur modèle à base d'arbres, mêmes hyperparamètres qu'en 4.5, on compare la version **sans** et **avec** les $\gamma_{ik}$ par le test de Diebold-Mariano.
- **Test principal : la validation 2020-2021** (modèles ajustés sur 2011-2019 ; le GMM aussi : aucun recouvrement).
- Test complémentaire : prévisions hors pli 2013-2021. Limite : le GMM a été ajusté sur 2011-2019 et a donc « vu » les états de marché de ces années (sans jamais voir de rendement futur). Ce test est légèrement optimiste pour la version avec γ.
"""),
code(r"""
gamma_cols = ["γ Calme", "γ Neutre", "γ Stress"]
gam = proba_all.rename(columns=dict(zip(["Calme", "Neutre", "Stress"], gamma_cols)))


def add_gamma(df):
    g = gam.reindex(df.index.get_level_values("date")).fillna(1 / 3)
    g.index = df.index
    return pd.concat([df, g], axis=1)


tr_g, va_g, trva_g = add_gamma(tr), add_gamma(va), add_gamma(trva)
FEATS_G, NUM_G = FEATS + gamma_cols, NUM_F + gamma_cols
tree_names = [n for n in cands.index if any(k in n for k in ("Forêt", "Boosting", "LightGBM", "XGBoost", "CatBoost", "Arbre"))]
gamma_models = ["Ridge", tree_names[0]]
rows_g, losses_g = {}, {}
for n in gamma_models:
    m = fc_fitted[n].named_steps["m"]
    p_sans = fc_fitted[n].predict(va[FEATS])
    pipe_g = make_pipeline(clone(m), NUM_G, CAT_FEATURES).fit(tr_g[FEATS_G], tr_g["target_reg"])
    p_avec = pipe_g.predict(va_g[FEATS_G])
    oof_avec = oof_predictions(make_pipeline(clone(m), NUM_G, CAT_FEATURES), trva_g, FEATS_G, cv_oof)
    l_val = {"sans": loss_by_date(va["target_reg"], p_sans), "avec": loss_by_date(va["target_reg"], p_avec)}
    l_oof = {"sans": loss_by_date(y_oof, oof[n].reindex(y_oof.index)),
             "avec": loss_by_date(y_oof, oof_avec.reindex(y_oof.index))}
    dv, do = diebold_mariano(l_val["sans"], l_val["avec"]), diebold_mariano(l_oof["sans"], l_oof["avec"])
    rows_g[n] = {"MSE val. sans γ": l_val["sans"].mean(), "MSE val. avec γ": l_val["avec"].mean(),
                 "DM validation": dv["DM"], "p validation": dv["p-valeur"],
                 "IC val. sans γ": forecast_report(p_sans, va)["IC moyen"],
                 "IC val. avec γ": forecast_report(p_avec, va_g)["IC moyen"],
                 "DM hors pli": do["DM"], "p hors pli": do["p-valeur"]}
gamma_tab = pd.DataFrame(rows_g).T
gamma_tab.to_csv(RESULTS_DIR / "agent_prevision_apport_gamma.csv")
RESULTS["prevision_apport_gamma"] = gamma_tab
print("DM > 0 et p < 0,05 : la version AVEC les probabilités de régime est meilleure.")
gamma_tab
"""),
md(r"""
### 5.6 Agent Alerte : détection d'anomalies (cours 1 et 2)
Une **anomalie** est une observation rare qui se distingue nettement de la majorité des données. Ici, une **journée de marché anormale** (krach, choc de liquidité). Deux usages dans le système : (i) **alerter** le conseiller et le client, (ii) **contrôler la qualité** des données téléchargées.

| Type | Méthode | Principe |
|---|---|---|
| Statistique | **z-score** | $\lvert r_t-\bar r\rvert/\sigma > 4$ sur le rendement du S&P 500 |
| Statistique | **Test de Grubbs** | teste si la valeur la plus extrême est aberrante sous hypothèse gaussienne ; répété tant qu'il rejette |
| ML | **Gaussienne multivariée** | densité $p(x;\mu,\Sigma)$ estimée sur l'entraînement ; anomalie si $p(x) \le \epsilon$ |
| ML | **Isolation Forest** | une anomalie s'isole en peu de coupures aléatoires |
| ML | **LOF** (Local Outlier Factor) | densité locale nettement plus faible que celle de ses voisins |
| ML | **One-Class SVM** | frontière qui entoure les données « normales » |

Il n'existe pas d'étiquette « anomalie » sur les marchés : la validation est **qualitative** (les crises connues sont-elles détectées ?) et repose sur l'**accord entre méthodes**. Seuils et paramètres sont appris sur 2011-2019 (taux d'anomalies attendu : 2 %).
"""),
code(r"""
from src.anomalies import zscore_flags, grubbs_iterative, fit_detectors, detect, event_recall

r_spy = returns["SPY"]
z_flags = zscore_flags(r_spy, TRAIN_END, threshold=4.0)
g_flags = grubbs_iterative(r_spy.loc[:TRAIN_END], alpha=0.01)
print(f"z-score > 4 : {int(z_flags.sum())} jours (dont {int(z_flags.loc[:TRAIN_END].sum())} en entraînement) | "
      f"Grubbs (entraînement) : {int(g_flags.sum())} jours aberrants")

detectors = fit_detectors(M_tr, contamination=0.02)
flags = detect(detectors, M_all)
flags["z-score S&P 500"] = z_flags.reindex(flags.index).fillna(False)
rates = flags.drop(columns="nb détecteurs").groupby(np.where(flags.index <= TRAIN_END, "2011-2019 (entraînement)",
                                                             "2020 → (hors échantillon)")).mean()
display(rates.style.format("{:.1%}"))
recall_ev = event_recall(flags)
display(recall_ev.replace({True: "✔", False: "✘"}))
display(flags.drop(columns="nb détecteurs").astype(float).corr().round(2))
"""),
code(r"""
ALERT_MIN = 3     # alerte si au moins 3 des 4 détecteurs ML signalent la journée (vote majoritaire)
alert = flags["nb détecteurs"] >= ALERT_MIN
fig, ax = plt.subplots(figsize=(15, 4.5))
spy_a = prices["SPY"].reindex(flags.index)
ax.plot(spy_a.index, spy_a.values, color="k", lw=.8)
ax.scatter(spy_a.index[alert], spy_a[alert], color="red", s=12, label=f"alerte (≥ {ALERT_MIN} détecteurs sur 4)", zorder=3)
ax.axvline(pd.Timestamp(TRAIN_END), color="blue", ls="--")
ax.legend()
ax.set_title("Agent Alerte : journées anormales détectées sur le S&P 500")
savefig("16b_anomalies")
plt.show()
print(f"Jours en alerte : {alert.loc[:TRAIN_END].mean():.1%} en entraînement, {alert.loc[TRAIN_END:].mean():.1%} ensuite")
alert_bundle = {"detectors": detectors, "features": reg_cols, "min_votes": ALERT_MIN}
chrono("5b. γ du GMM et Agent Alerte")
"""),
md(r"""
**Lecture** : toutes les crises connues sont signalées par plusieurs méthodes, et les détecteurs multivariés s'accordent largement entre eux. Le z-score, qui ne regarde que le rendement du jour, est peu corrélé aux autres : il repère les chocs brutaux d'une journée, alors que les méthodes multivariées repèrent des **états** de marché anormaux qui durent (volatilité, VIX et spread de crédit élevés ensemble).

**Dérive des données** (*data drift*, défi du ML vu au cours 1) : les détecteurs signalent environ 2 % des jours sur l'entraînement, comme prévu, mais une part bien plus élevée après 2019. Le marché d'après 2020 est plus agité que celui des années 2010. Pour limiter les fausses alertes, l'agent Alerte n'alerte que si **au moins 3 des 4 détecteurs** sont d'accord ; il faudra **recalibrer** périodiquement les seuils sur des données récentes.
"""),
]
