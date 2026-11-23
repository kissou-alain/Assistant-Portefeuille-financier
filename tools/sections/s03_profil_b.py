from cellules import code, md

CELLS = [
md(r"""
### 3.6 Classes déséquilibrées : poids de classes, SMOTE et ADASYN (cours 1)
Avec 1/3 de Prudents, un modèle non corrigé favorise la classe majoritaire et **rate des Prudents**, l'erreur la plus grave pour un conseiller.
- **Poids de classes** : chaque erreur sur la classe minoritaire compte davantage dans la fonction de coût.
- **SMOTE** : crée des exemples synthétiques de la classe minoritaire par interpolation entre un point et l'un de ses k voisins : $x_{new} = x_i + \lambda\,(x_{voisin} - x_i)$, avec $\lambda \sim U(0,1)$.
- **ADASYN** : variante de SMOTE qui génère davantage d'exemples là où la classe minoritaire est difficile à apprendre.

Le rééchantillonnage est placé dans un `imblearn.Pipeline` : il n'est appliqué **qu'aux plis d'entraînement**, jamais à la validation ni au test (sinon fuite de données). On compare les 4 stratégies sur la logistique et LightGBM, avec les hyperparamètres trouvés en 3.5.
"""),
code(r"""
def strip_weights(model):
    m = clone(model)
    if hasattr(m, "class_weight"):
        m.set_params(class_weight=None)
    return m


imb_rows = {}
for mname in ("Régression logistique", "LightGBM"):
    base_m = profile_fitted[mname].named_steps["m"]
    for strat, steps in (("aucune correction", [("m", strip_weights(base_m))]),
                         ("poids de classes", [("m", clone(base_m))]),
                         ("SMOTE", [("s", SMOTE(random_state=RANDOM_STATE)), ("m", strip_weights(base_m))]),
                         ("ADASYN", [("s", ADASYN(random_state=RANDOM_STATE)), ("m", strip_weights(base_m))])):
        p = Pipeline([("pre", pre_profile)] + steps).fit(Xp_tr, yp_tr)
        imb_rows[(mname, strat)] = eval_profile(p, Xp_va, yp_va)
imb_tab = pd.DataFrame(imb_rows).T
imb_tab.to_csv(RESULTS_DIR / "agent_profil_desequilibre.csv")
RESULTS["profil_desequilibre"] = imb_tab
imb_tab[["ROC-AUC", "Rappel Prudent", "Précision Prudent", "F1 macro", "Balanced acc."]]
"""),
md(r"""
**Lecture** : les 4 stratégies donnent presque la **même AUC**, car l'AUC mesure la qualité du **classement** des clients, indépendamment du seuil. Ce qui change, c'est le compromis rappel / précision au seuil de 0,5 : sans correction, environ 40 % des Prudents sont manqués ; avec les poids de classes, SMOTE ou ADASYN, le rappel des Prudents dépasse 75-80 %. Ces trois corrections ont un effet comparable à un simple **déplacement du seuil de décision** (3.8). On garde les poids de classes, la solution la plus simple, qui ne crée aucune donnée artificielle.

### 3.7 Choix du modèle (règle fixée en 3.5)
"""),
code(r"""
learned = profile_comparison.drop(index=[i for i in profile_comparison.index if i.startswith("Baseline")])
best_auc = learned["ROC-AUC"].max()
logit_auc = learned.loc["Régression logistique", "ROC-AUC"]
best_profile_name = "Régression logistique" if logit_auc >= best_auc - 0.005 else learned["ROC-AUC"].idxmax()
print(f"Meilleure AUC de validation : {best_auc:.4f} ({learned['ROC-AUC'].idxmax()}) | "
      f"logistique : {logit_auc:.4f}")
print("Modèle retenu pour l'agent Profil :", best_profile_name)
"""),
md(r"""
### 3.8 Seuil de décision (choisi sur la validation)
Règle métier : ne pas repérer un client prudent est l'erreur la plus coûteuse. On exige un **rappel d'au moins 80 %** sur la classe Prudent, puis on maximise l'accuracy équilibrée.
"""),
code(r"""
p_val = profile_fitted[best_profile_name].predict_proba(Xp_va)[:, 0]
yv_prud = (yp_va.values == 0)
grid_t = np.round(np.arange(0.10, 0.81, 0.01), 2)
bal_t = np.array([balanced_accuracy_score(yv_prud, p_val >= t) for t in grid_t])
rec_t = np.array([recall_score(yv_prud, p_val >= t) for t in grid_t])
ok = rec_t >= PRUDENT_RECALL_MIN
THRESH = float(grid_t[ok][np.argmax(bal_t[ok])]) if ok.any() else float(grid_t[np.argmax(rec_t)])
plt.figure(figsize=(7, 3.8))
plt.plot(grid_t, bal_t, label="Accuracy équilibrée")
plt.plot(grid_t, rec_t, label="Rappel « Prudent »")
plt.axhline(PRUDENT_RECALL_MIN, color="grey", ls=":", label=f"Rappel minimal ({PRUDENT_RECALL_MIN:.0%})")
plt.axvline(THRESH, color="r", ls="--")
plt.xlabel("Seuil sur P(Prudent)")
plt.legend()
plt.title(f"Choix du seuil de décision sur la validation : {THRESH:.2f}")
savefig("07b_profil_seuil")
plt.show()

# Compromis : que coûterait un rappel plus élevé ?
rows_t = []
for target in (0.80, 0.85, 0.90, 0.95):
    t = grid_t[rec_t >= target].max()
    pred = p_val >= t
    rows_t.append({"rappel visé": target, "seuil": t, "rappel Prudent": recall_score(yv_prud, pred),
                   "précision Prudent": precision_score(yv_prud, pred), "% clients classés Prudent": pred.mean(),
                   "% non-prudents classés Prudent": pred[~yv_prud].mean()})
display(pd.DataFrame(rows_t).set_index("rappel visé"))
"""),
md(r"""
**Lecture du compromis** : exiger 90 % ou 95 % de rappel ferait classer « Prudent » plus de la moitié des clients (alors qu'ils ne sont qu'environ un tiers) : le modèle perdrait son pouvoir de discrimination et proposerait des portefeuilles trop défensifs à de nombreux clients. Le seuil de 80 % est conservé. Le modèle n'est d'ailleurs qu'une des deux sécurités : le questionnaire déclaratif et la validation par le conseiller complètent le dispositif.
"""),
md(r"""
### 3.9 Diagnostic biais / variance : courbe d'apprentissage (cours 1)
On trace l'AUC sur l'entraînement ($J_{train}$) et en validation croisée ($J_{cv}$) quand la taille de l'entraînement augmente.
- Les deux courbes basses et proches → **biais élevé** (sous-apprentissage) : il faudrait un modèle plus riche ou de meilleures variables.
- Un grand écart entre les deux → **variance élevée** (surapprentissage) : il faudrait plus de données ou plus de régularisation.
"""),
code(r"""
sizes, sc_tr, sc_cv = learning_curve(clone(profile_fitted[best_profile_name]), Xp_tr, yp_tr, groups=gp_tr, cv=cv_p,
                                     train_sizes=np.linspace(0.1, 1.0, 4 if QUICK_RUN else 6),
                                     scoring="roc_auc", n_jobs=N_JOBS)
plt.figure(figsize=(7, 4))
plt.plot(sizes, sc_tr.mean(1), "o-", label="Entraînement")
plt.plot(sizes, sc_cv.mean(1), "o-", label="Validation croisée")
plt.fill_between(sizes, sc_cv.mean(1) - sc_cv.std(1), sc_cv.mean(1) + sc_cv.std(1), alpha=.2)
plt.xlabel("Nombre de lignes d'entraînement")
plt.ylabel("AUC")
plt.legend()
plt.title(f"Courbe d'apprentissage – {best_profile_name}")
savefig("07c_profil_courbe_apprentissage")
plt.show()
print(f"Écart final entraînement – validation : {sc_tr.mean(1)[-1] - sc_cv.mean(1)[-1]:.3f}")
"""),
md(r"""
### 3.10 Évaluation unique sur le test
Étape 4 de la méthode du cours : on ré-entraîne le modèle retenu sur **entraînement + validation**, avec les hyperparamètres choisis, puis on l'évalue **une seule fois** sur le test. La ligne « pondéré WGT » applique les poids d'enquête du SCF, qui rendent l'échantillon représentatif de la population américaine.
"""),
code(r"""
best_profile = clone(profile_fitted[best_profile_name]).fit(pd.concat([Xp_tr, Xp_va]), pd.concat([yp_tr, yp_va]))
profile_test = pd.DataFrame({
    f"{best_profile_name} (seuil 0,5)": eval_profile(best_profile, Xp_te, yp_te),
    f"{best_profile_name} (seuil {THRESH:.2f})": eval_profile(best_profile, Xp_te, yp_te, THRESH),
    f"{best_profile_name} (seuil {THRESH:.2f}, pondéré WGT)": eval_profile(best_profile, Xp_te, yp_te, THRESH,
                                                                           w_p[m_te].values),
    "Baseline – classe majoritaire": eval_profile(profile_fitted["Baseline – classe majoritaire"], Xp_te, yp_te),
}).T
profile_test.to_csv(RESULTS_DIR / "agent_profil_test.csv")
RESULTS["profil_test"] = profile_test
display(profile_test)

p_te_prud = best_profile.predict_proba(Xp_te)[:, 0]
pred_te_p = np.where(p_te_prud >= THRESH, 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(yp_te, pred_te_p, display_labels=PROFILE_ML_CLASSES, normalize="true",
                                        cmap="Blues", ax=axes[0], values_format=".2f")
axes[0].set_title(f"Matrice de confusion normalisée (test, seuil {THRESH:.2f})")
RocCurveDisplay.from_predictions((yp_te == 0).astype(int), p_te_prud, name=best_profile_name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("Courbe ROC – classe Prudent (test)")
plt.tight_layout()
savefig("07_profil_confusion_roc")
plt.show()
"""),
md(r"""
### 3.11 Validation externe sur une autre vague de l'enquête (SCF 2019)
Test plus exigeant : le modèle entraîné sur 2022, **sans ré-entraînement ni nouveau seuil**, est appliqué aux ménages de l'enquête **2019** (échantillon indépendant, interrogé avant le COVID). Les montants de 2019 sont convertis en dollars de 2022 avec l'indice des prix (CPI) de la FRED.
"""),
code(r"""
try:
    scf_ext = dl.download_scf(year=SCF_EXTERNAL_YEAR)
except Exception as exc:
    scf_ext = None
    print("⚠ SCF 2019 indisponible, validation externe ignorée :", exc)
if scf_ext is not None:
    X_ext, y_ext, g_ext, w_ext, rep_ext = prepare_scf(scf_ext)
    cpi = fred_raw["CPIAUCSL"].dropna()
    infl = cpi.loc[str(SCF_YEAR)].mean() / cpi.loc[str(SCF_EXTERNAL_YEAR)].mean()
    for c in [c for c in MONEY_FEATURES if c in X_ext.columns]:
        X_ext[c] = X_ext[c] * infl
    X_ext = X_ext.reindex(columns=X_p.columns)
    y_ext_int = y_ext.astype(str).map(CLS_TO_INT).astype(int)
    profil_externe = pd.DataFrame({
        f"Test interne – SCF {SCF_YEAR}": eval_profile(best_profile, Xp_te, yp_te, THRESH),
        f"Validation externe – SCF {SCF_EXTERNAL_YEAR}": eval_profile(best_profile, X_ext, y_ext_int, THRESH),
        f"Validation externe – SCF {SCF_EXTERNAL_YEAR} (pondéré WGT)": eval_profile(best_profile, X_ext, y_ext_int, THRESH,
                                                                                     w_ext.values),
    }).T
    profil_externe.to_csv(RESULTS_DIR / "agent_profil_validation_externe.csv")
    RESULTS["profil_externe"] = profil_externe
    display(profil_externe)
chrono("3. Agent Profil (modèles, test, validation externe)")
"""),
]
