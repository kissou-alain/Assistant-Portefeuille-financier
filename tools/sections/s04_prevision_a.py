from cellules import code, md

CELLS = [
md(r"""
## 4. Agent Prévision – apprentissage supervisé sur données temporelles

### 4.1 Formulation (cours 2, « cas des données temporelles »)
Pour chaque actif *i* et chaque date *t*, on prédit son **rendement excédentaire futur** sur un mois :

$$y_{i,t} = r_{i,\,t\to t+H} \;-\; \frac{1}{N_t}\sum_{j} r_{j,\,t\to t+H}$$

C'est une **régression**. On retire la moyenne de tous les actifs à la même date : l'effet « marché » commun disparaît et seul compte le **classement relatif**, l'information utile pour choisir *entre* les actifs.

Les données sont préparées par **fenêtre glissante** (cours 2) : à chaque date, les variables résument le passé observé (*lookback*), et la cible porte sur l'horizon futur (*forecast horizon*).
"""),
code(r"""
from sklearn.model_selection import RandomizedSearchCV
from src.features import (compute_asset_features, compute_market_features, sample_dates, temporal_split,
                          ASSET_NUM_FEATURES, MACRO_FEATURES, CAT_FEATURES, REGIME_FEATURES)
from src.temporal import (window_table, TemporalCV, describe_splits, regression_metrics, information_coefficient,
                          newey_west_tstat, quintile_spread, forecast_report, evaluate_three_ways,
                          loss_by_date, diebold_mariano, dm_table, H_STEPS)
from src.pca_analysis import kmo, bartlett_sphericity, ACP

panel = compute_asset_features(prices, macro)
FEATS = ASSET_NUM_FEATURES + [c for c in MACRO_FEATURES if c in panel.columns] + CAT_FEATURES
NUM_F = [c for c in FEATS if c not in CAT_FEATURES]
panel.to_csv(DATA_PROC / "panel_features.csv.gz", compression="gzip")
print("Panel (date × actif) :", panel.shape)
window_table(lookback=252, horizon=HORIZON, stride=SAMPLE_EVERY, input_size=len(FEATS))
"""),
md(r"""
**Variables d'entrée** (toutes calculées avec l'information disponible en *t*) :

| Famille | Variables |
|---|---|
| Momentum | rendements passés à 5 j, 1, 3, 6 et 12 mois ; momentum 12-1 (12 mois sauf le dernier) |
| Risque | volatilités à 1 et 3 mois, leur rapport, drawdown sur 1 an, asymétrie |
| Tendance | écarts aux moyennes mobiles 50 et 200 jours, RSI |
| Lien au marché | bêta et corrélation au S&P 500 sur 3 mois |
| Position relative | rangs transversaux du momentum et de la volatilité |
| Macro (FRED, décalée) | VIX et sa variation, pente de courbe, spread de crédit et sa variation, variation du taux 10 ans, inflation, chômage, production industrielle |
| Catégorielle | classe d'actifs (one-hot) |

**Choix du stride S = 5 jours** : avec S = 1, deux échantillons consécutifs auraient 20 jours de cible en commun (quasi-doublons) ; avec S = 21, il resterait trop peu de dates (≈ 110 en entraînement). Le chevauchement résiduel (4 échantillons) est corrigé dans les statistiques de test (Newey-West, Diebold-Mariano).

### 4.2 Découpage temporel purgé (avant tout prétraitement)
- **Entraînement** : 2011 → 2019 | **Validation** : 2020-2021 (choc COVID) | **Test** : 2022 → fin (inflation, hausse des taux).
- **Purge / embargo de 35 jours** avant chaque frontière : la cible d'une observation du 20 décembre 2019 dépend des prix de janvier 2020 (validation). Elle est supprimée, sinon le modèle apprendrait sur des rendements de la validation.
- Aucun mélange aléatoire : on ne prédit jamais le passé avec le futur.
"""),
code(r"""
panel_s = sample_dates(panel).dropna(subset=["target_reg"])
tr, va, te = temporal_split(panel_s)
for nm, d in (("entraînement", tr), ("validation", va), ("test", te)):
    dd = d.index.get_level_values("date")
    print(f"{nm:<13}: {dd.min().date()} → {dd.max().date()} | {len(d):>6} obs | {dd.nunique():>4} dates")
gap = va.index.get_level_values("date").min() - tr.index.get_level_values("date").max()
assert gap.days >= EMBARGO_DAYS
print(f"Écart entraînement → validation : {gap.days} jours (≥ embargo de {EMBARGO_DAYS} jours)")
d_tr = tr.index.get_level_values("date")
print(f"Écart-type de la cible (entraînement) : {tr['target_reg'].std():.4f} soit ±{tr['target_reg'].std():.1%} par mois")
"""),
md(r"""
### 4.3 Analyse exploratoire du pouvoir prédictif (entraînement)
Le **coefficient d'information (IC)** d'une variable est la corrélation de rang (Spearman) entre cette variable et le rendement futur, calculée date par date puis moyennée. En gestion d'actifs, un IC de 0,03 à 0,05 **stable** est déjà considéré comme exploitable.
"""),
code(r"""
ic_uni = pd.Series({f: information_coefficient(tr[f], tr["fwd_ret"]).mean() for f in NUM_F}).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ic_uni.plot.barh(ax=axes[0], color=np.where(ic_uni > 0, "seagreen", "indianred"))
axes[0].set_title("IC univarié moyen de chaque variable (entraînement)")
sns.heatmap(tr[ASSET_NUM_FEATURES].corr(), cmap="RdBu_r", center=0, ax=axes[1])
axes[1].set_title("Corrélations entre variables techniques (entraînement)")
plt.tight_layout()
savefig("10_prevision_ic_univarie")
plt.show()
print("IC univariés les plus forts (en valeur absolue) :")
print(ic_uni.reindex(ic_uni.abs().sort_values(ascending=False).index).head(6).round(3).to_string())
print(f"Médiane des |IC| : {ic_uni.abs().median():.3f}")
"""),
md(r"""
**Lecture** : la plupart des IC univariés sont faibles (médiane de |IC| ≈ 0,04), ce qui est **normal en finance**, les marchés étant proches de l'efficience. Les variables les plus « prédictives » (corrélation et bêta au S&P 500, volatilité, toutes de signe positif) repèrent en réalité **les actions par opposition aux obligations** : sur 2011-2019, marché haussier, les actifs risqués ont nettement battu les actifs défensifs. Il s'agit d'une **prime de classe d'actifs propre à la période d'entraînement**, pas d'un signal de sélection garanti : si le contexte change (2022 : actions et obligations baissent ensemble), ce signal peut disparaître. C'est un risque de **dérive des données** (*data drift*, cours 1). Les variables techniques sont par ailleurs **fortement corrélées entre elles** (différents horizons de momentum, différentes mesures de volatilité). D'où deux conséquences : il faudra des modèles **régularisés**, et une **réduction de dimension** est envisageable.

### 4.4 Réduction de dimension : Analyse en Composantes Principales (cours 2)
L'ACP remplace des variables corrélées par de nouvelles variables **non corrélées**, les composantes principales $F_k = X V_k$, où les $V_k$ sont les vecteurs propres de la matrice de variance-covariance $S = \frac{1}{n-1}X^tX$ des variables standardisées, rangés par valeur propre $\lambda_k$ décroissante. La part de variance expliquée par $F_k$ vaut $\lambda_k / \sum_j \lambda_j$.

**Vérification préalable** (N.B. du cours) : l'ACP n'a d'intérêt que si les variables sont corrélées → indice **KMO ≥ 0,6** et **test de sphéricité de Bartlett** significatif. Tout est calculé sur l'entraînement.
"""),
code(r"""
X_acp = tr[ASSET_NUM_FEATURES].dropna()
X_acp = X_acp.clip(X_acp.quantile(0.005), X_acp.quantile(0.995), axis=1)
kmo_global, kmo_var = kmo(X_acp)
bart = bartlett_sphericity(X_acp)
print(f"KMO global = {kmo_global:.3f}  (≥ 0,6 : ACP justifiée)")
print(f"Bartlett : χ² = {bart['chi2']:,.0f}, ddl = {bart['ddl']}, p-valeur = {bart['p-valeur']:.2g}")
acp = ACP().fit(X_acp)
var_tab = acp.variance_table()
crit = acp.n_components()
display(var_tab.head(10))
print("Nombre de facteurs selon chaque critère :", crit)
"""),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
lam = var_tab["valeur propre"]
axes[0].bar(range(1, len(lam) + 1), lam, color="steelblue")
axes[0].plot(range(1, len(lam) + 1), var_tab["% cumulé"] / 100 * lam.max(), "o-", color="darkorange",
             label="% de variance cumulé (échelle de droite)")
axes[0].axhline(1, color="r", ls="--", label="Critère de Kaiser (λ = 1)")
axes[0].set_xlabel("Composante")
axes[0].set_ylabel("Valeur propre")
axes[0].set_title("Éboulis des valeurs propres")
axes[0].legend()

C = acp.correlations(2)
axes[1].add_patch(plt.Circle((0, 0), 1, fill=False, color="grey"))
for var, (a, b) in C.iterrows():
    axes[1].arrow(0, 0, a, b, head_width=0.02, color="slateblue", alpha=.8)
    axes[1].text(a * 1.08, b * 1.08, var, fontsize=8, ha="center")
axes[1].axhline(0, color="k", lw=.5)
axes[1].axvline(0, color="k", lw=.5)
axes[1].set_xlim(-1.1, 1.1)
axes[1].set_ylim(-1.1, 1.1)
axes[1].set_aspect("equal")
axes[1].set_xlabel(f"F1 ({var_tab.iloc[0, 1]:.0f} %)")
axes[1].set_ylabel(f"F2 ({var_tab.iloc[1, 1]:.0f} %)")
axes[1].set_title("Cercle des corrélations")
plt.tight_layout()
savefig("10b_acp")
plt.show()
display(acp.correlations(4).round(2).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
chrono("4a. Prévision : données, EDA, ACP")
"""),
md(r"""
**Interprétation financière des composantes** (corrélations variables–facteurs) :
- **F1 – « Tendance »** : corrélé à tous les rendements passés (1 à 12 mois) et aux écarts aux moyennes mobiles. Un actif à F1 élevé est en tendance haussière.
- **F2 – « Risque »** : corrélé aux volatilités (1 et 3 mois) et à leur rang. Il oppose les actifs agités aux actifs calmes.
- **F3 – « Retournement »** : oppose le momentum long (6-12 mois) au momentum court (1 mois) ; il capte les actifs qui corrigent après une longue hausse.
- **F4 – « Sensibilité au marché »** : corrélé au bêta et à la corrélation au S&P 500.

Les critères ne s'accordent pas sur le nombre de facteurs (Kaiser : 5 ; 80 % de variance : 6 ; 90 % : 9 ; coude : 2). On testera en 4.6 les modèles **avec et sans ACP** dans le pipeline. L'ACP y est ajustée sur l'entraînement uniquement, comme toute autre étape de préparation.
"""),
]
