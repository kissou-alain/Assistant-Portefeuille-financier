from cellules import code, md

CELLS = [
md(r"""
## 5. Agent Régime et Agent Alerte – apprentissage non supervisé

### 5.1 Objectif et variables
L'agent Régime classe chaque journée de marché en **régimes** (Calme / Neutre / Stress) **sans étiquette** : c'est un problème de **clustering** (cours 2). L'état du marché à la date *t* est décrit par 5 variables **stationnaires** :
- rendement du S&P 500 sur 1 mois ;
- log-volatilité réalisée du S&P 500 sur 1 mois ;
- drawdown du S&P 500 (écart au plus haut sur 1 an) ;
- log du VIX (volatilité implicite) ;
- variation sur 3 mois du spread de crédit Baa − Trésor.

**Pourquoi des variables stationnaires** : la version 1 du projet utilisait des niveaux (pente de courbe, spread). La courbe des taux s'est inversée de 2022 à 2024, un niveau jamais vu à l'entraînement, et le modèle classait alors 91 % du test en « Stress ». Les logarithmes et les variations restent dans la plage de l'entraînement.

Les modèles sont ajustés sur **2011-2019 uniquement**, après standardisation apprise sur cette période, puis appliqués date par date (filtrage, sans regarder le futur).

### 5.2 Les quatre familles de clustering du cours
| Famille | Méthode | Principe | Nombre de groupes |
|---|---|---|---|
| Partition | **K-means** | minimise l'inertie intra-classe $J=\frac1n\sum_i\lVert x_i-\mu_{c^{(i)}}\rVert^2$ | fixé (coude, silhouette) |
| Hiérarchique | **CAH, critère de Ward** | fusionne à chaque étape les deux groupes dont la fusion augmente le moins l'inertie intra : $\frac{n_a n_b}{n_a+n_b}d^2(\mu_a,\mu_b)$ | choisi en coupant le dendrogramme |
| Densité | **DBSCAN** | un groupe = une zone dense ; les points isolés sont du **bruit** | déterminé par ε et min_samples |
| Probabiliste | **Mélange gaussien (GMM)** estimé par **EM** | $p(x)=\sum_k\pi_k\,\mathcal N(x;\mu_k,\Sigma_k)$ ; maximise la log-vraisemblance | fixé (critère BIC) |

**Règle fixée à l'avance** : l'agent Régime doit fournir des **probabilités d'appartenance** $\gamma_{ik}$ (affichées dans l'application, et réutilisées comme variables en 5.5). Seul le GMM en produit : c'est donc lui qui est utilisé par le système. Les trois autres méthodes servent de **comparaison** (indices de qualité) et de **contrôle de robustesse** (les régimes trouvés concordent-ils ?).
"""),
code(r"""
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from src.features import compute_market_features, REGIME_FEATURES
from src.regimes import (kmeans_elbow, gmm_bic, k_distance, fit_four_families, compare_families,
                         describe_clusters, label_by_volatility)

mkt_feat = compute_market_features(prices, macro)
reg_cols = [c for c in REGIME_FEATURES if c in mkt_feat]
M_tr = mkt_feat.loc[:TRAIN_END, reg_cols].dropna()
reg_scaler = StandardScaler().fit(M_tr)
Z_tr = reg_scaler.transform(M_tr)
print("Jours d'entraînement :", len(M_tr), "| variables :", reg_cols)

elbow = kmeans_elbow(Z_tr)
bic = gmm_bic(Z_tr)
kd = k_distance(Z_tr, k=10)
EPS = float(np.quantile(kd, 0.90))     # règle : ε = 90e centile des distances au 10e voisin
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
ax2 = axes[0].twinx()
elbow["inertie intra J_w"].plot(ax=axes[0], marker="o", color="steelblue", label="inertie intra")
elbow["silhouette"].plot(ax=ax2, marker="s", color="darkorange", label="silhouette")
axes[0].set_title("K-means : méthode du coude et silhouette")
axes[0].set_ylabel("Inertie intra-classe")
ax2.set_ylabel("Silhouette")
bic.plot(ax=axes[1], marker="o", color="purple")
axes[1].set_title("GMM : critère BIC (plus bas = meilleur)")
axes[2].plot(kd)
axes[2].axhline(EPS, color="r", ls="--", label=f"ε = {EPS:.2f}")
axes[2].set_title("DBSCAN : distance au 10e voisin (triée)")
axes[2].legend()
plt.tight_layout()
savefig("15_regime_choix_k")
plt.show()
"""),
md(r"""
**Choix K = 3** : le coude de K-means est peu marqué et le BIC du GMM continue de baisser au-delà de 3 (avec 2 400 jours, des groupes plus fins sont toujours un peu plus vraisemblables). On retient **3 régimes**, nombre directement **interprétable** et utilisable par un conseiller (Calme / Neutre / Stress). Le cours le rappelle : il faut choisir un nombre de groupes qui « met en lumière des structures intéressantes », pas seulement celui qui optimise un critère.
"""),
code(r"""
K = 3
fitted_cl = fit_four_families(Z_tr, k=K, eps=EPS, min_samples=10)
plt.figure(figsize=(12, 4))
dendrogram(fitted_cl["CAH (Ward)"][0], truncate_mode="lastp", p=30, color_threshold=None)
plt.title("CAH (Ward) : dendrogramme (30 derniers regroupements)")
plt.ylabel("Distance de fusion (Ward)")
savefig("15b_regime_dendrogramme")
plt.show()

cl_scores = compare_families(Z_tr, fitted_cl)
gmm_lab = fitted_cl["GMM (EM)"][1]
cl_scores["accord avec le GMM (indice de Rand ajusté)"] = [
    adjusted_rand_score(gmm_lab[lab >= 0], lab[lab >= 0]) for _, (_, lab) in fitted_cl.items()]
cl_scores.to_csv(RESULTS_DIR / "agent_regime_comparaison_clustering.csv")
RESULTS["regime_clustering"] = cl_scores
cl_scores
"""),
md(r"""
**Comment lire les indices** : **silhouette** (entre −1 et 1, ↑ = groupes compacts et séparés), **Davies-Bouldin** (↓), **Calinski-Harabasz** (↑), **% d'inertie expliquée** $J_b/(J_w+J_b)$ (↑). Pour DBSCAN, les indices sont calculés sans les points de bruit.

Le GMM obtient en général des indices géométriques moins bons que K-means et la CAH : c'est attendu. Ses groupes sont des **ellipses qui se chevauchent** (covariances différentes par régime), alors que ces indices favorisent des groupes sphériques bien séparés. Le GMM capte en revanche une structure que les autres ignorent : le régime de stress est **beaucoup plus dispersé** que le régime calme. L'**indice de Rand ajusté** mesure l'accord entre partitions (1 = identiques, 0 = accord dû au hasard).

### 5.3 Description des régimes du GMM (cours 2, « Décrire les clusters »)
"""),
code(r"""
gmm = fitted_cl["GMM (EM)"][0]
centers = pd.DataFrame(reg_scaler.inverse_transform(gmm.means_), columns=reg_cols)
label_map = label_by_volatility(centers, "mkt_logvol_21")
prof = describe_clusters(M_tr, gmm_lab)
prof.index = [label_map.get(i, i) if isinstance(i, (int, np.integer)) else i for i in prof.index]
prof["vol. annualisée"] = np.exp(prof["mkt_logvol_21"])
prof.loc["p-valeur Kruskal-Wallis", "vol. annualisée"] = np.nan
display(prof)

M_all = mkt_feat[reg_cols].dropna()
proba_all = pd.DataFrame(gmm.predict_proba(reg_scaler.transform(M_all)), index=M_all.index)
proba_all.columns = [label_map[c] for c in proba_all.columns]
proba_all = proba_all.T.groupby(level=0).sum().T[["Calme", "Neutre", "Stress"]]
reg_series = proba_all.idxmax(axis=1)
"""),
md(r"""
### 5.4 Validation économique hors échantillon
Un bon régime doit **annoncer** le risque à venir. On mesure, après chaque jour classé dans un régime, la volatilité réalisée et le rendement du S&P 500 **sur le mois suivant**. Le **η²** (part de la variance de la volatilité future expliquée par le régime) résume le pouvoir informatif du découpage. La validation 2020-2021 sert à juger le modèle ; les chiffres du test sont seulement **descriptifs** et ne servent à aucun choix.
"""),
code(r"""
fwd_ret_spy = prices["SPY"].shift(-21) / prices["SPY"] - 1
fwd_vol_spy = returns["SPY"].rolling(21).std().shift(-21) * np.sqrt(252)


def regime_eval(start, end):
    idx = reg_series.loc[start:end].index
    df = pd.DataFrame({"régime": reg_series.loc[idx], "rdt SPY 21 j futur": fwd_ret_spy.reindex(idx),
                       "vol. réalisée future": fwd_vol_spy.reindex(idx)}).dropna()
    tab = df.groupby("régime").agg(fréquence=("régime", "size"), **{
        "rdt moyen 21 j futur": ("rdt SPY 21 j futur", "mean"), "vol. future moyenne": ("vol. réalisée future", "mean")})
    tab["fréquence"] /= tab["fréquence"].sum()
    gm = df["vol. réalisée future"].mean()
    eta2 = sum(len(g) * (g["vol. réalisée future"].mean() - gm) ** 2 for _, g in df.groupby("régime")) / \
        ((df["vol. réalisée future"] - gm) ** 2).sum()
    return tab.reindex(["Calme", "Neutre", "Stress"]), float(eta2)


reg_eval = {}
for nom, (a, b) in {"Entraînement 2011-2019": (None, TRAIN_END), "Validation 2020-2021": ("2020-01-01", VAL_END),
                    "Test 2022 → (descriptif)": ("2022-01-01", None)}.items():
    tab, eta2 = regime_eval(a, b)
    print(f"{nom} : η² = {eta2:.2f}")
    display(tab)
    reg_eval[nom] = tab.assign(eta2=eta2)
reg_tab = pd.concat(reg_eval, names=["période", "régime"])
reg_tab.to_csv(RESULTS_DIR / "agent_regime_validation.csv")
RESULTS["regime"] = reg_tab
assert reg_eval["Validation 2020-2021"]["fréquence"].max() < 0.8, "Un régime domine : vérifier la stationnarité"
"""),
code(r"""
colors = {"Calme": "seagreen", "Neutre": "gold", "Stress": "crimson"}
fig, ax = plt.subplots(figsize=(15, 5))
spy = prices["SPY"].reindex(reg_series.index)
ax.plot(spy.index, spy.values, color="k", lw=.8)
for reg, col in colors.items():
    ax.fill_between(spy.index, 0, spy.max() * 1.05, where=(reg_series == reg).values, color=col, alpha=.25, label=reg)
ax.set_ylim(spy.min() * .9, spy.max() * 1.05)
for d, lab in ((TRAIN_END, "fin entraînement"), (VAL_END, "fin validation")):
    ax.axvline(pd.Timestamp(d), color="blue", ls="--")
    ax.text(pd.Timestamp(d), spy.max(), f" {lab}", color="blue")
ax.legend(loc="upper left")
ax.set_title("Régimes détectés par le GMM et S&P 500")
savefig("16_regimes")
plt.show()
trans = pd.crosstab(reg_series.shift(1), reg_series, normalize="index")
display(trans.rename_axis("de \\ vers").style.format("{:.1%}"))

regime_bundle = {"model": gmm, "scaler": reg_scaler, "features": reg_cols, "fill": M_tr.median().to_dict(),
                 "label_map": label_map}
mkt_feat.to_csv(DATA_PROC / "market_features.csv")
reg_series.rename("regime").to_csv(DATA_PROC / "regimes.csv")
chrono("5a. Agent Régime")
"""),
md(r"""
**Lecture** : sur les trois périodes, le régime « Stress » est suivi de la volatilité future la plus élevée : l'agent **repère bien le risque**. Une limite apparaît en validation : le régime « Calme » y est suivi d'une volatilité plus forte que le régime « Neutre ». C'est l'effet de **février 2020** : le marché était calme juste avant le krach du COVID. L'agent **réagit** au risque (avec quelques jours de retard) mais ne l'**anticipe** pas : aucune variable de marché ne pouvait annoncer une pandémie. En revanche, il n'annonce pas de baisse : les périodes de stress ont souvent été suivies de **rebonds** (rendement futur moyen plus élevé), ce qui rendra coûteuse une réduction systématique de l'exposition en régime de stress (point arbitré sur la validation en section 7). La matrice de transition montre une forte **persistance** des régimes (diagonale proche de 1), ce qui rend la détection en temps réel exploitable.
"""),
]
