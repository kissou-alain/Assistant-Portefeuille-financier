from cellules import code, md

CELLS = [
md(r"""
## 2. Prétraitement et analyse exploratoire des marchés

> **Rappel du cours (défis du ML)** : « des données de mauvaise qualité génèrent de faux résultats, donc de mauvaises décisions ». La qualité et la préparation des données passent avant le choix du modèle.

### 2.1 Diagnostic de qualité des données brutes
Problèmes typiques des données financières réelles :
- **calendriers différents** : le Bitcoin cote 7 jours sur 7, les ETF seulement les jours ouvrés de la bourse de New York (NYSE) ;
- **historiques de longueurs différentes** (Bitcoin disponible à partir de septembre 2014) ;
- **valeurs manquantes** ponctuelles (jours fériés, trous de l'API) ;
- **valeurs aberrantes** (pics isolés dus à des erreurs de cotation).
"""),
code(r"""
from src.preprocessing import clean_prices, compute_returns, build_macro_features, risk_free_daily

first = close_raw.apply(lambda s: s.first_valid_index())
quality = pd.DataFrame({
    "Première cotation": first.dt.date,
    "Nb observations": close_raw.count(),
    "% manquants après 1re cotation": [close_raw.loc[first[c]:, c].isna().mean() * 100 for c in close_raw],
    "Jours à volume nul": (volume_raw == 0).sum(),
})
quality
"""),
md(r"""
**Nettoyage** (`clean_prices`) : (1) tri et suppression des dates en double ; (2) alignement sur le calendrier NYSE ; (3) suppression des prix nuls ou négatifs ; (4) suppression des pics aberrants « aller-retour » de plus de 40 % en deux jours (hors Bitcoin) ; (5) propagation de la dernière valeur connue sur 5 jours au plus. Le remplissage se fait **uniquement vers l'avant** : remplir vers l'arrière utiliserait une information future.
"""),
code(r"""
prices, clean_report = clean_prices(close_raw)
for k in ("dates_brutes", "dates_calendrier_NYSE", "prix_non_positifs", "pics_aberrants"):
    print(f"{k:<28}: {clean_report[k]}")
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(prices.isna().T.astype(int).iloc[:, ::5], cbar=False, cmap="Greys", ax=ax)
ax.set_title("Disponibilité des données après nettoyage (noir = manquant)")
ax.set_xticks([])
savefig("01_disponibilite_donnees")
plt.show()
"""),
md(r"""
### 2.2 Rendements et valeurs extrêmes
On travaille sur les **rendements** $r_t = P_t / P_{t-1} - 1$, à peu près stationnaires, et non sur les prix, qui ne le sont pas.

Une valeur extrême n'est pas forcément une erreur. Le **z-score** (méthode statistique de détection d'anomalies du cours) repère les rendements à plus de 6 écarts-types. Ce sont de vrais événements de marché (krach de mars 2020…) : ils sont **conservés**, car ce sont précisément les risques que le système doit apprendre à gérer.
"""),
code(r"""
returns = compute_returns(prices)
z = (returns - returns.mean()) / returns.std()
extremes = (z.abs() > 6).sum().sort_values(ascending=False)
print("Nombre de rendements à plus de 6 écarts-types (conservés) :")
display(extremes[extremes > 0].to_frame("nb").T)
spy_ext = returns["SPY"][z["SPY"].abs() > 6].sort_values()
print("Exemples (S&P 500) :")
display(spy_ext.head(6).to_frame("rendement du jour"))
"""),
md(r"""
### 2.3 Variables macroéconomiques sans fuite temporelle
Une donnée macroéconomique n'est utilisable qu'à sa **date de publication**, pas à sa date de référence. Sinon, le modèle utiliserait une information qui n'existait pas encore : c'est une **fuite de données**.
- Séries quotidiennes (taux, VIX) : décalées d'un jour ouvré (publiées à J+1).
- Séries mensuelles (inflation, chômage) : la valeur du mois M n'est disponible qu'à partir du 15 du mois M+1.
"""),
code(r"""
macro = build_macro_features(fred_raw, prices.index)
rf_daily = risk_free_daily(macro)
display(macro.describe().T[["count", "mean", "std", "min", "max"]])

prices.to_csv(DATA_PROC / "prices_clean.csv")
returns.to_csv(DATA_PROC / "returns_daily.csv")
macro.to_csv(DATA_PROC / "macro_features.csv")
print("Données nettoyées enregistrées dans", DATA_PROC)
"""),
md(r"""
### 2.4 Analyse exploratoire
Pour ne pas orienter nos choix de modélisation avec des informations de la période de test, les statistiques qui **guident la modélisation** sont calculées sur la période d'entraînement (2010-2019). Les graphiques de contexte couvrent toute la période.
"""),
code(r"""
train_mask = returns.index <= TRAIN_END
R_tr = returns[train_mask]


def describe_returns(R: pd.DataFrame) -> pd.DataFrame:
    w = (1 + R.fillna(0)).cumprod()
    out = pd.DataFrame({
        "Classe": pd.Series(ASSET_CLASS),
        "Rdt annualisé": R.mean() * 252,
        "Vol annualisée": R.std() * np.sqrt(252),
        "Asymétrie": R.skew(),
        "Kurtosis (excès)": R.kurt(),
        "Max drawdown": (w / w.cummax() - 1).min(),
        "p-val Jarque-Bera": [stats.jarque_bera(R[c].dropna()).pvalue for c in R],
    })
    out["Sharpe (rf=0)"] = out["Rdt annualisé"] / out["Vol annualisée"]
    return out.loc[R.columns]


stats_train = describe_returns(R_tr)
stats_train.to_csv(RESULTS_DIR / "stats_actifs_train.csv")
stats_train.sort_values("Classe")
"""),
md(r"""
**Lecture** : tous les rendements journaliers ont une kurtosis en excès positive (**queues épaisses**), et la normalité est rejetée (p-valeurs de Jarque-Bera ≈ 0). Conséquences pour la suite : (i) mesurer le risque aussi par des quantiles historiques (VaR, CVaR), pas seulement par la volatilité ; (ii) écrêter les variables extrêmes (winsorisation) dans les pipelines ; (iii) préférer des modèles robustes aux valeurs extrêmes, comme les arbres.
"""),
code(r"""
show = ["SPY", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "VNQ", "DBC"]
norm = prices[show].dropna().pipe(lambda d: d / d.iloc[0] * 100)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
norm.plot(ax=axes[0], lw=1.2)
axes[0].set_title("Performance cumulée (base 100)")
(prices["BTC-USD"].dropna() / prices["BTC-USD"].dropna().iloc[0] * 100).plot(ax=axes[1], logy=True, color="darkorange")
axes[1].set_title("Bitcoin (échelle logarithmique)")
for d in ("2020-03-23", "2022-10-12"):
    for ax in axes:
        ax.axvline(pd.Timestamp(d), color="red", ls="--", alpha=.5)
savefig("02_performances_cumulees")
plt.show()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(R_tr.corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"shrink": .7})
ax.set_title("Corrélation des rendements quotidiens (période d'entraînement)")
savefig("03_correlations")
plt.show()
"""),
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
returns["SPY"].rolling(126).corr(returns["TLT"]).plot(ax=axes[0], color="purple")
axes[0].axhline(0, color="k", lw=.8)
axes[0].set_title("Corrélation glissante 6 mois actions / obligations longues")

r = returns["SPY"].dropna()
sns.histplot(r, bins=150, stat="density", ax=axes[1], color="steelblue")
xx = np.linspace(r.min(), r.max(), 300)
axes[1].plot(xx, stats.norm.pdf(xx, r.mean(), r.std()), "r", label="Loi normale")
axes[1].set_xlim(-0.06, 0.06)
axes[1].legend()
axes[1].set_title("Distribution des rendements du S&P 500")

lags = range(1, 31)
axes[2].bar([l - .2 for l in lags], [r.autocorr(l) for l in lags], width=.4, label="rendements")
axes[2].bar([l + .2 for l in lags], [r.abs().autocorr(l) for l in lags], width=.4, label="|rendements|")
axes[2].set_title("Autocorrélations : regroupement de volatilité")
axes[2].legend()
savefig("04_faits_stylises")
plt.show()

cols = [c for c in ["VIX", "T10Y2Y", "BAA10Y", "CPI_yoy", "UNRATE"] if c in macro]
macro[cols].plot(subplots=True, figsize=(13, 9), lw=1, title="Indicateurs macro-financiers (FRED)")
plt.tight_layout()
savefig("05_macro")
plt.show()
chrono("2. Prétraitement et EDA")
"""),
md(r"""
**Faits stylisés et conséquences pour la modélisation**
1. **Diversification instable** : la corrélation actions / obligations, négative pendant les années 2010, devient positive en 2022 (choc d'inflation). La covariance doit donc être réestimée en continu (fenêtre glissante d'un an).
2. **Queues épaisses** : voir ci-dessus.
3. **Regroupement de volatilité** : les rendements sont presque non autocorrélés, mais leurs valeurs absolues le sont fortement. Le **niveau de risque** est donc prévisible, beaucoup plus que la **direction** des prix. Cela justifie un agent Régime, et laisse attendre un pouvoir prédictif faible pour l'agent Prévision.
4. **Non-stationnarité des niveaux macro** : la courbe des taux s'est inversée de 2022 à 2024 (T10Y2Y < 0), un niveau jamais observé dans l'entraînement. Les modèles utiliseront de préférence des **variations** et des **logarithmes**, plus stables dans le temps.
"""),
]
