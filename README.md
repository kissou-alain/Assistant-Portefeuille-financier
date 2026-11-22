# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers

Projet de Machine Learning : un système de six agents spécialisés part du profil d'un investisseur et de l'état réel des marchés. Il en tire une allocation de portefeuille diversifiée, contrôlée en risque et expliquée.

```
Client ──► Agent Profil ─────┐
Marché ──► Agent Régime ─────┤
Panel  ──► Agent Prévision ──┼──► Agent Allocation ⇄ Agent Risque ──► Agent Explication ──► Recommandation
Prix   ──► Agent Risque ─────┘        (boucle de renégociation si une limite de risque est dépassée)
```

| Agent | Tâche | Méthodes |
|---|---|---|
| Profil | Classification 3 classes (Prudent / Équilibré / Dynamique) | Baselines, régression logistique, forêt aléatoire, LightGBM, XGBoost, MLP |
| Régime | Détection non supervisée Calme / Neutre / Stress | Mélange gaussien (algorithme EM), sélection par BIC |
| Prévision | Surperformance relative à 21 jours de 25 ETF | Baselines (classe majoritaire, momentum 12-1), régression logistique, forêt aléatoire, LightGBM, XGBoost, MLP |
| Risque | Covariance, volatilité, VaR, CVaR, contrôle des limites | Estimateur de Ledoit-Wolf |
| Allocation | Portefeuille optimal sous contraintes du profil | Optimisation moyenne-variance (SLSQP) |
| Explication | Justification en langage naturel | Synthèse des sorties et importance des variables |

## Données réelles utilisées

| Source | Contenu | Accès |
|---|---|---|
| Yahoo Finance (`yfinance`) | Cours ajustés quotidiens depuis 2010 de 25 actifs : SPY, QQQ, IWM, EFA, EEM, 8 ETF sectoriels, VNQ, SHY, IEF, TLT, LQD, HYG, TIP, EMB, GLD, SLV, DBC, BTC-USD | Gratuit |
| FRED (Federal Reserve Bank of St. Louis) | DGS10, DGS2, DTB3, T10Y2Y, BAA10Y, VIXCLS, CPIAUCSL, UNRATE, INDPRO | Gratuit, sans clé |
| Survey of Consumer Finances 2022 (Réserve fédérale) | Environ 4 600 ménages réels × 5 implicates : démographie, revenus, patrimoine, dettes, culture financière, tolérance au risque déclarée | Gratuit |

Le détail des sources et des traitements figure dans [`data/README_data.md`](data/README_data.md).

## Installation

```bash
git clone <url-du-depot> && cd projet
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

```bash
# 1. Télécharger les données réelles (mises en cache dans data/raw/)
python -m src.data_download

# 2. Exécuter le notebook complet : prétraitement, EDA, modèles, interprétabilité, backtest, sauvegarde
jupyter notebook notebooks/Projet_MultiAgent_Portefeuille.ipynb
#    ou en mode non interactif :
jupyter nbconvert --to notebook --execute --inplace notebooks/Projet_MultiAgent_Portefeuille.ipynb

# 3. Lancer l'application déployée
streamlit run app/streamlit_app.py
```

Pour un test rapide avec des grilles d'hyperparamètres réduites, mettez `QUICK_RUN = True` dans la première cellule de code du notebook.

## Reproduction des résultats

- Graine aléatoire fixée (`RANDOM_STATE = 42`, dans `src/config.py`).
- Les données brutes sont mises en cache : une fois téléchargées, les exécutions suivantes utilisent exactement les mêmes données. Pour figer une version, conservez `data/raw/`.
- Le notebook écrit tous les tableaux de résultats dans `results/` (CSV et `resultats_synthese.md`) et les figures dans `reports/figures/`.

## Prévention des fuites de données

1. **Agent Prévision** : découpage chronologique train (≤ 2019) / validation (2020-2021) / test (2022 → fin), fait **avant** tout prétraitement, avec une purge de 35 jours aux frontières (les cibles à 21 jours se chevauchent).
2. **Validation croisée temporelle purgée** (`PurgedTimeSeriesSplit`) pour le réglage des hyperparamètres.
3. **Variables macro** décalées à leur date de publication réelle : J+1 pour les séries quotidiennes, le 15 du mois suivant pour l'inflation et le chômage.
4. **SCF** : découpage **par ménage**, car les 5 implicates d'un même ménage restent dans le même ensemble.
5. Tous les imputeurs, écrêtages et normalisations sont dans des `Pipeline` scikit-learn ajustés sur l'entraînement uniquement.
6. **Backtest walk-forward** : modèle ré-entraîné chaque année avec les seules cibles connues à cette date.

## Arborescence

```
├── README.md
├── requirements.txt
├── data/
│   ├── README_data.md          # sources et traitements
│   ├── raw/                    # données brutes téléchargées (cache)
│   └── processed/              # données nettoyées (générées par le notebook)
├── src/
│   ├── config.py               # univers, dates, contraintes par profil
│   ├── data_download.py        # Yahoo Finance, FRED, SCF
│   ├── preprocessing.py        # nettoyage, macro sans fuite, transformateurs
│   ├── features.py             # variables, découpage purgé, CV temporelle, IC
│   ├── agents.py               # les 6 agents et l'orchestrateur
│   ├── questionnaire.py        # questionnaire noté de tolérance au risque
│   └── backtest.py             # simulation, benchmarks, métriques
├── notebooks/
│   └── Projet_MultiAgent_Portefeuille.ipynb
├── app/
│   └── streamlit_app.py        # déploiement
├── models/                     # modèles entraînés (.joblib)
├── results/                    # tableaux de performances
└── reports/                    # rapport, présentation, script vidéo, figures
```

## Avertissement

Projet académique. Les recommandations produites ne constituent pas un conseil en investissement.
