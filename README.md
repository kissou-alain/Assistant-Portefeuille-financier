# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers

Projet de Machine Learning – École Centrale de Casablanca, cours « Introduction aux techniques d'apprentissage automatique ».

**Application en ligne : https://assistant-portefeuille-financier.vercel.app** (détails en [fin de page](#version-en-ligne)).

Sept agents spécialisés coopèrent pour estimer le profil de risque d'un client, lire l'état du marché, détecter les journées anormales, construire une allocation diversifiée sous contraintes et l'expliquer. Le système est évalué sans fuite de données, avec des baselines et des tests statistiques, puis déployé dans une application web (API FastAPI + interface Next.js).

```
Client ──► Agent Profil ───────┐  profil retenu (questionnaire + capacité estimée), alerte d'incohérence
Marché ──► Agent Régime ───────┤  Calme / Neutre / Stress + probabilités
Marché ──► Agent Alerte ───────┤  journée anormale ? (4 détecteurs)
Panel  ──► Agent Prévision ────┤  rendements excédentaires prévus (garde-fou statistique)
Prix   ──► Agent Risque ───────┤  covariance de Ledoit-Wolf, VaR, CVaR
                               ▼
              Agent Allocation ⇄ Agent Risque   (renégociation si une limite est dépassée)
                               ▼
                       Agent Explication ──► recommandation justifiée + journal des échanges
```

| Agent | Tâche | Méthodes (cours) | Modèle retenu |
|---|---|---|---|
| Profil | Classification binaire Prudent / Non prudent | Logistique, KNN, SVM, arbre, forêt, LightGBM, XGBoost, CatBoost, MLP ; SMOTE, ADASYN | Régression logistique (AUC test 0,867) |
| Prévision | Régression du rendement excédentaire à 21 jours | Fenêtre glissante, MCO, Ridge, Lasso, Elastic Net, KNN, SVR, arbres, boosting, MLP, ensemble ; ACP ; 3 méthodes d'évaluation ; test de Diebold-Mariano | Ensemble, **non activé** (ne bat pas la prévision nulle) |
| Régime | Clustering des états du marché | K-means, CAH (Ward), DBSCAN, mélange gaussien (EM) | Mélange gaussien |
| Alerte | Détection d'anomalies | z-score, Grubbs, gaussienne multivariée, Isolation Forest, LOF, One-Class SVM | Vote ≥ 3 détecteurs sur 4 |
| Risque | Estimation et contrôle du risque | Ledoit-Wolf, VaR, CVaR | – |
| Allocation | Portefeuille optimal sous contraintes | Moyenne-variance + Black-Litterman ; références parité de risque, HRP ; agent PPO (renforcement) | Moyenne-variance |
| Explication | Justification | Coefficients, permutation, SHAP global et local | – |

## Résultats clés (période de test 2022 → novembre 2026)

- **Profil** : la régression logistique fait aussi bien que les ensembles d'arbres ; ses performances tiennent sur une autre vague de l'enquête (SCF 2019, AUC 0,843).
- **Prévision** : aucun des 13 modèles ne bat la prévision nulle de façon significative (test de Diebold-Mariano). Un hold-out unique aurait été trompeur. Le garde-fou, fixé à l'avance, empêche ces prévisions d'influencer l'allocation.
- **Régime et Alerte** : les régimes repèrent le risque à venir (volatilité future plus forte en Stress) ; toutes les crises connues sont détectées.
- **Allocation** : le risque de chaque profil est respecté ; pas de surperformance établie face à des règles simples de même risque (moins bien pour le profil Prudent). L'agent PPO ne se généralise pas au test.

Tous les tableaux : [`results/resultats_synthese.md`](results/resultats_synthese.md). Explication de chaque modèle : [`docs/fiches_modeles.md`](docs/fiches_modeles.md).

## Données (open data)

| Source | Contenu | Accès |
|---|---|---|
| Yahoo Finance (`yfinance`) | Cours ajustés quotidiens depuis 2010 de 25 actifs : SPY, QQQ, IWM, EFA, EEM, 8 ETF sectoriels, VNQ, SHY (monétaire), IEF, TLT, LQD, HYG, TIP, EMB, GLD, SLV, DBC, BTC-USD | Gratuit |
| FRED (Federal Reserve Bank of St. Louis) | DGS10, DGS2, DTB3, T10Y2Y, BAA10Y, VIXCLS, CPIAUCSL, UNRATE, INDPRO | Gratuit, sans clé |
| Survey of Consumer Finances 2022 et 2019 (Réserve fédérale) | Environ 4 600 ménages réels × 5 imputations : démographie, revenus, patrimoine, dettes, culture financière, tolérance au risque déclarée | Gratuit |

Détail des sources et des traitements : [`data/README_data.md`](data/README_data.md).

## Installation (Windows)

Prérequis : Python 3.10 ou plus récent (testé avec 3.12), Node.js 20 ou plus récent (testé avec 24), Git.

```powershell
git clone https://github.com/kissou-alain/Assistant-Portefeuille-financier.git
cd Assistant-Portefeuille-financier
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd web ; npm install ; cd ..
```

## Exécution

Le dépôt contient déjà les données, les modèles entraînés et les résultats : on peut lancer directement l'application (étape 3). Les étapes 1 et 2 servent à tout refaire depuis zéro.

```powershell
# 1. (Facultatif) Retélécharger les données réelles (mises en cache dans data/raw/)
.venv\Scripts\python.exe -m src.data_download

# 2. Construire et exécuter le notebook complet (≈ 22 minutes) : il entraîne les modèles,
#    écrit models/, results/ et reports/figures/
.venv\Scripts\python.exe tools\construire_notebook.py --executer

# 3. Lancer l'application (API sur le port 8000 + interface sur le port 3000, ouvre le navigateur)
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

- Le notebook final est `notebooks/Projet_MultiAgent_Portefeuille_v4.ipynb`. Il est généré à partir des fichiers de sections `tools/sections/sNN_*.py` (une section du plan par fichier) : on peut aussi l'ouvrir et l'exécuter directement dans Jupyter ou VS Code.
- **Mode rapide** : mettre `QUICK_RUN = True` dans la section 0 (`tools/sections/s00_introduction.py`) pour une exécution de test en environ 6 minutes (grilles réduites ; résultats non représentatifs).
- **Lancement manuel de l'application**, dans deux terminaux :
  `.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000` puis `cd web ; npm run dev`. Documentation de l'API : http://localhost:8000/docs
- **Application de repli** (Streamlit, version 1) : `.venv\Scripts\streamlit.exe run app/streamlit_app.py`
- **Mise à jour des données** depuis l'application (page Données) : téléchargement incrémental limité à 30 s, contrôles avant écriture, sauvegarde et restauration ; les modèles ne sont pas ré-entraînés.

## Reproduction des résultats

- Graine aléatoire fixée (`RANDOM_STATE = 42`, `src/config.py`).
- Les données brutes sont mises en cache : une fois téléchargées, les exécutions suivantes utilisent exactement les mêmes données. Pour figer une version, conservez `data/raw/`.
- Le notebook écrit tous les tableaux dans `results/` (CSV et `resultats_synthese.md`), les figures dans `reports/figures/`, et le temps de calcul de chaque étape dans la synthèse.

## Prévention des fuites de données

1. **Découpage avant tout prétraitement** : chronologique pour les marchés (entraînement ≤ 2019, validation 2020-2021, test 2022 →), avec une **purge de 35 jours** aux frontières (cibles à 21 jours qui se chevauchent) ; **par ménage** pour le SCF (les 5 imputations d'un ménage restent ensemble).
2. **Validation croisée temporelle purgée** (expanding et rolling window), jamais de validation croisée mélangée.
3. **Variables macro** décalées à leur date de publication réelle (J+1 pour les séries quotidiennes, le 15 du mois suivant pour l'inflation et le chômage).
4. Imputation, écrêtage, standardisation, ACP et rééchantillonnage SMOTE/ADASYN dans des `Pipeline` ajustés sur l'entraînement uniquement.
5. Tous les réglages (modèles, seuils, pénalité de rotation, réduction du risque, graine de PPO) sont choisis sur la validation ; la période de test sert à une **évaluation unique**. Limite assumée : cette période avait été consultée lors des versions précédentes du projet.
6. **Backtest walk-forward** : rebalancement mensuel, covariance sur l'année précédente, coûts de transaction de 10 points de base.

## Arborescence

```
├── README.md, requirements.txt, start.ps1
├── data/                 README_data.md ; raw/ (cache des téléchargements) ; processed/ (générées par le notebook)
├── src/                  code Python (importé par le notebook et l'API)
│   ├── config.py           univers, dates, contraintes par profil, libellés
│   ├── data_download.py    Yahoo Finance, FRED, SCF ; data_update.py : mise à jour sécurisée
│   ├── preprocessing.py    nettoyage, macro sans fuite, transformateurs
│   ├── features.py         variables, cibles, découpage purgé
│   ├── temporal.py         fenêtre glissante, CV expanding/rolling, métriques, Diebold-Mariano
│   ├── model_zoo.py        modèles de régression et baselines
│   ├── pca_analysis.py     ACP, KMO, Bartlett
│   ├── regimes.py          K-means, CAH, DBSCAN, GMM et indices
│   ├── anomalies.py        détection d'anomalies
│   ├── allocation.py       parité de risque, HRP
│   ├── rl_env.py           environnement Gymnasium et agent PPO
│   ├── explain.py          SHAP global et local
│   ├── agents.py           les agents et l'orchestrateur
│   ├── questionnaire.py    questionnaire noté de tolérance au risque
│   └── backtest.py         simulation, références, mesures de performance
├── notebooks/            Projet_MultiAgent_Portefeuille_v4.ipynb (version finale) et versions précédentes
├── tools/                construire_notebook.py et sections/ (sources du notebook) ;
│                         exporter_site.py et test_profil_parite.py (version en ligne)
├── .github/workflows/    mise à jour automatique quotidienne de la version en ligne
├── api/                  API FastAPI (main.py, services.py)
├── web/                  interface Next.js (pages : accueil, diagnostic, profil, portefeuille, marché,
│                         backtest, laboratoire, agents, données)
├── app/                  application Streamlit (repli)
├── models/               modèles entraînés (générés par le notebook)
├── results/              tableaux de résultats
├── reports/              rapport, présentation, script vidéo, figures
└── docs/                 plan de refonte, fiches explicatives des modèles
```

## Version en ligne

### 👉 [Ouvrir l'application : https://assistant-portefeuille-financier.vercel.app](https://assistant-portefeuille-financier.vercel.app)

Même interface que la version locale, publiée sur Vercel sous forme de site statique, sans serveur Python :

- **Les résultats viennent du code Python du projet.** `tools/exporter_site.py` appelle les mêmes fonctions que l'API (`api/services.py`) avec les modèles entraînés par le notebook. Il écrit leurs sorties dans `web/public/donnees/` : marché, régime, alertes, portefeuille de chaque profil, backtest, laboratoire.
- **Le profil du visiteur est calculé dans son navigateur** avec les paramètres appris du modèle : imputation, écrêtage, standardisation, encodage, coefficients de la régression logistique. L'explication SHAP est elle aussi calculée sur place. Aucune donnée saisie n'est envoyée à un serveur. Rien n'est ré-entraîné : c'est la même formule appliquée aux mêmes nombres. `tools/test_profil_parite.py` le vérifie sur 200 clients et 25 recommandations complètes, avec un écart maximal de 10⁻¹⁶ par rapport à l'API Python.
- **Mise à jour automatique** : chaque soir de semaine, après la clôture de New York, une tâche GitHub Actions (`.github/workflows/mise-a-jour-site.yml`) télécharge les nouveaux cours, relance l'export et enregistre le résultat dans le dépôt. Vercel republie alors le site. La tâche peut aussi être lancée à la main depuis l'onglet *Actions* du dépôt.
- La mise à jour manuelle des données (page Données) et la documentation de l'API ne sont disponibles que dans la version locale.

## Avertissement

Projet pédagogique. Les recommandations produites ne constituent pas un conseil en investissement. Les actifs sont cotés en dollars US ; l'effet de change et la fiscalité ne sont pas pris en compte.
