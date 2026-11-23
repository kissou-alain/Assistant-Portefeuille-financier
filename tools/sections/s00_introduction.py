from cellules import code, md

CELLS = [
md(r"""
# Assistant multi-agent pour la sélection intelligente de portefeuilles financiers

**Projet de Machine Learning** – École Centrale de Casablanca – cours « Introduction aux techniques d'apprentissage automatique ».
Notebook principal, version 4 (refonte alignée sur le cours).

## Problématique
Un conseiller doit proposer à chaque client un portefeuille **adapté à son profil** et **à l'état des marchés**, puis **justifier** sa recommandation. Nous construisons un système de plusieurs agents spécialisés qui coopèrent, chacun s'appuyant sur une famille de méthodes du cours :

| Agent | Question traitée | Paradigme du cours | Méthodes |
|---|---|---|---|
| **Profil** | Le client peut-il supporter des pertes ? | Supervisé – classification binaire | Logistique, KNN, SVM, arbres, forêts, boosting, MLP ; SMOTE / ADASYN |
| **Prévision** | Quels actifs vont surperformer le mois prochain ? | Supervisé – régression sur **données temporelles** | Ridge, Lasso, Elastic Net, KNN, SVR, arbres, boosting, MLP ; ACP |
| **Régime** | Le marché est-il calme ou stressé ? | Non supervisé – clustering | K-means, CAH, DBSCAN, mélange gaussien (EM) |
| **Alerte** | La journée est-elle anormale ? | Non supervisé – détection d'anomalies | z-score, Grubbs, gaussienne multivariée, Isolation Forest, LOF, One-Class SVM |
| **Risque** | Quel est le risque du portefeuille ? | Statistique | Covariance de Ledoit-Wolf, VaR, CVaR |
| **Allocation** | Quel portefeuille choisir ? | Optimisation / renforcement | Moyenne-variance (Black-Litterman), parité de risque, HRP, agent PPO |
| **Explication** | Pourquoi cette recommandation ? | Interprétabilité | Coefficients, permutation, SHAP global et local |

Les agents échangent par un **tableau noir** partagé, coordonnés par un **orchestrateur** : c'est une architecture d'**IA agentique**.

## Démarche (processus de développement d'un algorithme supervisé, cours 1)
compréhension du problème → collecte et nettoyage → compréhension des données (EDA) → préparation → séparation entraînement / validation / test → entraînement → évaluation → optimisation des hyperparamètres → déploiement.

## Plan du notebook
0. Configuration
1. Collecte des données réelles (open data)
2. Prétraitement et analyse exploratoire des marchés
3. **Agent Profil** – classification supervisée et classes déséquilibrées
4. **Agent Prévision** – apprentissage supervisé sur données temporelles (fenêtre glissante, ACP, régression régularisée, 3 méthodes d'évaluation, test de Diebold-Mariano)
5. **Agent Régime et Agent Alerte** – clustering et détection d'anomalies
6. Système multi-agent – orchestration et allocation
7. Backtest walk-forward sur la période de test
8. **Agent d'apprentissage par renforcement** (PPO)
9. Sauvegarde des artefacts pour l'application
10. Synthèse des résultats et conclusion

> **Règle de prévention des fuites de données** : les jeux d'entraînement, de validation et de test sont séparés **avant** tout prétraitement. Toute transformation qui apprend quelque chose des données (imputation, standardisation, ACP, rééchantillonnage SMOTE…) est placée dans un `Pipeline` et ajustée **sur l'entraînement uniquement**. La période de test ne sert à **aucun** choix.
"""),
md(r"""
## 0. Configuration

`QUICK_RUN = True` exécute le notebook en mode rapide (grilles réduites, moins de simulations), pour vérifier que tout fonctionne. Les résultats du rapport sont obtenus avec `QUICK_RUN = False`.
"""),
code(r"""
import sys
import json
import time
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from src.config import *
from src import data_download as dl

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.figsize": (12, 5), "axes.titleweight": "bold"})
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.max_columns", 30)
np.random.seed(RANDOM_STATE)

QUICK_RUN = False    # True : mise au point rapide | False : résultats finaux du rapport
N_ITER = 3 if QUICK_RUN else 10   # combinaisons d'hyperparamètres testées par modèle (recherche aléatoire)
N_JOBS = -1
RESULTS = {}         # tableaux collectés pour la synthèse finale

_T0 = _TLAST = time.time()
DUREES = {}


def chrono(etape: str):
    # Affiche le temps de calcul de l'étape qui vient de se terminer (objectif : notebook < 1 h 30)
    global _TLAST
    now = time.time()
    DUREES[etape] = now - _TLAST
    print(f"⏱ {etape} : {now - _TLAST:.0f} s (cumul {(now - _T0) / 60:.1f} min)")
    _TLAST = now


def savefig(name: str):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


print(f"Racine du projet : {ROOT} | mode rapide : {QUICK_RUN}")
"""),
]
