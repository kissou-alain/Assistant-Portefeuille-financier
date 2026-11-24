# Plan de refonte – Assistant multi-agent de sélection de portefeuille

*Rédigé le 22/11/2026, à valider avant tout développement. Remise : lundi 30/11/2026 (8 jours).*

---

## 1. Écarts entre le projet actuel et le cours (Pr. Y. Lamrani Alaoui)

| Notion du cours | Où dans le cours | État actuel du projet | Action |
|---|---|---|---|
| Processus supervisé complet (problème → collecte → EDA → préparation → split → entraînement → évaluation → optimisation → déploiement) | Cours 1, p. 56 | Présent, mais pas présenté selon ce schéma | Restructurer le notebook selon ce schéma |
| Préparation par fenêtre glissante : *lookback L*, *horizon H*, *stride S*, *input size* | Cours 2, p. 4-7 | Implicite (variables sur 252 j, H = 21, un échantillon / 5 j), jamais nommé | Tableau explicite + justification de S |
| Hold-out temporel, CV **expanding**, CV **rolling (sliding)** et leur comparaison | Cours 2, p. 8-14 | Hold-out purgé + CV expanding purgée uniquement | Ajouter le rolling, comparer et interpréter les 3 |
| MSE, RMSE, MAE, MAPE | Cours 2, p. 9 | Absentes (tâche formulée en classification) | Cible principale passée en **régression** |
| Test de **Diebold-Mariano** | Cours 2, p. 15-16 | Absent (bootstrap du Sharpe seulement) | Critère principal de comparaison à la baseline |
| Régression linéaire, **Ridge, Lasso, Elastic Net**, effet de λ | Cours 1, p. 67-72 | Absents | Ajout + chemin de régularisation |
| Diagnostic biais/variance, courbes d'apprentissage (J_train vs J_cv) | Cours 1, p. 42-49 | Absent | Courbes d'apprentissage + remède du cours |
| Descente de gradient, Adam | Cours 1, p. 61-66 | MLP présent sans lien explicite | MLP (solver Adam) relié au cours |
| Three-way holdout : ré-entraîner sur train + validation avant le test, puis sur tout avant déploiement | Cours 1, p. 84-88 | À vérifier dans le notebook | Appliquer et documenter les étapes 4 et 6 |
| **ACP**, KMO, Bartlett, Kaiser, coude, cercle des corrélations | Cours 2, p. 17-37 | Absente | Ajout + comparaison des modèles avec/sans ACP |
| Clustering : **K-means, CAH (Ward), DBSCAN, GMM (EM)** | Cours 2, p. 38-65 | GMM + BIC uniquement | Comparer les 4 familles |
| Silhouette, Davies-Bouldin, Calinski-Harabasz, inerties intra/inter, description des clusters | Cours 2, p. 66-77 | Absents | Ajout |
| GMM comme extraction de variables (γ_ik) | Cours 2, p. 90 | Absent | γ_ik ajoutées à l'agent Prévision, apport testé par DM |
| Détection d'anomalies : z-score, Grubbs, gaussienne multivariée, Isolation Forest, LOF, One-Class SVM | Cours 1, p. 24 ; Cours 2, p. 78-85 | Seulement les pics de prix aberrants dans `clean_prices` | Agent Alerte + contrôle qualité |
| Classes déséquilibrées : **SMOTE, ADASYN**, pas d'accuracy seule | Cours 1, p. 57-59 | Seuil de décision ajusté ; pas de rééchantillonnage | `imblearn.Pipeline`, comparaison avec les poids de classes |
| Logistique, softmax, KNN, SVM, arbre, CatBoost | Cours 1, p. 13, 73-82 | Logistique, RF, LightGBM, XGBoost, MLP | Compléter la liste |
| **Apprentissage par renforcement** (exemple « gestion de portefeuille ») | Cours 1, p. 28-31 | Mentionné en perspective seulement | Environnement Gymnasium + PPO |
| Interprétabilité intrinsèque, explicabilité globale **et locale** | Cours 1, p. 51-54 | SHAP global + permutation dans le notebook ; rien de local dans l'application | SHAP waterfall dans le notebook et dans l'application |
| IA agentique (chap. 6) | Plan du cours | Architecture multi-agent présente, mais pas reliée au vocabulaire du cours | Présenter l'orchestrateur comme un système agentique |
| Semi-supervisé (self-training) | Cours 1, p. 25-27 | Absent | **Perspective seulement** (aucune donnée non étiquetée réelle) |

**À conserver tel quel** : purge et embargo, CV purgée, macro décalée à sa date de publication, découpage SCF par ménage, validation externe SCF 2019, logique « tolérance déclarée + capacité ML → profil le plus prudent », mise à jour incrémentale sécurisée des données.

---

## 2. Architecture cible

```
src/
  config.py              (existant) + classe « Monétaire », constantes L/H/S
  data_download.py       (existant, inchangé)
  preprocessing.py       (existant) + pipelines imblearn
  features.py            (existant) + cible régression (rendement excédentaire vs moyenne transversale)
  temporal.py            NOUVEAU : fenêtres, splits hold-out / expanding / rolling purgés,
                                   MSE/RMSE/MAE, IC + Newey-West, test de Diebold-Mariano
  model_zoo.py           NOUVEAU : définition de tous les modèles et de leurs grilles
  pca_analysis.py        NOUVEAU : KMO, Bartlett, éboulis, cercle des corrélations
  regimes.py             NOUVEAU : K-means, CAH, DBSCAN, GMM + indices + description
  anomalies.py           NOUVEAU : z-score, Grubbs, gaussienne multivariée, IForest, LOF, OCSVM
  allocation.py          NOUVEAU : risk parity, HRP, 1/N à volatilité égale (Markowitz/BL déplacés ici)
  rl_env.py              NOUVEAU : environnement Gymnasium + entraînement/évaluation PPO
  explain.py             NOUVEAU : SHAP global/local réutilisable par le notebook et l'API
  agents.py              (existant) + Agent Alerte, journal enrichi
  backtest.py            (existant) + nouvelles références
api/                     NOUVEAU : FastAPI (charge les artefacts précalculés au démarrage)
  main.py  routes/{profil, recommandation, regime, backtest, modeles, donnees}.py
web/                     NOUVEAU : Next.js + TypeScript + Tailwind + shadcn/ui + Recharts
notebooks/Projet_MultiAgent_Portefeuille.ipynb   restructuré selon le plan du cours
app/streamlit_app.py     conservé comme solution de repli jusqu'à ce que web/ soit stable
start.ps1                lance l'API et le front en une commande
docs/PLAN_REFONTE.md, docs/fiches_modeles.md
```

Principe : **tout le calcul vit dans `src/`**. Le notebook appelle `src/` et écrit des artefacts (`models/`, `results/`). L'API ne fait que charger ces artefacts et appeler les agents. Une recommandation (optimisation SLSQP sur 25 actifs) prend déjà bien moins d'une seconde. Les graphiques lourds (frontière, Monte-Carlo, SHAP global) sont précalculés ou mis en cache.

---

## 3. Modèles retenus et justification

### 3.1 Agent Prévision : régression du rendement excédentaire à H = 21 jours

**Formalisation (vocabulaire du cours 2)**

| Terme | Valeur | Justification |
|---|---|---|
| Lookback L | 252 jours de bourse (fenêtre la plus longue des variables) | Momentum 12-1, volatilités, drawdown sur 1 an |
| Horizon H | 21 jours, méthode **directe à H** | Rebalancement mensuel |
| Stride S | 5 jours (un échantillon par semaine) | Compromis : S = 1 donne des cibles presque identiques ; S = 21 laisse trop peu de dates. Le chevauchement résiduel est corrigé par Newey-West et DM avec ⌈H/S⌉ − 1 = 4 retards |
| Input size | ~30 variables par pas de temps (20 techniques + 10 macro + classe d'actifs), plus K = 3 avec les γ_ik du GMM | — |

**Cible** : y = rendement à 21 j de l'actif − moyenne transversale des actifs ce jour-là. La classification (y > médiane) est conservée comme tâche secondaire, et un LightGBM Ranker comme variante.

| Famille | Modèles | Rôle |
|---|---|---|
| Baselines | prévision nulle (y = 0), moyenne historique, momentum 12-1 | Référence du test DM |
| Linéaires régularisés | OLS, Ridge, Lasso, Elastic Net | Explicables ; coefficients et chemin de λ |
| Voisinage / noyaux | KNN, SVR | Exigés par la liste du cours |
| Arbres | arbre de décision, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost | Non-linéarités, interactions |
| Réseau | MLP (Adam) | Lien avec le cours sur la descente de gradient |
| Ensemble | moyenne des 3 meilleurs en validation | Réduction de variance |
| Variante | chacun des meilleurs modèles **avec ACP** dans le pipeline, et **avec γ_ik du GMM** | Apport testé par DM |

Écartés, avec trace dans le rapport : **LSTM/GRU** (≈ 1 000 dates hebdomadaires de train, sur-apprentissage quasi certain ; temps insuffisant pour une validation honnête). C'est une perspective.

**Évaluation**
- 3 schémas temporels purgés : hold-out (train ≤ 2019, validation 2020-21, test 2022 →), CV expanding, CV rolling. Tableau comparatif et interprétation selon le cours (p. 14).
- MSE, RMSE, MAE ; IC de Spearman + t de Newey-West ; spread Q5 − Q1 **net de 10 pb de coûts** ; pour la classification : précision, rappel, F1, ROC-AUC.
- **Diebold-Mariano** : la perte d_t est calculée **date par date** (erreur quadratique moyenne transversale), ce qui donne une série temporelle. Variance HAC avec h − 1 = 4 retards. On compare chaque modèle à la baseline, puis les meilleurs entre eux.
- Courbes d'apprentissage (taille du train en abscisse, J_train et J_cv) et courbe de validation selon λ ou la profondeur.
- Hyperparamètres : GridSearch / RandomizedSearch avec CV temporelle purgée. Pas d'Optuna, inutile ici.

**Attente honnête** : R² hors échantillon proche de 0 et IC de 0,03 à 0,08. Il est probable que **la plupart des modèles ne battent pas la baseline de façon significative** au sens de DM. Ce sera écrit tel quel : c'est le résultat attendu sur des marchés efficients, et c'est défendable devant le jury.

### 3.2 Agent Régime : clustering non supervisé

- Variables d'état stationnaires existantes (rendement 21 j, log-volatilité, drawdown, log VIX, variation du spread de crédit), standardisées sur le train.
- K-means (coude + silhouette), CAH Ward (dendrogramme, plus grand saut), DBSCAN (ε choisi par le graphe des k-distances), GMM-EM (BIC).
- Indices : inerties intra et inter-classes, silhouette, Davies-Bouldin, Calinski-Harabasz. Description des clusters (profil moyen, ANOVA ou Kruskal-Wallis comme au cours 2 p. 77).
- Validation économique hors échantillon : volatilité et rendement futurs par régime.
- **GMM retenu dans le système** s'il reste compétitif, car il fournit des probabilités et les γ_ik. Le choix se fait sur le train et la validation.

### 3.3 Détection d'anomalies : Agent Alerte

z-score et Grubbs sur les rendements de SPY ; gaussienne multivariée (seuil ε choisi sur le train), Isolation Forest, LOF, One-Class SVM sur le vecteur journalier. Il n'y a pas de vraie étiquette : on fait une **validation qualitative** (les krachs connus de 2011, 2015, 2018, 2020 et 2022 sont-ils détectés ?) et on mesure l'accord entre méthodes. Rôle : alerte affichée dans l'application et contrôle qualité des données téléchargées.

### 3.4 Agent Profil : classification SCF

- Baselines (classe majoritaire, tirage stratifié), **logistique** (référence explicable), KNN, SVM, arbre, RF, XGBoost, LightGBM, CatBoost.
- Déséquilibre (~32 % de « Prudent ») : sans correction, avec poids de classes, avec **SMOTE**, avec **ADASYN**, dans un `imblearn.Pipeline` (le rééchantillonnage n'est appliqué qu'aux plis d'entraînement).
- Le multi-classes softmax est présenté en théorie, et la tâche à 3 classes reste en comparaison. La formulation binaire est conservée (la justification « Dynamique ≈ 5 % » reste valable).
- Seuil choisi en validation (rappel « Prudent » ≥ 80 %), test interne, puis validation externe SCF 2019.
- SHAP local (waterfall) pour un client, exposé par l'API.

### 3.5 Agent Allocation et agent RL

- Références : Markowitz/Black-Litterman actuel, risk parity, HRP, 1/N à volatilité égale, allocation stratégique par profil.
- **Poche monétaire** : SHY passe dans une classe « Monétaire ». La contrainte « obligations ≥ 50 % » du profil Prudent devient « défensif (obligations + monétaire) ≥ 50 % », avec un plafond sur la duration longue (TLT). Justification ex ante : un investisseur prudent à horizon court doit pouvoir détenir du quasi-monétaire sans risque de taux. C'est la logique des allocations MiFID / AMMC.
  - ⚠️ **Honnêteté** : la crise obligataire de 2022 est connue et la période de test a déjà été vue (versions v1 à v3 du notebook). Le rapport le dira explicitement, et le choix sera fixé **avant** toute nouvelle exécution sur le test.
- **RL** : environnement Gymnasium mensuel.
  - État : variables de marché, γ_ik du régime, poids actuels.
  - Action : vecteur de logits passé par un softmax, puis projeté sur les poids admissibles du profil.
  - Récompense : rendement − λ·variance − coûts de transaction.
  - Algorithme : PPO (`stable-baselines3`), entraînement sur 2011-2019, choix de λ, de γ et du nombre de pas en validation 2020-21, **une seule évaluation sur le test**, comparaison par bootstrap par blocs du Sharpe.
  - Risque majeur : environ 108 pas mensuels en entraînement, ce qui est très peu pour du RL. Mitigation : épisodes tirés aléatoirement dans la période d'entraînement, réseau petit, pas hebdomadaires si nécessaire.
  - Si PPO ne bat pas les références, ce sera dit, et l'application garde l'optimiseur classique.

### 3.6 Interprétabilité

Coefficients (logistique, Lasso/Ridge) ; SHAP global (beeswarm, |SHAP| moyen) et importance par permutation ; SHAP local (client, actif) dans le notebook **et** dans l'application ; analyse des erreurs par classe d'actifs, année, régime et tranche d'âge.

### 3.7 IA agentique

L'architecture (orchestrateur, tableau noir, agents, négociation Allocation ⇄ Risque, Agent Alerte) sera présentée dans le vocabulaire du chapitre 6. L'Agent Explication reste fondé sur des gabarits de texte et sur les valeurs SHAP. Aucun service externe n'est utilisé.

### 3.8 Hors périmètre (décision du 22/11)

Retirés du projet : LSTM/GRU, explication par LLM et RAG, Docker, apprentissage semi-supervisé. Le rapport les citera au plus comme perspectives, en une ligne chacun. L'objectif prioritaire est de pouvoir **expliquer la méthode et le choix de chaque modèle** : c'est le rôle de `docs/fiches_modeles.md` et des cellules Markdown du notebook, qui doivent rester cohérentes avec le code.

---

## 4. Application web

- **API FastAPI** : `/profil` (avec SHAP local), `/recommandation` (poids, risque, journal des agents, frontière, Monte-Carlo), `/regime` (probabilités, historique, anomalies, macro), `/backtest`, `/modeles` (tableaux, DM, ACP, clustering, fiches), `/donnees/maj` (logique sécurisée existante réutilisée telle quelle).
- **Front Next.js** (Node 24 déjà installé) : Tailwind, shadcn/ui, Recharts, Lightweight Charts. Neuf pages : accueil, onboarding en 3 étapes, résultat du profil, tableau de bord du portefeuille, marché, backtest, laboratoire des modèles, journal des agents, mise à jour des données. Mode clair/sombre, squelettes de chargement, interface mobile, avertissement réglementaire.
- `start.ps1` lance `uvicorn` et `next dev` (ou `next start` après build). Pas de Docker.
- L'application Streamlit reste en repli tant que la nouvelle n'est pas complète.

---

## 5. Planning jour par jour

| Jour | Phase | Contenu | Point d'étape |
|---|---|---|---|
| **Dim 22/11** | 0 | Validation du plan, installation de Git, dépôt + branche `refonte-v2`, sauvegarde de `data/`, `models/`, `results/`, installation des dépendances manquantes (catboost, imbalanced-learn, gymnasium, stable-baselines3, torch CPU, fastapi, statsmodels) | Plan validé |
| **Lun 23/11** | 1a | `temporal.py` (3 schémas d'évaluation, métriques, DM), cible régression, baselines + OLS/Ridge/Lasso/Elastic Net + chemin de λ | — |
| **Mar 24/11** | 1b | KNN, SVR, arbres, boosting, MLP, ensemble ; ACP (KMO, Bartlett, cercle) ; courbes d'apprentissage ; tableau DM ; classification secondaire + ranker | Point : chiffres de l'agent Prévision |
| **Mer 25/11** | 1c | Régime (4 familles + indices + γ_ik → DM), anomalies, Profil (SMOTE/ADASYN, nouveaux modèles, SHAP local), allocation (RP, HRP, poche monétaire), backtest | **Fin phase 1** |
| **Jeu 26/11** | 2 | API FastAPI ; front : charte graphique, accueil, onboarding, profil, portefeuille | — |
| **Ven 27/11** | 2 + 3 | Front : marché, backtest, laboratoire, journal, données ; `start.ps1`. En parallèle, entraînement PPO en arrière-plan | **Fin phase 2** |
| **Sam 28/11** | 3 + 4 | Évaluation RL ; notebook restructuré, Restart & Run All complet ; `resultats_synthese.md` ; `fiches_modeles.md` ; README | **Fin phase 3** |
| **Dim 29/11** | 4 | Rapport Word, PowerPoint, script vidéo, captures de l'application | Livrables relus |
| **Lun 30/11** | — | Marge, corrections, **enregistrement de la vidéo (par vous)**, remise | Remise |

---

## 6. Risques et solutions de repli

| Risque | Probabilité | Repli |
|---|---|---|
| Git absent de la machine | Certain | Installation de Git for Windows (`winget install Git.Git`) ; sinon dépôt GitHub créé depuis le navigateur |
| Échéance de 8 jours trop courte pour tout faire | Élevée | Priorités : phase 1 > phase 2 > RL > livrables. Périmètre déjà réduit (section 3.8) |
| Le front Next.js prend du retard | Moyenne | Pages prioritaires : onboarding, profil, portefeuille, laboratoire. Streamlit gardé en repli |
| PPO ne converge pas ou fait moins bien | Élevée | Présenté comme résultat négatif documenté ; l'optimiseur classique reste dans l'application |
| Aucun modèle ne bat la baseline (DM) | Élevée | Résultat assumé et argumenté (efficience des marchés, IC réalistes) ; la valeur du système vient du contrôle du risque et de l'adéquation au profil |
| Temps de calcul (12 modèles × 3 schémas × grilles) | Moyenne | Grilles resserrées, `QUICK_RUN`, résultats mis en cache dans `results/` |
| Période de test déjà « vue » lors des versions précédentes | Certain | Tous les choix sont figés sur la validation et documentés ; limite écrite noir sur blanc dans le rapport |
| Yahoo/FRED indisponibles | Faible | Cache local + sauvegarde/restauration existantes |
| Encodage Windows (`metadata.json` en cp1252) | Certain | Tous les fichiers écrits en UTF-8 explicite |

---

## 7. Décisions validées (22/11/2026)

1. **Git** : installé par Claude via `winget`.
2. **Périmètre** : LSTM, LLM/RAG, Docker et semi-supervisé retirés (section 3.8). Priorité à l'explicabilité des choix de modèles.
3. **Poche monétaire du profil Prudent** : approuvée, avec la mention explicite que la période de test a déjà été vue.
4. **Notebook** : un seul notebook principal. Chaque cellule de code est précédée d'une explication Markdown cohérente avec ce que fait le code (modèle, formule, raison du choix). Le calcul lourd est dans `src/`, les résultats sont mis en cache.
5. Les supports de cours ne font pas partie du livrable.
