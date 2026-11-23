# Fiches explicatives des modèles

Pour chaque méthode : **intuition**, **formule** (notations du cours), **hyperparamètres**, **pourquoi elle a été retenue ou écartée** (chiffres du notebook, exécution complète) et **comment l'expliquer en une minute à l'oral**.

> Règle commune : toutes les transformations (imputation, standardisation, ACP, SMOTE…) sont dans un Pipeline ajusté sur l'entraînement uniquement ; la période de test n'a servi à aucun choix.

---

## 1. Agent Profil – classification binaire (Prudent / Non prudent)

### Régression logistique – modèle retenu
- **Intuition** : on combine linéairement les caractéristiques du ménage, puis la fonction sigmoïde transforme ce score en probabilité d'être prudent. Chaque coefficient se lit directement : il augmente ou diminue la probabilité.
- **Formule** : p(y = 1 | x) = σ(β₀ + Σⱼ βⱼ xⱼ), avec σ(z) = 1 / (1 + e⁻ᶻ). Fonction de coût : entropie croisée binaire (BCE) J(β) = −(1/n) Σᵢ [yᵢ ln p(xᵢ) + (1 − yᵢ) ln(1 − p(xᵢ))] + λ ‖β‖², minimisée par descente de gradient.
- **Hyperparamètres** : C = 1/λ (force de la régularisation L2), poids de classes équilibrés, seuil de décision.
- **Pourquoi retenue** : meilleure AUC de validation (0,860), devant la SVM (0,853), CatBoost (0,850) et la forêt aléatoire (0,849) ; AUC de 0,867 sur le test et de 0,843 sur l'enquête 2019 (validation externe). Règle fixée à l'avance : à moins de 0,005 d'AUC du meilleur, on préfère le modèle le plus simple et le plus explicable.
- **Seuil** : 0,53 sur P(Prudent), choisi sur la validation pour repérer au moins 80 % des clients prudents.
- **À l'oral** : « Le modèle le plus simple a gagné. La relation entre situation financière et aversion au risque est presque monotone : plus de patrimoine, plus d'études et plus de culture financière rendent moins prudent. Un modèle linéaire la capte aussi bien qu'une forêt d'arbres, et il s'explique coefficient par coefficient. »

### Les autres classifieurs comparés
- **Baseline classe majoritaire** : prédit toujours « Non prudent » (AUC 0,5) ; sert de plancher.
- **KNN** (+ SMOTE) : vote des k plus proches voisins ; AUC 0,849 ; lent en prédiction et peu interprétable.
- **SVM à noyau RBF** : frontière de marge maximale ; AUC 0,853 mais rappel des prudents de 64 % seulement au seuil par défaut, et coût de calcul élevé (entraînée sur 4 000 lignes).
- **Arbre de décision** : questions successives ; très lisible mais moins précis (AUC 0,815).
- **Forêt aléatoire** (bagging) et **LightGBM, XGBoost, CatBoost** (boosting) : AUC de 0,84 à 0,85, sans gain sur la logistique.
- **MLP** (réseau de neurones, Adam) : le moins bon (AUC 0,787) : trop flexible pour ~16 000 lignes bruitées (variance élevée).

### Classes déséquilibrées : poids de classes, SMOTE, ADASYN
- **Intuition** : un tiers seulement des ménages est prudent ; sans correction, le modèle « oublie » cette classe.
- **SMOTE** : crée des exemples synthétiques x_new = xᵢ + λ (x_voisin − xᵢ), λ ~ U(0, 1). **ADASYN** : idem, en insistant sur les zones difficiles. Appliqués **uniquement** aux plis d'entraînement (imblearn.Pipeline).
- **Résultat** : l'AUC ne bouge pas (0,857 à 0,860) car elle mesure le classement ; le rappel des prudents passe de 56 % (sans correction) à 81-83 %. Les trois corrections équivalent à déplacer le seuil : on garde les poids de classes, la plus simple.
- **À l'oral** : « Rééquilibrer ne rend pas le modèle plus intelligent, cela change le compromis entre les deux erreurs. Nous l'avons fixé par une règle métier : ne pas rater plus d'un client prudent sur cinq. »

### Questionnaire noté (tolérance déclarée)
- 7 questions notées de 0 à 3 (score sur 21) : moins de 8 = Prudent, 8 à 14 = Équilibré, 15 et plus = Dynamique, avec des réponses éliminatoires (horizon de moins de 2 ans…).
- **Décision finale** : le plus prudent entre la tolérance déclarée et la capacité estimée par le modèle ; une divergence déclenche une alerte pour le conseiller (logique MiFID II).

---

## 2. Agent Prévision – régression sur données temporelles

### Préparation : fenêtre glissante
- **Lookback L = 252 jours**, **horizon H = 21 jours** (méthode directe), **stride S = 5 jours**, **input size = 31 variables**.
- **Cible** : yᵢ,ₜ = rᵢ,ₜ→ₜ₊₂₁ − moyenne des rendements de tous les actifs à la date t (rendement excédentaire relatif).
- **Anti-fuite** : découpage chronologique (entraînement 2011-2019, validation 2020-2021, test 2022 →) avec purge de 35 jours aux frontières.

### Baselines
- **Prévision nulle** (ŷ = 0), **moyenne historique par actif**, **momentum 12-1** (« les gagnants des 12 derniers mois continuent de gagner »).
- La moyenne par actif est la meilleure sur 2013-2021 (IC 0,12, t = 3,9) : elle capte la prime « actions contre obligations » de cette période. Sur le test, elle devient nettement moins bonne que la prévision nulle (R² ≈ −10 %) : la prime s'est retournée en 2022 (dérive des données).

### Régression linéaire, Ridge, Lasso, Elastic Net
- **Intuition** : une droite (un hyperplan) entre les variables et la cible ; la régularisation « tire » les coefficients vers zéro pour éviter le surapprentissage.
- **Formules** : MSE J(β) = (1/2n) ‖y − Xβ‖² ; Ridge : J(β) + λ ‖β‖₂² ; Lasso : J(β) + λ ‖β‖₁ (annule des coefficients : sélection de variables) ; Elastic Net : J(β) + λ [α ‖β‖₁ + (1 − α) ‖β‖₂²].
- **Hyperparamètres** : λ (et α pour Elastic Net), réglés en validation croisée temporelle.
- **Résultat** : la validation croisée choisit un Lasso dont **tous les coefficients sont nuls**, c'est-à-dire la prévision nulle. Ridge fait partie de l'ensemble retenu.
- **À l'oral** : « Quand on laisse la validation croisée choisir la pénalité, le Lasso met tout à zéro : il nous dit qu'aucune variable n'apporte assez d'information pour justifier un coefficient. C'est un résultat, pas un échec du code. »

### KNN, SVR, arbre de décision
- **KNN** : moyenne des k voisins ; **SVR** : régression à marge (perte ε-insensible), noyau RBF, entraînée sur 3 000 lignes ; **arbre** : partitions successives (profondeur 2 retenue).
- **Résultat** : la SVR est **significativement moins bonne** que la prévision nulle (DM = −3,2, p = 0,002) ; le KNN n'apporte rien.

### Forêt aléatoire, Gradient Boosting, LightGBM, XGBoost, CatBoost
- **Intuition** : combiner beaucoup d'arbres. Bagging (forêt) : arbres indépendants moyennés, réduit la variance. Boosting : arbres ajoutés un à un pour corriger les erreurs résiduelles, par descente de gradient sur la MSE.
- **Hyperparamètres** : profondeur, taille minimale des feuilles, taux d'apprentissage, régularisation L2 ; la recherche choisit presque toujours les réglages les plus régularisés.
- **Résultat** : aucun ne bat la prévision nulle de façon significative (p de 0,7 à 0,96).

### MLP (réseau de neurones, optimiseur Adam)
- **Intuition** : couches de neurones qui apprennent des combinaisons non linéaires ; poids ajustés par descente de gradient avec Adam (taux d'apprentissage adapté à chaque paramètre).
- **Résultat** : ne bat pas la prévision nulle ; c'est le modèle le plus sensible au bruit.

### Ensemble – modèle retenu (mais non activé)
- **Intuition** : moyenne des prévisions des 3 meilleurs modèles hors pli (Ridge, arbre de décision, forêt aléatoire) : réduit la variance.
- **Résultat** : meilleure MSE hors pli mais **pas de différence significative** avec la prévision nulle (DM = 0,82, p = 0,41). Sur le test : R² ≈ 0, IC ≈ 0,10, DM non significatif.
- **Garde-fou** (règle fixée avant le test) : le modèle n'influence l'allocation que s'il bat la prévision nulle au test de Diebold-Mariano. Ici, **α = 0** : ses prévisions sont affichées à titre indicatif.
- **À l'oral** : « Nous avons comparé 13 modèles et 3 baselines avec un test statistique. Aucun ne prévoit mieux qu'une prévision naïve. Le système le reconnaît et n'utilise pas ces prévisions : c'est la principale garantie d'honnêteté du projet. »

### Trois méthodes d'évaluation (hold-out, expanding, rolling)
- **Hold-out** : un seul découpage ; **expanding window** : la fenêtre d'apprentissage grandit ; **rolling window** : fenêtre de taille fixe qui glisse.
- **Résultat** : tous les modèles ont un R² positif en hold-out mais négatif ou nul en validation croisée : **le découpage unique était trop optimiste**. L'expanding fait mieux que le rolling : l'historique long est utile.

### ACP (analyse en composantes principales)
- **Intuition** : remplacer des variables très corrélées par quelques axes non corrélés qui résument l'essentiel de l'information.
- **Formule** : vecteurs propres de S = XᵗX / (n − 1) (variables standardisées), part de variance λₖ / Σⱼ λⱼ.
- **Vérifications** : KMO = 0,75 (≥ 0,6) et test de Bartlett significatif : l'ACP est justifiée. Kaiser : 5 axes ; 80 % de variance : 6 ; 90 % : 9 ; coude : 2.
- **Interprétation** : F1 « Tendance » (32 %), F2 « Risque » (19 %), F3 « Retournement », F4 « Sensibilité au marché ».
- **Résultat** : ajoutée dans le pipeline des deux meilleurs modèles (Elastic Net et Ridge, 15 composantes pour 90 % de variance), elle ne change **presque rien** (R² de validation 4,0 % contre 4,1 %, IC 0,16 dans les deux cas). Les modèles régularisés gèrent déjà bien les variables corrélées ; l'ACP apporte surtout une **lecture** des variables (tendance, risque), pas de gain de précision.

### Mesures et tests statistiques
- **MSE, RMSE, MAE** (cours) ; **R² hors échantillon** = 1 − MSE(modèle) / MSE(prévision nulle).
- **IC (coefficient d'information)** : corrélation de rang entre prévision et rendement réalisé, date par date. Son t-stat est corrigé par **Newey-West** car les cibles à 21 jours se chevauchent.
- **Test de Diebold-Mariano** : dₜ = L(e₁,ₜ) − L(e₂,ₜ) ; H₀ : E[dₜ] = 0 ; DM = d̄ / √Var(d̄), avec Var(d̄) = (γ̂₀ + 2 Σₖ γ̂ₖ) / T. DM > 0 et p < 0,05 : le modèle 2 est meilleur.
- **À l'oral** : « Une MSE un peu plus faible ne prouve rien : il faut que l'écart soit plus grand que le hasard. Le test de Diebold-Mariano le vérifie, en tenant compte du chevauchement des prévisions. »

---

## 3. Agent Régime – clustering non supervisé

### Variables
Rendement et log-volatilité du S&P 500 sur 1 mois, écart au plus haut sur 1 an, log du VIX, variation du spread de crédit sur 3 mois : toutes **stationnaires** (les niveaux, comme la pente des taux, sortent de leur plage historique après 2022).

### K-means
- **Intuition** : K centres ; chaque jour est affecté au centre le plus proche, puis les centres sont recalculés, jusqu'à convergence.
- **Coût** : J = (1/n) Σᵢ ‖xᵢ − μ_c(i)‖² (inertie intra-classe). Choix de K : méthode du coude et silhouette.
- **Résultat** : silhouette 0,27, Calinski-Harabasz le plus élevé ; pas de probabilités.

### CAH (classification ascendante hiérarchique, critère de Ward)
- **Intuition** : on part d'un groupe par jour et on fusionne à chaque étape les deux groupes les plus proches ; on coupe le dendrogramme au plus grand saut.
- **Critère de Ward** : distance (nₐ n_b / (nₐ + n_b)) d²(μₐ, μ_b) = hausse de l'inertie intra-classe causée par la fusion.
- **Résultat** : meilleure silhouette (0,36) et meilleur Davies-Bouldin ; pas de probabilités.

### DBSCAN
- **Intuition** : un groupe est une zone dense ; les points isolés sont du **bruit**. Paramètres ε (rayon) et min_samples ; ε choisi au 90e centile des distances au 10e voisin.
- **Résultat** : un grand groupe et 4 % de points de bruit : peu utile pour définir des régimes, mais ces points isolés sont les jours de crise (lien avec la détection d'anomalies).

### Mélange gaussien (GMM, algorithme EM) – modèle retenu
- **Intuition** : chaque régime est une loi normale multivariée (une « ellipse ») ; chaque jour reçoit une **probabilité** d'appartenir à chaque régime.
- **Formule** : p(x) = Σₖ πₖ N(x ; μₖ, Σₖ), paramètres estimés par l'algorithme EM qui maximise la log-vraisemblance Σᵢ ln Σₖ πₖ N(xᵢ ; μₖ, Σₖ). K choisi avec le BIC et l'interprétabilité (3 régimes).
- **Pourquoi retenu** (règle fixée à l'avance) : c'est la seule méthode qui fournit des probabilités, affichées dans l'application. Ses indices géométriques sont plus faibles (silhouette 0,12), ce qui est normal pour des ellipses qui se chevauchent.
- **Validation économique** : le régime Stress est suivi de la volatilité future la plus élevée sur les trois périodes (η² de 0,12 à 0,29). Limite : en février 2020, le marché était classé Calme juste avant le krach.
- **Probabilités γ comme variables** (extraction de caractéristiques) : aucun apport significatif à la prévision.
- **À l'oral** : « Le GMM ne dit pas si le marché va monter ou baisser ; il dit si le risque est élevé. C'est utile pour gérer le risque, pas pour prévoir les rendements. »

---

## 4. Agent Alerte – détection d'anomalies
- **z-score** : |rₜ − r̄| / σ > 4 sur le rendement du S&P 500 ; **test de Grubbs** : teste si la valeur la plus extrême est aberrante sous hypothèse gaussienne, répété tant qu'il rejette.
- **Gaussienne multivariée** : densité p(x ; μ, Σ) estimée sur 2011-2019 ; anomalie si p(x) ≤ ε (ε = 2e centile).
- **Isolation Forest** : une anomalie s'isole en peu de coupures aléatoires. **LOF** : densité locale plus faible que celle des voisins. **One-Class SVM** : frontière autour des données normales.
- **Décision** : alerte si au moins 3 des 4 détecteurs sont d'accord.
- **Résultat** : les 6 crises connues (2011, 2015, 2018, 2020, 2022, 2024) sont détectées. Dérive : 0,6 % des jours en alerte sur 2011-2019, 19 % ensuite ; il faudra recalibrer les seuils.

---

## 5. Agents Risque et Allocation

### Covariance de Ledoit-Wolf (agent Risque)
- **Intuition** : avec 25 actifs et un an de données, la covariance empirique est bruitée ; on la « rétrécit » vers une matrice simple, avec un dosage optimal calculé à partir des données.
- **Mesures** : volatilité √(wᵗΣw), VaR paramétrique 95 % à 1 mois = 1,645 σ √(21/252), CVaR historique, perte maximale.

### Optimisation moyenne-variance avec a priori de Black-Litterman (agent Allocation)
- **Problème** : max_w wᵗμ − (λ/2) wᵗΣw − (γ/2) ‖w − w_préc‖², sous les contraintes du profil (poids maximal, plafond actions, part défensive obligations + monétaire, plafond obligations longues, budget de volatilité). Résolu par SLSQP.
- **Black-Litterman** : μ = π + α, avec π = λ Σ w_ref (rendements implicites de l'allocation stratégique du profil). Sans signal (α = 0), l'optimiseur retrouve exactement w_ref (propriété vérifiée dans le notebook).
- **Négociation** : si l'agent Risque détecte un dépassement, la cible de volatilité est réduite de 10 % et l'allocation recalculée (jusqu'à 3 tours).
- **Réglages choisis sur la validation 2020-2021** : γ = 0,5 et aucune réduction du risque en régime de stress (les périodes de stress ont souvent été suivies de rebonds).

### Parité de risque et HRP (références)
- **Parité de risque** : poids tels que chaque actif contribue autant au risque : RCᵢ = wᵢ (Σw)ᵢ / σ_p égales.
- **HRP** : classification hiérarchique des actifs (distance √(½(1 − ρᵢⱼ))), puis répartition récursive du risque entre groupes ; n'inverse pas la covariance.
- **Résultat du backtest 2022-2026** : le multi-agent respecte le risque de chaque profil et fait jeu égal avec les références simples pour les profils Équilibré et Dynamique ; pour le profil Prudent, il fait significativement moins bien que le 1/N et la parité de risque à volatilité cible. Significativité mesurée par **bootstrap par blocs** de 21 jours sur la différence de ratio de Sharpe.

---

## 6. Agent d'apprentissage par renforcement (PPO)
- **Composantes** : agent = gestionnaire automatisé ; environnement = marché rejoué mois par mois ; état = variables de régime, probabilités γ, rendements et volatilités récents, poids actuels ; action = allocation entre 5 classes d'actifs, projetée sur les contraintes du profil ; récompense rₜ = Σⱼ Rⱼ − (λ/2) Σⱼ Rⱼ² − coûts.
- **Objectif** : maximiser Gₜ = Σₖ γᵏ rₜ₊ₖ (γ = 0,9). **PPO** est une méthode fondée sur la politique : un réseau de neurones donne directement π(a | s ; θ) ; l'exploration vient du caractère aléatoire de cette politique pendant l'entraînement.
- **Protocole** : entraînement sur 2011-2019 (30 000 pas), choix de la graine sur 2020-2021, évaluation unique sur le test.
- **Résultat** : bon sur la validation, mais Sharpe négatif sur le test pour les trois profils, significativement moins bon que le 1/N à volatilité cible : **surapprentissage** (environ 108 mois d'entraînement seulement). Non retenu dans l'application.
- **À l'oral** : « L'agent PPO a appris une politique qui marchait sur 2020-2021 mais pas après la hausse des taux de 2022. Avec une centaine de mois d'histoire, le renforcement n'a pas assez d'expérience pour généraliser. »

---

## 7. Interprétabilité
- **Intrinsèque** : coefficients de la régression logistique (Profil) et de Ridge (Prévision).
- **Importance par permutation** : baisse de performance quand on mélange une variable ; agnostique au modèle.
- **SHAP** (valeurs de Shapley) : prédictionᵢ = valeur de base + Σⱼ contributionⱼ. **Global** : moyenne des |contributions| (Profil : total des actifs, âge, durée des études, possession d'une entreprise, culture financière). **Local** : explication d'un client précis, affichée dans l'application (« Pourquoi ce profil ? »).
- **Analyse des erreurs** : le rappel des prudents est plus faible chez les moins de 35 ans ; les plus grosses erreurs de prévision viennent du Bitcoin.
