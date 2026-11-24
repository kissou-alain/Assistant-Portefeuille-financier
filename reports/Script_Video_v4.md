# Script de la vidéo – Assistant multi-agent de portefeuille (version 4)

**Durée visée : 18 minutes** (consigne : 15 à 20 minutes, avec démonstration).
**Support :** `reports/Presentation_Projet_MultiAgent_Portefeuille_v4.pptx`, enregistrée dans PowerPoint avec Caméo (caméra intégrée à la diapositive).
**Fil conducteur :** une cliente de 26 ans qui se dit « dynamique ». On l'annonce au début, on la retrouve dans la démonstration, et on conclut sur elle. C'est ce qui rend la présentation mémorable.
**Message à faire passer :** *un système d'apprentissage n'a de valeur que si l'on peut prouver ce qu'il apporte. Nous l'avons prouvé là où c'était vrai, et nous avons renoncé là où ce ne l'était pas.*

## Répartition proposée

| Partie | Diapositives | Intervenant | Durée |
|---|---|---|---|
| 1. Ouverture, problème, architecture | 1 à 4 | **Alain KISSOU** | 3 min 15 |
| 2. Données, protocole, agent Profil | 5 à 10 | **Mory Désiré OUATTARA** | 4 min 00 |
| 3. Prévision, Régime, Alerte | 11 à 13 | **Mahamoudou OUEDRAOGO** | 3 min 15 |
| 4. Allocation, renforcement | 14 à 16 | **Mouhamadou Bachir NGOM** | 2 min 15 |
| 5. Application et démonstration | 17 + démo | **Mouhamadou Bachir NGOM** | 3 min 30 |
| 6. Valeur, limites, conclusion | 18 à 20 | **Alain KISSOU** | 1 min 45 |
| **Total** | | | **≈ 18 min** |

> La répartition est une proposition : échangez les noms si vous le souhaitez, mais gardez une seule voix par partie.
> Rythme de lecture : environ 130 mots par minute. Le texte est écrit pour être **dit**, pas lu mot à mot. Chacun peut le reformuler avec ses mots, **mais les chiffres doivent rester exactement ceux-ci** : ce sont ceux des diapositives et du rapport.

Conventions : **[ACTION]** = ce qu'on fait à l'écran ; *en italique* = intention ou ton ; le reste = texte à dire.

---

## Partie 1 – Alain KISSOU (diapositives 1 à 4, 3 min 15)

### Diapositive 1 – Couverture (0:00 → 0:40)

*Ton posé, regard caméra. On commence par une histoire, pas par le titre.*

Imaginez une cliente de 26 ans qui entre chez son conseiller. Au questionnaire, elle répond qu'elle vise la croissance maximale, qu'elle investirait davantage si son portefeuille perdait 20 %. Elle se déclare donc « dynamique ». Mais elle gagne peu, elle n'a pas d'épargne, et elle a plus de dettes que d'actifs. Faut-il lui proposer un portefeuille risqué ?

C'est exactement le genre de décision que notre projet veut éclairer. Bonjour, nous sommes Alain Kissou, Mory Désiré Ouattara, Mahamoudou Ouedraogo et Mouhamadou Bachir Ngom. Nous vous présentons notre assistant multi-agent pour la sélection intelligente de portefeuilles financiers, réalisé sous l'encadrement du Professeur Lamrani Alaoui.

### Diapositive 2 – Programme (0:40 → 1:00)

Notre présentation suit cinq temps : le contexte et notre démarche ; les données et le protocole qui garantit des résultats honnêtes ; les agents d'apprentissage ; l'allocation et l'application, avec une démonstration en direct ; et enfin ce que le système apporte vraiment, ses limites, et ce que nous en avons appris.

### Diapositive 3 – Le problème (1:00 → 2:00)

Bien conseiller un investisseur, c'est faire trois choses. **Connaître le client** : pas seulement ce qu'il déclare, mais sa capacité réelle à supporter une perte. **Lire le marché** : un même portefeuille n'a pas le même risque en période calme et en pleine crise. Et **justifier** : un conseiller doit pouvoir expliquer chaque recommandation.

*Insister sur la deuxième moitié de la diapositive.*

Mais la littérature nous met en garde contre trois pièges. Prévoir les rendements est extrêmement difficile : Welch et Goyal l'ont montré en 2008. La règle la plus naïve, répartir à parts égales, le fameux 1/N, est très dure à battre. Et les fuites de données gonflent artificiellement les résultats. Notre objectif n'était donc pas seulement de construire un système, mais de **mesurer honnêtement ce qu'il apporte**.

### Diapositive 4 – Architecture (2:00 → 3:15)

Notre réponse : découper le problème en **sept agents spécialisés**, chacun traitant une seule question avec la méthode adaptée.

*Suivre le schéma de gauche à droite.*

L'agent **Profil** estime la capacité du client à prendre des risques, avec une régression logistique. L'agent **Régime** lit l'état du marché avec un mélange gaussien. L'agent **Alerte** détecte les journées anormales par un vote de quatre détecteurs. L'agent **Prévision** tente de prévoir les rendements. L'agent **Risque** estime la covariance et la VaR.

Ils ne s'appellent pas directement : ils publient tous sur un **tableau noir partagé**. L'agent **Allocation** construit le portefeuille, l'agent Risque le contrôle, et s'il dépasse une limite, il le refuse : c'est la flèche rouge, une renégociation en trois tours au plus. Enfin, l'agent **Explication** rédige la recommandation.

Le grand avantage de cette architecture : **chaque agent est évalué séparément, et peut être désactivé sans toucher aux autres**. Retenez ce point, il va devenir décisif. Mory va maintenant vous montrer sur quelles données tout cela repose.

---

## Partie 2 – Mory Désiré OUATTARA (diapositives 5 à 10, 4 min 00)

### Diapositive 5 – Intercalaire « Données et protocole » (3:15 → 3:25)

Merci Alain. Avant tout modèle, il faut des données fiables et un protocole qui interdit de tricher, même involontairement.

### Diapositive 6 – Les données (3:25 → 4:20)

Toutes nos données sont **réelles et ouvertes**, comme le demandent les consignes. **25 actifs cotés** : actions, obligations, monétaire, immobilier, matières premières et Bitcoin, de 2010 à aujourd'hui, depuis Yahoo Finance. **9 séries macroéconomiques** de la Réserve fédérale, la FRED : taux, VIX, crédit, inflation, chômage. Et **4 595 ménages réels** de l'enquête américaine sur les finances des ménages, le *Survey of Consumer Finances* 2022, plus la vague 2019 que nous gardons pour une validation externe.

L'analyse exploratoire nous a dicté trois choix. Les rendements ont des **queues épaisses** : nous mesurons donc le risque aussi par la VaR et la CVaR. La **diversification est instable** : en 2022, actions et obligations ont baissé ensemble, donc nous réestimons la covariance chaque mois. Et surtout, la **volatilité est prévisible, la direction des prix ne l'est pas**. Dès cette étape, nous savions que la prévision des rendements serait le point le plus difficile.

### Diapositive 7 – Six règles contre les fuites (4:20 → 5:20)

Voici notre protocole. Six règles, pour qu'aucune information du futur ne se glisse dans l'apprentissage.

*Pointer la frise en bas.*

Un : on **sépare avant de transformer**. Découpage chronologique pour les marchés : entraînement jusqu'en 2019, validation 2020-2021, test de 2022 à novembre 2026. Pour l'enquête, découpage **par ménage**, car chaque ménage y apparaît cinq fois. Deux : une **validation croisée temporelle**, toujours du passé vers le futur, avec une **purge de 35 jours** aux frontières, parce que nos cibles portent sur 21 jours. Trois : chaque série macro est **décalée à sa vraie date de publication**. Quatre : imputation, ACP et SMOTE sont dans des **pipelines** ajustés sur l'entraînement seul. Cinq : tous les réglages sont choisis sur la validation ; **le test ne sert qu'une seule fois**. Six : un **backtest walk-forward** qui, chaque mois, n'utilise que les données disponibles à cette date.

Et nous affichons notre seule limite : cette période de test avait été consultée lors des versions précédentes du projet.

### Diapositive 8 – Intercalaire « Les agents d'apprentissage » (5:20 → 5:30)

Passons aux agents qui apprennent, en commençant par celui qui connaît le client.

### Diapositive 9 – Agent Profil (5:30 → 6:25)

L'agent Profil répond à une question : ce client ressemble-t-il aux ménages prudents ? Nous avons comparé **neuf modèles du cours à une référence**, la classe majoritaire.

*Montrer le graphique.*

Résultat frappant : la **régression logistique**, le modèle le plus simple, obtient la meilleure AUC en validation, 0,860. Les forêts, le boosting, la SVM sont à quelques millièmes. Le réseau de neurones est le moins bon : il surapprend. Cela nous apprend que la relation entre situation financière et prudence est essentiellement monotone.

*Montrer les trois chiffres à droite.*

Sur le test, l'AUC atteint **0,867**, et le modèle repère **80 % des clients prudents**, exactement l'objectif fixé avant le test. Et sur l'enquête 2019, jamais vue, sans réentraînement : **0,843**. Le modèle a appris une relation stable dans le temps.

### Diapositive 10 – Déséquilibre, explication, conseiller (6:25 → 7:15)

Les prudents ne sont qu'un tiers des ménages. Nous avons comparé trois corrections : pondération, SMOTE et ADASYN. Leçon importante : **elles ne changent pas l'AUC, elles déplacent seulement le seuil**. Nous gardons donc la plus simple, la pondération, et nous choisissons le seuil explicitement : **0,53**, pour garantir 80 % de rappel.

*Pointer le schéma « Règle de décision ».*

Le profil final croise deux sources : le **questionnaire**, ce que le client déclare, et le **modèle**, ce que sa situation permet. On retient **le plus prudent des deux**, une alerte prévient le conseiller en cas d'incohérence, et **c'est toujours le conseiller qui valide**.

*Pointer la figure SHAP.*

Enfin, chaque décision est expliquée par les **valeurs SHAP**. Pour un modèle linéaire, elles sont **exactes**. Patrimoine, âge, études et culture financière pèsent le plus. Mahamoudou va maintenant vous parler de l'agent le plus ambitieux… et de pourquoi nous l'avons désactivé.

---

## Partie 3 – Mahamoudou OUEDRAOGO (diapositives 11 à 13, 3 min 15)

### Diapositive 11 – Agent Prévision : trois évaluations (7:15 → 8:20)

Merci Mory. L'agent Prévision cherche les actifs qui feront mieux que les autres le mois suivant. Nous avons comparé **treize modèles et trois références**.

*Montrer les barres vertes foncées.*

Sur la validation 2020-2021, tout semblait fonctionner : un R² positif pour tous les modèles. Mais nous avons appliqué les **trois méthodes d'évaluation du cours**, avec exactement les mêmes réglages.

*Montrer les barres négatives.*

En validation croisée, à fenêtre croissante ou glissante, **le R² devient négatif ou nul**, et le pouvoir de classement est divisé par quatre. Un seul découpage, tombé sur une période favorable, nous aurait fait croire à un modèle prédictif. **C'est l'enseignement méthodologique le plus important du projet.** D'ailleurs, le Lasso, à qui l'on laisse choisir ses variables, en garde **zéro sur trente-six** : il choisit de ne rien prédire. Et l'ACP, bien que justifiée, n'y change presque rien.

### Diapositive 12 – Diebold-Mariano et garde-fou (8:20 → 9:25)

Pour trancher, il faut un test statistique, pas quelques millièmes d'écart. Le **test de Diebold-Mariano** compare chaque modèle à la prévision nulle, sur 454 dates. Verdict : **aucun ne fait significativement mieux**.

*Pointer le « α = 0 ».*

Nous avions fixé la règle **avant** de regarder le test : l'agent n'influence l'allocation que s'il bat la prévision nulle. Ce n'est pas le cas. **L'agent est donc désactivé** : ses prévisions restent affichées, à titre indicatif.

Le test nous a donné raison : sur 2022-2026, notre ensemble a la même erreur que la prévision nulle, et la simple moyenne historique s'effondre à **moins 9,8 %**, parce que la prime des actions sur les obligations s'est retournée. Nous aurions pu réactiver l'agent après coup ; ce serait utiliser le test pour décider, donc une fuite. **Savoir renoncer à un modèle, preuves à l'appui, fait partie de la démarche.**

### Diapositive 13 – Régime et Alerte (9:25 → 10:30)

Les deux derniers agents travaillent **sans étiquettes** : apprentissage non supervisé.

*Montrer la figure de gauche.*

Pour l'agent Régime, nous avons comparé les quatre familles de clustering : K-means, classification hiérarchique, DBSCAN et **mélange gaussien**. Nous avons retenu ce dernier, bien qu'il ait la moins bonne silhouette, parce qu'il est le seul à donner des **probabilités** de régime. Et il est validé économiquement : le régime explique **12 à 29 %** de la volatilité du mois suivant. Mais soyons clairs : il réagit, il n'anticipe pas. Février 2020 était classé « Calme » juste avant le krach.

*Montrer la liste de droite.*

L'agent Alerte fait voter quatre détecteurs d'anomalies : il ne sonne que si trois sur quatre sont d'accord. Il retrouve **les six crises connues sur six**, de la crise de la dette en 2011 au choc de liquidité de 2024. Limite à noter : les détecteurs vieillissent et devront être recalibrés. Bachir va maintenant assembler tous ces agents.

---

## Partie 4 – Mouhamadou Bachir NGOM (diapositives 14 à 16, 2 min 15)

### Diapositive 14 – Intercalaire « Allocation et application » (10:30 → 10:40)

Merci Mahamoudou. Tous ces agents servent un seul but : construire un portefeuille. Voyons s'il tient ses promesses.

### Diapositive 15 – Backtest (10:40 → 11:40)

L'agent Allocation résout une optimisation **moyenne-variance**, avec un point de départ propre à chaque profil, l'a priori de **Black-Litterman**. Nous l'avons testé en conditions réelles : **soixante rebalancements mensuels, de janvier 2022 à novembre 2026, nets de coûts**, contre des règles simples ramenées au même risque.

*Montrer le graphique.*

Trois constats, honnêtes. Un : **le risque est tenu**. Volatilité réalisée de 6,3 %, 9,6 % et 13,3 %, conforme à chaque profil, même pendant le marché baissier de 2022. Deux : **pas de surperformance établie** : pour Équilibré et Dynamique, les écarts avec le 1/N ne sont pas significatifs au bootstrap par blocs. Trois : **le profil Prudent est notre point faible** : sa contrainte de 50 % d'actifs défensifs a coûté cher quand les obligations ont chuté en 2022. C'est exactement ce que prédisait la littérature.

### Diapositive 16 – PPO (11:40 → 12:45)

Nous avons aussi essayé une autre voie : **l'apprentissage par renforcement**. Un agent PPO apprend seul une politique d'allocation, par essais et erreurs, dans un environnement de marché que nous avons programmé.

*Pointer la carte de gauche, puis celle de droite.*

En validation, c'est prometteur : un Sharpe de **0,59 à 0,73**. Mais sur le test, il devient **négatif pour les trois profils**, il dépasse sa cible de risque, et il échange son portefeuille **de 3,6 à 6,6 fois par an**. C'est le symptôme classique du surapprentissage : avec seulement 108 mois d'entraînement, il a appris les régularités des années 2010, qui se sont retournées en 2022. Notre règle, fixée à l'avance, s'applique : **l'application garde l'optimiseur**, plus stable et plus facile à expliquer.

---

## Partie 5 – Mouhamadou Bachir NGOM (diapositive 17 + démonstration, 3 min 30)

### Diapositive 17 – L'application (12:45 → 13:15)

Tout ce travail est accessible dans **une vraie application web**, en ligne, sans installation. Trois points la distinguent. **Confidentialité** : le profil est calculé dans votre navigateur, rien n'est envoyé. **Mise à jour automatique** : chaque soir de semaine, les nouveaux cours sont téléchargés et le site est republié. Et un **test d'égalité** : sur 200 clients, le site donne exactement les mêmes réponses que notre code Python, à 3,5 × 10⁻¹⁶ près. Le mieux, c'est de vous la montrer.

### Démonstration en direct (13:15 → 16:15)

> **Préparation** : le site https://assistant-portefeuille-financier.vercel.app ouvert dans un onglet, zoom du navigateur à 110 %, aucune autre fenêtre visible. Voir la section « Enregistrer avec Caméo » pour l'insertion de la démonstration.

**[ACTION] Page d'accueil, puis cliquer sur « Diagnostic ».**
Reprenons notre cliente de 26 ans. D'abord, le questionnaire de tolérance au risque, sept questions.

**[ACTION] Répondre au questionnaire :**

| Question | Réponse à cliquer |
|---|---|
| Horizon | Plus de 10 ans |
| Objectif | Maximiser la croissance à long terme |
| Perte de 20 % | J'investis davantage |
| Baisse supportable | Plus de 20 % |
| Expérience | Comptes d'épargne et fonds prudents |
| Part du patrimoine | Moins de 10 % |
| Revenus futurs | Stables |

Elle se montre très confiante. **[ACTION] « Suivant ».**

**[ACTION] Remplir sa situation (devise : Dirham marocain, taux 10, laissé par défaut) :**

| Champ | Valeur |
|---|---|
| Âge | 26 ans |
| Enfants à charge | 1 |
| Revenu annuel du foyer | 220 000 MAD |
| Total des actifs | 60 000 MAD |
| Total des dettes | 140 000 MAD |
| Niveau d'études | Diplôme du secondaire |
| Situation professionnelle | Salarié |
| Connaissance des questions financières | 2 |
| Quiz de culture financière | 1 bonne réponse |
| Sexe | Femme |
| En couple | Non |
| Propriétaire | Non |
| A épargné l'an dernier | Non |

Notez que l'application accepte les dirhams : elle convertit en dollars, la monnaie du modèle. **[ACTION] « Suivant », puis « Voir mon profil ».**

**[ACTION] Page « Mon profil ».**
Voici le cœur du système. Le questionnaire donne **18 sur 21 : Dynamique**. Mais le modèle estime à **81 %** la probabilité que sa situation ressemble à celle des ménages prudents, bien au-dessus du seuil de 0,53. Le système retient donc **le plus prudent des deux : Prudent**, et il affiche une **alerte d'incohérence** pour le conseiller.

**[ACTION] Descendre jusqu'à « Pourquoi ce profil ? ».**
Et il explique pourquoi, variable par variable, avec les valeurs SHAP : d'un côté ce qui la rapproche des prudents, d'abord ses faibles actifs ; de l'autre ce qui l'en éloigne, comme son jeune âge.

**[ACTION] Remonter à « Validation du conseiller » (sans cliquer).**
Le conseiller garde le dernier mot : il peut confirmer, ou imposer un autre profil. Ce choix est alors signalé et enregistré.

**[ACTION] Cliquer sur « Portefeuille ».**
Le portefeuille recommandé : une majorité d'obligations et de monétaire, une poche d'actions limitée, la volatilité attendue, la VaR, et une projection en fourchettes, pas une promesse.

**[ACTION] Cliquer sur « Agents ».**
Et voici ce qui rend le système digne de confiance : **le dialogue des agents**. Seize messages, du questionnaire au rapport final. On voit le régime de marché, le contrôle des risques, et la décision du garde-fou : l'agent Prévision n'est pas utilisé. Chaque décision est tracée.

**[ACTION] Cliquer rapidement sur « Marché » (5 secondes), puis sur « Laboratoire » (5 secondes).**
Enfin, l'état du marché en temps réel, avec le régime actuel *[lire le régime affiché à l'écran]*, et le laboratoire, où tous les résultats que nous venons de présenter sont consultables par chacun. Je rends la parole à Alain pour conclure.

---

## Partie 6 – Alain KISSOU (diapositives 18 à 20, 1 min 45)

### Diapositive 18 – Valeur ajoutée et limites (16:15 → 17:05)

Merci Bachir. Alors, que vaut ce système ? Il ne promet pas de battre le marché. **Sa valeur, c'est la qualité du conseil.** Un profil vérifié deux fois, avec une alerte au conseiller : vous venez de le voir sur notre cliente. Un risque maîtrisé, même en 2022. Une explicabilité de bout en bout. La capacité à renoncer à un modèle qui n'a pas fait ses preuves. Et un outil réel, en ligne et entièrement reproductible.

Nous en connaissons aussi les limites : un modèle de profil appris sur des ménages américains, des actifs en dollars sans prise en compte du change, un historique court, et un profil Prudent trop contraint.

### Diapositive 19 – À retenir (17:05 → 17:40)

Si vous ne deviez retenir qu'une phrase : **un système d'apprentissage n'est utile que si l'on peut prouver ce qu'il apporte.** Nous avons fixé nos règles avant le test, et nous les avons appliquées, même contre nous : deux modèles écartés, la Prévision et PPO. Ce projet nous a appris que l'évaluation compte autant que le modèle. Prochaine étape : l'adapter au marché marocain et au dirham.

### Diapositive 20 – Merci (17:40 → 18:00)

Revenons à notre cliente de 26 ans. Un système qui l'aurait crue sur parole lui aurait proposé un portefeuille risqué. Le nôtre l'a protégée, a expliqué pourquoi, et a laissé le dernier mot à son conseiller. Merci de votre attention. Le site et tout le code sont accessibles aux adresses affichées.

---

## Enregistrer avec Caméo dans PowerPoint

1. **Insérer la caméra.** Sur la diapositive 1 : *Insertion › Caméo*. Bulle ronde d'environ 4 cm, en **bas à droite, au-dessus du bandeau gris** du pied de page. Copiez-la (Ctrl+C) puis collez-la (Ctrl+V) sur chaque diapositive : elle garde la même position.
2. **Vérifier chaque diapositive.** Sur les diapositives chargées (4, 9, 11, 15), la bulle peut cacher du contenu : réduisez-la ou déplacez-la sur un espace vide. Sur les intercalaires (5, 8, 14) et sur la couverture, elle peut être plus grande.
3. **Enregistrer.** *Diaporama › Enregistrer › Depuis le début*. Chaque intervenant enregistre **ses** diapositives, sur le même ordinateur ou en passant la main : PowerPoint permet de **réenregistrer une seule diapositive** sans toucher aux autres (*Enregistrer › Depuis la diapositive actuelle*). Personne n'a besoin de tout réussir d'une traite.
4. **La démonstration.** PowerPoint n'enregistre pas le navigateur pendant un enregistrement de diaporama. Deux possibilités :
   - **recommandé** : filmer la démonstration à part avec *Insertion › Enregistrement de l'écran* (PowerPoint) ou la Xbox Game Bar de Windows (Win + G), en parlant pendant l'enregistrement, puis l'insérer comme vidéo sur une **diapositive « Démonstration » placée juste après la 17** ;
   - ou partager l'écran pendant une visioconférence enregistrée (Teams, Zoom) et monter les deux vidéos.
5. **Exporter.** *Fichier › Exporter › Créer une vidéo*, qualité **Full HD (1080p)**, en utilisant les minutages et narrations enregistrés. Vérifier que la durée est **entre 15 et 20 minutes**.

## Conseils pour être convaincants

- **Les chiffres doivent être identiques** à ceux des diapositives. Un jury repère vite une incohérence entre ce qui est dit et ce qui est affiché.
- **Assumer les résultats négatifs avec assurance**, sans s'excuser : « nous avons désactivé cet agent », pas « malheureusement, ça n'a pas marché ». C'est notre point fort méthodologique.
- **Nommer explicitement les exigences des consignes** au passage : modèle de référence, prévention des fuites, interprétabilité SHAP, données ouvertes, déploiement. Le script le fait déjà ; ne supprimez pas ces mots.
- **Une transition par passage de parole** : chaque partie se termine en annonçant la suivante et son intervenant.
- **Regarder la caméra**, pas l'écran, pendant les phrases-clés (en gras dans le script).
- **Répéter une fois en conditions réelles** et chronométrer : si vous dépassez 19 minutes, raccourcissez d'abord les diapositives 6 et 13.
- **Avant d'enregistrer la démonstration**, faites-la une fois à blanc : le régime de marché affiché peut avoir changé depuis l'écriture de ce script.
