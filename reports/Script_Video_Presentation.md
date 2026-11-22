# Script de la présentation vidéo (15 à 20 minutes)

Conseil d'enregistrement : enregistrez l'écran avec OBS Studio ou l'enregistreur de Teams/Zoom, micro-casque, 1080p. Alternez entre les diapositives PowerPoint, le notebook déjà exécuté (avec ses sorties) et l'application Streamlit.

| # | Durée | À l'écran | Intervenant | Ce qu'il faut dire |
|---|---|---|---|---|
| 1 | 0:00 – 1:00 | Diapo 1 | Membre 1 | Présentation du groupe et du sujet : un assistant de six agents qui recommande une allocation de portefeuille personnalisée à partir de données réelles. |
| 2 | 1:00 – 2:30 | Diapos 2-3 | Membre 1 | Contexte : limites des allocations figées, krach de 2020, choc de 2022. Problématique et six objectifs. |
| 3 | 2:30 – 4:00 | Diapo 4 | Membre 2 | Architecture : rôle de chaque agent, tableau noir partagé, boucle de renégociation entre Allocation et Risque. |
| 4 | 4:00 – 6:00 | Diapos 5-6, puis notebook sections 1-2 | Membre 2 | Les trois sources réelles et leur téléchargement automatique. Montrer le tableau de qualité des données et la carte des valeurs manquantes. Expliquer les cinq protections contre les fuites, en insistant sur la purge et le découpage par ménage du SCF. |
| 5 | 6:00 – 7:30 | Notebook section 3 | Membre 3 | Analyse exploratoire : queues épaisses, regroupement de volatilité, corrélation actions/obligations devenue positive en 2022. Ce que cela implique pour la modélisation. |
| 6 | 7:30 – 9:30 | Diapo 7 + notebook section 4 | Membre 3 | Agent Profil : tableau de comparaison des modèles, écart par rapport aux baselines, matrice de confusion, SHAP, analyse des erreurs par âge. |
| 7 | 9:30 – 12:00 | Diapo 8 + notebook section 5 | Membre 4 | Agent Prévision : construction de la cible, validation croisée purgée, comparaison des modèles avec AUC et IC, ce qu'est un IC réaliste en finance, importance des variables, AUC par classe d'actifs et par année. |
| 8 | 12:00 – 13:00 | Diapo 9 + notebook section 6 | Membre 4 | Agent Régime : critère BIC, graphique des régimes, validation hors échantillon (volatilité future par régime). Contraintes par profil. |
| 9 | 13:00 – 15:00 | Diapo 10 + notebook section 8 | Membre 1 | Backtest : protocole walk-forward, tableau des performances, courbes de valeur et de drawdown, rôle de l'ablation sans ML, intervalles de confiance bootstrap. Conclure honnêtement sur la significativité. |
| 10 | 15:00 – 17:30 | Application Streamlit (démonstration en direct) | Membre 2 | Saisir deux clients contrastés (par exemple un retraité de 68 ans et un entrepreneur de 32 ans). Montrer le profil prédit, l'allocation, le risque, le texte d'explication et le journal des agents. Corriger le profil en tant que conseiller. Montrer l'onglet Backtest. |
| 11 | 17:30 – 19:00 | Diapos 13-14 | Membre 3 | Limites, perspectives (apprentissage par renforcement, Black-Litterman, agent conversationnel) et conclusion. |
| 12 | 19:00 – 19:30 | Diapo 14 | Tous | Remerciements. |

## Points à préparer avant l'enregistrement
- [ ] Notebook entièrement exécuté (toutes les sorties visibles), `QUICK_RUN = False`.
- [ ] Tableaux « à compléter » remplis dans le rapport et la présentation à partir de `results/resultats_synthese.md`.
- [ ] Figures insérées dans les cadres en pointillés des diapositives 7, 8, 10, 11 et 12.
- [ ] Application lancée à l'avance (`streamlit run app/streamlit_app.py`) pour éviter le temps de chargement.
- [ ] Phrase de prudence à dire : « outil pédagogique, pas un conseil en investissement ».
