# Synthèse des résultats

Données jusqu'au 2026-11-20 – mode rapide : True

- Agent Profil : **Régression logistique** (seuil 0.53)
- Agent Prévision : **Lasso** – non activé (garde-fou, α = 0) (IC hors pli = 0.0974)
- Réglages : réduction du risque « aucune », γ = 2.0


## Agent Profil – comparaison des modèles (validation)

|                               |   AUC CV (train) |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |   Durée (s) |
|:------------------------------|-----------------:|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|------------:|
| Régression logistique         |         0.837676 |  0.85858  |   0.765258 |        0.790772 |            0.618257 |         0.816438 |   0.781422 |    8.25858  |
| SVM (noyau RBF)               |         0.837245 |  0.855281 |   0.773964 |        0.768129 |            0.716418 |         0.657534 |   0.808418 |   11.6152   |
| XGBoost                       |         0.837908 |  0.852586 |   0.74527  |        0.78418  |            0.578624 |         0.860274 |   0.756459 |    7.28084  |
| Forêt aléatoire               |         0.837262 |  0.850132 |   0.763969 |        0.789952 |            0.615967 |         0.817352 |   0.779971 |   10.4092   |
| CatBoost                      |         0.838845 |  0.849723 |   0.745617 |        0.78006  |            0.583492 |         0.839269 |   0.758491 |   12.6778   |
| KNN (+SMOTE)                  |         0.832745 |  0.849348 |   0.752221 |        0.783366 |            0.594895 |         0.830137 |   0.766328 |    6.10136  |
| LightGBM                      |         0.837078 |  0.84919  |   0.744612 |        0.780978 |            0.580363 |         0.847489 |   0.756749 |    4.57577  |
| Arbre de décision             |         0.790808 |  0.800166 |   0.716186 |        0.747686 |            0.553083 |         0.794521 |   0.730624 |    2.77031  |
| MLP (Adam)                    |         0.78018  |  0.787034 |   0.701998 |        0.70634  |            0.579304 |         0.623744 |   0.73643  |   17.201    |
| Baseline – classe majoritaire |         0.5      |  0.5      |   0.405522 |        0.5      |            0        |         0        |   0.682148 |    0.641258 |


## Agent Profil – corrections du déséquilibre (validation)

|                                           |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Régression logistique / aucune correction |    0.8582 |     0.7645 |          0.75   |              0.7583 |           0.5872 |     0.8093 |
| Régression logistique / poids de classes  |    0.8586 |     0.7653 |          0.7908 |              0.6183 |           0.8164 |     0.7814 |
| Régression logistique / SMOTE             |    0.8583 |     0.7655 |          0.793  |              0.6156 |           0.8265 |     0.7808 |
| Régression logistique / ADASYN            |    0.8552 |     0.7641 |          0.7932 |              0.6117 |           0.8329 |     0.7788 |
| LightGBM / aucune correction              |    0.8498 |     0.7478 |          0.7375 |              0.7077 |           0.5881 |     0.7919 |
| LightGBM / poids de classes               |    0.8492 |     0.7446 |          0.781  |              0.5804 |           0.8475 |     0.7567 |
| LightGBM / SMOTE                          |    0.8516 |     0.7501 |          0.7827 |              0.5908 |           0.8347 |     0.7637 |
| LightGBM / ADASYN                         |    0.8513 |     0.7409 |          0.7775 |              0.576  |           0.8447 |     0.753  |


## Agent Profil – modèle retenu (test)

|                                                 |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:------------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Régression logistique (seuil 0,5)               |    0.8688 |     0.7674 |          0.7942 |              0.6184 |           0.8243 |     0.7832 |
| Régression logistique (seuil 0.53)              |    0.8688 |     0.7695 |          0.7895 |              0.6326 |           0.7923 |     0.7884 |
| Régression logistique (seuil 0.53, pondéré WGT) |    0.8245 |     0.7151 |          0.7372 |              0.5831 |           0.7926 |     0.7216 |
| Baseline – classe majoritaire                   |    0.5    |     0.4059 |          0.5    |              0      |           0      |     0.6832 |


## Agent Profil – validation externe (SCF 2019)

|                                             |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:--------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Test interne – SCF 2022                     |    0.8688 |     0.7695 |          0.7895 |              0.6326 |           0.7923 |     0.7884 |
| Validation externe – SCF 2019               |    0.8433 |     0.7447 |          0.7588 |              0.612  |           0.7325 |     0.7685 |
| Validation externe – SCF 2019 (pondéré WGT) |    0.8123 |     0.7195 |          0.7305 |              0.6214 |           0.76   |     0.7241 |


## Agent Prévision – comparaison des modèles (validation 2020-2021)

|                              | famille             |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   IC t-stat (Newey-West) |   % dates IC>0 |   Spread Q5-Q1 net (annualisé) |   Spread Q5-Q1 brut (annualisé) |
|:-----------------------------|:--------------------|-------:|-------:|-------:|----------------------:|-----------:|-------------------------:|---------------:|-------------------------------:|--------------------------------:|
| Baseline – moyenne par actif | Baseline            | 0.0048 | 0.0691 | 0.0406 |                0.0844 |     0.1964 |                   3.2977 |         0.7188 |                         0.2913 |                          0.2923 |
| Arbre de décision            | Arbre               | 0.0048 | 0.0691 | 0.0407 |                0.0843 |     0.1598 |                   3.3366 |         0.7083 |                         0.0625 |                          0.068  |
| Gradient Boosting            | Ensemble (boosting) | 0.005  | 0.0708 | 0.0416 |                0.039  |     0.1399 |                   2.9148 |         0.6562 |                         0.2355 |                          0.2557 |
| Forêt aléatoire              | Ensemble (bagging)  | 0.005  | 0.071  | 0.0418 |                0.0351 |     0.1672 |                   2.5493 |         0.6771 |                         0.2891 |                          0.303  |
| LightGBM                     | Ensemble (boosting) | 0.005  | 0.071  | 0.0422 |                0.0343 |     0.1358 |                   2.5318 |         0.6875 |                         0.2199 |                          0.2374 |
| XGBoost                      | Ensemble (boosting) | 0.0051 | 0.0711 | 0.0413 |                0.0308 |     0.1102 |                   1.8682 |         0.6146 |                         0.133  |                          0.1508 |
| Ridge                        | Linéaire régularisé | 0.0051 | 0.0716 | 0.0418 |                0.0174 |     0.0345 |                   0.6009 |         0.5625 |                         0.135  |                          0.1568 |
| Lasso                        | Linéaire régularisé | 0.0051 | 0.0717 | 0.0419 |                0.0159 |     0.0221 |                   0.3755 |         0.5208 |                         0.1529 |                          0.1752 |
| Elastic Net                  | Linéaire régularisé | 0.0051 | 0.0717 | 0.0419 |                0.0144 |     0.0222 |                   0.3756 |         0.5104 |                         0.1624 |                          0.1849 |
| Régression linéaire (MCO)    | Linéaire            | 0.0051 | 0.0717 | 0.0419 |                0.014  |     0.0227 |                   0.3852 |         0.5208 |                         0.1657 |                          0.1885 |
| Baseline – momentum 12-1     | Baseline            | 0.0052 | 0.0721 | 0.0413 |                0.0034 |     0.0139 |                   0.2732 |         0.4688 |                         0.2588 |                          0.2693 |
| Baseline – prévision nulle   | Baseline            | 0.0052 | 0.0722 | 0.0413 |                0      |            |                          |                |                                |                                 |
| SVR (noyau RBF)              | Noyau               | 0.0053 | 0.0729 | 0.0419 |               -0.0178 |    -0.0015 |                  -0.0267 |         0.4896 |                        -0.0946 |                         -0.0714 |
| KNN                          | Voisinage           | 0.0055 | 0.0739 | 0.0424 |               -0.0455 |     0.0426 |                   0.7185 |         0.6146 |                         0.0523 |                          0.0773 |
| CatBoost                     | Ensemble (boosting) | 0.0055 | 0.0743 | 0.042  |               -0.0576 |     0.1457 |                   2.4107 |         0.7083 |                         0.2429 |                          0.2594 |
| MLP (Adam)                   | Réseau de neurones  | 0.0124 | 0.1116 | 0.0731 |               -1.3842 |     0.0052 |                   0.1061 |         0.4688 |                        -0.0059 |                          0.0201 |


## Agent Prévision – trois méthodes d'évaluation

|                                                    |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   nb plis |   écart-type MSE |   écart-type IC |
|:---------------------------------------------------|-------:|-------:|-------:|----------------------:|-----------:|----------:|-----------------:|----------------:|
| Baseline – moyenne par actif / Hold-out temporel   | 0.0048 | 0.0691 | 0.0406 |                0.0844 |     0.1964 |         1 |                  |                 |
| Baseline – moyenne par actif / CV expanding window | 0.0034 | 0.0566 | 0.0315 |                0.0515 |     0.1699 |         3 |           0.0017 |          0.0461 |
| Baseline – moyenne par actif / CV rolling window   | 0.0034 | 0.0566 | 0.0315 |                0.0518 |     0.1803 |         3 |           0.0017 |          0.0484 |
| Arbre de décision / Hold-out temporel              | 0.0048 | 0.0691 | 0.0407 |                0.0843 |     0.1598 |         1 |                  |                 |
| Arbre de décision / CV expanding window            | 0.0037 | 0.0589 | 0.0326 |               -0.0258 |     0.068  |         3 |           0.0018 |          0.097  |
| Arbre de décision / CV rolling window              | 0.0038 | 0.0597 | 0.0338 |               -0.0508 |     0.0333 |         3 |           0.002  |          0.0698 |
| Gradient Boosting / Hold-out temporel              | 0.005  | 0.0708 | 0.0416 |                0.039  |     0.1399 |         1 |                  |                 |
| Gradient Boosting / CV expanding window            | 0.0036 | 0.0582 | 0.0322 |                0.003  |     0.1104 |         3 |           0.0018 |          0.0466 |
| Gradient Boosting / CV rolling window              | 0.0037 | 0.059  | 0.033  |               -0.0234 |     0.1266 |         3 |           0.0019 |          0.0262 |
| Forêt aléatoire / Hold-out temporel                | 0.005  | 0.071  | 0.0418 |                0.0351 |     0.1672 |         1 |                  |                 |
| Forêt aléatoire / CV expanding window              | 0.0036 | 0.0579 | 0.0317 |                0.0109 |     0.1522 |         3 |           0.0018 |          0.0142 |
| Forêt aléatoire / CV rolling window                | 0.0036 | 0.0581 | 0.032  |                0.006  |     0.1307 |         3 |           0.0018 |          0.0253 |
| Ridge / Hold-out temporel                          | 0.0051 | 0.0716 | 0.0418 |                0.0174 |     0.0345 |         1 |                  |                 |
| Ridge / CV expanding window                        | 0.0035 | 0.0579 | 0.0322 |                0.0076 |     0.0958 |         3 |           0.0018 |          0.0387 |
| Ridge / CV rolling window                          | 0.004  | 0.0614 | 0.0339 |               -0.1036 |     0.0954 |         3 |           0.0023 |          0.0314 |


## Agent Prévision – prévisions hors pli 2013-2021 et tests de Diebold-Mariano

|                                    |   MSE hors pli |   IC moyen |   IC t-stat (NW) |   DM vs prévision nulle |    p vs nulle |   DM vs moyenne par actif |   p vs moyenne par actif |
|:-----------------------------------|---------------:|-----------:|-----------------:|------------------------:|--------------:|--------------------------:|-------------------------:|
| Baseline – moyenne par actif       |         0.0034 |     0.1699 |           5.4208 |               1.2498    |   0.212091    |                 nan       |            nan           |
| Lasso                              |         0.0035 |     0.0974 |           3.5594 |               0.347454  |   0.72843     |                  -2.66826 |              0.00792915  |
| Ensemble (moyenne des 3 meilleurs) |         0.0036 |     0.0962 |           3.5191 |               0.320575  |   0.748697    |                  -2.69358 |              0.0073613   |
| Elastic Net                        |         0.0036 |     0.0961 |           3.5199 |               0.314385  |   0.75339     |                  -2.68949 |              0.0074504   |
| Régression linéaire (MCO)          |         0.0036 |     0.0958 |           3.5159 |               0.298994  |   0.765098    |                  -2.71977 |              0.00681254  |
| Ridge                              |         0.0036 |     0.0958 |           3.4701 |               0.351945  |   0.725061    |                  -2.62165 |              0.00907851  |
| Forêt aléatoire                    |         0.0036 |     0.1522 |           4.4172 |               0.944514  |   0.345467    |                  -1.21317 |              0.225769    |
| LightGBM                           |         0.0036 |     0.1196 |           3.9525 |               0.440703  |   0.659662    |                  -1.34553 |              0.179204    |
| Gradient Boosting                  |         0.0036 |     0.1104 |           3.775  |               0.104894  |   0.916511    |                  -1.60407 |              0.109473    |
| Baseline – momentum 12-1           |         0.0036 |     0.0469 |           1.4112 |               0.615347  |   0.538669    |                  -1.25962 |              0.208529    |
| XGBoost                            |         0.0036 |     0.1194 |           3.8659 |               0.0306436 |   0.975569    |                  -1.69545 |              0.0907543   |
| Baseline – prévision nulle         |         0.0036 |            |                  |             nan         | nan           |                  -1.2498  |              0.212091    |
| Arbre de décision                  |         0.0037 |     0.068  |           2.2563 |              -0.793062  |   0.428204    |                  -2.09617 |              0.0366845   |
| SVR (noyau RBF)                    |         0.0037 |     0.087  |           2.7348 |              -1.86171   |   0.0633649   |                  -1.68157 |              0.0934192   |
| KNN                                |         0.0038 |     0.0888 |           2.9853 |              -0.992133  |   0.321722    |                  -2.27869 |              0.023203    |
| CatBoost                           |         0.0039 |     0.127  |           4.1097 |              -1.16068   |   0.246452    |                  -2.3854  |              0.0175172   |
| MLP (Adam)                         |         0.01   |     0.0403 |           1.5732 |              -8.00048   |   1.30322e-14 |                  -8.38256 |              8.53867e-16 |


## Agent Prévision – test (2022 → fin)

|                              |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   IC t-stat (Newey-West) |   % dates IC>0 |   Spread Q5-Q1 net (annualisé) |   Spread Q5-Q1 brut (annualisé) |   DM vs prévision nulle |   p vs prévision nulle |   DM vs moyenne par actif |   p vs moyenne par actif |
|:-----------------------------|-------:|-------:|-------:|----------------------:|-----------:|-------------------------:|---------------:|-------------------------------:|--------------------------------:|------------------------:|-----------------------:|--------------------------:|-------------------------:|
| Lasso                        | 0.0028 | 0.053  | 0.0349 |               -0.1407 |     0.0147 |                   0.3929 |         0.5353 |                        -0.1009 |                         -0.0858 |               -2.39862  |               0.017221 |                  -2.96805 |                0.0033003 |
| Baseline – prévision nulle   | 0.0025 | 0.0497 | 0.0332 |                0      |            |                          |                |                                |                                 |              nan        |             nan        |                   1.5991  |                0.111113  |
| Baseline – moyenne par actif | 0.0027 | 0.052  | 0.034  |               -0.0978 |     0.0485 |                   1.0266 |         0.5643 |                        -0.0172 |                         -0.0168 |               -1.5991   |               0.111113 |                 nan       |              nan         |
| Baseline – momentum 12-1     | 0.0025 | 0.0496 | 0.0331 |                0.0038 |     0.0537 |                   1.644  |         0.5975 |                         0.0935 |                          0.1054 |                0.704872 |               0.481574 |                   1.67756 |                0.0947356 |


## Agent Régime – comparaison des 4 familles de clustering

|            |   nb clusters |   % bruit |   silhouette |   Davies-Bouldin |   Calinski-Harabasz |   inertie intra J_w |   inertie inter J_b |   % inertie expliquée |   accord avec le GMM (indice de Rand ajusté) |
|:-----------|--------------:|----------:|-------------:|-----------------:|--------------------:|--------------------:|--------------------:|----------------------:|---------------------------------------------:|
| K-means    |             3 |    0      |       0.2744 |           1.2872 |           1451.39   |             5395.89 |            6559.11  |               54.865  |                                       0.2653 |
| CAH (Ward) |             3 |    0      |       0.3584 |           1.1827 |           1164.57   |             6052.07 |            5902.93  |               49.3762 |                                       0.2471 |
| DBSCAN     |             3 |    3.9314 |       0.481  |           0.6327 |             98.9408 |             8627.63 |             744.224 |                7.9411 |                                       0.0079 |
| GMM (EM)   |             3 |    0      |       0.1198 |           1.7925 |            794.87   |             7177.07 |            4777.93  |               39.9659 |                                       1      |


## Agent Régime – validation économique

|                                   |   fréquence |   rdt moyen 21 j futur |   vol. future moyenne |   eta2 |
|:----------------------------------|------------:|-----------------------:|----------------------:|-------:|
| Entraînement 2011-2019 / Calme    |      0.3258 |                 0.0069 |                0.0967 | 0.2107 |
| Entraînement 2011-2019 / Neutre   |      0.4028 |                 0.0099 |                0.1226 | 0.2107 |
| Entraînement 2011-2019 / Stress   |      0.2714 |                 0.0214 |                0.174  | 0.2107 |
| Validation 2020-2021 / Calme      |      0.1347 |                -0.0323 |                0.2021 | 0.1198 |
| Validation 2020-2021 / Neutre     |      0.5188 |                 0.0149 |                0.1452 | 0.1198 |
| Validation 2020-2021 / Stress     |      0.3465 |                 0.0407 |                0.2693 | 0.1198 |
| Test 2022 → (descriptif) / Calme  |      0.2139 |                 0.0072 |                0.1148 | 0.2917 |
| Test 2022 → (descriptif) / Neutre |      0.3806 |                 0.0111 |                0.1277 | 0.2917 |
| Test 2022 → (descriptif) / Stress |      0.4055 |                 0.0124 |                0.2061 | 0.2917 |


## Apport des probabilités du GMM comme variables (Diebold-Mariano)

|                 |   MSE val. sans γ |   MSE val. avec γ |   DM validation |   p validation |   IC val. sans γ |   IC val. avec γ |   DM hors pli |   p hors pli |
|:----------------|------------------:|------------------:|----------------:|---------------:|-----------------:|-----------------:|--------------:|-------------:|
| Ridge           |            0.0051 |            0.0051 |         -0.6503 |         0.5171 |           0.0345 |           0.0339 |        1.4984 |       0.1348 |
| Forêt aléatoire |            0.005  |            0.005  |          0.0389 |         0.969  |           0.1672 |           0.1689 |       -0.3617 |       0.7178 |


## Réglages de l'allocation choisis sur la validation

|                |   Sharpe |   Max drawdown |   Rotation annuelle |
|:---------------|---------:|---------------:|--------------------:|
| aucune / 2.0   |   0.3663 |        -0.2366 |              1.429  |
| aucune / 10.0  |   0.3107 |        -0.2366 |              1.2325 |
| modérée / 2.0  |   0.2615 |        -0.2321 |              1.6353 |
| modérée / 10.0 |   0.2138 |        -0.2355 |              1.4728 |


## Backtest – performances sur le test

|                                         |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Sortino |   Max drawdown |   Calmar |   VaR 95% (jour) |   % jours positifs |   Rotation annuelle |
|:----------------------------------------|----------------------:|------------------------:|---------:|----------:|---------------:|---------:|-----------------:|-------------------:|--------------------:|
| Multi-agent – Prudent                   |                0.02   |                  0.0615 |  -0.2948 |   -0.4077 |        -0.1536 |   0.13   |           0.0061 |             0.5355 |              0.8332 |
| Multi-agent – Équilibré                 |                0.0465 |                  0.0953 |   0.1074 |    0.1464 |        -0.1839 |   0.2529 |           0.0095 |             0.5452 |              0.6809 |
| Multi-agent – Dynamique                 |                0.0778 |                  0.1324 |   0.3319 |    0.4499 |        -0.1994 |   0.3904 |           0.0132 |             0.5371 |              0.6341 |
| Équipondérée 1/N                        |                0.0709 |                  0.1167 |   0.3047 |    0.4176 |        -0.1957 |   0.3624 |           0.011  |             0.542  |              0.5991 |
| 60/40                                   |                0.0511 |                  0.1115 |   0.146  |    0.2086 |        -0.2095 |   0.244  |           0.011  |             0.5216 |              0.3883 |
| Variance minimale                       |                0.0132 |                  0.0486 |  -0.5252 |   -0.7453 |        -0.1367 |   0.0963 |           0.005  |             0.5338 |              1.1091 |
| Markowitz historique                    |                0.1695 |                  0.195  |   0.6978 |    0.8806 |        -0.2316 |   0.7318 |           0.0188 |             0.546  |              6.1569 |
| Référence stratégique – Prudent         |                0.0317 |                  0.0736 |  -0.0799 |   -0.1137 |        -0.1615 |   0.1962 |           0.0076 |             0.5403 |              0.4588 |
| 1/N vol. cible – Prudent                |                0.0476 |                  0.069  |   0.1316 |    0.1776 |        -0.1293 |   0.3679 |           0.0069 |             0.5452 |              0.8085 |
| Parité de risque vol. cible – Prudent   |                0.0413 |                  0.0657 |   0.0438 |    0.0592 |        -0.1411 |   0.2929 |           0.0063 |             0.5566 |              0.9028 |
| HRP vol. cible – Prudent                |                0.029  |                  0.0453 |  -0.2247 |   -0.3303 |        -0.1109 |   0.2615 |           0.0045 |             0.5395 |              1.8963 |
| Référence stratégique – Équilibré       |                0.0538 |                  0.1047 |   0.1726 |    0.2411 |        -0.1864 |   0.2883 |           0.0103 |             0.5412 |              0.5216 |
| 1/N vol. cible – Équilibré              |                0.062  |                  0.105  |   0.2464 |    0.3286 |        -0.1798 |   0.3448 |           0.0103 |             0.5428 |              0.9523 |
| Parité de risque vol. cible – Équilibré |                0.0485 |                  0.0742 |   0.1394 |    0.1942 |        -0.1514 |   0.3204 |           0.0073 |             0.5566 |              0.7087 |
| HRP vol. cible – Équilibré              |                0.0282 |                  0.0456 |  -0.2407 |   -0.3536 |        -0.1109 |   0.2541 |           0.0046 |             0.5428 |              1.9353 |
| Référence stratégique – Dynamique       |                0.0833 |                  0.1369 |   0.3626 |    0.4963 |        -0.1994 |   0.4179 |           0.0134 |             0.5379 |              0.5818 |
| 1/N vol. cible – Dynamique              |                0.0708 |                  0.1166 |   0.3042 |    0.4169 |        -0.1957 |   0.362  |           0.011  |             0.542  |              0.6064 |
| Parité de risque vol. cible – Dynamique |                0.0486 |                  0.0744 |   0.141  |    0.1966 |        -0.1514 |   0.3214 |           0.0073 |             0.5558 |              0.6952 |
| HRP vol. cible – Dynamique              |                0.0282 |                  0.0456 |  -0.2407 |   -0.3536 |        -0.1109 |   0.2541 |           0.0046 |             0.5428 |              1.9353 |


## Backtest – comparaison à risque comparable

|                                         |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Max drawdown |   Rotation annuelle |
|:----------------------------------------|----------------------:|------------------------:|---------:|---------------:|--------------------:|
| Prudent / Multi-agent                   |                0.02   |                  0.0615 |  -0.2948 |        -0.1536 |              0.8332 |
| Prudent / Référence stratégique         |                0.0317 |                  0.0736 |  -0.0799 |        -0.1615 |              0.4588 |
| Prudent / 1/N vol. cible                |                0.0476 |                  0.069  |   0.1316 |        -0.1293 |              0.8085 |
| Prudent / Parité de risque vol. cible   |                0.0413 |                  0.0657 |   0.0438 |        -0.1411 |              0.9028 |
| Prudent / HRP vol. cible                |                0.029  |                  0.0453 |  -0.2247 |        -0.1109 |              1.8963 |
| Équilibré / Multi-agent                 |                0.0465 |                  0.0953 |   0.1074 |        -0.1839 |              0.6809 |
| Équilibré / Référence stratégique       |                0.0538 |                  0.1047 |   0.1726 |        -0.1864 |              0.5216 |
| Équilibré / 1/N vol. cible              |                0.062  |                  0.105  |   0.2464 |        -0.1798 |              0.9523 |
| Équilibré / Parité de risque vol. cible |                0.0485 |                  0.0742 |   0.1394 |        -0.1514 |              0.7087 |
| Équilibré / HRP vol. cible              |                0.0282 |                  0.0456 |  -0.2407 |        -0.1109 |              1.9353 |
| Dynamique / Multi-agent                 |                0.0778 |                  0.1324 |   0.3319 |        -0.1994 |              0.6341 |
| Dynamique / Référence stratégique       |                0.0833 |                  0.1369 |   0.3626 |        -0.1994 |              0.5818 |
| Dynamique / 1/N vol. cible              |                0.0708 |                  0.1166 |   0.3042 |        -0.1957 |              0.6064 |
| Dynamique / Parité de risque vol. cible |                0.0486 |                  0.0744 |   0.141  |        -0.1514 |              0.6952 |
| Dynamique / HRP vol. cible              |                0.0282 |                  0.0456 |  -0.2407 |        -0.1109 |              1.9353 |


## Backtest – significativité (bootstrap par blocs)

|                                                        |   diff. de Sharpe observée |   IC 95 % bas |   IC 95 % haut |   P(diff ≤ 0) | significatif (5 %)   |
|:-------------------------------------------------------|---------------------------:|--------------:|---------------:|--------------:|:---------------------|
| Prudent : Multi-agent vs Référence stratégique         |                    -0.2152 |       -0.4248 |         0.0067 |        0.96   | non                  |
| Prudent : Multi-agent vs 1/N vol. cible                |                    -0.4271 |       -0.7726 |        -0.181  |        0.9967 | oui                  |
| Prudent : Multi-agent vs Parité de risque vol. cible   |                    -0.3391 |       -0.6026 |        -0.1521 |        0.9967 | oui                  |
| Prudent : Multi-agent vs HRP vol. cible                |                    -0.0701 |       -0.2668 |         0.2243 |        0.6667 | non                  |
| Équilibré : Multi-agent vs Référence stratégique       |                    -0.0652 |       -0.1689 |         0.0639 |        0.81   | non                  |
| Équilibré : Multi-agent vs 1/N vol. cible              |                    -0.1391 |       -0.4217 |         0.0724 |        0.89   | non                  |
| Équilibré : Multi-agent vs Parité de risque vol. cible |                    -0.032  |       -0.2344 |         0.1807 |        0.6633 | non                  |
| Équilibré : Multi-agent vs HRP vol. cible              |                     0.3487 |        0.0292 |         0.7317 |        0.02   | oui                  |
| Dynamique : Multi-agent vs Référence stratégique       |                    -0.0307 |       -0.0759 |         0.0249 |        0.83   | non                  |
| Dynamique : Multi-agent vs 1/N vol. cible              |                     0.0277 |       -0.1142 |         0.1535 |        0.38   | non                  |
| Dynamique : Multi-agent vs Parité de risque vol. cible |                     0.1909 |       -0.0994 |         0.5141 |        0.0867 | non                  |
| Dynamique : Multi-agent vs HRP vol. cible              |                     0.5733 |        0.1144 |         1.1076 |        0.0067 | oui                  |


## Renforcement (PPO) – comparaison sur le test

|                                   |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Sortino |   Max drawdown |   Calmar |   VaR 95% (jour) |   % jours positifs |   Rotation annuelle |
|:----------------------------------|----------------------:|------------------------:|---------:|----------:|---------------:|---------:|-----------------:|-------------------:|--------------------:|
| PPO – Prudent                     |                0.0336 |                  0.0813 |  -0.0424 |   -0.0591 |        -0.1726 |   0.1946 |           0.0085 |             0.5387 |              1.1483 |
| Multi-agent – Prudent             |                0.02   |                  0.0615 |  -0.2948 |   -0.4077 |        -0.1536 |   0.13   |           0.0061 |             0.5355 |              0.8332 |
| Référence stratégique – Prudent   |                0.0317 |                  0.0736 |  -0.0799 |   -0.1137 |        -0.1615 |   0.1962 |           0.0076 |             0.5403 |              0.4588 |
| 1/N vol. cible – Prudent          |                0.0476 |                  0.069  |   0.1316 |    0.1776 |        -0.1293 |   0.3679 |           0.0069 |             0.5452 |              0.8085 |
| PPO – Équilibré                   |                0.0396 |                  0.0987 |   0.0394 |    0.0533 |        -0.1878 |   0.2106 |           0.01   |             0.5355 |              2.2891 |
| Multi-agent – Équilibré           |                0.0465 |                  0.0953 |   0.1074 |    0.1464 |        -0.1839 |   0.2529 |           0.0095 |             0.5452 |              0.6809 |
| Référence stratégique – Équilibré |                0.0538 |                  0.1047 |   0.1726 |    0.2411 |        -0.1864 |   0.2883 |           0.0103 |             0.5412 |              0.5216 |
| 1/N vol. cible – Équilibré        |                0.062  |                  0.105  |   0.2464 |    0.3286 |        -0.1798 |   0.3448 |           0.0103 |             0.5428 |              0.9523 |
| PPO – Dynamique                   |                0.0454 |                  0.1038 |   0.0964 |    0.1304 |        -0.1918 |   0.2366 |           0.0105 |             0.533  |              2.2724 |
| Multi-agent – Dynamique           |                0.0778 |                  0.1324 |   0.3319 |    0.4499 |        -0.1994 |   0.3904 |           0.0132 |             0.5371 |              0.6341 |
| Référence stratégique – Dynamique |                0.0833 |                  0.1369 |   0.3626 |    0.4963 |        -0.1994 |   0.4179 |           0.0134 |             0.5379 |              0.5818 |
| 1/N vol. cible – Dynamique        |                0.0708 |                  0.1166 |   0.3042 |    0.4169 |        -0.1957 |   0.362  |           0.011  |             0.542  |              0.6064 |


## Renforcement (PPO) – significativité

|                                          |   diff. de Sharpe observée |   IC 95 % bas |   IC 95 % haut |   P(diff ≤ 0) | significatif (5 %)   |
|:-----------------------------------------|---------------------------:|--------------:|---------------:|--------------:|:---------------------|
| Prudent : PPO vs Multi-agent             |                     0.2528 |       -0.07   |         0.5238 |        0.0467 | non                  |
| Prudent : PPO vs Référence stratégique   |                     0.0376 |       -0.1192 |         0.1964 |        0.34   | non                  |
| Prudent : PPO vs 1/N vol. cible          |                    -0.1742 |       -0.5416 |         0.096  |        0.91   | non                  |
| Équilibré : PPO vs Multi-agent           |                    -0.0681 |       -0.3358 |         0.1762 |        0.7467 | non                  |
| Équilibré : PPO vs Référence stratégique |                    -0.1333 |       -0.352  |         0.0689 |        0.8867 | non                  |
| Équilibré : PPO vs 1/N vol. cible        |                    -0.2072 |       -0.5654 |         0.0965 |        0.9233 | non                  |
| Dynamique : PPO vs Multi-agent           |                    -0.2357 |       -0.5529 |         0.0656 |        0.9433 | non                  |
| Dynamique : PPO vs Référence stratégique |                    -0.2665 |       -0.575  |         0.0386 |        0.9533 | non                  |
| Dynamique : PPO vs 1/N vol. cible        |                    -0.2081 |       -0.5194 |         0.0701 |        0.9267 | non                  |


## Temps de calcul par étape (s)

|                                                       |   secondes |
|:------------------------------------------------------|-----------:|
| 1. Collecte des données                               |     0.2111 |
| 2. Prétraitement et EDA                               |     5.3001 |
| 3. Agent Profil (modèles, test, validation externe)   |   100.526  |
| 3. Agent Profil (interprétabilité)                    |     5.4302 |
| 4a. Prévision : données, EDA, ACP                     |    16.0059 |
| 4b. Prévision : modèles, régularisation, ACP          |    19.5378 |
| 4c. Prévision : 3 évaluations, Diebold-Mariano, choix |    26.5849 |
| 4d. Prévision : test et interprétabilité              |     5.7597 |
| 5a. Agent Régime                                      |     6.4684 |
| 5b. γ du GMM et Agent Alerte                          |     3.0419 |
| 6. Système multi-agent                                |     4.4813 |
| 7a. Backtest : réglages et simulation                 |    86.7175 |
| 7b. Backtest : résultats et significativité           |     9.6408 |
| 8. Apprentissage par renforcement (PPO)               |    36.2611 |
| 9. Sauvegarde                                         |     2.2129 |
