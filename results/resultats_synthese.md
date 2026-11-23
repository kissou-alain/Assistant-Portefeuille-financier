# Synthèse des résultats

Données jusqu'au 2026-11-20 – mode rapide : False

- Agent Profil : **Régression logistique** (seuil 0.53)
- Agent Prévision : **Ensemble (moyenne des 3 meilleurs)** – non activé (garde-fou, α = 0) (IC hors pli = 0.0472)
- Réglages : réduction du risque « aucune », γ = 0.5


## Agent Profil – comparaison des modèles (validation)

|                               |   AUC CV (train) |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |   Durée (s) |
|:------------------------------|-----------------:|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|------------:|
| Régression logistique         |         0.838616 |  0.85976  |   0.766864 |        0.791286 |            0.621943 |         0.812785 |   0.783454 |   11.5849   |
| SVM (noyau RBF)               |         0.841924 |  0.852563 |   0.763333 |        0.757082 |            0.704225 |         0.639269 |   0.8      |   26.9849   |
| CatBoost                      |         0.83921  |  0.849723 |   0.745617 |        0.78006  |            0.583492 |         0.839269 |   0.758491 |   28.8958   |
| KNN (+SMOTE)                  |         0.832929 |  0.849348 |   0.752221 |        0.783366 |            0.594895 |         0.830137 |   0.766328 |   11.3599   |
| XGBoost                       |         0.836834 |  0.849157 |   0.75578  |        0.784851 |            0.601734 |         0.823744 |   0.770682 |   30.2787   |
| Forêt aléatoire               |         0.838697 |  0.848923 |   0.754131 |        0.78153  |            0.602027 |         0.813699 |   0.769811 |   39.8615   |
| LightGBM                      |         0.838964 |  0.841678 |   0.750354 |        0.769275 |            0.610626 |         0.76621  |   0.770392 |   22.9275   |
| Arbre de décision             |         0.793317 |  0.814748 |   0.724214 |        0.75175  |            0.566424 |         0.782648 |   0.740493 |   14.5838   |
| MLP (Adam)                    |         0.791229 |  0.787034 |   0.701998 |        0.70634  |            0.579304 |         0.623744 |   0.73643  |   35.7719   |
| Baseline – classe majoritaire |         0.5      |  0.5      |   0.405522 |        0.5      |            0        |         0        |   0.682148 |    0.590034 |


## Agent Profil – corrections du déséquilibre (validation)

|                                           |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Régression logistique / aucune correction |    0.8596 |     0.7539 |          0.7385 |              0.754  |           0.5626 |     0.8026 |
| Régression logistique / poids de classes  |    0.8598 |     0.7669 |          0.7913 |              0.6219 |           0.8128 |     0.7835 |
| Régression logistique / SMOTE             |    0.8592 |     0.7711 |          0.7968 |              0.6251 |           0.8237 |     0.7869 |
| Régression logistique / ADASYN            |    0.857  |     0.7619 |          0.7905 |              0.6095 |           0.8283 |     0.7768 |
| LightGBM / aucune correction              |    0.8429 |     0.755  |          0.7455 |              0.7114 |           0.6055 |     0.7965 |
| LightGBM / poids de classes               |    0.8417 |     0.7504 |          0.7693 |              0.6106 |           0.7662 |     0.7704 |
| LightGBM / SMOTE                          |    0.8441 |     0.7573 |          0.774  |              0.6236 |           0.7626 |     0.7782 |
| LightGBM / ADASYN                         |    0.8444 |     0.7525 |          0.766  |              0.624  |           0.7397 |     0.7756 |


## Agent Profil – modèle retenu (test)

|                                                 |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:------------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Régression logistique (seuil 0,5)               |    0.8667 |     0.7728 |          0.8005 |              0.6238 |           0.8344 |     0.7881 |
| Régression logistique (seuil 0.53)              |    0.8667 |     0.7761 |          0.7969 |              0.6393 |           0.8042 |     0.7942 |
| Régression logistique (seuil 0.53, pondéré WGT) |    0.8232 |     0.7219 |          0.7447 |              0.5896 |           0.8035 |     0.7282 |
| Baseline – classe majoritaire                   |    0.5    |     0.4059 |          0.5    |              0      |           0      |     0.6832 |


## Agent Profil – validation externe (SCF 2019)

|                                             |   ROC-AUC |   F1 macro |   Balanced acc. |   Précision Prudent |   Rappel Prudent |   Accuracy |
|:--------------------------------------------|----------:|-----------:|----------------:|--------------------:|-----------------:|-----------:|
| Test interne – SCF 2022                     |    0.8667 |     0.7761 |          0.7969 |              0.6393 |           0.8042 |     0.7942 |
| Validation externe – SCF 2019               |    0.8435 |     0.7458 |          0.7611 |              0.6109 |           0.7404 |     0.7687 |
| Validation externe – SCF 2019 (pondéré WGT) |    0.812  |     0.7217 |          0.7335 |              0.6218 |           0.7693 |     0.7258 |


## Agent Prévision – comparaison des modèles (validation 2020-2021)

|                              | famille             |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   IC t-stat (Newey-West) |   % dates IC>0 |   Spread Q5-Q1 net (annualisé) |   Spread Q5-Q1 brut (annualisé) |
|:-----------------------------|:--------------------|-------:|-------:|-------:|----------------------:|-----------:|-------------------------:|---------------:|-------------------------------:|--------------------------------:|
| Baseline – moyenne par actif | Baseline            | 0.0048 | 0.0691 | 0.0406 |                0.0844 |     0.1964 |                   3.2977 |         0.7188 |                         0.2913 |                          0.2923 |
| Elastic Net                  | Linéaire régularisé | 0.005  | 0.0707 | 0.0416 |                0.0412 |     0.1634 |                   2.6368 |         0.6562 |                         0.3628 |                          0.3716 |
| Ridge                        | Linéaire régularisé | 0.005  | 0.0709 | 0.0414 |                0.036  |     0.1566 |                   2.3708 |         0.6354 |                         0.3572 |                          0.3656 |
| XGBoost                      | Ensemble (boosting) | 0.005  | 0.071  | 0.0424 |                0.0338 |     0.1588 |                   2.5143 |         0.6771 |                         0.2486 |                          0.263  |
| LightGBM                     | Ensemble (boosting) | 0.0051 | 0.0712 | 0.0417 |                0.0291 |     0.1257 |                   2.1305 |         0.6875 |                         0.2634 |                          0.28   |
| Forêt aléatoire              | Ensemble (bagging)  | 0.0051 | 0.0714 | 0.0414 |                0.023  |     0.1618 |                   2.4609 |         0.6771 |                         0.3502 |                          0.3617 |
| Gradient Boosting            | Ensemble (boosting) | 0.0051 | 0.0715 | 0.0415 |                0.0219 |     0.0981 |                   1.7025 |         0.6771 |                         0.1553 |                          0.1723 |
| Arbre de décision            | Arbre               | 0.0051 | 0.0717 | 0.0421 |                0.0151 |     0.1151 |                   2.1107 |         0.6667 |                        -0.0115 |                         -0.0039 |
| Régression linéaire (MCO)    | Linéaire            | 0.0051 | 0.0717 | 0.0419 |                0.014  |     0.0227 |                   0.3852 |         0.5208 |                         0.1657 |                          0.1885 |
| KNN                          | Voisinage           | 0.0052 | 0.0719 | 0.0412 |                0.0105 |     0.0781 |                   1.25   |         0.6146 |                        -0.0001 |                          0.0213 |
| Baseline – momentum 12-1     | Baseline            | 0.0052 | 0.0721 | 0.0413 |                0.0034 |     0.0139 |                   0.2732 |         0.4688 |                         0.2588 |                          0.2693 |
| Baseline – prévision nulle   | Baseline            | 0.0052 | 0.0722 | 0.0413 |                0      |            |                          |                |                                |                                 |
| Lasso                        | Linéaire régularisé | 0.0052 | 0.0722 | 0.0413 |               -0      |            |                          |                |                                |                                 |
| SVR (noyau RBF)              | Noyau               | 0.0052 | 0.0724 | 0.0418 |               -0.004  |    -0.0048 |                  -0.0966 |         0.5    |                         0.0467 |                          0.0699 |
| CatBoost                     | Ensemble (boosting) | 0.0053 | 0.0729 | 0.0415 |               -0.0183 |     0.1291 |                   1.922  |         0.6354 |                         0.1656 |                          0.1819 |
| MLP (Adam)                   | Réseau de neurones  | 0.0058 | 0.0759 | 0.0457 |               -0.1043 |     0.2174 |                   3.9857 |         0.75   |                         0.2339 |                          0.2535 |


## Agent Prévision – trois méthodes d'évaluation

|                                                    |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   nb plis |   écart-type MSE |   écart-type IC |
|:---------------------------------------------------|-------:|-------:|-------:|----------------------:|-----------:|----------:|-----------------:|----------------:|
| Baseline – moyenne par actif / Hold-out temporel   | 0.0048 | 0.0691 | 0.0406 |                0.0844 |     0.1964 |         1 |                  |                 |
| Baseline – moyenne par actif / CV expanding window | 0.0033 | 0.055  | 0.0312 |               -0.0299 |     0.1166 |         5 |           0.0019 |          0.0872 |
| Baseline – moyenne par actif / CV rolling window   | 0.0035 | 0.0565 | 0.032  |               -0.087  |     0.0458 |         5 |           0.002  |          0.038  |
| Elastic Net / Hold-out temporel                    | 0.005  | 0.0707 | 0.0416 |                0.0412 |     0.1634 |         1 |                  |                 |
| Elastic Net / CV expanding window                  | 0.0034 | 0.0552 | 0.0313 |               -0.0165 |     0.0345 |         5 |           0.0022 |          0.1551 |
| Elastic Net / CV rolling window                    | 0.0035 | 0.056  | 0.0318 |               -0.0479 |    -0.0058 |         5 |           0.0022 |          0.1404 |
| Ridge / Hold-out temporel                          | 0.005  | 0.0709 | 0.0414 |                0.036  |     0.1566 |         1 |                  |                 |
| Ridge / CV expanding window                        | 0.0033 | 0.0549 | 0.0311 |               -0.0011 |     0.0414 |         5 |           0.0022 |          0.176  |
| Ridge / CV rolling window                          | 0.0034 | 0.0551 | 0.0311 |               -0.0058 |    -0.0149 |         5 |           0.0022 |          0.1479 |
| XGBoost / Hold-out temporel                        | 0.005  | 0.071  | 0.0424 |                0.0338 |     0.1588 |         1 |                  |                 |
| XGBoost / CV expanding window                      | 0.0034 | 0.0554 | 0.0318 |               -0.0344 |     0.0426 |         5 |           0.0021 |          0.1793 |
| XGBoost / CV rolling window                        | 0.0035 | 0.0565 | 0.0322 |               -0.0735 |    -0.0211 |         5 |           0.0022 |          0.1443 |


## Agent Prévision – prévisions hors pli 2013-2021 et tests de Diebold-Mariano

|                                    |   MSE hors pli |   IC moyen |   IC t-stat (NW) |   DM vs prévision nulle |   p vs nulle |   DM vs moyenne par actif |   p vs moyenne par actif |
|:-----------------------------------|---------------:|-----------:|-----------------:|------------------------:|-------------:|--------------------------:|-------------------------:|
| Baseline – moyenne par actif       |         0.0033 |     0.1164 |           3.8929 |                0.370743 |   0.711002   |                nan        |            nan           |
| Ensemble (moyenne des 3 meilleurs) |         0.0033 |     0.0472 |           1.3108 |                0.820731 |   0.412231   |                 -0.223792 |              0.82302     |
| Ridge                              |         0.0033 |     0.0411 |           1.1996 |                0.615329 |   0.538646   |                 -0.243302 |              0.807881    |
| Arbre de décision                  |         0.0034 |     0.0237 |           0.7815 |                0.390503 |   0.696348   |                 -0.32879  |              0.742466    |
| Forêt aléatoire                    |         0.0034 |     0.0396 |           1.1052 |                0.359873 |   0.71911    |                 -0.35108  |              0.725691    |
| Baseline – prévision nulle         |         0.0034 |            |                  |              nan        | nan          |                 -0.370743 |              0.711002    |
| LightGBM                           |         0.0034 |     0.0279 |           0.7973 |               -0.0518   |   0.958711   |                 -0.471655 |              0.6374      |
| Elastic Net                        |         0.0034 |     0.0342 |           1.0217 |               -0.107629 |   0.914338   |                 -0.502935 |              0.615254    |
| XGBoost                            |         0.0034 |     0.0423 |           1.208  |               -0.103936 |   0.917266   |                 -0.542836 |              0.58751     |
| Lasso                              |         0.0034 |    -0.0372 |          -0.5922 |               -1.63038  |   0.103716   |                 -0.432119 |              0.66586     |
| Baseline – momentum 12-1           |         0.0034 |    -0.0488 |          -1.4788 |               -1.28814  |   0.198356   |                 -0.474499 |              0.635372    |
| Gradient Boosting                  |         0.0034 |     0.0339 |           1.047  |               -0.321175 |   0.748226   |                 -0.590428 |              0.555198    |
| KNN                                |         0.0034 |     0.0039 |           0.1132 |               -0.844645 |   0.398755   |                 -0.543192 |              0.587265    |
| SVR (noyau RBF)                    |         0.0034 |    -0.0296 |          -1.0293 |               -3.1639   |   0.00166147 |                 -0.697131 |              0.486079    |
| MLP (Adam)                         |         0.0035 |     0.0353 |           1.1723 |               -1.36114  |   0.174146   |                 -1.2834   |              0.200009    |
| CatBoost                           |         0.0035 |     0.0248 |           0.7209 |               -0.870777 |   0.384337   |                 -1.30416  |              0.19284     |
| Régression linéaire (MCO)          |         0.0035 |     0.005  |           0.1597 |               -0.746331 |   0.455855   |                 -3.85436  |              0.000132802 |


## Agent Prévision – test (2022 → fin)

|                                    |    MSE |   RMSE |    MAE |   R² hors échantillon |   IC moyen |   IC t-stat (Newey-West) |   % dates IC>0 |   Spread Q5-Q1 net (annualisé) |   Spread Q5-Q1 brut (annualisé) |   DM vs prévision nulle |   p vs prévision nulle |   DM vs moyenne par actif |   p vs moyenne par actif |
|:-----------------------------------|-------:|-------:|-------:|----------------------:|-----------:|-------------------------:|---------------:|-------------------------------:|--------------------------------:|------------------------:|-----------------------:|--------------------------:|-------------------------:|
| Ensemble (moyenne des 3 meilleurs) | 0.0025 | 0.0496 | 0.0331 |                0.0015 |     0.1002 |                   2.3523 |         0.5934 |                         0.0984 |                          0.1048 |               0.0759081 |               0.939555 |                   2.08562 |                0.0380694 |
| Baseline – prévision nulle         | 0.0025 | 0.0497 | 0.0332 |                0      |            |                          |                |                                |                                 |             nan         |             nan        |                   1.5991  |                0.111113  |
| Baseline – moyenne par actif       | 0.0027 | 0.052  | 0.034  |               -0.0978 |     0.0485 |                   1.0266 |         0.5643 |                        -0.0172 |                         -0.0168 |              -1.5991    |               0.111113 |                 nan       |              nan         |
| Baseline – momentum 12-1           | 0.0025 | 0.0496 | 0.0331 |                0.0038 |     0.0537 |                   1.644  |         0.5975 |                         0.0935 |                          0.1054 |               0.704872  |               0.481574 |                   1.67756 |                0.0947356 |


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

|                   |   MSE val. sans γ |   MSE val. avec γ |   DM validation |   p validation |   IC val. sans γ |   IC val. avec γ |   DM hors pli |   p hors pli |
|:------------------|------------------:|------------------:|----------------:|---------------:|-----------------:|-----------------:|--------------:|-------------:|
| Ridge             |            0.005  |            0.005  |         -1.6538 |         0.1015 |           0.1566 |           0.1553 |       -1.4254 |       0.1547 |
| Arbre de décision |            0.0051 |            0.0051 |                 |                |           0.1151 |           0.1151 |               |              |


## Réglages de l'allocation choisis sur la validation

|                |   Sharpe |   Max drawdown |   Rotation annuelle |
|:---------------|---------:|---------------:|--------------------:|
| aucune / 0.5   |   0.3879 |        -0.2366 |              1.5676 |
| aucune / 2.0   |   0.3663 |        -0.2366 |              1.429  |
| légère / 0.5   |   0.358  |        -0.2332 |              1.6354 |
| aucune / 5.0   |   0.3353 |        -0.2366 |              1.3117 |
| légère / 2.0   |   0.3322 |        -0.2353 |              1.4985 |
| aucune / 10.0  |   0.3107 |        -0.2366 |              1.2325 |
| légère / 5.0   |   0.3017 |        -0.236  |              1.3963 |
| modérée / 0.5  |   0.2947 |        -0.2259 |              1.8163 |
| légère / 10.0  |   0.2745 |        -0.2363 |              1.3253 |
| modérée / 2.0  |   0.2615 |        -0.2321 |              1.6353 |
| modérée / 5.0  |   0.2384 |        -0.2344 |              1.5443 |
| modérée / 10.0 |   0.2138 |        -0.2355 |              1.4728 |
| forte / 0.5    |   0.2007 |        -0.2163 |              2.1154 |
| forte / 2.0    |   0.1803 |        -0.2242 |              1.8854 |
| forte / 5.0    |   0.1671 |        -0.2263 |              1.7415 |
| forte / 10.0   |   0.1523 |        -0.2265 |              1.6199 |


## Backtest – performances sur le test

|                                         |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Sortino |   Max drawdown |   Calmar |   VaR 95% (jour) |   % jours positifs |   Rotation annuelle |
|:----------------------------------------|----------------------:|------------------------:|---------:|----------:|---------------:|---------:|-----------------:|-------------------:|--------------------:|
| Multi-agent – Prudent                   |                0.0217 |                  0.0627 |  -0.2608 |   -0.3603 |        -0.1496 |   0.1449 |           0.0062 |             0.5379 |              0.9276 |
| Multi-agent – Équilibré                 |                0.0473 |                  0.0965 |   0.1152 |    0.1571 |        -0.182  |   0.2601 |           0.0093 |             0.546  |              0.7406 |
| Multi-agent – Dynamique                 |                0.081  |                  0.1335 |   0.3523 |    0.4782 |        -0.1994 |   0.4063 |           0.0132 |             0.5346 |              0.6708 |
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
| Prudent / Multi-agent                   |                0.0217 |                  0.0627 |  -0.2608 |        -0.1496 |              0.9276 |
| Prudent / Référence stratégique         |                0.0317 |                  0.0736 |  -0.0799 |        -0.1615 |              0.4588 |
| Prudent / 1/N vol. cible                |                0.0476 |                  0.069  |   0.1316 |        -0.1293 |              0.8085 |
| Prudent / Parité de risque vol. cible   |                0.0413 |                  0.0657 |   0.0438 |        -0.1411 |              0.9028 |
| Prudent / HRP vol. cible                |                0.029  |                  0.0453 |  -0.2247 |        -0.1109 |              1.8963 |
| Équilibré / Multi-agent                 |                0.0473 |                  0.0965 |   0.1152 |        -0.182  |              0.7406 |
| Équilibré / Référence stratégique       |                0.0538 |                  0.1047 |   0.1726 |        -0.1864 |              0.5216 |
| Équilibré / 1/N vol. cible              |                0.062  |                  0.105  |   0.2464 |        -0.1798 |              0.9523 |
| Équilibré / Parité de risque vol. cible |                0.0485 |                  0.0742 |   0.1394 |        -0.1514 |              0.7087 |
| Équilibré / HRP vol. cible              |                0.0282 |                  0.0456 |  -0.2407 |        -0.1109 |              1.9353 |
| Dynamique / Multi-agent                 |                0.081  |                  0.1335 |   0.3523 |        -0.1994 |              0.6708 |
| Dynamique / Référence stratégique       |                0.0833 |                  0.1369 |   0.3626 |        -0.1994 |              0.5818 |
| Dynamique / 1/N vol. cible              |                0.0708 |                  0.1166 |   0.3042 |        -0.1957 |              0.6064 |
| Dynamique / Parité de risque vol. cible |                0.0486 |                  0.0744 |   0.141  |        -0.1514 |              0.6952 |
| Dynamique / HRP vol. cible              |                0.0282 |                  0.0456 |  -0.2407 |        -0.1109 |              1.9353 |


## Backtest – significativité (bootstrap par blocs)

|                                                        |   diff. de Sharpe observée |   IC 95 % bas |   IC 95 % haut |   P(diff ≤ 0) | significatif (5 %)   |
|:-------------------------------------------------------|---------------------------:|--------------:|---------------:|--------------:|:---------------------|
| Prudent : Multi-agent vs Référence stratégique         |                    -0.1811 |       -0.3813 |         0.0616 |        0.928  | non                  |
| Prudent : Multi-agent vs 1/N vol. cible                |                    -0.393  |       -0.7069 |        -0.1275 |        0.9965 | oui                  |
| Prudent : Multi-agent vs Parité de risque vol. cible   |                    -0.305  |       -0.5488 |        -0.1172 |        0.999  | oui                  |
| Prudent : Multi-agent vs HRP vol. cible                |                    -0.036  |       -0.2352 |         0.2531 |        0.501  | non                  |
| Équilibré : Multi-agent vs Référence stratégique       |                    -0.0574 |       -0.1511 |         0.0624 |        0.789  | non                  |
| Équilibré : Multi-agent vs 1/N vol. cible              |                    -0.1313 |       -0.3712 |         0.0535 |        0.924  | non                  |
| Équilibré : Multi-agent vs Parité de risque vol. cible |                    -0.0242 |       -0.2252 |         0.2031 |        0.565  | non                  |
| Équilibré : Multi-agent vs HRP vol. cible              |                     0.3565 |        0.0367 |         0.7901 |        0.016  | oui                  |
| Dynamique : Multi-agent vs Référence stratégique       |                    -0.0103 |       -0.051  |         0.0423 |        0.59   | non                  |
| Dynamique : Multi-agent vs 1/N vol. cible              |                     0.0481 |       -0.0981 |         0.1906 |        0.2655 | non                  |
| Dynamique : Multi-agent vs Parité de risque vol. cible |                     0.2114 |       -0.0692 |         0.54   |        0.061  | non                  |
| Dynamique : Multi-agent vs HRP vol. cible              |                     0.5937 |        0.1251 |         1.1544 |        0.0045 | oui                  |


## Renforcement (PPO) – comparaison sur le test

|                                   |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Sortino |   Max drawdown |   Calmar |   VaR 95% (jour) |   % jours positifs |   Rotation annuelle |
|:----------------------------------|----------------------:|------------------------:|---------:|----------:|---------------:|---------:|-----------------:|-------------------:|--------------------:|
| PPO – Prudent                     |                0.0191 |                  0.0846 |  -0.2037 |   -0.2833 |        -0.1875 |   0.1021 |           0.0092 |             0.5281 |              3.6151 |
| Multi-agent – Prudent             |                0.0217 |                  0.0627 |  -0.2608 |   -0.3603 |        -0.1496 |   0.1449 |           0.0062 |             0.5379 |              0.9276 |
| Référence stratégique – Prudent   |                0.0317 |                  0.0736 |  -0.0799 |   -0.1137 |        -0.1615 |   0.1962 |           0.0076 |             0.5403 |              0.4588 |
| 1/N vol. cible – Prudent          |                0.0476 |                  0.069  |   0.1316 |    0.1776 |        -0.1293 |   0.3679 |           0.0069 |             0.5452 |              0.8085 |
| PPO – Équilibré                   |                0.0194 |                  0.1139 |  -0.1239 |   -0.1696 |        -0.2353 |   0.0822 |           0.012  |             0.5281 |              5.6396 |
| Multi-agent – Équilibré           |                0.0473 |                  0.0965 |   0.1152 |    0.1571 |        -0.182  |   0.2601 |           0.0093 |             0.546  |              0.7406 |
| Référence stratégique – Équilibré |                0.0538 |                  0.1047 |   0.1726 |    0.2411 |        -0.1864 |   0.2883 |           0.0103 |             0.5412 |              0.5216 |
| 1/N vol. cible – Équilibré        |                0.062  |                  0.105  |   0.2464 |    0.3286 |        -0.1798 |   0.3448 |           0.0103 |             0.5428 |              0.9523 |
| PPO – Dynamique                   |                0.0167 |                  0.1499 |  -0.08   |   -0.1095 |        -0.2835 |   0.0589 |           0.0157 |             0.5232 |              6.6289 |
| Multi-agent – Dynamique           |                0.081  |                  0.1335 |   0.3523 |    0.4782 |        -0.1994 |   0.4063 |           0.0132 |             0.5346 |              0.6708 |
| Référence stratégique – Dynamique |                0.0833 |                  0.1369 |   0.3626 |    0.4963 |        -0.1994 |   0.4179 |           0.0134 |             0.5379 |              0.5818 |
| 1/N vol. cible – Dynamique        |                0.0708 |                  0.1166 |   0.3042 |    0.4169 |        -0.1957 |   0.362  |           0.011  |             0.542  |              0.6064 |


## Renforcement (PPO) – significativité

|                                          |   diff. de Sharpe observée |   IC 95 % bas |   IC 95 % haut |   P(diff ≤ 0) | significatif (5 %)   |
|:-----------------------------------------|---------------------------:|--------------:|---------------:|--------------:|:---------------------|
| Prudent : PPO vs Multi-agent             |                     0.0572 |       -0.295  |         0.3573 |        0.444  | non                  |
| Prudent : PPO vs Référence stratégique   |                    -0.1239 |       -0.3625 |         0.1107 |        0.867  | non                  |
| Prudent : PPO vs 1/N vol. cible          |                    -0.3357 |       -0.7473 |        -0.0255 |        0.984  | oui                  |
| Équilibré : PPO vs Multi-agent           |                    -0.2394 |       -0.5885 |         0.069  |        0.9485 | non                  |
| Équilibré : PPO vs Référence stratégique |                    -0.2968 |       -0.6118 |         0.0045 |        0.9735 | non                  |
| Équilibré : PPO vs 1/N vol. cible        |                    -0.3707 |       -0.8194 |        -0.0257 |        0.9825 | oui                  |
| Dynamique : PPO vs Multi-agent           |                    -0.4326 |       -0.8351 |        -0.1141 |        0.9935 | oui                  |
| Dynamique : PPO vs Référence stratégique |                    -0.4429 |       -0.8241 |        -0.1261 |        0.996  | oui                  |
| Dynamique : PPO vs 1/N vol. cible        |                    -0.3845 |       -0.8325 |        -0.038  |        0.985  | oui                  |


## Temps de calcul par étape (s)

|                                                       |   secondes |
|:------------------------------------------------------|-----------:|
| 1. Collecte des données                               |     0.2027 |
| 2. Prétraitement et EDA                               |     4.8543 |
| 3. Agent Profil (modèles, test, validation externe)   |   244.755  |
| 3. Agent Profil (interprétabilité)                    |    10.054  |
| 4a. Prévision : données, EDA, ACP                     |    17.3118 |
| 4b. Prévision : modèles, régularisation, ACP          |    69.9536 |
| 4c. Prévision : 3 évaluations, Diebold-Mariano, choix |    58.1262 |
| 4d. Prévision : test et interprétabilité              |    15.0359 |
| 5a. Agent Régime                                      |     9.79   |
| 5b. γ du GMM et Agent Alerte                          |     3.5782 |
| 6. Système multi-agent                                |     9.2755 |
| 7a. Backtest : réglages et simulation                 |   357.653  |
| 7b. Backtest : résultats et significativité           |    11.7111 |
| 8. Apprentissage par renforcement (PPO)               |   506.807  |
| 9. Sauvegarde                                         |     4.7114 |
