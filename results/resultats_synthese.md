# Synthèse des résultats

Données jusqu'au 2026-11-18 – modèle Profil : **Forêt aléatoire** – modèle Prévision : **Forêt aléatoire** (IC validation = 0.1718).


## Agent Profil – comparaison des modèles (validation)

| Modèle                        |   F1 macro CV (train) |   Accuracy |   Balanced acc. |   F1 macro |   ROC-AUC |   Rappel Prudent |   Durée (s) |
|:------------------------------|----------------------:|-----------:|----------------:|-----------:|----------:|-----------------:|------------:|
| Forêt aléatoire               |                0.7359 |     0.8009 |          0.7672 |     0.7689 |    0.8472 |           0.6749 |     59.1099 |
| Régression logistique         |                0.7319 |     0.7814 |          0.7908 |     0.7653 |    0.8586 |           0.8164 |      7.2705 |
| XGBoost                       |                0.7304 |     0.7797 |          0.7663 |     0.7553 |    0.8441 |           0.7297 |     24.6766 |
| LightGBM                      |                0.7372 |     0.7846 |          0.7553 |     0.7534 |    0.8359 |           0.6749 |     22.824  |
| MLP                           |                0.6943 |     0.7115 |          0.6622 |     0.6638 |    0.7337 |           0.5269 |     34.4255 |
| Baseline – tirage stratifié   |                0.4983 |     0.5713 |          0.5058 |     0.5058 |    0.5058 |           0.326  |      8.518  |
| Baseline – classe majoritaire |                0.4056 |     0.6821 |          0.5    |     0.4055 |    0.5    |           0      |      8.9683 |


## Agent Profil – modèle retenu vs baselines (test)

|                                       |   Accuracy |   Balanced acc. |   F1 macro |   ROC-AUC |   Rappel Prudent |
|:--------------------------------------|-----------:|----------------:|-----------:|----------:|-----------------:|
| Forêt aléatoire                       |     0.8058 |          0.7703 |     0.7732 |    0.86   |           0.6734 |
| Forêt aléatoire + seuil 0.40          |     0.7814 |          0.7917 |     0.7653 |    0.86   |           0.8198 |
| Forêt aléatoire + seuil – pondéré WGT |     0.7243 |          0.7494 |     0.7202 |    0.8226 |           0.8387 |
| Baseline – tirage stratifié           |     0.5678 |          0.5014 |     0.5014 |    0.5014 |           0.3202 |
| Baseline – classe majoritaire         |     0.6832 |          0.5    |     0.4059 |    0.5    |           0      |


## Agent Profil – validation externe (autre vague SCF)

|                                             |   Accuracy |   Balanced acc. |   F1 macro |   ROC-AUC |   Rappel Prudent |
|:--------------------------------------------|-----------:|----------------:|-----------:|----------:|-----------------:|
| Test interne – SCF 2022                     |     0.7814 |          0.7917 |     0.7653 |    0.86   |           0.8198 |
| Validation externe – SCF 2019               |     0.7496 |          0.7513 |     0.7297 |    0.8373 |           0.7557 |
| Validation externe – SCF 2019 (pondéré WGT) |     0.7017 |          0.7178 |     0.6998 |    0.8035 |           0.7928 |
| Baseline majoritaire – SCF 2019             |     0.6837 |          0.5    |     0.4061 |    0.5    |           0      |


## Agent Prévision – comparaison des modèles (validation)

|                               |   Accuracy |   Balanced acc. |     F1 |   ROC-AUC |   IC moyen |   IC t-stat (Newey-West) |   Spread Q5-Q1 (21j) |   % dates IC>0 |   AUC CV (train) |   Durée (s) |
|:------------------------------|-----------:|----------------:|-------:|----------:|-----------:|-------------------------:|---------------------:|---------------:|-----------------:|------------:|
| Forêt aléatoire               |     0.5637 |          0.5618 | 0.5299 |    0.6058 |     0.1718 |                   2.8774 |               0.0127 |         0.7812 |           0.5577 |     43.0465 |
| Régression logistique         |     0.5613 |          0.5622 | 0.5614 |    0.5866 |     0.1604 |                   2.4728 |               0.0202 |         0.6979 |           0.5589 |     10.332  |
| LightGBM                      |     0.555  |          0.5518 | 0.5046 |    0.5871 |     0.1465 |                   2.331  |               0.0134 |         0.6979 |           0.5547 |      8.013  |
| XGBoost                       |     0.545  |          0.5443 | 0.526  |    0.5805 |     0.1165 |                   1.9031 |               0.0104 |         0.6875 |           0.5545 |     10.6171 |
| Baseline – momentum 12-1      |     0.5183 |          0.5192 | 0.5183 |    0.5146 |     0.0139 |                   0.2771 |               0.0154 |         0.4688 |         nan      |    nan      |
| MLP                           |     0.5229 |          0.526  | 0.548  |    0.5238 |    -0.0096 |                  -0.1497 |              -0.0169 |         0.4896 |           0.538  |      9.141  |
| Baseline – classe majoritaire |     0.52   |          0.5    | 0      |    0.5    |   nan      |                 nan      |               0      |       nan      |         nan      |    nan      |


## Agent Prévision – modèle retenu vs baselines (test)

|                               |   Accuracy |   Balanced acc. |     F1 |   ROC-AUC |   IC moyen |   IC t-stat (Newey-West) |   Spread Q5-Q1 (21j) |   % dates IC>0 |
|:------------------------------|-----------:|----------------:|-------:|----------:|-----------:|-------------------------:|---------------------:|---------------:|
| Forêt aléatoire               |     0.5461 |          0.5396 | 0.4447 |    0.5652 |     0.0846 |                   1.9579 |               0.0086 |         0.6017 |
| Baseline – classe majoritaire |     0.52   |          0.5    | 0      |    0.5    |   nan      |                 nan      |               0      |       nan      |
| Baseline – momentum 12-1      |     0.5285 |          0.5293 | 0.5285 |    0.5307 |     0.0537 |                   1.6312 |               0.0088 |         0.5975 |


## Agent Régime – validation hors échantillon

| régime   |   fréquence |   rdt moyen 21j futur |   vol future |
|:---------|------------:|----------------------:|-------------:|
| Calme    |      0.1884 |               -0.001  |       0.133  |
| Neutre   |      0.4254 |                0.0127 |       0.1337 |
| Stress   |      0.3861 |                0.0198 |       0.2228 |


## Réglages de l'allocation choisis sur la validation (moyenne des 3 profils)

|                   |   Sharpe |   Max drawdown |   Rotation annuelle |
|:------------------|---------:|---------------:|--------------------:|
| ('aucune', 0.5)   |   0.5299 |        -0.23   |              2.4328 |
| ('aucune', 2.0)   |   0.5833 |        -0.2244 |              1.7357 |
| ('aucune', 5.0)   |   0.5678 |        -0.2227 |              1.4572 |
| ('aucune', 10.0)  |   0.5442 |        -0.2222 |              1.3261 |
| ('forte', 0.5)    |   0.456  |        -0.2056 |              2.9185 |
| ('forte', 2.0)    |   0.4994 |        -0.2036 |              2.2652 |
| ('forte', 5.0)    |   0.496  |        -0.2034 |              1.9661 |
| ('forte', 10.0)   |   0.477  |        -0.2033 |              1.7843 |
| ('légère', 0.5)   |   0.5048 |        -0.229  |              2.47   |
| ('légère', 2.0)   |   0.5557 |        -0.224  |              1.8058 |
| ('légère', 5.0)   |   0.5398 |        -0.2225 |              1.5488 |
| ('légère', 10.0)  |   0.5143 |        -0.2221 |              1.4263 |
| ('modérée', 0.5)  |   0.4494 |        -0.2249 |              2.6344 |
| ('modérée', 2.0)  |   0.497  |        -0.2213 |              1.9864 |
| ('modérée', 5.0)  |   0.4911 |        -0.2205 |              1.7266 |
| ('modérée', 10.0) |   0.4699 |        -0.2203 |              1.5893 |


## Backtest – performances sur la période de test

|                                   |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Sortino |   Max drawdown |   Calmar |   VaR 95% (jour) |   % jours positifs |   Rotation annuelle |
|:----------------------------------|----------------------:|------------------------:|---------:|----------:|---------------:|---------:|-----------------:|-------------------:|--------------------:|
| Multi-agent – Prudent             |                0.0187 |                  0.0662 |  -0.2886 |   -0.3996 |        -0.159  |   0.1173 |           0.0064 |             0.5437 |              1.1955 |
| Multi-agent sans ML – Prudent     |                0.0137 |                  0.0646 |  -0.3735 |   -0.5184 |        -0.1679 |   0.0813 |           0.0065 |             0.5314 |              0.9244 |
| Multi-agent – Équilibré           |                0.0513 |                  0.1025 |   0.1512 |    0.2062 |        -0.1901 |   0.2698 |           0.0102 |             0.542  |              1.1437 |
| Multi-agent sans ML – Équilibré   |                0.0524 |                  0.095  |   0.1661 |    0.2268 |        -0.1833 |   0.2856 |           0.0094 |             0.5469 |              0.6762 |
| Multi-agent – Dynamique           |                0.0925 |                  0.1509 |   0.3979 |    0.5434 |        -0.2199 |   0.4205 |           0.0149 |             0.5306 |              1.0801 |
| Multi-agent sans ML – Dynamique   |                0.0876 |                  0.1315 |   0.4022 |    0.5473 |        -0.1974 |   0.4441 |           0.0125 |             0.538  |              0.6313 |
| Équipondérée 1/N                  |                0.078  |                  0.1163 |   0.3623 |    0.4978 |        -0.1957 |   0.3987 |           0.011  |             0.5429 |              0.596  |
| 60/40                             |                0.0574 |                  0.1112 |   0.2    |    0.2864 |        -0.2095 |   0.2741 |           0.0109 |             0.5224 |              0.3822 |
| Variance minimale                 |                0.0131 |                  0.0486 |  -0.526  |   -0.7466 |        -0.1367 |   0.0958 |           0.005  |             0.5339 |              1.1091 |
| Markowitz historique              |                0.1764 |                  0.1949 |   0.7279 |    0.9173 |        -0.2316 |   0.7614 |           0.0186 |             0.5469 |              6.1706 |
| Référence stratégique – Prudent   |                0.0287 |                  0.0825 |  -0.0976 |   -0.1399 |        -0.1861 |   0.1544 |           0.0087 |             0.5371 |              0.4465 |
| 1/N vol. cible – Prudent          |                0.0516 |                  0.0688 |   0.1872 |    0.2529 |        -0.1293 |   0.3988 |           0.0068 |             0.5461 |              0.8066 |
| Référence stratégique – Équilibré |                0.0596 |                  0.1042 |   0.2259 |    0.316  |        -0.1858 |   0.3207 |           0.0099 |             0.542  |              0.5186 |
| 1/N vol. cible – Équilibré        |                0.069  |                  0.1046 |   0.3102 |    0.4152 |        -0.1798 |   0.384  |           0.0102 |             0.5437 |              0.9441 |
| Référence stratégique – Dynamique |                0.0931 |                  0.1358 |   0.4303 |    0.5908 |        -0.1973 |   0.4717 |           0.0131 |             0.5396 |              0.5812 |
| 1/N vol. cible – Dynamique        |                0.078  |                  0.1162 |   0.3619 |    0.4971 |        -0.1957 |   0.3983 |           0.011  |             0.5429 |              0.6033 |


## Backtest – comparaison à risque comparable

|                                        |   Rendement annualisé |   Volatilité annualisée |   Sharpe |   Max drawdown |   Rotation annuelle |
|:---------------------------------------|----------------------:|------------------------:|---------:|---------------:|--------------------:|
| ('Prudent', 'Multi-agent')             |                0.0187 |                  0.0662 |  -0.2886 |        -0.159  |              1.1955 |
| ('Prudent', 'Multi-agent sans ML')     |                0.0137 |                  0.0646 |  -0.3735 |        -0.1679 |              0.9244 |
| ('Prudent', 'Référence stratégique')   |                0.0287 |                  0.0825 |  -0.0976 |        -0.1861 |              0.4465 |
| ('Prudent', '1/N vol. cible')          |                0.0516 |                  0.0688 |   0.1872 |        -0.1293 |              0.8066 |
| ('Équilibré', 'Multi-agent')           |                0.0513 |                  0.1025 |   0.1512 |        -0.1901 |              1.1437 |
| ('Équilibré', 'Multi-agent sans ML')   |                0.0524 |                  0.095  |   0.1661 |        -0.1833 |              0.6762 |
| ('Équilibré', 'Référence stratégique') |                0.0596 |                  0.1042 |   0.2259 |        -0.1858 |              0.5186 |
| ('Équilibré', '1/N vol. cible')        |                0.069  |                  0.1046 |   0.3102 |        -0.1798 |              0.9441 |
| ('Dynamique', 'Multi-agent')           |                0.0925 |                  0.1509 |   0.3979 |        -0.2199 |              1.0801 |
| ('Dynamique', 'Multi-agent sans ML')   |                0.0876 |                  0.1315 |   0.4022 |        -0.1974 |              0.6313 |
| ('Dynamique', 'Référence stratégique') |                0.0931 |                  0.1358 |   0.4303 |        -0.1973 |              0.5812 |
| ('Dynamique', '1/N vol. cible')        |                0.078  |                  0.1162 |   0.3619 |        -0.1957 |              0.6033 |


## Backtest – significativité (bootstrap par blocs)

|                                              |   diff. Sharpe observée |   IC 95% bas |   IC 95% haut |   P(diff ≤ 0) |
|:---------------------------------------------|------------------------:|-------------:|--------------:|--------------:|
| Prudent : complet vs sans ML                 |                  0.0851 |      -0.0527 |        0.2085 |        0.1215 |
| Prudent : complet vs référence stratégique   |                 -0.1913 |      -0.4604 |        0.1102 |        0.888  |
| Prudent : complet vs 1/N vol. cible          |                 -0.4764 |      -0.7847 |       -0.1907 |        0.999  |
| Équilibré : complet vs sans ML               |                 -0.0149 |      -0.2005 |        0.1599 |        0.555  |
| Équilibré : complet vs référence stratégique |                 -0.0747 |      -0.2859 |        0.143  |        0.721  |
| Équilibré : complet vs 1/N vol. cible        |                 -0.1592 |      -0.3802 |        0.0147 |        0.965  |
| Dynamique : complet vs sans ML               |                 -0.0043 |      -0.2076 |        0.189  |        0.496  |
| Dynamique : complet vs référence stratégique |                 -0.0324 |      -0.2337 |        0.1738 |        0.599  |
| Dynamique : complet vs 1/N vol. cible        |                  0.036  |      -0.2204 |        0.2705 |        0.409  |
