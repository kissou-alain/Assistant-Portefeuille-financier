# Données du projet

Toutes les données sont **réelles** et proviennent de sources publiques officielles ou reconnues. Elles sont téléchargées automatiquement par `python -m src.data_download` (ou par la section 1 du notebook) et mises en cache ici.

## 1. Fichiers bruts (`data/raw/`)

| Fichier | Source | Description |
|---|---|---|
| `yahoo_close_raw.csv` | Yahoo Finance, bibliothèque `yfinance` | Cours de clôture **ajustés** des dividendes et des divisions (`auto_adjust=True`). Une ligne par date, une colonne par ticker, depuis le 01/01/2010. |
| `yahoo_volume_raw.csv` | Yahoo Finance | Volumes échangés (contrôle qualité). |
| `fred_raw.csv` | FRED, https://fred.stlouisfed.org | Séries aux fréquences d'origine : DGS10, DGS2, DTB3, T10Y2Y, BAA10Y, VIXCLS (quotidiennes) ; CPIAUCSL, UNRATE, INDPRO (mensuelles). |
| `scfp2022.csv` | Federal Reserve Board, Survey of Consumer Finances 2022, *Summary Extract Public Data* : https://www.federalreserve.gov/econres/scfindex.htm | Une ligne par implicate (5 par ménage). Variables utilisées : YY1, WGT, AGE, KIDS, INCOME, NETWORTH, ASSET, DEBT, LEVRATIO, DEBT2INC, KNOWL, FINLIT, HHSEX, EDCL, MARRIED, OCCAT1, LIFECL, HOUSECL, SAVED, NOFINRISK, YESFINRISK. |

### Univers d'investissement (25 actifs)
Actions : SPY, QQQ, IWM, EFA, EEM · Secteurs US : XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLU · Immobilier : VNQ · Monétaire : SHY · Obligations : IEF, TLT, LQD, HYG, TIP, EMB · Matières premières : GLD, SLV, DBC · Crypto : BTC-USD.

## 2. Traitements appliqués

### Cours (`src/preprocessing.py::clean_prices`)
1. Tri chronologique, suppression des dates en double.
2. Alignement sur le calendrier NYSE (jours de cotation de SPY). Le Bitcoin, coté 7 j/7, est ré-échantillonné sur ces dates.
3. Suppression des prix nuls ou négatifs.
4. Détection des pics aberrants (aller-retour de plus de 40 % en 2 jours, hors crypto), remplacés par une valeur manquante.
5. Propagation de la dernière valeur connue sur 5 jours maximum. Jamais de remplissage vers l'arrière, qui introduirait de l'information future.
6. Les actifs à l'historique plus court (BTC-USD à partir de 2014) gardent des valeurs manquantes avant leur première cotation. Ils sont exclus de l'univers tant qu'ils n'ont pas 1 an d'historique.
7. Les rendements extrêmes réels (krach COVID, par exemple) sont **conservés**.

### Macroéconomie (`build_macro_features`)
- Séries quotidiennes décalées d'1 jour ouvré (publication à J+1).
- Séries mensuelles disponibles seulement à partir du 15 du mois suivant (délai de publication).
- Variables dérivées : inflation sur 12 mois, variation du chômage sur 3 mois, croissance de la production industrielle, variation du VIX sur 21 jours, des taux et du spread de crédit sur 63 jours.

### SCF (`prepare_scf`)
- Cible (formulation binaire, retenue) : `NOFINRISK=1` → Prudent ; sinon → Non prudent. La formulation à 3 classes (`YESFINRISK=1` → Dynamique, sinon Équilibré) est conservée pour comparaison : la classe Dynamique (≈ 5 % des ménages) n'est pas séparable avec ces variables. Le passage Équilibré → Dynamique est décidé par le questionnaire noté.
- Exclusion des variables de composition du portefeuille financier, qui sont des conséquences de la tolérance au risque (fuite de la cible).
- Les codes négatifs de non-réponse sont convertis en valeurs manquantes. Les variables catégorielles sont converties en texte.
- Découpage **par ménage** (`YY1`), car les 5 implicates d'un ménage sont quasi identiques.
- Imputation, transformation logarithmique signée des montants, écrêtage 1 %/99 % et standardisation sont faits dans un `Pipeline` ajusté sur l'entraînement.

## 3. Fichiers produits (`data/processed/`)
| Fichier | Contenu |
|---|---|
| `prices_clean.csv` | Cours nettoyés (date × ticker) |
| `returns_daily.csv` | Rendements quotidiens |
| `macro_features.csv` | Variables macro alignées et décalées |
| `panel_features.csv.gz` | Panel (date, ticker) : 20 variables techniques + 10 variables macro + classe d'actifs + rendement futur + cible |
| `panel_recent.csv` | Dernière année du panel (utilisée par l'application) |
| `market_features.csv` | Variables de l'agent Régime |
| `regimes.csv` | Régime détecté chaque jour |

## 4. Conditions d'utilisation
Les données Yahoo Finance sont réservées à un usage personnel et académique. Les données FRED et SCF sont publiques. Citer : *Board of Governors of the Federal Reserve System, Survey of Consumer Finances 2022* ; *Federal Reserve Bank of St. Louis, FRED*.
