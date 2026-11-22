"""
Configuration centrale du projet
« Assistant multi-agent pour la sélection intelligente de portefeuilles financiers ».

Toutes les constantes (univers d'actifs, sources de données, dates de découpage,
contraintes par profil de risque) sont regroupées ici pour garantir la
reproductibilité des expériences.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Arborescence
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
for _p in (DATA_RAW, DATA_PROC, MODELS_DIR, RESULTS_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Univers d'investissement (ETF cotés, données Yahoo Finance)
# ticker -> (libellé, classe d'actifs)
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Actions – indices larges
    "SPY": ("S&P 500", "Actions"),
    "QQQ": ("Nasdaq 100", "Actions"),
    "IWM": ("Russell 2000 (petites capitalisations)", "Actions"),
    "EFA": ("Actions pays développés hors US", "Actions"),
    "EEM": ("Actions pays émergents", "Actions"),
    # Actions – secteurs US
    "XLK": ("Secteur Technologie", "Actions"),
    "XLF": ("Secteur Finance", "Actions"),
    "XLV": ("Secteur Santé", "Actions"),
    "XLE": ("Secteur Énergie", "Actions"),
    "XLY": ("Secteur Consommation discrétionnaire", "Actions"),
    "XLP": ("Secteur Consommation de base", "Actions"),
    "XLI": ("Secteur Industrie", "Actions"),
    "XLU": ("Secteur Services publics", "Actions"),
    # Immobilier coté
    "VNQ": ("Immobilier coté US (REIT)", "Immobilier"),
    # Poche monétaire / quasi-monétaire (duration ≈ 2 ans, risque de taux faible)
    "SHY": ("Bons du Trésor US 1-3 ans", "Monétaire"),
    # Obligations
    "IEF": ("Obligations Trésor US 7-10 ans", "Obligations"),
    "TLT": ("Obligations Trésor US 20+ ans", "Obligations"),
    "LQD": ("Obligations entreprises Investment Grade", "Obligations"),
    "HYG": ("Obligations entreprises High Yield", "Obligations"),
    "TIP": ("Obligations indexées sur l'inflation", "Obligations"),
    "EMB": ("Dette souveraine émergente (USD)", "Obligations"),
    # Matières premières
    "GLD": ("Or", "Matières premières"),
    "SLV": ("Argent", "Matières premières"),
    "DBC": ("Panier de matières premières", "Matières premières"),
    # Crypto-actif
    "BTC-USD": ("Bitcoin", "Crypto"),
}
TICKERS = list(UNIVERSE)
ASSET_CLASS = {t: c for t, (_, c) in UNIVERSE.items()}
ASSET_NAME = {t: n for t, (n, _) in UNIVERSE.items()}
MARKET_TICKER = "SPY"          # référence de marché (calendrier NYSE, bêta)
BOND_PROXY = "IEF"              # pour le benchmark 60/40

START_DATE = "2010-01-01"
END_DATE = None                 # None = jusqu'à aujourd'hui

# ---------------------------------------------------------------------------
# Séries macroéconomiques FRED (Federal Reserve Bank of St. Louis)
# ---------------------------------------------------------------------------
FRED_DAILY = {
    "DGS10": "Taux Trésor US 10 ans (%)",
    "DGS2": "Taux Trésor US 2 ans (%)",
    "DTB3": "Taux T-Bill 3 mois (%) – taux sans risque",
    "T10Y2Y": "Pente de la courbe 10 ans – 2 ans (%)",
    "BAA10Y": "Spread de crédit Moody's Baa – Trésor 10 ans (%)",
    "VIXCLS": "Indice de volatilité implicite VIX",
}
FRED_MONTHLY = {
    "CPIAUCSL": "Indice des prix à la consommation",
    "UNRATE": "Taux de chômage US (%)",
    "INDPRO": "Indice de production industrielle",
}
# Délai de publication conservateur appliqué aux séries mensuelles :
# la valeur du mois M n'est utilisable qu'à partir du 15 du mois M+1
# (ex. l'inflation de janvier est publiée mi-février) -> évite la fuite temporelle.
MONTHLY_RELEASE_LAG_DAYS = 15
# Les séries quotidiennes FRED sont décalées d'un jour ouvré (publication J+1).
DAILY_RELEASE_LAG = 1

# ---------------------------------------------------------------------------
# Survey of Consumer Finances (Réserve fédérale US) – agent Profil investisseur
# ---------------------------------------------------------------------------
SCF_YEAR = 2022
SCF_URL = f"https://www.federalreserve.gov/econres/files/scfp{SCF_YEAR}excel.zip"
# Vague antérieure utilisée pour une VALIDATION EXTERNE de l'agent Profil (modèle entraîné
# sur 2022, testé sur des ménages d'une autre enquête, 2019, avant le COVID)
SCF_EXTERNAL_YEAR = 2019
SCF_HOUSEHOLD_ID = "YY1"   # identifiant du ménage (5 implicates par ménage)
SCF_WEIGHT = "WGT"

# Variables explicatives retenues (celles absentes du fichier sont ignorées).
# On EXCLUT volontairement les variables de composition du patrimoine financier
# (part en actions, fonds, etc.) qui sont des conséquences directes de la
# tolérance au risque : elles constitueraient une fuite de la cible.
SCF_NUM_FEATURES = [
    "AGE", "KIDS", "INCOME", "NORMINC", "WAGEINC", "NETWORTH", "ASSET", "DEBT",
    "LEVRATIO", "DEBT2INC", "KNOWL", "FINLIT", "EDUC",
]
SCF_CAT_FEATURES = [
    # socio-démographie
    "HHSEX", "EDCL", "MARRIED", "OCCAT1", "OCCAT2", "LF", "LIFECL", "HOUSECL", "HBUS",
    # comportement d'épargne et de crédit (attitudes, pas la composition du portefeuille)
    "SAVED", "WSAVED", "SPENDMOR", "SPENDLESS", "EXPENSHILO", "LATE", "HPAYDAY",
    "BNKRUPLAST5", "FORECLLAST5", "TURNDOWN", "EMERGSAV", "EMERGBORR", "EMERGCUT", "EMERGPSTP",
    # motifs d'épargne
    "SAVRES1", "SAVRES2", "SAVRES3", "SAVRES4", "SAVRES5", "SAVRES6", "SAVRES7", "SAVRES8", "SAVRES9",
    # façon de prendre les décisions d'investissement
    "ISHOPNONE", "ISHOPGRDL", "ISHOPMODR", "IFINPRO", "ISELF", "IINTERNET", "IFRIENDWORK",
    "IMAGZNEWS", "ICALL", "INTERNET",
]
# Libellés lisibles des variables SCF (explications SHAP du notebook et de l'application)
SCF_LABELS = {
    "AGE": "Âge", "KIDS": "Enfants à charge", "INCOME": "Revenu annuel", "NORMINC": "Revenu habituel",
    "WAGEINC": "Revenus salariaux", "NETWORTH": "Patrimoine net", "ASSET": "Total des actifs",
    "DEBT": "Total des dettes", "LEVRATIO": "Dettes / actifs", "DEBT2INC": "Dettes / revenu",
    "KNOWL": "Connaissances financières (0-10)", "FINLIT": "Quiz de culture financière (0-3)",
    "EDUC": "Durée des études", "HHSEX": "Sexe", "EDCL": "Niveau d'études", "MARRIED": "En couple",
    "OCCAT1": "Situation professionnelle", "OCCAT2": "Catégorie professionnelle", "LF": "Actif sur le marché du travail",
    "LIFECL": "Étape du cycle de vie", "HOUSECL": "Propriétaire du logement", "HBUS": "Possède une entreprise",
    "SAVED": "A épargné l'an dernier", "WSAVED": "Comportement d'épargne", "SPENDMOR": "Dépenses plus élevées que prévu",
    "SPENDLESS": "Dépenses plus faibles que prévu", "EXPENSHILO": "Dépenses inhabituelles", "LATE": "Retards de paiement",
    "HPAYDAY": "Prêt sur salaire", "BNKRUPLAST5": "Faillite (5 ans)", "FORECLLAST5": "Saisie immobilière (5 ans)",
    "TURNDOWN": "Crédit refusé", "EMERGSAV": "Épargne de précaution", "EMERGBORR": "Emprunterait en cas d'urgence",
    "EMERGCUT": "Réduirait ses dépenses en cas d'urgence", "EMERGPSTP": "Reporterait ses paiements en cas d'urgence",
    "SAVRES1": "Ne peut pas épargner", "SAVRES2": "Épargne pour les études", "SAVRES3": "Épargne pour la famille",
    "SAVRES4": "Épargne pour le logement", "SAVRES5": "Épargne pour des achats", "SAVRES6": "Épargne pour la retraite",
    "SAVRES7": "Épargne de précaution / liquidité", "SAVRES8": "Épargne pour investir", "SAVRES9": "Épargne sans but précis",
    "ISHOPNONE": "Ne compare pas les placements", "ISHOPGRDL": "Compare beaucoup les placements",
    "ISHOPMODR": "Compare modérément les placements", "IFINPRO": "Conseiller financier", "ISELF": "Décide seul",
    "IINTERNET": "S'informe sur internet", "IFRIENDWORK": "Conseils d'amis / collègues", "IMAGZNEWS": "Presse spécialisée",
    "ICALL": "Démarchage téléphonique", "INTERNET": "Banque en ligne",
}

PROFILE_CLASSES = ["Prudent", "Équilibré", "Dynamique"]

# Formulation de la tâche de l'agent Profil :
# - "binaire" (recommandé) : le modèle ML apprend l'AVERSION AU RISQUE (Prudent vs Non prudent).
#   La classe « Dynamique » ne représente que ~5 % des ménages du SCF et n'est pas séparable
#   d'« Équilibré » avec des variables socio-économiques (F1 macro plafonné à ~0,5 quelle que
#   soit la méthode). Le passage Équilibré -> Dynamique est décidé par deux questions explicites
#   du questionnaire (horizon, perte acceptable), comme dans les questionnaires réglementaires.
# - "3classes" : formulation d'origine, conservée pour comparaison.
PROFILE_TARGET_MODE = "binaire"
PROFILE_ML_CLASSES = ["Prudent", "Non prudent"] if PROFILE_TARGET_MODE == "binaire" else PROFILE_CLASSES
DYNAMIC_RULE = {"horizon_min_ans": 10, "perte_acceptable_min_pct": 20}

# ---------------------------------------------------------------------------
# Agent Prévision – paramètres de la tâche supervisée
# ---------------------------------------------------------------------------
HORIZON = 21          # horizon de prévision : 21 jours de bourse (~1 mois)
SAMPLE_EVERY = 5      # un échantillon par semaine pour l'apprentissage
LOOKBACK_COV = 252    # fenêtre d'estimation de la covariance (1 an)

# Découpage temporel (fait AVANT tout prétraitement ajusté sur les données)
TRAIN_END = "2019-12-31"   # entraînement : 2010/2011 -> 2019
VAL_END = "2021-12-31"     # validation   : 2020 -> 2021 (inclut le choc COVID)
#                            test         : 2022 -> fin (inclut le marché baissier 2022)
EMBARGO_DAYS = 35          # purge : ~21 jours de bourse + marge (étiquettes chevauchantes)

# ---------------------------------------------------------------------------
# Contraintes d'allocation par profil de risque
# ---------------------------------------------------------------------------
# defensive_min : part minimale « défensive » = obligations + monétaire.
# Décision de la refonte (22/11/2026), prise AVANT toute nouvelle exécution sur le test :
# un investisseur prudent doit pouvoir détenir du quasi-monétaire (SHY) sans risque de taux,
# au lieu d'être forcé vers des obligations longues (logique des allocations MiFID II / AMMC).
# Limite assumée : la crise obligataire de 2022 était connue au moment de ce choix
# (la période de test avait été vue lors des versions précédentes du projet).
# long_bond_max : plafond sur les obligations du Trésor à 20 ans et plus (TLT), très sensibles aux taux.
PROFILE_CONSTRAINTS = {
    "Prudent": dict(target_vol=0.06, risk_aversion=10.0, w_max=0.25, cash_max=0.40,
                    equity_max=0.35, defensive_min=0.50, long_bond_max=0.10,
                    commodity_max=0.10, crypto_max=0.00),
    "Équilibré": dict(target_vol=0.10, risk_aversion=5.0, w_max=0.20, cash_max=0.30,
                      equity_max=0.60, defensive_min=0.25, long_bond_max=0.15,
                      commodity_max=0.15, crypto_max=0.02),
    "Dynamique": dict(target_vol=0.16, risk_aversion=2.5, w_max=0.20, cash_max=0.20,
                      equity_max=0.90, defensive_min=0.00, long_bond_max=0.20,
                      commodity_max=0.20, crypto_max=0.05),
}
LONG_BOND_TICKERS = ["TLT"]
CASH_TICKER = "SHY"
# Allocation stratégique de référence par profil (poids par classe d'actifs, répartis à parts
# égales entre les actifs disponibles de la classe). Elle sert à calculer les rendements
# d'équilibre implicites π = λ·Σ·w_ref (Black-Litterman) : sans signal ML, l'optimiseur
# retrouve cette allocation au lieu de dériver vers la variance minimale.
PROFILE_BENCHMARK = {
    "Prudent": {"Actions": 0.25, "Immobilier": 0.05, "Obligations": 0.40, "Monétaire": 0.25,
                "Matières premières": 0.05},
    "Équilibré": {"Actions": 0.50, "Immobilier": 0.05, "Obligations": 0.33, "Monétaire": 0.05,
                  "Matières premières": 0.07},
    "Dynamique": {"Actions": 0.75, "Immobilier": 0.05, "Obligations": 0.10, "Matières premières": 0.08,
                  "Crypto": 0.02},
}
# Ajustement de la cible de risque selon le régime de marché détecté
# Valeur par défaut. Le notebook (section 8.0) choisit l'intensité sur la période de VALIDATION
# parmi plusieurs jeux (aucune / légère / modérée / forte réduction) et l'enregistre avec le
# modèle de régime ; l'application utilise alors la valeur choisie.
REGIME_VOL_MULTIPLIER = {"Calme": 1.00, "Neutre": 0.95, "Stress": 0.75}
REGIME_MULTIPLIER_GRID = {
    "aucune":  {"Calme": 1.00, "Neutre": 1.00, "Stress": 1.00},
    "légère":  {"Calme": 1.00, "Neutre": 1.00, "Stress": 0.90},
    "modérée": {"Calme": 1.00, "Neutre": 0.95, "Stress": 0.75},
    "forte":   {"Calme": 1.00, "Neutre": 0.90, "Stress": 0.60},
}

TRANSACTION_COST_BPS = 10   # coût de transaction (aller simple) en points de base

# Pénalité quadratique sur la rotation du portefeuille (γ dans ‖w − w_préc‖²).
# Le notebook la choisit sur la période de VALIDATION 2020-2021 (section 8) parmi
# {0,5 ; 2 ; 5 ; 10} et l'enregistre avec le modèle de prévision ; 5 est la valeur obtenue
# lors du diagnostic (rotation divisée par ~2 sans perte de Sharpe en validation).
TURNOVER_PENALTY = 5.0

# Réduction de l'IC mesuré en validation avant de l'utiliser dans α = IC·σ·z.
# L'IC de validation surestime en général la performance future (sélection du meilleur
# modèle parmi plusieurs, non-stationnarité) : on le divise par deux, pratique courante.
IC_SHRINK = 0.5

# Seuil de décision de l'agent Profil : rappel minimal exigé sur la classe « Prudent »
# (ne pas repérer un client prudent est l'erreur la plus coûteuse pour un conseiller).
PRUDENT_RECALL_MIN = 0.80

# Devises de saisie dans l'application. Le modèle Profil a été entraîné sur des montants en
# dollars US (SCF 2022) : les montants saisis sont convertis en USD. Taux par défaut
# (1 USD = x devise, novembre 2026), modifiables dans l'application.
CURRENCIES = {
    "MAD": {"label": "Dirham marocain (MAD)", "symbol": "MAD", "per_usd": 10.0},
    "EUR": {"label": "Euro (EUR)", "symbol": "€", "per_usd": 0.91},
    "USD": {"label": "Dollar américain (USD)", "symbol": "$", "per_usd": 1.0},
}
