from cellules import code, md

CELLS = [
md(r"""
## 1. Collecte des données réelles

Toutes les données sont **ouvertes** (open data) et téléchargées automatiquement par `src/data_download.py`.

| Source | Contenu | Fréquence | Utilisée par |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | Cours de clôture ajustés (dividendes, divisions) de 25 actifs : ETF actions, secteurs, obligations, immobilier, matières premières, monétaire, Bitcoin | Quotidienne, 2010 → aujourd'hui | Prévision, Régime, Alerte, Risque, Allocation |
| **FRED** (Federal Reserve Bank of St. Louis) | Taux 3 mois / 2 ans / 10 ans, pente de courbe, spread de crédit Baa, VIX, inflation, chômage, production industrielle | Quotidienne / mensuelle | Prévision, Régime |
| **Survey of Consumer Finances 2022** (Réserve fédérale) | ~4 600 ménages américains (× 5 imputations) : âge, revenus, patrimoine, dettes, éducation, connaissances financières, **tolérance au risque déclarée** | Coupe transversale | Profil |
| **Survey of Consumer Finances 2019** | Autre vague de la même enquête | Coupe transversale | Validation externe du Profil |

Les fichiers bruts sont mis en cache dans `data/raw/` : une nouvelle exécution ne retélécharge rien, ce qui garantit la **reproductibilité**.
"""),
code(r"""
univers = pd.DataFrame([(t, n, c) for t, (n, c) in UNIVERSE.items()],
                       columns=["Ticker", "Libellé", "Classe d'actifs"]).set_index("Ticker")
display(univers.groupby("Classe d'actifs").size().rename("nb actifs").to_frame().T)
univers
"""),
code(r"""
close_raw, volume_raw = dl.download_yahoo()
fred_raw = dl.download_fred()
scf_raw = dl.download_scf()

print("Cours Yahoo :", close_raw.shape, close_raw.index.min().date(), "→", close_raw.index.max().date())
print("Séries FRED :", fred_raw.shape)
print("SCF 2022    :", scf_raw.shape)
chrono("1. Collecte des données")
"""),
]
