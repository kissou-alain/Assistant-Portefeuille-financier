"""
Mise à jour sécurisée des données de marché, partagée par l'API et l'application Streamlit.

- Téléchargement RAPIDE (seulement les derniers jours, limite de temps stricte : ~30 s).
- Tout est recalculé en mémoire et contrôlé ; les fichiers ne sont remplacés que si tout est
  complet et au moins aussi récent. Sinon une exception est levée et rien n'est modifié.
- Les données actuelles sont sauvegardées dans data/backup/ avant toute écriture, et peuvent
  être restaurées.
- Les modèles ne sont PAS ré-entraînés (relancer le notebook pour cela).
"""
from __future__ import annotations

import os
import shutil

import pandas as pd

from .config import DATA_PROC, ROOT

BACKUP_DIR = ROOT / "data" / "backup"


def data_files() -> list:
    from .data_download import FRED_FILE, YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE
    return [YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE, FRED_FILE, DATA_PROC / "prices_clean.csv",
            DATA_PROC / "panel_recent.csv", DATA_PROC / "market_features.csv"]


def backup_data() -> None:
    """Copie les données actuelles dans data/backup/ (on ne sauvegarde jamais un fichier vide)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in data_files():
        if f.exists() and f.stat().st_size > 1000:
            shutil.copy2(f, BACKUP_DIR / f.name)


def has_backup() -> bool:
    return all((BACKUP_DIR / f.name).exists() for f in data_files())


def restore_data() -> None:
    """Remet en place les données sauvegardées avant la dernière mise à jour."""
    if not has_backup():
        raise RuntimeError("aucune sauvegarde disponible")
    for f in data_files():
        tmp = f.with_name(f.name + ".tmp")
        shutil.copy2(BACKUP_DIR / f.name, tmp)
        os.replace(tmp, f)


def refresh_market_data(timeout_yahoo: float = 20, timeout_fred: float = 10) -> pd.Timestamp:
    """Télécharge les derniers cours (Yahoo) et séries macro (FRED), recalcule les variables,
    contrôle le résultat, sauvegarde l'existant puis écrit. Retourne la nouvelle dernière date."""
    from .data_download import (FRED_FILE, YAHOO_CLOSE_FILE, YAHOO_VOLUME_FILE, _atomic_to_csv,
                                update_fred_incremental, update_yahoo_incremental)
    from .features import compute_asset_features, compute_market_features
    from .preprocessing import build_macro_features, clean_prices

    close, volume = update_yahoo_incremental(timeout=timeout_yahoo)     # erreur si incomplet ou trop lent
    try:
        fred = update_fred_incremental(timeout=timeout_fred)
    except Exception:
        fred = pd.read_csv(FRED_FILE, index_col=0, parse_dates=True)   # macro non bloquante
    prices_new, _ = clean_prices(close)
    macro = build_macro_features(fred, prices_new.index)
    recent_prices = prices_new.tail(700)                                # 252 j d'historique + 260 dates utiles
    panel = compute_asset_features(recent_prices, macro.loc[recent_prices.index])
    mkt_new = compute_market_features(prices_new, macro)
    if panel.empty or mkt_new.dropna(how="all").empty:
        raise RuntimeError("les variables recalculées sont vides")
    old = DATA_PROC / "prices_clean.csv"
    old_last = pd.read_csv(old, index_col=0, parse_dates=True).index.max() if old.exists() else None
    if old_last is not None and pd.notna(old_last) and prices_new.index.max() < old_last:
        raise RuntimeError(f"les nouvelles données s'arrêtent au {prices_new.index.max():%d/%m/%Y}, "
                           f"avant les données actuelles ({old_last:%d/%m/%Y})")

    backup_data()
    dates = panel.index.get_level_values("date").unique().sort_values()[-260:]
    _atomic_to_csv(close, YAHOO_CLOSE_FILE)
    _atomic_to_csv(volume, YAHOO_VOLUME_FILE)
    _atomic_to_csv(fred, FRED_FILE)
    _atomic_to_csv(prices_new, DATA_PROC / "prices_clean.csv")
    _atomic_to_csv(panel[panel.index.get_level_values("date").isin(dates)], DATA_PROC / "panel_recent.csv")
    _atomic_to_csv(mkt_new, DATA_PROC / "market_features.csv")
    return prices_new.index.max()
