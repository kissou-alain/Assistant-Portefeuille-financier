"""
Téléchargement des données RÉELLES depuis des sources publiques et fiables :

1. Yahoo Finance (via la bibliothèque `yfinance`) : cours ajustés (dividendes et
   splits) et volumes de 25 ETF / actifs cotés.
2. FRED – Federal Reserve Bank of St. Louis : séries macroéconomiques et de taux.
3. Survey of Consumer Finances (SCF) – Board of Governors of the Federal Reserve
   System : enquête réelle sur ~4 600 ménages américains (x5 implicates),
   contenant la tolérance au risque financier déclarée.

Chaque fonction met en cache le fichier brut dans data/raw/ : une fois les
données téléchargées, les exécutions suivantes sont reproductibles à l'identique
(même sans connexion). Utiliser force=True pour rafraîchir.

Usage en ligne de commande :
    python -m src.data_download            # télécharge tout
    python -m src.data_download --force    # force la mise à jour
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile

import pandas as pd
import requests

from .config import (DATA_RAW, END_DATE, FRED_DAILY, FRED_MONTHLY, SCF_EXTERNAL_YEAR,
                     SCF_YEAR, START_DATE, TICKERS)

HEADERS = {"User-Agent": "Mozilla/5.0 (projet-academique-ML; contact: etudiant)"}

YAHOO_CLOSE_FILE = DATA_RAW / "yahoo_close_raw.csv"
YAHOO_VOLUME_FILE = DATA_RAW / "yahoo_volume_raw.csv"
FRED_FILE = DATA_RAW / "fred_raw.csv"


# ---------------------------------------------------------------------------
# 1. Yahoo Finance
# ---------------------------------------------------------------------------
def _strip_tz(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


class DataValidationError(RuntimeError):
    """Données téléchargées incomplètes : elles ne doivent PAS remplacer le cache."""


def validate_prices(close: pd.DataFrame, tickers=TICKERS, market: str = "SPY",
                    min_coverage: float = 0.95, recent_days: int = 10) -> None:
    """Vérifie qu'un téléchargement Yahoo est complet avant toute écriture.

    Lève DataValidationError si : le marché de référence (SPY) est absent ou vide,
    moins de `min_coverage` des actifs ont des cours sur les `recent_days` derniers jours
    de bourse, ou l'historique est anormalement court.
    """
    if close is None or close.empty:
        raise DataValidationError("aucune donnée reçue de Yahoo Finance")
    if market not in close.columns or close[market].notna().sum() == 0:
        raise DataValidationError(f"le marché de référence {market} est absent ou vide "
                                  "(probable limitation de débit de Yahoo)")
    spy_dates = close.index[close[market].notna()]
    if len(spy_dates) < 1000:
        raise DataValidationError(f"historique trop court pour {market} ({len(spy_dates)} jours)")
    recent = close.loc[spy_dates[-recent_days:]]
    ok = [t for t in tickers if t in recent.columns and recent[t].notna().any()]
    missing = [t for t in tickers if t not in ok]
    if len(ok) < min_coverage * len(tickers):
        raise DataValidationError(f"{len(missing)} actifs sur {len(tickers)} sans cours récents : "
                                  + ", ".join(missing))


def download_yahoo(tickers=TICKERS, start=START_DATE, end=END_DATE,
                   force: bool = False, retries: int = 3, save: bool = True):
    """Retourne (close, volume) : DataFrames date x ticker des cours ajustés.

    Le cache data/raw/ n'est remplacé QUE si le nouveau téléchargement passe
    validate_prices(). Avec save=False, rien n'est écrit (l'appelant décide).
    """
    if YAHOO_CLOSE_FILE.exists() and not force:
        close = pd.read_csv(YAHOO_CLOSE_FILE, index_col=0, parse_dates=True)
        volume = pd.read_csv(YAHOO_VOLUME_FILE, index_col=0, parse_dates=True)
        print(f"[Yahoo] cache chargé : {close.shape[0]} dates x {close.shape[1]} actifs")
        return close, volume

    import yfinance as yf

    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                              progress=False, group_by="column", threads=False)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # erreurs réseau / limitation de débit
            print(f"[Yahoo] tentative {attempt} échouée : {exc}")
        time.sleep(5 * attempt)
    if raw is None or raw.empty:
        raise DataValidationError("Téléchargement Yahoo Finance impossible. Vérifiez la connexion "
                                  "ou réessayez plus tard (limitation de débit).")

    if isinstance(raw.columns, pd.MultiIndex):
        close, volume = raw["Close"].copy(), raw["Volume"].copy()
    else:  # un seul ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    close.index, volume.index = _strip_tz(close.index), _strip_tz(volume.index)

    # Re-téléchargement individuel (espacé) des tickers manquants
    for t in tickers:
        if t not in close.columns or close[t].isna().all():
            print(f"[Yahoo] nouvel essai individuel pour {t}")
            time.sleep(2)
            try:
                h = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                h.index = _strip_tz(h.index)
                close[t], volume[t] = h["Close"], h["Volume"]
            except Exception as exc:
                print(f"[Yahoo] {t} indisponible : {exc}")

    close = close[[t for t in tickers if t in close.columns]].sort_index()
    volume = volume[[t for t in tickers if t in volume.columns]].sort_index()
    validate_prices(close, tickers)          # lève une erreur AVANT toute écriture
    if save:
        _atomic_to_csv(close, YAHOO_CLOSE_FILE)
        _atomic_to_csv(volume, YAHOO_VOLUME_FILE)
        print(f"[Yahoo] {close.shape[0]} dates x {close.shape[1]} actifs -> {YAHOO_CLOSE_FILE}")
    return close, volume


def _atomic_to_csv(df: pd.DataFrame, path, **kwargs) -> None:
    """Écrit d'abord dans un fichier temporaire, puis le renomme : un fichier existant
    n'est jamais laissé à moitié écrit en cas d'erreur."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Mise à jour RAPIDE (incrémentale) – utilisée par l'application
# ---------------------------------------------------------------------------
def _run_with_deadline(fn, timeout: float, what: str):
    """Exécute fn() dans un thread et abandonne l'attente après `timeout` secondes."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except FutTimeout:
        raise DataValidationError(f"{what} : pas de réponse en {timeout:.0f} s "
                                  "(serveur lent ou limitation de débit), réessayez plus tard") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _splice(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Raccorde les nouveaux cours ajustés à l'historique en cache.

    Les cours « ajustés » de Yahoo sont recalculés après chaque dividende : on remet
    l'ancien historique à l'échelle du nouveau, actif par actif, grâce aux jours communs
    (méthode standard de raccordement). Les actifs absents du nouveau téléchargement
    gardent leur historique."""
    first_new = new.index.min()
    out = old.loc[old.index < first_new].copy()
    for t in new.columns:
        if t not in out.columns or new[t].notna().sum() == 0:
            continue
        common = old[t].dropna().index.intersection(new[t].dropna().index)
        if len(common):
            ratio = float((new.loc[common, t] / old.loc[common, t]).median())
            if pd.notna(ratio) and ratio > 0:
                out[t] = out[t] * ratio
    merged = pd.concat([out, new.reindex(columns=old.columns)])
    # actifs non reçus cette fois : on conserve leurs anciens cours récents
    tail_old = old.loc[old.index >= first_new]
    merged = merged.combine_first(tail_old)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def update_yahoo_incremental(tickers=TICKERS, overlap_days: int = 15, timeout: float = 20):
    """Télécharge uniquement les derniers jours (depuis la dernière date en cache moins
    `overlap_days`) en UNE requête groupée, avec une limite de temps stricte.
    Rien n'est écrit : retourne (close, volume) complets et validés."""
    close_old = pd.read_csv(YAHOO_CLOSE_FILE, index_col=0, parse_dates=True)
    vol_old = pd.read_csv(YAHOO_VOLUME_FILE, index_col=0, parse_dates=True)
    start = (close_old.index.max() - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")

    def _dl():
        import yfinance as yf
        return yf.download(list(tickers), start=start, auto_adjust=True, progress=False,
                           group_by="column", threads=True, timeout=10)

    raw = _run_with_deadline(_dl, timeout, "Yahoo Finance")
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise DataValidationError("Yahoo Finance n'a renvoyé aucune donnée (limitation de débit probable)")
    close_new, vol_new = raw["Close"].copy(), raw["Volume"].copy()
    close_new.index, vol_new.index = _strip_tz(close_new.index), _strip_tz(vol_new.index)
    got = [t for t in tickers if t in close_new.columns and close_new[t].notna().any()]
    if "SPY" not in got:
        raise DataValidationError("SPY absent de la réponse de Yahoo (limitation de débit probable)")
    if len(got) < 0.95 * len(tickers):
        missing = [t for t in tickers if t not in got]
        raise DataValidationError(f"{len(missing)} actifs non reçus : " + ", ".join(missing))
    close = _splice(close_old, close_new[got])
    volume = vol_old.combine_first(vol_new[got]) if not vol_new.empty else vol_old
    volume.update(vol_new[got])
    validate_prices(close, tickers)
    return close, volume


def update_fred_incremental(lookback_days: int = 400, timeout: float = 15) -> pd.DataFrame:
    """Met à jour les séries FRED en parallèle (derniers mois seulement), limite de temps
    stricte. Une série qui ne répond pas garde ses anciennes valeurs (les variables macro
    sont de toute façon décalées à leur date de publication)."""
    from concurrent.futures import ThreadPoolExecutor, wait
    fred_old = pd.read_csv(FRED_FILE, index_col=0, parse_dates=True)
    start = (fred_old.index.max() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    ids = list(FRED_DAILY) + list(FRED_MONTHLY)
    ex = ThreadPoolExecutor(max_workers=len(ids))
    futs = {ex.submit(_fred_series, sid, start, 10): sid for sid in ids}
    done, _ = wait(futs, timeout=timeout)
    ex.shutdown(wait=False, cancel_futures=True)
    new = [f.result() for f in done if f.exception() is None]
    if not new:
        raise DataValidationError("FRED ne répond pas, réessayez plus tard")
    fresh = pd.concat(new, axis=1)
    fred = fresh.combine_first(fred_old)
    fred.update(fresh)
    return fred.sort_index()


# ---------------------------------------------------------------------------
# 2. FRED
# ---------------------------------------------------------------------------
def _fred_series(series_id: str, start: str, timeout: float = 60) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s


def download_fred(start=START_DATE, force: bool = False, save: bool = True) -> pd.DataFrame:
    """Retourne un DataFrame (date x série) aux fréquences d'origine (jour/mois)."""
    if FRED_FILE.exists() and not force:
        fred = pd.read_csv(FRED_FILE, index_col=0, parse_dates=True)
        print(f"[FRED] cache chargé : {fred.shape[1]} séries")
        return fred
    # on remonte 2 ans plus tôt pour pouvoir calculer les variations annuelles
    start_ext = str(pd.Timestamp(start) - pd.DateOffset(years=2))[:10]
    series = []
    for sid in list(FRED_DAILY) + list(FRED_MONTHLY):
        for attempt in range(3):
            try:
                series.append(_fred_series(sid, start_ext))
                print(f"[FRED] {sid} OK")
                break
            except Exception as exc:
                print(f"[FRED] {sid} tentative {attempt + 1} échouée : {exc}")
                time.sleep(3)
    if not series:
        raise DataValidationError("aucune série FRED téléchargée")
    fred = pd.concat(series, axis=1).sort_index()
    missing = [sid for sid in list(FRED_DAILY) + list(FRED_MONTHLY)
               if sid not in fred or fred[sid].notna().sum() == 0]
    if missing:
        raise DataValidationError("séries FRED manquantes : " + ", ".join(missing))
    if save:
        _atomic_to_csv(fred, FRED_FILE)
    return fred


# ---------------------------------------------------------------------------
# 3. Survey of Consumer Finances
# ---------------------------------------------------------------------------
def download_scf(force: bool = False, year: int = SCF_YEAR) -> pd.DataFrame:
    """Fichier « Summary Extract Public Data » du SCF (format CSV dans un zip).
    `year` permet de télécharger une autre vague (ex. 2019 pour la validation externe)."""
    path = DATA_RAW / f"scfp{year}.csv"
    url = f"https://www.federalreserve.gov/econres/files/scfp{year}excel.zip"
    if path.exists() and not force:
        scf = pd.read_csv(path)
        print(f"[SCF {year}] cache chargé : {scf.shape}")
        return scf
    print(f"[SCF {year}] téléchargement de {url} (≈ 30 Mo)…")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=300)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Téléchargement SCF impossible. Téléchargez manuellement le fichier "
            "« Summary Extract Public Data – Excel/CSV » sur "
            "https://www.federalreserve.gov/econres/scfindex.htm, "
            f"puis enregistrez le CSV sous {path}") from exc
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
        with zf.open(member) as fh:
            scf = pd.read_csv(fh)
    scf.columns = scf.columns.str.upper()
    scf.to_csv(path, index=False)
    print(f"[SCF {year}] {scf.shape} -> {path}")
    return scf


def download_all(force: bool = False):
    close, volume = download_yahoo(force=force)
    fred = download_fred(force=force)
    scf = download_scf(force=force)
    download_scf(force=force, year=SCF_EXTERNAL_YEAR)   # validation externe de l'agent Profil
    return close, volume, fred, scf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Téléchargement des données réelles")
    parser.add_argument("--force", action="store_true", help="ignorer le cache")
    args = parser.parse_args()
    download_all(force=args.force)
