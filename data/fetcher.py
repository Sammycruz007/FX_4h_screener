"""
data/fetcher.py
---------------
Data fetching layer for the FX Scanner pipeline.

LOGICAL FLOW:
─────────────
UNIVERSE SOURCING (fixed, not discovered):
   Unlike the stock project (NASDAQ FTP scrape of ~6,000 tickers → Stage 1
   filter → ~1,500), forex has a small, fixed universe: 28 pairs built from
   the 8 major currencies (7 majors × 8 currencies, standard FX convention).
   There is no filtering funnel — every pair in config.yaml's
   `universe.pairs` gets fetched and scanned directly. The universe comes
   straight from config, not from any external source.

FETCH INTERVAL (Weekly):
   We fetch native Weekly (1wk) data from yfinance, NOT a resample of a
   finer interval. This is a deliberate switch away from this project's
   earlier 1H-fetched-and-resampled-to-4H design — the model's target
   horizon moved to "next 2 weekly candles," so the natural fetch unit
   moved with it. Like Daily, Weekly data has no yfinance ceiling the way
   1H data does (that ceiling was the whole reason the old design fetched
   1H and resampled rather than just asking yfinance directly) — a single
   request can pull decades of weekly bars.

   NO RESAMPLING, NO SESSION-ANCHORING: the old 4H design needed
   session-anchored resampling specifically to handle a partial Sunday-
   open 1H candle cleanly rolling up into 4H buckets. Native weekly bars
   from yfinance don't have an equivalent problem — a week is a week,
   yfinance handles the bucket boundaries itself, and there is no
   intermediate interval being rolled up here at all.

WHAT'S DROPPED FROM THE STOCK PROJECT (deliberately, not oversight):
   - NASDAQ FTP universe discovery         → fixed 28-pair list from config
   - Stage 1 filter (price/volume)         → doesn't map to FX; no funnel needed
   - Sector metadata fetch                 → currencies have no sectors
   - `filters` / `indices` / `sectors` config sections → removed upstream

SMART FETCH (unchanged in shape from the stock project):
   First run  → full HISTORICAL_DAYS of weekly history per pair (config
                now sized for a long weekly lookback — e.g. 25 years —
                since weekly bars carry no yfinance-imposed ceiling and
                the basket models want a long, multi-regime window)
   Later runs → incremental INCREMENTAL_DAYS fetch for pairs already tracked

PARALLEL BATCHING (unchanged in shape, smaller in scale):
   Downloads happen in parallel batches (config: batch_size × max_workers).
   With only 28 pairs total this typically means a single batch, but the
   batching/threading code is kept identical to the stock project for
   consistency and in case the universe list grows.

MACRO DRIVERS (added alongside the basket redesign):
   Alongside the 28 FX pairs, this module also fetches a small set of
   external macro drivers (DXY, gold, US 10Y yield, WTI crude, VIX) —
   see config.yaml's `macro` section. These are fetched and stored
   through the identical smart_fetch/write_raw_prices path as FX
   pairs, just keyed by a macro symbol name (e.g. "DXY") instead of a
   6-char FX pair, and kept non-fatal in run_data_pipeline so a
   macro-source hiccup never blocks the core FX fetch.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import os
import yaml

_is_cloud = bool(os.getenv("SUPABASE_DB_URL"))

if _is_cloud:
    from data.database_cloud import (
        get_last_fetch_dates_bulk,
        write_last_fetch_dates,
    )
    # Cloud has no per-pair raw_prices table — prices live in Storage
    # (see data/storage_cloud.py). write_raw_prices / get_last_fetch_date
    # below are the LOCAL SQLite versions, kept for the local dev path.
    from data.database import write_raw_prices, get_last_fetch_date
else:
    from data.database import (
        write_raw_prices,
        get_last_fetch_date,
    )
from utils.logging import get_fetcher_logger
from utils.error_handler import (
    retry,
    graceful,
    validate_dataframe,
    DataFetchError,
    DataValidationError,
)

from utils.yf_session import YF_SESSION

from curl_cffi import requests as curl_requests

logger = get_fetcher_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
FETCHER_CFG  = config["fetcher"]
UNIVERSE_CFG = config["universe"]
MACRO_CFG    = config["macro"]

BATCH_SIZE       = FETCHER_CFG["batch_size"]
MAX_WORKERS      = FETCHER_CFG["max_workers"]
RETRY_ATTEMPTS   = FETCHER_CFG["retry_attempts"]
RETRY_DELAY      = FETCHER_CFG["retry_delay_seconds"]
HISTORICAL_DAYS  = FETCHER_CFG["historical_days"]
INCREMENTAL_DAYS = FETCHER_CFG["incremental_days"]
FETCH_INTERVAL   = FETCHER_CFG["fetch_interval"]        # "1wk" — native weekly bars

MACRO_DRIVERS    = MACRO_CFG["drivers"]   # {"DXY": {"yf_ticker": ..., "storage_symbol": ...}, ...}


# =============================================================================
# FIXED FX UNIVERSE
# =============================================================================

def get_full_universe() -> list[str]:
    """
    Return the fixed FX pair universe from config, converted to yfinance
    ticker format.

    Returns:
        List of yfinance-format FX tickers, e.g. ["EURUSD=X", "GBPUSD=X", ...]
    """
    pairs = UNIVERSE_CFG["pairs"]
    tickers = [f"{pair}=X" for pair in pairs]

    logger.info(f"FX universe loaded from config | {len(tickers)} pairs")
    return tickers


def _strip_yf_suffix(ticker: str) -> str:
    """Convert a yfinance FX ticker back to the plain pair name.
    'EURUSD=X' -> 'EURUSD'."""
    return ticker.replace("=X", "")


# =============================================================================
# MACRO DRIVER UNIVERSE (DXY, gold, US10Y, WTI, VIX)
# =============================================================================
# Fetched and stored through the exact same path as FX pairs — same
# write_raw_prices/read_price_history calls, just with a "pair name"
# that's a macro symbol (e.g. "DXY") instead of a 6-char FX pair, and
# a yfinance ticker that isn't a plain "XXXYYY=X" FX ticker. Added
# because the EURUSD-only reference project's edge came heavily from
# these external, non-price-derived drivers — the original basket
# design had none of them (see engines/macro.py for the feature side).

# yfinance ticker -> the storage_symbol we file it under (reverse of
# MACRO_DRIVERS, built once at import time).
_YF_TICKER_TO_STORAGE_SYMBOL = {
    driver_cfg["yf_ticker"]: driver_cfg["storage_symbol"]
    for driver_cfg in MACRO_DRIVERS.values()
}


def get_macro_universe() -> list[str]:
    """
    Return the macro driver universe's yfinance tickers, e.g.
    ["DX-Y.NYB", "GC=F", "^TNX", "CL=F", "^VIX"] (deduped — CHF and
    JPY both map to VIX, so this only fetches it once).
    """
    tickers = sorted(set(_YF_TICKER_TO_STORAGE_SYMBOL.keys()))
    logger.info(f"Macro driver universe loaded from config | {len(tickers)} symbols")
    return tickers


def _macro_storage_symbol(ticker: str) -> str:
    """Map a macro yfinance ticker back to its storage symbol name.
    'DX-Y.NYB' -> 'DXY'. Unlike _strip_yf_suffix (FX pairs), macro
    tickers have irregular yfinance formats (^VIX, GC=F, DX-Y.NYB),
    so this is a config-driven lookup rather than string slicing."""
    return _YF_TICKER_TO_STORAGE_SYMBOL[ticker]


# =============================================================================
# SINGLE PAIR OHLC FETCH (Weekly)
# =============================================================================

@retry(
    attempts   = RETRY_ATTEMPTS,
    delay_seconds = RETRY_DELAY,
    exceptions = (DataFetchError, Exception),
)
def fetch_single_pair(
    ticker     : str,
    start_date : str,
    end_date   : str,
    min_rows   : int = 50,
    pair_name_resolver = _strip_yf_suffix,
) -> Optional[pd.DataFrame]:
    """
    Fetch Weekly OHLC data for a single FX pair (or macro driver) via
    yfinance.

    NOTE ON VOLUME: FX is decentralized (OTC) — there is no real,
    centralized traded volume the way exchanges provide for stocks. We
    deliberately do not fetch or keep a volume column at all here.
    Macro drivers (DXY, gold futures, etc.) DO have real volume, but
    we drop it here too for schema consistency with the FX side —
    volume isn't used by any engine in this project.

    Args:
        ticker    : yfinance ticker, e.g. "EURUSD=X" (FX) or "^VIX" (macro)
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD
        min_rows  : Minimum row count to pass validation — MUST reflect
                    whether this call is part of a full or incremental
                    fetch (see validate_dataframe's docstring in
                    utils/error_handler.py: using the same floor for
                    both fetch modes silently rejects every legitimate
                    incremental fetch). Callers (smart_fetch) pass the
                    correct value per mode.
        pair_name_resolver: Function mapping the yfinance ticker to
                    the name stored in the 'pair' column. Defaults to
                    the FX '=X'-suffix stripper; macro callers pass
                    _macro_storage_symbol instead so "^VIX" is stored
                    as "VIX", not the raw ticker string.

    Returns:
        Clean Weekly OHLC DataFrame (columns: pair, datetime, open,
        high, low, close) or None if fetch/validation fails
    """
    local_session = curl_requests.Session(impersonate="chrome")

    try:
        raw = yf.download(
            ticker,
            start       = start_date,
            end         = end_date,
            interval    = FETCH_INTERVAL,   # "1wk" — native weekly bars, no resampling
            auto_adjust = True,
            progress    = False,
            threads     = False,
            session     = local_session,
        )

        local_session.close()

        if raw.empty:
            logger.warning(f"{ticker} | yfinance returned empty DataFrame")
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]

        required = ["open", "high", "low", "close"]
        missing  = [c for c in required if c not in raw.columns]
        if missing:
            logger.warning(f"{ticker} | Missing columns: {missing}")
            return None

        df = raw[required].copy()

        # Force UTC-aware index to standardize datetimes
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        pair_name       = pair_name_resolver(ticker)
        df["pair"]      = pair_name
        df["datetime"]  = df.index
        df              = df.reset_index(drop=True)

        if not validate_dataframe(df, pair_name, required, min_rows=min_rows):
            return None

        df = df.dropna(subset=required)
        df = df[df["close"] > 0]

        logger.debug(
            f"{ticker} | {len(df)} weekly rows | "
            f"{start_date} to {end_date}"
        )
        return df

    except Exception as e:
        raise DataFetchError(f"{ticker} fetch failed: {e}") from e


# =============================================================================
# BATCH PARALLEL FETCH
# =============================================================================

def fetch_batch(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
    min_rows   : int = 50,
    pair_name_resolver = _strip_yf_suffix,
) -> pd.DataFrame:
    """
    Fetch Weekly OHLC data for a batch of pairs in parallel.

    Args:
        tickers   : List of yfinance tickers (FX or macro)
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD
        min_rows  : Passed through to fetch_single_pair's validation
        pair_name_resolver: Passed through to fetch_single_pair — see
                    its docstring. Defaults to the FX resolver.

    Returns:
        Combined DataFrame for all successful pairs in batch
    """
    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_single_pair, ticker, start_date, end_date, min_rows,
                pair_name_resolver,
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    failed.append(ticker)
            except Exception as e:
                logger.warning(f"{ticker} | Batch fetch failed: {e}")
                failed.append(ticker)

    if failed:
        logger.warning(f"Batch: {len(failed)} pairs failed: {failed}")

    if not results:
        logger.warning("Batch returned no data")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# FULL UNIVERSE FETCH — orchestrates batching
# =============================================================================

def fetch_universe(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
    min_rows   : int = 50,
    pair_name_resolver = _strip_yf_suffix,
) -> pd.DataFrame:
    """
    Fetch Weekly OHLC data for the entire FX universe (or macro driver
    set) in batches.

    Args:
        tickers   : Full list of yfinance tickers (FX or macro)
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD
        min_rows  : Passed through to fetch_batch/fetch_single_pair
        pair_name_resolver: Passed through to fetch_batch — see
                    fetch_single_pair's docstring. Defaults to the FX
                    resolver.

    Returns:
        Combined DataFrame for all pairs
    """
    total   = len(tickers)
    batches = [
        tickers[i : i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]
    all_data = []

    logger.info(
        f"Fetching {total} symbols | "
        f"{len(batches)} batch(es) of up to {BATCH_SIZE} | "
        f"Period: {start_date} to {end_date} | "
        f"Interval: {FETCH_INTERVAL} | min_rows: {min_rows}"
    )

    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)} | {len(batch)} symbols")
        batch_df = fetch_batch(batch, start_date, end_date, min_rows, pair_name_resolver)

        if not batch_df.empty:
            all_data.append(batch_df)

        if i < len(batches):
            _time.sleep(3)

    if not all_data:
        logger.error("fetch_universe: No data returned for any pair")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"fetch_universe complete | "
        f"Total weekly rows: {len(combined)} | "
        f"Pairs with data: {combined['pair'].nunique()}"
    )
    return combined


# =============================================================================
# SMART FETCH — full vs incremental per pair
# =============================================================================

def smart_fetch(
    tickers: list[str],
    pair_name_resolver = _strip_yf_suffix,
) -> pd.DataFrame:
    """
    Intelligently decide full vs incremental fetch per pair, in both
    local (SQLite) and cloud (Supabase) modes.

    Args:
        tickers: yfinance tickers to fetch (FX pairs or macro drivers)
        pair_name_resolver: Maps each ticker to its stored 'pair' name.
                    Defaults to the FX '=X'-suffix stripper; macro
                    callers (smart_fetch_macro) pass
                    _macro_storage_symbol instead.

    Returns:
        Combined DataFrame of all new data fetched
    """
    today    = datetime.now(timezone.utc)
    end_date = today.strftime("%Y-%m-%d")

    full_tickers        = []
    incremental_tickers = []

    if _is_cloud:
        last_fetch_map = get_last_fetch_dates_bulk()
    else:
        last_fetch_map = None

    for ticker in tickers:
        pair_name = pair_name_resolver(ticker)

        if _is_cloud:
            last_date = last_fetch_map.get(pair_name)
        else:
            last_date = get_last_fetch_date(pair_name)

        if last_date is None:
            full_tickers.append(ticker)
        else:
            incremental_tickers.append(ticker)

    logger.info(
        f"smart_fetch | Cloud mode: {_is_cloud} | "
        f"Full fetch needed: {len(full_tickers)} symbols | "
        f"Incremental: {len(incremental_tickers)} symbols"
    )

    all_data = []

    FULL_FETCH_MIN_ROWS = 50

    if full_tickers:
        start_full = (
            today - timedelta(days=HISTORICAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Full fetch: {start_full} to {end_date}")
        df_full = fetch_universe(
            full_tickers, start_full, end_date,
            min_rows=FULL_FETCH_MIN_ROWS,
            pair_name_resolver=pair_name_resolver,
        )

        if not df_full.empty:
            all_data.append(df_full)

    INCREMENTAL_FETCH_MIN_ROWS = 1

    if incremental_tickers:
        start_incr = (
            today - timedelta(days=INCREMENTAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Incremental fetch: {start_incr} to {end_date}")
        df_incr = fetch_universe(
            incremental_tickers, start_incr, end_date,
            min_rows=INCREMENTAL_FETCH_MIN_ROWS,
            pair_name_resolver=pair_name_resolver,
        )

        if not df_incr.empty:
            all_data.append(df_incr)

    if not all_data:
        logger.warning("smart_fetch: No new data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    if _is_cloud and not combined.empty:
        max_dates = (
            combined.groupby("pair")["datetime"]
            .max()
            .astype(str)
            .to_dict()
        )
        write_last_fetch_dates(max_dates)

    logger.info(f"smart_fetch complete | Total weekly rows: {len(combined)}")
    return combined


def smart_fetch_macro() -> pd.DataFrame:
    """
    Fetch all configured macro drivers (DXY, gold, US10Y, WTI, VIX),
    full-vs-incremental, through the exact same smart_fetch path as FX
    pairs — just with the macro ticker list and the macro pair-name
    resolver swapped in. Storage is identical (write_raw_prices), so
    downstream engines/features.py reads macro history the same way
    it reads any FX pair's history.
    """
    macro_tickers = get_macro_universe()
    return smart_fetch(macro_tickers, pair_name_resolver=_macro_storage_symbol)


# =============================================================================
# SHAPE ADAPTER — long-form fetch output -> per-pair dict for engines
# =============================================================================

def to_pair_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert a long-form OHLC DataFrame into dict[pair_name -> DataFrame].

    Returns:
        Dict mapping pair name -> that pair's OHLC DataFrame, sorted
        datetime ascending, 'pair' column dropped
    """
    if df.empty:
        logger.warning("to_pair_dict: input DataFrame is empty, returning {}")
        return {}

    if "pair" not in df.columns:
        raise DataValidationError(
            "to_pair_dict: input DataFrame has no 'pair' column"
        )

    pair_dict = {}
    for pair, group in df.groupby("pair"):
        pair_dict[pair] = (
            group.sort_values("datetime")
            .drop(columns=["pair"])
            .reset_index(drop=True)
        )

    logger.debug(f"to_pair_dict: reshaped {len(df)} rows into {len(pair_dict)} pairs")
    return pair_dict


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_data_pipeline() -> dict:
    """
    Main entry point for the FX data pipeline.

    Returns:
        Summary dict with counts for scheduler logging and monitoring
    """
    logger.info("=" * 60)
    logger.info("FX DATA PIPELINE STARTED")
    logger.info("=" * 60)

    summary = {}

    try:
        tickers                  = get_full_universe()
        summary["universe_size"] = len(tickers)

        df                      = smart_fetch(tickers)
        summary["rows_fetched"] = len(df)

        if df.empty:
            logger.error("Data pipeline: No data fetched. Aborting.")
            return summary

        rows_written            = write_raw_prices(df)
        summary["rows_written"] = rows_written

        # Macro drivers (DXY, gold, US10Y, WTI, VIX) — fetched and
        # stored through the same write_raw_prices path, but kept
        # non-fatal: a macro-source hiccup (e.g. yfinance rate-limits
        # ^VIX) should never block writing the FX prices the rest of
        # the pipeline depends on.
        try:
            macro_df = smart_fetch_macro()
            summary["macro_rows_fetched"] = len(macro_df)

            if not macro_df.empty:
                macro_rows_written = write_raw_prices(macro_df)
                summary["macro_rows_written"] = macro_rows_written
            else:
                logger.warning("Macro fetch returned no data this run")
        except Exception as e:
            logger.error(f"Macro fetch failed (non-fatal, FX data still written): {e}", exc_info=True)
            summary["macro_fetch_error"] = str(e)

        logger.info("=" * 60)
        logger.info(f"FX DATA PIPELINE COMPLETE | Summary: {summary}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.critical(f"FX data pipeline failed: {e}", exc_info=True)
        raise
