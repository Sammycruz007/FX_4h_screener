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

FETCH INTERVAL (Daily):
   We fetch native Daily (1D) data from yfinance. Unlike 1H data, daily data 
   has no 730-day yfinance ceiling, allowing us to seamlessly fetch 8+ years 
   of historical data in a single request. 

WHAT'S DROPPED FROM THE STOCK PROJECT (deliberately, not oversight):
   - NASDAQ FTP universe discovery         → fixed 28-pair list from config
   - Stage 1 filter (price/volume)         → doesn't map to FX; no funnel needed
   - Sector metadata fetch                 → currencies have no sectors
   - `filters` / `indices` / `sectors` config sections → removed upstream

SMART FETCH (unchanged in shape from the stock project):
   First run  → full HISTORICAL_DAYS of 1D history per pair
   Later runs → incremental INCREMENTAL_DAYS fetch for pairs already tracked

PARALLEL BATCHING (unchanged in shape, smaller in scale):
   Downloads happen in parallel batches (config: batch_size × max_workers).
   With only 28 pairs total this typically means a single batch, but the
   batching/threading code is kept identical to the stock project for
   consistency and in case the universe list grows.
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

BATCH_SIZE       = FETCHER_CFG["batch_size"]
MAX_WORKERS      = FETCHER_CFG["max_workers"]
RETRY_ATTEMPTS   = FETCHER_CFG["retry_attempts"]
RETRY_DELAY      = FETCHER_CFG["retry_delay_seconds"]
HISTORICAL_DAYS  = FETCHER_CFG["historical_days"]      
INCREMENTAL_DAYS = FETCHER_CFG["incremental_days"]
FETCH_INTERVAL   = FETCHER_CFG["fetch_interval"]        # "1d" — native daily bars


# =============================================================================
# FIXED FX UNIVERSE
# Unlike the stock project's NASDAQ FTP scrape, the FX universe is a small,
# fixed list defined directly in config.yaml — no discovery, no filtering
# funnel. yfinance ticker format for FX is "EURUSD=X" (base+quote+"=X").
# =============================================================================

def get_full_universe() -> list[str]:
    """
    Return the fixed FX pair universe from config, converted to yfinance
    ticker format.

    FLOW:
    1. Read `universe.pairs` from config.yaml (28 pairs, e.g. "EURUSD")
    2. Append yfinance's FX suffix "=X" to each (e.g. "EURUSD=X")
    3. Return as a stable, ordered list (already deduplicated at config
       validation time — no set()/sort() reshuffling needed here, unlike
       the stock project's dynamically-discovered universe)

    Returns:
        List of yfinance-format FX tickers, e.g. ["EURUSD=X", "GBPUSD=X", ...]
    """
    pairs = UNIVERSE_CFG["pairs"]
    tickers = [f"{pair}=X" for pair in pairs]

    logger.info(f"FX universe loaded from config | {len(tickers)} pairs")
    return tickers


def _strip_yf_suffix(ticker: str) -> str:
    """Convert a yfinance FX ticker back to the plain pair name.
    'EURUSD=X' -> 'EURUSD'. Used when writing to storage/DB, where we
    key on the plain pair name (matches config.yaml's `universe.pairs`
    and `universe.currencies`), not the yfinance-specific suffix.
    """
    return ticker.replace("=X", "")


# =============================================================================
# SINGLE PAIR OHLC FETCH (Daily)
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
) -> Optional[pd.DataFrame]:
    """
    Fetch Daily OHLC data for a single FX pair via yfinance.

    FLOW:
    1. Download Daily OHLC from yfinance (no volume — see note below)
    2. Flatten MultiIndex columns if present
    3. Standardise column names to lowercase
    4. Force UTC timezone alignment on the index
    5. Add pair and datetime columns
    6. Validate data quality
    7. Drop nulls and non-positive prices
    8. Return clean, Daily DataFrame

    NOTE ON VOLUME: FX is decentralized (OTC) — there is no real,
    centralized traded volume the way exchanges provide for stocks. What
    yfinance reports for FX tickers is synthetic "tick volume" (count of
    price changes, not actual transaction size). We deliberately do not
    fetch or keep a volume column at all here — training on tick-volume-
    as-if-it-were-real-volume would be learning from noise dressed up as
    signal. This is a hard drop, not an oversight.

    Args:
        ticker    : yfinance FX ticker, e.g. "EURUSD=X"
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD
        min_rows  : Minimum row count to pass validation —
                    MUST reflect whether this call is part of a full or
                    incremental fetch.

    Returns:
        Clean Daily OHLC DataFrame (columns: pair, datetime, open, high,
        low, close) or None if fetch/validation fails
    """
    local_session = curl_requests.Session(impersonate="chrome")

    try:
        raw = yf.download(
            ticker,
            start       = start_date,
            end         = end_date,
            interval    = FETCH_INTERVAL,   
            auto_adjust = True,
            progress    = False,
            threads     = False,
            session     = local_session,
        )

        local_session.close()

        if raw.empty:
            logger.warning(f"{ticker} | yfinance returned empty DataFrame")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise column names
        raw.columns = [c.lower() for c in raw.columns]

        # Keep only OHLC — no volume column for FX (see docstring note)
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

        pair_name = _strip_yf_suffix(ticker)
        df["pair"]     = pair_name
        df["datetime"] = df.index
        df             = df.reset_index(drop=True)

        if not validate_dataframe(df, pair_name, required, min_rows=min_rows):
            return None

        # Drop rows with null OHLC or non-positive close
        df = df.dropna(subset=required)
        df = df[df["close"] > 0]

        logger.debug(
            f"{ticker} | {len(df)} Daily rows | "
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
) -> pd.DataFrame:
    """
    Fetch Daily OHLC data for a batch of pairs in parallel.

    FLOW:
    1. Submit all pairs to ThreadPoolExecutor simultaneously
    2. Collect results as they complete
    3. Log failed pairs
    4. Combine successful results into single DataFrame

    Args:
        tickers   : List of yfinance FX tickers
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD
        min_rows  : Passed through to fetch_single_pair's validation —
                    MUST reflect full vs incremental fetch mode (see
                    fetch_single_pair's docstring)

    Returns:
        Combined DataFrame for all successful pairs in batch
    """
    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_single_pair, ticker, start_date, end_date, min_rows): ticker
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
) -> pd.DataFrame:
    """
    Fetch Daily OHLC data for the entire FX universe in batches.

    FLOW:
    1. Split full ticker list into batches of BATCH_SIZE
       (with only 28 pairs total, this is typically a single batch —
       kept batched for consistency with the stock project and to scale
       gracefully if the universe list grows)
    2. Process each batch sequentially
       (parallelism happens WITHIN each batch via ThreadPoolExecutor)
    3. Combine all batch results
    4. Log summary statistics

    Args:
        tickers   : Full list of yfinance FX tickers
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD
        min_rows  : Passed through to fetch_batch/fetch_single_pair's
                    validation — MUST reflect full vs incremental fetch
                    mode (see fetch_single_pair's docstring). smart_fetch
                    is the only caller that should decide this value;
                    it calls fetch_universe once per mode with the
                    correct min_rows each time.

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
        f"Fetching {total} FX pairs | "
        f"{len(batches)} batch(es) of up to {BATCH_SIZE} | "
        f"Period: {start_date} to {end_date} | "
        f"Interval: {FETCH_INTERVAL} | "
        f"min_rows: {min_rows}"
    )

    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)} | {len(batch)} pairs")
        batch_df = fetch_batch(batch, start_date, end_date, min_rows)

        if not batch_df.empty:
            all_data.append(batch_df)

        # Pause between batches to avoid Yahoo Finance rate limiting
        if i < len(batches):
            _time.sleep(3)

    if not all_data:
        logger.error("fetch_universe: No data returned for any pair")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"fetch_universe complete | "
        f"Total Daily rows: {len(combined)} | "
        f"Pairs with data: {combined['pair'].nunique()}"
    )
    return combined


# =============================================================================
# SMART FETCH — full vs incremental per pair
# =============================================================================

def smart_fetch(tickers: list[str]) -> pd.DataFrame:
    """
    Intelligently decide full vs incremental fetch per pair, in both
    local (SQLite) and cloud (Supabase) modes.

    DECISION LOGIC per pair:
    - Not previously tracked → full HISTORICAL_DAYS fetch
    - Already tracked        → incremental INCREMENTAL_DAYS fetch (buffer
      window covering weekend gaps and any missed pipeline runs)

    This means:
    - First ever run: downloads massive multi-year historical data.
    - All subsequent runs: only fetch INCREMENTAL_DAYS of daily data per pair.

    Cloud mode tracks last-fetched date per pair via a small Postgres
    table (fetch_tracker), same pattern as the stock project — one bulk
    query for the whole universe, not one query per pair.

    NOTE: last-fetch tracking here is keyed on plain pair name (e.g.
    "EURUSD"), matching what fetch_single_pair writes to the `pair`
    column — not the yfinance "=X" ticker format.

    Args:
        tickers: Full universe ticker list (yfinance format, e.g. "EURUSD=X")

    Returns:
        Combined DataFrame of all new data fetched
    """
    today    = datetime.now(timezone.utc)
    end_date = today.strftime("%Y-%m-%d")

    full_tickers        = []
    incremental_tickers = []

    # ── Determine each pair's last-fetched date ────────────────────────────
    if _is_cloud:
        # ONE bulk query for the whole universe, not one query per pair
        last_fetch_map = get_last_fetch_dates_bulk()
    else:
        last_fetch_map = None  # local path looks up per-pair via SQLite below

    for ticker in tickers:
        pair_name = _strip_yf_suffix(ticker)

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
        f"Full fetch needed: {len(full_tickers)} pairs | "
        f"Incremental: {len(incremental_tickers)} pairs"
    )

    all_data = []

    # ── Full historical fetch ───────────────────────────────────────────────
    FULL_FETCH_MIN_ROWS = 50

    if full_tickers:
        start_full = (
            today - timedelta(days=HISTORICAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Full fetch: {start_full} to {end_date}")
        df_full = fetch_universe(full_tickers, start_full, end_date, min_rows=FULL_FETCH_MIN_ROWS)

        if not df_full.empty:
            all_data.append(df_full)

    # ── Incremental fetch ────────────────────────────────────────────────────
    INCREMENTAL_FETCH_MIN_ROWS = 1

    if incremental_tickers:
        start_incr = (
            today - timedelta(days=INCREMENTAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Incremental fetch: {start_incr} to {end_date}")
        df_incr = fetch_universe(incremental_tickers, start_incr, end_date, min_rows=INCREMENTAL_FETCH_MIN_ROWS)

        if not df_incr.empty:
            all_data.append(df_incr)

    if not all_data:
        logger.warning("smart_fetch: No new data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # ── Update the fetch tracker with what was ACTUALLY fetched ────────────
    if _is_cloud and not combined.empty:
        max_dates = (
            combined.groupby("pair")["datetime"]
            .max()
            .astype(str)
            .to_dict()
        )
        write_last_fetch_dates(max_dates)

    logger.info(f"smart_fetch complete | Total Daily rows: {len(combined)}")
    return combined


# =============================================================================
# SHAPE ADAPTER — long-form fetch output -> per-pair dict for engines
# Every engine (linreg, smc, adx, csi) takes tickers_data as
# dict[pair_name -> DataFrame], matching the stock project's convention.
# fetch_universe()/smart_fetch() return a single long-form DataFrame
# (all pairs stacked, via pd.concat) — this is the same shape the
# original stock fetcher.py produced, and reshaping it into per-ticker
# dicts was always something the orchestrator did upstream of the
# engines, not fetcher.py itself. Added here as a small, clearly-scoped
# utility so that piece of glue code exists in one obvious place rather
# than being silently assumed or re-implemented ad hoc later.
# =============================================================================

def to_pair_dict(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert a long-form OHLC DataFrame (all pairs stacked, one 'pair'
    column) into the dict[pair_name -> DataFrame] shape every engine
    (linreg, smc, adx, csi) expects as its `tickers_data` argument.

    FLOW:
    1. Group by 'pair'
    2. For each pair, sort its rows by datetime ascending (engines
       assume chronological order — e.g. LinReg/ADX read the last N
       rows as "most recent")
    3. Drop the now-redundant 'pair' column from each per-pair frame
       (it's already the dict key)
    4. Return the dict, ready to pass straight into run_csi_engine,
       run_linreg_engine, etc.

    Args:
        df: Long-form DataFrame with columns including 'pair' and
            'datetime', as produced by fetch_universe()/smart_fetch()

    Returns:
        Dict mapping pair name (e.g. 'EURUSD') -> that pair's OHLC
        DataFrame, sorted datetime ascending, 'pair' column dropped
    """
    if df.empty:
        logger.warning("to_pair_dict: input DataFrame is empty, returning {}")
        return {}

    if "pair" not in df.columns:
        raise DataValidationError(
            "to_pair_dict: input DataFrame has no 'pair' column — "
            "expected long-form output from fetch_universe()/smart_fetch()"
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
# Called by the scheduler (see config `scheduler` section — cadence still
# open per the build-order discussion: once daily vs. every H4 close).
# =============================================================================

def run_data_pipeline() -> dict:
    """
    Main entry point for the FX data pipeline.

    FULL FLOW:
    1. Load fixed 28-pair universe from config
    2. Smart fetch OHLC data (full or incremental per pair), fetching Daily
       data directly.
    3. Write raw (Daily) prices to storage

    Unlike the stock project, there is no Stage 1 filter step and no
    sector metadata fetch — both are structurally absent for FX, not
    skipped conditionally. See module docstring.

    Returns:
        Summary dict with counts for scheduler logging and monitoring
    """
    logger.info("=" * 60)
    logger.info("FX DATA PIPELINE STARTED")
    logger.info("=" * 60)

    summary = {}

    try:
        # ── Step 1: Get fixed universe ──────────────────────────────────────
        tickers                  = get_full_universe()
        summary["universe_size"] = len(tickers)

        # ── Step 2: Smart fetch Daily OHLC ──────────────────────────────────
        df                      = smart_fetch(tickers)
        summary["rows_fetched"] = len(df)

        if df.empty:
            logger.error("Data pipeline: No data fetched. Aborting.")
            return summary

        # ── Step 3: Write raw prices ─────────────────────────────────────────
        rows_written            = write_raw_prices(df)
        summary["rows_written"] = rows_written

        logger.info("=" * 60)
        logger.info(f"FX DATA PIPELINE COMPLETE | Summary: {summary}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.critical(f"FX data pipeline failed: {e}", exc_info=True)
        raise

