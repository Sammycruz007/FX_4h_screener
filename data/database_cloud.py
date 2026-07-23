"""
data/database_cloud.py
-----------------------
PostgreSQL version of database.py for cloud deployment (FX Scanner).

Used by:
- run_pipeline_cloud.py  (GitHub Actions, twice daily per
  config.yaml's scheduler.run_times)
- dashboard/app_cloud.py (Streamlit Cloud)

DIFFERENCE FROM database.py:
- Uses psycopg2 (PostgreSQL) instead of sqlite3
- Reads connection string from SUPABASE_DB_URL environment variable
- Only stores RESULT tables (no raw_prices — too large for Supabase
  free tier). Raw 4H OHLC lives in Supabase Storage as Parquet
  snapshots instead (see data/storage_cloud.py), not here.

TABLES STORED IN SUPABASE:
- indicator_results   (engine outputs per pair per 4H-candle datetime)
- scan_results        (ranked candidates per scan run)
- model_metrics       (ML model performance)
- fetch_tracker       (per-pair incremental-fetch bookkeeping)

WHAT'S DROPPED FROM THE STOCK PROJECT'S VERSION OF THIS FILE:
- filtered_universe table + write_filtered_universe/
  read_filtered_universe — no Stage 1 filtering funnel for FX; the
  universe is a fixed 28-pair list from config.yaml, every pair
  scanned directly (see fetcher.py/screener.py's module docstrings)
- ticker_metadata's sector_name/sector_etf columns + write_sector_metadata/
  read_sector_metadata — currencies have no sectors
- gft_watchlist_results table + write_gft_watchlist_results/
  read_latest_gft_watchlist_results — GFT_TICKERS was a 15-stock
  evaluation-account watchlist with no FX equivalent, dropped entirely
  per project decision (see ml/signal_ranker.py's module docstring)
- volume_signal column everywhere it appeared — no real volume data
  exists for FX (decentralized OTC market, see fetcher.py's docstring)

WHAT'S DIFFERENT IN SHAPE, NOT JUST RENAMED:
- ticker -> pair, date -> datetime EVERYWHERE — consistent with every
  other FX file (fetcher.py, engines/csi.py, features.py, screener.py,
  train_models.py)
- scan_results is now keyed on (pair, datetime, direction), NOT
  (pair, scan_date, direction) — the stock project scanned once daily,
  so a DATE was enough to identify a run uniquely. This project scans
  TWICE daily (config.yaml's scheduler.run_times: ["12:00", "20:00"]
  UTC) — keying on date alone would silently collide the 12:00 and
  20:00 runs' candidates into the same row via ON CONFLICT, with the
  second write overwriting the first. datetime (the actual 4H candle
  timestamp being scanned) disambiguates the two runs correctly.
- indicator_results holds ENGINE outputs only (LinReg, SMC, ADX, CSI's
  4 features, candlestick's raw pattern flags) — matching the boundary
  already established in the stock project's version of this table.
  ATR-normalized distances, session flags, and other features.py-time
  derived values are computed fresh at feature-matrix-build time, not
  persisted here, same as dist_to_mean/band_penetration were never
  persisted in the stock version either.
"""

import os
import psycopg2
import psycopg2.extras
import pandas as pd
from contextlib import contextmanager
from typing import Optional
from utils.logging import get_database_logger
from utils.error_handler import DatabaseError

logger = get_database_logger()


# =============================================================================
# CONNECTION
# =============================================================================

def _get_db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise DatabaseError(
            "SUPABASE_DB_URL environment variable not set. "
            "Add it to GitHub Secrets or .streamlit/secrets.toml"
        )
    return url


@contextmanager
def get_connection():
    """Context manager for PostgreSQL connections."""
    conn = None
    try:
        conn = psycopg2.connect(_get_db_url(), sslmode="require", connect_timeout=10)
        yield conn
        conn.commit()
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"PostgreSQL error: {e}") from e
    finally:
        if conn:
            conn.close()


# =============================================================================
# INITIALISE TABLES
# =============================================================================

def initialise_database() -> None:
    """
    Create all result tables in Supabase if they don't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    tables = [
        """
        CREATE TABLE IF NOT EXISTS indicator_results (
            id                   SERIAL PRIMARY KEY,
            pair                 TEXT NOT NULL,
            datetime             TIMESTAMPTZ NOT NULL,
            linreg_value         REAL,
            linreg_slope         REAL,
            linreg_slope_up      INTEGER,
            sd1_upper            REAL,
            sd1_lower            REAL,
            sd2_upper            REAL,
            sd2_lower            REAL,
            sd3_upper            REAL,
            sd3_lower            REAL,
            price_sd_position    REAL,
            smc_structure        TEXT,
            has_valid_zone       INTEGER DEFAULT 0,
            adx_value            REAL,
            plus_di              REAL,
            minus_di             REAL,
            csi_rs               REAL,
            csi_diff_zscore      REAL,
            csi_diff_roc         REAL,
            csi_commodity_bloc   REAL,
            is_hammer            INTEGER DEFAULT 0,
            is_shooting_star     INTEGER DEFAULT 0,
            created_at           TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, datetime)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id             SERIAL PRIMARY KEY,
            pair           TEXT NOT NULL,
            datetime       TIMESTAMPTZ NOT NULL,
            direction      TEXT NOT NULL,
            currency_bloc  TEXT,
            sd_position    REAL,
            has_valid_zone INTEGER DEFAULT 0,
            ml_score       REAL,
            ml_rank        INTEGER,
            created_at     TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, datetime, direction)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id              SERIAL PRIMARY KEY,
            model_name      TEXT NOT NULL,
            train_date      TEXT NOT NULL,
            precision_score REAL,
            recall_score    REAL,
            pr_auc_score    REAL,
            auc_roc_score   REAL,
            n_samples       INTEGER,
            created_at      TIMESTAMP DEFAULT NOW(),
            UNIQUE(model_name, train_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fetch_tracker (
            pair            TEXT PRIMARY KEY,
            last_fetch_date DATE NOT NULL,
            updated_at      TIMESTAMP DEFAULT NOW()
        )
        """,
    ]

    with get_connection() as conn:
        cursor = conn.cursor()
        for ddl in tables:
            cursor.execute(ddl)

    logger.info("Supabase database initialised — all tables ready")


# =============================================================================
# WRITE FUNCTIONS
# =============================================================================

def write_indicator_results(df: pd.DataFrame) -> int:
    """
    Upsert engine-output indicator rows (LinReg, SMC, ADX, CSI,
    candlestick raw flags) — one row per (pair, datetime).

    Columns not present in df default to NULL/0 via .get() with a
    default, so this function tolerates a df that's missing e.g. CSI
    or candlestick columns (partial engine runs) rather than KeyError-
    crashing the whole write.
    """
    if df.empty:
        return 0

    cols = [
        "pair", "datetime", "linreg_value", "linreg_slope", "linreg_slope_up",
        "sd1_upper", "sd1_lower", "sd2_upper", "sd2_lower", "sd3_upper", "sd3_lower",
        "price_sd_position", "smc_structure", "has_valid_zone",
        "adx_value", "plus_di", "minus_di",
        "csi_rs", "csi_diff_zscore", "csi_diff_roc", "csi_commodity_bloc",
        "is_hammer", "is_shooting_star",
    ]

    records = []
    for _, row in df.iterrows():
        record = {c: row.get(c) for c in cols}

        # Integer-typed columns need an explicit default for BOTH a
        # missing column (.get() -> None) AND an actual NaN value in a
        # present column (e.g. a float64 column with some missing
        # entries — a common pandas artifact). NOTE: `x or 0` does NOT
        # catch NaN — NaN is truthy in Python, so `int(nan or 0)` still
        # raises ValueError. pd.isna() is required to catch both cases
        # correctly; this was a real bug caught in testing.
        def _int_default(value, default=0):
            if value is None or pd.isna(value):
                return default
            return int(value)

        record["has_valid_zone"]   = _int_default(record.get("has_valid_zone"))
        record["is_hammer"]        = _int_default(record.get("is_hammer"))
        record["is_shooting_star"] = _int_default(record.get("is_shooting_star"))
        records.append(record)

    sql = """
        INSERT INTO indicator_results (
            pair, datetime, linreg_value, linreg_slope, linreg_slope_up,
            sd1_upper, sd1_lower, sd2_upper, sd2_lower, sd3_upper, sd3_lower,
            price_sd_position, smc_structure, has_valid_zone,
            adx_value, plus_di, minus_di,
            csi_rs, csi_diff_zscore, csi_diff_roc, csi_commodity_bloc,
            is_hammer, is_shooting_star
        ) VALUES (
            %(pair)s, %(datetime)s, %(linreg_value)s, %(linreg_slope)s, %(linreg_slope_up)s,
            %(sd1_upper)s, %(sd1_lower)s, %(sd2_upper)s, %(sd2_lower)s,
            %(sd3_upper)s, %(sd3_lower)s, %(price_sd_position)s,
            %(smc_structure)s, %(has_valid_zone)s,
            %(adx_value)s, %(plus_di)s, %(minus_di)s,
            %(csi_rs)s, %(csi_diff_zscore)s, %(csi_diff_roc)s, %(csi_commodity_bloc)s,
            %(is_hammer)s, %(is_shooting_star)s
        )
        ON CONFLICT (pair, datetime) DO UPDATE SET
            linreg_value       = EXCLUDED.linreg_value,
            linreg_slope       = EXCLUDED.linreg_slope,
            linreg_slope_up    = EXCLUDED.linreg_slope_up,
            sd1_upper          = EXCLUDED.sd1_upper,
            sd1_lower          = EXCLUDED.sd1_lower,
            sd2_upper          = EXCLUDED.sd2_upper,
            sd2_lower          = EXCLUDED.sd2_lower,
            sd3_upper          = EXCLUDED.sd3_upper,
            sd3_lower          = EXCLUDED.sd3_lower,
            price_sd_position  = EXCLUDED.price_sd_position,
            smc_structure      = EXCLUDED.smc_structure,
            has_valid_zone     = EXCLUDED.has_valid_zone,
            adx_value          = EXCLUDED.adx_value,
            plus_di            = EXCLUDED.plus_di,
            minus_di           = EXCLUDED.minus_di,
            csi_rs             = EXCLUDED.csi_rs,
            csi_diff_zscore    = EXCLUDED.csi_diff_zscore,
            csi_diff_roc       = EXCLUDED.csi_diff_roc,
            csi_commodity_bloc = EXCLUDED.csi_commodity_bloc,
            is_hammer          = EXCLUDED.is_hammer,
            is_shooting_star   = EXCLUDED.is_shooting_star
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_indicator_results: {len(records)} rows upserted")
    return len(records)


def write_scan_results(df: pd.DataFrame, run_datetime: str) -> int:
    """
    Upsert scanner candidate rows for one scan run.

    Args:
        df          : Scanner output [pair, direction, currency_bloc,
                      sd_position, has_valid_zone, ml_score, ml_rank]
        run_datetime: ISO timestamp string of the 4H candle this scan
                      run evaluated (NOT just a date — see module
                      docstring on why datetime, not scan_date, is the
                      uniqueness key for this twice-daily schedule)

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    def _safe_int(value, default=0):
        # Same NaN-safety fix as write_indicator_results — row.get(key,
        # default) only substitutes the default when the KEY is
        # missing, not when the key exists but holds NaN. int(nan)
        # raises ValueError regardless of any 'or default' guard, since
        # NaN is truthy in Python.
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(value)

    records = []
    for _, row in df.iterrows():
        records.append({
            "pair"          : row["pair"],
            "datetime"      : run_datetime,
            "direction"     : row["direction"],
            "currency_bloc" : row.get("currency_bloc"),
            "sd_position"   : float(row.get("sd_position", 0)),
            "has_valid_zone": _safe_int(row.get("has_valid_zone", 0)),
            "ml_score"      : float(row.get("ml_score", 0)),
            "ml_rank"       : _safe_int(row.get("ml_rank", 0)),
        })

    sql = """
        INSERT INTO scan_results (
            pair, datetime, direction, currency_bloc, sd_position,
            has_valid_zone, ml_score, ml_rank
        ) VALUES (
            %(pair)s, %(datetime)s, %(direction)s, %(currency_bloc)s, %(sd_position)s,
            %(has_valid_zone)s, %(ml_score)s, %(ml_rank)s
        )
        ON CONFLICT (pair, datetime, direction) DO UPDATE SET
            ml_score       = EXCLUDED.ml_score,
            ml_rank        = EXCLUDED.ml_rank,
            currency_bloc  = EXCLUDED.currency_bloc,
            has_valid_zone = EXCLUDED.has_valid_zone
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_scan_results: {len(records)} candidates written for {run_datetime}")
    return len(records)


def write_model_metrics(
    model_name : str,
    train_date : str,
    precision  : float,
    auc_roc    : float,
    n_samples  : int,
    recall     : float = 0.0,
    pr_auc     : float = 0.0,
) -> None:
    """Unchanged from the stock version — asset/timeframe-agnostic already."""
    sql = """
        INSERT INTO model_metrics
            (model_name, train_date, precision_score, recall_score, pr_auc_score, auc_roc_score, n_samples)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, train_date) DO UPDATE SET
            precision_score = EXCLUDED.precision_score,
            recall_score    = EXCLUDED.recall_score,
            pr_auc_score    = EXCLUDED.pr_auc_score,
            auc_roc_score   = EXCLUDED.auc_roc_score,
            n_samples       = EXCLUDED.n_samples
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (model_name, train_date, precision, recall, pr_auc, auc_roc, n_samples))

    logger.info(f"write_model_metrics: {model_name} written")


# =============================================================================
# READ FUNCTIONS (used by Streamlit Cloud dashboard)
# =============================================================================

def read_latest_indicator_results() -> pd.DataFrame:
    sql = """
        SELECT * FROM indicator_results
        WHERE datetime = (SELECT MAX(datetime) FROM indicator_results)
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def read_latest_scan_results(direction: Optional[str] = None) -> pd.DataFrame:
    if direction:
        sql = """
            SELECT * FROM scan_results
            WHERE datetime = (SELECT MAX(datetime) FROM scan_results)
              AND direction = %s
            ORDER BY ml_rank ASC
        """
        with get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql, (direction,))
            rows = cursor.fetchall()
            return pd.DataFrame(rows)
    else:
        sql = """
            SELECT * FROM scan_results
            WHERE datetime = (SELECT MAX(datetime) FROM scan_results)
            ORDER BY ml_rank ASC
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn)


def read_latest_model_metrics() -> pd.DataFrame:
    sql = """
        SELECT DISTINCT ON (model_name) *
        FROM model_metrics
        ORDER BY model_name, train_date DESC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_last_fetch_dates_bulk() -> dict:
    """
    Read every tracked pair's last-fetched date in ONE query, instead
    of one query per pair. Used by fetcher.py's smart_fetch() to
    decide full vs. incremental per pair for the whole universe at once.

    Returns:
        Dict mapping pair -> last_fetch_date (as string 'YYYY-MM-DD')
    """
    sql = "SELECT pair, last_fetch_date FROM fetch_tracker"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

    return {pair: str(date) for pair, date in rows}


def write_last_fetch_dates(pair_dates: dict) -> int:
    """
    Update the fetch tracker after a successful fetch.

    Args:
        pair_dates: Dict mapping pair -> the MAX datetime actually
                    fetched for that pair (not just "today" — a pair
                    might have gaps or fail partway through a run)

    Returns:
        Number of pairs updated
    """
    if not pair_dates:
        return 0

    records = [
        {"pair": p, "last_fetch_date": d}
        for p, d in pair_dates.items()
    ]

    sql = """
        INSERT INTO fetch_tracker (pair, last_fetch_date, updated_at)
        VALUES (%(pair)s, %(last_fetch_date)s, NOW())
        ON CONFLICT (pair) DO UPDATE SET
            last_fetch_date = EXCLUDED.last_fetch_date,
            updated_at      = NOW()
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_last_fetch_dates: {len(records)} pairs updated")
    return len(records)


def prune_old_prices(max_days: int = 65) -> int:
    """No-op for cloud — raw_prices not stored in Supabase Postgres.
    See data/storage_cloud.py's consolidate_snapshots() for the actual
    retention/pruning mechanism, which operates on Storage instead."""
    return 0
