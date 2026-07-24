"""
data/database.py
-----------------
Local SQLite version of the database layer, for dev/debugging only.

NOT used by the production deployment path — run_pipeline_cloud.py
(GitHub Actions) and dashboard/app_cloud.py (Streamlit Cloud) both
import exclusively from data.database_cloud and data.storage_cloud.
This file exists so the pipeline can be run and iterated on locally
(e.g. testing a new feature or engine change against a small local
dataset) without needing live Supabase credentials.

WHY THIS FILE EXISTS AT ALL, GIVEN CLOUD IS THE ONLY DEPLOYMENT TARGET:
fetcher.py and train_models.py both branch on
`_is_cloud = bool(os.getenv("SUPABASE_DB_URL"))` and import from this
module in the ELSE (local) branch — that branch was already present in
the code before this file existed, so without it, running either
script locally (SUPABASE_DB_URL unset) would ImportError immediately.
This module makes that local branch actually work, matching what
database_cloud.py's docstring already assumed existed.

SCHEMA PARITY WITH database_cloud.py:
indicator_results/scan_results/model_metrics/fetch_tracker tables here
mirror database_cloud.py's Postgres schema as closely as SQLite's type
system allows, so a feature/engine tested locally behaves the same way
once deployed to Supabase. One deliberate difference: this file DOES
store raw_prices locally (SQLite has no free-tier size pressure the
way Supabase Postgres does — see database_cloud.py's docstring for why
raw prices live in Supabase Storage, not Postgres, in production).
Locally, a plain raw_prices table is simpler than standing up a local
Parquet-snapshot equivalent of storage_cloud.py just for dev use.

COLUMN NAMING (pair/datetime, not ticker/date):
Consistent with every other FX file in this project.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import yaml

from utils.logging import get_database_logger
from utils.error_handler import DatabaseError

logger = get_database_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config   = _load_config()
DB_PATH  = Path(__file__).resolve().parents[1] / config["database"]["path"]


# =============================================================================
# CONNECTION
# =============================================================================

@contextmanager
def get_connection():
    """Context manager for local SQLite connections."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"SQLite error: {e}") from e
    finally:
        if conn:
            conn.close()


# =============================================================================
# INITIALISE TABLES
# =============================================================================

def initialise_database() -> None:
    """
    Create all local tables if they don't exist. Safe to call multiple
    times. Mirrors database_cloud.py's schema (see module docstring for
    the one deliberate difference: raw_prices lives here locally,
    unlike in the cloud/Supabase deployment path).
    """
    tables = [
        """
        CREATE TABLE IF NOT EXISTS raw_prices (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            pair     TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open     REAL,
            high     REAL,
            low      REAL,
            close    REAL,
            UNIQUE(pair, datetime)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS indicator_results (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            pair                 TEXT NOT NULL,
            datetime             TEXT NOT NULL,
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
            csi_base_zscore      REAL,
            csi_quote_zscore     REAL,
            is_hammer            INTEGER DEFAULT 0,
            is_shooting_star     INTEGER DEFAULT 0,
            created_at           TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, datetime)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            pair           TEXT NOT NULL,
            datetime       TEXT NOT NULL,
            direction      TEXT NOT NULL,
            currency_bloc  TEXT,
            sd_position    REAL,
            has_valid_zone INTEGER DEFAULT 0,
            ml_score       REAL,
            ml_rank        INTEGER,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, datetime, direction)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT NOT NULL,
            train_date      TEXT NOT NULL,
            precision_score REAL,
            recall_score    REAL,
            pr_auc_score    REAL,
            auc_roc_score   REAL,
            n_samples       INTEGER,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, train_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fetch_tracker (
            pair            TEXT PRIMARY KEY,
            last_fetch_date TEXT NOT NULL,
            updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]

    # SQLite has no ALTER TABLE ADD COLUMN IF NOT EXISTS — unlike
    # database_cloud.py's Postgres migration approach, so a schema
    # change here means dropping the local dev DB file and letting it
    # get recreated, not migrating an existing one in place. Acceptable
    # for a local dev-only database with no production data at stake.
    with get_connection() as conn:
        cursor = conn.cursor()
        for ddl in tables:
            cursor.execute(ddl)

    logger.info(f"Local database initialised at {DB_PATH}")


# =============================================================================
# WRITE — raw prices (called from fetcher.py's local/dev branch)
# =============================================================================

def write_raw_prices(df: pd.DataFrame) -> int:
    """
    Upsert raw 4H OHLC rows, one row per (pair, datetime).

    Args:
        df: DataFrame with columns [pair, datetime, open, high, low, close]

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    records = df[["pair", "datetime", "open", "high", "low", "close"]].to_dict("records")

    sql = """
        INSERT INTO raw_prices (pair, datetime, open, high, low, close)
        VALUES (:pair, :datetime, :open, :high, :low, :close)
        ON CONFLICT(pair, datetime) DO UPDATE SET
            open  = excluded.open,
            high  = excluded.high,
            low   = excluded.low,
            close = excluded.close
    """

    with get_connection() as conn:
        conn.executemany(sql, records)

    logger.info(f"write_raw_prices: {len(records)} rows upserted")
    return len(records)


def read_raw_prices(pair: str) -> pd.DataFrame:
    """
    Read all stored raw 4H OHLC rows for one pair, sorted by datetime.

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'

    Returns:
        DataFrame [pair, datetime, open, high, low, close], sorted
        ascending. Empty DataFrame if nothing stored for this pair yet.
    """
    sql = "SELECT * FROM raw_prices WHERE pair = ? ORDER BY datetime ASC"

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(pair,))

    if df.empty:
        logger.debug(f"read_raw_prices: no data for {pair}")

    return df


# =============================================================================
# FETCH TRACKER — used by fetcher.py's smart_fetch() local branch
# =============================================================================

def get_last_fetch_date(pair: str) -> Optional[str]:
    """
    Read one pair's last-fetched date, for smart_fetch()'s local
    (non-cloud) branch, which looks this up per-pair rather than in
    bulk (database_cloud.py's get_last_fetch_dates_bulk() is the
    cloud/bulk equivalent — see fetcher.py's smart_fetch() for how the
    two branches differ).

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'

    Returns:
        Last fetch date as a string, or None if this pair has never
        been tracked (triggers a full fetch in smart_fetch())
    """
    sql = "SELECT last_fetch_date FROM fetch_tracker WHERE pair = ?"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (pair,))
        row = cursor.fetchone()

    return row[0] if row else None


def write_last_fetch_dates(pair_dates: dict) -> int:
    """
    Update the fetch tracker after a successful fetch — same shape and
    purpose as database_cloud.py's write_last_fetch_dates(), for the
    local (non-cloud) path.

    Args:
        pair_dates: Dict mapping pair -> the MAX datetime actually
                    fetched for that pair

    Returns:
        Number of pairs updated
    """
    if not pair_dates:
        return 0

    records = [{"pair": p, "last_fetch_date": d} for p, d in pair_dates.items()]

    sql = """
        INSERT INTO fetch_tracker (pair, last_fetch_date, updated_at)
        VALUES (:pair, :last_fetch_date, CURRENT_TIMESTAMP)
        ON CONFLICT(pair) DO UPDATE SET
            last_fetch_date = excluded.last_fetch_date,
            updated_at      = CURRENT_TIMESTAMP
    """

    with get_connection() as conn:
        conn.executemany(sql, records)

    logger.info(f"write_last_fetch_dates: {len(records)} pairs updated")
    return len(records)


# =============================================================================
# WRITE — indicator results / scan results (local dev parity with
# database_cloud.py, not currently called by fetcher.py/train_models.py
# but kept for full local dev parity with the cloud path)
# =============================================================================

def write_indicator_results(df: pd.DataFrame) -> int:
    """Local SQLite equivalent of database_cloud.py's write_indicator_results."""
    if df.empty:
        return 0

    cols = [
        "pair", "datetime", "linreg_value", "linreg_slope", "linreg_slope_up",
        "sd1_upper", "sd1_lower", "sd2_upper", "sd2_lower", "sd3_upper", "sd3_lower",
        "price_sd_position", "smc_structure", "has_valid_zone",
        "adx_value", "plus_di", "minus_di",
        "csi_rs", "csi_diff_zscore", "csi_diff_roc", "csi_commodity_bloc",
        "csi_base_zscore", "csi_quote_zscore",
        "is_hammer", "is_shooting_star",
    ]

    def _int_default(value, default=0):
        if value is None or pd.isna(value):
            return default
        return int(value)

    records = []
    for _, row in df.iterrows():
        record = {c: row.get(c) for c in cols}
        record["has_valid_zone"]   = _int_default(record.get("has_valid_zone"))
        record["is_hammer"]        = _int_default(record.get("is_hammer"))
        record["is_shooting_star"] = _int_default(record.get("is_shooting_star"))
        records.append(record)

    placeholders = ", ".join(f":{c}" for c in cols)
    update_clause = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("pair", "datetime"))

    sql = f"""
        INSERT INTO indicator_results ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(pair, datetime) DO UPDATE SET
            {update_clause}
    """

    with get_connection() as conn:
        conn.executemany(sql, records)

    logger.info(f"write_indicator_results: {len(records)} rows upserted")
    return len(records)


def write_scan_results(df: pd.DataFrame, run_datetime: str) -> int:
    """Local SQLite equivalent of database_cloud.py's write_scan_results."""
    if df.empty:
        return 0

    def _safe_int(value, default=0):
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
            :pair, :datetime, :direction, :currency_bloc, :sd_position,
            :has_valid_zone, :ml_score, :ml_rank
        )
        ON CONFLICT(pair, datetime, direction) DO UPDATE SET
            ml_score       = excluded.ml_score,
            ml_rank        = excluded.ml_rank,
            currency_bloc  = excluded.currency_bloc,
            has_valid_zone = excluded.has_valid_zone
    """

    with get_connection() as conn:
        conn.executemany(sql, records)

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
    """Local SQLite equivalent of database_cloud.py's write_model_metrics."""
    sql = """
        INSERT INTO model_metrics
            (model_name, train_date, precision_score, recall_score, pr_auc_score, auc_roc_score, n_samples)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name, train_date) DO UPDATE SET
            precision_score = excluded.precision_score,
            recall_score    = excluded.recall_score,
            pr_auc_score    = excluded.pr_auc_score,
            auc_roc_score   = excluded.auc_roc_score,
            n_samples       = excluded.n_samples
    """
    with get_connection() as conn:
        conn.execute(sql, (model_name, train_date, precision, recall, pr_auc, auc_roc, n_samples))

    logger.info(f"write_model_metrics: {model_name} written")


# =============================================================================
# READ — for local dashboard use (dev only)
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
              AND direction = ?
            ORDER BY ml_rank ASC
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=(direction,))
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
        SELECT model_name, train_date, precision_score, recall_score,
               pr_auc_score, auc_roc_score, n_samples
        FROM model_metrics m
        WHERE train_date = (
            SELECT MAX(train_date) FROM model_metrics
            WHERE model_name = m.model_name
        )
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def prune_old_prices(max_days: int = 65) -> int:
    """
    Delete raw_prices rows older than max_days. Local-dev-only utility
    — no equivalent needed for the cloud path (database_cloud.py never
    stores raw_prices at all; see that module's docstring). Not called
    automatically by anything — a manual dev utility.

    Args:
        max_days: Keep rows within this many days, delete the rest

    Returns:
        Number of rows deleted
    """
    sql = """
        DELETE FROM raw_prices
        WHERE datetime < datetime('now', ?)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (f"-{max_days} days",))
        deleted = cursor.rowcount

    logger.info(f"prune_old_prices: deleted {deleted} rows older than {max_days} days")
    return deleted
