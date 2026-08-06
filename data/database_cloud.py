"""
data/database_cloud.py
-----------------------
PostgreSQL version of database.py for cloud deployment (FX directional
prediction pipeline).

Used by:
- run_pipeline_cloud.py  (GitHub Actions)
- dashboard/app_cloud.py (Streamlit Cloud)

DIFFERENCE FROM database.py:
- Uses psycopg2 (PostgreSQL) instead of sqlite3
- Reads connection string from SUPABASE_DB_URL environment variable
- Only stores RESULT tables (no raw_prices — too large for Supabase
  free tier). Raw Weekly OHLC lives in Supabase Storage as Parquet
  snapshots instead (see data/storage_cloud.py), not here.

TABLES STORED IN SUPABASE:
- indicator_results   (ADX + CSI + candlestick pattern flags per pair
  per weekly-candle datetime)
- prediction_results  (basket-model directional predictions that
  cleared the display threshold)
- model_metrics       (per-basket ML model performance)
- fetch_tracker       (per-pair incremental-fetch bookkeeping)
- all_predictions_log (EVERY basket-model prediction, every run,
  regardless of display threshold — added for live-tracking/
  continuation-vs-flip comparisons, since prediction_results only
  ever holds threshold-clearing rows and can't answer "what did this
  pair predict yesterday" if yesterday's prediction happened to be
  sub-threshold)
- prediction_outcomes (once a prediction's validity window has
  elapsed, whether price actually moved in the predicted direction —
  the live accuracy tracker, evaluated against actual closes as they
  become available, separate from the original backtest's AUC/
  precision, which only ever measured historical held-out data)

WHAT'S DIFFERENT FROM THE PRIOR (SIGNAL RANKER SCANNER) VERSION OF
THIS FILE — THIS IS A REDESIGN OF THE SCHEMA, NOT A RENAME:

1. indicator_results DROPS EVERY LINREG/SMC COLUMN. LinReg and SMC are
   dropped from this project's scope entirely (per project decision —
   see ml/features.py's module docstring). This table no longer has
   linreg_value, linreg_slope, linreg_slope_up, sd1/2/3_upper/lower,
   price_sd_position, smc_structure, has_valid_zone — none of these
   are computed anywhere in the pipeline anymore. What remains:
   adx_value/plus_di/minus_di, the 6 CSI columns, and
   is_hammer/is_shooting_star (now raw pattern flags with no extreme
   gate — see engines/candlestick.py's compute_raw_pattern_flags).

2. scan_results IS GONE, REPLACED BY prediction_results. The old table
   held scanner CANDIDATES (pair, direction, sd_position,
   has_valid_zone, ml_score, ml_rank) — that whole concept no longer
   exists. prediction_results holds basket-model directional
   PREDICTIONS instead: (pair, datetime, basket, up_probability,
   direction, confidence). Key differences:
     - Keyed on (pair, datetime, BASKET) not (pair, datetime,
       direction) — a pair belongs to exactly one basket, and the
       basket identity matters for knowing which model produced a
       given prediction.
     - up_probability is the model's raw output; direction ("up"/
       "down") and confidence (symmetric around 0.5) are DERIVED from
       it, not independently modeled.
     - This table ONLY EVER holds rows that already cleared the
       display threshold (config.yaml's ml.high_probability_threshold)
       — per project decision, rows that don't clear it are never
       written here at all. The old scan_results table held EVERY
       candidate regardless of ml_score; this is a deliberate change.

3. model_metrics is UNCHANGED IN SCHEMA — model_name now holds values
   like "directional_basket1_usd" instead of "signal_ranker", but the
   table structure needed no changes — it was already model-agnostic.

4. fetch_tracker is UNCHANGED — still keyed on plain pair name.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other FX file.
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
            datetime             DATE NOT NULL,
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
            created_at           TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, datetime)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id                   SERIAL PRIMARY KEY,
            pair                 TEXT NOT NULL,
            datetime             DATE NOT NULL,
            basket               TEXT NOT NULL,
            up_probability       REAL NOT NULL,
            direction            TEXT NOT NULL,
            confidence           REAL NOT NULL,
            signal_status        TEXT,
            previous_probability REAL,
            created_at           TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, datetime, basket)
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
        """
        CREATE TABLE IF NOT EXISTS all_predictions_log (
            id             SERIAL PRIMARY KEY,
            pair           TEXT NOT NULL,
            datetime       DATE NOT NULL,
            basket         TEXT NOT NULL,
            up_probability REAL NOT NULL,
            direction      TEXT NOT NULL,
            created_at     TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, datetime, basket)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id                  SERIAL PRIMARY KEY,
            pair                TEXT NOT NULL,
            basket              TEXT NOT NULL,
            prediction_datetime DATE NOT NULL,
            direction           TEXT NOT NULL,
            up_probability      REAL NOT NULL,
            valid_through_date  DATE NOT NULL,
            actual_close_change REAL,
            outcome             TEXT,
            evaluated_at        TIMESTAMP,
            created_at          TIMESTAMP DEFAULT NOW(),
            UNIQUE(pair, basket, prediction_datetime)
        )
        """,
    ]

    # Migration statements — for anyone with the OLD schema already
    # initialised in their Supabase instance. CREATE TABLE IF NOT
    # EXISTS is a no-op on a table that already exists (with the OLD
    # LinReg/SMC columns), so indicator_results needs explicit column
    # DROPs here. scan_results' data is NOT migrated forward — it held
    # scanner candidates, a concept that no longer exists, so there is
    # nothing meaningful to carry over.
    migrations = [
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS linreg_value",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS linreg_slope",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS linreg_slope_up",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd1_upper",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd1_lower",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd2_upper",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd2_lower",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd3_upper",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS sd3_lower",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS price_sd_position",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS smc_structure",
        "ALTER TABLE indicator_results DROP COLUMN IF EXISTS has_valid_zone",
        # prediction_results already existed in deployed instances before
        # signal_status/previous_probability were added to its
        # CREATE TABLE statement above — CREATE TABLE IF NOT EXISTS is a
        # no-op against an already-existing table, so those columns
        # never actually got added anywhere they were deployed before
        # this line existed, and every write_prediction_results() call
        # failed outright with "column does not exist" (silently, since
        # run_pipeline_cloud.py's STEP 10 catches and logs the
        # exception rather than crashing the whole run). Explicit
        # ADD COLUMN IF NOT EXISTS here is the fix, and is safe to
        # rerun indefinitely on every initialise_database() call.
        "ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS signal_status TEXT",
        "ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS previous_probability REAL",
        # Convert candle-date columns from TIMESTAMPTZ to DATE on
        # already-deployed tables — same "CREATE TABLE IF NOT EXISTS
        # is a no-op against an existing table" issue as above, so an
        # explicit ALTER is required here too, not just the schema
        # change in the CREATE TABLE statements. USING datetime::date
        # is safe here specifically because every value in these
        # tables was always midnight UTC anyway (candle dates, not
        # real timestamps) — this drops the time-of-day/tz component
        # that was never meaningful, losing no information.
        #
        # THIS IS THE FIX for the recurring family of tz-aware vs.
        # tz-naive comparison bugs this project hit repeatedly
        # (engines/macro.py's driver index, the continuation-lookup
        # comparison, and the STEP 8.5 "Cannot pass a datetime or
        # Timestamp with tzinfo" crash) — a plain DATE has no
        # timezone to be aware or naive ABOUT, so comparisons between
        # these columns and Python's datetime.date objects (what
        # _add_business_days already returns) now just work, with no
        # tz_localize/tz_convert dance needed anywhere.
        "ALTER TABLE indicator_results ALTER COLUMN datetime TYPE DATE USING datetime::date",
        "ALTER TABLE prediction_results ALTER COLUMN datetime TYPE DATE USING datetime::date",
        "ALTER TABLE all_predictions_log ALTER COLUMN datetime TYPE DATE USING datetime::date",
        "ALTER TABLE prediction_outcomes ALTER COLUMN prediction_datetime TYPE DATE USING prediction_datetime::date",
    ]

    with get_connection() as conn:
        cursor = conn.cursor()
        for ddl in tables:
            cursor.execute(ddl)
        for ddl in migrations:
            cursor.execute(ddl)

    logger.info("Supabase database initialised — all tables ready")


# =============================================================================
# WRITE FUNCTIONS
# =============================================================================

def write_indicator_results(df: pd.DataFrame) -> int:
    """
    Upsert engine-output indicator rows (ADX, CSI, candlestick raw
    flags — NO LinReg, NO SMC) — one row per (pair, datetime).
    """
    if df.empty:
        return 0

    cols = [
        "pair", "datetime",
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
        record["is_hammer"]        = _int_default(record.get("is_hammer"))
        record["is_shooting_star"] = _int_default(record.get("is_shooting_star"))
        records.append(record)

    sql = """
        INSERT INTO indicator_results (
            pair, datetime,
            adx_value, plus_di, minus_di,
            csi_rs, csi_diff_zscore, csi_diff_roc, csi_commodity_bloc,
            csi_base_zscore, csi_quote_zscore,
            is_hammer, is_shooting_star
        ) VALUES (
            %(pair)s, %(datetime)s,
            %(adx_value)s, %(plus_di)s, %(minus_di)s,
            %(csi_rs)s, %(csi_diff_zscore)s, %(csi_diff_roc)s, %(csi_commodity_bloc)s,
            %(csi_base_zscore)s, %(csi_quote_zscore)s,
            %(is_hammer)s, %(is_shooting_star)s
        )
        ON CONFLICT (pair, datetime) DO UPDATE SET
            adx_value          = EXCLUDED.adx_value,
            plus_di             = EXCLUDED.plus_di,
            minus_di            = EXCLUDED.minus_di,
            csi_rs              = EXCLUDED.csi_rs,
            csi_diff_zscore     = EXCLUDED.csi_diff_zscore,
            csi_diff_roc        = EXCLUDED.csi_diff_roc,
            csi_commodity_bloc  = EXCLUDED.csi_commodity_bloc,
            csi_base_zscore     = EXCLUDED.csi_base_zscore,
            csi_quote_zscore    = EXCLUDED.csi_quote_zscore,
            is_hammer           = EXCLUDED.is_hammer,
            is_shooting_star    = EXCLUDED.is_shooting_star
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_indicator_results: {len(records)} rows upserted")
    return len(records)


def write_prediction_results(df: pd.DataFrame, run_datetime: str) -> int:
    """
    Upsert basket-model directional prediction rows for one run —
    ONLY rows that already cleared the project's display threshold
    (that filtering happens in run_pipeline_cloud.py, before this
    function is ever called; this function does not re-check it).

    Args:
        df          : Prediction output [pair, basket, up_probability,
                      direction, confidence, signal_status,
                      previous_probability]. The last two are optional
                      — added for live-tracking continuation/flip
                      display; older callers or a df missing these
                      columns still work (they'll be written as NULL).
        run_datetime: ISO timestamp string of the weekly candle this
                      prediction run evaluated

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "pair"                 : row["pair"],
            "datetime"             : run_datetime,
            "basket"               : row["basket"],
            "up_probability"       : float(row["up_probability"]),
            "direction"            : row["direction"],
            "confidence"           : float(row["confidence"]),
            "signal_status"        : row.get("signal_status"),
            "previous_probability" : (
                float(row["previous_probability"])
                if pd.notna(row.get("previous_probability"))
                else None
            ),
        })

    sql = """
        INSERT INTO prediction_results (
            pair, datetime, basket, up_probability, direction, confidence,
            signal_status, previous_probability
        ) VALUES (
            %(pair)s, %(datetime)s, %(basket)s, %(up_probability)s,
            %(direction)s, %(confidence)s, %(signal_status)s,
            %(previous_probability)s
        )
        ON CONFLICT (pair, datetime, basket) DO UPDATE SET
            up_probability       = EXCLUDED.up_probability,
            direction            = EXCLUDED.direction,
            confidence           = EXCLUDED.confidence,
            signal_status        = EXCLUDED.signal_status,
            previous_probability = EXCLUDED.previous_probability
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_prediction_results: {len(records)} predictions written for {run_datetime}")
    return len(records)


def write_all_predictions_log(df: pd.DataFrame, run_datetime: str) -> int:
    """
    Upsert EVERY basket-model prediction for one run, regardless of
    whether it cleared the display threshold — the raw feed backing
    live-tracking's continuation-vs-flip comparison.

    WHY THIS IS A SEPARATE TABLE FROM prediction_results: that table
    only ever holds threshold-clearing rows (by design — see its own
    docstring above). If pair X predicted BUY at 0.60 yesterday
    (below threshold, never written to prediction_results) and BUY at
    0.70 today (clears threshold, shown on the dashboard), there would
    be no way to look up "what did X predict yesterday" to correctly
    label today's prediction as a CONTINUATION rather than a first
    signal — prediction_results simply wouldn't have yesterday's row
    at all. This table holds ALL 28 pairs' raw probabilities every
    single run specifically so that lookup always succeeds.

    Args:
        df          : ALL basket predictions for this run (not just
                      threshold-clearing ones) — [pair, basket,
                      up_probability, direction]
        run_datetime: ISO timestamp string of the candle this
                      prediction run evaluated

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "pair"           : row["pair"],
            "datetime"       : run_datetime,
            "basket"         : row["basket"],
            "up_probability" : float(row["up_probability"]),
            "direction"      : row["direction"],
        })

    sql = """
        INSERT INTO all_predictions_log (
            pair, datetime, basket, up_probability, direction
        ) VALUES (
            %(pair)s, %(datetime)s, %(basket)s, %(up_probability)s,
            %(direction)s
        )
        ON CONFLICT (pair, datetime, basket) DO UPDATE SET
            up_probability = EXCLUDED.up_probability,
            direction      = EXCLUDED.direction
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_all_predictions_log: {len(records)} predictions logged for {run_datetime}")
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
    """Unchanged in schema — model_name now holds values like
    'directional_basket1_usd' instead of 'signal_ranker'."""
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


def get_previous_predictions(pair_basket_pairs: list) -> dict:
    """
    For each (pair, basket) tuple, look up that pair's MOST RECENT
    prior prediction from all_predictions_log — regardless of whether
    it cleared the display threshold — for continuation-vs-flip
    comparison against today's fresh prediction.

    Args:
        pair_basket_pairs: List of (pair, basket) tuples to look up,
                           e.g. [("EURUSD", "usd"), ("AUDCHF", "chf")]

    Returns:
        Dict mapping (pair, basket) -> {"direction": ..., 
        "up_probability": ..., "datetime": ...} for the most recent
        prior row found, OR omitted entirely if no prior row exists
        for that pair (e.g. a brand new pair, or a gap in fetch
        history) — caller should treat a missing key as "no prior
        prediction available," not as any particular direction.
    """
    if not pair_basket_pairs:
        return {}

    # NOTE: "WHERE (pair, basket) IN %s" with a Python list of tuples
    # passed as a single psycopg2 parameter does NOT reliably perform
    # a composite-key IN match — psycopg2 substitutes %s based on the
    # outer sequence, not as nested Postgres row-value tuples, so this
    # previously matched incorrectly (silently, with no error) rather
    # than scoping each lookup to its specific (pair, basket) pair.
    # That bug produced wrong "previous prediction" rows — e.g.
    # matching a different basket's row for the same pair, or the
    # wrong pair's row entirely — which is what caused GBPNZD's
    # genuine direction flip (down -> up) to be mislabeled as
    # "continuation," and EURAUD's real previous probability (0.5615)
    # to be reported as something close to its OWN current value
    # instead. Fixed by explicitly joining against a VALUES list of
    # the exact (pair, basket) pairs requested, which Postgres matches
    # correctly as a composite key.
    values_clause = ", ".join(["(%s, %s)"] * len(pair_basket_pairs))
    params = [item for pair_basket in pair_basket_pairs for item in pair_basket]

    sql = f"""
        SELECT DISTINCT ON (apl.pair, apl.basket)
            apl.pair, apl.basket, apl.direction, apl.up_probability, apl.datetime
        FROM all_predictions_log apl
        JOIN (VALUES {values_clause}) AS wanted(pair, basket)
            ON apl.pair = wanted.pair AND apl.basket = wanted.basket
        ORDER BY apl.pair, apl.basket, apl.datetime DESC
    """

    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return {
        (row["pair"], row["basket"]): {
            "direction"      : row["direction"],
            "up_probability" : row["up_probability"],
            "datetime"       : row["datetime"],
        }
        for row in rows
    }


def write_prediction_outcomes(records: list) -> int:
    """
    Insert prediction-outcome rows once a prediction's validity window
    has elapsed and the actual price move is known.

    Args:
        records: List of dicts with keys pair, basket,
                prediction_datetime, direction, up_probability,
                valid_through_date, actual_close_change, outcome
                ("correct"/"incorrect")

    Returns:
        Number of rows written
    """
    if not records:
        return 0

    sql = """
        INSERT INTO prediction_outcomes (
            pair, basket, prediction_datetime, direction, up_probability,
            valid_through_date, actual_close_change, outcome, evaluated_at
        ) VALUES (
            %(pair)s, %(basket)s, %(prediction_datetime)s, %(direction)s,
            %(up_probability)s, %(valid_through_date)s,
            %(actual_close_change)s, %(outcome)s, NOW()
        )
        ON CONFLICT (pair, basket, prediction_datetime) DO UPDATE SET
            actual_close_change = EXCLUDED.actual_close_change,
            outcome             = EXCLUDED.outcome,
            evaluated_at        = NOW()
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_prediction_outcomes: {len(records)} outcomes recorded")
    return len(records)


def read_unevaluated_predictions(as_of_date: str) -> pd.DataFrame:
    """
    Find predictions with NO outcome recorded yet in prediction_outcomes
    — candidates for evaluation, not yet filtered by whether their
    validity window has actually expired.

    IMPORTANT: this function does NOT filter by valid_through_date —
    it can't, cleanly, in SQL alone, since valid_through_date isn't a
    stored column here (it's derived from the prediction's date via
    business-day arithmetic, which skips weekends — not a single plain
    SQL date comparison). The as_of_date argument is accepted for
    interface consistency but currently unused inside this function's
    SQL; every row with no outcome yet comes back, expired or not. The
    caller (run_pipeline_cloud.py's STEP 8.5) is responsible for
    computing each row's valid_through_date via _add_business_days()
    and skipping (continuing past) any row that hasn't actually
    expired yet. This split was previously undocumented and looked
    like a bug (a docstring here claimed date-filtering that was never
    implemented) — now stated plainly so it isn't mistaken for one
    again.

    Only looks at predictions that cleared the display threshold
    (prediction_results), since those are the only ones the live
    accuracy tracker needs to grade — sub-threshold rows in
    all_predictions_log exist purely for continuation lookups, not for
    outcome tracking.

    Args:
        as_of_date: Accepted for interface consistency with the
                   caller's naming, but not used in this function's
                   SQL — see the note above. The real filtering by
                   expiry happens in run_pipeline_cloud.py's STEP 8.5,
                   row by row, after this function returns.

    Returns:
        DataFrame [pair, basket, datetime, direction, up_probability],
        one row per unscored prediction — expired or not; the caller
        filters further.
    """
    sql = """
        SELECT pr.pair, pr.basket, pr.datetime, pr.direction, pr.up_probability
        FROM prediction_results pr
        LEFT JOIN prediction_outcomes po
            ON pr.pair = po.pair
            AND pr.basket = po.basket
            AND pr.datetime = po.prediction_datetime
        WHERE po.id IS NULL
        ORDER BY pr.datetime ASC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def read_prediction_outcomes(limit_days: int = 30) -> pd.DataFrame:
    """
    Read recent scored prediction outcomes for the live-tracking
    dashboard section — the actual "how is this doing in production"
    view, separate from the original backtest's AUC/precision.

    Args:
        limit_days: Only return outcomes for predictions made in the
                   last N days.

    Returns:
        DataFrame ordered by prediction_datetime descending.
    """
    sql = """
        SELECT * FROM prediction_outcomes
        WHERE prediction_datetime >= NOW() - INTERVAL '%s days'
        ORDER BY prediction_datetime DESC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=(limit_days,))


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


def read_latest_prediction_results(
    basket: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read display-threshold predictions for a specific candle date,
    optionally filtered to one basket.

    Args:
        basket    : Optional basket name to filter to.
        as_of_date: 'YYYY-MM-DD' string for the candle date to read.
                   If omitted, falls back to the table's own MAX(datetime)
                   — kept for backward compatibility with any existing
                   caller, but NOT recommended for a live dashboard: if
                   TODAY'S run had zero predictions clear the display
                   threshold (a real, expected outcome — see
                   run_pipeline_cloud.py's STEP 10 log,
                   "Predictions clearing display threshold: 0/28"), the
                   table's global MAX(datetime) silently falls back to
                   whatever the last run WITH qualifying predictions
                   was, which could be days old. That produced exactly
                   this symptom: a dashboard showing a stale pair
                   (e.g. EURAUD @ 0.70 from days ago) as if it were
                   today's result, with no indication it wasn't fresh.
                   Callers that need "today, honestly, even if empty"
                   MUST pass as_of_date explicitly (see
                   app_cloud.py, which passes the same latest-candle
                   date already computed for its header caption).

    Returns:
        DataFrame ordered by confidence descending — replaces the old
        ml_rank ordering, since there's no per-candidate rank concept
        anymore, just confidence. Empty DataFrame if as_of_date was
        given and that specific date has no qualifying rows — this is
        the correct, honest result, not an error.
    """
    if as_of_date:
        if basket:
            sql = """
                SELECT * FROM prediction_results
                WHERE datetime = %s
                  AND basket = %s
                ORDER BY confidence DESC
            """
            params = (as_of_date, basket)
        else:
            sql = """
                SELECT * FROM prediction_results
                WHERE datetime = %s
                ORDER BY confidence DESC
            """
            params = (as_of_date,)

        with get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return pd.DataFrame(rows)

    # Backward-compatible fallback path (no as_of_date given) — see
    # docstring warning above about why this isn't safe for a live
    # "did today produce anything" dashboard view.
    if basket:
        sql = """
            SELECT * FROM prediction_results
            WHERE datetime = (SELECT MAX(datetime) FROM prediction_results)
              AND basket = %s
            ORDER BY confidence DESC
        """
        with get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql, (basket,))
            rows = cursor.fetchall()
            return pd.DataFrame(rows)
    else:
        sql = """
            SELECT * FROM prediction_results
            WHERE datetime = (SELECT MAX(datetime) FROM prediction_results)
            ORDER BY confidence DESC
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
    Read every tracked pair's last-fetched date in ONE query.

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
                    fetched for that pair

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
