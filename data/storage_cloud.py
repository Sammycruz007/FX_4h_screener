"""
data/storage_cloud.py
----------------------
FULL raw 4H-price history for cloud training, stored as Parquet
snapshots in Supabase Storage (NOT Postgres — keeps Supabase DB rows
free-tier friendly).

WHY THIS EXISTS:
run_pipeline_cloud.py fetches OHLCV and discards it after computing
indicators — Supabase Postgres was never meant to hold raw_prices (see
database_cloud.py docstring). But train_models.py needs a long window
of raw 4H OHLCV history to backfill indicators and give the model
exposure to multiple market regimes. This module holds that history
cheaply.

NOT A SIMPLE ROLLING WINDOW, BUT NOT UNBOUNDED EITHER — CONSOLIDATION:
This module accumulates history and is not a naive "last N days"
rolling window (a rolling read would silently lose the large first-run
backfill once it aged out — see read_price_history()'s docstring).
But leaving snapshot FILES to accumulate forever isn't right either:
Supabase's free tier caps total object count as well as total bytes,
and with runs happening twice a day (see scheduler.run_times in
config.yaml), file count would grow unbounded even though the
underlying DATA volume stays small and bounded by
storage.retention_days (~2.5 years). consolidate_snapshots() is called
every pipeline run: it reads everything, dedupes on (pair, datetime),
trims anything older than retention_days, writes ONE fresh
consolidated file, then deletes every old chunk — safe because the
new file already captured everything worth keeping before the old
ones are removed. See consolidate_snapshots()'s own docstring for the
full mechanics and why this is safer than a naive age-based delete.

WHY WE DON'T JUST DELETE SNAPSHOTS RIGHT AFTER EACH RUN, THOUGH (a
real question, worth stating explicitly): these snapshots are NOT a
cache for the run that wrote them. Postgres never holds raw prices at
all (see above) — this bucket is the ONLY persistent copy of price
history that exists anywhere in this architecture. train_models.py's
read_price_history() reconstructs the FULL backfill window from all
accumulated history — that is the entire mechanism by which training
gets multi-regime historical data. Deleting a run's data right after
that run would permanently lose it (not re-fetchable later, since
yfinance's history window is finite — fetcher.py's 729-day 1H
ceiling), and would break fetcher.py's incremental fetch design, which
relies on accumulated Storage history for everything older than its
tiny per-run delta. Consolidation solves this correctly: it keeps
every row inside the retention window, and only removes rows/files
once they're genuinely outside it — never data the project still
needs.

TWICE-DAILY SCAN SCHEDULE CHANGES THE SNAPSHOT KEY (FX-specific fix):
Per config.yaml's scheduler.run_times (["12:00", "20:00"] UTC), this
pipeline runs TWICE per calendar day, unlike the stock project's once-
daily cadence. The original filename scheme
("YYYY-MM-DD_partNNN.parquet") keys snapshots by DATE ONLY — two runs
on the same day would collide on the same filename prefix, and since
writes use x-upsert=true, the SECOND run's write would silently
overwrite the FIRST run's snapshot data for that day. Fixed here by
keying filenames on a full UTC timestamp
("YYYY-MM-DDTHHMM_partNNN.parquet", e.g. "2026-07-22T1200_part000.parquet"
and "2026-07-22T2000_part000.parquet" as two genuinely distinct files),
so both of a day's runs are preserved independently. See
_parse_run_timestamp_from_filename() for the corresponding read-side fix.

INCREMENTAL FETCH CHANGES THE SNAPSHOT SHAPE:
Run 1's snapshot is large (full ~729-day backfill for every pair,
since nothing is tracked yet). Every run after is tiny (just each
pair's new 4H candles since the last run, via fetcher.py's genuine
cloud incremental fetch). Both shapes are handled by the same
chunked-write / read-everything logic below — no special-casing needed.

BUCKET LAYOUT (chunked — see note below):
    prices/
        2026-07-22T1200_part000.parquet   ← first scan of the day
        2026-07-22T2000_part000.parquet   ← second scan, same day
        2026-07-23T1200_part000.parquet
        ...

WHY CHUNKED, NOT ONE FILE PER RUN:
Supabase Storage free-tier buckets cap individual file uploads at
50MB. A full-universe, full-history snapshot can be far larger than
that as a single Parquet file. So each snapshot is split into multiple
smaller part files (CHUNK_ROWS rows each) instead of one large file.
Reading transparently reassembles all parts.

USAGE:
    write_snapshot(raw_df, run_timestamp)   # called from run_pipeline_cloud.py, every run
    consolidate_snapshots(retention_days)   # called from run_pipeline_cloud.py, every run
    df = read_price_history()               # called from train_models.py — reads ALL history
    # prune_old_snapshots() still exists but is SUPERSEDED by
    # consolidate_snapshots() above and is NOT called anywhere in this
    # project — see its own docstring for why it's kept only for
    # reference.

REQUIRES:
    No extra package — uses plain HTTP calls to Supabase's Storage
    REST API via `requests` (already a pinned dependency), instead of
    the `supabase` client package. This avoids any ambiguity around
    client-library version support for Supabase's newer sb_secret_...
    API key format.

    Two GitHub Secrets needed: SUPABASE_URL, SUPABASE_KEY
    (SUPABASE_KEY = your sb_secret_... key — NOT the sb_publishable_...
    key, and NOT the same value as SUPABASE_DB_URL, which is the
    separate Postgres connection string).

DEBUGGING "Invalid API key" ERRORS:
    Because this module uses plain HTTP, you can test the exact same
    credentials directly with curl, outside of the pipeline:

        curl "$SUPABASE_URL/storage/v1/bucket" \\
             -H "apikey: $SUPABASE_KEY" \\
             -H "Authorization: Bearer $SUPABASE_KEY"

    If that curl call also returns "Invalid API key", the problem is
    the secret value itself (wrong key copied, extra whitespace, or
    using the publishable key instead of the secret key) — not this
    code. If curl succeeds, re-check the GitHub secret values for
    typos or trailing whitespace.
"""

import os
import io
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from utils.logging import get_database_logger
from utils.error_handler import DatabaseError

logger = get_database_logger()

BUCKET_NAME = "price-history"
PREFIX      = "prices"

# Rows per chunk file. Supabase free-tier caps individual uploads at
# 50MB. 500k rows of OHLCV data compresses (snappy+parquet) to well
# under that with headroom — adjust down if you still see 413s.
CHUNK_ROWS = 500_000

# Filename timestamp format — includes hour+minute (not just date) so
# the twice-daily schedule's two runs never collide on the same
# filename. See module docstring's "TWICE-DAILY SCAN SCHEDULE" section.
FILENAME_TS_FORMAT = "%Y-%m-%dT%H%M"

# Consolidated snapshots (written by consolidate_snapshots(), see below)
# are named "consolidated_{ts}_part000.parquet" to distinguish them from
# individual per-run snapshots — defined here, near FILENAME_TS_FORMAT,
# since _parse_run_timestamp_from_filename() needs to strip this prefix
# before parsing the timestamp portion of a consolidated file's name.
CONSOLIDATED_PREFIX = "consolidated"


# =============================================================================
# CONFIG / HEADERS
# =============================================================================

def _get_config():
    """
    Read SUPABASE_URL and SUPABASE_KEY from the environment and build
    the base URL + auth headers used by every Storage REST call.

    Returns:
        Tuple of (base_url, headers)
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise DatabaseError(
            "SUPABASE_URL / SUPABASE_KEY environment variables not set. "
            "Add them to GitHub Secrets (Storage needs the project URL + "
            "the sb_secret_... key, separate from SUPABASE_DB_URL)."
        )

    base_url = url.rstrip("/") + "/storage/v1"
    headers  = {
        "apikey"       : key,
        "Authorization": f"Bearer {key}",
    }
    return base_url, headers


def _parse_run_timestamp_from_filename(name: str) -> Optional[datetime]:
    """
    Extract the snapshot run timestamp from a chunk filename, e.g.
    "2026-07-22T1200_part003.parquet" -> datetime(2026, 7, 22, 12, 0)

    Replaces the date-only parser from the stock project's version —
    FX's twice-daily schedule needs hour+minute precision to tell two
    same-day snapshots apart (see module docstring).
    """
    if not name.endswith(".parquet"):
        return None

    stem = name.replace(".parquet", "")
    ts_str = stem.split("_part")[0] if "_part" in stem else stem

    # Consolidated files are named "consolidated_{ts}_part000.parquet" —
    # strip that prefix before parsing, or strptime fails and this
    # function returns None, silently making every consolidated
    # snapshot invisible to read_price_history() (a real bug caught in
    # testing: consolidation would "succeed" — write the new file,
    # delete the old ones — but the new file could never be read back).
    if ts_str.startswith(f"{CONSOLIDATED_PREFIX}_"):
        ts_str = ts_str[len(f"{CONSOLIDATED_PREFIX}_"):]

    try:
        return datetime.strptime(ts_str, FILENAME_TS_FORMAT)
    except ValueError:
        return None


# =============================================================================
# WRITE — called from run_pipeline_cloud.py after each OHLCV fetch
# =============================================================================

def write_snapshot(df: pd.DataFrame, run_timestamp: datetime) -> bool:
    """
    Write a run's raw 4H OHLC DataFrame to Supabase Storage as chunked
    Parquet files (each under the 50MB per-file Storage limit).

    Args:
        df           : Raw 4H OHLC DataFrame (pair, datetime, open,
                       high, low, close)
        run_timestamp: This pipeline run's UTC timestamp (NOT just a
                       date — see module docstring on why the
                       twice-daily schedule needs this) — used to build
                       the filename prefix, e.g. "2026-07-22T1200"

    Returns:
        True if ALL chunks were written successfully, False otherwise
        (non-fatal — pipeline should continue even if this fails)
    """
    if df.empty:
        logger.warning("write_snapshot: empty DataFrame, skipping")
        return False

    ts_str = run_timestamp.strftime(FILENAME_TS_FORMAT)

    try:
        base_url, headers = _get_config()

        n_chunks = max(1, (len(df) + CHUNK_ROWS - 1) // CHUNK_ROWS)
        upload_headers = {
            **headers,
            "Content-Type": "application/octet-stream",
            "x-upsert"    : "true",
        }

        total_bytes    = 0
        chunks_written = 0

        for i in range(n_chunks):
            chunk = df.iloc[i * CHUNK_ROWS : (i + 1) * CHUNK_ROWS]
            if chunk.empty:
                continue

            buffer = io.BytesIO()
            chunk.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
            buffer.seek(0)
            raw_bytes = buffer.read()

            path = f"{PREFIX}/{ts_str}_part{i:03d}.parquet"

            resp = requests.post(
                f"{base_url}/object/{BUCKET_NAME}/{path}",
                headers=upload_headers,
                data=raw_bytes,
                timeout=60,
            )

            if resp.status_code not in (200, 201):
                logger.warning(
                    f"write_snapshot: chunk {i} failed | "
                    f"HTTP {resp.status_code} — {resp.text[:300]}"
                )
                continue

            total_bytes    += len(raw_bytes)
            chunks_written  += 1

        if chunks_written == 0:
            logger.warning("write_snapshot: no chunks written successfully")
            return False

        size_mb = total_bytes / (1024 * 1024)
        logger.info(
            f"write_snapshot: {len(df)} rows written across "
            f"{chunks_written}/{n_chunks} chunks for run {ts_str} ({size_mb:.1f} MB total)"
        )
        return chunks_written == n_chunks

    except Exception as e:
        logger.warning(f"write_snapshot failed: {e} — continuing without it")
        return False


# =============================================================================
# READ — called from train_models.py to rebuild history for backfilling
# =============================================================================

def _list_snapshot_files(base_url: str, headers: dict) -> list[dict]:
    """
    List all files under the prices/ prefix in the bucket via the
    Storage REST API's list endpoint.
    """
    resp = requests.post(
        f"{base_url}/object/list/{BUCKET_NAME}",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "prefix": PREFIX,
            "limit": 1000,
            "sortBy": {"column": "name", "order": "asc"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def read_price_history(days: Optional[int] = None) -> pd.DataFrame:
    """
    Download and concatenate ALL accumulated Parquet snapshot chunks by
    default — NOT a rolling window. See module docstring's "WHY WE
    DON'T DELETE SNAPSHOTS" section for the full reasoning.

    WHY NO DEFAULT WINDOW: the incremental fetcher means the first
    run's snapshot is large (full ~729-day backfill for every pair,
    since nothing is tracked yet) and every run after is tiny (just
    each pair's new 4H candles). A rolling "last N days" read would
    silently lose almost the entire dataset once the first run's
    snapshot aged out of the window. Reading everything and relying on
    drop_duplicates(subset=["pair","datetime"]) to merge correctly is
    the simplest fix at this data scale.

    Args:
        days: Optional — if given, restricts to snapshots from the last
              N days only (rolling window). Leave as None (default) to
              read the full accumulated history, which is what training
              needs for full regime coverage.

    Returns:
        Combined DataFrame across all available snapshot chunks.
        Empty DataFrame if none found or on failure.
    """
    try:
        base_url, headers = _get_config()

        files = _list_snapshot_files(base_url, headers)
        if not files:
            logger.warning("read_price_history: no snapshots found in bucket")
            return pd.DataFrame()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)) if days is not None else None

        frames     = []
        runs_seen  = set()

        for f in files:
            name = f["name"]  # e.g. "2026-07-22T1200_part003.parquet"
            file_ts = _parse_run_timestamp_from_filename(name)
            if file_ts is None:
                continue
            if cutoff is not None and file_ts.replace(tzinfo=timezone.utc) < cutoff:
                continue

            dl_resp = requests.get(
                f"{base_url}/object/{BUCKET_NAME}/{PREFIX}/{name}",
                headers=headers,
                timeout=60,
            )
            dl_resp.raise_for_status()

            df = pd.read_parquet(io.BytesIO(dl_resp.content), engine="pyarrow")
            frames.append(df)
            runs_seen.add(file_ts.strftime(FILENAME_TS_FORMAT))

        if not frames:
            window_desc = f"within last {days} days" if days is not None else "at all"
            logger.warning(f"read_price_history: no snapshots found {window_desc}")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["pair", "datetime"])

        logger.info(
            f"read_price_history: {len(combined)} rows loaded from "
            f"{len(frames)} chunks across {len(runs_seen)} snapshot runs | "
            f"Pairs: {combined['pair'].nunique()}"
        )
        return combined

    except Exception as e:
        logger.error(f"read_price_history failed: {e}")
        return pd.DataFrame()


def read_raw_prices_cloud(pair: str, days: Optional[int] = None) -> pd.DataFrame:
    """
    Drop-in replacement for database.py's read_raw_prices(pair), but
    backed by Parquet snapshots instead of SQLite.

    NOTE: Less efficient than a real per-pair query — this reads the
    full rolling window then filters. For a one-off call this is fine;
    train_models.py should prefer read_price_history() once and filter
    in memory across the whole pair loop instead of calling this per
    pair.

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'
        days: Rolling window size

    Returns:
        DataFrame for this pair only, sorted by datetime
    """
    history = read_price_history(days=days)
    if history.empty:
        return pd.DataFrame()

    df = history[history["pair"] == pair].sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# CONSOLIDATE — called every pipeline run from run_pipeline_cloud.py
# Bounds both total Storage bytes AND object count (Supabase free tier
# caps both) WITHOUT ever losing data still inside the retention
# window — unlike a naive age-based prune, which could delete a chunk
# containing rows that are still within the window just because the
# chunk itself happens to be old.
# =============================================================================

def consolidate_snapshots(retention_days: int) -> bool:
    """
    Collapse ALL accumulated snapshot chunks (individual run deltas
    AND any previous consolidated file) into ONE fresh, de-duplicated,
    retention-trimmed snapshot, then delete every old chunk.

    WHY THIS EXISTS: without any pruning, every pipeline run adds new
    chunk files forever — twice a day, indefinitely. The underlying
    data volume stays small (28 pairs, 4H candles, bounded by
    retention_days — comfortably under Supabase's free-tier limits),
    but the FILE COUNT does not stay bounded on its own, and Supabase's
    free tier caps total object count as well as total bytes. A naive
    age-based delete (the stock project's prune_old_snapshots, kept
    below but never called) risks deleting a chunk that still contains
    in-window rows, since a chunk's OWN age doesn't tell you the age of
    the OLDEST row inside it once run-to-run overlaps get involved.
    Consolidation sidesteps that entirely: read everything, keep only
    rows that are actually still in-window, write that as the new
    single source of truth, THEN delete the old files — by which point
    every row worth keeping has already been captured in the new file.

    FLOW:
    1. Read every accumulated chunk via the existing read_price_history()
       machinery (handles both individual run deltas and any prior
       consolidated file identically — dedup logic doesn't care which
       file a row came from)
    2. Drop rows older than retention_days
    3. Write the result as ONE new consolidated snapshot (still
       chunked via the existing CHUNK_ROWS logic if it exceeds the
       50MB/file limit)
    4. Delete every old chunk file (both individual-run and any prior
       consolidated file) that existed BEFORE this consolidation pass
       — safe because step 3 already wrote everything worth keeping

    Args:
        retention_days: Rows older than this (from the current UTC
                        time) are dropped during consolidation. Read
                        live from config.yaml's storage.retention_days
                        by the caller — not hardcoded here.

    Returns:
        True if consolidation completed and the old files were
        cleaned up successfully, False on any failure (non-fatal —
        pipeline should continue regardless; the old files simply
        accumulate one more run's worth until the next successful
        consolidation)
    """
    try:
        base_url, headers = _get_config()

        pre_consolidation_files = _list_snapshot_files(base_url, headers)
        if not pre_consolidation_files:
            logger.info("consolidate_snapshots: no existing snapshots, nothing to do")
            return True

        # ── Step 1 & 2: read everything, dedupe, trim to retention window ──
        history = read_price_history()
        if history.empty:
            logger.warning("consolidate_snapshots: read_price_history returned nothing, aborting")
            return False

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        # datetime column may be tz-aware (from fetcher.py's UTC-localized
        # index) — compare accordingly rather than assume naive.
        dt_col = pd.to_datetime(history["datetime"], utc=True)
        before_trim = len(history)
        history = history[dt_col >= cutoff].reset_index(drop=True)

        logger.info(
            f"consolidate_snapshots: {before_trim} rows read | "
            f"{len(history)} rows within {retention_days}-day retention window "
            f"({before_trim - len(history)} trimmed as too old)"
        )

        if history.empty:
            logger.warning(
                "consolidate_snapshots: all rows fell outside the retention "
                "window — this would delete everything, aborting instead of proceeding"
            )
            return False

        # ── Step 3: write ONE new consolidated snapshot ─────────────────────
        consolidation_ts = datetime.now(timezone.utc).strftime(FILENAME_TS_FORMAT)
        write_ok = _write_consolidated(history, consolidation_ts)

        if not write_ok:
            logger.warning(
                "consolidate_snapshots: failed to write consolidated snapshot — "
                "aborting BEFORE deleting old files, so no data is lost"
            )
            return False

        # ── Step 4: delete every file that existed before this pass ────────
        to_delete = [f"{PREFIX}/{f['name']}" for f in pre_consolidation_files]

        resp = requests.delete(
            f"{base_url}/object/{BUCKET_NAME}",
            headers={**headers, "Content-Type": "application/json"},
            json={"prefixes": to_delete},
            timeout=30,
        )
        resp.raise_for_status()

        logger.info(
            f"consolidate_snapshots: complete | "
            f"Wrote 1 new consolidated snapshot ({len(history)} rows) | "
            f"Deleted {len(to_delete)} old chunk files"
        )
        return True

    except Exception as e:
        logger.warning(f"consolidate_snapshots failed: {e} — continuing without it")
        return False


def _write_consolidated(df: pd.DataFrame, consolidation_ts: str) -> bool:
    """
    Write the fully-consolidated DataFrame as chunked Parquet files
    under a CONSOLIDATED_PREFIX-tagged filename, distinguishing it from
    individual per-run snapshots (both are read identically by
    read_price_history() — the tag matters only for consolidate_snapshots()
    itself to know what to delete on the NEXT consolidation pass).

    Args:
        df               : The trimmed, de-duplicated combined history
        consolidation_ts : UTC timestamp string for this consolidation
                           pass, used in the filename

    Returns:
        True if all chunks wrote successfully, False otherwise
    """
    try:
        base_url, headers = _get_config()

        n_chunks = max(1, (len(df) + CHUNK_ROWS - 1) // CHUNK_ROWS)
        upload_headers = {
            **headers,
            "Content-Type": "application/octet-stream",
            "x-upsert"    : "true",
        }

        chunks_written = 0

        for i in range(n_chunks):
            chunk = df.iloc[i * CHUNK_ROWS : (i + 1) * CHUNK_ROWS]
            if chunk.empty:
                continue

            buffer = io.BytesIO()
            chunk.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
            buffer.seek(0)
            raw_bytes = buffer.read()

            path = f"{PREFIX}/{CONSOLIDATED_PREFIX}_{consolidation_ts}_part{i:03d}.parquet"

            resp = requests.post(
                f"{base_url}/object/{BUCKET_NAME}/{path}",
                headers=upload_headers,
                data=raw_bytes,
                timeout=60,
            )

            if resp.status_code not in (200, 201):
                logger.warning(
                    f"_write_consolidated: chunk {i} failed | "
                    f"HTTP {resp.status_code} — {resp.text[:300]}"
                )
                continue

            chunks_written += 1

        return chunks_written == n_chunks

    except Exception as e:
        logger.warning(f"_write_consolidated failed: {e}")
        return False


# =============================================================================
# LEGACY PRUNE — superseded by consolidate_snapshots() above, kept only
# for reference. NOT called anywhere in this project.
# =============================================================================

def prune_old_snapshots(max_days: int = 60) -> int:
    """
    Delete Parquet snapshot chunks older than max_days from Supabase Storage.

    ⚠️ SUPERSEDED by consolidate_snapshots() above, which is the
    function actually wired into run_pipeline_cloud.py. This blunt,
    age-based version is kept only for reference — do NOT call it. It
    has a real correctness gap consolidate_snapshots() was written to
    avoid: a chunk's own filename age doesn't tell you whether every
    row inside it is actually outside the retention window — deleting
    purely by chunk age risks losing in-window data if any run's chunk
    boundaries don't line up cleanly with the age cutoff.

    Kept here only in case retention policy changes deliberately in
    the future (e.g. capping to a fixed rolling window once the model
    is mature) — not called automatically by anything in this project.

    Args:
        max_days: Keep snapshots within this many days, delete the rest

    Returns:
        Number of chunk files deleted
    """
    try:
        base_url, headers = _get_config()

        files = _list_snapshot_files(base_url, headers)
        if not files:
            return 0

        cutoff    = datetime.now(timezone.utc) - timedelta(days=max_days)
        to_delete = []

        for f in files:
            name = f["name"]
            file_ts = _parse_run_timestamp_from_filename(name)
            if file_ts is None:
                continue

            if file_ts.replace(tzinfo=timezone.utc) < cutoff:
                to_delete.append(f"{PREFIX}/{name}")

        if not to_delete:
            logger.info("prune_old_snapshots: nothing to prune")
            return 0

        resp = requests.delete(
            f"{base_url}/object/{BUCKET_NAME}",
            headers={**headers, "Content-Type": "application/json"},
            json={"prefixes": to_delete},
            timeout=30,
        )
        resp.raise_for_status()

        logger.info(f"prune_old_snapshots: deleted {len(to_delete)} old snapshot chunks")
        return len(to_delete)

    except Exception as e:
        logger.warning(f"prune_old_snapshots failed: {e} — continuing")
        return 0
