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

NOT A ROLLING WINDOW — this one accumulates and keeps ALL history by
design. The whole point of accumulating snapshots run over run is full
regime coverage in training, so pruning old snapshots here would
actively work against that goal. read_price_history() defaults to
reading everything ever written.

WHY WE DON'T DELETE SNAPSHOTS AFTER EACH PIPELINE RUN (a real question,
not a hypothetical — worth stating explicitly): these snapshots are
NOT a cache for the run that wrote them. Postgres never holds raw
prices at all (see above) — this bucket is the ONLY persistent copy of
price history that exists anywhere in this architecture.
train_models.py's read_price_history() reconstructs the FULL backfill
window from ALL accumulated snapshots — that is the entire mechanism
by which training gets multi-regime historical data. Deleting a
snapshot right after its run would permanently lose that day's data —
not re-fetchable later, since yfinance's history window is finite
(fetcher.py's 729-day 1H ceiling). It would also break fetcher.py's
incremental fetch design: smart_fetch() deliberately pulls only tiny
NEW deltas on every run after the first, relying on accumulated
Storage history for everything older. Delete-per-run would force a
choice between re-fetching the full 729-day history every single run
(defeating incremental fetch entirely) or permanently losing all
history older than one run (breaking multi-regime training). Neither
is acceptable, so snapshots accumulate indefinitely by design, exactly
as this module's original stock-project version also concluded.

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
    write_snapshot(raw_df, run_timestamp)   # called from run_pipeline_cloud.py
    df = read_price_history()               # called from train_models.py — reads ALL history
    # prune_old_snapshots() exists but should NOT be called in this
    # project's normal flow — see docstring on that function. NOT
    # wired into run_pipeline_cloud.py, matching this module's own
    # design intent (accumulate forever, prune only on a deliberate,
    # separate retention-policy decision later).

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
# PRUNE — exists but is NOT called anywhere in this project's normal flow
# =============================================================================

def prune_old_snapshots(max_days: int = 60) -> int:
    """
    Delete Parquet snapshot chunks older than max_days from Supabase Storage.

    ⚠️ DO NOT call this in this project's normal pipeline flow, and it
    is deliberately NOT wired into run_pipeline_cloud.py. This
    project's entire purpose for accumulating snapshots is preserving
    FULL history for regime coverage in training — calling this would
    delete exactly the data the project depends on. See this module's
    top-level docstring, "WHY WE DON'T DELETE SNAPSHOTS AFTER EACH
    PIPELINE RUN", for the full reasoning (incremental fetch relies on
    this accumulated history existing; deleting it forces either a
    full 729-day re-fetch every run or permanent data loss).

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
