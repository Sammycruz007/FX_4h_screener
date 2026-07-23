"""
run_pipeline_cloud.py
----------------------
Main orchestrator for the FX Scanner cloud pipeline (GitHub Actions).

Runs TWICE daily per config.yaml's scheduler.run_times (["12:00",
"20:00"] UTC — see that config section's comment for why these two
times specifically). Each run:
   1. Fetches 4H OHLC data (1H fetched + session-anchored resampled,
      see data/fetcher.py) for the fixed 28-pair universe
   2. Snapshots raw prices to Supabase Storage, then consolidates
      accumulated snapshots (see data/storage_cloud.py)
   3. Runs LinReg + SMC + ADX per-pair, and CSI once across the whole
      aligned universe (CSI is inherently cross-pair, see
      engines/csi.py — cannot be computed inside a per-pair loop)
   4. Writes indicator results to Supabase Postgres
   5. Runs the scanner (slope + SD-zone gate only — see
      scanner/screener.py's module docstring for why there's no
      market/sector waterfall for FX)
   6. Scores candidates with the Signal Ranker (CSI features included,
      candlestick-at-extreme computed HERE — see STEP 8.5 below for
      why this can't happen earlier)
   7. Writes ranked scan results to Supabase Postgres

WHAT'S DROPPED FROM THE STOCK PROJECT'S VERSION OF THIS FILE:
   - Stage 1 filter (Step 4) — no filtering funnel for FX, fixed
     28-pair universe scanned directly (see fetcher.py/screener.py)
   - Sector metadata fetch (Step 5) — no sectors for currencies
   - Volume Classifier scoring — no real volume data exists for FX
     (decentralized OTC market, see fetcher.py's docstring); Signal
     Ranker is the only model
   - GFT watchlist filtering + separate write (Step 11) — a 15-stock
     evaluation-account diagnostic with no FX equivalent, dropped
     entirely per project decision (see ml/signal_ranker.py's
     module docstring)
   - prune_old_snapshots() — never actually called in the stock
     version either in principle (data/storage_cloud.py's docstring
     says not to), but the stock run_pipeline_cloud.py called it
     anyway; that call is NOT carried over here. Replaced by
     consolidate_snapshots(), which IS correctly wired in below
     (see STEP 3.5) — it bounds Storage growth safely, unlike a
     blunt age-based prune (see storage_cloud.py's own docstring for
     why the difference matters).

WHY CANDLESTICK FEATURES ARE COMPUTED AFTER THE SCANNER, NOT WITH THE
OTHER INDICATOR ENGINES (STEP 7):
   engines/candlestick.py's hammer_at_extreme is direction-aware — it
   needs to know whether a LONG or a SHORT is being evaluated for a
   given (pair, datetime) to decide whether a detected pattern "counts"
   (see that module's docstring). But direction doesn't exist as a
   concept until the scanner (Step 8) has actually identified which
   pairs are long candidates vs. short candidates. Computing candlestick
   features for a pair in BOTH directions before knowing which one(s)
   apply would be pure wasted work for the ~24 of 28 pairs that don't
   qualify as a candidate in either direction on a given run. So
   candlestick features are computed in STEP 8.5, strictly after the
   scanner and strictly only for the actual candidates it produced —
   this is the one point in the pipeline where the per-pair engine
   sequence genuinely depends on the scanner's output, not the other
   way around.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other file in this project.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone
import pandas as pd
import yaml

from data.fetcher import get_full_universe, smart_fetch, to_pair_dict
from data.database_cloud import (
    initialise_database,
    write_indicator_results,
    write_scan_results,
)
from data.storage_cloud import write_snapshot, consolidate_snapshots

from engines.linreg import compute_linreg_latest
from engines.smc    import compute_smc
from engines.adx    import compute_adx_latest
from engines.csi    import run_csi_engine
from engines.candlestick import compute_candlestick_latest

from scanner.screener import run_scanner
from ml.signal_ranker  import score_candidates, load_signal_ranker

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config      = _load_config()
STORAGE_CFG = config["storage"]

RETENTION_DAYS = STORAGE_CFG["retention_days"]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    run_start   = datetime.now(timezone.utc)
    run_dt_str  = run_start.isoformat()

    logger.info("=" * 70)
    logger.info("FX SCANNER — CLOUD PIPELINE RUN")
    logger.info(f"Run datetime (UTC): {run_dt_str}")
    logger.info("=" * 70)

    # ── STEP 1: Initialise Supabase ─────────────────────────────────────────
    logger.info("\n[STEP 1] Initialising Supabase...")
    try:
        initialise_database()
        logger.info("Supabase initialised")
    except Exception as e:
        logger.critical(f"Supabase initialisation failed: {e}", exc_info=True)
        return

    # ── STEP 2: Fixed 28-pair universe (no discovery/filtering funnel) ──────
    logger.info("\n[STEP 2] Loading fixed FX pair universe...")
    try:
        tickers = get_full_universe()
        logger.info(f"Universe: {len(tickers)} pairs")
    except Exception as e:
        logger.critical(f"Universe load failed: {e}", exc_info=True)
        return

    # ── STEP 3: Fetch 4H OHLC (1H fetched + session-anchored resampled) ─────
    logger.info(
        "\n[STEP 3] Fetching 4H OHLC data "
        "(full 729-day backfill or incremental, see fetcher.py)..."
    )
    try:
        raw_df = smart_fetch(tickers)
        if raw_df.empty:
            logger.error("No OHLC data fetched. Aborting pipeline run.")
            return
        logger.info(
            f"Fetched {len(raw_df)} rows for {raw_df['pair'].nunique()} pairs"
        )
    except Exception as e:
        logger.critical(f"Fetch failed: {e}", exc_info=True)
        return

    # ── STEP 3.4: Snapshot raw prices to Supabase Storage ───────────────────
    logger.info("\n[STEP 3.4] Writing raw price snapshot to Supabase Storage...")
    try:
        snapshot_ok = write_snapshot(raw_df, run_timestamp=run_start)
        if snapshot_ok:
            logger.info("Snapshot written successfully")
        else:
            logger.warning(
                "Snapshot write failed or partial — continuing pipeline "
                "anyway (non-fatal, see storage_cloud.py's docstring)"
            )
    except Exception as e:
        logger.warning(f"Snapshot write raised an exception: {e} — continuing")

    # ── STEP 3.5: Consolidate accumulated snapshots ─────────────────────────
    # Bounds both Storage bytes AND object count without ever losing
    # in-window data — see storage_cloud.py's consolidate_snapshots()
    # docstring for the full mechanics and why this replaces the stock
    # project's blunt age-based prune_old_snapshots (never actually
    # wired into that project's orchestrator either, despite being
    # called there — see this module's docstring).
    logger.info(
        f"\n[STEP 3.5] Consolidating Storage snapshots "
        f"(retention: {RETENTION_DAYS} days)..."
    )
    try:
        consolidate_ok = consolidate_snapshots(retention_days=RETENTION_DAYS)
        if consolidate_ok:
            logger.info("Snapshot consolidation complete")
        else:
            logger.warning(
                "Snapshot consolidation failed or was skipped — continuing "
                "pipeline anyway (non-fatal; old snapshots simply "
                "accumulate one more run's worth until the next success)"
            )
    except Exception as e:
        logger.warning(f"Consolidation raised an exception: {e} — continuing")

    # ── STEP 6: Reshape long-form fetch output into per-pair dict ───────────
    # (Numbered to match the stock project's step numbering where the
    # analogous "load price data for engines" step lived — Stage 1/
    # sector steps 4-5 don't exist for FX, so this is the next real step.)
    logger.info("\n[STEP 6] Reshaping price data for indicator engines...")
    try:
        tickers_data = to_pair_dict(raw_df)
        if not tickers_data:
            logger.error("to_pair_dict produced an empty dict. Aborting.")
            return
        logger.info(f"Loaded data for {len(tickers_data)} pairs")
    except Exception as e:
        logger.critical(f"Reshape failed: {e}", exc_info=True)
        return

    # ── STEP 7: Run indicator engines ────────────────────────────────────────
    # LinReg + SMC + ADX are genuinely per-pair — computed in a loop.
    # CSI is NOT — it requires all 28 pairs' aligned history at once
    # (see engines/csi.py's module docstring), so it runs ONCE across
    # the whole universe, separately from the per-pair loop below. This
    # mirrors the exact same restructuring done in ml/train_models.py's
    # backfill, for the same reason.
    logger.info("\n[STEP 7] Running indicator engines...")

    scan_datetime = None
    indicator_rows = []

    for pair, df in tickers_data.items():
        if df.empty:
            continue

        latest_dt = str(df["datetime"].iloc[-1])
        if scan_datetime is None:
            scan_datetime = latest_dt

        lr = compute_linreg_latest(pair, df, latest_dt)
        if lr is None:
            continue

        smc = compute_smc(
            pair, df, latest_dt,
            sd1_lower=lr.get("sd1_lower"),
            sd3_lower=lr.get("sd3_lower"),
            sd1_upper=lr.get("sd1_upper"),
            sd3_upper=lr.get("sd3_upper"),
        )
        if smc is None:
            continue

        adx = compute_adx_latest(pair, df, latest_dt)
        if adx is None:
            continue

        indicator_rows.append({
            "pair"    : pair,
            "datetime": latest_dt,
            **{k: v for k, v in lr.items() if k not in ("pair", "datetime")},
            "smc_structure" : smc["smc_structure"],
            "has_valid_zone": smc["has_valid_zone"],
            "adx_value"     : adx["adx_value"],
            "plus_di"       : adx["plus_di"],
            "minus_di"      : adx["minus_di"],
        })

    if not indicator_rows:
        logger.error("No indicator rows computed for any pair. Aborting.")
        return

    indicator_df = pd.DataFrame(indicator_rows)
    logger.info(f"LinReg + SMC + ADX: {len(indicator_df)} pairs computed")

    # CSI — separate cross-pair phase, computed once for the whole universe
    try:
        csi_df = run_csi_engine(tickers_data, date=scan_datetime)
        logger.info(f"CSI: {len(csi_df)} pairs computed")
    except Exception as e:
        logger.warning(f"CSI engine failed: {e} — continuing with empty csi_df")
        csi_df = pd.DataFrame()

    # Merge CSI's 4 features onto indicator_df so write_indicator_results
    # persists them alongside LinReg/SMC/ADX in one row per pair.
    if not csi_df.empty:
        indicator_df = indicator_df.merge(
            csi_df[["pair", "csi_rs", "csi_diff_zscore", "csi_diff_roc", "csi_commodity_bloc"]],
            on="pair",
            how="left",
        )
    else:
        for col in ("csi_rs", "csi_diff_zscore", "csi_diff_roc", "csi_commodity_bloc"):
            indicator_df[col] = None

    try:
        rows_written = write_indicator_results(indicator_df)
        logger.info(f"Indicator results written: {rows_written} rows")
    except Exception as e:
        logger.error(f"Failed to write indicator results: {e}", exc_info=True)
        # Non-fatal — the scanner can still run on the in-memory indicator_df
        # even if the Supabase write failed.

    # ── STEP 8: Run scanner ───────────────────────────────────────────────────
    # Slope + SD-zone gate only — no market/sector waterfall for FX (see
    # scanner/screener.py's module docstring). This gate MUST match
    # ml/train_models.py's relaxed candidate gate exactly, or the model
    # is scored on a different definition of "candidate" than it was
    # trained on — verified identical via testing at build time.
    logger.info("\n[STEP 8] Running scanner...")
    try:
        candidates_df = run_scanner(indicator_df, datetime_str=scan_datetime)
        if candidates_df.empty:
            logger.info("No candidates found this run. Pipeline complete (nothing to score/write).")
            _log_pipeline_complete(run_start)
            return
        logger.info(
            f"Scanner found {len(candidates_df)} candidates | "
            f"Long: {(candidates_df['direction']=='long').sum()} | "
            f"Short: {(candidates_df['direction']=='short').sum()}"
        )
    except Exception as e:
        logger.critical(f"Scanner failed: {e}", exc_info=True)
        return

    # ── STEP 8.5: Candlestick-at-extreme, computed HERE — see module ────────
    # docstring's "WHY CANDLESTICK FEATURES ARE COMPUTED AFTER THE
    # SCANNER" section for the full reasoning. Only computed for actual
    # candidates, not the whole 28-pair universe — direction-aware, so
    # doing this earlier for every pair in both directions would be
    # mostly wasted work.
    logger.info("\n[STEP 8.5] Computing candlestick-at-extreme features for candidates...")
    hammer_results = []
    for _, cand_row in candidates_df.iterrows():
        pair      = cand_row["pair"]
        direction = cand_row["direction"]
        sd_pos    = float(cand_row["sd_position"])

        px = tickers_data.get(pair)
        if px is None or px.empty:
            continue

        result = compute_candlestick_latest(
            pair            = pair,
            df              = px,
            signal_datetime = scan_datetime,
            sd_position     = sd_pos,
            direction       = direction,
        )
        if result is not None:
            hammer_results.append({
                "pair"     : pair,
                "direction": direction,
                **result,
            })

    if hammer_results:
        hammer_df = pd.DataFrame(hammer_results)
        # Merge candlestick's raw pattern flags back onto indicator_df
        # (keyed by pair only — candlestick patterns aren't direction-
        # specific themselves, only the interaction feature is) so
        # write_indicator_results persists is_hammer/is_shooting_star
        # alongside everything else.
        pattern_flags = hammer_df[["pair", "is_hammer", "is_shooting_star"]].drop_duplicates(subset=["pair"])
        indicator_df = indicator_df.merge(pattern_flags, on="pair", how="left")
        try:
            write_indicator_results(indicator_df)
        except Exception as e:
            logger.warning(f"Failed to re-write indicator results with candlestick flags: {e}")
        logger.info(f"Candlestick features computed for {len(hammer_results)} candidate rows")
    else:
        logger.warning("No candlestick results computed for any candidate")

    # ── STEP 9: ML scoring (Signal Ranker only — no Volume Classifier) ──────
    logger.info("\n[STEP 9] Running Signal Ranker scoring...")
    try:
        pipeline = load_signal_ranker()
        if pipeline is None:
            logger.warning(
                "Signal Ranker model not found — candidates will be ranked "
                "by SD position only (see ml/signal_ranker.py's score_candidates "
                "fallback behaviour). Train the model via ml/train_models.py."
            )

        scored_df = score_candidates(
            candidates_df   = candidates_df,
            prices_df       = raw_df,
            indicators_df   = indicator_df,
            csi_df          = csi_df,
            signal_datetime = scan_datetime,
            pipeline        = pipeline,
        )
        logger.info("Signal Ranker scoring complete")
    except Exception as e:
        logger.error(f"ML scoring failed: {e} — writing candidates unscored", exc_info=True)
        scored_df = candidates_df

    # ── STEP 10: Write scan results to Supabase ─────────────────────────────
    logger.info("\n[STEP 10] Writing scan results to Supabase...")
    try:
        if not scored_df.empty:
            rows = write_scan_results(scored_df, run_datetime=scan_datetime)
            logger.info(f"Scan results written: {rows} candidates")
        else:
            logger.info("No candidates to write")
    except Exception as e:
        logger.error(f"Failed to write scan results: {e}", exc_info=True)

    _log_pipeline_complete(run_start)


def _log_pipeline_complete(run_start: datetime):
    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds() / 60
    logger.info("\n" + "=" * 70)
    logger.info("CLOUD PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}m")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
