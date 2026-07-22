"""
ml/train_models.py
-------------------
Backfills a rolling indicator history from raw 4H prices, generates
labels, builds the feature matrix, and trains the Signal Ranker.

WHAT'S DIFFERENT FROM THE STOCK PROJECT:
1. No Volume Classifier — no real volume data exists for FX
   (decentralized OTC market, see fetcher.py's docstring). This file
   trains exactly ONE model: the Signal Ranker.
2. No universe filtering — read_filtered_universe() doesn't exist for
   FX. The universe is the fixed 28-pair list from config.yaml's
   universe.pairs, read live (not hardcoded) — matching
   fetcher.py's get_full_universe().
3. CSI is computed as a SEPARATE backfill phase, not inside the
   per-pair backfill_ticker() loop. CSI is inherently cross-pair (see
   engines/csi.py's module docstring) — you cannot compute
   CSI_EUR from EURUSD's data alone, you need all 28 pairs' aligned
   history at once. LinReg/SMC/ADX stay in the per-pair parallel
   backfill (each is genuinely single-pair), but CSI runs once, after,
   across the whole universe via engines.csi.compute_csi_series().
4. Relaxed scan-hit gate is slope + SD-zone ONLY (two conditions, not
   three). The stock project's third gate was volume_signal ==
   accumulation/distribution — there is no FX substitute for this.
   CSI-RS was explicitly considered and rejected as a replacement
   HARD gate: the whole point of the stock project's own design was
   that has_valid_zone and CHoCH were demoted from hard gates to ML
   FEATURES so the model learns to weight them — volume was the one
   exception that stayed a hard gate. Making CSI a hard gate here
   would be an asymmetric, arbitrary choice with no better
   justification than any other feature (ADX, ATR ratio, candlestick)
   also being promoted to a gate. CSI stays a feature (already in
   SIGNAL_FEATURE_COLS), consistent with every other engine output in
   this pipeline. See project discussion for the full reasoning.
5. Column naming: 'pair'/'datetime' throughout, not 'ticker'/'date' —
   consistent with every other FX file in this project.

Every threshold/period/list below is read live from config.yaml —
nothing in this file hardcodes a value that config already owns.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

if os.getenv("SUPABASE_DB_URL"):
    from data.database_cloud import (
        initialise_database,
        write_model_metrics,
    )
    from data.storage_cloud import read_price_history
    _CLOUD_MODE = True
else:
    from data.database import (
        initialise_database,
        read_raw_prices,
        write_model_metrics,
    )
    _CLOUD_MODE = False

from engines.linreg import compute_linreg_latest, PERIOD as LINREG_PERIOD
from engines.smc    import compute_smc
from engines.adx    import compute_adx_latest
from engines.csi    import compute_csi_series
from ml.labeller  import label_scanner_hits
from ml.features  import build_signal_feature_matrix
from ml.signal_ranker import train_signal_ranker
from utils.logging import get_ml_logger

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
ML_CFG       = config["ml"]
UNIVERSE_CFG = config["universe"]
SCANNER_CFG  = config["scanner"]

FORWARD = ML_CFG["label_forward_periods"]
STRIDE  = ML_CFG["backfill_stride"]

MIN_SIGNAL_SAMPLES = ML_CFG["min_training_samples"]

# SD zone thresholds from scanner config
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]   # -1
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]   # -3
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]  # +1
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]  # +3

# Fixed 28-pair universe — read live from config, matching
# fetcher.py's get_full_universe(). No discovery/filtering funnel for FX.
PAIRS = UNIVERSE_CFG["pairs"]

# Cap for quick test runs — set to None for full universe
MAX_PAIRS_FOR_TRAINING = None


# =============================================================================
# ROLLING BACKFILL — per pair (LinReg, SMC, ADX only)
# CSI is NOT computed here — see the CSI BACKFILL section below and the
# module docstring for why it's a separate, cross-pair phase.
# =============================================================================

def backfill_pair(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run LinReg, SMC, and ADX engines on rolling windows of df.

    For each step i (i = LINREG_PERIOD .. len(df)-1, stride STRIDE):
        window = df.iloc[:i+1]
        dt     = df.iloc[i]["datetime"]
        -> compute_linreg_latest, compute_smc, compute_adx_latest

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'
        df  : Full OHLC DataFrame for this pair, sorted datetime ascending

    Returns:
        DataFrame with one row per backfilled (pair, datetime)
    """
    rows = []
    n    = len(df)

    # Need LINREG_PERIOD candles for LinReg + FORWARD candles for labelling
    if n < LINREG_PERIOD + FORWARD + 1:
        logger.debug(
            f"{pair} | Insufficient data for backfill: "
            f"{n} rows, need {LINREG_PERIOD + FORWARD + 1}"
        )
        return pd.DataFrame()

    for i in range(LINREG_PERIOD, n, STRIDE):
        window = df.iloc[: i + 1]
        dt     = str(df.iloc[i]["datetime"])

        lr = compute_linreg_latest(pair, window, dt)
        if lr is None:
            continue

        smc = compute_smc(
            pair, window, dt,
            sd1_lower = lr.get("sd1_lower"),
            sd3_lower = lr.get("sd3_lower"),
            sd1_upper = lr.get("sd1_upper"),
            sd3_upper = lr.get("sd3_upper"),
        )
        if smc is None:
            continue

        adx = compute_adx_latest(pair, window, dt)
        if adx is None:
            continue

        rows.append({
            "pair"          : pair,
            "datetime"      : dt,
            **{k: v for k, v in lr.items() if k not in ("pair", "datetime")},
            "smc_structure" : smc["smc_structure"],
            "has_valid_zone": smc["has_valid_zone"],
            "adx_value"     : adx["adx_value"],
            "plus_di"       : adx["plus_di"],
            "minus_di"      : adx["minus_di"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# RELAXED SCAN HIT DETECTION
# Two conditions only: slope + SD-zone. No third hard gate — CSI, ADX,
# ATR ratio, candlestick-at-extreme, and has_valid_zone are all ML
# FEATURES the model learns to weight, not filters that block a row
# from becoming a training example. See module docstring for the full
# reasoning on why CSI specifically was NOT promoted to a hard gate.
# =============================================================================

def _is_long_candidate_relaxed(row: pd.Series) -> bool:
    """
    Relaxed long candidate check for training data generation.
    Requires: slope up + price in -1 to -3 SD zone.
    Does NOT require: has_valid_zone, CHoCH absence, CSI direction
    agreement. These become ML features instead of hard filters.
    """
    if int(row.get("linreg_slope_up", 0)) != 1:
        return False

    sd_pos = float(row.get("price_sd_position", 0))
    if not (LONG_SD_MAX <= sd_pos <= LONG_SD_MIN):   # -3 <= sd <= -1
        return False

    return True


def _is_short_candidate_relaxed(row: pd.Series) -> bool:
    """
    Relaxed short candidate check for training data generation.
    Requires: slope down + price in +1 to +3 SD zone.
    """
    if int(row.get("linreg_slope_up", 1)) != 0:
        return False

    sd_pos = float(row.get("price_sd_position", 0))
    if not (SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX):   # +1 <= sd <= +3
        return False

    return True


def build_historical_scan_hits(
    indicators_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find all historical (pair, datetime) combinations that would have
    qualified as long or short candidates using the relaxed conditions.

    Unlike the stock project, there is no excluded-tickers set to
    build (no indices/sectors mixed into the universe — every row in
    indicators_history_df is a real, scannable pair).

    Args:
        indicators_history_df: Full backfilled indicator history

    Returns:
        DataFrame with columns [pair, datetime, direction]
    """
    hits = []
    for _, row in indicators_history_df.iterrows():
        if _is_long_candidate_relaxed(row):
            hits.append({
                "pair"     : row["pair"],
                "datetime" : row["datetime"],
                "direction": "long",
            })
        if _is_short_candidate_relaxed(row):
            hits.append({
                "pair"     : row["pair"],
                "datetime" : row["datetime"],
                "direction": "short",
            })

    logger.info(f"build_historical_scan_hits: {len(hits)} historical setups found")
    return pd.DataFrame(hits) if hits else pd.DataFrame(
        columns=["pair", "datetime", "direction"]
    )


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_training():
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    initialise_database()

    # Delete stale model to force clean retraining with new fixes
    from ml.signal_ranker import MODEL_PATH as SIG_PATH

    if SIG_PATH.exists():
        SIG_PATH.unlink()
        logger.info(f"Deleted stale model: {SIG_PATH}")

    # ── Step 1: Fixed 28-pair universe (no filtering funnel for FX) ─────────
    pairs = list(PAIRS)
    if MAX_PAIRS_FOR_TRAINING:
        pairs = pairs[:MAX_PAIRS_FOR_TRAINING]

    logger.info(
        f"Backfilling indicators for {len(pairs)} pairs "
        f"(stride={STRIDE}, linreg_period={LINREG_PERIOD})"
    )

    # ── Step 2: Rolling backfill per pair (PARALLELIZED) ────────────────────
    indicator_history = []
    prices_all        = []
    backfill_failed    = 0

    if _CLOUD_MODE:
        # Load the full accumulated history once from Supabase Storage,
        # then filter per pair in memory — avoids re-downloading on
        # every loop iteration. No days= window — this project needs
        # the full accumulated history for regime coverage, not a
        # recent-only slice.
        logger.info("Loading full price history from Supabase Storage...")
        full_history = read_price_history()
        logger.info(f"Loaded {len(full_history)} total rows across all snapshots")

        if full_history.empty:
            logger.error(
                "No price history snapshots available yet. "
                "Snapshots accumulate from fetcher.py's runs — "
                "wait for more runs before training."
            )
            return

        logger.info("Pre-grouping and sorting history by pair...")
        grouped_history = {
            name: group.sort_values("datetime").reset_index(drop=True)
            for name, group in full_history.groupby("pair")
        }

    # Phase A: Prepare per-pair DataFrames in memory
    logger.info("Preparing pair datasets for parallel processing...")
    pair_tasks = []
    for pair in pairs:
        if _CLOUD_MODE:
            df = grouped_history.get(pair, pd.DataFrame())
        else:
            df = read_raw_prices(pair)

        if df.empty:
            backfill_failed += 1
            continue

        pair_tasks.append((pair, df))

    # Phase B: Process tasks in parallel across available CPU cores
    logger.info(f"Launching parallel backfill with {len(pair_tasks)} workers...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(backfill_pair, task_pair, task_df): (task_pair, task_df)
            for task_pair, task_df in pair_tasks
        }

        for n, future in enumerate(as_completed(futures), 1):
            task_pair, task_df = futures[future]

            try:
                hist = future.result()
                prices_all.append(task_df)

                if hist is not None and not hist.empty:
                    indicator_history.append(hist)
                else:
                    backfill_failed += 1
            except Exception as e:
                logger.error(f"Worker crashed processing pair {task_pair}: {e}")
                backfill_failed += 1

            if n % 10 == 0:
                logger.info(
                    f"Backfill progress: {n}/{len(pair_tasks)} pairs processed | "
                    f"History rows collected: {sum(len(h) for h in indicator_history)}"
                )

    if not indicator_history:
        logger.error(
            "No indicator history produced — "
            "pairs may not have enough candles yet. "
            "Run the pipeline for more days to accumulate data."
        )
        return

    indicators_history_df = pd.concat(indicator_history, ignore_index=True)
    prices_all_df          = pd.concat(prices_all, ignore_index=True)

    logger.info(
        f"Per-pair backfill complete | "
        f"History rows: {len(indicators_history_df)} | "
        f"Pairs with history: {indicators_history_df['pair'].nunique()} | "
        f"Failed: {backfill_failed}"
    )

    # ── Step 3: CSI backfill — SEPARATE cross-pair phase ────────────────────
    # CSI cannot be computed inside backfill_pair()'s per-pair loop — it
    # requires all 28 pairs' aligned history simultaneously (see
    # engines/csi.py's module docstring). Requires a dict[pair -> df] of
    # the SAME per-pair price DataFrames used above, built once here.
    logger.info("Computing CSI series across the full aligned universe...")

    prices_by_pair = {
        pair: group.sort_values("datetime").reset_index(drop=True)
        for pair, group in prices_all_df.groupby("pair")
    }

    csi_series_df = compute_csi_series(prices_by_pair)

    if csi_series_df.empty:
        logger.warning(
            "CSI backfill produced no rows — csi_rs, csi_diff_zscore, "
            "csi_diff_roc, and csi_commodity_bloc will all be 0.0 for "
            "all training examples. This will weaken the model but "
            "training will still proceed."
        )
    else:
        logger.info(f"CSI backfill complete | {len(csi_series_df)} (pair, datetime) rows")

    # ── Step 4: Build historical scan hits for Signal Ranker training ──────
    logger.info("Building historical scan hits for Signal Ranker training...")

    scan_hits = build_historical_scan_hits(indicators_history_df)

    if scan_hits.empty:
        logger.warning(
            "No historical scan hits found. "
            "This means no pairs had the right combination of: "
            "slope up/down + price in SD zone. "
            "Run the pipeline for more days to accumulate diverse market conditions."
        )
        logger.info("=" * 60)
        logger.info("ML TRAINING PIPELINE COMPLETE (Signal Ranker skipped)")
        logger.info("=" * 60)
        return

    logger.info(
        f"Scan hits found: {len(scan_hits)} | "
        f"Longs: {(scan_hits['direction']=='long').sum()} | "
        f"Shorts: {(scan_hits['direction']=='short').sum()}"
    )

    # Generate labels for scan hits
    sig_labels = label_scanner_hits(prices_all_df, indicators_history_df, scan_hits)

    if sig_labels.empty:
        logger.warning("No signal labels generated — Signal Ranker skipped")
        return

    pos = (sig_labels["label"] == 1).sum()
    neg = (sig_labels["label"] == 0).sum()
    logger.info(
        f"Signal labels | Total: {len(sig_labels)} | "
        f"Positive: {pos} ({pos/len(sig_labels)*100:.1f}%) | "
        f"Negative: {neg} ({neg/len(sig_labels)*100:.1f}%)"
    )

    # ── Step 5: Build feature matrix ────────────────────────────────────────
    sig_matrix = build_signal_feature_matrix(
        prices_df      = prices_all_df,
        indicators_df  = indicators_history_df,
        labels_df      = sig_labels,
        csi_series_df  = csi_series_df,
    )

    if sig_matrix.empty:
        logger.warning("Signal feature matrix is empty — Signal Ranker skipped")
        return

    logger.info(f"Signal feature matrix: {len(sig_matrix)} rows")

    # ── Step 6: Train Signal Ranker ─────────────────────────────────────────
    # Allow even small sample sizes (model still ranks by probability) —
    # temporarily lower the module's MIN_SAMPLES threshold rather than
    # skip training outright, restoring it afterward.
    if len(sig_matrix) < MIN_SIGNAL_SAMPLES:
        logger.warning(
            f"Signal Ranker has only {len(sig_matrix)} training samples "
            f"(minimum is {MIN_SIGNAL_SAMPLES}). "
            f"Temporarily lowering threshold to train anyway."
        )
        import ml.signal_ranker as sr_module
        original_min = sr_module.MIN_SAMPLES
        sr_module.MIN_SAMPLES = max(2, len(sig_matrix))

    try:
        sig_pipeline, sig_metrics = train_signal_ranker(sig_matrix)
        write_model_metrics(
            model_name = sig_metrics["model_name"],
            train_date = sig_metrics["train_date"],
            precision  = sig_metrics["precision"],
            auc_roc    = sig_metrics["auc_roc"],
            n_samples  = sig_metrics["n_train"] + sig_metrics["n_test"],
            recall     = sig_metrics.get("recall", 0.0),
            pr_auc     = sig_metrics.get("pr_auc", 0.0),
        )

        logger.info(
            f"Signal Ranker trained | "
            f"Precision: {sig_metrics['precision']:.4f} | "
            f"AUC-ROC: {sig_metrics['auc_roc']:.4f} | "
        )
    finally:
        if len(sig_matrix) < MIN_SIGNAL_SAMPLES:
            sr_module.MIN_SAMPLES = original_min

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()
