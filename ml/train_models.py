"""
ml/train_models.py
-------------------
Backfills weekly indicator history, generates directional labels,
builds feature matrices, and trains FIVE separate basket-grouped
directional models (one XGBoost classifier per currency basket),
instead of one Signal Ranker across all 28 pairs.

WHAT'S DIFFERENT FROM THE PRIOR (SIGNAL RANKER) VERSION OF THIS FILE —
THIS IS A COMPLETE REDESIGN, NOT AN INCREMENTAL CHANGE:

1. FIVE MODELS, NOT ONE. Per project decision: 28 pairs are grouped
   into 5 currency baskets (defined in config.yaml's universe.baskets),
   and ONE model is trained PER BASKET rather than one model across
   all 28 pairs pooled together. Reasoning: JPY-cross pairs share
   structural drivers (BOJ policy, carry dynamics) genuinely different
   from CHF crosses (SNB policy, safe-haven flows) or CAD crosses (oil
   sensitivity) — a single pooled model has to learn one function
   averaging over several different regimes; basket models let each
   specialize. See project discussion for the full reasoning and the
   basket definitions themselves (verified against the real 28-pair
   universe — no typos, no gaps, no duplicates, across all 5 baskets).

2. NO SCANNER / CANDIDATE GATE AT ALL. The prior version's
   _is_long_candidate_relaxed / _is_short_candidate_relaxed /
   build_historical_scan_hits are GONE ENTIRELY. There is no "setup"
   concept anymore — every weekly candle for every pair gets a
   directional label and a feature row, unconditionally. This mirrors
   the reference EURUSD project's own design ("every pair, every
   cycle") and follows directly from the label redefinition in
   ml/labeller.py (label_direction: a plain "will price be higher in
   HORIZON candles" check, not "did a flagged setup hit its target").

3. NO LINREG, NO SMC, ANYWHERE. backfill_pair() previously ran LinReg +
   SMC + ADX per pair. LinReg and SMC are dropped from this project's
   scope entirely (per project decision) — this file's backfill now
   runs ONLY ADX per pair. CSI stays exactly as before: a separate,
   cross-pair phase (see CSI BACKFILL section) — that was never
   LinReg/SMC-dependent in the first place.

4. WEEKLY TIMEFRAME, NOT 4H. data/fetcher.py now fetches native Weekly
   bars (see that module's docstring) — no more session-anchored 4H
   resampling. This file's backfill loop and its minimum-row
   requirements are sized in weekly-candle units, not 4H-candle units.
   ml.label_forward_periods is now 2 (2 WEEKLY candles ahead), read
   from the SAME config key as before — see ml/labeller.py's module
   docstring for why this is a repurposed key, not a new one.

5. Column naming: 'pair'/'datetime' throughout — unchanged from every
   other file in this project.

6. MACRO DRIVER FEATURES ADDED (per project decision, after comparing
   against the EURUSD reference project's feature set). Macro history
   (DXY, gold, US10Y, WTI, VIX) is fetched and stored through the
   IDENTICAL path as FX pairs (see data/fetcher.py's smart_fetch_macro),
   so in cloud mode it's already present in read_price_history()'s
   result — this file just pulls it out by symbol name rather than
   re-fetching it. engines/macro.py's get_macro_features_for_basket()
   is called ONCE PER BASKET (basket-scoped, unlike CSI which is
   universe-scoped) and threaded into build_directional_feature_matrix.
   Each basket's model now also uses ml.features.get_directional_feature_cols
   (basket_name), NOT the old flat DIRECTIONAL_FEATURE_COLS constant,
   since macro columns differ per basket (12 for usd, 4 for the rest).

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

from engines.adx import compute_adx_latest
from engines.csi import compute_csi_series
from engines.macro import get_macro_features_for_basket
from ml.labeller  import label_direction
from ml.features  import build_directional_feature_matrix, compute_atr_series, ATR_FAST_PERIOD
from ml.signal_ranker import train_directional_model
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
ADX_CFG      = config["adx"]
MACRO_CFG    = config["macro"]

HORIZON = ML_CFG["label_forward_periods"]   # 2 weekly candles, see labeller.py
STRIDE  = ML_CFG["backfill_stride"]

MIN_SAMPLES_PER_BASKET = ML_CFG["min_training_samples"]

# ADX needs its own warm-up window for a meaningful reading — same
# reasoning as the old LINREG_PERIOD warm-up guard, just against ADX's
# own period instead (LinReg no longer exists to size this against).
ADX_PERIOD = ADX_CFG["period"]
# A small multiple of ADX_PERIOD gives ADX's own smoothing enough
# history to stabilise before we trust its output.
MIN_ROWS_FOR_BACKFILL = ADX_PERIOD * 3

# Fixed 28-pair universe, grouped into 5 currency baskets — read live
# from config, matching fetcher.py's get_full_universe() and the
# project's basket-model decision.
PAIRS   = UNIVERSE_CFG["pairs"]
BASKETS = UNIVERSE_CFG["baskets"]   # dict: basket_name -> list of pairs

# Macro driver config — same source engines/macro.py reads, used here
# to know which storage symbols (DXY, GOLD, US10Y, WTI, VIX) to pull
# out of the price history that's already been read for FX pairs
# (macro symbols are fetched and stored through the identical
# write_raw_prices path as FX pairs — see data/fetcher.py's
# smart_fetch_macro — so they're already present in read_price_history()'s
# result, just under a macro 'pair' name instead of an FX pair name).
MACRO_DRIVERS      = MACRO_CFG["drivers"]          # {"DXY": {...}, ...}
MACRO_BASKET_MAP    = MACRO_CFG["basket_drivers"]   # {"usd": ["DXY", ...], ...}
MACRO_STORAGE_SYMBOLS = sorted({
    driver_cfg["storage_symbol"] for driver_cfg in MACRO_DRIVERS.values()
})

# Cap for quick test runs — set to None for full universe
MAX_PAIRS_FOR_TRAINING = None


# =============================================================================
# ROLLING BACKFILL — per pair (ADX only)
# LinReg and SMC are GONE — see module docstring point 3. CSI is NOT
# computed here either — see the CSI BACKFILL section below.
# =============================================================================

def backfill_pair(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ADX engine on rolling windows of df.

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'
        df  : Full OHLC DataFrame for this pair, sorted datetime ascending

    Returns:
        DataFrame with one row per backfilled (pair, datetime)
    """
    rows = []
    n    = len(df)

    if n < MIN_ROWS_FOR_BACKFILL + HORIZON + 1:
        logger.debug(
            f"{pair} | Insufficient data for backfill: "
            f"{n} rows, need {MIN_ROWS_FOR_BACKFILL + HORIZON + 1}"
        )
        return pd.DataFrame()

    for i in range(MIN_ROWS_FOR_BACKFILL, n, STRIDE):
        window = df.iloc[: i + 1]
        dt     = str(df.iloc[i]["datetime"])

        adx = compute_adx_latest(pair, window, dt)
        if adx is None:
            continue

        rows.append({
            "pair"      : pair,
            "datetime"  : dt,
            "adx_value" : adx["adx_value"],
            "plus_di"   : adx["plus_di"],
            "minus_di"  : adx["minus_di"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN TRAINING PIPELINE — trains 5 separate basket models
# =============================================================================

def run_training():
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE STARTED (basket-grouped directional models)")
    logger.info("=" * 60)

    initialise_database()

    pairs = list(PAIRS)
    if MAX_PAIRS_FOR_TRAINING:
        pairs = pairs[:MAX_PAIRS_FOR_TRAINING]

    logger.info(
        f"Backfilling ADX for {len(pairs)} pairs across {len(BASKETS)} "
        f"baskets (stride={STRIDE}, horizon={HORIZON} weekly candles)"
    )

    prices_all        = []
    indicator_history  = []
    backfill_failed    = 0

    if _CLOUD_MODE:
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

    # ── Macro driver history — same storage path as FX pairs (see
    # data/fetcher.py's smart_fetch_macro), just keyed by a macro
    # symbol name (DXY, GOLD, US10Y, WTI, VIX) instead of an FX pair.
    # In cloud mode this data is ALREADY present in grouped_history
    # (read_price_history() reads everything in storage, macro symbols
    # included) — no separate fetch needed here, just pulled out by name.
    logger.info(f"Loading macro driver history: {MACRO_STORAGE_SYMBOLS}")
    macro_data = {}
    for storage_symbol in MACRO_STORAGE_SYMBOLS:
        if _CLOUD_MODE:
            macro_df = grouped_history.get(storage_symbol, pd.DataFrame())
        else:
            macro_df = read_raw_prices(storage_symbol)

        if macro_df.empty:
            logger.warning(
                f"No history found for macro driver '{storage_symbol}' — "
                f"any basket using it will get 0.0 for its features from "
                f"this driver (see engines/macro.py's fallback behaviour)"
            )
        macro_data[storage_symbol] = macro_df

    macro_rows_total = sum(len(df) for df in macro_data.values())
    logger.info(f"Macro driver history loaded | {macro_rows_total} total rows across {len(MACRO_STORAGE_SYMBOLS)} symbols")

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

    logger.info(f"Launching parallel ADX backfill with {len(pair_tasks)} workers...")

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
            "pairs may not have enough weekly candles yet."
        )
        return

    indicators_history_df = pd.concat(indicator_history, ignore_index=True)
    prices_all_df          = pd.concat(prices_all, ignore_index=True)

    logger.info(
        f"Per-pair ADX backfill complete | "
        f"History rows: {len(indicators_history_df)} | "
        f"Pairs with history: {indicators_history_df['pair'].nunique()} | "
        f"Failed: {backfill_failed}"
    )

    # ── CSI backfill — SEPARATE cross-pair phase, computed ONCE across the
    # WHOLE universe (not per-basket — CSI needs all 28 pairs' aligned
    # history regardless of which basket a pair ends up training in) ────
    logger.info("Computing CSI series across the full aligned universe...")

    prices_by_pair = {
        pair: group.sort_values("datetime").reset_index(drop=True)
        for pair, group in prices_all_df.groupby("pair")
    }

    csi_series_df = compute_csi_series(prices_by_pair)

    if csi_series_df.empty:
        logger.warning(
            "CSI backfill produced no rows — all 4 CSI features will be "
            "0.0 for all training examples across every basket."
        )
    else:
        logger.info(f"CSI backfill complete | {len(csi_series_df)} (pair, datetime) rows")

    # ── ATR (fast) — computed ONCE PER PAIR across each pair's full
    # sorted history, then reassembled back into prices_all_df.
    #
    # ml/labeller.py's label_direction() needs an atr_fast[t] value for
    # EVERY historical row t (it's the move-size threshold: a row only
    # labels BUY/SELL if the net move over the horizon clears
    # min_move_atr_multiple * atr_fast[t]). features.py's _compute_atr()
    # can't supply this — it deliberately returns a single scalar "ATR
    # as of the latest row," built for live/scoring use where a caller
    # has already sliced px down to "up to right now" for one pair. Use
    # compute_atr_series() instead, the rolling per-row counterpart
    # (same True Range formula, rolling(window).mean() instead of
    # .tail().mean(), same no-look-ahead guarantee — row t only uses
    # rows <= t).
    logger.info(
        f"Computing rolling ATR ({ATR_FAST_PERIOD}-period) per pair for "
        f"the labeller's move-size threshold..."
    )

    for pair, px in prices_by_pair.items():
        px["atr_fast"] = compute_atr_series(px, ATR_FAST_PERIOD)

    prices_all_df = pd.concat(prices_by_pair.values(), ignore_index=True)

    # ── Macro features — computed ONCE PER BASKET (not per-pair, not
    # cross-basket like CSI) — each basket only needs its OWN
    # configured driver(s)' history (see config.yaml's
    # macro.basket_drivers), so this is naturally basket-scoped rather
    # than universe-scoped the way CSI is.
    logger.info("Computing macro driver features per basket...")

    macro_features_by_basket = {}
    for basket_name in BASKETS:
        basket_macro_df = get_macro_features_for_basket(basket_name, macro_data)
        macro_features_by_basket[basket_name] = basket_macro_df

        if basket_macro_df.empty:
            logger.warning(
                f"{basket_name}: no macro features available — all macro "
                f"driver columns will be 0.0 for this basket's training "
                f"examples"
            )
        else:
            logger.info(
                f"{basket_name}: macro features ready | "
                f"{len(basket_macro_df)} rows | "
                f"columns: {list(basket_macro_df.columns)}"
            )

    # ── Directional labels — UNCONDITIONAL, every row, every pair. No
    # scanner gate, no candidate concept.
    logger.info("Generating directional labels for the full universe...")

    labels_df = label_direction(prices_all_df)

    if labels_df.empty:
        logger.error(
            "No directional labels generated at all — cannot train any "
            "basket model."
        )
        return

    pos = (labels_df["label"] == 1).sum()
    neg = (labels_df["label"] == 0).sum()
    logger.info(
        f"Directional labels | Total: {len(labels_df)} | "
        f"Up: {pos} ({pos/len(labels_df)*100:.1f}%) | "
        f"Down: {neg} ({neg/len(labels_df)*100:.1f}%)"
    )

    # ── Train ONE model PER BASKET. CSI/ADX/prices are all pre-computed
    # above ONCE for the whole universe; this loop just FILTERS to each
    # basket's pairs before building that basket's own feature matrix.
    basket_results = {}

    for basket_name, basket_pairs in BASKETS.items():
        logger.info("=" * 60)
        logger.info(f"BASKET: {basket_name} | Pairs: {basket_pairs}")
        logger.info("=" * 60)

        basket_labels = labels_df[labels_df["pair"].isin(basket_pairs)]
        basket_prices = prices_all_df[prices_all_df["pair"].isin(basket_pairs)]

        if basket_labels.empty:
            logger.warning(
                f"{basket_name}: no labels for any pair in this basket — skipped"
            )
            basket_results[basket_name] = {"status": "skipped_no_labels"}
            continue

        logger.info(f"{basket_name}: {len(basket_labels)} labelled examples")

        basket_matrix = build_directional_feature_matrix(
            prices_df      = basket_prices,
            labels_df      = basket_labels,
            csi_series_df  = csi_series_df,
            macro_features_by_basket = {basket_name: macro_features_by_basket.get(basket_name, pd.DataFrame())},
        )

        if basket_matrix.empty:
            logger.warning(f"{basket_name}: feature matrix is empty — skipped")
            basket_results[basket_name] = {"status": "skipped_empty_matrix"}
            continue

        logger.info(f"{basket_name}: feature matrix built | {len(basket_matrix)} rows")

        if len(basket_matrix) < MIN_SAMPLES_PER_BASKET:
            logger.warning(
                f"{basket_name}: only {len(basket_matrix)} training samples "
                f"(minimum is {MIN_SAMPLES_PER_BASKET}) — skipped."
            )
            basket_results[basket_name] = {"status": "skipped_too_few_samples"}
            continue

        try:
            basket_pipeline, basket_metrics = train_directional_model(
                basket_matrix, basket_name=basket_name,
            )
            write_model_metrics(
                model_name = basket_metrics["model_name"],
                train_date = basket_metrics["train_date"],
                precision  = basket_metrics["precision"],
                auc_roc    = basket_metrics["auc_roc"],
                n_samples  = basket_metrics["n_train"] + basket_metrics["n_test"],
                recall     = basket_metrics.get("recall", 0.0),
                pr_auc     = basket_metrics.get("pr_auc", 0.0),
            )

            logger.info(
                f"{basket_name} trained | "
                f"Precision: {basket_metrics['precision']:.4f} | "
                f"AUC-ROC: {basket_metrics['auc_roc']:.4f}"
            )
            basket_results[basket_name] = {"status": "trained", "metrics": basket_metrics}

        except Exception as e:
            logger.error(f"{basket_name}: training failed: {e}", exc_info=True)
            basket_results[basket_name] = {"status": "failed", "error": str(e)}

    logger.info("=" * 60)
    logger.info("BASKET TRAINING SUMMARY")
    logger.info("=" * 60)
    for basket_name, result in basket_results.items():
        logger.info(f"{basket_name}: {result['status']}")

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()
