"""
ml/labeller.py
--------------
Programmatic label generation for the FX Signal Ranker.

LOGICAL FLOW:
─────────────
We never manually label data. Labels are derived from historical
price outcomes, using the same fixed-LinReg-snapshot approach the
stock project used.

WHAT'S DROPPED FROM THE STOCK PROJECT:
   The stock project had TWO labellers: a Volume Classifier labeller
   (label_volume_patterns, based on volume pattern structure) and a
   Signal Ranker labeller (label_scanner_hits, based on price outcome).
   The Volume Classifier is dropped entirely for FX — there is no real,
   centralized volume in a decentralized OTC market, so there is nothing
   for a volume-pattern labeller to label. This file now contains ONLY
   the Signal Ranker labeller. This is a structural simplification, not
   an oversight — see the project's config.yaml / features.py comments
   for the same reasoning applied throughout the FX rework.

SIGNAL RANKER LABELS (carried over, timeframe-agnostic in shape):
   For every historical scanner hit (a pair that met scanner criteria
   at a given 4H candle), we look forward FORWARD_PERIODS candles and
   ask: "Did price CLOSE at or beyond the LinReg mean?"

   Uses CLOSE price (not high/low intraday touch) against a FIXED
   LinReg value (snapshot at signal datetime, not rolling future).
   This prevents the near-universal positive labelling problem.

   Yes -> label = 1 (successful setup)
   No  -> label = 0 (failed setup)

   FORWARD_PERIODS is read from ml.label_forward_periods (30 4H-candles
   ~= 5 trading days, per config.yaml) — already timeframe-correct for
   FX, no change needed to this parameter itself.

COLUMN NAMING (pair/datetime, not ticker/date):
   All inputs/outputs use 'pair' (e.g. 'EURUSD') and 'datetime'
   (timestamp, not a date string) — matching fetcher.py's actual output
   shape and engines/csi.py's convention. This naming is used
   consistently across every FX file from here forward.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import graceful, MLError

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()
ML_CFG = config["ml"]

FORWARD_PERIODS = ML_CFG["label_forward_periods"]   # 30 4H-candles, per config.yaml

SCANNER_CFG  = config["scanner"]
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]   # -1
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]   # -3
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]  # +1
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]  # +3


# =============================================================================
# SIGNAL RANKER LABELLER
# Labels historical scanner hits as successful or failed setups.
# Uses FIXED LinReg snapshot + CLOSE price to prevent universal positives.
# Carried over from the stock project's label_scanner_hits — the logic
# itself is timeframe/asset-agnostic; only column names changed.
# =============================================================================

def label_scanner_hits(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    scan_hits_df : pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate labels for the Signal Ranker.

    FLOW:
    1. Bulk-merge scan hits with their FIXED LinReg value at signal time
       (one merge, not a per-row lookup — same optimization as the
       stock project)
    2. Pre-group prices per pair into fast NumPy arrays
    3. For each hit, slice the next FORWARD_PERIODS closes using
       np.searchsorted (O(log n) lookup) and check if the CLOSE ever
       reached the fixed LinReg target in that window
    4. Return DataFrame of (pair, datetime, direction, label)

    Args:
        prices_df    : Full OHLC DataFrame [pair, datetime, open, high,
                       low, close]
        indicators_df: Indicator results [pair, datetime, linreg_value,
                       price_sd_position, ...]
        scan_hits_df : Historical scanner hits [pair, datetime, direction]

    Returns:
        DataFrame with columns [pair, datetime, direction, label]
    """
    logger.info("Generating Signal Ranker labels...")

    if scan_hits_df.empty:
        logger.warning("Signal Ranker labeller: No scan hits provided")
        return pd.DataFrame()

    # 1. BULK LOOKUP: merge indicators to get the fixed LinReg value in one shot
    hits_with_linreg = pd.merge(
        scan_hits_df,
        indicators_df[["pair", "datetime", "linreg_value"]],
        on=["pair", "datetime"],
        how="inner",
    )

    if hits_with_linreg.empty:
        return pd.DataFrame()

    # Normalise the merged 'datetime' column to TIMEZONE-NAIVE pandas
    # Timestamps (UTC-interpreted, then tz stripped). THREE distinct
    # failure modes were hit getting to this fix, not just one — worth
    # recording all three since the underlying cause is a genuinely
    # subtle pandas/numpy interop gotcha, not a simple typo:
    #   1. str vs Timestamp entirely -> "'<' not supported between int
    #      and str" (numpy compares Timestamps as ints internally)
    #   2. tz-naive vs tz-aware Timestamp (pd.to_datetime() without
    #      utc=True can silently parse a string with a '+00:00' offset
    #      as tz-naive in some pandas paths) -> "can't compare
    #      offset-naive and offset-aware datetimes"
    #   3. THE ACTUAL ROOT CAUSE: calling .values on a tz-AWARE pandas
    #      Series (as done below for price_dict's numpy arrays) SILENTLY
    #      STRIPS the tz info, converting datetime64[us, UTC] to plain
    #      datetime64[us] (tz-naive) numpy values — even though the
    #      Series itself was correctly tz-aware right up until that
    #      conversion. Meanwhile itertuples() (used for the per-row
    #      loop below) preserves pandas' tz-aware Timestamp type. So the
    #      array side silently goes tz-naive while the per-row needle
    #      side stays tz-aware, and np.searchsorted fails on the same
    #      "offset-naive and offset-aware" error as #2 — but this time
    #      caused by a numpy conversion behaviour, not the earlier
    #      to_datetime() call. Fix: interpret as UTC first (so any
    #      offset in the input is correctly applied), THEN strip
    #      tz-awareness explicitly with tz_localize(None) — applied
    #      identically to both the price array and the per-row needle,
    #      so neither side silently drifts tz-aware/tz-naive relative
    #      to the other after any pandas/numpy conversion downstream.
    hits_with_linreg["datetime"] = (
        pd.to_datetime(hits_with_linreg["datetime"], utc=True).dt.tz_localize(None)
    )

    # 2. PRE-PROCESS PRICES: group by pair into fast NumPy arrays
    logger.info("Pre-processing price data for fast lookup...")
    price_dict = {}

    sorted_prices = prices_df.sort_values(["pair", "datetime"]).reset_index(drop=True)
    # Same UTC-interpret-then-strip-tz normalisation applied to the
    # price side — see the detailed comment above for why both steps
    # (utc=True AND tz_localize(None)) are both required, not either
    # alone.
    sorted_prices["datetime"] = (
        pd.to_datetime(sorted_prices["datetime"], utc=True).dt.tz_localize(None)
    )

    for pair, group in sorted_prices.groupby("pair"):
        price_dict[pair] = {
            "datetimes": group["datetime"].values,
            "closes"   : group["close"].values,
        }

    all_labels = []

    # 3. FAST ITERATION: itertuples + np.searchsorted
    logger.info("Evaluating forward returns...")
    for row in hits_with_linreg.itertuples():
        pair              = row.pair
        dt                = row.datetime
        direction         = row.direction
        linreg_at_signal  = float(row.linreg_value)

        if pair not in price_dict:
            continue

        p_datetimes = price_dict[pair]["datetimes"]
        p_closes    = price_dict[pair]["closes"]

        # O(log n) lookup for the datetime index — both p_datetimes and
        # dt are now guaranteed-comparable pandas Timestamps (via the
        # normalisation above), not a str-vs-Timestamp mismatch.
        idx = np.searchsorted(p_datetimes, dt, side="right")

        # Slice the next FORWARD_PERIODS candles directly from the array
        future_closes = p_closes[idx : idx + FORWARD_PERIODS]

        if len(future_closes) < 1:
            continue

        # Check if CLOSE reached the FIXED LinReg target
        if direction == "long":
            reached = bool(np.any(future_closes >= linreg_at_signal))
        else:
            reached = bool(np.any(future_closes <= linreg_at_signal))

        all_labels.append({
            "pair"     : pair,
            "datetime" : dt,
            "direction": direction,
            "label"    : 1 if reached else 0,
        })

    result = pd.DataFrame(all_labels)

    if result.empty:
        logger.warning("Signal Ranker labeller: No labels generated")
        return result

    pos = (result["label"] == 1).sum()
    neg = (result["label"] == 0).sum()
    logger.info(
        f"Signal Ranker labels generated | "
        f"Total: {len(result)} | "
        f"Positive (1): {pos} ({pos/len(result)*100:.1f}%) | "
        f"Negative (0): {neg} ({neg/len(result)*100:.1f}%)"
    )

    return result
