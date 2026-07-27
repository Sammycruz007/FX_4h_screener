"""
ml/labeller.py
--------------
Programmatic label generation for the FX directional models.

LOGICAL FLOW — COMPLETE REDESIGN FROM THE PRIOR SIGNAL RANKER LABELLER:
─────────────
This project's task changed from "will a scanner-flagged setup reach a
specific LinReg-band target within N candles" to a much simpler,
unconditional question, asked of EVERY weekly candle for EVERY pair:

    "Is price higher 2 weekly candles from now than it is today?"

    target = 1 if close[t + HORIZON] > close[t] else 0
    target = 0 otherwise

This directly mirrors the EURUSD reference project's own label
definition (see that project's README) — a plain forward-return sign
check, not a fixed-target-hit search.

WHAT'S DROPPED FROM THE PRIOR VERSION OF THIS FILE, AND WHY:
   - label_scanner_hits() is GONE ENTIRELY. That function existed to
     label scanner-flagged (pair, datetime, direction) CANDIDATES —
     rows that had already passed a slope+SD-zone gate — against a
     FIXED LinReg-snapshot target using np.searchsorted to check if
     price ever touched that target within a forward window. NONE of
     that applies anymore:
       - There is no scanner/candidate concept in the new design —
         every pair gets a directional prediction every week,
         unconditionally, matching the reference project's "every
         pair, every cycle" pattern. There is no "direction" input to
         condition the label on, because the model predicts direction
         itself rather than being scored against a pre-chosen one.
       - There is no LinReg-derived target level to search for at all
         — LinReg is dropped from this project's feature set entirely
         per the project's redefined scope (28 pairs / basket models /
         CSI+ADX+candlestick features only).
       - The target itself changed from "did price EVER reach a level
         within a window" (needs a forward SEARCH across the whole
         window) to "is price higher at exactly one future point" (a
         plain point-in-time comparison) — this is a fundamentally
         simpler operation, no O(log n) searchsorted machinery needed.
   - This means the labeller is now dramatically smaller. That's a
     reflection of the task being genuinely simpler, not a shortcut.

WHAT'S KEPT FROM THE PRIOR VERSION, AND WHY:
   The hard-won datetime normalisation logic is kept nearly verbatim.
   That code fixed two REAL, production-hit bugs unrelated to the old
   target definition — (1) mismatched string-vs-Timestamp datetime
   representations across DataFrames causing SILENT zero-match merges
   (no exception, just empty results), and (2) a numpy 2.x compatibility
   gap where searchsorted (and, by the same underlying comparison
   mechanism, any datetime equality/inequality check) can fail between
   a pandas.Timestamp and a datetime64 array unless explicitly
   convertedfirst. Both risks apply just as much to the new simple
   shift-based label as they did to the old search-based one, so this
   defensive normalisation stays.

LABEL DEFINITION:
   For each pair, sorted by datetime ascending:
       target[t] = 1 if close[t + HORIZON] > close[t] else 0
   The last HORIZON rows of each pair's series have no valid future
   close to compare against — those rows get target = NaN and are
   DROPPED, not zero-filled (training on a fabricated "down" label for
   rows where we simply don't know the outcome yet would corrupt the
   model, same reasoning applied everywhere else in this project's
   NaN-handling).

   HORIZON is read from ml.label_forward_periods, which must now be
   set to 2 (2 WEEKLY candles ahead) in config.yaml — this project
   moved from a 4H timeframe (where this same config key held 30, ~1
   week's worth of 4H candles) to native Weekly bars (see
   data/fetcher.py's module docstring), so 2 weekly candles is now the
   correct value for the SAME config key, not a new one.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other FX file — 'pair' and 'datetime'
   throughout.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import MLError

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

# HORIZON: number of WEEKLY candles ahead the label looks. Same config
# key as the old 4H-era labeller (ml.label_forward_periods), now
# expected to be 2 (2 weekly candles), not 30 (the old ~1-week-in-4H-
# candles value). See module docstring for the full reasoning.
HORIZON = ML_CFG["label_forward_periods"]

if HORIZON < 1:
    raise MLError(
        f"ml.label_forward_periods={HORIZON} is invalid — must be a "
        f"positive integer number of weekly candles to look ahead "
        f"(the reference design uses 2)."
    )


# =============================================================================
# DATETIME NORMALISATION
# Kept from the prior version — fixes two real, previously production-hit
# bugs (silent string-vs-Timestamp merge mismatches, and a numpy 2.x
# Timestamp-vs-datetime64 comparison gap). Both risks are independent of
# what the label itself measures, so this defensive logic stays as-is.
# =============================================================================

def _normalise_datetime_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Coerce a DataFrame's 'datetime' column to timezone-naive pandas
    Timestamps (UTC-interpreted, then tz stripped), defensively.

    See module docstring's "WHAT'S KEPT" section for why this exists:
    a Timestamp-vs-string or tz-aware-vs-tz-naive mismatch between two
    DataFrames can silently produce wrong or empty merge/comparison
    results with no exception raised anywhere, and this project has
    been bitten by exactly that twice already.

    Also guards against a 'datetime' column arriving as raw numeric
    values (e.g. after a Parquet round-trip) — pd.to_datetime on a
    numeric column silently assumes Unix epoch SECONDS rather than
    raising an error, which would silently corrupt every label with a
    wildly wrong date rather than fail loudly.

    Args:
        df   : DataFrame with a 'datetime' column to normalise
        label: Name for this DataFrame, used only in the error message
               if the numeric-input guard fires

    Returns:
        A copy of df, with 'datetime' coerced to tz-naive Timestamps
    """
    col = df["datetime"]

    if pd.api.types.is_numeric_dtype(col):
        raise MLError(
            f"{label}['datetime'] is numeric (dtype={col.dtype}) — "
            f"refusing to guess whether these are epoch seconds, "
            f"milliseconds, or nanoseconds. This usually means a "
            f"datetime column lost its proper dtype somewhere "
            f"upstream (e.g. a Parquet round-trip) — fix the source, "
            f"don't guess the unit here."
        )

    df = df.copy()
    df["datetime"] = pd.to_datetime(col, utc=True).dt.tz_localize(None)
    return df


# =============================================================================
# DIRECTIONAL LABEL GENERATION
# Replaces label_scanner_hits() entirely. Unconditional — every row for
# every pair gets a label, not just scanner-flagged candidates (there
# is no scanner/candidate concept in this design at all).
# =============================================================================

def label_direction(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate directional labels for every (pair, datetime) row.

    FLOW:
    1. Normalise the datetime column defensively (see module docstring)
    2. Sort each pair's rows chronologically
    3. For each pair, compare close[t + HORIZON] against close[t] via a
       plain forward shift — NOT a searchsorted/window-scan, since the
       new target is a single future point, not "did it ever touch a
       level within a window"
    4. Drop the last HORIZON rows per pair, where the future close
       needed for comparison doesn't exist yet (NaN, not zero-filled —
       we don't fabricate an outcome for rows we genuinely don't know
       yet)
    5. Return one row per (pair, datetime) with a binary label

    NO LOOK-AHEAD BIAS: target[t] is computed from close[t] and
    close[t+HORIZON] only — both are already-realised prices by the
    time this label would ever be used for scoring at t+HORIZON. The
    label is deliberately NOT usable to predict anything at time t
    itself (that's the model's job, using only data available up to
    and including t) — this function only ever runs on historical
    data where the future is already known, for training purposes.

    Args:
        prices_df: Full OHLC DataFrame [pair, datetime, open, high,
                   low, close], one row per pair per weekly candle

    Returns:
        DataFrame with columns [pair, datetime, label], one row per
        (pair, datetime) that has a valid HORIZON-candles-ahead close
        to compare against. label is 1 (up) or 0 (down/flat).
    """
    logger.info(f"Generating directional labels | HORIZON={HORIZON} weekly candles")

    if prices_df.empty:
        logger.warning("label_direction: prices_df is empty")
        return pd.DataFrame(columns=["pair", "datetime", "label"])

    prices_df = _normalise_datetime_column(prices_df, "prices_df")

    all_labels = []

    for pair, group in prices_df.groupby("pair"):
        group = group.sort_values("datetime").reset_index(drop=True)

        if len(group) <= HORIZON:
            logger.debug(
                f"{pair} | Only {len(group)} rows, need > {HORIZON} "
                f"for even one valid label — skipped"
            )
            continue

        future_close = group["close"].shift(-HORIZON)

        # BUG FIX (caught in testing, not a style choice): comparing a
        # plain float against NaN via `future_close > close` does NOT
        # produce NaN for the missing rows — it silently evaluates to
        # False, and casting that False to "Int64" afterward just gives
        # 0, not <NA>. That meant the intended "drop the last HORIZON
        # rows" behaviour never actually fired — every row got a label,
        # including the ones with no real future close to compare
        # against, silently mislabelling them as "down." Fixed by
        # explicitly using .where() to force NaN wherever future_close
        # itself is NaN, BEFORE the comparison, so the missing-ness is
        # real and explicit rather than assumed to propagate on its own.
        is_up = future_close > group["close"]
        label = is_up.where(future_close.notna()).astype("Int64")

        pair_labels = pd.DataFrame({
            "pair"    : pair,
            "datetime": group["datetime"],
            "label"   : label,
        })
        pair_labels = pair_labels.dropna(subset=["label"])
        pair_labels["label"] = pair_labels["label"].astype(int)

        all_labels.append(pair_labels)

    if not all_labels:
        logger.warning("label_direction: no pair produced any valid labels")
        return pd.DataFrame(columns=["pair", "datetime", "label"])

    result = pd.concat(all_labels, ignore_index=True)

    pos = (result["label"] == 1).sum()
    neg = (result["label"] == 0).sum()
    logger.info(
        f"Directional labels generated | "
        f"Total: {len(result)} | "
        f"Up (1): {pos} ({pos/len(result)*100:.1f}%) | "
        f"Down (0): {neg} ({neg/len(result)*100:.1f}%)"
    )

    return result
