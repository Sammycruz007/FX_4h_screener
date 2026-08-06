"""
ml/labeller.py
--------------
Programmatic label generation for the FX directional models.

LOGICAL FLOW — LABEL REDESIGN (ATR-relative move + majority-candle
confirmation, replacing the plain forward-return-sign label):
─────────────
The prior version of this file asked a simple, unconditional question
of every daily candle for every pair:

    "Is price higher H candles from now than it is today?"

    target = 1 if close[t + H] > close[t] else 0

That label technically counts as "correct" any path where price nets
out positive by t+H, INCLUDING paths that dropped, rallied, then
partially reversed again — a small, noisy net move that doesn't
resemble a tradeable, sustained directional move at all. This is the
exact failure mode described in the FX_Directional_Model_Redesign
spec: a "successful" BUY often nets only 15-40 pips of genuine signal
buried in what was mostly noise.

NEW LABEL DEFINITION (see FX_Directional_Model_Redesign spec):
   For each (pair, t), looking H candles ahead (H = HORIZON, 3 daily
   candles):

       net_move = close[t+H] - close[t]

       BUY (1)  if net_move >=  MIN_MOVE_ATR_MULTIPLE * atr_fast[t]
                AND at least MIN_CONFIRMING_CANDLES of the H candles
                in (t, t+H] close bullish (close > open)

       SELL (0) if net_move <= -MIN_MOVE_ATR_MULTIPLE * atr_fast[t]
                AND at least MIN_CONFIRMING_CANDLES of the H candles
                in (t, t+H] close bearish (close < open)

       otherwise: discard (no label — NOT zero-filled, NOT treated as
       an ambiguous class of its own; the row is dropped from
       training entirely)

   This directly targets the "dip-then-reverse-then-technically-still-
   positive" pattern: a small choppy net move that clears neither the
   ATR-relative move-size bar nor the majority-of-candles-agree bar
   gets discarded, not counted as a BUY/SELL.

   MIN_MOVE_ATR_MULTIPLE and MIN_CONFIRMING_CANDLES are read from
   ml.min_move_atr_multiple (0.8) and ml.min_confirming_candles (2) in
   config.yaml. atr_fast is consumed as-is from prices_df (produced by
   features.py's _compute_atr(px, ATR_FAST_PERIOD)) — this file does
   NOT compute ATR itself.

   This is a discard-heavy design BY INTENT, not a bug: the spec
   explicitly trades label quantity for label quality (expected
   precision 0.84-0.92 at threshold, recall as low as 0.15) — training
   only on genuinely clean, directional examples rather than every
   sample that happened to net positive/negative. Expect a meaningful
   drop in usable rows per basket vs. the old label, especially on
   choppy days, which are common in FX, not rare.

WHAT'S KEPT FROM THE PRIOR VERSION, AND WHY:
   The hard-won datetime normalisation logic is kept nearly verbatim.
   That code fixed two REAL, production-hit bugs unrelated to the
   label definition — (1) mismatched string-vs-Timestamp datetime
   representations across DataFrames causing SILENT zero-match merges
   (no exception, just empty results), and (2) a numpy 2.x compatibility
   gap where searchsorted (and, by the same underlying comparison
   mechanism, any datetime equality/inequality check) can fail between
   a pandas.Timestamp and a datetime64 array unless explicitly
   converted first. Both risks are independent of what the label
   itself measures, so this defensive normalisation stays.

   The "drop last H rows per pair, NaN not zero-filled" principle also
   carries over unchanged — those rows have no valid future close to
   compare against yet, and fabricating an outcome for them would
   corrupt the model the same way it would have under the old label.

HORIZON:
   HORIZON is read from ml.label_forward_periods, now 3 (3 DAILY
   candles — this project moved from native Weekly bars to native
   Daily bars; see data/fetcher.py's fetch_interval, already "1d").
   The same config key is reused rather than introducing a new one.

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

# HORIZON: number of DAILY candles ahead the label looks. Same config
# key as before (ml.label_forward_periods), now expected to be 3 (3
# daily candles), not 2 (the old weekly-era value). See module
# docstring for the full reasoning.
HORIZON = ML_CFG["label_forward_periods"]

if HORIZON < 1:
    raise MLError(
        f"ml.label_forward_periods={HORIZON} is invalid — must be a "
        f"positive integer number of daily candles to look ahead "
        f"(the current design uses 3)."
    )

# MIN_MOVE_ATR_MULTIPLE: net move over the horizon must be at least
# this many multiples of atr_fast[t] (in the label's direction) to
# count as BUY/SELL rather than being discarded. Part of the
# ATR-relative move-size filter — see module docstring.
MIN_MOVE_ATR_MULTIPLE = ML_CFG["min_move_atr_multiple"]

if MIN_MOVE_ATR_MULTIPLE <= 0:
    raise MLError(
        f"ml.min_move_atr_multiple={MIN_MOVE_ATR_MULTIPLE} is invalid "
        f"— must be a positive number (the current design uses 0.8)."
    )

# MIN_CONFIRMING_CANDLES: at least this many of the HORIZON candles
# after t must individually close in the label's direction (bullish
# for BUY, bearish for SELL) for the sample to count. Part of the
# majority-candle-confirmation filter — see module docstring.
MIN_CONFIRMING_CANDLES = ML_CFG["min_confirming_candles"]

if not (1 <= MIN_CONFIRMING_CANDLES <= HORIZON):
    raise MLError(
        f"ml.min_confirming_candles={MIN_CONFIRMING_CANDLES} is "
        f"invalid — must be between 1 and ml.label_forward_periods "
        f"(HORIZON={HORIZON}) inclusive (the current design uses 2 "
        f"of 3)."
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
# ATR-relative move + majority-candle-confirmation. Unconditional — every
# row for every pair is evaluated, but many will be discarded (no label)
# rather than forced into BUY/SELL. See module docstring for the full
# label definition and reasoning.
# =============================================================================

def label_direction(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate directional labels for every (pair, datetime) row.

    FLOW:
    1. Normalise the datetime column defensively (see module docstring)
    2. Sort each pair's rows chronologically
    3. For each pair, compute:
       - net_move[t]  = close[t+HORIZON] - close[t]        (forward shift)
       - bull_count[t] = count of bullish candles (close > open) among
                         the HORIZON candles at t+1 .. t+HORIZON
       - bear_count[t] = count of bearish candles (close < open) among
                         the same HORIZON candles
    4. Classify each row:
       - BUY (1)  if net_move  >=  MIN_MOVE_ATR_MULTIPLE * atr_fast[t]
                  and bull_count >= MIN_CONFIRMING_CANDLES
       - SELL (0) if net_move  <= -MIN_MOVE_ATR_MULTIPLE * atr_fast[t]
                  and bear_count >= MIN_CONFIRMING_CANDLES
       - otherwise: discard (NaN label, dropped)
    5. Drop the last HORIZON rows per pair, where the future closes
       and candles needed for the calculation don't exist yet (NaN,
       not zero-filled — we don't fabricate an outcome for rows we
       genuinely don't know yet)
    6. Return one row per (pair, datetime) that received a real label

    NO LOOK-AHEAD BIAS: every quantity used (net_move, bull_count,
    bear_count) is computed from close[t] and the HORIZON candles
    strictly after t — all already-realised prices by the time this
    label would ever be used for scoring at t+HORIZON. This function
    only ever runs on historical data where the future is already
    known, for training purposes.

    Args:
        prices_df: Full OHLC DataFrame [pair, datetime, open, high,
                   low, close, atr_fast, ...], one row per pair per
                   daily candle. atr_fast must already be present
                   (produced by features.py) — this function does not
                   compute it.

    Returns:
        DataFrame with columns [pair, datetime, label], one row per
        (pair, datetime) that cleared either the BUY or SELL bar.
        label is 1 (BUY) or 0 (SELL). Rows that were discarded (didn't
        clear either bar) are simply absent, not present with a NaN
        or third-class label.
    """
    logger.info(
        f"Generating directional labels | HORIZON={HORIZON} daily candles | "
        f"min_move_atr_multiple={MIN_MOVE_ATR_MULTIPLE} | "
        f"min_confirming_candles={MIN_CONFIRMING_CANDLES}/{HORIZON}"
    )

    if prices_df.empty:
        logger.warning("label_direction: prices_df is empty")
        return pd.DataFrame(columns=["pair", "datetime", "label"])

    if "atr_fast" not in prices_df.columns:
        raise MLError(
            "label_direction: prices_df is missing 'atr_fast' — this "
            "column must be computed upstream by features.py "
            "(_compute_atr(px, ATR_FAST_PERIOD)) before labelling. "
            "This function does not compute ATR itself."
        )

    prices_df = _normalise_datetime_column(prices_df, "prices_df")

    all_labels = []
    total_discarded = 0

    for pair, group in prices_df.groupby("pair"):
        group = group.sort_values("datetime").reset_index(drop=True)

        if len(group) <= HORIZON:
            logger.debug(
                f"{pair} | Only {len(group)} rows, need > {HORIZON} "
                f"for even one valid label — skipped"
            )
            continue

        close = group["close"]
        open_ = group["open"]
        atr_fast = group["atr_fast"]

        future_close = close.shift(-HORIZON)
        net_move = future_close - close

        # Per-candle bullish/bearish flags, aligned to their own row —
        # shifted into place below so that, for row t, we can sum the
        # flags over rows t+1 .. t+HORIZON.
        is_bullish_candle = (close > open_)
        is_bearish_candle = (close < open_)

        bull_count = pd.Series(0, index=group.index, dtype="float64")
        bear_count = pd.Series(0, index=group.index, dtype="float64")
        for k in range(1, HORIZON + 1):
            bull_count = bull_count.add(
                is_bullish_candle.shift(-k).astype("float64"), fill_value=0
            )
            bear_count = bear_count.add(
                is_bearish_candle.shift(-k).astype("float64"), fill_value=0
            )

        # BUG-PRONE PATTERN (see prior version's note): comparing floats
        # against NaN silently evaluates comparisons to False rather
        # than propagating NaN, which would let rows past the horizon
        # (with no real future data) sneak in as false negatives/
        # discards instead of being explicitly dropped. Guard explicitly
        # via .notna() on future_close BEFORE combining conditions.
        has_valid_future = future_close.notna()

        buy_move_ok  = net_move >=  MIN_MOVE_ATR_MULTIPLE * atr_fast
        sell_move_ok = net_move <= -MIN_MOVE_ATR_MULTIPLE * atr_fast

        is_buy  = has_valid_future & buy_move_ok  & (bull_count >= MIN_CONFIRMING_CANDLES)
        is_sell = has_valid_future & sell_move_ok & (bear_count >= MIN_CONFIRMING_CANDLES)

        # A row cannot be both — buy_move_ok and sell_move_ok are
        # mutually exclusive by construction (net_move can't be both
        # >= a positive threshold and <= its negative). Rows matching
        # neither are the discard case: no label at all.
        label = pd.Series(np.nan, index=group.index, dtype="float64")
        label = label.mask(is_buy, 1.0)
        label = label.mask(is_sell, 0.0)

        pair_labels = pd.DataFrame({
            "pair"    : pair,
            "datetime": group["datetime"],
            "label"   : label,
        })

        n_eligible = int(has_valid_future.sum())
        n_labelled = int(label.notna().sum())
        total_discarded += (n_eligible - n_labelled)

        pair_labels = pair_labels.dropna(subset=["label"])
        pair_labels["label"] = pair_labels["label"].astype(int)

        if n_eligible > 0:
            logger.debug(
                f"{pair} | eligible={n_eligible} | labelled={n_labelled} | "
                f"discarded={n_eligible - n_labelled} "
                f"({(n_eligible - n_labelled) / n_eligible * 100:.1f}%)"
            )
        else:
            logger.debug(f"{pair} | no eligible rows")

        all_labels.append(pair_labels)

    if not all_labels:
        logger.warning("label_direction: no pair produced any valid labels")
        return pd.DataFrame(columns=["pair", "datetime", "label"])

    result = pd.concat(all_labels, ignore_index=True)

    if result.empty:
        # Every eligible row across every pair was discarded — a real,
        # expected possibility under this filter (e.g. a uniformly
        # choppy period), not necessarily a bug. Log plainly rather
        # than dividing by zero.
        logger.warning(
            f"label_direction: {total_discarded} row(s) were eligible "
            f"but ALL were discarded — no BUY/SELL labels produced "
            f"this run."
        )
        return result

    pos = (result["label"] == 1).sum()
    neg = (result["label"] == 0).sum()
    logger.info(
        f"Directional labels generated | "
        f"Total: {len(result)} | "
        f"BUY (1): {pos} ({pos/len(result)*100:.1f}%) | "
        f"SELL (0): {neg} ({neg/len(result)*100:.1f}%) | "
        f"Discarded: {total_discarded}"
    )

    return result
