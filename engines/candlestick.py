"""
engines/candlestick.py
----------------------
Candlestick-pattern-at-extreme interaction engine for the FX Signal
Ranker. Hand-crafted, not left for XGBoost to discover on its own — per
project discussion, FX's smaller training-data volume (28 pairs vs
~1,800 stocks) makes explicit interactions more valuable here than they
were for the stock project.

LOGICAL FLOW:
─────────────
STEP 1 — Pattern detection (Hammer / Shooting Star):
   Both patterns are wick-to-body ratio tests on a single candle,
   using the user's own detection logic verbatim (not re-derived):

   Hammer (bullish reversal shape):
       body       = |close - open|
       lower_wick = min(open, close) - low
       upper_wick = high - max(open, close)
       is_hammer  = (lower_wick >= 2 * body) AND (upper_wick <= body)

   Shooting Star (bearish reversal shape — mirror of Hammer):
       upper_wick     = high - max(open, close)
       lower_wick     = min(open, close) - low
       is_shooting_star = (upper_wick >= 2 * body) AND (lower_wick <= body)

STEP 2 — "At extreme" gate:
   A pattern only counts as meaningful at a genuine price extreme, not
   mid-range noise. Extreme is defined as |sd_position| > 1 — i.e.
   anywhere inside the scanner's own long/short entry band (SD -1 to
   -3 for long, +1 to +3 for short), NOT a separately-tuned threshold.
   This intentionally matches the scanner's existing entry-band
   definition rather than introducing a second, inconsistent notion of
   "extreme" — decided explicitly over config.yaml's candlestick.
   extreme_sd_threshold (1.5), which is now STALE and should be
   updated/removed next time config.yaml is touched (flagged, not
   silently fixed here, since this file doesn't own config.yaml).

STEP 3 — Direction-aware interaction (the hand-crafted part):
   The pattern + extreme gate is only informative when it AGREES with
   the setup direction being evaluated:
       Hammer at extreme LOW (sd_position < -1)  + direction == 'long'
           -> hammer_at_extreme = 1   (bullish reversal shape, at a
                                        genuine oversold extreme, while
                                        we're evaluating a long)
       Shooting Star at extreme HIGH (sd_position > +1) + direction == 'short'
           -> hammer_at_extreme = 1   (bearish reversal shape, at a
                                        genuine overbought extreme,
                                        while we're evaluating a short)
       Anything else (pattern present but direction doesn't match, no
       pattern, or not at an extreme) -> hammer_at_extreme = 0

   This is deliberately asymmetric-by-design: a Hammer at an extreme
   HIGH, or while evaluating a short, is NOT given credit — the whole
   point of the hand-crafted interaction is to sharpen the signal for
   the SPECIFIC setup being scored, not to flag "some reversal pattern
   happened somewhere" as a direction-agnostic feature.

OUTPUT per (pair, datetime, direction):
   - is_hammer          : raw pattern flag (0/1), no extreme gating
   - is_shooting_star   : raw pattern flag (0/1), no extreme gating
   - hammer_at_extreme  : the direction-aware interaction feature (0/1)
                          — THIS is what features.py consumes; the two
                          raw flags above are exposed for debugging/
                          analysis, not fed into the model directly.

NAMING NOTE: earlier project discussion referred to this engine as
"Hammer/Spinning-Top-at-extreme." The user's actual detection code
covers Hammer and Shooting Star (not Spinning Top) — this engine is
built around what was actually provided, not a forced Spinning Top
interpretation. If Spinning Top detection is added later, it slots in
as a third pattern check feeding the same extreme-gated, direction-
aware interaction logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_candlestick_logger
from utils.error_handler import graceful, EngineError

logger = get_candlestick_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()

# NOTE: extreme threshold is intentionally NOT read from
# config["candlestick"]["extreme_sd_threshold"] (1.5) — per explicit
# decision, "at extreme" here matches the scanner's own long/short
# entry band (|sd_position| > 1), not a separately-tuned value. See
# module docstring. config.yaml's extreme_sd_threshold is stale as of
# this engine and should be updated/removed next time config.yaml is
# touched.
EXTREME_SD_THRESHOLD = 1.0

MIN_CANDLES_REQUIRED = 1   # pattern detection only needs the current candle


# =============================================================================
# STEP 1 — PATTERN DETECTION (user's logic, verbatim math)
# =============================================================================

def _compute_patterns(open_: float, high: float, low: float, close: float) -> dict:
    """
    Detect Hammer and Shooting Star on a single candle's OHLC values.

    Uses the exact wick-to-body ratio logic as provided: a pattern
    requires its "long" wick to be at least 2x the candle body, and
    its "short" wick to be no larger than the body itself.

    Args:
        open_, high, low, close: Single candle's OHLC values

    Returns:
        Dict with is_hammer (bool) and is_shooting_star (bool)
    """
    body       = abs(close - open_)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)

    is_hammer = bool(
        (lower_wick >= 2 * body) and (upper_wick <= body)
    )
    is_shooting_star = bool(
        (upper_wick >= 2 * body) and (lower_wick <= body)
    )

    return {
        "is_hammer"       : is_hammer,
        "is_shooting_star": is_shooting_star,
    }


# =============================================================================
# RAW PATTERN FLAGS — no extreme gate, no direction interaction
# Added for the project's redefined directional-prediction task: with
# LinReg (and its sd_position) dropped from the feature set entirely,
# and no externally-supplied trade direction to gate against (the model
# now PREDICTS direction rather than being scored against a chosen
# one), the hand-crafted "at extreme, direction-aware" interaction this
# module was originally built around no longer has anything to anchor
# to. Per project decision: don't rebuild the extreme gate on a new
# basis — just expose the raw pattern flags and let the model learn
# when they matter, using ADX/CSI/ATR-regime as its context instead.
# compute_candlestick_latest and _compute_hammer_at_extreme below are
# KEPT, not deleted — they're harmless, well-tested code that simply
# isn't called by the new features.py. Removing them would be a bigger,
# riskier change than leaving unused-but-correct code in place.
# =============================================================================

def compute_raw_pattern_flags(
    pair: str,
    df  : pd.DataFrame,
) -> Optional[dict]:
    """
    Detect Hammer / Shooting Star on the latest candle only — no
    extreme-SD gate, no direction interaction. This is the entry point
    the redefined-task features.py uses; compute_candlestick_latest
    (below) remains for any caller that still has sd_position/direction
    available and wants the original gated interaction.

    Args:
        pair: FX pair symbol, e.g. 'EURUSD' (used for logging only)
        df  : OHLC DataFrame for this pair, sorted datetime ascending,
              already filtered to <= the signal datetime by the caller

    Returns:
        Dict with is_hammer (int 0/1) and is_shooting_star (int 0/1),
        or None if there's no candle to evaluate
    """
    if df is None or df.empty:
        logger.debug(f"{pair} | No candle available for pattern detection")
        return None

    last_candle = df.iloc[-1]

    patterns = _compute_patterns(
        open_ = float(last_candle["open"]),
        high  = float(last_candle["high"]),
        low   = float(last_candle["low"]),
        close = float(last_candle["close"]),
    )

    return {
        "is_hammer"       : int(patterns["is_hammer"]),
        "is_shooting_star": int(patterns["is_shooting_star"]),
    }


# =============================================================================
# STEP 2 & 3 — EXTREME GATE + DIRECTION-AWARE INTERACTION
# Kept for backward compatibility — not called by the redefined-task
# features.py (see compute_raw_pattern_flags above), but left in place
# rather than deleted since it's correct, tested code.
# =============================================================================

def _compute_hammer_at_extreme(
    is_hammer       : bool,
    is_shooting_star: bool,
    sd_position     : float,
    direction       : str,
) -> int:
    """
    Combine pattern detection, the extreme-SD gate, and setup direction
    into the single hand-crafted interaction feature.

    Args:
        is_hammer        : Hammer pattern detected on this candle
        is_shooting_star  : Shooting Star pattern detected on this candle
        sd_position      : Signed SD distance from the LinReg mean
                           (negative = below mean, positive = above)
        direction        : 'long' or 'short' — the setup currently
                           being evaluated

    Returns:
        1 if the pattern agrees with both the extreme gate and the
        setup direction, else 0
    """
    at_extreme_low  = sd_position < -EXTREME_SD_THRESHOLD
    at_extreme_high = sd_position > EXTREME_SD_THRESHOLD

    if direction == "long" and is_hammer and at_extreme_low:
        return 1
    if direction == "short" and is_shooting_star and at_extreme_high:
        return 1
    return 0


# =============================================================================
# SINGLE (pair, datetime, direction) LOOKUP
# Matches the calling shape of compute_adx_latest / compute_linreg_latest
# etc. — @graceful-wrapped, minimum-row-count guard, returns Optional[dict].
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,))
def compute_candlestick_latest(
    pair           : str,
    df             : pd.DataFrame,
    signal_datetime,
    sd_position    : float,
    direction      : str,
) -> Optional[dict]:
    """
    Compute candlestick-at-extreme features for a single pair at a
    single signal datetime, for a specific setup direction.

    FLOW:
    1. Guard: need at least the current candle (MIN_CANDLES_REQUIRED)
    2. Extract the candle at (or most recent at-or-before) signal_datetime
    3. Detect Hammer / Shooting Star on that single candle
    4. Apply the extreme gate + direction-aware interaction

    Args:
        pair           : FX pair symbol, e.g. 'EURUSD' (used for logging only)
        df             : OHLC DataFrame for this pair, sorted datetime
                         ascending, already filtered to <= signal_datetime
                         by the caller (matches the convention used by
                         other engines' *_latest functions)
        signal_datetime: Timestamp of the signal
        sd_position    : This pair's current SD distance from LinReg mean
                         (passed in — this engine does not recompute
                         LinReg itself, matching how ADX/other engines
                         don't recompute each other's indicators)
        direction      : 'long' or 'short' — the setup being evaluated

    Returns:
        Dict with is_hammer, is_shooting_star, hammer_at_extreme, or
        None if insufficient data
    """
    if df is None or len(df) < MIN_CANDLES_REQUIRED:
        logger.debug(f"{pair} | Insufficient candles for candlestick detection")
        return None

    if direction not in ("long", "short"):
        logger.warning(f"{pair} | Unrecognised direction '{direction}', expected 'long'/'short'")
        return None

    last_candle = df.iloc[-1]

    patterns = _compute_patterns(
        open_ = float(last_candle["open"]),
        high  = float(last_candle["high"]),
        low   = float(last_candle["low"]),
        close = float(last_candle["close"]),
    )

    hammer_at_extreme = _compute_hammer_at_extreme(
        is_hammer        = patterns["is_hammer"],
        is_shooting_star = patterns["is_shooting_star"],
        sd_position      = sd_position,
        direction        = direction,
    )

    return {
        "is_hammer"         : int(patterns["is_hammer"]),
        "is_shooting_star"  : int(patterns["is_shooting_star"]),
        "hammer_at_extreme" : hammer_at_extreme,
    }


# =============================================================================
# BATCH RUNNER
# Matches the calling shape of run_linreg_engine / run_smc_engine /
# run_adx_engine — dict[pair -> DataFrame] in, DataFrame out. Unlike
# those engines, this one also needs sd_position and direction per
# row, since the interaction is direction-aware and SD-position-gated
# — these come from indicators_df (LinReg output) and the scan hit
# being evaluated, not from this engine's own OHLC data.
# =============================================================================

def run_candlestick_engine(
    tickers_data  : dict[str, pd.DataFrame],
    indicators_df : pd.DataFrame,
    scan_hits_df  : pd.DataFrame,
    date          : str,
) -> pd.DataFrame:
    """
    Run the candlestick-at-extreme engine across a batch of scan hits.

    Unlike linreg/smc/adx (one row of output per pair, from that
    pair's own OHLC history alone), this engine's output is keyed by
    (pair, datetime, direction) — one row per scan hit being evaluated
    — since the same candle can be a "1" for a long evaluation and a
    "0" for a short evaluation simultaneously.

    FLOW:
    1. For each scan hit, look up that pair's price data (already
       filtered to <= that hit's datetime, matching the convention
       used elsewhere) and its sd_position from indicators_df
    2. Call compute_candlestick_latest for that (pair, datetime, direction)
    3. Collect into a DataFrame

    Args:
        tickers_data  : Dict mapping pair name -> OHLC DataFrame
                        (datetime ascending)
        indicators_df : Indicator results [pair, datetime,
                        price_sd_position, ...] — used to look up
                        sd_position per scan hit
        scan_hits_df  : Scan hits to evaluate [pair, datetime, direction]
        date          : Date string for logging only

    Returns:
        DataFrame [pair, datetime, direction, is_hammer,
        is_shooting_star, hammer_at_extreme]
    """
    logger.info(
        f"Candlestick engine starting | "
        f"{len(scan_hits_df)} scan hits | Date: {date}"
    )

    if scan_hits_df.empty:
        logger.warning("Candlestick engine: no scan hits provided")
        return pd.DataFrame()

    results = []
    skipped = 0

    for row in scan_hits_df.itertuples(index=False):
        pair      = row.pair
        dt        = row.datetime
        direction = row.direction

        if pair not in tickers_data:
            skipped += 1
            continue

        px = tickers_data[pair]
        px = px[px["datetime"] <= dt].sort_values("datetime")

        if px.empty:
            skipped += 1
            continue

        ind_row = indicators_df[
            (indicators_df["pair"] == pair) &
            (indicators_df["datetime"] == dt)
        ]
        if ind_row.empty:
            skipped += 1
            continue

        sd_position = float(ind_row.iloc[0]["price_sd_position"])

        result = compute_candlestick_latest(
            pair            = pair,
            df              = px,
            signal_datetime = dt,
            sd_position     = sd_position,
            direction       = direction,
        )

        if result is None:
            skipped += 1
            continue

        results.append({
            "pair"      : pair,
            "datetime"  : dt,
            "direction" : direction,
            **result,
        })

    logger.info(
        f"Candlestick engine complete | "
        f"Computed: {len(results)} | Skipped: {skipped}"
    )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
