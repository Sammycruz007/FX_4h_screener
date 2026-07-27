"""
ml/features.py
--------------
Feature engineering for the FX directional models (basket-grouped
XGBoost classifiers, one per currency basket).

LOGICAL FLOW — MAJOR REDESIGN FROM THE PRIOR SIGNAL RANKER VERSION:
─────────────
The project's task changed from "score a scanner-flagged long/short
CANDIDATE against a fixed LinReg-band target" to "predict price
direction 2 weekly candles ahead, for every pair, unconditionally, no
candidate/direction input at all." This changes both WHAT is computed
and the function SIGNATURE — there is no longer a `direction` argument
anywhere in this file, because the model predicts direction rather
than being scored against a pre-chosen one.

WHAT'S DROPPED, AND WHY (this is the bulk of the change):
   - LinReg entirely: sd_position, dist_to_mean, band_penetration,
     linreg_slope, days_in_trend. LinReg is no longer part of this
     project's feature set at all (per project scope redefinition —
     28 pairs / basket models / CSI+ADX+candlestick only). There is no
     replacement for these — they're just gone, not swapped for
     something else.
   - SMC entirely: has_valid_zone, bos_strength_vs_atr (Break-of-
     Structure is an SMC concept). Same reasoning — SMC dropped from
     project scope entirely.
   - direction_flag: this existed only to encode "which direction is
     currently being scored" for the old candidate-scoring task. There
     is no external direction to encode anymore — the label itself
     (see ml/labeller.py's label_direction) IS the direction, and it's
     the model's target, not an input feature.
   - hammer_at_extreme (the direction-aware, SD-extreme-gated
     candlestick interaction): this depended on BOTH sd_position
     (LinReg, now dropped) and an external direction (now nonexistent)
     to decide whether a detected pattern "counts." With neither input
     available, per project decision: don't rebuild the gate on a new
     basis — just expose the RAW pattern flags (is_hammer,
     is_shooting_star) with no gating at all, and let the model learn
     when they matter using ADX/CSI/ATR-regime as context. See
     engines/candlestick.py's compute_raw_pattern_flags().

WHAT'S KEPT, UNCHANGED (none of these ever depended on LinReg or SMC):
   - atr_fast / atr_slow / atr_ratio — volatility-regime feature,
     computed the same way, same ATR_FAST_PERIOD/ATR_SLOW_PERIOD config.
   - session_tokyo / session_london / session_new_york / session_overlap
     — derived purely from the candle's own UTC timestamp.
   - adx_value / plus_di / minus_di — pure price-action, unrelated to
     LinReg/SMC.
   - csi_rs / csi_diff_zscore / csi_diff_roc / csi_commodity_bloc —
     engines/csi.py's Currency Strength Index features, entirely
     independent of LinReg/SMC (CSI only ever needed close prices
     across the 28-pair universe).
   - is_hammer / is_shooting_star — kept as RAW, ungated pattern flags
     (see "WHAT'S DROPPED" above for why the gating is gone but the
     underlying pattern detection isn't).

TIMEFRAME NOTE: this project moved from 4H bars to native Weekly bars
(see data/fetcher.py's module docstring). ATR_FAST_PERIOD/
ATR_SLOW_PERIOD and the session-hour boundaries are still read from
the same config keys — if those keys' VALUES were tuned in units of
4H-candles, they need re-tuning for weekly bars (e.g. an "ATR_100"
over 100 weekly candles is ~2 years, not the ~2.3-month window it was
at 100 4H-candles) — that's a config decision, not something this file
can correct on its own; it just reads whatever config.yaml says.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other FX file.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import graceful, MLError
from engines.adx import compute_adx_latest
from engines.csi import run_csi_engine
from engines.candlestick import compute_raw_pattern_flags

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config      = _load_config()
ATR_CFG     = config["atr"]
SESSION_CFG = config["session"]

ATR_FAST_PERIOD = ATR_CFG["fast_period"]
ATR_SLOW_PERIOD = ATR_CFG["slow_period"]

TOKYO_OPEN_UTC     = SESSION_CFG["tokyo_open_utc"]
TOKYO_CLOSE_UTC    = SESSION_CFG["tokyo_close_utc"]
LONDON_OPEN_UTC    = SESSION_CFG["london_open_utc"]
LONDON_CLOSE_UTC   = SESSION_CFG["london_close_utc"]
NEW_YORK_OPEN_UTC  = SESSION_CFG["new_york_open_utc"]
NEW_YORK_CLOSE_UTC = SESSION_CFG["new_york_close_utc"]


# =============================================================================
# ATR — shared volatility measure
# Unchanged from the prior version — never depended on LinReg/SMC.
# =============================================================================

def _compute_atr(px: pd.DataFrame, period: int) -> float:
    """
    Compute Average True Range over the last `period` candles.

    Args:
        px    : OHLC DataFrame, sorted datetime ascending, already
                filtered to <= signal datetime by the caller
        period: Number of trailing candles to average over

    Returns:
        ATR value, or a 1%-of-price fallback if fewer than 2 candles
        are available
    """
    recent = px.tail(period)

    if len(recent) >= 2:
        high_low   = recent["high"] - recent["low"]
        high_close = (recent["high"] - recent["close"].shift(1)).abs()
        low_close  = (recent["low"]  - recent["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
    else:
        atr = float(px["close"].iloc[-1]) * 0.01

    return float(atr)


# =============================================================================
# SESSION INDICATOR
# Unchanged from the prior version — derived purely from the candle's
# own UTC timestamp, no engine dependency at all.
# =============================================================================

def _compute_session_flags(dt: pd.Timestamp) -> dict:
    """
    Determine which FX trading session(s) a candle's timestamp falls
    within, from UTC hour ranges in config.

    Args:
        dt: UTC timestamp of the candle

    Returns:
        Dict with session_tokyo, session_london, session_new_york
        (each 0/1), and session_overlap (1 if 2+ sessions active)
    """
    hour = dt.hour

    in_tokyo    = TOKYO_OPEN_UTC    <= hour < TOKYO_CLOSE_UTC
    in_london   = LONDON_OPEN_UTC   <= hour < LONDON_CLOSE_UTC
    in_new_york = NEW_YORK_OPEN_UTC <= hour < NEW_YORK_CLOSE_UTC

    active_count = int(in_tokyo) + int(in_london) + int(in_new_york)

    return {
        "session_tokyo"    : int(in_tokyo),
        "session_london"   : int(in_london),
        "session_new_york" : int(in_new_york),
        "session_overlap"  : int(active_count >= 2),
    }


# =============================================================================
# DIRECTIONAL FEATURES
# Replaces compute_signal_features() entirely. No `direction` parameter
# — this function computes context features for a (pair, datetime)
# point, to be used by a model that PREDICTS direction, not one that
# scores a pre-chosen direction.
# =============================================================================

def compute_directional_features(
    pair          : str,
    signal_datetime,
    prices_df     : pd.DataFrame,
    csi_df        : Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Compute all features for the directional model for a single
    pair/datetime.

    FEATURE GROUPS:

    GROUP 1 — Volatility regime (3 features):
    - atr_fast  : ATR over ATR_FAST_PERIOD candles
    - atr_slow  : ATR over ATR_SLOW_PERIOD candles
    - atr_ratio : atr_fast / atr_slow — >1 = expanding vol, <1 = compression

    GROUP 2 — Session (4 features):
    - session_tokyo, session_london, session_new_york, session_overlap

    GROUP 3 — Raw candlestick pattern flags (2 features, NO extreme
    gate, NO direction interaction — see module docstring for why):
    - is_hammer, is_shooting_star

    GROUP 4 — ADX (3 features, unchanged, direction-independent):
    - adx_value, plus_di, minus_di

    GROUP 5 — Currency Strength Index (4 features, unchanged):
    - csi_rs, csi_diff_zscore, csi_diff_roc, csi_commodity_bloc

    Args:
        pair           : FX pair symbol e.g. 'EURUSD'
        signal_datetime: Timestamp of the signal (UTC)
        prices_df      : OHLC data for this pair, up to and including
                         signal_datetime (caller filters; this function
                         also defensively filters again below)
        csi_df         : Output of run_csi_engine() / compute_csi_series()
                         for this signal_datetime — pre-computed once
                         for the whole universe per call, not recomputed
                         per pair (CSI is inherently cross-pair). Falls
                         back to 0.0 for all 4 CSI features if not
                         provided or pair not found.

    Returns:
        Dict of all features or None if insufficient data
    """
    # ── Normalise datetime types before ANY comparison ──────────────────────
    # Same defensive normalisation this project has needed in every
    # datetime-comparing function so far (labeller.py, the prior
    # features.py) — a Timestamp-vs-string or tz-aware-vs-tz-naive
    # mismatch silently produces wrong/empty results with no exception
    # raised anywhere. Applied here even though this function has fewer
    # datetime comparisons than its predecessor, since the risk is the
    # same regardless of how many comparisons there are.
    signal_datetime = pd.Timestamp(signal_datetime)
    if signal_datetime.tzinfo is not None:
        signal_datetime = signal_datetime.tz_localize(None)

    if not pd.api.types.is_datetime64_any_dtype(prices_df["datetime"]):
        prices_df = prices_df.copy()
        prices_df["datetime"] = pd.to_datetime(prices_df["datetime"], utc=True).dt.tz_localize(None)
    elif isinstance(prices_df["datetime"].dtype, pd.DatetimeTZDtype):
        prices_df = prices_df.copy()
        prices_df["datetime"] = prices_df["datetime"].dt.tz_localize(None)

    # ── Get price data up to signal datetime ───────────────────────────────
    px = prices_df[
        prices_df["datetime"] <= signal_datetime
    ].sort_values("datetime")

    if "pair" in px.columns:
        px = px[px["pair"] == pair]

    if len(px) < ATR_SLOW_PERIOD:
        # Need at least ATR_SLOW_PERIOD candles for the slow ATR to be
        # meaningful. On weekly bars, ATR_SLOW_PERIOD candles is a much
        # longer real-world span than it was on 4H bars for the same
        # config value — see module docstring's TIMEFRAME NOTE.
        return None

    # ── GROUP 1: Volatility regime ───────────────────────────────────────────
    atr_fast  = _compute_atr(px, ATR_FAST_PERIOD)
    atr_slow  = _compute_atr(px, ATR_SLOW_PERIOD)
    atr_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0

    # ── GROUP 2: Session flags ────────────────────────────────────────────────
    session_flags = _compute_session_flags(signal_datetime)

    # ── GROUP 3: Raw candlestick pattern flags (no gate, no direction) ──────
    pattern_result = compute_raw_pattern_flags(pair, px)
    if pattern_result is not None:
        is_hammer        = pattern_result["is_hammer"]
        is_shooting_star = pattern_result["is_shooting_star"]
    else:
        is_hammer        = 0
        is_shooting_star = 0

    # ── GROUP 4: ADX (trend strength, direction-independent) ────────────────
    adx_result = compute_adx_latest(pair, px, str(signal_datetime))
    if adx_result is not None:
        adx_value = adx_result["adx_value"]
        plus_di   = adx_result["plus_di"]
        minus_di  = adx_result["minus_di"]
    else:
        adx_value = 0.0
        plus_di   = 0.0
        minus_di  = 0.0

    # ── GROUP 5: Currency Strength Index ─────────────────────────────────────
    csi_rs             = 0.0
    csi_diff_zscore    = 0.0
    csi_diff_roc       = 0.0
    csi_commodity_bloc = 0.0
    if csi_df is not None and not csi_df.empty:
        csi_row = csi_df[csi_df["pair"] == pair]
        if not csi_row.empty:
            row0 = csi_row.iloc[0]
            csi_rs_val   = row0.get("csi_rs", np.nan)
            zscore_val   = row0.get("csi_diff_zscore", np.nan)
            roc_val      = row0.get("csi_diff_roc", np.nan)
            bloc_val     = row0.get("csi_commodity_bloc", np.nan)
            csi_rs             = float(csi_rs_val) if pd.notna(csi_rs_val) else 0.0
            csi_diff_zscore    = float(zscore_val) if pd.notna(zscore_val) else 0.0
            csi_diff_roc       = float(roc_val)    if pd.notna(roc_val)    else 0.0
            csi_commodity_bloc = float(bloc_val)   if pd.notna(bloc_val)   else 0.0

    # ── Assemble all features ─────────────────────────────────────────────────
    features = {
        # Identifiers (not used in training — dropped before fit)
        "pair"               : pair,
        "datetime"           : signal_datetime,

        # Group 1: Volatility regime
        "atr_fast"           : round(atr_fast,  6),
        "atr_slow"           : round(atr_slow,  6),
        "atr_ratio"          : round(atr_ratio, 4),

        # Group 2: Session
        **session_flags,

        # Group 3: Raw candlestick pattern flags
        "is_hammer"          : is_hammer,
        "is_shooting_star"   : is_shooting_star,

        # Group 4: ADX
        "adx_value"          : adx_value,
        "plus_di"            : plus_di,
        "minus_di"           : minus_di,

        # Group 5: Currency Strength Index
        "csi_rs"             : round(csi_rs,             6),
        "csi_diff_zscore"    : round(csi_diff_zscore,    6),
        "csi_diff_roc"       : round(csi_diff_roc,       6),
        "csi_commodity_bloc" : round(csi_commodity_bloc, 6),
    }

    return features


# =============================================================================
# FEATURE MATRIX BUILDER
# Replaces build_signal_feature_matrix() entirely. No `direction`
# column anywhere — labels_df now comes from labeller.label_direction()
# and has columns [pair, datetime, label] only.
# =============================================================================

def build_directional_feature_matrix(
    prices_df    : pd.DataFrame,
    labels_df    : pd.DataFrame,
    csi_series_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for training a directional model.

    Args:
        prices_df     : Full OHLC data [pair, datetime, open, high,
                        low, close]
        labels_df     : Output of labeller.label_direction()
                        [pair, datetime, label]
        csi_series_df : Output of engines.csi.compute_csi_series() —
                        full historical CSI series for all pairs. If
                        None, all 4 CSI features default to 0.0 for
                        every training example (logged as a warning).

    Returns:
        DataFrame with one row per labelled example, ready for model
        training (label column included, pair/datetime kept as
        identifiers to drop before fit)
    """
    logger.info(
        f"Building directional feature matrix | "
        f"{len(labels_df)} labelled examples"
    )

    if csi_series_df is None or csi_series_df.empty:
        logger.warning(
            "build_directional_feature_matrix: no csi_series_df provided — "
            "all 4 CSI features will be 0.0 for all training examples"
        )

    # ── Normalise ALL datetime columns ONCE, here, before the loop ─────────
    # Same class of bug this project has hit twice already (silent
    # zero-match comparisons from mismatched datetime representations)
    # — normalising once here, before the loop, both fixes it and
    # avoids redundant per-row work at scale.
    def _normalise_dt(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "datetime" not in df.columns:
            return df
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        elif isinstance(df["datetime"].dtype, pd.DatetimeTZDtype):
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        return df

    prices_df = _normalise_dt(prices_df)
    labels_df = _normalise_dt(labels_df)
    if csi_series_df is not None and not csi_series_df.empty:
        csi_series_df = _normalise_dt(csi_series_df)

    rows   = []
    failed = 0

    # Pre-group prices once, per pair
    prices_dict = {p: df.sort_values("datetime") for p, df in prices_df.groupby("pair")}

    for row in labels_df.itertuples(index=False):
        pair  = row.pair
        dt    = row.datetime
        label = row.label

        px = prices_dict.get(pair)

        if px is None or px.empty:
            failed += 1
            continue

        csi_snapshot = None
        if csi_series_df is not None and not csi_series_df.empty:
            csi_snapshot = csi_series_df[csi_series_df["datetime"] == dt]

        features = compute_directional_features(
            pair            = pair,
            signal_datetime = dt,
            prices_df       = px,
            csi_df          = csi_snapshot,
        )

        if features is None:
            failed += 1
            continue

        features["label"] = label
        rows.append(features)

    result = pd.DataFrame(rows)

    if "datetime" in result.columns:
        result = result.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"Directional feature matrix built | "
        f"Rows: {len(result)} | "
        f"Failed: {failed}"
    )

    return result


# =============================================================================
# FEATURE COLUMNS — used at both train and inference time
# =============================================================================

DIRECTIONAL_FEATURE_COLS = [
    # Volatility regime
    "atr_fast",
    "atr_slow",
    "atr_ratio",
    # Session
    "session_tokyo",
    "session_london",
    "session_new_york",
    "session_overlap",
    # Raw candlestick pattern flags (no extreme gate, no direction)
    "is_hammer",
    "is_shooting_star",
    # ADX — trend strength, direction-independent
    "adx_value",
    "plus_di",
    "minus_di",
    # Currency Strength Index
    "csi_rs",
    "csi_diff_zscore",
    "csi_diff_roc",
    "csi_commodity_bloc",
]
