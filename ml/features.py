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

ADDED (per project decision, after comparing against the EURUSD-only
reference project's 0.76-AUC feature set — this basket redesign had
originally dropped several of its core signal families without a
deliberate decision to do so):
   - rsi (Wilder RSI, own-pair) + rsi_oversold / rsi_overbought flags
     — was a top feature in the reference project; had no LinReg/SMC
     dependency and should never have been dropped.
   - macd / macd_signal / macd_hist — same reasoning as RSI.
   - price_vs_sma20 / price_vs_sma50 / sma_cross — trend-position
     features, direction-independent, no LinReg dependency (LinReg was
     a regression channel; a plain SMA is not that).
   - up_streak / down_streak / vol_expanding — regime/streak features
     from the reference project, direction-independent.
   - MACRO FEATURES (genuinely new, not a restoration): per-basket
     external drivers — DXY/gold/US10Y for the usd basket, WTI for
     cad, VIX for chf/jpy/crosses (config.yaml's macro.basket_drivers).
     See engines/macro.py. This is the one category of feature the
     reference project's edge relied on that this project never had
     an equivalent for at all (CSI is cross-pair but still fully
     internal to the 28-pair universe's own price action).

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
from engines.macro import get_macro_features_for_basket

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
BOLLINGER_CFG = config["bollinger"]
MOMENTUM_CFG  = config["momentum"]
RSI_CFG       = config["rsi"]
MACD_CFG      = config["macd"]
TREND_CFG     = config["trend"]
STOCH_CFG     = config["stochastic"]
LAG_CFG       = config["lag_features"]
UNIVERSE_CFG  = config["universe"]

ATR_FAST_PERIOD = ATR_CFG["fast_period"]
ATR_SLOW_PERIOD = ATR_CFG["slow_period"]

TOKYO_OPEN_UTC     = SESSION_CFG["tokyo_open_utc"]
TOKYO_CLOSE_UTC    = SESSION_CFG["tokyo_close_utc"]
LONDON_OPEN_UTC    = SESSION_CFG["london_open_utc"]
LONDON_CLOSE_UTC   = SESSION_CFG["london_close_utc"]
NEW_YORK_OPEN_UTC  = SESSION_CFG["new_york_open_utc"]
NEW_YORK_CLOSE_UTC = SESSION_CFG["new_york_close_utc"]

# Bollinger Bands — replaces LinReg's "where is price relative to its
# recent range" role, computed from a plain rolling SMA/STD instead of
# a regression channel. 20-period SMA, 2 standard deviations — the
# conventional default, chosen deliberately for weekly bars (~5 months
# of history per band), not inherited from a daily-bar config.
BOLLINGER_PERIOD    = BOLLINGER_CFG["period"]
BOLLINGER_NUM_STD   = BOLLINGER_CFG["num_std"]

# Rate of Change — the momentum feature. 4-period lookback (~1 month
# on weekly bars), matching the reference EURUSD project's short-
# horizon momentum feel.
ROC_PERIOD = MOMENTUM_CFG["roc_period"]

# RSI — Wilder's standard period, timeframe-invariant convention.
RSI_PERIOD = RSI_CFG["period"]

# MACD — standard 12/26/9, timeframe-invariant convention.
MACD_FAST_PERIOD   = MACD_CFG["fast_period"]
MACD_SLOW_PERIOD   = MACD_CFG["slow_period"]
MACD_SIGNAL_PERIOD = MACD_CFG["signal_period"]

# Trend/SMA position — matches the reference EURUSD project's
# price_vs_sma20 / price_vs_sma50 / sma_cross features.
SMA_FAST_PERIOD = TREND_CFG["sma_fast"]
SMA_SLOW_PERIOD = TREND_CFG["sma_slow"]
# sma_10/ema_10 — the reference project's add_trend_features also
# computes these (sma_10 feeds its own sma_cross = sma_10 > sma_20;
# ema_10 is its own feature). Same window length as the reference,
# carried over 1:1 in candle-count (weekly instead of daily).
SMA_SHORT_PERIOD = TREND_CFG["sma_short"]
EMA_SHORT_PERIOD = TREND_CFG["ema_short"]

# Stochastic Oscillator — stoch_k was the reference project's 2nd
# most important single feature; entirely missing from this project
# until now. Same window as the reference, carried over 1:1 in
# candle-count.
STOCH_PERIOD = STOCH_CFG["period"]

# Simple N-candle-ago returns + rolling return-volatility windows —
# from the reference project's add_lag_features. Same lag set / window
# lengths as the reference, carried over 1:1 in candle-count.
RETURN_LAGS       = LAG_CFG["lags"]
VOLATILITY_SHORT  = LAG_CFG["volatility_short"]
VOLATILITY_LONG   = LAG_CFG["volatility_long"]

# pair -> basket name lookup, built once from config.yaml's
# universe.baskets, so compute_directional_features can find which
# basket (and therefore which macro drivers) a given pair belongs to
# without the caller needing to pass it explicitly every time.
_PAIR_TO_BASKET = {
    pair: basket_name
    for basket_name, pairs in UNIVERSE_CFG["baskets"].items()
    for pair in pairs
}

# basket -> expected macro feature column names, e.g.
# "usd" -> ["dxy_return_short", "dxy_return_long", "dxy_above_sma20",
#           "dxy_rsi", "gold_return_short", ..., "us10y_rsi"].
# Built once from config.yaml's macro.basket_drivers + macro.drivers,
# matching the column-naming convention engines/macro.py's
# get_macro_features_for_basket() produces (add_prefix with the
# lowercased driver storage_symbol). Used so every row for a given
# basket has the SAME macro columns present (falling back to 0.0),
# even on datetimes where macro_features_df has no as-of row yet.
MACRO_CFG          = config["macro"]
_MACRO_DRIVERS     = MACRO_CFG["drivers"]
_MACRO_BASKET_MAP  = MACRO_CFG["basket_drivers"]
_MACRO_FEATURE_SUFFIXES = ["return_short", "return_long", "above_sma20", "rsi"]

MACRO_FEATURE_COLS_BY_BASKET = {
    basket_name: [
        f"{_MACRO_DRIVERS[driver_symbol]['storage_symbol'].lower()}_{suffix}"
        for driver_symbol in driver_symbols
        if driver_symbol in _MACRO_DRIVERS
        for suffix in _MACRO_FEATURE_SUFFIXES
    ]
    for basket_name, driver_symbols in _MACRO_BASKET_MAP.items()
}

# basket -> one-hot pair-identity column names, e.g.
# "usd" -> ["pair_is_eurusd", "pair_is_gbpusd", "pair_is_audusd",
#           "pair_is_nzdusd"]. Per project decision: within a basket,
# pairs sharing the same quote currency (e.g. EURUSD/AUDUSD both in
# "usd") still behave differently under the same macro regime — the
# model previously had no way to distinguish which pair a row
# belonged to. One-hot (not ordinal) since pairs have no natural
# ordering and a basket's pair set is small and fixed, so the extra
# columns are cheap. Built once per basket from config.yaml's
# universe.baskets, in the SAME order every time (sorted), so column
# position is stable across training and inference for a given basket.
PAIR_IDENTITY_COLS_BY_BASKET = {
    basket_name: [f"pair_is_{pair.lower()}" for pair in sorted(pairs)]
    for basket_name, pairs in UNIVERSE_CFG["baskets"].items()
}


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
# BOLLINGER BANDS
# Replaces LinReg's "where is price relative to its recent range" role
# — per the reference EURUSD project's README, Bollinger-derived
# features were its SECOND most important feature group. Computed from
# a plain rolling SMA/STD, not a regression channel, so this has no
# LinReg dependency at all.
# =============================================================================

def _compute_bollinger(px: pd.DataFrame) -> dict:
    """
    Compute Bollinger Band position for the latest candle.

    MATHS:
    - sma    = rolling mean of close over BOLLINGER_PERIOD candles
    - std    = rolling std of close over BOLLINGER_PERIOD candles
    - upper  = sma + BOLLINGER_NUM_STD * std
    - lower  = sma - BOLLINGER_NUM_STD * std
    - %B     = (close - lower) / (upper - lower) — 0 = at lower band,
               1 = at upper band, 0.5 = at the middle SMA. Unbounded
               beyond [0,1] when price closes outside the bands
               entirely (a genuine, informative signal — not clipped).
    - bandwidth = (upper - lower) / sma — a volatility-regime measure,
               analogous in spirit to ATR ratio but band-based rather
               than true-range-based.

    Args:
        px: OHLC DataFrame, sorted datetime ascending, already
            filtered to <= signal datetime by the caller

    Returns:
        Dict with bollinger_pct_b and bollinger_bandwidth. Falls back
        to pct_b=0.5 (i.e. "at the middle, no information") and
        bandwidth=0.0 if fewer than BOLLINGER_PERIOD candles are
        available — a neutral default, not a fabricated extreme.
    """
    recent = px["close"].tail(BOLLINGER_PERIOD)

    if len(recent) < BOLLINGER_PERIOD:
        return {"bollinger_pct_b": 0.5, "bollinger_bandwidth": 0.0}

    sma = recent.mean()
    std = recent.std()

    upper = sma + BOLLINGER_NUM_STD * std
    lower = sma - BOLLINGER_NUM_STD * std

    current_close = px["close"].iloc[-1]

    band_range = upper - lower

    # Use a RELATIVE tolerance (band_range vs. price scale), not a
    # strict > 0 check — floating-point std() on genuinely-flat prices
    # (e.g. all closes exactly 1.10) does not return exact 0.0, it
    # returns something like 8e-16 due to float representation. A
    # strict > 0 check treats that near-zero as "a real band," and
    # computes pct_b from an almost-zero-width band — producing a
    # meaningless, wildly sensitive value (e.g. 0.25 instead of the
    # intended neutral 0.5) instead of correctly falling back. Caught
    # in testing: a flat-price fixture failed the "neutral fallback"
    # assertion for exactly this reason.
    if band_range > (1e-8 * max(abs(sma), 1e-8)):
        pct_b     = (current_close - lower) / band_range
        bandwidth = band_range / sma if sma > 0 else 0.0
    else:
        # Effectively zero band width (flat price over the whole
        # window) — no meaningful band to position against; neutral
        # defaults, not a divide-by-near-zero.
        pct_b     = 0.5
        bandwidth = 0.0

    return {
        "bollinger_pct_b"    : float(pct_b),
        "bollinger_bandwidth": float(bandwidth),
    }


# =============================================================================
# RATE OF CHANGE — the momentum feature
# Per the reference EURUSD project's README momentum feature group.
# =============================================================================

def _compute_roc(px: pd.DataFrame) -> float:
    """
    Compute Rate of Change: percentage price change over ROC_PERIOD
    candles (4 weekly candles ≈ 1 month, per project decision).

    MATHS:
        roc = (close[t] - close[t - ROC_PERIOD]) / close[t - ROC_PERIOD]

    Args:
        px: OHLC DataFrame, sorted datetime ascending, already
            filtered to <= signal datetime by the caller

    Returns:
        ROC as a plain float (e.g. 0.02 = +2% over the period).
        Falls back to 0.0 (no momentum signal) if fewer than
        ROC_PERIOD + 1 candles are available.
    """
    if len(px) < ROC_PERIOD + 1:
        return 0.0

    current_close = px["close"].iloc[-1]
    past_close    = px["close"].iloc[-(ROC_PERIOD + 1)]

    if past_close == 0:
        return 0.0

    return float((current_close - past_close) / past_close)


# =============================================================================
# RSI, MACD, SMA-TREND, REGIME/STREAK FEATURES
# Restored from the EURUSD reference project — none of these ever
# depended on LinReg or SMC, and their absence from the original
# basket redesign was an oversight, not a deliberate scope decision
# (see module docstring's "ADDED" section).
# =============================================================================

def _compute_rsi(px: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    """
    Standard Wilder RSI on this pair's own close series.

    Returns:
        RSI value (0-100). Falls back to 50.0 (neutral) if fewer than
        period+1 candles are available.
    """
    close = px["close"]
    if len(close) < period + 1:
        return 50.0

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    latest_avg_gain = avg_gain.iloc[-1]
    latest_avg_loss = avg_loss.iloc[-1]

    if pd.isna(latest_avg_gain) or pd.isna(latest_avg_loss):
        return 50.0
    if latest_avg_loss == 0:
        return 100.0 if latest_avg_gain > 0 else 50.0

    rs  = latest_avg_gain / latest_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def _compute_macd(px: pd.DataFrame) -> dict:
    """
    Standard MACD: EMA_fast - EMA_slow, signal = EMA of MACD line,
    histogram = MACD - signal.

    Returns:
        Dict with macd, macd_signal, macd_hist. Falls back to all
        0.0 (neutral — no trend signal) if fewer than
        MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD candles are available.
    """
    close = px["close"]
    min_required = MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD
    if len(close) < min_required:
        return {"macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0}

    ema_fast   = close.ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    hist        = macd_line - signal_line

    return {
        "macd"       : float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist"  : float(hist.iloc[-1]),
    }


def _compute_sma_trend(px: pd.DataFrame) -> dict:
    """
    Price-vs-trend position + trend/regime flags, matching the
    EURUSD reference project's add_trend_features + add_regime_features
    exactly (window lengths carried over 1:1 in candle-count, weekly
    instead of daily):

    - sma_10, ema_10  : the short-window trend features themselves
                        (not just used for a ratio — the reference
                        project keeps these as standalone features)
    - price_vs_sma20/50: how far current close sits above/below its
                        own SMA_FAST/SMA_SLOW, expressed as a fraction
                        of price (not raw price units — FX pairs quote
                        at very different decimal scales, e.g. JPY
                        pairs in 2 decimals vs. others in 4-5, so a raw
                        price difference isn't comparable across pairs
                        the way a % difference is)
    - above_sma20/50  : 1/0 flags — is price currently above its own
                        SMA_FAST/SMA_SLOW (reference project's
                        add_regime_features)
    - sma_cross       : 1 if SMA_SHORT (10) > SMA_FAST (20), else 0 —
                        NOTE this matches the reference project's own
                        definition (sma_10 > sma_20) exactly. This is
                        NOT "fast SMA > slow SMA" in the 20/50 sense —
                        it's specifically the short/fast pairing the
                        reference project used, corrected from an
                        earlier version of this file that had used
                        20-vs-50 instead.

    Falls back to 0.0 / 0 for any feature whose required SMA/EMA
    window exceeds available history.
    """
    close = px["close"]
    current_close = close.iloc[-1]

    def _sma(period):
        if len(close) < period:
            return None
        return float(close.tail(period).mean())

    sma10 = _sma(SMA_SHORT_PERIOD)
    sma20 = _sma(SMA_FAST_PERIOD)
    sma50 = _sma(SMA_SLOW_PERIOD)

    if len(close) < EMA_SHORT_PERIOD:
        ema10 = None
    else:
        ema10 = float(close.ewm(span=EMA_SHORT_PERIOD, adjust=False).mean().iloc[-1])

    price_vs_sma20 = float((current_close - sma20) / sma20) if sma20 and sma20 > 0 else 0.0
    price_vs_sma50 = float((current_close - sma50) / sma50) if sma50 and sma50 > 0 else 0.0
    above_sma20    = int(current_close > sma20) if sma20 is not None else 0
    above_sma50    = int(current_close > sma50) if sma50 is not None else 0
    sma_cross      = int(sma10 > sma20) if (sma10 is not None and sma20 is not None) else 0

    return {
        "sma_10"         : sma10 if sma10 is not None else 0.0,
        "ema_10"         : ema10 if ema10 is not None else 0.0,
        "price_vs_sma20" : price_vs_sma20,
        "price_vs_sma50" : price_vs_sma50,
        "above_sma20"    : above_sma20,
        "above_sma50"    : above_sma50,
        "sma_cross"      : sma_cross,
    }


def _compute_stochastic(px: pd.DataFrame, period: int = STOCH_PERIOD) -> dict:
    """
    Stochastic Oscillator (%K, %D) — the EURUSD reference project's
    2nd most important single feature (stoch_k), entirely missing from
    this project until now. Standard definition:
        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %D = 3-period SMA of %K
    Same window (STOCH_PERIOD) as the reference project, carried over
    1:1 in candle-count.

    Returns:
        Dict with stoch_k, stoch_d. Falls back to 50.0/50.0 (neutral,
        mid-range) if fewer than period+3 candles are available (the
        +3 covers %D's own 3-period smoothing of %K).
    """
    if len(px) < period + 3:
        return {"stoch_k": 50.0, "stoch_d": 50.0}

    high = px["high"]
    low  = px["low"]
    close = px["close"]

    lowest_low   = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    denom        = (highest_high - lowest_low).replace(0, np.nan)

    pct_k = 100 * (close - lowest_low) / denom
    pct_d = pct_k.rolling(3).mean()

    k_val = pct_k.iloc[-1]
    d_val = pct_d.iloc[-1]

    return {
        "stoch_k": float(k_val) if pd.notna(k_val) else 50.0,
        "stoch_d": float(d_val) if pd.notna(d_val) else 50.0,
    }


def _compute_lag_features(px: pd.DataFrame) -> dict:
    """
    Simple N-candle-ago returns + rolling return-volatility windows,
    matching the EURUSD reference project's add_lag_features exactly
    (return_lag_5 ranked in its top 15; distinct from price_roc, which
    uses a single fixed lookback). Same lag set / window lengths as
    the reference, carried over 1:1 in candle-count.

    Returns:
        Dict with return_lag_{N} for each N in RETURN_LAGS, hl_range
        (High-Low)/Close for the latest candle, and volatility_short/
        volatility_long (rolling stdev of returns). Each individual
        lag/window falls back to 0.0 if insufficient history for that
        specific window — shorter lags/windows still populate even if
        a longer one doesn't have enough history yet.
    """
    close = px["close"]
    current_close = close.iloc[-1]

    result = {}
    for lag in RETURN_LAGS:
        if len(close) < lag + 1:
            result[f"return_lag_{lag}"] = 0.0
        else:
            past_close = close.iloc[-1 - lag]
            result[f"return_lag_{lag}"] = float((current_close - past_close) / past_close) if past_close != 0 else 0.0

    latest_high  = px["high"].iloc[-1]
    latest_low   = px["low"].iloc[-1]
    result["hl_range"] = float((latest_high - latest_low) / current_close) if current_close != 0 else 0.0

    returns = close.pct_change()

    if len(returns.dropna()) < VOLATILITY_SHORT:
        result["volatility_short"] = 0.0
    else:
        result["volatility_short"] = float(returns.tail(VOLATILITY_SHORT).std())

    if len(returns.dropna()) < VOLATILITY_LONG:
        result["volatility_long"] = 0.0
    else:
        result["volatility_long"] = float(returns.tail(VOLATILITY_LONG).std())

    return result


def _compute_regime_flags(px: pd.DataFrame, rsi_value: float, volatility_short: float, volatility_long: float) -> dict:
    """
    Regime/streak features from the reference project: how extended is
    the current move, not just its raw indicator value.

    STREAK DEFINITION — matches the reference project's
    add_regime_features exactly: up_streak/down_streak are the
    consecutive-run LENGTH multiplied by whether the LATEST candle's
    return was itself positive/negative (the reference project's
    groupby/cumcount-then-gate-by-latest-sign construction). In
    practice, for the single latest row this reduces to: count
    consecutive up (or down) closes ending at the latest candle, same
    as counting the run length directly — this implementation counts
    the run directly rather than reproducing the groupby machinery,
    since both produce the identical value for "the latest row's
    streak length," which is all a single-timestamp feature needs.

    Returns:
        Dict with:
        - up_streak / down_streak: consecutive up/down closes ending
          at the latest candle (one of these is always 0)
        - rsi_oversold / rsi_overbought: 1/0 flags at the conventional
          30/70 RSI thresholds
        - vol_expanding: 1 if volatility_short > volatility_long
          (rolling return-stdev currently expanding vs. its own slower
          baseline), else 0 — matches the reference project's own
          definition (compares its volatility_10/volatility_20
          return-stdev features), NOT atr_fast/atr_slow (which is a
          separate, already-existing feature in this project measuring
          the same underlying concept a different way).
    """
    closes = px["close"].values
    up_streak   = 0
    down_streak = 0

    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if down_streak > 0:
                break
            up_streak += 1
        elif closes[i] < closes[i - 1]:
            if up_streak > 0:
                break
            down_streak += 1
        else:
            break

    return {
        "up_streak"      : up_streak,
        "down_streak"    : down_streak,
        "rsi_oversold"   : int(rsi_value <= 30),
        "rsi_overbought" : int(rsi_value >= 70),
        "vol_expanding"  : int(volatility_short > volatility_long) if volatility_long > 0 else 0,
    }


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
    macro_features_df: Optional[pd.DataFrame] = None,
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

    GROUP 6 — Bollinger Bands (2 features): replaces LinReg's "where is
    price relative to its recent range" role, per the reference
    project's README (its second most important feature group):
    - bollinger_pct_b   : (close - lower) / (upper - lower), 0=lower
      band, 1=upper band, 0.5=middle SMA
    - bollinger_bandwidth: (upper - lower) / sma — volatility-regime
      measure, band-based analog to atr_ratio

    GROUP 7 — Rate of Change (1 feature): the momentum feature, per
    the reference project's README:
    - roc : (close[t] - close[t-ROC_PERIOD]) / close[t-ROC_PERIOD]

    GROUP 8 — RSI (3 features): restored from the reference project.
    - rsi, rsi_oversold, rsi_overbought

    GROUP 9 — MACD (3 features): restored from the reference project.
    - macd, macd_signal, macd_hist

    GROUP 10 — SMA trend position (3 features): restored from the
    reference project.
    - price_vs_sma20, price_vs_sma50, sma_cross

    GROUP 11 — Regime/streak (3 features): restored from the
    reference project.
    - up_streak, down_streak, vol_expanding

    GROUP 12 — Macro drivers (variable count per basket): NEW, per
    project decision — external, non-price-derived features mapped to
    this pair's basket (e.g. dxy_return, gold_rsi for usd-basket
    pairs; wti_return for cad-basket pairs). See engines/macro.py.
    Falls back to 0.0 for every configured driver column if
    macro_features_df is not provided or has no row for this datetime.

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
        macro_features_df: Output of engines.macro.get_macro_features_for_basket()
                         for THIS pair's basket — pre-computed once per
                         basket per training run (mirrors how csi_df is
                         pre-computed once for the whole universe), not
                         recomputed per pair. Columns are looked up by
                         nearest datetime <= signal_datetime.

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

    # ── GROUP 6: Bollinger Bands ─────────────────────────────────────────────
    bollinger = _compute_bollinger(px)

    # ── GROUP 7: Rate of Change (momentum) ──────────────────────────────────
    # Named price_roc (not bare "roc") to avoid any confusion with
    # csi_diff_roc above — genuinely different things (price momentum
    # vs. the rate of change of the CSI relative-strength gap).
    price_roc = _compute_roc(px)

    # ── GROUP 8: RSI ─────────────────────────────────────────────────────────
    rsi_value = _compute_rsi(px)

    # ── GROUP 9: MACD ────────────────────────────────────────────────────────
    macd_result = _compute_macd(px)

    # ── GROUP 10: SMA trend position (+ sma_10/ema_10/above_sma20/50) ───────
    sma_trend = _compute_sma_trend(px)

    # ── GROUP 13: Stochastic Oscillator ─────────────────────────────────────
    # NEW — reference project's 2nd most important single feature,
    # missing from this project until now.
    stochastic = _compute_stochastic(px)

    # ── GROUP 14: Lag returns + hl_range + rolling return-volatility ────────
    # NEW — return_lag_5 ranked in the reference project's top 15;
    # volatility_short/long feed GROUP 11's vol_expanding below.
    lag_features = _compute_lag_features(px)

    # ── GROUP 11: Regime/streak flags ───────────────────────────────────────
    # vol_expanding now compares lag_features' own volatility_short/long
    # (matching the reference project's definition) rather than
    # atr_fast/atr_slow (which remains its own separate feature, Group 1).
    regime_flags = _compute_regime_flags(
        px, rsi_value,
        lag_features["volatility_short"], lag_features["volatility_long"],
    )

    # ── GROUP 12: Macro drivers (per this pair's basket) ────────────────────
    # Falls back to 0.0 for every driver column this pair's basket is
    # configured for (see config.yaml's macro.basket_drivers), rather
    # than silently omitting the columns — keeps get_directional_feature_cols
    # consistent across every row regardless of macro data availability.
    basket_name    = _PAIR_TO_BASKET.get(pair)
    macro_features = {}

    if basket_name is not None:
        expected_macro_cols = MACRO_FEATURE_COLS_BY_BASKET.get(basket_name, [])

        if macro_features_df is not None and not macro_features_df.empty:
            macro_asof = macro_features_df[macro_features_df.index <= signal_datetime]
            if not macro_asof.empty:
                latest_macro_row = macro_asof.iloc[-1]
                for col in expected_macro_cols:
                    val = latest_macro_row.get(col, np.nan)
                    macro_features[col] = float(val) if pd.notna(val) else 0.0

        # Fill in 0.0 for any expected column not populated above
        # (no macro_features_df provided, or no row as-of this date yet).
        for col in expected_macro_cols:
            macro_features.setdefault(col, 0.0)

    # ── GROUP 15: Pair identity (one-hot, per basket) ───────────────────────
    # NEW, per project decision: within a basket, pairs sharing the same
    # quote currency still behave differently under the same macro
    # regime — this lets the model tell them apart. Exactly one column
    # is 1 (this row's own pair), all others in the basket's set are 0.
    pair_identity_features = {}
    if basket_name is not None:
        expected_pair_cols = PAIR_IDENTITY_COLS_BY_BASKET.get(basket_name, [])
        this_pair_col = f"pair_is_{pair.lower()}"
        for col in expected_pair_cols:
            pair_identity_features[col] = 1 if col == this_pair_col else 0

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

        # Group 6: Bollinger Bands
        "bollinger_pct_b"    : round(bollinger["bollinger_pct_b"],     6),
        "bollinger_bandwidth": round(bollinger["bollinger_bandwidth"], 6),

        # Group 7: Rate of Change (momentum)
        "price_roc"          : round(price_roc, 6),

        # Group 8: RSI
        "rsi"                : round(rsi_value, 4),

        # Group 9: MACD
        "macd"               : round(macd_result["macd"],        6),
        "macd_signal"        : round(macd_result["macd_signal"], 6),
        "macd_hist"          : round(macd_result["macd_hist"],   6),

        # Group 10: SMA trend position
        "sma_10"             : round(sma_trend["sma_10"], 6),
        "ema_10"             : round(sma_trend["ema_10"], 6),
        "price_vs_sma20"     : round(sma_trend["price_vs_sma20"], 6),
        "price_vs_sma50"     : round(sma_trend["price_vs_sma50"], 6),
        "above_sma20"        : sma_trend["above_sma20"],
        "above_sma50"        : sma_trend["above_sma50"],
        "sma_cross"          : sma_trend["sma_cross"],

        # Group 11: Regime/streak flags
        "up_streak"          : regime_flags["up_streak"],
        "down_streak"        : regime_flags["down_streak"],
        "rsi_oversold"       : regime_flags["rsi_oversold"],
        "rsi_overbought"     : regime_flags["rsi_overbought"],
        "vol_expanding"      : regime_flags["vol_expanding"],

        # Group 12: Macro drivers (per basket — column set varies)
        **{col: round(val, 6) for col, val in macro_features.items()},

        # Group 13: Stochastic Oscillator
        "stoch_k"            : round(stochastic["stoch_k"], 4),
        "stoch_d"            : round(stochastic["stoch_d"], 4),

        # Group 14: Lag returns + hl_range + rolling return-volatility
        **{
            f"return_lag_{lag}": round(lag_features[f"return_lag_{lag}"], 6)
            for lag in RETURN_LAGS
        },
        "hl_range"           : round(lag_features["hl_range"], 6),
        "volatility_short"   : round(lag_features["volatility_short"], 6),
        "volatility_long"    : round(lag_features["volatility_long"],  6),

        # Group 15: Pair identity (one-hot, per basket — column set varies)
        **pair_identity_features,
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
    macro_features_by_basket: Optional[dict[str, pd.DataFrame]] = None,
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
        macro_features_by_basket: Dict mapping basket name -> output of
                        engines.macro.get_macro_features_for_basket()
                        for that basket, e.g. {"usd": <df>, "cad": <df>,
                        ...} — computed ONCE per basket per training
                        run (mirrors csi_series_df's "compute once for
                        the whole universe" pattern), not recomputed
                        per row. If None, all macro features default to
                        0.0 for every training example (logged as a
                        warning).

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

    if not macro_features_by_basket:
        logger.warning(
            "build_directional_feature_matrix: no macro_features_by_basket "
            "provided — all macro driver features will be 0.0 for all "
            "training examples"
        )
        macro_features_by_basket = {}

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

    # Macro feature DataFrames are indexed by datetime (not a column —
    # see engines/macro.py's get_macro_features_for_basket), so they
    # need index normalisation rather than the column-based helper above.
    normalised_macro_by_basket = {}
    for basket_name, macro_df in macro_features_by_basket.items():
        if macro_df is None or macro_df.empty:
            normalised_macro_by_basket[basket_name] = macro_df
            continue
        macro_df = macro_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(macro_df.index):
            macro_df.index = pd.to_datetime(macro_df.index, utc=True).tz_localize(None)
        elif isinstance(macro_df.index.dtype, pd.DatetimeTZDtype):
            macro_df.index = macro_df.index.tz_localize(None)
        normalised_macro_by_basket[basket_name] = macro_df

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

        basket_name = _PAIR_TO_BASKET.get(pair)
        macro_snapshot = normalised_macro_by_basket.get(basket_name) if basket_name else None

        features = compute_directional_features(
            pair            = pair,
            signal_datetime = dt,
            prices_df       = px,
            csi_df          = csi_snapshot,
            macro_features_df = macro_snapshot,
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
# NOTE ON MACRO COLUMNS AND PAIR-IDENTITY COLUMNS: unlike every other
# feature group, these two are NOT the same across all 5 basket
# models:
#   - macro: usd gets 12 columns (DXY+GOLD+US10Y x 4 features each),
#     cad/chf/jpy/crosses each get 4 (their single driver x 4 features)
#   - pair identity: usd/crosses get 4-6 one-hot columns, cad gets 5,
#     chf/jpy get 6-7 — one column per pair IN THAT BASKET (see
#     PAIR_IDENTITY_COLS_BY_BASKET)
# A single flat DIRECTIONAL_FEATURE_COLS list would be wrong for every
# basket in a different way. get_directional_feature_cols(basket_name)
# below is the correct way to get a basket's column set — train_models.py
# and any inference code should call it per-basket rather than using
# a single shared constant.

_BASE_DIRECTIONAL_FEATURE_COLS = [
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
    # Bollinger Bands (replaces LinReg's band-position role)
    "bollinger_pct_b",
    "bollinger_bandwidth",
    # Rate of Change (momentum)
    "price_roc",
    # RSI (restored from the reference project)
    "rsi",
    "rsi_oversold",
    "rsi_overbought",
    # MACD (restored from the reference project)
    "macd",
    "macd_signal",
    "macd_hist",
    # SMA trend position (restored from the reference project, with
    # sma_10/ema_10/above_sma20/above_sma50 added to match it exactly)
    "sma_10",
    "ema_10",
    "price_vs_sma20",
    "price_vs_sma50",
    "above_sma20",
    "above_sma50",
    "sma_cross",
    # Regime/streak flags (restored from the reference project)
    "up_streak",
    "down_streak",
    "vol_expanding",
    # Stochastic Oscillator (NEW — reference project's 2nd most
    # important single feature, was missing from this project)
    "stoch_k",
    "stoch_d",
    # Lag returns + hl_range + rolling return-volatility (NEW —
    # return_lag_5 ranked in the reference project's top 15)
    *[f"return_lag_{lag}" for lag in RETURN_LAGS],
    "hl_range",
    "volatility_short",
    "volatility_long",
]


def get_directional_feature_cols(basket_name: str) -> list[str]:
    """
    Return the full ordered feature-column list for a given basket's
    model — the shared base columns (identical across every basket)
    plus that basket's specific macro driver columns AND pair-identity
    one-hot columns (both differ per basket — see the module-level
    note above).

    Args:
        basket_name: e.g. "usd", "cad", "chf", "jpy", "crosses"

    Returns:
        List of column names, in the same order compute_directional_features
        produces them, for use as the X columns when training or
        scoring this basket's model.
    """
    macro_cols = MACRO_FEATURE_COLS_BY_BASKET.get(basket_name, [])
    pair_cols  = PAIR_IDENTITY_COLS_BY_BASKET.get(basket_name, [])
    return _BASE_DIRECTIONAL_FEATURE_COLS + macro_cols + pair_cols


# Kept for any existing caller that imports this name directly — this
# is ONLY the shared base set and does NOT include any basket's macro
# or pair-identity columns. New code (train_models.py, inference)
# should call get_directional_feature_cols(basket_name) instead, which
# returns the correct full set for a specific basket.
DIRECTIONAL_FEATURE_COLS = _BASE_DIRECTIONAL_FEATURE_COLS
