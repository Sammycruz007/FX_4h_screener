"""
ml/features.py
--------------
Feature engineering for the FX Signal Ranker.

LOGICAL FLOW:
─────────────
Unlike the stock project (two models: Volume Classifier + Signal
Ranker), FX has ONE model — the Signal Ranker. There is no volume
data to train a Volume Classifier on (FX is decentralized OTC — see
fetcher.py's docstring), so that whole model, its features, and its
labeller are dropped, not adapted. This is a structural simplification
of the project, not a missing feature.

WHAT'S DROPPED FROM THE STOCK PROJECT:
   - compute_volume_features() and all 10 volume features — no real
     volume data exists for FX (yfinance's FX "volume" is synthetic
     tick-count, not real traded volume — training on it would be
     learning from noise dressed up as signal)
   - build_volume_feature_matrix() — no Volume Classifier to train
   - build_sector_price_cache() / relative_strength_sector — currencies
     have no sectors
   - relative_strength vs. a single benchmark (SPY) — no single
     equity-market-style benchmark exists for currencies
   - market_slope_avg / market_ind_df (Market Pulse from SPY/QQQ/DIA)
     — no equivalent benchmark trio exists for FX
   - vol_clf_score, volume_signal_encoded — no Volume Classifier output
     to feed in

WHAT'S ADDED, REPLACING THE ABOVE:
   - csi_rs (CSI_base - CSI_quote) — replaces both relative_strength
     and relative_strength_sector. See engines/csi.py for the full
     Currency Strength Index computation. This is the primary
     relative-strength feature: "how is this pair's base doing
     broadly vs. how is its quote doing broadly."
   - csi_commodity_bloc — replaces Market Pulse's market-context role.
     A regime signal (is the AUD/NZD/CAD commodity bloc moving
     together right now), shared across all pairs at a given
     timestamp, not pair-specific.
   - atr_fast / atr_slow / atr_ratio — volatility-regime feature
     (ATR_14/ATR_100 ratio: >1 = expanding vol, <1 = compression).
     Fills the "activity/conviction" role volume played for stocks,
     and doubles as the normalization basis below.
   - ATR-normalization applied to EVERY distance-based feature
     (dist_to_mean, band_penetration — bos_strength was already
     ATR-normalized in the stock project and needed no change). This
     is a correctness requirement for FX, not polish — price LEVEL is
     an arbitrary quoting convention here (EURUSD ~1.08, USDJPY ~150,
     JPY pairs quote 2dp vs 4-5dp elsewhere), unlike stocks where
     price at least shares a common dollar unit.
   - session_tokyo / session_london / session_new_york / session_overlap
     — categorical session indicator, derived free from the candle's
     own UTC timestamp. Captures that FX behavior differs meaningfully
     by session (London-NY overlap = highest volatility, Asian session
     = often quieter/ranging).
   - hammer_at_extreme — Hammer/Shooting-Star-at-extreme interaction,
     computed by engines/candlestick.py. Direction-aware: only counts
     when the pattern agrees with the setup direction being evaluated
     (Hammer-at-low for a long, Shooting-Star-at-high for a short).
     Extreme is |sd_position| > 1, matching the scanner's own
     long/short entry band (not config.yaml's stale
     candlestick.extreme_sd_threshold — see engines/candlestick.py).

FEATURE ENGINEERING PRINCIPLES (carried over unchanged):
   1. All features are normalised where needed — ATR ratios instead of
      raw price distances, so features are comparable across pairs
      with very different price levels and pip conventions.
   2. Missing values are handled with documented fallbacks (0.0 for
      CSI/ADX when insufficient history — same reasoning as the stock
      project: we don't want to lose training examples for pairs still
      ramping up their aligned history).
   3. No lookahead bias — we only use data available at or before the
      signal datetime, never future data.
   4. Features are computed fresh for each prediction — the model sees
      exactly what was available at that candle close.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with fetcher.py, engines/csi.py, and labeller.py — every
   FX file uses 'pair' and 'datetime' throughout, not the stock
   project's 'ticker'/'date'.
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

ATR_FAST_PERIOD = ATR_CFG["fast_period"]    # 14
ATR_SLOW_PERIOD = ATR_CFG["slow_period"]    # 100

# NOTE: no EXTREME_SD_THRESHOLD read here — engines/candlestick.py owns
# its own threshold (1.0, matching the scanner's entry band; see that
# module's docstring for why config.yaml's candlestick.
# extreme_sd_threshold is stale and not used).

TOKYO_OPEN_UTC     = SESSION_CFG["tokyo_open_utc"]
TOKYO_CLOSE_UTC    = SESSION_CFG["tokyo_close_utc"]
LONDON_OPEN_UTC    = SESSION_CFG["london_open_utc"]
LONDON_CLOSE_UTC   = SESSION_CFG["london_close_utc"]
NEW_YORK_OPEN_UTC  = SESSION_CFG["new_york_open_utc"]
NEW_YORK_CLOSE_UTC = SESSION_CFG["new_york_close_utc"]


# =============================================================================
# ATR — shared volatility measure
# Used both as its own volatility-regime feature (atr_ratio) and as the
# normalization basis for every distance-based feature below. Computed
# once per call, not recomputed per feature.
# =============================================================================

def _compute_atr(px: pd.DataFrame, period: int) -> float:
    """
    Compute Average True Range over the last `period` candles.

    MATHS (standard ATR, simple rolling mean of True Range — matching
    the stock project's existing ATR14 calc in compute_signal_features,
    just parameterised by period so it can serve both the fast and
    slow windows):
    - True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    - ATR = mean(True Range) over the last `period` candles

    Args:
        px    : OHLC DataFrame, sorted datetime ascending, already
                filtered to <= signal datetime by the caller
        period: Number of trailing candles to average over

    Returns:
        ATR value, or a 1%-of-price fallback if fewer than 2 candles
        are available (mirrors the stock project's fallback behaviour)
    """
    recent = px.tail(period)

    if len(recent) >= 2:
        high_low   = recent["high"] - recent["low"]
        high_close = (recent["high"] - recent["close"].shift(1)).abs()
        low_close  = (recent["low"]  - recent["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
    else:
        atr = float(px["close"].iloc[-1]) * 0.01   # 1% fallback

    return float(atr)


# =============================================================================
# SESSION INDICATOR
# Derived purely from the candle's own UTC timestamp — no new data
# source needed, essentially free to compute.
# =============================================================================

def _compute_session_flags(dt: pd.Timestamp) -> dict:
    """
    Determine which FX trading session(s) a candle's timestamp falls
    within, from UTC hour ranges in config.

    Sessions can and do overlap (e.g. London-NY overlap, the highest-
    volatility window) — each session gets its own independent binary
    flag rather than a single mutually-exclusive category, so the
    model can see overlap periods as a distinct combination rather
    than forcing a single label onto genuinely overlapping regimes.

    Args:
        dt: UTC timestamp of the candle (session/datetime column)

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
# CANDLESTICK-AT-EXTREME
# Wired to the real engine (engines/candlestick.py) — Hammer/Shooting
# Star pattern detection, gated by |sd_position| > 1 (matching the
# scanner's own long/short entry band, NOT config.yaml's
# candlestick.extreme_sd_threshold, which is now stale — see
# engines/candlestick.py's module docstring for why), and made
# direction-aware: only counts when the pattern agrees with the setup
# direction currently being evaluated (Hammer-at-low for a long,
# Shooting-Star-at-high for a short).
# =============================================================================

from engines.candlestick import compute_candlestick_latest


def _compute_hammer_at_extreme_feature(
    pair       : str,
    px         : pd.DataFrame,
    signal_dt,
    sd_position: float,
    direction  : str,
) -> float:
    """
    Thin wrapper around engines.candlestick.compute_candlestick_latest
    for use inside compute_signal_features. Falls back to 0.0 (not a
    crash) if the engine returns None — same reasoning as ADX's
    fallback just below: we don't want to lose a training example
    over one engine's insufficient history.
    """
    result = compute_candlestick_latest(
        pair            = pair,
        df              = px,
        signal_datetime = signal_dt,
        sd_position     = sd_position,
        direction       = direction,
    )
    if result is None:
        return 0.0
    return float(result["hammer_at_extreme"])



# =============================================================================
# SIGNAL RANKER FEATURES
# The only feature set now — there is no Volume Classifier for FX.
# =============================================================================

def compute_signal_features(
    pair          : str,
    signal_datetime,
    direction     : str,
    prices_df     : pd.DataFrame,
    indicators_df : pd.DataFrame,
    csi_df        : Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Compute all features for the Signal Ranker for a single pair/datetime.

    FEATURE GROUPS:

    GROUP 1 — Price position (3 features, ATR-normalized):
    - sd_position       : How deep in the band (-1.8 = 1.8 SDs below mean)
    - dist_to_mean       : Distance from current price to LinReg mean,
                           normalised by ATR_fast (NOT by price — price
                           level is an arbitrary FX quoting convention,
                           see module docstring)
    - band_penetration   : How far through the band is price? (0 = at
                           ±1SD, 1 = at ±3SD) — unchanged in shape from
                           the stock project (already SD-relative, not
                           price-relative), kept here as its own group
                           for parity with the stock feature grouping

    GROUP 2 — Trend strength (2 features):
    - linreg_slope       : Steepness of LinReg slope (normalised by price
                           at fit time — unchanged, computed upstream by
                           engines/linreg.py)
    - days_in_trend      : How many consecutive 4H-candles has slope
                           been in the same direction?

    GROUP 3 — Volatility regime (3 features, replacing stock's Volume group):
    - atr_fast           : ATR over ATR_FAST_PERIOD (14) candles
    - atr_slow           : ATR over ATR_SLOW_PERIOD (100) candles
    - atr_ratio           : atr_fast / atr_slow — >1 = expanding vol
                           (often precedes/accompanies breakouts),
                           <1 = compression/squeeze

    GROUP 4 — Session (4 features):
    - session_tokyo, session_london, session_new_york : binary flags
      from the candle's own UTC timestamp
    - session_overlap   : 1 if 2+ sessions are simultaneously active
      (e.g. London-NY overlap, typically highest volatility)

    GROUP 5 — Candlestick-at-extreme (1 feature):
    - hammer_at_extreme : direction-aware Hammer/Shooting-Star-at-
      extreme interaction from engines/candlestick.py

    GROUP 6 — ADX / trend strength (3 features, unchanged from stock
    project — pure price-action, timeframe/asset-agnostic):
    - adx_value, plus_di, minus_di

    GROUP 7 — Currency Strength Index (2 features, replacing Market
    Pulse AND both relative_strength features):
    - csi_rs             : CSI_base - CSI_quote — this pair's relative
                           strength (e.g. EURUSD's csi_rs = CSI_EUR -
                           CSI_USD)
    - csi_commodity_bloc : mean CSI across AUD/NZD/CAD — shared regime
                           signal, same value for every pair at a
                           given timestamp

    Args:
        pair           : FX pair symbol e.g. 'EURUSD'
        signal_datetime: Timestamp of the signal (UTC)
        direction      : 'long' or 'short'
        prices_df      : Full OHLC data for this pair
        indicators_df  : Full indicator results (all pairs) — must
                         include linreg_value, linreg_slope,
                         linreg_slope_up, price_sd_position,
                         has_valid_zone for this pair/datetime
        csi_df         : Output of run_csi_engine() / compute_csi_series()
                         for this signal_datetime — pre-computed once
                         for the whole universe per call, not recomputed
                         per pair (CSI is inherently cross-pair, see
                         engines/csi.py). Falls back to 0.0 for both CSI
                         features if not provided or pair not found.

    Returns:
        Dict of all features or None if insufficient data
    """
    # ── Get indicator row for this pair on signal datetime ────────────────
    ind_row = indicators_df[
        (indicators_df["pair"]     == pair) &
        (indicators_df["datetime"] == signal_datetime)
    ]

    if ind_row.empty:
        return None

    ind = ind_row.iloc[0]

    # ── Get price data up to signal datetime ───────────────────────────────
    px = prices_df[
        (prices_df["pair"]     == pair) &
        (prices_df["datetime"] <= signal_datetime)
    ].sort_values("datetime")

    if len(px) < ATR_SLOW_PERIOD:
        # Need at least ATR_SLOW_PERIOD candles for the slow ATR to be
        # meaningful — mirrors the stock project's AVG_VOLUME_PERIOD
        # minimum-history gate, just using the (now larger) ATR window
        # since there's no volume period concept anymore.
        return None

    # ── GROUP 1: Price position features (ATR-normalized) ──────────────────
    sd_position   = float(ind["price_sd_position"])
    linreg_val    = float(ind["linreg_value"])
    current_close = px.iloc[-1]["close"]

    atr_fast = _compute_atr(px, ATR_FAST_PERIOD)
    atr_slow = _compute_atr(px, ATR_SLOW_PERIOD)
    atr_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0

    # Distance from price to LinReg, normalised by ATR_fast — NOT by
    # price level. Price level is an arbitrary FX quoting convention
    # (EURUSD ~1.08, USDJPY ~150) and dividing by it would make this
    # feature nearly meaningless across pairs — this is the correctness
    # fix flagged as critical in the project discussion, applied here.
    dist_to_mean = abs(current_close - linreg_val) / atr_fast if atr_fast > 0 else 0.0

    # How far through the band is price? Already SD-relative, not
    # price-relative — no change needed from the stock project's calc.
    abs_sd = abs(sd_position)
    band_penetration = np.clip((abs_sd - 1.0) / 2.0, 0.0, 1.0)

    # ── GROUP 2: Trend strength features ────────────────────────────────────
    linreg_slope = float(ind["linreg_slope"])

    pair_ind = indicators_df[
        indicators_df["pair"] == pair
    ].sort_values("datetime")

    current_slope = int(ind["linreg_slope_up"])

    days_in_trend = 0
    for val in reversed(pair_ind["linreg_slope_up"].tolist()):
        if val == current_slope:
            days_in_trend += 1
        else:
            break

    # ── GROUP 3: Volatility regime (replaces stock's volume group) ─────────
    # atr_fast/atr_slow already computed above (needed for GROUP 1's
    # normalization) — exposing them as their own trend features too is
    # nearly free, per project discussion.

    # ── BOS strength (candle body / ATR_fast) — unchanged from stock ───────
    # Already ATR-normalized in the stock project, no rework needed here.
    last_candle  = px.iloc[-1]
    bos_body     = abs(last_candle["close"] - last_candle["open"])
    bos_strength = bos_body / atr_fast if atr_fast > 0 else 0.0

    # ── GROUP 4: Session flags ──────────────────────────────────────────────
    session_flags = _compute_session_flags(pd.Timestamp(signal_datetime))

    # ── GROUP 5: Candlestick-at-extreme ─────────────────────────────────────
    hammer_at_extreme = _compute_hammer_at_extreme_feature(
        pair        = pair,
        px          = px,
        signal_dt   = signal_datetime,
        sd_position = sd_position,
        direction   = direction,
    )

    # ── GROUP 6: ADX (trend strength, direction-independent) ────────────────
    # Unchanged from stock project — pure price-action, timeframe-agnostic.
    # Falls back to 0.0 if insufficient history, same reasoning as before:
    # we don't want to lose training examples for pairs still ramping up
    # their history.
    adx_result = compute_adx_latest(pair, px, str(signal_datetime))
    if adx_result is not None:
        adx_value = adx_result["adx_value"]
        plus_di   = adx_result["plus_di"]
        minus_di  = adx_result["minus_di"]
    else:
        adx_value = 0.0
        plus_di   = 0.0
        minus_di  = 0.0

    # ── GROUP 7: Currency Strength Index ─────────────────────────────────────
    # csi_df is pre-computed once per call site for the whole universe at
    # this signal_datetime (CSI is inherently cross-pair — see
    # engines/csi.py) — this function just looks up this pair's row.
    csi_rs             = 0.0
    csi_commodity_bloc = 0.0
    if csi_df is not None and not csi_df.empty:
        csi_row = csi_df[csi_df["pair"] == pair]
        if not csi_row.empty:
            csi_rs_val = csi_row.iloc[0]["csi_rs"]
            bloc_val   = csi_row.iloc[0]["csi_commodity_bloc"]
            csi_rs             = float(csi_rs_val) if pd.notna(csi_rs_val) else 0.0
            csi_commodity_bloc = float(bloc_val) if pd.notna(bloc_val) else 0.0

    # ── Direction encoding ────────────────────────────────────────────────────
    # 1 = long setup, 0 = short setup
    direction_flag = 1 if direction == "long" else 0

    # ── Zone flag — unchanged, SMC is unaffected by the FX rework ───────────
    has_valid_zone = int(ind.get("has_valid_zone", 0))

    # ── Assemble all features ─────────────────────────────────────────────────
    features = {
        # Identifiers (not used in training — dropped before fit)
        "pair"               : pair,
        "datetime"           : signal_datetime,
        "direction"          : direction,

        # Group 1: Price position (ATR-normalized)
        "sd_position"        : round(sd_position,      4),
        "dist_to_mean"       : round(dist_to_mean,      6),
        "band_penetration"   : round(band_penetration,  4),

        # Group 2: Trend strength
        "linreg_slope"       : round(linreg_slope,      6),
        "days_in_trend"      : days_in_trend,
        "bos_strength_vs_atr": round(bos_strength,      4),

        # Group 3: Volatility regime
        "atr_fast"           : round(atr_fast,           6),
        "atr_slow"           : round(atr_slow,           6),
        "atr_ratio"          : round(atr_ratio,           4),

        # Group 4: Session
        **session_flags,

        # Group 5: Candlestick-at-extreme
        "hammer_at_extreme"  : hammer_at_extreme,

        # Group 6: ADX
        "adx_value"          : adx_value,
        "plus_di"            : plus_di,
        "minus_di"           : minus_di,

        # Group 7: Currency Strength Index
        "csi_rs"             : round(csi_rs,             6),
        "csi_commodity_bloc" : round(csi_commodity_bloc, 6),

        # Direction
        "direction_flag"     : direction_flag,
        # Zone flag
        "has_valid_zone"     : has_valid_zone,
    }

    return features


# =============================================================================
# FEATURE MATRIX BUILDER
# Builds the full feature matrix for Signal Ranker training.
# =============================================================================

def build_signal_feature_matrix(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    labels_df    : pd.DataFrame,
    csi_series_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for Signal Ranker training.

    Unlike the stock project's build_signal_feature_matrix, there is
    no vol_scores parameter (no Volume Classifier), no benchmark/sector
    price lookups (no market pulse, no sectors) — replaced by a single
    pre-computed CSI series covering the whole training window.

    Args:
        prices_df     : Full OHLC data [pair, datetime, open, high,
                        low, close]
        indicators_df : Full indicator results (all pairs)
        labels_df     : Output of labeller.label_scanner_hits()
                        [pair, datetime, direction, label]
        csi_series_df : Output of engines.csi.compute_csi_series() —
                        full historical CSI/RS series for all pairs.
                        If None, csi_rs and csi_commodity_bloc will be
                        0.0 for every training example (logged as a
                        warning, since this would silently weaken the
                        model rather than crash it).

    Returns:
        DataFrame with one row per labelled example, ready for model
        training (label column included, pair/datetime/direction kept
        as identifiers to drop before fit)
    """
    logger.info(
        f"Building Signal Ranker feature matrix | "
        f"{len(labels_df)} labelled examples"
    )

    if csi_series_df is None or csi_series_df.empty:
        logger.warning(
            "build_signal_feature_matrix: no csi_series_df provided — "
            "csi_rs and csi_commodity_bloc will be 0.0 for all training examples"
        )

    rows   = []
    failed = 0

    # Pre-group prices once, per pair
    prices_dict = {p: df.sort_values("datetime") for p, df in prices_df.groupby("pair")}

    for row in labels_df.itertuples(index=False):
        pair      = row.pair
        dt        = row.datetime
        direction = row.direction
        label     = row.label

        px = prices_dict.get(pair)

        if px is None or px.empty:
            failed += 1
            continue

        # Slice this signal datetime's CSI snapshot from the pre-computed
        # full series (CSI is cross-pair, so we don't recompute it here —
        # same reasoning as compute_signal_features' csi_df parameter)
        csi_snapshot = None
        if csi_series_df is not None and not csi_series_df.empty:
            csi_snapshot = csi_series_df[csi_series_df["datetime"] == dt]

        features = compute_signal_features(
            pair            = pair,
            signal_datetime = dt,
            direction       = direction,
            prices_df       = px,
            indicators_df   = indicators_df,
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
        f"Signal feature matrix built | "
        f"Rows: {len(result)} | "
        f"Failed: {failed}"
    )

    return result


# =============================================================================
# FEATURE COLUMNS — used at both train and inference time
# =============================================================================

SIGNAL_FEATURE_COLS = [
    # Price position (ATR-normalized)
    "sd_position",
    "dist_to_mean",
    "band_penetration",
    # Trend
    "linreg_slope",
    "days_in_trend",
    "bos_strength_vs_atr",       # BOS candle body / ATR_fast
    # Volatility regime (replaces stock's volume group)
    "atr_fast",
    "atr_slow",
    "atr_ratio",
    # Session
    "session_tokyo",
    "session_london",
    "session_new_york",
    "session_overlap",
    # Candlestick-at-extreme
    "hammer_at_extreme",
    # ADX — trend strength, direction-independent
    "adx_value",
    "plus_di",
    "minus_di",
    # Currency Strength Index (replaces Market Pulse + both RS features)
    "csi_rs",
    "csi_commodity_bloc",
    # Direction
    "direction_flag",
    "has_valid_zone",            # 1 = valid demand/supply zone exists
]
