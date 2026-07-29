"""
engines/macro.py
-----------------
Macro driver engine for the FX Scanner pipeline.

WHY THIS EXISTS:
   The EURUSD-only reference project's strongest features weren't
   price-derived at all — DXY, the US 10Y yield, and gold gave the
   model genuine EXTERNAL signal (real fundamental drivers), distinct
   from anything derivable by re-slicing a pair's own OHLCV. The
   original basket redesign had no equivalent: CSI (engines/csi.py)
   is cross-pair, but it's still entirely internal to the 28-pair
   universe's own price action. This engine adds that missing
   external layer back in, mapped per basket to each quote currency's
   real driver rather than reused generically everywhere:

     usd basket     -> DXY, gold, US 10Y yield   (matches EURUSD ref)
     cad basket     -> WTI crude oil             (Canada's #1 export)
     chf/jpy basket -> VIX                       (safe-haven / carry-
                                                    unwind flows — no
                                                    free, reliable
                                                    CHF/JPY-specific
                                                    yield series exists
                                                    on yfinance, and
                                                    both currencies
                                                    trade far more on
                                                    risk sentiment than
                                                    domestic yield)
     crosses basket -> VIX                       (general risk-
                                                    sentiment proxy;
                                                    this basket has no
                                                    single shared quote
                                                    currency)

   See config.yaml's `macro` section for the exact ticker/basket
   mapping — this engine is generic over whichever driver symbol is
   passed in, so one set of functions serves all five drivers rather
   than one-off code per symbol.

LOGICAL FLOW:
─────────────
STEP 1 — Per-driver feature computation (generic, symbol-agnostic):
   For a single macro driver's own OHLC history:
     macro_return       : 1-candle % return (close/close.shift(1) - 1)
     macro_return_4wk   : 4-weekly-candle % return — a slower-moving
                          trend read, roughly comparable in spirit to
                          the reference project's 5d/10d ROC pairing
     macro_above_sma20  : 1 if close > its own 20-period SMA else 0 —
                          "is this driver in an uptrend right now"
     macro_rsi          : driver's own 14-period RSI — is the driver
                          itself overbought/oversold (e.g. is DXY
                          already stretched, making further USD
                          strength less likely on mean-reversion
                          grounds)
   All four are computed directly from the driver's own close series
   — no cross-referencing the FX pair being scored, which keeps this
   engine fully independent of and reusable across every basket.

STEP 2 — Per-basket assembly:
   config.yaml's `macro.basket_drivers` maps each basket name to the
   list of driver symbols it should receive (e.g. "usd" -> ["DXY",
   "GOLD", "US10Y"]). get_macro_features_for_basket() looks up that
   list and returns one row per timestamp with columns prefixed by
   driver name (dxy_return, dxy_above_sma20, ..., gold_return, ...),
   so features.py can join them onto a basket's feature matrix by
   datetime without any per-basket special-casing.

OUTPUT (per driver, per timestamp):
   {symbol}_return, {symbol}_return_4wk, {symbol}_above_sma20,
   {symbol}_rsi — all lowercased driver symbol as prefix, e.g.
   "dxy_return", "vix_rsi".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_macro_logger
from utils.error_handler import graceful, EngineError

logger = get_macro_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config      = _load_config()
MACRO_CFG   = config["macro"]

DRIVERS         = MACRO_CFG["drivers"]         # {"DXY": {...}, "GOLD": {...}, ...}
BASKET_DRIVERS  = MACRO_CFG["basket_drivers"]  # {"usd": ["DXY", "GOLD", "US10Y"], ...}

SMA_PERIOD      = 20   # matches trend.sma_fast used elsewhere in features.py
RSI_PERIOD      = 14   # matches rsi.period used elsewhere in features.py
RETURN_4WK_LAG  = 4    # weekly candles — slower-moving companion to the 1-candle return


# =============================================================================
# STEP 1 — PER-DRIVER FEATURE COMPUTATION (generic, symbol-agnostic)
# =============================================================================

def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Standard Wilder RSI on a single close-price series. Symbol-agnostic
    — used identically whether the input is a driver's close series
    or (elsewhere in features.py) an FX pair's own close series.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is exactly 0 (pure uptrend run), RSI is 100 by
    # definition rather than NaN from the divide-by-zero guard above.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_driver_features(driver_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full historical feature series for a single macro
    driver, from its own OHLC history.

    Args:
        driver_df: DataFrame with 'datetime' and 'close' columns,
                   sorted datetime ascending (matches fetcher.py's
                   per-symbol output shape — same shape as an FX
                   pair's DataFrame, just keyed by a macro symbol
                   name instead of a pair name).

    Returns:
        DataFrame indexed by datetime with columns: return, return_4wk,
        above_sma20, rsi. NOT yet prefixed with the driver symbol —
        get_macro_features_for_basket() does that when assembling
        multiple drivers together.
    """
    if driver_df.empty or "close" not in driver_df.columns:
        logger.warning("compute_driver_features: empty or malformed driver_df")
        return pd.DataFrame()

    df = driver_df.sort_values("datetime").set_index("datetime")
    close = df["close"]

    out = pd.DataFrame(index=df.index)
    out["return"]      = close.pct_change(1)
    out["return_4wk"]  = close.pct_change(RETURN_4WK_LAG)
    sma20              = close.rolling(SMA_PERIOD).mean()
    out["above_sma20"] = (close > sma20).astype(float)
    out["rsi"]         = _compute_rsi(close, RSI_PERIOD)

    return out


# =============================================================================
# STEP 2 — PER-BASKET ASSEMBLY
# =============================================================================

def get_macro_features_for_basket(
    basket_name: str,
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Assemble the full historical macro feature set for one basket, by
    looking up which driver(s) that basket uses (config.yaml's
    macro.basket_drivers) and joining their per-driver features
    together on datetime.

    Args:
        basket_name: e.g. "usd", "cad", "chf", "jpy", "crosses"
        macro_data : Dict mapping driver storage_symbol -> that
                     driver's OHLC DataFrame (e.g. "DXY" -> df,
                     "GOLD" -> df, "VIX" -> df, ...) — same shape
                     fetcher.py's to_pair_dict() produces, just for
                     macro symbols instead of FX pairs.

    Returns:
        DataFrame indexed by datetime with one column group per
        driver, prefixed by lowercased driver symbol, e.g. for the
        "usd" basket: dxy_return, dxy_return_4wk, dxy_above_sma20,
        dxy_rsi, gold_return, ..., us10y_rsi. Empty DataFrame if the
        basket has no configured drivers or none of its drivers'
        data is available yet.
    """
    driver_symbols = BASKET_DRIVERS.get(basket_name, [])
    if not driver_symbols:
        logger.warning(f"get_macro_features_for_basket: no drivers configured for basket '{basket_name}'")
        return pd.DataFrame()

    driver_feature_frames = []

    for driver_symbol in driver_symbols:
        driver_cfg      = DRIVERS.get(driver_symbol)
        if driver_cfg is None:
            logger.warning(f"'{driver_symbol}' not found in macro.drivers config, skipping")
            continue

        storage_symbol  = driver_cfg["storage_symbol"]
        driver_df       = macro_data.get(storage_symbol)

        if driver_df is None or driver_df.empty:
            logger.warning(
                f"Basket '{basket_name}': no data available yet for driver "
                f"'{storage_symbol}', skipping (basket will be missing this "
                f"driver's features until its history is fetched)"
            )
            continue

        features = compute_driver_features(driver_df)
        if features.empty:
            continue

        prefix = storage_symbol.lower()
        features = features.add_prefix(f"{prefix}_")
        driver_feature_frames.append(features)

    if not driver_feature_frames:
        logger.warning(f"Basket '{basket_name}': no macro features available from any configured driver")
        return pd.DataFrame()

    # Outer-join on datetime, NOT inner-join like CSI's cross-pair
    # alignment — macro drivers are independent external series, not
    # a set that all need to align for one shared computation, so a
    # gap in one driver shouldn't truncate the others. Downstream
    # feature-matrix building (features.py) already drops rows with
    # any NaN feature, so this doesn't risk leaking incomplete rows
    # into training.
    combined = driver_feature_frames[0]
    for frame in driver_feature_frames[1:]:
        combined = combined.join(frame, how="outer")

    logger.info(
        f"Basket '{basket_name}' macro features assembled | "
        f"Drivers: {[f.columns[0].split('_')[0] for f in driver_feature_frames]} | "
        f"Rows: {len(combined)}"
    )

    return combined
