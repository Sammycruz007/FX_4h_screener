"""
engines/linreg.py
------------------
Single-pair 5-minute LinReg slope confirmation — the execution-filter
gate for the FX directional pipeline.

LOGICAL FLOW:
─────────────
This is NOT the multi-leg (stock + sector ETF + market indices) design
from the stock-screener reference project this was adapted from. This
project's execution filter is single-pair only: when Stage 1's daily
model emits a BUY or SELL for a pair, this engine fits a LinReg channel
on that SAME pair's fresh 5-minute intraday closes and checks whether
the slope agrees with the model's direction.

    Model says BUY  -> require an UP slope on the 5-min LinReg
    Model says SELL -> require a DOWN slope on the 5-min LinReg
    Disagreement (or missing/insufficient data) -> the signal is
    dropped entirely — never written to Supabase, never shown on the
    dashboard (see run_pipeline_cloud.py, which calls this right after
    predictions are generated and before anything is persisted).

DATA WINDOW:
   A full month of 5-minute bars (execution_filter.fetch_days) is
   fetched fresh into memory for the pair being checked — never
   persisted, this is a point-in-time confirmation check, not a stored
   feature. Of that month's worth of bars, only the most recent
   execution_filter.linreg_period (1600) closes are actually used to
   fit the regression line — the rest is headroom so the fit always
   has a full window even accounting for weekend/holiday gaps in FX
   trading hours, not a lookback the regression itself consumes.

STEP 1 — Fit the Linear Regression line:
   Using the last linreg_period (1600) 5-minute closes, fit a straight
   line through the data using least squares regression (scipy). The
   slope of this line is the short-term trend direction.

STEP 2 — Normalise the slope:
   Raw slope is in price units per candle — not meaningfully
   comparable across pairs at very different price scales (e.g.
   EURUSD ~1.08 vs USDJPY ~150). Dividing by the current close gives a
   percentage slope. This engine only ever evaluates one pair at a
   time (no cross-pair comparison happens here), but normalising is
   kept anyway for consistency with how slope is reported elsewhere in
   this project, and because it costs nothing.

STEP 3 — Determine slope direction:
   Positive slope -> up. Negative slope -> down. That's the entire
   gate — no SD-band / entry-zone logic. The stock-screener reference
   project's "where does price sit relative to the bands" pullback
   check doesn't apply here; this project's rule is simple direction
   agreement, nothing more.

STEP 4 — check_direction_agreement():
   Given the model's predicted direction ("BUY" or "SELL") and this
   engine's LinReg result for that same pair, returns whether the
   5-min slope agrees — this project's actual pass/fail verdict.

NOT INCLUDED HERE (dropped vs. both reference projects it was adapted
from):
  - Sector ETF / market-index legs — this gate is single-pair only,
    there is no FX equivalent of "sector" or "market index" here.
  - SD bands / price_sd_position — no entry-zone concept in this
    project's rule, only slope-direction agreement.
  - compute_linreg_series() / run_linreg_engine() batch runner — this
    engine is called once per Stage-1-qualifying pair (never a full-
    universe batch) and has no dashboard-charting requirement.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_linreg_logger
from utils.error_handler import graceful, EngineError

logger = get_linreg_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config   = _load_config()
EXEC_CFG = config["execution_filter"]

PERIOD = EXEC_CFG["linreg_period"]   # 1600 x 5-min bars used to fit the line

if PERIOD < 3:
    raise EngineError(
        f"execution_filter.linreg_period={PERIOD} is invalid — need at "
        f"least 3 candles for a meaningful regression (the current "
        f"design uses 1600)."
    )


# =============================================================================
# CORE LINREG CALCULATION
# =============================================================================

def _compute_linreg(closes: np.ndarray) -> dict:
    """
    Fit a linear regression line through an array of 5-minute closing
    prices and determine its slope direction.

    MATHS:
    - x = [0, 1, ..., N-1] — candle position (not timestamp — every
      candle is treated as equally spaced, which is what makes slope
      meaningful as a single number at all)
    - y = closing prices
    - Fit: y = slope * x + intercept (least squares)

    Args:
        closes: numpy array of closing prices, oldest first, length PERIOD

    Returns:
        Dict with linreg_value, linreg_slope (normalised), and
        linreg_slope_up (bool) — all for the MOST RECENT candle
    """
    n = len(closes)
    x = np.arange(n)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes)

    y_fitted       = slope * x + intercept
    current_linreg = y_fitted[-1]
    current_close  = closes[-1]

    # Normalise slope by price level — see module docstring's STEP 2.
    normalised_slope = slope / current_close if current_close > 0 else slope

    return {
        "linreg_value"   : round(float(current_linreg), 6),
        "linreg_slope"   : round(float(normalised_slope), 8),
        "linreg_slope_up": bool(slope > 0),
    }


# =============================================================================
# PER-PAIR ENTRY POINT
# Called once per Stage-1-qualifying pair by run_pipeline_cloud.py, right
# after a BUY/SELL prediction is generated and before anything is written
# to Supabase. Wrapped with @graceful so a bad/missing intraday fetch for
# one pair never crashes the rest of the run.
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_linreg_latest(
    pair : str,
    df   : pd.DataFrame,
) -> Optional[dict]:
    """
    Compute the 5-minute LinReg slope for the most recent candle of a
    single FX pair.

    Args:
        pair : Pair symbol e.g. 'EURUSD'
        df   : 5-minute OHLC DataFrame for this pair, sorted datetime
               ascending, fetched fresh and in-memory — a month's
               worth of bars, comfortably more than PERIOD (see module
               docstring's DATA WINDOW section). Never persisted.

    Returns:
        Dict with pair / linreg_value / linreg_slope / linreg_slope_up
        for this pair, or None on failure / insufficient data
        (@graceful handles exceptions; the explicit length check below
        handles "ran but not enough rows")
    """
    if len(df) < PERIOD:
        logger.warning(f"{pair} | Only {len(df)} 5-min rows — need {PERIOD}+ for LinReg")
        return None

    closes = df["close"].values[-PERIOD:]

    result = {
        "pair": pair,
        **_compute_linreg(closes),
    }

    logger.debug(
        f"{pair} | LinReg: {result['linreg_value']} | "
        f"Slope: {'UP' if result['linreg_slope_up'] else 'DOWN'}"
    )

    return result


# =============================================================================
# DIRECTION AGREEMENT CHECK — the actual pass/fail gate
# =============================================================================

def check_direction_agreement(
    model_direction : str,
    linreg_result   : Optional[dict],
) -> dict:
    """
    Given Stage 1's predicted direction for a pair and this engine's
    5-min LinReg result for that SAME pair, determine whether the
    signal is confirmed.

    RULE (see conversation — deliberate, project-specific, and
    deliberately simpler than the stock-screener reference project's
    multi-leg + SD-zone design):
    - model_direction == "BUY"  requires linreg_slope_up == True
    - model_direction == "SELL" requires linreg_slope_up == False
    - A missing/None linreg_result (intraday fetch failed, or
      insufficient 5-min history) fails the candidate outright.

    Args:
        model_direction: "BUY" or "SELL" — Stage 1's call for this pair
        linreg_result  : compute_linreg_latest() output for this SAME
                         pair, or None

    Returns:
        Dict with:
          - "passed"          : bool
          - "reason"          : short string explaining a fail (for
                                 logging/audit), None if passed
          - "linreg_slope_up" : bool or None
    """
    if model_direction not in ("BUY", "SELL"):
        return {
            "passed"          : False,
            "reason"          : f"unrecognised model_direction={model_direction!r}",
            "linreg_slope_up" : None,
        }

    if linreg_result is None:
        return {
            "passed"          : False,
            "reason"          : (
                "missing LinReg result (5-min fetch failed or "
                "insufficient intraday history)"
            ),
            "linreg_slope_up" : None,
        }

    slope_up = linreg_result["linreg_slope_up"]

    if model_direction == "BUY" and slope_up:
        return {"passed": True, "reason": None, "linreg_slope_up": slope_up}

    if model_direction == "SELL" and not slope_up:
        return {"passed": True, "reason": None, "linreg_slope_up": slope_up}

    return {
        "passed"          : False,
        "reason"          : (
            f"model_direction={model_direction} but 5-min LinReg slope "
            f"is {'UP' if slope_up else 'DOWN'} — disagreement"
        ),
        "linreg_slope_up" : slope_up,
    }
