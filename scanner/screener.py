"""
scanner/screener.py
-------------------
Scanner for the FX Signal Ranker pipeline.

LOGICAL FLOW:
─────────────
WHY THIS IS NOT A WATERFALL (unlike the stock project):
The stock scanner used a 3-level waterfall: Market Health (SPY/QQQ/DIA)
-> Sector Health (11 sector ETFs) -> Stock Setup. Neither Level 1 nor
Level 2 has an FX equivalent — there is no single equity-market-style
benchmark for currencies, and currencies have no sectors. The FX
project's actual replacement for "is this currency's broader context
healthy" is CSI (Currency Strength Index, engines/csi.py) — but CSI is
a FEATURE the ML model learns to weight, not a hard gate a candidate
must clear before being surfaced (see ml/train_models.py's module
docstring for the full reasoning on why CSI was deliberately NOT
promoted to a hard gate, matching how has_valid_zone and CHoCH were
already demoted from gates to features in the stock project's own
design). So there is only ONE level here:

LEVEL 1 — PAIR SETUP CHECK (the only level):
   For LONG candidates:
   - LinReg sloping UP
   - Price SD position between -1 and -3 (in the buy zone)

   For SHORT candidates:
   - LinReg sloping DOWN
   - Price SD position between +1 and +3 (in the sell zone)

   THIS MUST MATCH ml/train_models.py's _is_long_candidate_relaxed /
   _is_short_candidate_relaxed EXACTLY — those two conditions are what
   the model was trained to recognize as "a candidate" in the first
   place. If this file's gate drifts from that file's gate, the model
   gets scored on candidates defined differently than the ones it was
   trained on. This is not a style preference; it's the same
   train/serve consistency contract the rest of this pipeline enforces
   (e.g. features.py used identically by both files).

FINAL OUTPUT:
   A ranked list of long and short candidates with all their
   indicator values attached, plus a currency_bloc display label. ML
   scoring is applied separately (ml/signal_ranker.py's score_candidates)
   — here we just filter and tag. Candidates get ml_score = 0.0 and
   ml_rank = 0 as placeholders until that step runs.

WHAT'S DROPPED FROM THE STOCK PROJECT:
   - _check_market_health / _check_sector_health — no market or sector
     concept exists for FX; CSI (a feature, not a gate) replaces both
     roles. See ml/train_models.py's module docstring for the decision.
   - _load_sector_lookup / SECTOR_ETF_NAMES / read_sector_metadata —
     no sectors, no sector metadata table
   - get_market_sector_status — a dashboard export for the above two
     levels; nothing to export once neither level exists
   - the `excluded` set built from config["universe"]["indices"] +
     config["universe"]["sectors"] — these config keys don't exist for
     FX (fixed 28-pair universe, no indices/sectors mixed into it —
     every row in indicator_df is a real, scannable pair already)
   - volume_signal field on the candidate row, and all previously-
     commented-out CHoCH/has_valid_zone hard-gate conditions — already
     established as FX features-not-gates, consistent with CSI's
     treatment here
   - "sector" display field — replaced with currency_bloc (see below)

WHAT'S ADDED:
   - currency_bloc — a display-only label derived from the pair's
     base/quote currencies (e.g. "EUR/USD", tagged additionally as
     "Commodity" if either side is AUD/NZD/CAD — the same bloc
     engines/csi.py's csi_commodity_bloc feature covers). Purely for
     dashboard grouping/display; carries NO gating logic and does not
     affect which candidates pass the scan.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other FX file — 'pair' throughout, not
   'ticker'. This file doesn't use 'datetime' directly (indicator_df
   passed in already represents one point in time per scan run), but
   is written to be trivially extended if that changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_scanner_logger
from utils.error_handler import ScannerError, handle_critical_error

logger = get_scanner_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config      = _load_config()
SCANNER_CFG = config["scanner"]

# Entry zone boundaries — SAME config keys ml/train_models.py's relaxed
# candidate gate reads, by design (train/serve consistency, see module
# docstring above).
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]    # -1 (upper boundary)
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]    # -3 (lower boundary)
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]   # +1 (lower boundary)
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]   # +3 (upper boundary)

# Currencies making up the "Commodity" bloc tag — SAME grouping
# engines/csi.py's csi_commodity_bloc feature uses, kept consistent so
# the dashboard label and the model's actual regime feature refer to
# the same bloc definition.
COMMODITY_BLOC_CURRENCIES = {"AUD", "NZD", "CAD"}


def _split_pair(pair: str) -> tuple[str, str]:
    """
    Split a 6-character FX pair string into (base, quote).
    e.g. 'EURUSD' -> ('EUR', 'USD'). Same pure string-slicing approach
    as engines/csi.py's _split_pair — every pair in the fixed universe
    is exactly 6 characters, no separators needed.
    """
    return pair[:3], pair[3:]


def _currency_bloc_label(pair: str) -> str:
    """
    Build a display-only currency-bloc label for a pair, for dashboard
    grouping. NOT used for any gating/filtering decision — purely
    cosmetic, per the project decision to include this in place of the
    stock scanner's "sector" field with no replacement gating logic.

    Args:
        pair: FX pair symbol, e.g. 'EURUSD'

    Returns:
        A label like "EUR/USD", with " (Commodity)" appended if either
        the base or quote currency is part of the AUD/NZD/CAD bloc
        (matching engines/csi.py's csi_commodity_bloc grouping).
    """
    base, quote = _split_pair(pair)
    label = f"{base}/{quote}"

    if base in COMMODITY_BLOC_CURRENCIES or quote in COMMODITY_BLOC_CURRENCIES:
        label += " (Commodity)"

    return label


# =============================================================================
# LEVEL 1 — PAIR SETUP CHECK (the only level — see module docstring)
# =============================================================================

def _is_long_candidate(row: pd.Series) -> bool:
    """
    Check if a pair qualifies as a LONG candidate.

    MUST MATCH ml/train_models.py's _is_long_candidate_relaxed exactly
    — see module docstring's train/serve consistency note.

    Requires: slope up + price in -1 to -3 SD zone. Does NOT require
    CSI direction agreement, has_valid_zone, or any candlestick
    pattern — those are ML features the model weighs, not filters
    that block a row from being scanned.
    """
    if int(row["linreg_slope_up"]) != 1:
        return False

    sd_pos = float(row["price_sd_position"])
    if not (LONG_SD_MAX <= sd_pos <= LONG_SD_MIN):   # -3 <= sd <= -1
        return False

    return True


def _is_short_candidate(row: pd.Series) -> bool:
    """
    Check if a pair qualifies as a SHORT candidate.
    Mirror of _is_long_candidate with reversed conditions. MUST MATCH
    ml/train_models.py's _is_short_candidate_relaxed exactly.
    """
    if int(row["linreg_slope_up"]) != 0:
        return False

    sd_pos = float(row["price_sd_position"])
    if not (SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX):   # +1 <= sd <= +3
        return False

    return True


# =============================================================================
# CANDIDATE BUILDER
# Assembles the final candidate rows with all data attached
# =============================================================================

def _build_candidate_row(row: pd.Series, direction: str) -> dict:
    """
    Build a complete candidate row.

    Args:
        row      : One row from indicator_df (a single pair's latest
                   indicator values)
        direction: 'long' or 'short'

    Returns:
        Dict representing one scanner candidate
    """
    pair = row["pair"]

    return {
        "pair"            : pair,
        "direction"       : direction,
        "currency_bloc"   : _currency_bloc_label(pair),
        "sd_position"     : float(row["price_sd_position"]),
        "has_valid_zone"  : int(row.get("has_valid_zone", 0)),
        "ml_score"        : 0.0,
        "ml_rank"         : 0,
    }


# =============================================================================
# MAIN SCANNER
# =============================================================================

def run_scanner(
    indicator_df : pd.DataFrame,
    datetime_str : str,
) -> pd.DataFrame:
    """
    Run the FX scanner across the fixed 28-pair universe.

    Unlike the stock project, there is no market-bias gate deciding
    whether to scan longs/shorts at all — every pair is checked for
    both directions independently every run (CSI's regime-context role
    is a feature the model sees per-candidate, not a global switch that
    turns long or short scanning off entirely).

    Args:
        indicator_df: Latest indicator results for all 28 pairs
                      [pair, linreg_slope_up, price_sd_position,
                      has_valid_zone, ...]
        datetime_str: Timestamp string of this scan run (for logging)

    Returns:
        DataFrame of ranked candidates, both directions, with
        currency_bloc display labels and ml_score/ml_rank placeholders
        (0.0 / 0) pending ml/signal_ranker.py's score_candidates()
    """
    logger.info("=" * 60)
    logger.info("SCANNER STARTING")
    logger.info(f"Datetime: {datetime_str}")
    logger.info("=" * 60)

    if indicator_df.empty:
        logger.warning("Scanner: indicator_df is empty, no candidates possible")
        return pd.DataFrame()

    long_candidates  = []
    short_candidates = []

    logger.info(f"Scanning {len(indicator_df)} pairs...")

    for _, row in indicator_df.iterrows():
        pair = row["pair"]

        if _is_long_candidate(row):
            long_candidates.append(_build_candidate_row(row, "long"))

        if _is_short_candidate(row):
            short_candidates.append(_build_candidate_row(row, "short"))

    logger.info(
        f"Scanner complete | "
        f"Long: {len(long_candidates)} | "
        f"Short: {len(short_candidates)}"
    )

    all_candidates = long_candidates + short_candidates

    if not all_candidates:
        logger.warning("Scanner: No candidates found this run")
        return pd.DataFrame()

    df = pd.DataFrame(all_candidates)

    # Preliminary ranking by SD position depth — real ranking happens
    # in ml/signal_ranker.py's score_candidates() once ML scores exist.
    # Kept here only so the output is meaningfully ordered even before
    # ML scoring runs (e.g. if the model isn't trained yet).
    longs  = df[df["direction"] == "long"].sort_values("sd_position", ascending=False)
    shorts = df[df["direction"] == "short"].sort_values("sd_position", ascending=True)

    longs  = longs.reset_index(drop=True)
    shorts = shorts.reset_index(drop=True)
    longs["ml_rank"]  = longs.index + 1
    shorts["ml_rank"] = shorts.index + 1

    result = pd.concat([longs, shorts], ignore_index=True)

    logger.info(f"Final candidates | Total: {len(result)}")
    return result
