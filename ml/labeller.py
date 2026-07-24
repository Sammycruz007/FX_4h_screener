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

    def _normalise_datetime_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Coerce a DataFrame's 'datetime' column to timezone-naive pandas
        Timestamps (UTC-interpreted, then tz stripped), defensively.

        WHY THIS EXISTS, AND WHY IT RUNS BEFORE THE MERGE (not after):
        this function was originally applied only to hits_with_linreg
        AFTER the merge below. That's a real gap — if scan_hits_df's
        and indicators_df's 'datetime' columns have ANY subtle
        inconsistency between them BEFORE the merge (different string
        precision, one already a Timestamp and the other a string,
        etc.), pd.merge's exact-match join can silently produce zero
        or wrong matches — a merge that already went wrong can't be
        fixed by normalising its output afterward. Normalising each
        input independently, before the merge, closes off that whole
        class of failure, not just the searchsorted-level symptom of
        it.

        GUARD AGAINST SILENT MISINTERPRETATION: pd.to_datetime on a
        raw numeric (int/float) column does NOT raise an error — it
        silently interprets the numbers as Unix EPOCH SECONDS by
        default, producing a wildly wrong date (e.g. year 1970) rather
        than failing loudly. If a 'datetime' column ever arrives as
        raw nanosecond-since-epoch integers (a real risk after a
        Parquet round-trip through Supabase Storage, depending on
        pyarrow's schema handling), this would silently corrupt every
        label rather than crash — a much worse failure mode than an
        exception. We explicitly detect and reject raw numeric input
        here instead of letting pd.to_datetime guess.

        Args:
            df   : DataFrame with a 'datetime' column to normalise IN PLACE
            label: Name for this DataFrame, used only in the error
                   message if the numeric-input guard fires

        Returns:
            The same DataFrame, with 'datetime' coerced to tz-naive
            Timestamps
        """
        col = df["datetime"]

        if pd.api.types.is_numeric_dtype(col):
            raise MLError(
                f"{label}['datetime'] is numeric (dtype={col.dtype}) — "
                f"refusing to guess whether these are epoch seconds, "
                f"milliseconds, or nanoseconds. pd.to_datetime's default "
                f"epoch-seconds assumption would silently produce wrong "
                f"dates rather than fail loudly. This usually means a "
                f"datetime column lost its proper dtype somewhere "
                f"upstream (e.g. a Parquet round-trip) — fix the source, "
                f"don't guess the unit here."
            )

        df = df.copy()
        df["datetime"] = pd.to_datetime(col, utc=True).dt.tz_localize(None)
        return df

    scan_hits_df  = _normalise_datetime_column(scan_hits_df, "scan_hits_df")
    indicators_df = _normalise_datetime_column(indicators_df, "indicators_df")

    # 1. BULK LOOKUP: merge indicators to get the fixed LinReg value in one shot
    # Both sides are now guaranteed-normalised BEFORE this merge — closing
    # off the risk of a silent zero/partial-match merge from any type
    # inconsistency between the two inputs (see _normalise_datetime_column's
    # docstring for why this matters more than fixing it after the fact).
    hits_with_linreg = pd.merge(
        scan_hits_df,
        indicators_df[["pair", "datetime", "linreg_value"]],
        on=["pair", "datetime"],
        how="inner",
    )

    if hits_with_linreg.empty:
        return pd.DataFrame()

    # 2. PRE-PROCESS PRICES: group by pair into fast NumPy arrays
    logger.info("Pre-processing price data for fast lookup...")
    price_dict = {}

    prices_df = _normalise_datetime_column(prices_df, "prices_df")
    sorted_prices = prices_df.sort_values(["pair", "datetime"]).reset_index(drop=True)

    for pair, group in sorted_prices.groupby("pair"):
        # .to_numpy(dtype="datetime64[ns]") rather than the bare .values
        # used previously — .values on a tz-aware Series silently
        # strips tz info in a way that's easy to lose track of (see the
        # historical note below); an explicit target dtype makes the
        # array's actual representation unambiguous at the point it's
        # built, rather than relying on whatever .values happens to
        # infer.
        price_dict[pair] = {
            "datetimes": group["datetime"].to_numpy(dtype="datetime64[ns]"),
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

        # O(log n) lookup for the datetime index.
        #
        # THE ACTUAL ROOT CAUSE (numpy 2.x, confirmed by reproducing the
        # exact production error locally): p_datetimes is a genuine
        # numpy datetime64[ns] array and dt is a genuine pandas.Timestamp
        # — both individually correct, tz-naive, same precision — yet
        # numpy 2.x's searchsorted no longer implicitly coerces a
        # pandas.Timestamp needle against a datetime64 array the way
        # earlier numpy versions did, and fails with the exact
        # "'<' not supported between instances of 'int' and 'Timestamp'"
        # error seen in production. This was NOT the tz-awareness bug
        # fixed earlier (that was real too, but a different, now-closed
        # issue) — this is a distinct numpy-version compatibility gap.
        # Explicitly converting the needle to np.datetime64 removes any
        # ambiguity about what numpy should coerce, and works
        # identically across numpy versions.
        idx = np.searchsorted(p_datetimes, np.datetime64(dt), side="right")

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
