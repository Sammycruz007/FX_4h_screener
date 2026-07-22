"""
engines/csi.py
--------------
Currency Strength Index (CSI) engine for the FX Scanner pipeline.
Replaces Market Pulse (SPY/QQQ/DIA) and sector-ETF-based Relative
Strength from the stock project — currencies have no single benchmark
index and no sectors, but they DO have something stocks never had: every
pair is a relationship between exactly two currencies, so strength is
directly, natively measurable at the currency level.

LOGICAL FLOW:
─────────────
STEP 1 — Align all pairs onto a shared timestamp index:
   CSI is a CROSS-PAIR calculation — you cannot compute CSI_EUR from
   EURUSD's data alone, you need EUR's performance across all 7 pairs
   it appears in, at the SAME candle. All 28 pairs' 4H series are
   inner-joined on datetime before any CSI math happens. This is the
   real correctness risk in this engine, not the arithmetic — one
   misaligned pair silently corrupts every currency it touches. We
   inner-join (drop non-overlapping timestamps) rather than
   forward-fill gaps, so we never fabricate a currency's strength
   reading from stale data.

STEP 2 — Per-pair rolling return:
   For each pair, over `csi.lookback_period` candles:
       pair_return[t] = (close[t] / close[t - lookback_period]) - 1
   Uses only close[t] and close[t-lookback], both already known at
   candle close — no look-ahead, safe for training and live scoring.

STEP 3 — Signed contribution per currency:
   For pair BASEQUOTE (e.g. EURUSD), a positive pair_return means base
   strengthened / quote weakened:
       contributes +pair_return to BASE's strength
       contributes -pair_return to QUOTE's strength

STEP 4 — CSI per currency:
   Each of the 8 majors appears in exactly 7 of the 28 pairs (by
   construction of the fixed universe). CSI_X[t] is the mean of X's
   7 signed contributions at time t. Computed ONCE per pipeline run
   across all 8 currencies, not recomputed per-pair.

STEP 5 — Pair-level RS feature:
   csi_rs[t] = CSI_BASE[t] - CSI_QUOTE[t]
   e.g. EURUSD's csi_rs = CSI_EUR - CSI_USD. This is the primary
   relative-strength feature, replacing sector RS.

STEP 6 — Commodity-bloc cohesion (secondary, complementary feature):
   csi_commodity_bloc[t] = mean(CSI_AUD[t], CSI_NZD[t], CSI_CAD[t])
   A regime signal ("is the commodity bloc moving together right now"),
   not subtracted from anything — exposed flat, same value for every
   pair at a given timestamp. Cheap given CSI already exists.

OUTPUT per pair (latest candle):
   - csi_base           : CSI of the pair's base currency
   - csi_quote          : CSI of the pair's quote currency
   - csi_rs             : csi_base - csi_quote (primary RS feature)
   - csi_diff_zscore    : csi_rs normalised against its own rolling
                          ZSCORE_WINDOW-candle history — "is this
                          pair's RS gap at a macro extreme right now?"
                          Makes csi_rs comparable across pairs with very
                          different typical RS-gap magnitudes.
   - csi_diff_roc       : csi_rs differenced over ROC_LAG candles — is
                          the RS gap accelerating or exhausting/rolling
                          over? csi_rs alone is a single snapshot and
                          can't see this on its own.
   - csi_commodity_bloc : mean CSI across AUD/NZD/CAD (regime feature)

WHY THIS MATTERS FOR THE MODEL:
   CSI_base - CSI_quote answers "how is this pair's base doing broadly
   vs. how is its quote doing broadly" — pair-specific and theoretically
   cleaner than benchmarking against a single index the way stocks did
   against SPY/QQQ/DIA. The commodity-bloc feature answers a genuinely
   different question (cross-pair cohesion within a bloc) that the
   pair-specific delta can't fully see on its own — kept as a
   complementary feature, not a replacement, per the project discussion.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_csi_logger
from utils.error_handler import graceful, EngineError

logger = get_csi_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
CSI_CFG      = config["csi"]
UNIVERSE_CFG = config["universe"]

LOOKBACK_PERIOD = CSI_CFG["lookback_period"]   # 20 4H-candles
ZSCORE_WINDOW   = CSI_CFG["zscore_window"]     # 50 4H-candles — rolling window
                                                # csi_diff_zscore is normalised
                                                # against (macro-overextension
                                                # feature, added per project
                                                # discussion)
ROC_LAG         = CSI_CFG["roc_lag"]           # 2 4H-candles — lag csi_diff_roc
                                                # (momentum/exhaustion feature)
                                                # is differenced over
CURRENCIES      = UNIVERSE_CFG["currencies"]   # the 8 majors CSI is computed for
PAIRS           = UNIVERSE_CFG["pairs"]        # the 28 fixed pairs

# Currencies making up the commodity-bloc cohesion feature (secondary,
# per project discussion — AUD/NZD/CAD are the classic commodity-currency
# group that tends to move together on risk sentiment / commodities).
COMMODITY_BLOC_CURRENCIES = ["AUD", "NZD", "CAD"]


def _split_pair(pair: str) -> tuple[str, str]:
    """
    Split a 6-character FX pair string into (base, quote).
    e.g. 'EURUSD' -> ('EUR', 'USD'). Pure string slicing — every pair
    in our fixed universe is exactly 6 characters (two 3-letter ISO
    currency codes), no separators, so this is safe and doesn't need
    a lookup table.
    """
    return pair[:3], pair[3:]


# =============================================================================
# STEP 1 — ALIGN ALL PAIRS ONTO A SHARED TIMESTAMP INDEX
# The real correctness risk in this engine — see module docstring.
# =============================================================================

def _align_pairs(tickers_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Inner-join all pairs' close-price series onto a single shared
    datetime index.

    FLOW:
    1. For each pair, extract [datetime, close] and rename close to
       the pair name (e.g. 'EURUSD')
    2. Inner-join all pairs together on datetime — a timestamp only
       survives if EVERY pair has data for it
    3. Sort by datetime ascending

    We deliberately do NOT forward-fill gaps. Inner-joining means we
    lose some rows at each pair's individual data boundaries, but we
    never fabricate a flat/stale close price for a currency's strength
    calc — a wrong-but-plausible-looking CSI value is worse than a
    slightly shorter aligned history.

    Args:
        tickers_data: Dict mapping pair name -> OHLC DataFrame, each
                      with a 'datetime' column and 'close' column,
                      sorted datetime ascending (matches fetcher.py's
                      output shape, one DataFrame per pair)

    Returns:
        DataFrame indexed by datetime, one column per pair, containing
        only timestamps present in ALL pairs
    """
    close_series = []

    for pair, df in tickers_data.items():
        if df is None or df.empty:
            logger.warning(f"CSI align | {pair} | empty DataFrame, excluded from alignment")
            continue

        s = df.set_index("datetime")["close"].rename(pair)
        close_series.append(s)

    if not close_series:
        logger.error("CSI align | No non-empty pair DataFrames to align")
        return pd.DataFrame()

    # Inner join on index — pd.concat with join="inner" does this cleanly
    # across an arbitrary number of Series at once
    aligned = pd.concat(close_series, axis=1, join="inner")
    aligned = aligned.sort_index()

    logger.info(
        f"CSI align | {len(close_series)} pairs | "
        f"{len(aligned)} aligned timestamps (inner join)"
    )

    return aligned


# =============================================================================
# STEP 2 & 3 — PER-PAIR RETURNS AND SIGNED CURRENCY CONTRIBUTIONS
# =============================================================================

def _compute_signed_contributions(aligned_closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each pair's rolling return, then attribute it as a signed
    contribution to both its base and quote currency.

    MATHS:
    - pair_return[t] = (close[t] / close[t - LOOKBACK_PERIOD]) - 1
    - Contributes +pair_return to the base currency's contribution series
    - Contributes -pair_return to the quote currency's contribution series
      (inverting, since a base-currency gain is a quote-currency loss)

    A currency that appears as BASE in some pairs and QUOTE in others
    (every currency in our fixed universe does) simply accumulates one
    signed contribution column per pair it appears in — these get
    averaged together in step 4.

    Args:
        aligned_closes: Output of _align_pairs() — datetime-indexed,
                        one column per pair

    Returns:
        DataFrame, datetime-indexed, with one column per
        (currency, pair) contribution, named '{currency}__from__{pair}'
        — long-form so step 4 can group by currency and average
    """
    pair_returns = aligned_closes / aligned_closes.shift(LOOKBACK_PERIOD) - 1

    contributions = {}

    for pair in aligned_closes.columns:
        if pair not in PAIRS:
            # Guard against unexpected columns (shouldn't happen given
            # tickers_data is built from the fixed universe, but a
            # silent skip here is safer than crashing on a stray column)
            logger.warning(f"CSI contributions | Unrecognised pair '{pair}', skipped")
            continue

        base, quote = _split_pair(pair)
        col = f"{base}__from__{pair}"
        contributions[col] = pair_returns[pair]

        col_q = f"{quote}__from__{pair}"
        contributions[col_q] = -pair_returns[pair]

    return pd.DataFrame(contributions)


# =============================================================================
# STEP 4 — CSI PER CURRENCY
# =============================================================================

def _compute_csi_per_currency(contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Average each currency's signed contributions (one per pair it
    appears in) into a single CSI series per currency.

    Each of the 8 majors appears in exactly 7 of the 28 pairs by
    construction of the fixed universe — so this is a mean over 7
    columns per currency, not a variable-width average.

    Args:
        contributions: Output of _compute_signed_contributions() —
                       long-form, columns named '{currency}__from__{pair}'

    Returns:
        DataFrame, datetime-indexed, one column per currency
        (e.g. 'CSI_EUR', 'CSI_USD', ...)
    """
    csi = {}

    for currency in CURRENCIES:
        prefix = f"{currency}__from__"
        matching_cols = [c for c in contributions.columns if c.startswith(prefix)]

        if len(matching_cols) != 7:
            # Not fatal — but a real signal something's off with the
            # universe (a currency should appear in exactly 7 of 28
            # pairs). Log loudly rather than silently averaging over
            # whatever happened to be there.
            logger.warning(
                f"CSI per-currency | {currency} | expected 7 contributing "
                f"pairs, found {len(matching_cols)}"
            )

        csi[f"CSI_{currency}"] = contributions[matching_cols].mean(axis=1)

    return pd.DataFrame(csi)


# =============================================================================
# STEP 5 & 6 — PAIR-LEVEL RS AND COMMODITY-BLOC COHESION
# =============================================================================

def _compute_pair_features(csi_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Derive the two pair-facing features from the per-currency CSI series:
    csi_rs (base - quote) for every pair, and the shared
    csi_commodity_bloc regime signal.

    Args:
        csi_df: Output of _compute_csi_per_currency() — datetime-indexed,
               one column per currency ('CSI_EUR', 'CSI_USD', ...)

    Returns:
        Dict mapping pair name -> single-column DataFrame of csi_rs,
        plus a special key '__commodity_bloc__' holding the shared
        csi_commodity_bloc series (same for every pair)
    """
    result = {}

    for pair in PAIRS:
        base, quote = _split_pair(pair)
        base_col  = f"CSI_{base}"
        quote_col = f"CSI_{quote}"

        if base_col not in csi_df.columns or quote_col not in csi_df.columns:
            logger.warning(f"CSI pair features | {pair} | missing CSI column(s), skipped")
            continue

        result[pair] = csi_df[base_col] - csi_df[quote_col]

    commodity_bloc_cols = [f"CSI_{c}" for c in COMMODITY_BLOC_CURRENCIES]
    missing = [c for c in commodity_bloc_cols if c not in csi_df.columns]
    if missing:
        logger.warning(f"CSI commodity bloc | Missing columns: {missing}")
        commodity_bloc = pd.Series(np.nan, index=csi_df.index)
    else:
        commodity_bloc = csi_df[commodity_bloc_cols].mean(axis=1)

    result["__commodity_bloc__"] = commodity_bloc

    return result


# =============================================================================
# CSI_DIFF Z-SCORE AND RATE-OF-CHANGE
# Two additional pair-facing features derived from csi_rs itself (per
# project discussion — these sharpen csi_rs from a raw, unbounded
# number into something comparable across pairs and over time):
#
# csi_diff_zscore — how extreme is THIS pair's current csi_rs relative
#   to its OWN recent history (rolling ZSCORE_WINDOW candles)? A raw
#   csi_rs of -0.03 might be a huge outlier for a normally-tight pair
#   and unremarkable for a volatile one — z-scoring against the pair's
#   own rolling mean/std makes the feature comparable across all 28
#   pairs, and gives the model a genuine "macro overextension" signal
#   to combine with price-level SD extremes (mean-reversion setups).
#
# csi_diff_roc — is the relative-strength gap accelerating or rolling
#   over? A simple ROC_LAG-period difference of csi_rs. Captures
#   momentum/exhaustion on the RS gap itself, which csi_rs alone
#   (a single snapshot) cannot.
# =============================================================================

def _compute_zscore_and_roc(pair_features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Compute csi_diff_zscore and csi_diff_roc for every pair, from each
    pair's own csi_rs series.

    MATHS:
        csi_diff_zscore[t] = (csi_rs[t] - rolling_mean_50(csi_rs)[t])
                              / rolling_std_50(csi_rs)[t]
        csi_diff_roc[t]     = csi_rs[t] - csi_rs[t - ROC_LAG]

    Both use ONLY past/current values at each t (rolling window and a
    backward-looking lag) — no look-ahead, same guarantee as csi_rs
    itself.

    Args:
        pair_features: Output of _compute_pair_features() — dict
                       mapping pair name -> csi_rs Series, plus the
                       '__commodity_bloc__' key (ignored here)

    Returns:
        Dict mapping pair name -> DataFrame with columns
        ['csi_diff_zscore', 'csi_diff_roc'], datetime-indexed
    """
    result = {}

    for pair in PAIRS:
        if pair not in pair_features:
            continue

        csi_rs_series = pair_features[pair]

        rolling_mean = csi_rs_series.rolling(window=ZSCORE_WINDOW).mean()
        rolling_std  = csi_rs_series.rolling(window=ZSCORE_WINDOW).std()

        # Guard against division by zero (a pair whose csi_rs is
        # perfectly flat for ZSCORE_WINDOW candles — extremely
        # unlikely with real price data, but a real edge case to
        # handle explicitly rather than silently produce inf).
        zscore = (csi_rs_series - rolling_mean) / rolling_std.replace(0, np.nan)

        roc = csi_rs_series - csi_rs_series.shift(ROC_LAG)

        result[pair] = pd.DataFrame({
            "csi_diff_zscore": zscore,
            "csi_diff_roc"   : roc,
        })

    return result


# =============================================================================
# BATCH RUNNER
# Matches the calling shape of run_linreg_engine / run_smc_engine /
# run_adx_engine — dict[pair -> DataFrame] in, DataFrame out — but
# internally the computation is genuinely cross-pair (see module
# docstring), unlike those single-pair engines.
# =============================================================================

def run_csi_engine(
    tickers_data : dict[str, pd.DataFrame],
    date         : str,
) -> pd.DataFrame:
    """
    Run the full CSI engine across the whole aligned FX universe and
    return the latest-candle CSI features for every pair.

    FLOW:
    1. Align all pairs onto a shared timestamp index (inner join)
    2. Compute per-pair rolling returns and signed currency contributions
    3. Average into 8 per-currency CSI series
    4. Derive csi_rs per pair and the shared csi_commodity_bloc feature
    5. Derive csi_diff_zscore (csi_rs normalised against its own rolling
       ZSCORE_WINDOW history) and csi_diff_roc (csi_rs differenced over
       ROC_LAG candles) per pair
    6. Extract the LATEST row's values per pair, package into records

    NaN handling at the edges: the first LOOKBACK_PERIOD aligned rows
    have no valid return for any pair, so csi_rs is undefined there too.
    csi_diff_zscore needs a further ZSCORE_WINDOW candles of valid
    csi_rs on top of that warm-up. Both warm-up windows are naturally
    NaN (not zero-filled) and simply won't be selected since we only
    take the latest row per pair; the historical engine (backfill) is
    expected to drop them explicitly rather than train on a fabricated
    flat-strength signal.

    Args:
        tickers_data : Dict mapping pair name -> OHLC DataFrame, each
                       with 'datetime' and 'close' columns
        date         : Today's date string YYYY-MM-DD (for keying,
                       consistent with the other engines even though
                       CSI's real timestamp is the aligned datetime)

    Returns:
        DataFrame with one row per pair: [pair, date, csi_base,
        csi_quote, csi_rs, csi_diff_zscore, csi_diff_roc,
        csi_commodity_bloc]
    """
    logger.info(f"CSI engine starting | {len(tickers_data)} pairs | Date: {date}")

    aligned = _align_pairs(tickers_data)
    if aligned.empty:
        logger.warning("CSI engine: alignment produced no rows, aborting")
        return pd.DataFrame()

    # Minimum rows needed: LOOKBACK_PERIOD for a single valid csi_rs
    # reading, PLUS ZSCORE_WINDOW more for a single valid zscore reading
    # on top of that (the zscore's own rolling window needs its own
    # warm-up, on top of csi_rs's warm-up).
    min_required = LOOKBACK_PERIOD + ZSCORE_WINDOW
    if len(aligned) <= min_required:
        logger.warning(
            f"CSI engine: only {len(aligned)} aligned rows, need "
            f"{min_required + 1}+ for a single valid csi_diff_zscore reading"
        )
        return pd.DataFrame()

    contributions = _compute_signed_contributions(aligned)
    csi_df        = _compute_csi_per_currency(contributions)
    pair_features = _compute_pair_features(csi_df)
    zscore_roc    = _compute_zscore_and_roc(pair_features)

    commodity_bloc_latest = pair_features["__commodity_bloc__"].iloc[-1]

    results = []
    skipped = 0

    for pair in PAIRS:
        if pair not in pair_features:
            skipped += 1
            continue

        base, quote = _split_pair(pair)
        base_col, quote_col = f"CSI_{base}", f"CSI_{quote}"

        if base_col not in csi_df.columns or quote_col not in csi_df.columns:
            skipped += 1
            continue

        csi_rs_series = pair_features[pair]
        latest_rs      = csi_rs_series.iloc[-1]

        if pd.isna(latest_rs):
            logger.debug(f"{pair} | Latest csi_rs is NaN (inside lookback warm-up), skipped")
            skipped += 1
            continue

        latest_zscore = zscore_roc[pair]["csi_diff_zscore"].iloc[-1]
        latest_roc    = zscore_roc[pair]["csi_diff_roc"].iloc[-1]

        if pd.isna(latest_zscore) or pd.isna(latest_roc):
            logger.debug(f"{pair} | Latest csi_diff_zscore/roc is NaN (inside zscore warm-up), skipped")
            skipped += 1
            continue

        # Round base/quote FIRST, then derive csi_rs from the rounded
        # values — guarantees csi_rs == csi_base - csi_quote EXACTLY as
        # stored, so anyone recomputing the delta from the saved columns
        # gets a bit-for-bit match rather than a ~1e-6 rounding artifact
        # from rounding all three independently.
        csi_base_rounded  = round(float(csi_df[base_col].iloc[-1]), 6)
        csi_quote_rounded = round(float(csi_df[quote_col].iloc[-1]), 6)

        results.append({
            "pair"               : pair,
            "date"               : date,
            "csi_base"           : csi_base_rounded,
            "csi_quote"          : csi_quote_rounded,
            "csi_rs"             : round(csi_base_rounded - csi_quote_rounded, 6),
            "csi_diff_zscore"    : round(float(latest_zscore), 6),
            "csi_diff_roc"       : round(float(latest_roc), 6),
            "csi_commodity_bloc" : round(float(commodity_bloc_latest), 6)
                                    if not pd.isna(commodity_bloc_latest) else None,
        })

    logger.info(
        f"CSI engine complete | "
        f"Computed: {len(results)} | "
        f"Skipped: {skipped}"
    )

    if not results:
        logger.warning("CSI engine: No results produced")
        return pd.DataFrame()

    return pd.DataFrame(results)


# =============================================================================
# FULL SERIES — for feature-building over history (train_models.py backfill)
# Mirrors linreg.py's compute_linreg_series: returns EVERY aligned candle's
# CSI/RS values, not just the latest, for building training features
# across a historical window rather than a single scan date.
# =============================================================================

def compute_csi_series(tickers_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute the full historical CSI/RS series for every pair, for use
    when building training features over a historical window (unlike
    run_csi_engine, which returns only the latest candle per pair for
    live scanning).

    Args:
        tickers_data : Dict mapping pair name -> OHLC DataFrame, each
                       with 'datetime' and 'close' columns

    Returns:
        Long-form DataFrame: [pair, datetime, csi_base, csi_quote,
        csi_rs, csi_diff_zscore, csi_diff_roc, csi_commodity_bloc],
        one row per (pair, aligned timestamp). Rows inside the
        LOOKBACK_PERIOD + ZSCORE_WINDOW combined warm-up window (where
        csi_rs or csi_diff_zscore is NaN) are dropped — callers must
        not train on fabricated warm-up values.
    """
    aligned = _align_pairs(tickers_data)
    min_required = LOOKBACK_PERIOD + ZSCORE_WINDOW
    if aligned.empty or len(aligned) <= min_required:
        logger.warning("compute_csi_series: insufficient aligned history")
        return pd.DataFrame()

    contributions = _compute_signed_contributions(aligned)
    csi_df        = _compute_csi_per_currency(contributions)
    pair_features = _compute_pair_features(csi_df)
    zscore_roc    = _compute_zscore_and_roc(pair_features)

    commodity_bloc = pair_features["__commodity_bloc__"]

    rows = []
    for pair in PAIRS:
        if pair not in pair_features:
            continue

        base, quote = _split_pair(pair)
        base_col, quote_col = f"CSI_{base}", f"CSI_{quote}"
        if base_col not in csi_df.columns or quote_col not in csi_df.columns:
            continue

        pair_df = pd.DataFrame({
            "pair"               : pair,
            "datetime"           : aligned.index,
            "csi_base"           : csi_df[base_col].values,
            "csi_quote"          : csi_df[quote_col].values,
            "csi_rs"             : pair_features[pair].values,
            "csi_diff_zscore"    : zscore_roc[pair]["csi_diff_zscore"].values,
            "csi_diff_roc"       : zscore_roc[pair]["csi_diff_roc"].values,
            "csi_commodity_bloc" : commodity_bloc.values,
        })
        rows.append(pair_df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    # Drop warm-up rows — no fabricated flat-strength signal for training.
    # Dropping on csi_diff_zscore is sufficient (its warm-up window is a
    # strict superset of csi_rs's), but checking both is cheap and makes
    # the intent explicit rather than relying on that being true forever.
    combined = combined.dropna(subset=["csi_rs", "csi_diff_zscore"])

    return combined
