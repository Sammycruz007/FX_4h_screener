"""
run_pipeline_cloud.py
----------------------
Main orchestrator for the FX directional-prediction cloud pipeline
(GitHub Actions).

WHAT'S DIFFERENT FROM THE PRIOR (SIGNAL RANKER SCANNER) VERSION —
THIS IS A COMPLETE REDESIGN, NOT AN INCREMENTAL CHANGE:

1. WEEKLY, NOT 4H. data/fetcher.py now fetches native Weekly bars — no
   1H fetch, no session-anchored resampling. See that module's
   docstring.

2. NO LINREG, NO SMC, NO SCANNER, NO CANDIDATES. The prior pipeline's
   steps for LinReg + SMC per pair, the scanner slope+SD-zone gate,
   and direction-aware candlestick-at-extreme computed only for
   candidates are GONE ENTIRELY. There is no "setup" or "candidate"
   concept in this design — every pair in every basket gets an
   unconditional up/down PREDICTION every run. Candlestick features
   are now RAW pattern flags (is_hammer, is_shooting_star), computed
   for every pair up front alongside ADX/CSI, with no gating logic and
   no dependency on a scanner result that no longer exists (see
   ml/features.py and engines/candlestick.py's module docstrings).

3. FIVE BASKET MODELS, NOT ONE SIGNAL RANKER. ML scoring now loops
   over config.yaml's universe.baskets, calling
   ml.signal_ranker.predict_direction() once per basket — each basket
   has its OWN trained model (models/directional_{basket_name}.pkl),
   scored only against that basket's own pairs.

4. DISPLAY THRESHOLD APPLIED HERE, EXPLICITLY, NOT INSIDE THE MODEL.
   Per project decision: each model only surfaces a prediction if its
   probability of success passes the configured threshold (e.g.
   >=70%), and only THAT gets displayed on the dashboard.
   predict_direction() itself returns EVERY pair's raw probability,
   unfiltered — this orchestrator is where config.yaml's
   ml.high_probability_threshold is actually applied, so that decision
   lives in exactly one place, not duplicated or buried in the model.

5. Every pair gets a row in indicator_results (ADX + CSI + candlestick
   flags), but prediction_results (replacing the old scan_results
   table) ONLY includes rows that (a) had enough history to score at
   all and (b) cleared the display threshold. Rows that didn't clear
   the threshold are NOT written.

6. MACRO DRIVERS WIRED IN (per project decision, after comparing
   against the EURUSD reference project's feature set):
   - STEP 3.2 fetches macro driver data (DXY, gold, US10Y, WTI, VIX)
     via data.fetcher.smart_fetch_macro(), right after the FX fetch.
     Kept non-fatal — a macro-source hiccup never blocks the FX
     pipeline (matches data/fetcher.py's own run_data_pipeline).
   - STEP 3.4's snapshot write now includes macro rows alongside FX
     rows (same storage schema), so macro history actually persists
     to Storage for future runs (including ml/train_models.py's
     backfill) to read back — without this, smart_fetch_macro()
     fetching successfully wouldn't be enough; the data has to be
     written too.
   - STEP 7.5 computes each basket's macro features via
     engines.macro.get_macro_features_for_basket(), reusing STEP 6's
     tickers_data dict directly (it already contains the macro
     symbols alongside the 28 FX pairs, since working_df was read
     back from the same Storage snapshot).
   - STEP 9 passes each basket's macro_features_df into
     predict_direction(), which threads it into
     compute_directional_features() the same way training does.

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other file in this project.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone
import pandas as pd
import yaml

from data.fetcher import get_full_universe, smart_fetch, smart_fetch_macro, to_pair_dict
from data.database_cloud import (
    initialise_database,
    write_indicator_results,
    write_prediction_results,
    write_all_predictions_log,
    get_previous_predictions,
    read_unevaluated_predictions,
    write_prediction_outcomes,
)
from data.storage_cloud import write_snapshot, consolidate_snapshots, read_price_history

from engines.adx import compute_adx_latest
from engines.csi import run_csi_engine
from engines.candlestick import compute_raw_pattern_flags
from engines.macro import get_macro_features_for_basket

from ml.signal_ranker import predict_direction, load_directional_model

from utils.logging import get_pipeline_logger

logger = get_pipeline_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
STORAGE_CFG  = config["storage"]
ML_CFG       = config["ml"]
UNIVERSE_CFG = config["universe"]
MACRO_CFG    = config["macro"]

RETENTION_DAYS     = STORAGE_CFG["retention_days"]
DISPLAY_THRESHOLD  = ML_CFG["high_probability_threshold"]
LABEL_FORWARD_PERIODS = ML_CFG["label_forward_periods"]  # 2 business days ahead, daily bars

import datetime as _datetime

def _add_business_days(start_date: _datetime.date, n: int) -> _datetime.date:
    """Add n business days (Mon-Fri) to start_date, skipping weekends —
    matches dashboard/app_cloud.py's identical helper, used here to
    compute each prediction's valid_through_date for outcome tracking."""
    current = start_date
    added   = 0
    while added < n:
        current += _datetime.timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current
BASKETS            = UNIVERSE_CFG["baskets"]

MACRO_DRIVERS         = MACRO_CFG["drivers"]
MACRO_STORAGE_SYMBOLS = sorted({
    driver_cfg["storage_symbol"] for driver_cfg in MACRO_DRIVERS.values()
})


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    run_start   = datetime.now(timezone.utc)
    run_dt_str  = run_start.isoformat()

    logger.info("=" * 70)
    logger.info("FX DIRECTIONAL PREDICTION — CLOUD PIPELINE RUN")
    logger.info(f"Run datetime (UTC): {run_dt_str}")
    logger.info("=" * 70)

    logger.info("\n[STEP 1] Initialising Supabase...")
    try:
        initialise_database()
        logger.info("Supabase initialised")
    except Exception as e:
        logger.critical(f"Supabase initialisation failed: {e}", exc_info=True)
        return

    logger.info("\n[STEP 2] Loading fixed FX pair universe...")
    try:
        tickers = get_full_universe()
        logger.info(f"Universe: {len(tickers)} pairs across {len(BASKETS)} baskets")
    except Exception as e:
        logger.critical(f"Universe load failed: {e}", exc_info=True)
        return

    logger.info("\n[STEP 3] Fetching Weekly OHLC data...")
    try:
        raw_df = smart_fetch(tickers)
        if raw_df.empty:
            logger.error("No OHLC data fetched. Aborting pipeline run.")
            return
        logger.info(
            f"Fetched {len(raw_df)} rows for {raw_df['pair'].nunique()} pairs"
        )
    except Exception as e:
        logger.critical(f"Fetch failed: {e}", exc_info=True)
        return

    logger.info("\n[STEP 3.2] Fetching macro driver data (DXY, gold, US10Y, WTI, VIX)...")
    try:
        macro_raw_df = smart_fetch_macro()
        if macro_raw_df.empty:
            logger.warning(
                "No macro driver data fetched this run — macro features "
                "will fall back to 0.0 for every basket this scan (see "
                "engines/macro.py's fallback behaviour). FX prices are "
                "unaffected — continuing pipeline."
            )
        else:
            logger.info(
                f"Fetched {len(macro_raw_df)} rows for "
                f"{macro_raw_df['pair'].nunique()} macro symbols"
            )
    except Exception as e:
        # Non-fatal, matching data/fetcher.py's run_data_pipeline: a
        # macro-source hiccup should never block the core FX pipeline.
        logger.error(f"Macro fetch failed (non-fatal): {e} — continuing without macro data this run")
        macro_raw_df = pd.DataFrame()

    logger.info("\n[STEP 3.4] Writing raw price snapshot to Supabase Storage...")
    try:
        # Combine FX + macro rows into ONE snapshot write — same
        # storage path/schema for both (see data/fetcher.py's
        # smart_fetch_macro), so they belong in the same snapshot file
        # rather than a separate one. Without this, macro rows would
        # never persist to Storage, and every future run (including
        # train_models.py's backfill) would see an empty macro history
        # even though smart_fetch_macro() successfully fetched it.
        if not macro_raw_df.empty:
            snapshot_input_df = pd.concat([raw_df, macro_raw_df], ignore_index=True)
        else:
            snapshot_input_df = raw_df

        snapshot_ok = write_snapshot(snapshot_input_df, run_timestamp=run_start)
        if snapshot_ok:
            logger.info("Snapshot written successfully")
        else:
            logger.warning(
                "Snapshot write failed or partial — continuing pipeline anyway"
            )
    except Exception as e:
        logger.warning(f"Snapshot write raised an exception: {e} — continuing")

    logger.info(
        f"\n[STEP 3.5] Consolidating Storage snapshots "
        f"(retention: {RETENTION_DAYS} days)..."
    )
    try:
        consolidate_ok = consolidate_snapshots(retention_days=RETENTION_DAYS)
        if consolidate_ok:
            logger.info("Snapshot consolidation complete")
        else:
            logger.warning("Snapshot consolidation failed or was skipped — continuing")
    except Exception as e:
        logger.warning(f"Consolidation raised an exception: {e} — continuing")

    logger.info("\n[STEP 3.6] Reading back full consolidated history for engines...")
    try:
        working_df = read_price_history()
        if working_df.empty:
            logger.warning(
                "read_price_history returned empty after consolidation — "
                "falling back to this run's raw_df"
            )
            working_df = raw_df
        else:
            logger.info(
                f"Loaded {len(working_df)} rows of full history for "
                f"{working_df['pair'].nunique()} pairs"
            )
    except Exception as e:
        logger.warning(
            f"read_price_history failed: {e} — falling back to this run's raw_df"
        )
        working_df = raw_df

    logger.info("\n[STEP 6] Reshaping price data for indicator engines...")
    try:
        tickers_data = to_pair_dict(working_df)
        if not tickers_data:
            logger.error("to_pair_dict produced an empty dict. Aborting.")
            return
        logger.info(f"Loaded data for {len(tickers_data)} pairs")
    except Exception as e:
        logger.critical(f"Reshape failed: {e}", exc_info=True)
        return

    logger.info("\n[STEP 7] Running indicator engines (ADX, candlestick, CSI)...")

    scan_datetime = None
    indicator_rows = []

    for pair, df in tickers_data.items():
        if df.empty:
            continue

        latest_dt = str(df["datetime"].iloc[-1])
        if scan_datetime is None:
            scan_datetime = latest_dt

        adx = compute_adx_latest(pair, df, latest_dt)
        if adx is None:
            continue

        pattern_flags = compute_raw_pattern_flags(pair, df)
        if pattern_flags is None:
            pattern_flags = {"is_hammer": 0, "is_shooting_star": 0}

        indicator_rows.append({
            "pair"             : pair,
            "datetime"         : latest_dt,
            "adx_value"        : adx["adx_value"],
            "plus_di"          : adx["plus_di"],
            "minus_di"         : adx["minus_di"],
            "is_hammer"        : pattern_flags["is_hammer"],
            "is_shooting_star" : pattern_flags["is_shooting_star"],
        })

    if not indicator_rows:
        logger.error("No indicator rows computed for any pair. Aborting.")
        return

    indicator_df = pd.DataFrame(indicator_rows)
    logger.info(f"ADX + candlestick: {len(indicator_df)} pairs computed")

    try:
        csi_df = run_csi_engine(tickers_data, date=scan_datetime)
        logger.info(f"CSI: {len(csi_df)} pairs computed")
    except Exception as e:
        logger.warning(f"CSI engine failed: {e} — continuing with empty csi_df")
        csi_df = pd.DataFrame()

    csi_cols = [
        "pair", "csi_rs", "csi_diff_zscore", "csi_diff_roc", "csi_commodity_bloc",
        "csi_base_zscore", "csi_quote_zscore",
    ]
    if not csi_df.empty:
        indicator_df = indicator_df.merge(csi_df[csi_cols], on="pair", how="left")
    else:
        for col in csi_cols:
            if col != "pair":
                indicator_df[col] = None

    try:
        rows_written = write_indicator_results(indicator_df)
        logger.info(f"Indicator results written: {rows_written} rows")
    except Exception as e:
        logger.error(f"Failed to write indicator results: {e}", exc_info=True)

    logger.info("\n[STEP 7.5] Computing macro driver features per basket...")
    # tickers_data (built in STEP 6 from working_df, which includes
    # macro rows since STEP 3.4 now writes them into the same
    # snapshot) already has the DXY/GOLD/US10Y/WTI/VIX entries
    # alongside the 28 FX pairs — same dict shape
    # get_macro_features_for_basket() expects, so it's reused directly
    # rather than re-fetched or re-shaped.
    macro_features_by_basket = {}
    for basket_name in BASKETS:
        try:
            macro_features_by_basket[basket_name] = get_macro_features_for_basket(
                basket_name, tickers_data,
            )
        except Exception as e:
            logger.warning(
                f"  {basket_name}: macro feature computation failed: {e} — "
                f"this basket's macro columns will fall back to 0.0"
            )
            macro_features_by_basket[basket_name] = pd.DataFrame()

        if macro_features_by_basket[basket_name].empty:
            logger.warning(
                f"  {basket_name}: no macro features this run (missing "
                f"driver history) — falling back to 0.0 for this basket's "
                f"macro columns"
            )

    logger.info("\n[STEP 8.5] Evaluating outcomes for expired predictions...")
    try:
        today_date = scan_datetime.date() if hasattr(scan_datetime, "date") else pd.Timestamp(scan_datetime).date()
        unevaluated = read_unevaluated_predictions(as_of_date=today_date.isoformat())

        if unevaluated.empty:
            logger.info("No predictions awaiting outcome evaluation")
        else:
            outcome_records = []
            for _, pred_row in unevaluated.iterrows():
                # pred_row["datetime"] now comes back as a plain Python
                # datetime.date — prediction_results.datetime was
                # migrated from TIMESTAMPTZ to DATE (see
                # database_cloud.py's initialise_database migrations),
                # since every value here was always midnight UTC
                # anyway (a candle DATE, never a real time-of-day).
                # psycopg2 maps a DATE column straight to
                # datetime.date, so no tz_localize/tz_convert dance is
                # needed here at all anymore — a plain date has no
                # timezone to be aware or naive ABOUT. This eliminates
                # the entire bug class that caused "Cannot pass a
                # datetime or Timestamp with tzinfo with the tz
                # parameter" here, and the earlier engines/macro.py and
                # continuation-lookup tz bugs elsewhere in this project
                # — all three were the same root cause (candle dates
                # stored/passed as full datetimes, sometimes tz-aware,
                # sometimes not), now structurally impossible for this
                # column.
                pred_date = pred_row["datetime"]
                if not isinstance(pred_date, _datetime.date):
                    pred_date = pd.Timestamp(pred_date).date()

                valid_through = _add_business_days(pred_date, LABEL_FORWARD_PERIODS)

                # Only evaluate predictions whose window has GENUINELY
                # elapsed (today is after valid_through) — not ones
                # still in-flight. read_unevaluated_predictions' SQL
                # does NOT filter by expiry itself (see its docstring
                # in database_cloud.py) — this is the actual, only
                # place that check happens.
                if today_date <= valid_through:
                    continue

                pair_prices = working_df[working_df["pair"] == pred_row["pair"]].sort_values("datetime")

                # working_df["datetime"] is the raw OHLC price history
                # from read_price_history() (a Storage/Parquet read) —
                # a DIFFERENT column than prediction_results.datetime,
                # untouched by the DATE migration above, and likely
                # still a real tz-aware timestamp (candles have an
                # actual fetch/close moment, unlike a prediction's
                # candle-date reference). Comparing a plain
                # pred_date/valid_through (datetime.date) against this
                # column needs pred_date/valid_through converted to
                # match whatever working_df's dtype actually is, not
                # the other way around — normalize once here, at the
                # single remaining comparison point in this function
                # that still spans the two different column types.
                prices_dt = pair_prices["datetime"]
                if pd.api.types.is_datetime64_any_dtype(prices_dt):
                    if prices_dt.dt.tz is not None:
                        pred_datetime_cmp     = pd.Timestamp(pred_date, tz=prices_dt.dt.tz)
                        valid_through_cmp     = pd.Timestamp(valid_through, tz=prices_dt.dt.tz)
                    else:
                        pred_datetime_cmp     = pd.Timestamp(pred_date)
                        valid_through_cmp     = pd.Timestamp(valid_through)
                else:
                    # working_df["datetime"] isn't a proper pandas
                    # datetime dtype at all (e.g. plain objects/strings
                    # from an unusual read path) — fall back to plain
                    # date comparison, which pandas can still evaluate
                    # correctly against date-like strings/objects.
                    pred_datetime_cmp = pred_date
                    valid_through_cmp = valid_through

                price_at_prediction = pair_prices[prices_dt <= pred_datetime_cmp]
                price_at_expiry     = pair_prices[prices_dt <= valid_through_cmp]

                if price_at_prediction.empty or price_at_expiry.empty:
                    logger.warning(
                        f"  {pred_row['pair']}: insufficient price history to "
                        f"evaluate outcome (need data through {valid_through.isoformat()}, "
                        f"have through {prices_dt.max() if not prices_dt.empty else 'nothing'}) "
                        f"— skipping this run, will retry next run"
                    )
                    continue

                close_at_prediction = price_at_prediction.iloc[-1]["close"]
                close_at_expiry     = price_at_expiry.iloc[-1]["close"]
                actual_change       = float((close_at_expiry - close_at_prediction) / close_at_prediction)

                predicted_up = pred_row["direction"] == "up"
                actual_up    = actual_change > 0
                outcome      = "correct" if predicted_up == actual_up else "incorrect"

                outcome_records.append({
                    "pair"                : pred_row["pair"],
                    "basket"              : pred_row["basket"],
                    "prediction_datetime" : pred_date.isoformat(),
                    "direction"           : pred_row["direction"],
                    "up_probability"      : float(pred_row["up_probability"]),
                    "valid_through_date"  : valid_through.isoformat(),
                    "actual_close_change" : round(actual_change, 6),
                    "outcome"             : outcome,
                })

            if outcome_records:
                written = write_prediction_outcomes(outcome_records)
                logger.info(f"Outcomes evaluated and recorded: {written}")
            else:
                logger.info("No predictions had both expired AND had sufficient price history yet")
    except Exception as e:
        logger.error(f"Outcome evaluation failed (non-fatal): {e}", exc_info=True)

    logger.info("\n[STEP 9] Running basket-grouped directional predictions...")

    all_predictions = []

    for basket_name, basket_pairs in BASKETS.items():
        logger.info(f"  Basket: {basket_name} | Pairs: {basket_pairs}")

        try:
            pipeline = load_directional_model(basket_name)
            if pipeline is None:
                logger.warning(
                    f"  {basket_name}: model not found — skipping this basket"
                )
                continue

            basket_predictions = predict_direction(
                basket_name     = basket_name,
                basket_pairs    = basket_pairs,
                prices_df       = working_df,
                csi_df          = csi_df,
                signal_datetime = scan_datetime,
                pipeline        = pipeline,
                macro_features_df = macro_features_by_basket.get(basket_name),
            )

            if not basket_predictions.empty:
                basket_predictions["basket"] = basket_name
                all_predictions.append(basket_predictions)

        except Exception as e:
            logger.error(f"  {basket_name}: prediction failed: {e}", exc_info=True)
            continue

    if not all_predictions:
        logger.warning("No predictions produced by any basket model this run.")
        _log_pipeline_complete(run_start)
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"Total predictions across all baskets: {len(predictions_df)}")

    predictions_df["direction"] = predictions_df["up_probability"].apply(
        lambda p: "up" if p >= 0.5 else "down"
    )
    predictions_df["confidence"] = predictions_df["up_probability"].apply(
        lambda p: p if p >= 0.5 else 1 - p
    )

    display_df = predictions_df[predictions_df["confidence"] >= DISPLAY_THRESHOLD].copy()

    logger.info(
        f"Predictions clearing display threshold ({DISPLAY_THRESHOLD}): "
        f"{len(display_df)}/{len(predictions_df)}"
    )

    # Continuation vs flip — for each DISPLAYED prediction, look up
    # that pair's most recent PRIOR prediction (any probability, not
    # just threshold-clearing) from all_predictions_log. This answers
    # "is today's signal a continuation of yesterday's direction, or a
    # flip" even when yesterday's prediction never appeared on the
    # dashboard because it was below threshold.
    #
    # CRITICAL ORDERING: this lookup MUST happen BEFORE today's own
    # predictions are written to all_predictions_log below. Doing it
    # after (as an earlier version of this code did) meant
    # get_previous_predictions() would find TODAY's own just-written
    # row as the "most recent prior prediction" for every pair — since
    # it orders by datetime DESC and today's row is now the newest —
    # producing "continuation" labels that compared each prediction
    # against ITSELF (same direction and probability, trivially
    # "continuation" every single time, which is exactly the bug
    # reported: EURAUD/GBPNZD showing "was up @ 0.81"/"was up @ 0.82"
    # matching their OWN current probability, even on pairs whose
    # actual prior signal was a SELL). The previous code had a
    # `prior["datetime"] == scan_datetime` guard meant to catch this,
    # but relying on exact datetime equality to detect "is this row
    # the one I just wrote" is fragile (this project has hit tz-aware
    # vs tz-naive comparison bugs before) — reordering so the lookup
    # simply cannot see today's row removes the failure mode
    # structurally instead of guarding against it.
    if not display_df.empty:
        lookup_pairs = list(zip(display_df["pair"], display_df["basket"]))
        try:
            previous = get_previous_predictions(lookup_pairs)
        except Exception as e:
            logger.error(f"Failed to fetch previous predictions for continuation check (non-fatal): {e}", exc_info=True)
            previous = {}

        def _continuation_label(row):
            prior = previous.get((row["pair"], row["basket"]))
            if prior is None:
                # No prior row exists at all for this pair (first time
                # ever logged, or a gap in fetch history).
                return "first signal", None

            # Report the prior prediction's DIRECTION-RELATIVE
            # confidence, not its raw up_probability. up_probability is
            # always "probability of UP" regardless of which direction
            # was actually predicted — e.g. a DOWN call with
            # up_probability=0.4368 means the model was 1-0.4368=0.5632
            # (~56%) confident in DOWN, not 44% confident in anything.
            # Displaying the raw up_probability as if it were "how
            # confident was the prior call" is misleading for any prior
            # DOWN prediction specifically (up_probability < 0.5 in
            # that case reads as LOW confidence when it's actually
            # reporting the flip side of a confident down call).
            # This mirrors exactly how predictions_df["confidence"]
            # is computed for today's own row, earlier in this
            # function — same transformation, applied consistently to
            # both today's and the prior's probability.
            prior_up_probability = prior["up_probability"]
            prior_confidence = (
                prior_up_probability if prior_up_probability >= 0.5
                else 1 - prior_up_probability
            )

            if prior["direction"] == row["direction"]:
                return "continuation", prior_confidence
            return "flip", prior_confidence

        labels = display_df.apply(_continuation_label, axis=1, result_type="expand")
        display_df["signal_status"]        = labels[0]
        display_df["previous_probability"] = labels[1]

    # Log EVERY prediction (all 28 pairs, regardless of threshold) —
    # this is the raw feed continuation/flip comparisons read from,
    # since prediction_results (below) only ever holds threshold-
    # clearing rows and can't answer "what did this pair predict
    # yesterday" if yesterday's happened to be sub-threshold. Written
    # AFTER the continuation lookup above, deliberately — see the note
    # above explaining why this order matters.
    try:
        log_rows = write_all_predictions_log(predictions_df, run_datetime=scan_datetime)
        logger.info(f"All-predictions log written: {log_rows} rows")
    except Exception as e:
        logger.error(f"Failed to write all-predictions log (non-fatal): {e}", exc_info=True)

    if not display_df.empty:
        display_df = display_df.sort_values("confidence", ascending=False).reset_index(drop=True)
        logger.info(
            f"Top display predictions:\n"
            f"{display_df[['pair','basket','direction','confidence','signal_status']].to_string(index=False)}"
        )

    logger.info("\n[STEP 10] Writing display-threshold predictions to Supabase...")
    try:
        if not display_df.empty:
            rows = write_prediction_results(display_df, run_datetime=scan_datetime)
            logger.info(f"Prediction results written: {rows} rows")
        else:
            logger.info("No predictions cleared the display threshold this run — nothing written")
    except Exception as e:
        logger.error(f"Failed to write prediction results: {e}", exc_info=True)

    _log_pipeline_complete(run_start)


def _log_pipeline_complete(run_start: datetime):
    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds() / 60
    logger.info("\n" + "=" * 70)
    logger.info("CLOUD PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}m")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
