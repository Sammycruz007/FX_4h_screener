"""
dashboard/app_cloud.py
-----------------------
Streamlit Cloud version of the FX Scanner dashboard.

DIFFERENCES from a plain local dashboard:
- Reads from Supabase (PostgreSQL) instead of SQLite
- Gets DB credentials from st.secrets (Streamlit Cloud secrets)
- Sets SUPABASE_DB_URL env var so database_cloud.py can connect
- Entry point for Streamlit Cloud deployment

DEPLOYMENT:
- Streamlit Cloud looks for a file called streamlit_app.py at root
- We create streamlit_app.py that simply imports this file

WHAT REPLACES THE STOCK PROJECT'S MARKET PULSE / SECTOR HEALTH SECTIONS:
   The stock dashboard had two separate sections: "Market Pulse" (3
   gauges for SPY/QQQ/DIA) and "Sector Health" (11 tiles for sector
   ETFs) — two distinct levels of market-context display, matching the
   two-level waterfall in the stock scanner. FX collapses both into
   ONE "Currency Strength" section: 8 gauges, one per major currency,
   each showing that currency's CSI z-score (engines/csi.py's
   csi_base_zscore/csi_quote_zscore — display-only, NOT a model
   feature, added specifically so this section has something bounded
   and gauge-readable to show, matching the existing SD-position gauge
   style). There is no second, separate tier underneath this one — CSI
   IS the per-currency-level reading; there's no natural "sector" layer
   below it for FX the way there was for stocks. A small
   csi_commodity_bloc callout sits alongside the grid, since that's a
   genuinely distinct regime signal (AUD/NZD/CAD cohesion) that
   doesn't belong to any single currency's own gauge.

WHAT'S DROPPED FROM THE STOCK PROJECT'S DASHBOARD:
   - Market Pulse section (SPY/QQQ/DIA) — replaced by Currency Strength
   - Sector Health section (11 sector ETF tiles) — folded into Currency
     Strength above; no second tier for FX
   - get_market_sector_status import — that function doesn't exist in
     scanner/screener.py; the whole waterfall/status concept it served
     doesn't apply to FX (see screener.py's module docstring)
   - GFT Watchlist section entirely, and its
     read_latest_gft_watchlist_results import — a 15-stock evaluation-
     account diagnostic with no FX equivalent, dropped per project
     decision (see ml/signal_ranker.py's module docstring)
   - volume_signal column from the Scanner Results table — no real
     volume data exists for FX

WHAT'S RENAMED:
   - ticker -> pair, throughout
   - "date" -> "datetime" for the last-run caption

WHAT CHANGED IN THIS PASS — SECTION 2 IS A REDESIGN, NOT A RENAME:
   The project moved from a scanner-flagged-candidate design (LinReg +
   SMC + a slope/SD-zone gate, one Signal Ranker scoring pre-chosen
   long/short candidates) to an unconditional, basket-grouped
   directional-prediction design (no scanner, no candidates — every
   pair in every basket gets an up/down probability every run, and
   only predictions clearing the project's display threshold — e.g.
   >=70% confidence — are ever written to Supabase at all; see
   run_pipeline_cloud.py's STEP 9.5 and data/database_cloud.py's
   module docstring for where that filtering happens). Concretely:
     - read_latest_scan_results (direction="long"/"short") is GONE —
       replaced by read_latest_prediction_results(basket=...), which
       reads the NEW prediction_results table
     - "Scanner Results" (Long/Short candidate tabs) is replaced by
       "Predictions" — one section per basket, since each basket has
       its own independently-trained model, showing pair/direction/
       confidence rather than pair/currency_bloc/sd_position/ml_score
     - ml_rank (a per-candidate rank) is gone — replaced by sorting on
       confidence, since there's no candidate-ranking concept anymore
     - currency_bloc, sd_position, has_valid_zone, ml_score — all
       LinReg/SMC/scanner-era fields — are gone; predictions now show
       pair, direction, up_probability, confidence
   Section 1 (Currency Strength) and Section 3 (Model Health) needed
   NO structural changes — CSI's per-currency z-score gauges and the
   model-metrics table were never scanner/candidate-dependent.

WHAT CHANGED IN THIS PASS — DAILY TIMEFRAME, LATEST-CANDLE CAPTION:
   The project moved from Weekly to Daily bars (config.yaml's
   fetcher.fetch_interval), and label_forward_periods now means "2
   business days ahead," not "2 weekly candles ahead." The header
   caption previously showed indicator_results' latest datetime as a
   generic "Last run" timestamp — this was misleading twice over: (1)
   it's a scan-completion time, not the underlying candle's date, and
   (2) "last run" invites the reader to wonder how stale the run is,
   when what actually matters for trading is how stale the PRICE DATA
   is and whether the prediction it produced has already expired. The
   caption now reads the fetch_tracker table directly (via
   data.database_cloud.get_last_fetch_dates_bulk(), already used
   elsewhere in this project for the same table) and shows the actual
   latest fetched candle date plus its computed validity window (candle
   date + 2 business days, matching label_forward_periods=2 on daily
   bars) — e.g. "Latest candle: 2026-07-29 — valid through 2026-07-31
   (expires 2026-08-01)". If different pairs somehow show different
   last_fetch_date values (a partial fetch failure), the caption shows
   the OLDEST one and warns, rather than silently reporting the newest
   and hiding that some pairs are stale.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG — must be the FIRST Streamlit command in the script, before
# any other st.* call (including st.error/st.stop below). Calling it later
# throws StreamlitAPIException and masks the real secrets error underneath.
# =============================================================================

SYSTEM_NAME = "Eagle Logic FX System"
SYSTEM_ICON = "🦅"

st.set_page_config(
    page_title = SYSTEM_NAME,
    page_icon  = SYSTEM_ICON,
    layout     = "wide",
)

# ── Set Supabase credentials from Streamlit secrets ───────────────────────────
# Streamlit Cloud reads from .streamlit/secrets.toml
# GitHub Actions reads from environment variable directly
if "SUPABASE_DB_URL" in st.secrets:
    os.environ["SUPABASE_DB_URL"] = st.secrets["SUPABASE_DB_URL"]
elif "SUPABASE_DB_URL" not in os.environ:
    st.error(
        "SUPABASE_DB_URL not found in secrets or environment. "
        "Add it to Streamlit Cloud secrets or .streamlit/secrets.toml"
    )
    st.stop()

# ── Import cloud DB functions ─────────────────────────────────────────────────
from data.database_cloud import (
    initialise_database,
    read_latest_indicator_results,
    read_latest_prediction_results,
    read_latest_model_metrics,
    get_last_fetch_dates_bulk,
    read_prediction_outcomes,
)

# ── Initialise DB tables (safe — IF NOT EXISTS) ───────────────────────────────
initialise_database()


# =============================================================================
# HEADER
# =============================================================================

header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.markdown(
        f"<div style='font-size:64px;line-height:1;'>{SYSTEM_ICON}</div>",
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        f"<h1 style='margin-bottom:0;'>{SYSTEM_NAME}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#9ca3af;margin-top:0;'>"
        "Basket-Grouped Directional Predictions — ADX + Currency Strength + "
        "Bollinger + Momentum"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Load indicator data ───────────────────────────────────────────────────────
indicator_df = read_latest_indicator_results()

if indicator_df.empty:
    st.warning("No scan data yet. Pipeline has not run or Supabase is empty.")
    st.stop()

# ── Latest candle date + validity window ──────────────────────────────────
# Replaces the old "Last run: <indicator_results timestamp> UTC" caption.
# That showed when the SCAN completed, not when the underlying PRICE
# DATA is actually from — misleading for judging whether a prediction
# is still actionable. This reads fetch_tracker directly (the same
# table/function data/fetcher.py itself writes to and reads from) to
# show the real latest candle date, plus how long that candle's
# prediction remains valid (label_forward_periods business days ahead
# on daily bars — see config.yaml's ml.label_forward_periods).
import datetime as _datetime
import yaml as _yaml

def _get_label_forward_periods() -> int:
    """Read ml.label_forward_periods from config.yaml directly, so this
    caption never drifts out of sync with the value the pipeline and
    labeller actually train/predict against (this constant was
    previously hardcoded here and silently went stale when the config
    value changed)."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = _yaml.safe_load(f)
    return cfg["ml"]["label_forward_periods"]

def _add_business_days(start_date: _datetime.date, n: int) -> _datetime.date:
    """Add n business days (Mon-Fri) to start_date, skipping weekends —
    matches FX market closure, consistent with how label_direction()
    itself must skip non-trading days when building the N-day-ahead
    label this validity window is meant to mirror."""
    current = start_date
    added   = 0
    while added < n:
        current += _datetime.timedelta(days=1)
        if current.weekday() < 5:  # Mon=0 ... Fri=4
            added += 1
    return current

LABEL_FORWARD_PERIODS = _get_label_forward_periods()  # from config.yaml's ml.label_forward_periods
                            # (daily bars: "N business days ahead")

fetch_dates = get_last_fetch_dates_bulk()  # {pair: "YYYY-MM-DD"}

if not fetch_dates:
    st.caption("Latest candle date unavailable — fetch_tracker is empty.")
else:
    parsed_dates = {
        pair: _datetime.date.fromisoformat(date_str)
        for pair, date_str in fetch_dates.items()
    }
    oldest_date  = min(parsed_dates.values())
    newest_date  = max(parsed_dates.values())

    # valid_through = the Nth business day itself (the last day the
    # prediction still covers). expires_on = the calendar day
    # immediately after valid_through (the exclusive boundary — once
    # this date arrives, a fresh candle/prediction should exist and
    # the old one should no longer be acted on).
    valid_through = _add_business_days(oldest_date, LABEL_FORWARD_PERIODS)
    expires_on    = valid_through + _datetime.timedelta(days=1)

    if oldest_date == newest_date:
        st.caption(
            f"📅 Latest candle: **{oldest_date.isoformat()}** — "
            f"valid through **{valid_through.isoformat()}** "
            f"(expires {expires_on.isoformat()}). "
            f"Predicts {LABEL_FORWARD_PERIODS} business days ahead."
        )
    else:
        # Pairs disagree on last_fetch_date — a partial fetch failure
        # somewhere. Show the OLDEST (most conservative/limiting) date
        # as the basis for validity, and warn explicitly rather than
        # silently reporting the newest and hiding that some pairs are
        # running on stale data.
        stale_pairs = [p for p, d in parsed_dates.items() if d == oldest_date]
        st.caption(
            f"📅 Latest candle: **{oldest_date.isoformat()}** to "
            f"**{newest_date.isoformat()}** (pairs disagree — "
            f"{len(stale_pairs)} pair(s) on the oldest date) — "
            f"valid through **{valid_through.isoformat()}** "
            f"(expires {expires_on.isoformat()}), based on the OLDEST "
            f"fetched pair. Predicts {LABEL_FORWARD_PERIODS} business "
            f"days ahead."
        )
        st.warning(
            f"⚠️ {len(stale_pairs)} pair(s) have an older last_fetch_date "
            f"than the rest — their predictions may be based on stale "
            f"data: {', '.join(sorted(stale_pairs))}"
        )

BADGE = {"bullish": "🟢", "bearish": "🔴", "broken": "🟡", "unknown": "⚪"}


# =============================================================================
# SECTION 1 — CURRENCY STRENGTH
# Replaces Market Pulse + Sector Health (see module docstring). 8
# gauges, one per major currency, reading csi_base_zscore/
# csi_quote_zscore from indicator_results — display-only values,
# bounded roughly [-3, +3] to match the existing SD-position gauge
# style (see engines/csi.py's module docstring for why these are
# separate from the model-feature csi_diff_zscore).
# =============================================================================

st.header("Currency Strength")
st.caption(
    "Each currency's CSI z-scored against its own rolling history — "
    "positive means stronger than its recent average, negative means weaker."
)

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
COMMODITY_BLOC_CURRENCIES = {"AUD", "NZD", "CAD"}


def _get_currency_zscore(currency: str) -> float | None:
    """
    Look up one currency's zscore from indicator_df. Any row where
    this currency appears as base OR quote holds the same value (CSI
    is currency-level, not pair-level — verified identical across
    every pair sharing a currency at build time), so we just take the
    first match we find, checking base first then quote.
    """
    for _, row in indicator_df.iterrows():
        pair = row["pair"]
        if len(pair) != 6:
            continue
        base, quote = pair[:3], pair[3:]

        if base == currency and pd.notna(row.get("csi_base_zscore")):
            return float(row["csi_base_zscore"])
        if quote == currency and pd.notna(row.get("csi_quote_zscore")):
            return float(row["csi_quote_zscore"])

    return None


currency_cols = st.columns(4)
for i, currency in enumerate(CURRENCIES):
    with currency_cols[i % 4]:
        zscore = _get_currency_zscore(currency)
        bloc_tag = " 🌾" if currency in COMMODITY_BLOC_CURRENCIES else ""

        st.subheader(f"{currency}{bloc_tag}")

        if zscore is None:
            st.caption("No data yet")
            continue

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = zscore,
            title = {"text": "CSI Z-Score", "font": {"color": "#e5e7eb"}},
            gauge = {
                "axis"   : {"range": [-3.5, 3.5], "tickcolor": "#e5e7eb"},
                "bar"    : {"color": "#16a34a" if zscore > 0 else "#dc2626"},
                "bgcolor": "#1c2128",
                "steps"  : [
                    {"range": [-3.5, -1], "color": "#7f1d1d"},
                    {"range": [-1, 1],    "color": "#1c2128"},
                    {"range": [1, 3.5],   "color": "#14532d"},
                ],
                "threshold": {
                    "line" : {"color": "white", "width": 2},
                    "value": zscore,
                },
            },
            number = {"font": {"color": "#e5e7eb"}},
        ))
        fig.update_layout(
            height        = 220,
            margin        = dict(l=20, r=20, t=40, b=20),
            paper_bgcolor = "#1c2128",
            font          = {"color": "#e5e7eb"},
        )
        st.plotly_chart(fig, use_container_width=True, key=f"csi_gauge_{currency}")

# Commodity-bloc cohesion callout — a distinct regime signal, not
# specific to any one currency's own gauge above (see module docstring).
commodity_bloc_rows = indicator_df[indicator_df["csi_commodity_bloc"].notna()]
if not commodity_bloc_rows.empty:
    bloc_value = float(commodity_bloc_rows.iloc[0]["csi_commodity_bloc"])
    st.metric(
        "Commodity Bloc Cohesion (AUD/NZD/CAD mean CSI)",
        f"{bloc_value:+.5f}",
    )


# =============================================================================
# SECTION 2 — PREDICTIONS (replaces Scanner Results — see module docstring)
# One tab per basket (config.yaml's universe.baskets, read dynamically
# so this file doesn't need editing whenever baskets are added/renamed).
# Each basket's model is independent, so predictions are shown grouped
# by basket rather than by long/short direction — direction is now a
# COLUMN within each basket's table, not a separate tab, since a single
# basket's predictions can be a mix of "up" and "down" calls.
# =============================================================================

st.header("Predictions")
st.caption(
    "Only predictions clearing the display confidence threshold are "
    "shown — see config.yaml's ml.high_probability_threshold. Each "
    "basket has its own independently-trained model."
)

import yaml as _yaml

def _load_baskets() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = _yaml.safe_load(f)
    return cfg.get("universe", {}).get("baskets", {})

BASKETS = _load_baskets()

if not BASKETS:
    st.warning(
        "No baskets configured in config.yaml's universe.baskets — "
        "cannot show per-basket predictions."
    )
else:
    basket_tabs = st.tabs([f"📊 {name}" for name in BASKETS.keys()])

    cols_to_show = ["pair", "direction", "up_probability", "confidence", "signal_status", "previous_probability"]

    for tab, basket_name in zip(basket_tabs, BASKETS.keys()):
        with tab:
            basket_predictions = read_latest_prediction_results(
                basket=basket_name,
                as_of_date=oldest_date.isoformat() if fetch_dates else None,
            )

            if basket_predictions.empty:
                date_note = f" for {oldest_date.isoformat()}" if fetch_dates else ""
                st.info(
                    f"No predictions cleared the display threshold for "
                    f"{basket_name}{date_note}. This is a normal, honest "
                    f"result — it means today's candle didn't produce a "
                    f"high-confidence call for this basket, not a display "
                    f"error."
                )
                continue

            display_cols = [c for c in cols_to_show if c in basket_predictions.columns]

            # Human-readable signal_status + previous_probability combo,
            # e.g. "Continuation (was up @ 0.60)" or "Flip (was down @
            # 0.30)" — folds two raw columns into one readable string
            # rather than showing them as separate numeric/text columns.
            if "signal_status" in basket_predictions.columns and "previous_probability" in basket_predictions.columns:
                def _format_signal_status(row):
                    status = row.get("signal_status")
                    prev_p = row.get("previous_probability")
                    if status == "first signal" or pd.isna(prev_p):
                        return "First signal"
                    label = "Continuation" if status == "continuation" else "Flip"
                    return f"{label} (was {row['direction'] if status == 'continuation' else ('down' if row['direction']=='up' else 'up')} @ {prev_p:.2f})"

                basket_predictions["Signal"] = basket_predictions.apply(_format_signal_status, axis=1)
                display_cols = [c for c in display_cols if c not in ("signal_status", "previous_probability")] + ["Signal"]

            # Colour by direction: green background for "up" rows, red
            # for "down" — background_gradient alone (as the old
            # ml_score-based version used) doesn't make sense here since
            # up/down calls need visually distinct treatment, not just a
            # single-direction intensity gradient.
            def _highlight_direction(row):
                color = "#14532d" if row.get("direction") == "up" else "#7f1d1d"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                basket_predictions[display_cols].style.apply(_highlight_direction, axis=1),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# SECTION 2.5 — LIVE TRACKING
# =============================================================================
# Tracks REAL, ongoing prediction accuracy — separate from the
# original backtest's AUC/precision, which only ever measured
# historical held-out data. Once a prediction's validity window has
# elapsed (see run_pipeline_cloud.py's STEP 8.5), the pipeline scores
# it against the actual price move and records the outcome here. This
# is the honest, ongoing answer to "is the deployed model actually
# performing the way the backtest suggested it would" — the backtest
# is a one-time historical estimate; this is live evidence that
# accumulates over time.

st.header("Live Tracking")
st.caption(
    "Real outcomes for expired predictions — was the direction call "
    "actually correct? This tracks live performance separately from "
    "the original backtest metrics shown under Model Health below."
)

outcomes_df = read_prediction_outcomes(limit_days=30)

if outcomes_df.empty:
    st.info(
        "No scored outcomes yet — predictions are scored once their "
        "validity window elapses (2 business days after the "
        "prediction's candle date)."
    )
else:
    # Rolling accuracy summary per basket — the headline number to
    # compare against each basket's backtest AUC/precision over time.
    summary = (
        outcomes_df.groupby("basket")["outcome"]
        .apply(lambda s: (s == "correct").mean())
        .reset_index()
        .rename(columns={"outcome": "accuracy"})
    )
    summary["n_predictions"] = outcomes_df.groupby("basket").size().values

    summary_cols = st.columns(len(summary)) if len(summary) > 0 else []
    for col, (_, row) in zip(summary_cols, summary.iterrows()):
        with col:
            st.metric(
                row["basket"],
                f"{row['accuracy']:.1%}",
                help=f"{int(row['n_predictions'])} predictions scored in the last 30 days",
            )

    st.divider()

    # Detailed outcome table — most recent first.
    display_outcomes = outcomes_df.copy()
    display_outcomes["prediction_datetime"] = pd.to_datetime(
        display_outcomes["prediction_datetime"]
    ).dt.date

    # Show CONFIDENCE, not raw up_probability — up_probability is
    # always "probability of UP" regardless of which direction was
    # actually predicted, so a DOWN call with up_probability=0.20
    # reads as "20% confidence" when the model was actually 80%
    # confident IN THE DOWN CALL. Same fix already applied to the
    # continuation-label logic in run_pipeline_cloud.py — applying it
    # here too so Live Tracking doesn't reintroduce the same
    # misleading framing for outcomes specifically.
    if "up_probability" in display_outcomes.columns and "direction" in display_outcomes.columns:
        display_outcomes["confidence"] = display_outcomes.apply(
            lambda row: row["up_probability"] if row["direction"] == "up" else 1 - row["up_probability"],
            axis=1,
        )

    outcome_cols = [
        "pair", "basket", "prediction_datetime", "direction",
        "confidence", "valid_through_date", "actual_close_change", "outcome",
    ]
    outcome_cols = [c for c in outcome_cols if c in display_outcomes.columns]

    def _highlight_outcome(row):
        color = "#14532d" if row.get("outcome") == "correct" else "#7f1d1d"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        display_outcomes[outcome_cols].style.apply(_highlight_outcome, axis=1).format({"confidence": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# SECTION 3 — MODEL HEALTH
# =============================================================================

st.header("Model Health")

metrics = read_latest_model_metrics()

if metrics.empty:
    st.info("ML models not trained yet.")
else:
    # Table form, one row per basket model — replaces the previous
    # per-model st.metric() card grid, which took much more vertical
    # space to show the same numbers and made cross-basket comparison
    # (e.g. "which basket has the best AUC-ROC?") harder than a plain
    # table with sortable columns.
    display_cols = {
        "model_name"     : "Model",
        "precision_score": "Precision",
        "auc_roc_score"  : "AUC-ROC",
        "recall_score"   : "Recall",
        "pr_auc_score"   : "PR-AUC",
        "train_date"     : "Trained",
        "n_samples"      : "Samples",
    }
    # Only include columns that actually exist in this deployment's
    # model_metrics table — recall_score/pr_auc_score weren't always
    # populated for every model run historically.
    available_cols = [c for c in display_cols if c in metrics.columns]

    health_table = metrics[available_cols].rename(columns=display_cols)

    # Format percentage-style columns as readable strings; leave
    # numeric AUC/PR-AUC as plain floats (already 0-1 scale, not a %).
    if "Precision" in health_table.columns:
        health_table["Precision"] = health_table["Precision"].map(
            lambda v: f"{v:.2%}" if pd.notna(v) else "—"
        )
    if "Recall" in health_table.columns:
        health_table["Recall"] = health_table["Recall"].map(
            lambda v: f"{v:.2%}" if pd.notna(v) else "—"
        )
    if "AUC-ROC" in health_table.columns:
        health_table["AUC-ROC"] = health_table["AUC-ROC"].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
    if "PR-AUC" in health_table.columns:
        health_table["PR-AUC"] = health_table["PR-AUC"].map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )

    st.dataframe(health_table, hide_index=True, use_container_width=True)
