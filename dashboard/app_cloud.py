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
   Header/caption text updated: this project is now Weekly-timeframe,
   not 4H, and the actual run cadence should be read from
   scheduler.run_times in config.yaml directly rather than hardcoded
   here — the old "twice daily (12:00 and 20:00 UTC)" caption was
   specific to the 4H-era schedule and is very likely stale now that
   this project predicts 2-weekly-candles-ahead; this file no longer
   guesses at a specific cadence in its caption text.
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
        "Bollinger + Momentum — Weekly"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Load indicator data ───────────────────────────────────────────────────────
indicator_df = read_latest_indicator_results()

if indicator_df.empty:
    st.warning("No scan data yet. Pipeline has not run or Supabase is empty.")
    st.stop()

latest_datetime = indicator_df["datetime"].max()
st.caption(
    f"Last run: {latest_datetime} UTC — "
    f"prices may have moved since this run. This project predicts "
    f"direction 2 weekly candles ahead — see the deployment's "
    f"scheduler.run_times in config.yaml for the actual run cadence "
    f"(not hardcoded here, since it's a deployment-specific setting "
    f"this dashboard file shouldn't need to track)."
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

    cols_to_show = ["pair", "direction", "up_probability", "confidence"]

    for tab, basket_name in zip(basket_tabs, BASKETS.keys()):
        with tab:
            basket_predictions = read_latest_prediction_results(basket=basket_name)

            if basket_predictions.empty:
                st.info(
                    f"No predictions cleared the display threshold for "
                    f"{basket_name} this run."
                )
                continue

            display_cols = [c for c in cols_to_show if c in basket_predictions.columns]

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
# SECTION 3 — MODEL HEALTH
# =============================================================================

st.header("Model Health")

metrics = read_latest_model_metrics()

if metrics.empty:
    st.info("ML models not trained yet.")
else:
    mcols = st.columns(2)
    for i, row in metrics.iterrows():
        with mcols[i % 2]:
            st.subheader(row["model_name"])
            st.metric("Precision", f"{row['precision_score']:.2%}")
            st.metric("AUC-ROC",   f"{row['auc_roc_score']:.3f}")

            if "recall_score" in row and pd.notna(row.get("recall_score")):
                st.metric("Recall", f"{row['recall_score']:.2%}")
            if "pr_auc_score" in row and pd.notna(row.get("pr_auc_score")):
                st.metric("PR-AUC", f"{row['pr_auc_score']:.3f}")

            st.caption(
                f"Trained: {row['train_date']} | "
                f"Samples: {row['n_samples']}"
            )
