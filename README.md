# Eagle Logic FX System

**Author:** Idongesit Sampson

A machine-learning filter layer for discretionary FX trading — basket-grouped
directional models that predict short-horizon price direction, used to
confirm or discard Smart Money Concepts (SMC) + Linear Regression channel
trade setups before they're taken.

---

## Problem Statement

Discretionary FX trading built on SMC (market structure — BOS, CHOCH, Order
Blocks) and a Linear Regression channel with standard-deviation bands (for
direction and overstretched-price zones) produces clean, well-formed setups
regularly — but even a textbook setup fails often enough that structure and
mean-reversion zones alone aren't a sufficient entry condition. There was no
independent, data-driven check on *direction* to filter out setups that look
right visually but are fighting the underlying short-term trend.

Early attempts at solving this ran into a second, more subtle problem: a
plain forward-return label (`will price be higher in N candles?`) rewards
*any* net-positive path, including sharp reversals — a candle dropping hard,
partially recovering, and technically closing a few pips above where it
started still counts as a correct "BUY" under that definition, even though
no discretionary trader would call that a good call. Live-tracking review
surfaced this directly: "successful" BUY signals were regularly showing
15–40 pip net moves that had actually involved a drop-then-reverse pattern,
not sustained directional pressure.

## Objectives

- Provide a **daily directional bias** (BUY/SELL) per FX pair, over a fixed
  forward horizon, as a **filter** layer — not a standalone signal generator.
  A prediction only becomes a potential trade when it agrees with a clean
  SMC + LinReg setup found independently on a lower timeframe.
- Label and train on **sustained, confirmed moves only** — discard
  directionally ambiguous or reversal-driven price action rather than force
  every historical candle into a BUY/SELL bucket.
- Group pairs into **currency baskets** (rather than one model per pair) to
  give each basket-level model enough training data, while still letting
  each basket lean on the macro drivers that actually move it.
- Run as a **fully automated, scheduled cloud pipeline** — daily data fetch
  and prediction, weekly retraining — with a dashboard for reviewing
  predictions, currency strength, and live (not just backtested)
  performance.
- Track **live outcomes**, not just historical backtest metrics, so
  real-world performance can be verified against what training predicted,
  on an ongoing basis.

## Architecture

```
                    ┌─────────────────────┐
                    │   yfinance (fetch)   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          FX pairs (28)              Macro drivers (5)
     DXY, gold, US10Y, WTI, VIX — mapped per basket
                 │                           │
                 └─────────────┬─────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Supabase Storage     │
                    │  (raw price snapshots,│
                    │   Parquet, chunked +  │
                    │   consolidated)       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       Indicator engines   Macro engine    Directional labeller
      (ADX, CSI, candle-   (per-basket     (BUY/SELL/discard —
       stick patterns)      driver features) see Design Rationale)
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Feature matrix       │
                    │  (per basket: shared   │
                    │   base features +      │
                    │   basket-specific      │
                    │   macro columns)       │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        Training (weekly,            Prediction (daily,
        train_models.yml)            daily_scan.yml)
        XGBoost + isotonic           Loads the current
        calibration, per basket      .pkl per basket
                 │                           │
                 └─────────────┬─────────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Supabase Postgres    │
                    │  (indicator_results,   │
                    │  prediction_results,   │
                    │  all_predictions_log,  │
                    │  prediction_outcomes,  │
                    │  model_metrics,        │
                    │  fetch_tracker)        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Streamlit dashboard   │
                    │  (cached reads —       │
                    │  Currency Strength,    │
                    │  Predictions, Live     │
                    │  Tracking, Model       │
                    │  Health)               │
                    └───────────────────────┘
```

**Currency baskets:** `usd`, `cad`, `chf`, `jpy`, `crosses` — 28 pairs total,
each trained as its own independent model rather than one model per pair or
one shared model for everything. Each basket also carries its own macro
driver set (e.g. `usd` → DXY + gold + US10Y; `cad` → WTI; `chf`/`jpy`/
`crosses` → VIX), reflecting what actually moves that currency rather than
applying one generic macro feature set everywhere.

## Project Structure

```
.
├── config/
│   └── config.yaml              # single source of truth for every
│                                 # threshold, period, and schedule
├── data/
│   ├── fetcher.py                # yfinance fetch (FX + macro), incremental
│   ├── storage_cloud.py          # Parquet snapshot read/write/consolidate
│   └── database_cloud.py         # Postgres schema, all read/write functions
├── engines/
│   ├── adx.py                    # trend-strength indicator
│   ├── csi.py                    # Currency Strength Index (cross-pair)
│   ├── candlestick.py             # raw candlestick pattern flags
│   └── macro.py                  # per-basket macro driver features
├── ml/
│   ├── labeller.py                # BUY/SELL/discard label generation
│   ├── features.py                # feature matrix construction (per basket)
│   ├── signal_ranker.py           # training, calibration, prediction
│   └── train_models.py            # training orchestrator (weekly)
├── utils/
│   ├── logging.py                 # shared logger setup
│   └── error_handler.py           # shared exception types
├── models/                        # trained .pkl per basket (committed by CI)
├── app_cloud.py                   # Streamlit dashboard
├── run_pipeline_cloud.py          # daily fetch + predict orchestrator
└── .github/workflows/
    ├── daily_scan.yml             # daily: fetch + predict (00:50 UTC)
    └── train_models.yml           # weekly: fetch + retrain (03:00 UTC Sat)
```

## Design Rationale

**Why basket-grouped models, not one model per pair or one global model.**
A single pair's history alone was too thin to train a reliable model on;
one shared model across all 28 pairs would blur together currencies with
genuinely different drivers (JPY's risk-sentiment sensitivity vs. CAD's oil
linkage). Grouping by shared quote currency was the middle ground — enough
pooled data per basket, while keeping baskets economically coherent enough
that shared features remain meaningful across the pairs inside them.

**Why a redesigned label, not a plain forward-return check.** The original
label (`close[t+H] > close[t]`) rewarded any net-positive path, including
sharp reversals — exactly the failure mode live-tracking surfaced. The
current label requires three things together: the net move over the horizon
clears a minimum size (`0.65 × Daily ATR`), and at least 2 of the 3 forward
candles individually confirm the direction close-to-close against their own
prior candle — not just the endpoint, and not just the candle's own open-to-
close body. Samples that satisfy neither BUY nor SELL are discarded, not
forced into a label. This trades data volume for label quality: roughly 90%
of candidate rows are discarded under the strictest settings tested, and the
project settled on a looser (but still confirming) variant after comparing
discard rates and out-of-sample precision across several parameter
combinations — the tightest setting produced unusably small samples for the
smaller baskets (`cad` in particular).

**Why close-to-close confirmation, not candle-body direction.** An earlier
version defined a "bullish candle" as `close > open`. That measures whether
a candle finished green regardless of where the *previous* candle closed — a
candle can gap down, open low, and still count as bullish while closing
below the prior candle's close, which happens often during a choppy
pullback and doesn't represent real confirmation. Close-to-close
(`close[t+i] > close[t+i-1]`) is both a more direct reading of "did price
make net progress" and, in practice, a substantially looser bar — switching
to it roughly quadrupled the number of surviving labeled examples per
basket without giving back the precision gains the redesign was meant to
produce.

**Why "2 of 3 individually confirm," not a summed net-body check.** An
earlier draft summed the three candles' bodies together as the confirmation
check, which is mathematically almost redundant with the net-move check
already being applied — a single large outlier candle can make that sum
positive even while the other candles were bearish, which does not protect
against the drop-then-reverse pattern this redesign exists to filter out.
Counting how many candles *individually* agree with the net direction is
the only part of the design that inspects the path rather than just the
endpoint magnitude.

**Why a 3-day horizon, not 2.** The 2-of-3 confirmation rule needs three
candles to mean anything as a genuine majority check; under a 2-day horizon
it would collapse to "both candles must agree," a harsher bar than intended
and one that gives a dip-then-reverse pattern no room to either confirm as
real or get correctly discarded.

**Why a separate full-history refit before deployment.** Every basket model
is evaluated on a genuinely held-out final ~30% of history (by date, with a
gap to prevent label leakage), which is the honest way to measure whether
the approach generalizes. But the model actually deployed for live
prediction is a *second*, separate fit on the full dataset — refitting on
the held-out portion once the approach is validated is standard practice,
and withholding a live model's most recent, most-regime-relevant data from
it for no further benefit would be wasteful.

**Why continuation vs. flip is tracked on every displayed prediction.**
Because predictions are re-issued daily over a multi-day validity window,
consecutive predictions for the same pair necessarily overlap — a SELL
issued yesterday and a BUY issued today can both be "valid" at once. Only
the most recent prediction should ever be acted on; an older, not-yet-
expired prediction is not still in force once a newer one exists. Each
displayed prediction is labeled a continuation or a flip relative to that
pair's actual most recent prior prediction — looked up from the full,
unfiltered prediction log so the comparison is correct even when
yesterday's prediction never cleared the display threshold — so a same-
direction streak versus a direction change is visible without cross-
referencing history by hand.

 Streamlit reruns the
entire script on every user interaction; with no caching, a single dashboard
session could trigger nine or more database round-trips per rerun. This
became a real problem when the project's Supabase organization exceeded its
free-tier egress quota. Every dashboard read is now wrapped in
`st.cache_data` with an hour-long TTL — safe because every underlying table
changes at most once a day (the pipeline runs daily, models retrain weekly),
so an hour of staleness never actually hides new data from the user.

**Why the pipeline runs daily rather than only at signal expiration.** A
directional prediction with a multi-day validity window is meant to be
re-evaluated on every new candle, not just checked once at expiry — running
only at expiration would silently discard roughly half the entry
opportunities the model is capable of generating, and would mean trading
decisions rely on stale information for most of each validity window.

**Why predictions and their live outcomes are tracked separately from the
original backtest.** A held-out backtest is a one-time historical estimate;
it is not a guarantee of future performance, and this project has already
seen the same historical candle produce different model outputs across
runs once model files or feature computation timing changed. Live Tracking
scores every prediction against what price actually did once its validity
window has elapsed, independent of the labeler or model version that
produced it, so real performance can be verified on an ongoing basis rather
than assumed from a single backtest.

## Disclaimer

This system is a decision-support filter for a discretionary trading
process, not a standalone trading system or investment advice. Its
predictions are probabilistic, trained on historical data, and are not
guaranteed to hold under future market conditions. Backtested and
cross-validated performance figures describe how the modeling approach
generalized on historical held-out data; they are not a forecast of future
results. All trading decisions, position sizing, and risk management remain
the sole responsibility of the trader using this system. Past performance,
whether backtested or live-tracked, is not indicative of future results.
