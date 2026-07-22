"""
ml/signal_ranker.py
--------------------
Signal Ranker ML model for the FX Scanner pipeline.

LOGICAL FLOW:
─────────────
This model takes every FX pair that passed the scanner's relaxed
candidate check and assigns it a probability score: how likely is
this setup to succeed (price CLOSING at or beyond the fixed LinReg
mean within FORWARD_PERIODS 4H-candles)?

It is the final ranking layer before the dashboard display.

THIS IS THE ONLY MODEL IN THE FX PROJECT:
   Unlike the stock project (Volume Classifier + Signal Ranker,
   two-stage pipeline), FX has no real volume data (decentralized OTC
   market — see fetcher.py's docstring) and therefore no Volume
   Classifier to train or to feed a vol_clf_score into this model.
   This is a structural simplification, not an oversight — see
   ml/features.py and ml/labeller.py's module docstrings for the same
   reasoning applied consistently across the FX rework.

TRAINING FLOW:
1. Load labelled scan hits from labeller.label_scanner_hits()
2. Build full feature matrix from features.build_signal_feature_matrix()
   (includes CSI relative-strength, ATR-normalized distances, ATR
   ratio, session flags, candlestick-at-extreme interaction, ADX —
   no volume, no sector RS, no Market Pulse)
3. Handle class imbalance
4. Train XGBoost with walk-forward cross-validation
5. Evaluate (Precision + AUC-ROC + PR-AUC)
6. Save model to disk
7. Write metrics to SQLite / Supabase

INFERENCE FLOW (per scan run):
1. Scanner returns N candidates (pairs meeting the relaxed
   slope + SD-zone conditions — see train_models.py)
2. For each candidate, compute full feature vector (CSI looked up
   from a pre-computed universe-wide snapshot, not recomputed per pair
   — CSI is inherently cross-pair, see engines/csi.py)
3. Run predict_proba() -> probability of success
4. Sort candidates by probability descending, within each direction
5. Assign ml_rank (1 = highest probability)
6. Write ranked results to storage
7. Dashboard reads and displays

COLUMN NAMING (pair/datetime, not ticker/date):
   Consistent with every other FX file — 'pair' and 'datetime'
   throughout, matching fetcher.py's actual output shape.

WHAT'S DROPPED FROM THE STOCK PROJECT:
   - vol_clf_score feature and the whole Volume Classifier relationship
     described in the original module docstring
   - build_sector_price_cache / RS_BENCHMARK (SPY) — no sectors, no
     single-benchmark relative strength for currencies (CSI_rs, a
     feature already in SIGNAL_FEATURE_COLS, replaces both)
   - market_ind_df (Market Pulse from SPY/QQQ/DIA) — replaced by
     csi_commodity_bloc, already part of the feature vector
   - GFT_TICKERS watchlist diagnostic — a 15-stock evaluation-account
     artifact specific to the stock project, with no FX equivalent.
     Dropped entirely rather than stubbed with a placeholder list.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from ml.features import SIGNAL_FEATURE_COLS, compute_signal_features
from utils.logging import get_ml_logger
from utils.error_handler import MLError

logger = get_ml_logger()


# =============================================================================
# CONFIG
# Every threshold/period below is read live from config.yaml — nothing
# in this file hardcodes a value that config already owns.
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config     = _load_config()
ML_CFG     = config["ml"]
LINREG_CFG = config["linreg"]

MIN_SAMPLES         = ML_CFG["min_training_samples"]
HIGH_PROB_THRESHOLD = ML_CFG["high_probability_threshold"]
GAP                 = ML_CFG["label_forward_periods"]
LINREG_PERIOD       = LINREG_CFG["period"]

MODEL_DIR  = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "signal_ranker.pkl"

# The train/test and CV gap must be AT LEAST as large as the biggest
# feature lookback window, or validation rows can share overlapping
# history with training rows (leakage) even though they're on
# "different" datetimes. LinReg features look back up to LINREG_PERIOD
# (200 4H-candles) candles — this is read live from config, not
# hardcoded, so a future config change to linreg.period automatically
# widens this gap too.
SAFETY_GAP = max(LINREG_PERIOD, GAP)


# =============================================================================
# MODEL BUILDER
# XGBoost hyperparameters are asset/timeframe-agnostic tuning choices —
# carried over unchanged from the stock project, no FX-specific reason
# to retune them here.
# =============================================================================

def _build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    Build the sklearn Pipeline for the Signal Ranker.

    Args:
        scale_pos_weight: Ratio of negative to positive samples

    Returns:
        sklearn Pipeline
    """
    base_model = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        reg_alpha          = 0.5,
        reg_lambda         = 1.0,
        scale_pos_weight   = scale_pos_weight,
        eval_metric        = "logloss",
        random_state       = 42,
        n_jobs             = -1,
    )

    # Isotonic calibration — cv=5 uses out-of-fold predictions to map
    # probabilities without leaking data or overfitting to train set.
    calibrated_model = CalibratedClassifierCV(
        estimator = base_model,
        method    = "isotonic",
        cv        = 5,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   calibrated_model),
    ])

    return pipeline


# =============================================================================
# TRAINING
# =============================================================================

def train_signal_ranker(
    feature_matrix: pd.DataFrame,
) -> Tuple[Pipeline, dict]:
    """
    Train the Signal Ranker on the labelled feature matrix.

    FLOW:
    1. Validate minimum sample count
    2. Separate features (X) from labels (y), drop identifier columns
    3. Compute class imbalance ratio
    4. Build pipeline
    5. Cut a final OOS holdout FIRST (never touched during CV), with a
       SAFETY_GAP buffer to prevent LinReg-lookback leakage across the
       train/test boundary
    6. Walk-forward cross-validation on TRAIN only
    7. Train final model on TRAIN only
    8. Evaluate on the untouched OOS test set
    9. Save model to disk

    Args:
        feature_matrix: Output of build_signal_feature_matrix()
                        Must contain SIGNAL_FEATURE_COLS + 'label'

    Returns:
        Tuple of (trained Pipeline, metrics dict)
    """
    logger.info("=" * 60)
    logger.info("SIGNAL RANKER TRAINING STARTING")
    logger.info("=" * 60)

    # ── Step 1: Validate sample count ────────────────────────────────────────
    if len(feature_matrix) < MIN_SAMPLES:
        raise MLError(
            f"Insufficient training samples: {len(feature_matrix)} "
            f"(need {MIN_SAMPLES})"
        )

    # ── Step 2: Separate features, labels, AND datetime ─────────────────────
    # Keep datetime out of X — only used to split time.
    df = feature_matrix.copy()
    X  = df[SIGNAL_FEATURE_COLS].copy()
    y  = df["label"].values

    # Features are already sorted chronologically by build_signal_feature_matrix()
    if "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"])
    else:
        dates = pd.Series(range(len(df)), index=df.index)
        logger.warning(
            "No 'datetime' column in feature matrix — "
            "using row index as time proxy for walk-forward split"
        )

    # ── Step 3: Class imbalance ratio ─────────────────────────────────────────
    n_negative       = (y == 0).sum()
    n_positive       = (y == 1).sum()
    raw_ratio        = n_negative / n_positive if n_positive > 0 else 1.0
    scale_pos_weight = raw_ratio

    logger.info(
        f"Class balance | "
        f"Positive: {n_positive} | Negative: {n_negative} | "
        f"Raw ratio: {raw_ratio:.2f} | scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ── Step 4: Build pipeline ────────────────────────────────────────────────
    pipeline = _build_pipeline(scale_pos_weight)

    # ── Step 5: Cut a FINAL HOLDOUT FIRST. Never touch this during CV ───────
    # Last 30% by time = walk-forward reality check. SAFETY_GAP rows are
    # dropped between train and test so test rows near the boundary can't
    # share overlapping LinReg lookback history with training rows right
    # before them.
    split_idx  = int(len(dates) * 0.70)
    test_start = split_idx + SAFETY_GAP

    if test_start >= len(dates):
        logger.warning(
            f"SAFETY_GAP ({SAFETY_GAP}) leaves no room for an OOS test set "
            f"with only {len(dates)} samples — falling back to no gap."
        )
        test_start = split_idx

    train_mask = dates.index < dates.index[split_idx]
    test_mask  = dates.index >= dates.index[test_start]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(
        f"Walk-Forward Split | Train: {len(X_train)} | "
        f"Gap: {test_start - split_idx} rows | "
        f"OOS Test: {len(X_test)} | Ratio: {raw_ratio:.2f}:1"
    )

    if "datetime" in df.columns:
        train_dates    = dates[train_mask]
        test_dates     = dates[test_mask]
        train_pos_rate = y_train.mean() if len(y_train) > 0 else float("nan")
        test_pos_rate  = y_test.mean()  if len(y_test)  > 0 else float("nan")
        logger.info(
            f"Train window | {train_dates.min()} -> {train_dates.max()} | "
            f"Positive rate: {train_pos_rate:.4f}"
        )
        logger.info(
            f"OOS window   | {test_dates.min()} -> {test_dates.max()} | "
            f"Positive rate: {test_pos_rate:.4f}"
        )

    # ── Step 6: Cross-validation on TRAIN only ───────────────────────────────
    # CV folds are all drawn from X_train (the oldest 70% of data) — none
    # of them ever see the OOS test window above.
    cv = TimeSeriesSplit(n_splits=5, gap=SAFETY_GAP)

    cv_auc_roc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
    cv_pr_auc  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="average_precision", n_jobs=-1)

    precision_scorer = make_scorer(precision_score, zero_division=0)
    cv_precision = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                    scoring=precision_scorer, n_jobs=-1)

    logger.info(
        f"Cross-validation | "
        f"AUC-ROC:   {cv_auc_roc.mean():.4f} +/- {cv_auc_roc.std():.4f} | "
        f"PR-AUC:    {cv_pr_auc.mean():.4f} +/- {cv_pr_auc.std():.4f} | "
        f"Precision: {cv_precision.mean():.4f} +/- {cv_precision.std():.4f}"
    )

    # ── Step 7: Train FINAL model on TRAIN only ─────────────────────────────
    logger.info("Training final model on TRAIN set only...")
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.65).astype(int)
    logger.info(
        f"OOS Probability Spread | "
        f"Max: {y_pred_proba.max():.4f} | Mean: {y_pred_proba.mean():.4f}"
    )

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc_roc   = roc_auc_score(y_test, y_pred_proba)
    pr_auc    = average_precision_score(y_test, y_pred_proba)

    logger.info(
        f"OOS Test Metrics | Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}"
    )

    # ── Precision at top 5% of signals — a real-trading-relevant metric ─────
    results_df = pd.DataFrame({
        "true_label" : y_test,
        "probability": y_pred_proba,
    }).sort_values("probability", ascending=False)

    top_5_percent_cutoff = max(1, int(len(results_df) * 0.05))
    top_signals          = results_df.head(top_5_percent_cutoff)
    top_precision        = top_signals["true_label"].mean()

    logger.info(
        f"Real Trading Metrics | "
        f"Win Rate of Top 5% Signals: {top_precision:.4f} "
        f"(Baseline: {y_test.mean():.4f})"
    )

    metrics = {
        "model_name"        : "signal_ranker",
        "train_date"        : datetime.today().strftime("%Y-%m-%d"),
        "precision"         : round(precision, 4),
        "recall"            : round(recall, 4),
        "f1"                : round(f1, 4),
        "auc_roc"           : round(auc_roc, 4),
        "pr_auc"            : round(pr_auc, 4),
        "cv_auc_mean"       : round(cv_auc_roc.mean(), 4),
        "cv_auc_std"        : round(cv_auc_roc.std(), 4),
        "cv_pr_mean"        : round(cv_pr_auc.mean(), 4),
        "cv_pr_std"         : round(cv_pr_auc.std(), 4),
        "cv_precision_mean" : round(cv_precision.mean(), 4),
        "cv_precision_std"  : round(cv_precision.std(), 4),
        "n_train"           : len(X_train),
        "n_test"            : len(X_test),
    }

    logger.info(
        f"Final metrics | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"PR-AUC: {pr_auc:.4f} | "
        f"AUC-ROC: {auc_roc:.4f} | "
        f"CV AUC-ROC: {cv_auc_roc.mean():.4f} +/- {cv_auc_roc.std():.4f}"
    )

    # ── Step 9: Save model to disk ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Signal Ranker saved to {MODEL_PATH}")
    logger.info("=" * 60)
    logger.info("SIGNAL RANKER TRAINING COMPLETE")
    logger.info("=" * 60)

    return pipeline, metrics


# =============================================================================
# INFERENCE
# =============================================================================

def load_signal_ranker() -> Optional[Pipeline]:
    """
    Load the trained Signal Ranker from disk.

    Returns:
        Trained Pipeline or None if model not found
    """
    if not MODEL_PATH.exists():
        logger.warning(
            f"Signal Ranker model not found at {MODEL_PATH}. "
            f"Train the model first."
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    logger.info(f"Signal Ranker loaded from {MODEL_PATH}")
    return pipeline


def score_candidates(
    candidates_df : pd.DataFrame,
    prices_df     : pd.DataFrame,
    indicators_df : pd.DataFrame,
    csi_df        : pd.DataFrame,
    signal_datetime,
    pipeline      : Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Score all scanner candidates and rank them by probability.

    FLOW:
    1. Load model if not provided
    2. For each candidate (pair + direction):
       a. Compute full signal feature vector (CSI looked up from the
          pre-computed universe-wide snapshot passed in, not
          recomputed per pair)
       b. Run predict_proba() -> success probability
       c. Tag as high/normal probability
    3. Sort by ml_score descending, within each direction
    4. Assign ml_rank (1 = best)
    5. Return updated candidates DataFrame

    If model is not available yet (first run before training):
    - All candidates get ml_score = 0.5 (neutral)
    - Ranked by SD position instead

    Args:
        candidates_df  : Output of run_scanner() — unranked candidates
                         [pair, direction, ...]
        prices_df      : Full OHLC data, all pairs
        indicators_df  : Full indicator results, all pairs
        csi_df         : Output of run_csi_engine() for this
                         signal_datetime — computed ONCE for the whole
                         universe, not recomputed per candidate (CSI is
                         inherently cross-pair, see engines/csi.py)
        signal_datetime: Timestamp of this scan run
        pipeline       : Optional pre-loaded model

    Returns:
        candidates_df with ml_score and ml_rank columns filled
    """
    if pipeline is None:
        pipeline = load_signal_ranker()

    if pipeline is None:
        logger.warning(
            "Signal Ranker not available — "
            "using SD position as preliminary ranking"
        )
        candidates_df = candidates_df.copy()
        candidates_df["ml_score"] = 0.5
        candidates_df = candidates_df.sort_values(
            "sd_position",
            key=lambda x: x.abs(),
            ascending=True,
        )
        candidates_df["ml_rank"] = range(1, len(candidates_df) + 1)
        return candidates_df

    results = []

    for _, row in candidates_df.iterrows():
        pair      = row["pair"]
        direction = row["direction"]

        px = prices_df[
            prices_df["pair"] == pair
        ].sort_values("datetime")

        try:
            features = compute_signal_features(
                pair            = pair,
                signal_datetime = signal_datetime,
                direction       = direction,
                prices_df       = px,
                indicators_df   = indicators_df,
                csi_df          = csi_df,
            )

            if features is None:
                ml_score = 0.5
            else:
                X        = pd.DataFrame([features])[SIGNAL_FEATURE_COLS]
                ml_score = float(pipeline.predict_proba(X)[0][1])

        except Exception as e:
            logger.warning(f"{pair} | Signal scoring failed: {e}")
            ml_score = 0.5

        result_row = row.to_dict()
        result_row["ml_score"] = round(ml_score, 4)
        results.append(result_row)

    result_df = pd.DataFrame(results)

    # ── Sort by ml_score descending within each direction ─────────────────────
    longs = result_df[result_df["direction"] == "long"].sort_values(
        "ml_score", ascending=False
    ).reset_index(drop=True)

    shorts = result_df[result_df["direction"] == "short"].sort_values(
        "ml_score", ascending=False
    ).reset_index(drop=True)

    longs["ml_rank"]  = longs.index + 1
    shorts["ml_rank"] = shorts.index + 1

    final = pd.concat([longs, shorts], ignore_index=True)

    high_prob = final[final["ml_score"] >= HIGH_PROB_THRESHOLD]
    logger.info(
        f"Signal Ranker scoring complete | "
        f"Total candidates: {len(final)} | "
        f"High probability (>={HIGH_PROB_THRESHOLD}): {len(high_prob)}"
    )

    if not high_prob.empty:
        logger.info(
            f"Top candidates:\n"
            f"{high_prob[['pair','direction','ml_score','ml_rank']].to_string(index=False)}"
        )

    return final
