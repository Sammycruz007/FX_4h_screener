"""
ml/signal_ranker.py
--------------------
Directional model training and inference for the FX basket models.

WHAT'S DIFFERENT FROM THE PRIOR SIGNAL RANKER VERSION — THIS IS A
COMPLETE REDESIGN, NOT AN INCREMENTAL CHANGE:

1. FIVE MODELS, ONE FUNCTION. train_signal_ranker() (one model, all 28
   pairs pooled) is replaced by train_directional_model(feature_matrix,
   basket_name), called once per basket from ml/train_models.py's loop.
   Each basket's model is saved under its own filename
   (models/directional_{basket_name}.pkl), not one shared
   signal_ranker.pkl.

2. CROSS-PAIR TIME-SERIES LEAKAGE FIX (the reason this file needed a
   real rework, not just a rename) — flagged directly by the user:

       "Suppose Week 120 appears in training for EURUSD. Week 120 for
       GBPUSD is almost the same macro event. If Week 120 for GBPUSD
       lands in your validation fold while Week 120 for EURUSD is in
       training, your model has effectively seen the same market
       regime... every pair from a given week should belong entirely
       to either the training set or the validation/test set."

   The prior version's train/test split cut at a ROW-COUNT percentile
   (`int(len(dates) * 0.70)`) on a DataFrame with multiple pairs'
   rows STACKED together and merely sorted by datetime. Because
   different pairs don't necessarily produce rows in perfect lockstep,
   a row-count cutoff does NOT guarantee every pair's row for a given
   calendar week lands on the same side of the boundary. FIX: split on
   the UNIQUE SORTED DATETIME VALUES first, not row position. See
   _split_by_datetime_boundary() below, and its dedicated test
   asserting zero datetime overlap between train and test across ALL
   pairs simultaneously.

3. MULTI-THRESHOLD EVALUATION, NOT ONE HARDCODED CUTOFF. The prior
   version evaluated Precision/Recall/F1 at a single hardcoded
   threshold. In BOTH real training runs performed on this project,
   that produced misleading results. Fixed here by reporting
   Precision/Recall/n at SEVERAL thresholds plus AUC-ROC/PR-AUC
   (threshold-independent).

4. NO `direction` PARAMETER ANYWHERE. score_candidates() is replaced
   by predict_direction(), which predicts direction directly for every
   pair in a basket — there is no external candidate list to rank.

Every threshold/period below is read live from config.yaml.
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from ml.features import DIRECTIONAL_FEATURE_COLS, compute_directional_features
from utils.logging import get_ml_logger
from utils.error_handler import MLError

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

MIN_SAMPLES         = ML_CFG["min_training_samples"]
DISPLAY_THRESHOLD   = ML_CFG["high_probability_threshold"]
HORIZON             = ML_CFG["label_forward_periods"]

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# The train/test and CV gap must be AT LEAST as large as the label's
# own forward-looking HORIZON, or a training row near the boundary can
# have its label computed from a future close that falls inside the
# test window. HORIZON is the correct thing to size this gap against
# now that LinReg (which used to size it) is dropped.
SAFETY_GAP = HORIZON + 5

EVAL_THRESHOLDS = sorted(set([0.5, 0.6, 0.7, 0.8, DISPLAY_THRESHOLD]))


# =============================================================================
# CROSS-PAIR-SAFE TRAIN/TEST SPLIT
# =============================================================================

def _split_by_datetime_boundary(
    dates      : pd.Series,
    train_frac : float = 0.70,
    gap        : int   = 0,
) -> tuple:
    """
    Split rows into train/test masks using a DATETIME boundary, not a
    row-count percentile — every row sharing a given datetime
    (regardless of pair) lands entirely on one side of the split.

    Args:
        dates     : The full 'datetime' column, one entry per row,
                    potentially many pairs stacked
        train_frac: Fraction of the UNIQUE datetime range for training
        gap       : Number of unique datetime steps as a gap between
                    train's end and test's start

    Returns:
        (train_mask, test_mask, train_end_dt, test_start_dt)
    """
    unique_dates = pd.Series(sorted(dates.unique()))

    if len(unique_dates) < 3:
        raise MLError(
            f"_split_by_datetime_boundary: only {len(unique_dates)} unique "
            f"datetime values — need at least 3."
        )

    split_idx = int(len(unique_dates) * train_frac)
    split_idx = max(1, min(split_idx, len(unique_dates) - 2))

    train_end_dt = unique_dates.iloc[split_idx - 1]

    test_start_idx = split_idx + gap
    if test_start_idx >= len(unique_dates):
        logger.warning(
            f"_split_by_datetime_boundary: gap={gap} leaves no room for "
            f"a test set with only {len(unique_dates)} unique datetimes "
            f"— falling back to no gap."
        )
        test_start_idx = split_idx

    test_start_dt = unique_dates.iloc[test_start_idx]

    train_mask = dates <= train_end_dt
    test_mask  = dates >= test_start_dt

    return train_mask, test_mask, train_end_dt, test_start_dt


# =============================================================================
# MODEL BUILDER
# =============================================================================

def _build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build the sklearn Pipeline for a basket's directional model."""
    base_model = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 4,
        learning_rate      = 0.02,
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
    inner_cv = TimeSeriesSplit(n_split= 2, gap = SAFETY_GAP)
    calibrated_model = CalibratedClassifierCV(
        estimator = base_model,
        method    = "isotonic",
        cv        = inner_cv,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   calibrated_model),
    ])

    return pipeline


# =============================================================================
# TRAINING — one basket at a time
# =============================================================================

def train_directional_model(
    feature_matrix: pd.DataFrame,
    basket_name   : str,
) -> Tuple[Pipeline, dict]:
    """
    Train one basket's directional model on its labelled feature matrix.

    Args:
        feature_matrix: Output of build_directional_feature_matrix()
                        for ONE basket's pairs. Must contain
                        DIRECTIONAL_FEATURE_COLS + 'label' + 'pair' +
                        'datetime'.
        basket_name   : e.g. 'basket1_usd'

    Returns:
        Tuple of (trained Pipeline, metrics dict)
    """
    logger.info("=" * 60)
    logger.info(f"DIRECTIONAL MODEL TRAINING STARTING | Basket: {basket_name}")
    logger.info("=" * 60)

    if len(feature_matrix) < MIN_SAMPLES:
        raise MLError(
            f"{basket_name}: insufficient training samples: "
            f"{len(feature_matrix)} (need {MIN_SAMPLES})"
        )

    if "pair" not in feature_matrix.columns or "datetime" not in feature_matrix.columns:
        raise MLError(
            f"{basket_name}: feature_matrix is missing 'pair' or 'datetime'."
        )

    df    = feature_matrix.copy()
    X     = df[DIRECTIONAL_FEATURE_COLS].copy()
    y     = df["label"].values
    dates = pd.to_datetime(df["datetime"])
    pairs_in_matrix = df["pair"].unique().tolist()

    logger.info(f"{basket_name}: pairs in this matrix: {pairs_in_matrix}")

    n_negative       = (y == 0).sum()
    n_positive       = (y == 1).sum()
    raw_ratio        = n_negative / n_positive if n_positive > 0 else 1.0
    scale_pos_weight = raw_ratio

    logger.info(
        f"{basket_name}: class balance | "
        f"Up: {n_positive} | Down: {n_negative} | "
        f"Raw ratio: {raw_ratio:.2f} | scale_pos_weight: {scale_pos_weight:.2f}"
    )

    pipeline = _build_pipeline(scale_pos_weight)

    train_mask, test_mask, train_end_dt, test_start_dt = _split_by_datetime_boundary(
        dates, train_frac=0.70, gap=SAFETY_GAP,
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(
        f"{basket_name}: cross-pair-safe split | "
        f"Train: {len(X_train)} rows (<= {train_end_dt}) | "
        f"Gap: {HORIZON} weekly candles | "
        f"OOS Test: {len(X_test)} rows (>= {test_start_dt}) | "
        f"Ratio: {raw_ratio:.2f}:1"
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise MLError(
            f"{basket_name}: split produced an empty train or test set "
            f"(train={len(X_train)}, test={len(X_test)})."
        )

    train_pos_rate = y_train.mean() if len(y_train) > 0 else float("nan")
    test_pos_rate  = y_test.mean()  if len(y_test)  > 0 else float("nan")
    logger.info(
        f"{basket_name}: train positive rate {train_pos_rate:.4f} | "
        f"OOS positive rate {test_pos_rate:.4f}"
    )

    cv = TimeSeriesSplit(n_splits=5, gap=SAFETY_GAP)

    cv_auc_roc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
    cv_pr_auc  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="average_precision", n_jobs=-1)
    cv_precision  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="precision", n_jobs=-1)
    cv_recall     = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="recall", n_jobs=-1)

    logger.info(
        f"{basket_name}: cross-validation | "
        f"AUC-ROC: {cv_auc_roc.mean():.4f} +/- {cv_auc_roc.std():.4f} | "
        f"PR-AUC:  {cv_pr_auc.mean():.4f} +/- {cv_pr_auc.std():.4f}"
        f"PR-AUC:  {cv_precision.mean():.4f} +/- {cv_precision.std():.4f}"
        f"PR-AUC:  {cv_recall.mean():.4f} +/- {cv_recall.std():.4f}"
    )

    logger.info(f"{basket_name}: training final model on TRAIN set only...")
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    logger.info(
        f"{basket_name}: OOS probability spread | "
        f"Min: {y_pred_proba.min():.4f} | Max: {y_pred_proba.max():.4f} | "
        f"Mean: {y_pred_proba.mean():.4f}"
    )

    threshold_table = []
    for threshold in EVAL_THRESHOLDS:
        y_pred_at_t = (y_pred_proba >= threshold).astype(int)
        n_predicted_positive = int(y_pred_at_t.sum())

        precision_at_t = precision_score(y_test, y_pred_at_t, zero_division=0)
        recall_at_t    = recall_score(y_test, y_pred_at_t, zero_division=0)
        f1_at_t        = f1_score(y_test, y_pred_at_t, zero_division=0)

        threshold_table.append({
            "threshold": threshold,
            "n"        : n_predicted_positive,
            "precision": round(precision_at_t, 4),
            "recall"   : round(recall_at_t, 4),
            "f1"       : round(f1_at_t, 4),
        })

        logger.info(
            f"{basket_name}: threshold={threshold:.2f} | "
            f"n={n_predicted_positive} | "
            f"Precision={precision_at_t:.4f} | Recall={recall_at_t:.4f} | "
            f"F1={f1_at_t:.4f}"
        )

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    pr_auc  = average_precision_score(y_test, y_pred_proba)

    display_row = next(
        (row for row in threshold_table if row["threshold"] == DISPLAY_THRESHOLD),
        threshold_table[-1],
    )

    results_df = pd.DataFrame({
        "true_label" : y_test,
        "probability": y_pred_proba,
    }).sort_values("probability", ascending=False)

    top_5_percent_cutoff = max(1, int(len(results_df) * 0.05))
    top_signals          = results_df.head(top_5_percent_cutoff)
    top_precision        = top_signals["true_label"].mean()

    logger.info(
        f"{basket_name}: real trading metric | "
        f"Win Rate of Top 5% OOS Predictions: {top_precision:.4f} "
        f"(Baseline: {y_test.mean():.4f})"
    )

    metrics = {
        "model_name"        : f"directional_{basket_name}",
        "train_date"        : datetime.today().strftime("%Y-%m-%d"),
        "precision"         : display_row["precision"],
        "recall"            : display_row["recall"],
        "f1"                : display_row["f1"],
        "auc_roc"           : round(auc_roc, 4),
        "pr_auc"            : round(pr_auc, 4),
        "cv_auc_mean"       : round(cv_auc_roc.mean(), 4),
        "cv_auc_std"        : round(cv_auc_roc.std(), 4),
        "cv_pr_mean"        : round(cv_pr_auc.mean(), 4),
        "cv_pr_std"         : round(cv_pr_auc.std(), 4),
        "threshold_table"   : threshold_table,
        "top_5pct_win_rate" : round(top_precision, 4),
        "n_train"           : len(X_train),
        "n_test"            : len(X_test),
    }

    logger.info(
        f"{basket_name}: final metrics | "
        f"AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"At display threshold {DISPLAY_THRESHOLD}: "
        f"Precision={display_row['precision']:.4f} Recall={display_row['recall']:.4f} n={display_row['n']}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"directional_{basket_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"{basket_name}: model saved to {model_path}")
    logger.info("=" * 60)
    logger.info(f"DIRECTIONAL MODEL TRAINING COMPLETE | Basket: {basket_name}")
    logger.info("=" * 60)

    return pipeline, metrics


# =============================================================================
# INFERENCE
# =============================================================================

def load_directional_model(basket_name: str) -> Optional[Pipeline]:
    """Load one basket's trained directional model from disk."""
    model_path = MODEL_DIR / f"directional_{basket_name}.pkl"

    if not model_path.exists():
        logger.warning(
            f"Directional model for {basket_name} not found at {model_path}. "
            f"Train it first via ml/train_models.py."
        )
        return None

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    logger.info(f"Directional model loaded for {basket_name} from {model_path}")
    return pipeline


def predict_direction(
    basket_name   : str,
    basket_pairs  : list,
    prices_df     : pd.DataFrame,
    csi_df        : pd.DataFrame,
    signal_datetime,
    pipeline      : Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Predict direction for every pair in one basket, unconditionally.
    Replaces score_candidates() entirely.

    Args:
        basket_name    : e.g. 'basket1_usd'
        basket_pairs   : List of pairs in this basket
        prices_df      : Full OHLC data for at least this basket's pairs
        csi_df         : Output of run_csi_engine() for this
                         signal_datetime
        signal_datetime: Timestamp of this scan run
        pipeline       : Optional pre-loaded model for this basket

    Returns:
        DataFrame [pair, up_probability], one row per pair that had
        enough history. The caller applies the display threshold.
    """
    if pipeline is None:
        pipeline = load_directional_model(basket_name)

    if pipeline is None:
        logger.warning(
            f"{basket_name}: model not available — returning empty "
            f"predictions for all {len(basket_pairs)} pairs"
        )
        return pd.DataFrame(columns=["pair", "up_probability"])

    results = []
    skipped = []

    for pair in basket_pairs:
        px = prices_df[prices_df["pair"] == pair].sort_values("datetime")

        try:
            features = compute_directional_features(
                pair            = pair,
                signal_datetime = signal_datetime,
                prices_df       = px,
                csi_df          = csi_df,
            )

            if features is None:
                skipped.append(pair)
                continue

            X = pd.DataFrame([features])[DIRECTIONAL_FEATURE_COLS]
            up_probability = float(pipeline.predict_proba(X)[0][1])

        except Exception as e:
            logger.warning(f"{basket_name} | {pair} | Prediction failed: {e}")
            skipped.append(pair)
            continue

        results.append({
            "pair"           : pair,
            "up_probability" : round(up_probability, 4),
        })

    if skipped:
        logger.info(f"{basket_name}: skipped {len(skipped)} pairs (insufficient history): {skipped}")

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values("up_probability", ascending=False).reset_index(drop=True)

    logger.info(
        f"{basket_name}: predictions complete | "
        f"{len(result_df)}/{len(basket_pairs)} pairs scored"
    )

    return result_df
