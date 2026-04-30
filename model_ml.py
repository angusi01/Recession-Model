"""
model_ml.py — ML-based recession forecasting with true walk-forward validation.

Architecture:
  1. Walk-forward backtest: train on [start → t], predict at t+1
     (re-trains every 6 months for speed; predicts every month)
  2. Final models: trained on ALL available labelled data
  3. Calibration: isotonic regression applied to walk-forward OOS predictions
  4. Current forecast: final models + calibrators → P(recession_3m), P(recession_6m)

Models:
  - Logistic Regression  (interpretable baseline, L2 regularised, class_weight='balanced')
  - Gradient Boosting    (nonlinear, shallow trees to avoid overfit)
  - Ensemble: simple average of LR and GB calibrated probabilities
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_TRAIN_MONTHS = 60        # 5 years minimum training window
RETRAIN_INTERVAL = 6         # re-train every N months during walk-forward backtest
GB_PARAMS = dict(
    n_estimators=60,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42,
)
LR_PARAMS = dict(
    C=0.5,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])


def _make_gb_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(**GB_PARAMS)),
    ])


# ── Walk-forward validation ───────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def walk_forward_predict(
    features: pd.DataFrame,
    y_3m: pd.Series,
    y_6m: pd.Series,
    min_train: int = MIN_TRAIN_MONTHS,
    retrain_every: int = RETRAIN_INTERVAL,
) -> pd.DataFrame:
    """
    True expanding-window walk-forward validation.

    For each test point t (starting at min_train):
      - Trains LR and GB on data [0 … t-1]
      - Predicts probability at t
      - Re-trains every `retrain_every` months for speed

    Returns a DataFrame indexed by the feature matrix's PeriodIndex with columns:
      y_3m, y_6m, p_lr_3m, p_gb_3m, p_lr_6m, p_gb_6m
    """
    n = len(features)
    if n < min_train + 1:
        logger.warning(f"Insufficient data for walk-forward (need {min_train+1}, got {n})")
        return pd.DataFrame()

    X = features.values.astype(float)
    y3 = y_3m.values.astype(int)
    y6 = y_6m.values.astype(int)

    records = []

    # Cache models to avoid re-fitting every single month
    lr3, gb3, lr6, gb6 = None, None, None, None
    last_train_t = -999

    for t in range(min_train, n):
        should_retrain = (t - last_train_t >= retrain_every) or (lr3 is None)

        X_train = X[:t]
        y3_train = y3[:t]
        y6_train = y6[:t]

        # Only train if we have at least one positive example per target
        can_train_3m = y3_train.sum() >= 1
        can_train_6m = y6_train.sum() >= 1

        if should_retrain:
            if can_train_3m:
                try:
                    lr3 = _make_lr_pipeline()
                    lr3.fit(X_train, y3_train)
                    gb3 = _make_gb_pipeline()
                    gb3.fit(X_train, y3_train)
                except Exception as e:
                    logger.debug(f"Walk-forward train error at t={t} (3m): {e}")
                    lr3, gb3 = None, None

            if can_train_6m:
                try:
                    lr6 = _make_lr_pipeline()
                    lr6.fit(X_train, y6_train)
                    gb6 = _make_gb_pipeline()
                    gb6.fit(X_train, y6_train)
                except Exception as e:
                    logger.debug(f"Walk-forward train error at t={t} (6m): {e}")
                    lr6, gb6 = None, None

            last_train_t = t

        X_test = X[t: t + 1]

        def _prob(model):
            if model is None:
                return np.nan
            try:
                return float(model.predict_proba(X_test)[0, 1])
            except Exception:
                return np.nan

        records.append({
            "date": features.index[t],
            "y_3m": int(y3[t]),
            "y_6m": int(y6[t]),
            "p_lr_3m": _prob(lr3),
            "p_gb_3m": _prob(gb3),
            "p_lr_6m": _prob(lr6),
            "p_gb_6m": _prob(gb6),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    # Ensemble = mean of LR and GB; if one model returned NaN (failed), the
    # other model's value is used automatically (pandas skipna=True by default).
    df["p_ens_3m"] = df[["p_lr_3m", "p_gb_3m"]].mean(axis=1)
    df["p_ens_6m"] = df[["p_lr_6m", "p_gb_6m"]].mean(axis=1)
    return df


# ── Isotonic calibration ──────────────────────────────────────────────────────

def calibrate_probabilities(
    wf_df: pd.DataFrame,
    raw_col: str,
    target_col: str,
) -> IsotonicRegression | None:
    """
    Fit an isotonic regression calibrator on walk-forward OOS predictions.
    Returns fitted IsotonicRegression or None if insufficient data.
    """
    valid = wf_df[[raw_col, target_col]].dropna()
    if valid[target_col].sum() < 1 or len(valid) < 10:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(valid[raw_col].values, valid[target_col].values)
    return iso


def _apply_calibrator(
    raw_proba: float | np.ndarray,
    calibrator: IsotonicRegression | None,
) -> float | np.ndarray:
    """Apply calibrator if available, else return raw probability."""
    if calibrator is None:
        return raw_proba
    arr = np.atleast_1d(raw_proba).astype(float)
    if np.isnan(arr).any():
        return raw_proba
    return calibrator.predict(arr)


# ── Final model training ──────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def train_final_models(
    features: pd.DataFrame,
    y_3m: pd.Series,
    y_6m: pd.Series,
    wf_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Train final LR and GB models on all available labelled data.
    Also fits isotonic calibrators from the walk-forward OOS predictions.

    Returns a dict with keys:
      lr_3m, gb_3m, lr_6m, gb_6m  — fitted sklearn Pipelines
      cal_lr_3m, cal_gb_3m, cal_lr_6m, cal_gb_6m  — isotonic calibrators (or None)
      feature_names  — list of feature column names
    """
    X = features.values.astype(float)
    y3 = y_3m.values.astype(int)
    y6 = y_6m.values.astype(int)

    models: dict[str, Any] = {"feature_names": list(features.columns)}

    for label, y, suffix in [("3m", y3, "_3m"), ("6m", y6, "_6m")]:
        if y.sum() < 1:
            logger.warning(f"No positive examples for {label} target — skipping model")
            models[f"lr{suffix}"] = None
            models[f"gb{suffix}"] = None
            continue

        try:
            lr = _make_lr_pipeline()
            lr.fit(X, y)
            models[f"lr{suffix}"] = lr
        except Exception as e:
            logger.error(f"LR training failed for {label}: {e}")
            models[f"lr{suffix}"] = None

        try:
            gb = _make_gb_pipeline()
            gb.fit(X, y)
            models[f"gb{suffix}"] = gb
        except Exception as e:
            logger.error(f"GB training failed for {label}: {e}")
            models[f"gb{suffix}"] = None

    # Calibrators from walk-forward OOS
    if not wf_df.empty:
        for suffix, raw_col, tgt_col in [
            ("_3m", "p_ens_3m", "y_3m"),
            ("_6m", "p_ens_6m", "y_6m"),
        ]:
            models[f"cal{suffix}"] = calibrate_probabilities(wf_df, raw_col, tgt_col)
    else:
        models["cal_3m"] = None
        models["cal_6m"] = None

    return models


# ── Current forecast ──────────────────────────────────────────────────────────

def get_current_forecast(
    current_row: pd.Series,
    models: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate current 3m and 6m recession probability forecasts.

    Returns:
      p_3m          : calibrated ensemble probability (0–100 %)
      p_6m          : calibrated ensemble probability (0–100 %)
      p_lr_3m       : LR raw probability
      p_gb_3m       : GB raw probability
      p_lr_6m       : LR raw probability
      p_gb_6m       : GB raw probability
      feature_importance : dict of {feature_name: importance}  (from LR coefficients)
    """
    if current_row is None or models.get("lr_3m") is None:
        return {
            "p_3m": None, "p_6m": None,
            "p_lr_3m": None, "p_gb_3m": None,
            "p_lr_6m": None, "p_gb_6m": None,
            "feature_importance": {},
        }

    X_now = current_row.values.reshape(1, -1).astype(float)
    result: dict[str, Any] = {}

    for suffix in ("_3m", "_6m"):
        lr_model = models.get(f"lr{suffix}")
        gb_model = models.get(f"gb{suffix}")
        cal = models.get(f"cal{suffix}")

        lr_p = float(lr_model.predict_proba(X_now)[0, 1]) if lr_model else np.nan
        gb_p = float(gb_model.predict_proba(X_now)[0, 1]) if gb_model else np.nan

        raw_ens = np.nanmean([lr_p, gb_p]) if not (np.isnan(lr_p) and np.isnan(gb_p)) else np.nan

        if cal is not None and not np.isnan(raw_ens):
            cal_p = float(cal.predict([raw_ens])[0])
        else:
            cal_p = raw_ens

        label = suffix[1:]  # "3m" or "6m"
        result[f"p_{label}"] = round(cal_p * 100, 1) if not np.isnan(cal_p) else None
        result[f"p_lr_{label}"] = round(lr_p * 100, 1) if not np.isnan(lr_p) else None
        result[f"p_gb_{label}"] = round(gb_p * 100, 1) if not np.isnan(gb_p) else None

    # Feature importance: LR coefficients (absolute value, 3m model)
    feature_importance = {}
    lr_3m = models.get("lr_3m")
    if lr_3m is not None and hasattr(lr_3m.named_steps.get("clf"), "coef_"):
        coef = lr_3m.named_steps["clf"].coef_[0]
        names = models.get("feature_names", [])
        for name, c in zip(names, coef):
            feature_importance[name] = float(c)  # keep sign for direction

    result["feature_importance"] = feature_importance
    return result


# ── Backtest metrics ──────────────────────────────────────────────────────────

def compute_backtest_metrics(wf_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute ROC-AUC, Brier score, and early-detection metrics from walk-forward OOS results.

    Early detection: for each recession onset month, how many months before onset
    did the ensemble 3m probability first cross 50%?

    Returns a dict with metric names and values (or None if not computable).
    """
    if wf_df.empty:
        return {}

    metrics: dict[str, Any] = {}

    for suffix, p_col, y_col in [("3m", "p_ens_3m", "y_3m"), ("6m", "p_ens_6m", "y_6m")]:
        valid = wf_df[[p_col, y_col]].dropna()
        if valid.empty or valid[y_col].sum() == 0:
            metrics[f"roc_auc_{suffix}"] = None
            metrics[f"brier_{suffix}"] = None
            continue

        try:
            metrics[f"roc_auc_{suffix}"] = round(
                roc_auc_score(valid[y_col], valid[p_col]), 3
            )
        except Exception:
            metrics[f"roc_auc_{suffix}"] = None

        try:
            metrics[f"brier_{suffix}"] = round(
                brier_score_loss(valid[y_col], valid[p_col]), 4
            )
        except Exception:
            metrics[f"brier_{suffix}"] = None

    # Early detection: months before recession onset where p_ens_3m > 0.5
    if "p_ens_3m" in wf_df.columns and "y_3m" in wf_df.columns:
        early_leads = []
        y3 = wf_df["y_3m"]
        prob = wf_df["p_ens_3m"]

        # Find recession onset months (transition from 0 to 1)
        transitions = (y3 == 1) & (y3.shift(1, fill_value=0) == 0)
        onset_dates = y3.index[transitions]

        for onset in onset_dates:
            # Take up to 12 months before onset:
            # tail(13) = onset + 12 prior months; iloc[:-1] drops the onset month itself.
            lookback = wf_df.loc[:onset].iloc[-13:-1]
            above_50 = lookback[lookback["p_ens_3m"] >= 0.5]
            if not above_50.empty:
                # Earliest month with probability >= 50%
                first_signal_pos = list(lookback.index).index(above_50.index[0])
                months_early = len(lookback) - first_signal_pos
                early_leads.append(months_early)

        if early_leads:
            metrics["avg_months_early"] = round(np.mean(early_leads), 1)
            metrics["early_lead_details"] = early_leads
        else:
            metrics["avg_months_early"] = None

    return metrics
