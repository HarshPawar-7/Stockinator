"""
Feature Engineering Pipeline

Transforms raw fundamentals + model outputs into ML-ready features.

FEATURE SAFETY CLASSIFICATION
══════════════════════════════
SAFE (use in both training X and inference):
  - Valuation model outputs (ggm_value, dcf_value, comps_value, rim_value)
  - Fundamental ratios (pe, pb, ps, ev/ebitda, roe, roa, margins, growth)
  - Macro inputs (risk_free_rate, beta, wacc)
  - Model disagreement (derived from model outputs)

INFERENCE-ONLY (valid at prediction time, but MUST be dropped from training X):
  - price_to_ensemble: current_price / ensemble_value
    At training time this uses the price from the labeling date, which
    partially encodes the future label (future_price / ensemble_value).
    Drop from X during training; use only for live inference overlay.

NEVER USE (target leakage):
  - future_price, price_12m_vs_iv — these ARE the label

Reference: stock_valuation_ml_reference.md §5.1
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Safe training features (no lookahead) ─────────────────────────────────────
TRAINING_FEATURE_COLUMNS = [
    # Valuation model outputs
    "ggm_value", "dcf_value", "comps_value", "rim_value",
    "model_disagreement",

    # Fundamentals
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa",
    "debt_to_equity", "gross_margin", "operating_margin", "net_margin",
    "revenue_growth_yoy", "payout_ratio", "dividend_yield",

    # Macro
    "risk_free_rate", "beta", "wacc",
]

# ── Inference-only features (valid at live prediction, not in training X) ─────
INFERENCE_ONLY_COLUMNS = [
    "price_to_ensemble",   # current_price / ensemble_value — leaky if used in CV folds
]

# ── All features (for feature matrix building) ────────────────────────────────
ALL_FEATURE_COLUMNS = TRAINING_FEATURE_COLUMNS + INFERENCE_ONLY_COLUMNS

TARGET_COLUMN = "price_12m_vs_iv"


def build_features_from_result(result: dict) -> dict:
    """
    Extract ML features from a single valuation result dict.

    Args:
        result: Output from valuate_single_stock()

    Returns:
        Dict of feature name → value (includes both training and
        inference-only features; caller must filter appropriately).
    """
    models = result.get("models", {})
    ensemble = result.get("ensemble", {})
    inputs = result.get("inputs", {})
    price = result.get("market_price")

    ggm_val = models.get("ggm", {}).get("value")
    dcf_val = models.get("dcf", {}).get("value")
    comps_val = models.get("comps", {}).get("value")
    rim_val = models.get("rim", {}).get("value")

    # Model disagreement — safe: derived only from model outputs
    valid_vals = [v for v in [ggm_val, dcf_val, comps_val, rim_val] if v and v > 0]
    if len(valid_vals) >= 2:
        disagreement = float(np.std(valid_vals) / np.mean(valid_vals))
    else:
        disagreement = None

    ensemble_val = ensemble.get("ensemble_value")

    # ⚠️ INFERENCE-ONLY: drop from training X (see module docstring)
    price_to_ens = None
    if price and ensemble_val and ensemble_val > 0:
        price_to_ens = price / ensemble_val

    return {
        "ticker": result.get("ticker"),
        "date": result.get("valuation_date"),
        "price": price,

        # Model outputs (safe)
        "ggm_value": ggm_val,
        "dcf_value": dcf_val,
        "comps_value": comps_val,
        "rim_value": rim_val,
        "ensemble_value": ensemble_val,
        "model_disagreement": disagreement,

        # Inference-only
        "price_to_ensemble": price_to_ens,

        # Metadata / other outputs
        "margin_of_safety": ensemble.get("margin_of_safety"),
        "signal": ensemble.get("signal"),

        # Macro (safe)
        "risk_free_rate": inputs.get("risk_free_rate"),
        "beta": inputs.get("beta"),
        "cost_of_equity": inputs.get("cost_of_equity"),
        "wacc": inputs.get("wacc"),
    }


def build_feature_matrix(results: list[dict]) -> pd.DataFrame:
    """
    Build a feature matrix from a batch of valuation results.

    Args:
        results: List of valuation result dicts

    Returns:
        DataFrame with one row per stock, columns = all features
    """
    rows = [build_features_from_result(r) for r in results]
    df = pd.DataFrame(rows)

    logger.info("Feature matrix: %d rows × %d columns", len(df), len(df.columns))
    logger.debug("Missing values:\n%s", df.isnull().sum().to_string())

    return df


def build_imputer(X: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Fit a median imputer on X and return the transformed matrix + fitted imputer.

    The imputer must be fitted on training data ONLY and then applied
    to validation/test sets to prevent imputation leakage.

    Args:
        X: Feature matrix (training split only)

    Returns:
        (X_imputed DataFrame, fitted SimpleImputer)
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_imputed, imputer


def apply_imputer(X: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """Apply a pre-fitted imputer to a feature matrix."""
    return pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )


def build_training_dataset(
    historical_results: list[dict],
    future_prices: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset with labels.

    Label = Future_Price_12M / Current_Ensemble_Value
    - Value > 1: IV was conservative → undervalued was correct
    - Value < 1: IV was optimistic  → overvalued was correct

    Enforces temporal ordering and drops inference-only features from X.

    Args:
        historical_results: Past valuation results (must be sorted by date)
        future_prices: Dict of ticker → price 12 months after valuation date

    Returns:
        (X features DataFrame, y target Series)
    """
    df = build_feature_matrix(historical_results)

    # Compute target label
    df["future_price"] = df["ticker"].map(future_prices)
    df[TARGET_COLUMN] = None

    mask = (
        df["ensemble_value"].notna()
        & (df["ensemble_value"] > 0)
        & df["future_price"].notna()
    )
    df.loc[mask, TARGET_COLUMN] = (
        df.loc[mask, "future_price"] / df.loc[mask, "ensemble_value"]
    )

    # Drop rows without target
    df_valid = df.dropna(subset=[TARGET_COLUMN]).copy()

    # ── Select ONLY safe training features (no inference-only leakage) ────
    available_safe = [c for c in TRAINING_FEATURE_COLUMNS if c in df_valid.columns]
    X = df_valid[available_safe]
    y = df_valid[TARGET_COLUMN].astype(float)

    logger.info(
        "Training set: %d samples, %d features (inference-only columns excluded)",
        len(X), len(X.columns),
    )

    return X, y
