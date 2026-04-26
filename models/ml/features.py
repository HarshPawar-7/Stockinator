"""
Feature Engineering Pipeline

Transforms raw fundamentals + model outputs into ML-ready features.
All features use only data available at prediction time (no lookahead bias).

Reference: stock_valuation_ml_reference.md §5.1
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature definitions
FEATURE_COLUMNS = [
    # Valuation model outputs (ensemble inputs)
    "ggm_value", "dcf_value", "comps_value", "rim_value",
    "model_disagreement",

    # Fundamentals
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa",
    "debt_to_equity", "gross_margin", "operating_margin", "net_margin",
    "revenue_growth_yoy", "payout_ratio", "dividend_yield",

    # Price-based
    "price_to_ensemble",

    # Macro
    "risk_free_rate",
]

TARGET_COLUMN = "price_12m_vs_iv"


def build_features_from_result(result: dict) -> dict:
    """
    Extract ML features from a single valuation result dict.

    Args:
        result: Output from valuate_single_stock()

    Returns:
        Dict of feature name → value
    """
    models = result.get("models", {})
    ensemble = result.get("ensemble", {})
    inputs = result.get("inputs", {})
    price = result.get("market_price")

    ggm_val = models.get("ggm", {}).get("value")
    dcf_val = models.get("dcf", {}).get("value")
    comps_val = models.get("comps", {}).get("value")
    rim_val = models.get("rim", {}).get("value")

    # Model disagreement
    valid_vals = [v for v in [ggm_val, dcf_val, comps_val, rim_val] if v and v > 0]
    if len(valid_vals) >= 2:
        disagreement = float(np.std(valid_vals) / np.mean(valid_vals))
    else:
        disagreement = None

    ensemble_val = ensemble.get("ensemble_value")
    price_to_ens = None
    if price and ensemble_val and ensemble_val > 0:
        price_to_ens = price / ensemble_val

    return {
        "ticker": result.get("ticker"),
        "date": result.get("valuation_date"),
        "price": price,

        # Model outputs
        "ggm_value": ggm_val,
        "dcf_value": dcf_val,
        "comps_value": comps_val,
        "rim_value": rim_val,
        "ensemble_value": ensemble_val,
        "model_disagreement": disagreement,

        # Price-based
        "price_to_ensemble": price_to_ens,
        "margin_of_safety": ensemble.get("margin_of_safety"),
        "signal": ensemble.get("signal"),

        # Macro
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
        DataFrame with one row per stock, columns = features
    """
    rows = [build_features_from_result(r) for r in results]
    df = pd.DataFrame(rows)

    logger.info("Feature matrix: %d rows × %d columns", len(df), len(df.columns))
    logger.info("Missing values per column:\n%s", df.isnull().sum().to_string())

    return df


def build_training_dataset(
    historical_results: list[dict],
    future_prices: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset with labels.

    Label = Future_Price_12M / Current_IV_Estimate
    Value < 1: overvalued signal confirmed
    Value > 1: undervalued signal confirmed

    Args:
        historical_results: Past valuation results
        future_prices: Dict of ticker → price 12 months later

    Returns:
        (X features DataFrame, y target Series)
    """
    df = build_feature_matrix(historical_results)

    # Compute target
    df["future_price"] = df["ticker"].map(future_prices)
    df[TARGET_COLUMN] = None

    mask = (df["ensemble_value"].notna()) & (df["ensemble_value"] > 0) & (df["future_price"].notna())
    df.loc[mask, TARGET_COLUMN] = df.loc[mask, "future_price"] / df.loc[mask, "ensemble_value"]

    # Drop rows without target
    df_valid = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Select feature columns (only those that exist)
    available = [c for c in FEATURE_COLUMNS if c in df_valid.columns]
    X = df_valid[available]
    y = df_valid[TARGET_COLUMN].astype(float)

    logger.info("Training set: %d samples, %d features", len(X), len(X.columns))

    return X, y
