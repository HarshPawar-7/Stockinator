"""
ML Prediction Pipeline

Uses trained ML model to adjust ensemble valuations with
calibrated confidence intervals.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.ml.features import build_features_from_result

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = Path(__file__).parent / "saved_models"


def predict(
    result: dict,
    model_type: str = "xgboost",
) -> dict:
    """
    Use trained ML model to predict price-to-IV ratio.

    The ML model predicts: future_price / current_IV
    - Ratio > 1: model thinks IV is conservative (potential upside)
    - Ratio < 1: model thinks IV is optimistic (potential downside)
    - Ratio ≈ 1: model agrees with IV estimate

    Args:
        result: Valuation result dict from batch_valuation
        model_type: Which saved model to use

    Returns:
        Dict with ML-adjusted valuation and signal
    """
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    if not model_path.exists():
        logger.warning("No trained model found at %s — skipping ML adjustment", model_path)
        return {"ml_available": False}

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Extract features
    features = build_features_from_result(result)
    feature_df = pd.DataFrame([features])

    # Select only numeric columns that model expects
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove non-feature columns
    exclude = {"price", "future_price", "price_12m_vs_iv"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    X = feature_df[numeric_cols].fillna(-999)

    # Predict
    try:
        ratio_pred = float(model.predict(X)[0])
    except Exception as e:
        logger.error("ML prediction failed: %s", e)
        return {"ml_available": False, "error": str(e)}

    # Compute ML-adjusted value
    ensemble_val = result.get("ensemble", {}).get("ensemble_value")
    price = result.get("market_price")

    ml_adjusted_value = None
    ml_signal = None
    if ensemble_val and ensemble_val > 0:
        ml_adjusted_value = ensemble_val * ratio_pred

        if price and price > 0 and ml_adjusted_value > 0:
            ml_mos = (ml_adjusted_value - price) / ml_adjusted_value
            if ml_mos > 0.30:
                ml_signal = "STRONG_BUY"
            elif ml_mos > 0.20:
                ml_signal = "BUY"
            elif ml_mos < -0.30:
                ml_signal = "STRONG_SELL"
            elif ml_mos < -0.20:
                ml_signal = "SELL"
            else:
                ml_signal = "HOLD"

    return {
        "ml_available": True,
        "predicted_ratio": round(ratio_pred, 4),
        "ml_adjusted_value": round(ml_adjusted_value, 2) if ml_adjusted_value else None,
        "ml_signal": ml_signal,
        "model_type": model_type,
    }


def batch_predict(results: list[dict], model_type: str = "xgboost") -> list[dict]:
    """Apply ML predictions to a batch of valuation results."""
    enriched = []
    for r in results:
        ml_output = predict(r, model_type=model_type)
        r["ml"] = ml_output
        enriched.append(r)
    return enriched
