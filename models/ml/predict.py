"""
ML Prediction Pipeline

Uses trained ML model to adjust ensemble valuations with
calibrated confidence intervals.

The ML model predicts: future_price_12m / current_ensemble_value
- Ratio > 1: model thinks IV is conservative (potential upside)
- Ratio < 1: model thinks IV is optimistic (potential downside)
- Ratio ≈ 1: model agrees with IV estimate
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.ml.features import (
    TRAINING_FEATURE_COLUMNS,
    build_features_from_result,
)

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = Path(__file__).parent / "saved_models"


def predict(
    result: dict,
    model_type: str = "xgboost",
) -> dict:
    """
    Use trained ML model to predict price-to-IV ratio.

    Args:
        result: Valuation result dict from batch_valuation
        model_type: Which saved model to use

    Returns:
        Dict with ML-adjusted valuation and signal, or
        {"ml_available": False} if no model is trained yet.
    """
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    if not model_path.exists():
        logger.debug("No trained model found at %s — skipping ML adjustment", model_path)
        return {"ml_available": False}

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load imputer (required to apply the same transformation as training)
    imputer_path = MODEL_SAVE_DIR / f"stockinator_{model_type}_imputer.pkl"
    imputer = None
    if imputer_path.exists():
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)

    # Extract features (full set, including metadata)
    features = build_features_from_result(result)
    feature_df = pd.DataFrame([features])

    # Use ONLY the safe training features in the same order as training
    available = [c for c in TRAINING_FEATURE_COLUMNS if c in feature_df.columns]
    X = feature_df[available]

    # Apply imputer (use training-time imputer if available, else median fill)
    if imputer is not None:
        try:
            X_imp = pd.DataFrame(
                imputer.transform(X),
                columns=X.columns,
                index=X.index,
            )
        except Exception:
            X_imp = X.fillna(X.median())
    else:
        X_imp = X.fillna(X.median())

    # Predict
    try:
        ratio_pred = float(model.predict(X_imp)[0])
    except Exception as e:
        logger.error("ML prediction failed: %s", e)
        return {"ml_available": False, "error": str(e)}

    # Compute ML-adjusted value
    ensemble_val = result.get("ensemble", {}).get("ensemble_value")
    price = result.get("market_price")

    ml_adjusted_value = None
    ml_signal = None
    ml_mos = None

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
        "ml_margin_of_safety": round(ml_mos, 4) if ml_mos is not None else None,
        "ml_signal": ml_signal,
        "model_type": model_type,
        "features_used": len(available),
    }


def batch_predict(results: list[dict], model_type: str = "xgboost") -> list[dict]:
    """Apply ML predictions to a batch of valuation results."""
    enriched = []
    for r in results:
        ml_output = predict(r, model_type=model_type)
        r["ml"] = ml_output
        enriched.append(r)
    return enriched
