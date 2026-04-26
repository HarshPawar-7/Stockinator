"""
ML Training Pipeline

Trains XGBoost/LightGBM ensemble on historical valuation data.
Uses strict temporal train/test split to avoid lookahead bias.

Reference: stock_valuation_ml_reference.md §5.2
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = Path(__file__).parent / "saved_models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_splits: int = 5,
) -> dict:
    """
    Train ML model with temporal cross-validation.

    CRITICAL: Uses TimeSeriesSplit — no random shuffle to prevent lookahead bias.

    Args:
        X: Feature matrix
        y: Target (price_12m / ensemble_value)
        model_type: "xgboost", "lightgbm", or "ridge"
        n_splits: Number of time-series CV folds

    Returns:
        Dict with trained model, metrics, and feature importances
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Select model
    model = _create_model(model_type)

    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Handle NaN
        X_train = X_train.fillna(-999)
        X_val = X_val.fillna(-999)

        if model_type in ("xgboost", "lightgbm"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = {
            "fold": fold,
            "mae": mean_absolute_error(y_val, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
            "r2": r2_score(y_val, y_pred),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }
        fold_metrics.append(metrics)
        logger.info("Fold %d: MAE=%.4f, RMSE=%.4f, R²=%.4f", fold, metrics["mae"], metrics["rmse"], metrics["r2"])

    # Final model: train on all data
    X_full = X.fillna(-999)
    model.fit(X_full, y)

    # Feature importance
    importance = _get_feature_importance(model, X.columns.tolist())

    # Save model
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_path)

    avg_metrics = {
        "avg_mae": np.mean([m["mae"] for m in fold_metrics]),
        "avg_rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "avg_r2": np.mean([m["r2"] for m in fold_metrics]),
    }

    logger.info(
        "Training complete: Avg MAE=%.4f, RMSE=%.4f, R²=%.4f",
        avg_metrics["avg_mae"], avg_metrics["avg_rmse"], avg_metrics["avg_r2"],
    )

    return {
        "model": model,
        "model_type": model_type,
        "model_path": str(model_path),
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "feature_importance": importance,
        "n_features": len(X.columns),
        "n_samples": len(X),
    }


def _create_model(model_type: str):
    """Create ML model instance."""
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42,
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_feature_importance(model, feature_names: list[str]) -> list[dict]:
    """Extract feature importance from trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return []

        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": name, "importance": float(imp)} for name, imp in pairs]
    except Exception:
        return []


def load_model(model_type: str = "xgboost"):
    """Load a saved model from disk."""
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)
