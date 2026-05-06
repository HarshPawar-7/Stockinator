"""
ML Training Pipeline

Trains XGBoost/LightGBM/Ridge ensemble on historical valuation data.
Uses strict temporal TimeSeriesSplit to prevent lookahead bias.
Uses median SimpleImputer (fitted on train only) to prevent imputation leakage.
Optionally logs experiments to MLflow.

Reference: stock_valuation_ml_reference.md §5.2
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from models.ml.features import apply_imputer, build_imputer

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = Path(__file__).parent / "saved_models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_splits: int = 5,
    experiment_name: str = "stockinator_valuation",
) -> dict:
    """
    Train ML model with temporal cross-validation.

    CRITICAL: Uses TimeSeriesSplit — no random shuffle to prevent lookahead bias.
    CRITICAL: Imputer is fitted on train fold only — no imputation leakage.

    Args:
        X: Feature matrix (safe features only, no inference-only columns)
        y: Target (future_price_12m / ensemble_value)
        model_type: "xgboost", "lightgbm", or "ridge"
        n_splits: Number of time-series CV folds
        experiment_name: MLflow experiment name

    Returns:
        Dict with trained model, imputer, metrics, and feature importances
    """
    # ── MLflow Setup (optional) ────────────────────────────────────────────
    mlflow_active = False
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"{model_type}_training")
        mlflow_active = True
        logger.info("MLflow tracking enabled — experiment: %s", experiment_name)
        # Log params
        mlflow.log_params({
            "model_type": model_type,
            "n_splits": n_splits,
            "n_features": X.shape[1],
            "n_samples": len(X),
        })
    except ImportError:
        logger.info("MLflow not installed — skipping experiment tracking. Run: pip install mlflow")

    # ── Temporal Cross-Validation ──────────────────────────────────────────
    model = _create_model(model_type)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Fit imputer on train fold ONLY — prevents leakage across folds
        X_train_imp, fold_imputer = build_imputer(X_train_fold)
        X_val_imp = apply_imputer(X_val_fold, fold_imputer)

        if model_type in ("xgboost", "lightgbm"):
            model.fit(
                X_train_imp, y_train_fold,
                eval_set=[(X_val_imp, y_val_fold)],
                verbose=False,
            )
        else:
            model.fit(X_train_imp, y_train_fold)

        y_pred = model.predict(X_val_imp)

        metrics = {
            "fold": fold,
            "mae": mean_absolute_error(y_val_fold, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_val_fold, y_pred))),
            "r2": r2_score(y_val_fold, y_pred),
            "n_train": len(X_train_fold),
            "n_val": len(X_val_fold),
        }
        fold_metrics.append(metrics)
        logger.info(
            "Fold %d: MAE=%.4f, RMSE=%.4f, R²=%.4f (train=%d, val=%d)",
            fold, metrics["mae"], metrics["rmse"], metrics["r2"],
            metrics["n_train"], metrics["n_val"],
        )
        if mlflow_active:
            mlflow.log_metrics({
                f"fold_{fold}_mae": metrics["mae"],
                f"fold_{fold}_rmse": metrics["rmse"],
                f"fold_{fold}_r2": metrics["r2"],
            })

    # ── Final Model: train on all data ─────────────────────────────────────
    X_full_imp, final_imputer = build_imputer(X)
    if model_type in ("xgboost", "lightgbm"):
        model.fit(X_full_imp, y, verbose=False)
    else:
        model.fit(X_full_imp, y)

    # ── Feature Importance ─────────────────────────────────────────────────
    importance = _get_feature_importance(model, X.columns.tolist())

    # ── Save Model + Imputer ───────────────────────────────────────────────
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    imputer_path = MODEL_SAVE_DIR / f"stockinator_{model_type}_imputer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(imputer_path, "wb") as f:
        pickle.dump(final_imputer, f)

    logger.info("Model saved to %s", model_path)
    logger.info("Imputer saved to %s", imputer_path)

    # ── Aggregate Metrics ──────────────────────────────────────────────────
    avg_metrics = {
        "avg_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "avg_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "avg_r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }

    logger.info(
        "Training complete: Avg MAE=%.4f, RMSE=%.4f, R²=%.4f",
        avg_metrics["avg_mae"], avg_metrics["avg_rmse"], avg_metrics["avg_r2"],
    )

    # ── MLflow Final Logging ───────────────────────────────────────────────
    if mlflow_active:
        mlflow.log_metrics(avg_metrics)
        if importance:
            mlflow.log_dict(
                {"feature_importance": importance[:20]},
                "feature_importance.json",
            )
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass  # Non-sklearn models (XGBoost/LGBM) may need specific logging
        mlflow.end_run()
        logger.info("MLflow run logged — view with: mlflow ui")

    return {
        "model": model,
        "imputer": final_imputer,
        "model_type": model_type,
        "model_path": str(model_path),
        "imputer_path": str(imputer_path),
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "feature_importance": importance,
        "n_features": len(X.columns),
        "n_samples": len(X),
        "feature_names": X.columns.tolist(),
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


def load_model(model_type: str = "xgboost") -> tuple:
    """
    Load a saved model and its imputer from disk.

    Returns:
        (model, imputer) tuple
    """
    model_path = MODEL_SAVE_DIR / f"stockinator_{model_type}.pkl"
    imputer_path = MODEL_SAVE_DIR / f"stockinator_{model_type}_imputer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    imputer = None
    if imputer_path.exists():
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)
    else:
        logger.warning("No imputer found at %s — using fallback median fill", imputer_path)

    return model, imputer
