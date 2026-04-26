"""
Conformal Prediction — Distribution-Free Confidence Intervals

Uses MAPIE (Model Agnostic Prediction Interval Estimator)
for calibrated 95% confidence intervals on valuation estimates.

Conformal prediction guarantees valid coverage regardless of
the underlying data distribution.

Reference: stock_valuation_ml_reference.md §5.3
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_with_conformal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_type: str = "xgboost",
    alpha: float = 0.05,
) -> dict:
    """
    Train model with MAPIE conformal prediction wrapper.

    Produces prediction intervals with guaranteed (1-alpha) coverage.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features for prediction
        model_type: Base model type
        alpha: Significance level (0.05 = 95% CI)

    Returns:
        Dict with predictions, intervals, and diagnostics
    """
    try:
        from mapie.regression import MapieRegressor
    except ImportError:
        logger.warning("MAPIE not installed — falling back to bootstrap CI")
        return _bootstrap_ci(X_train, y_train, X_test, model_type, alpha)

    from models.ml.train import _create_model

    base_model = _create_model(model_type)

    # Fill NaN
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # MAPIE wrapper
    mapie = MapieRegressor(
        estimator=base_model,
        method="plus",
        cv=5,
    )
    mapie.fit(X_train, y_train)

    # Predict with intervals
    y_pred, y_ci = mapie.predict(X_test, alpha=alpha)

    results = {
        "predictions": y_pred.tolist(),
        "ci_lower": y_ci[:, 0, 0].tolist(),
        "ci_upper": y_ci[:, 1, 0].tolist(),
        "alpha": alpha,
        "method": "conformal_plus",
        "coverage_guarantee": f"{(1-alpha)*100:.0f}%",
    }

    # Coverage width
    widths = y_ci[:, 1, 0] - y_ci[:, 0, 0]
    results["avg_ci_width"] = float(np.mean(widths))
    results["median_ci_width"] = float(np.median(widths))

    logger.info(
        "Conformal prediction: %d samples, avg CI width=%.4f, coverage=%s",
        len(y_pred), results["avg_ci_width"], results["coverage_guarantee"],
    )

    return results


def _bootstrap_ci(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_type: str,
    alpha: float,
    n_bootstrap: int = 100,
) -> dict:
    """
    Fallback: Bootstrap confidence intervals when MAPIE is not available.

    Less rigorous than conformal prediction but still useful.
    """
    from models.ml.train import _create_model

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    predictions = []
    n_samples = len(X_train)

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_b = X_train.iloc[idx]
        y_b = y_train.iloc[idx]

        model = _create_model(model_type)
        if model_type == "ridge":
            model.fit(X_b, y_b)
        else:
            model.fit(X_b, y_b, verbose=False)

        preds = model.predict(X_test)
        predictions.append(preds)

    pred_array = np.array(predictions)
    y_pred = np.mean(pred_array, axis=0)
    ci_lower = np.percentile(pred_array, (alpha / 2) * 100, axis=0)
    ci_upper = np.percentile(pred_array, (1 - alpha / 2) * 100, axis=0)

    widths = ci_upper - ci_lower

    return {
        "predictions": y_pred.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
        "alpha": alpha,
        "method": "bootstrap",
        "n_bootstrap": n_bootstrap,
        "avg_ci_width": float(np.mean(widths)),
        "median_ci_width": float(np.median(widths)),
    }
