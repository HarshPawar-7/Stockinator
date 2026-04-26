"""
Ensemble Valuation Combiner

Combines outputs from GGM, DCF, Comps, and RIM with configurable weights.
Produces a final intrinsic value, confidence interval, and BUY/HOLD/SELL signal.

Reference: stock_valuation_ml_reference.md §2.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import MODEL_WEIGHTS, SIGNAL_THRESHOLDS, HIGH_DISAGREEMENT_BUY_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Combined valuation result from all models."""
    ticker: str
    current_price: float | None
    ensemble_value: float | None
    ci_lower_95: float | None
    ci_upper_95: float | None
    margin_of_safety: float | None
    signal: str  # BUY, HOLD, SELL, STRONG_BUY, STRONG_SELL, INSUFFICIENT_DATA
    model_values: dict[str, float | None]
    model_weights_used: dict[str, float]
    model_disagreement: float | None
    all_warnings: list[str] = field(default_factory=list)
    valid_model_count: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "ensemble_value": self.ensemble_value,
            "ci_95": [self.ci_lower_95, self.ci_upper_95],
            "margin_of_safety": self.margin_of_safety,
            "signal": self.signal,
            "model_values": self.model_values,
            "weights": self.model_weights_used,
            "model_disagreement": self.model_disagreement,
            "valid_model_count": self.valid_model_count,
            "warnings": self.all_warnings,
        }


def _compute_signal(mos: float, high_disagreement: bool) -> str:
    """Determine investment signal from margin of safety."""
    if high_disagreement:
        # Require higher MOS when models disagree
        if mos > HIGH_DISAGREEMENT_BUY_THRESHOLD:
            return "BUY"
        elif mos < -HIGH_DISAGREEMENT_BUY_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"

    if mos > SIGNAL_THRESHOLDS["strong_buy"]:
        return "STRONG_BUY"
    elif mos > SIGNAL_THRESHOLDS["buy"]:
        return "BUY"
    elif mos < SIGNAL_THRESHOLDS["strong_sell"]:
        return "STRONG_SELL"
    elif mos < SIGNAL_THRESHOLDS["sell"]:
        return "SELL"
    else:
        return "HOLD"


def weighted_ensemble(
    ticker: str,
    ggm_value: float | None,
    dcf_value: float | None,
    comps_value: float | None,
    rim_value: float | None,
    current_price: float | None = None,
    custom_weights: dict[str, float] | None = None,
    all_warnings: list[str] | None = None,
) -> EnsembleResult:
    """
    Combine valuation model outputs into a weighted ensemble.

    When a model is invalid (None), its weight is redistributed
    proportionally among valid models.

    Args:
        ticker: Stock ticker symbol
        ggm_value: Gordon Growth Model intrinsic value per share
        dcf_value: DCF intrinsic value per share
        comps_value: Comps intrinsic value per share
        rim_value: RIM intrinsic value per share
        current_price: Current market price
        custom_weights: Override default model weights
        all_warnings: Accumulated warnings from all models

    Returns:
        EnsembleResult with combined valuation and signal
    """
    warnings = list(all_warnings or [])
    weights = dict(custom_weights or MODEL_WEIGHTS)

    model_values = {
        "ggm": ggm_value,
        "dcf": dcf_value,
        "comps": comps_value,
        "rim": rim_value,
    }

    # ── Identify valid models ──────────────────────────────────────
    valid = {k: v for k, v in model_values.items() if v is not None and v > 0}
    valid_count = len(valid)

    # ── Insufficient data check ────────────────────────────────────
    if valid_count < 2:
        return EnsembleResult(
            ticker=ticker, current_price=current_price,
            ensemble_value=list(valid.values())[0] if valid_count == 1 else None,
            ci_lower_95=None, ci_upper_95=None, margin_of_safety=None,
            signal="INSUFFICIENT_DATA", model_values=model_values,
            model_weights_used={}, model_disagreement=None,
            all_warnings=warnings + [
                f"INSUFFICIENT_MODELS: Only {valid_count} valid model(s); need ≥2 for signal"
            ],
            valid_model_count=valid_count,
        )

    # ── Redistribute weights among valid models ────────────────────
    active_weights = {k: weights.get(k, 0) for k in valid}
    total_w = sum(active_weights.values())
    if total_w > 0:
        active_weights = {k: w / total_w for k, w in active_weights.items()}
    else:
        # Equal weighting fallback
        active_weights = {k: 1.0 / valid_count for k in valid}

    # ── Weighted average ───────────────────────────────────────────
    ensemble_value = sum(valid[k] * active_weights[k] for k in valid)

    # ── Model disagreement ─────────────────────────────────────────
    values_arr = np.array(list(valid.values()))
    mean_val = np.mean(values_arr)
    std_val = np.std(values_arr)
    disagreement = float(std_val / mean_val) if mean_val > 0 else 0.0

    high_disagreement = disagreement > 0.40
    if high_disagreement:
        warnings.append(
            f"HIGH_MODEL_DISAGREEMENT: {disagreement:.2%} — "
            f"widen CI, reduce position sizing"
        )

    # ── Confidence Interval (bootstrap-style for Phase 1) ──────────
    # Use ±1.96×std as approximate 95% CI, bounded by model range
    ci_half = 1.96 * std_val
    ci_lower = max(ensemble_value - ci_half, min(values_arr) * 0.9)
    ci_upper = ensemble_value + ci_half

    # ── Margin of Safety & Signal ──────────────────────────────────
    mos = None
    signal = "INSUFFICIENT_DATA"
    if current_price and current_price > 0 and ensemble_value > 0:
        mos = (ensemble_value - current_price) / ensemble_value
        signal = _compute_signal(mos, high_disagreement)

    logger.info(
        "Ensemble [%s]: %.2f (CI: %.2f–%.2f) | MOS: %s | Signal: %s | Models: %d",
        ticker, ensemble_value, ci_lower, ci_upper,
        f"{mos:.2%}" if mos is not None else "N/A",
        signal, valid_count,
    )

    return EnsembleResult(
        ticker=ticker,
        current_price=current_price,
        ensemble_value=round(ensemble_value, 2),
        ci_lower_95=round(ci_lower, 2),
        ci_upper_95=round(ci_upper, 2),
        margin_of_safety=round(mos, 4) if mos is not None else None,
        signal=signal,
        model_values=model_values,
        model_weights_used={k: round(v, 4) for k, v in active_weights.items()},
        model_disagreement=round(disagreement, 4),
        all_warnings=warnings,
        valid_model_count=valid_count,
    )
