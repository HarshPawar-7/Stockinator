"""
Data Quality Validation Gates

Enforces data quality checks before any valuation signal is acted upon.
Money is involved — correctness over convenience.

Reference: stock_valuation_ml_reference.md §6
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from config import (
    MAX_FUNDAMENTALS_AGE_DAYS,
    MODEL_DISAGREEMENT_THRESHOLD,
    MIN_VALID_MODELS,
    EXTREME_PE_UPPER,
    EXTREME_PE_LOWER,
)

logger = logging.getLogger(__name__)


def validate_stock_data(data, ticker: str) -> list[str]:
    """
    Run all data quality checks on fetched stock data.

    Returns list of warning strings. Empty list = all clear.
    """
    warnings = []

    # ── Staleness ──────────────────────────────────────────────────
    if hasattr(data, "fetch_date") and data.fetch_date:
        try:
            fetch = datetime.strptime(data.fetch_date, "%Y-%m-%d")
            age = (datetime.now() - fetch).days
            if age > MAX_FUNDAMENTALS_AGE_DAYS:
                warnings.append(
                    f"STALE_DATA: Data is {age} days old (threshold: {MAX_FUNDAMENTALS_AGE_DAYS})"
                )
        except ValueError:
            pass

    # ── Missing critical fields ────────────────────────────────────
    missing = []
    if not data.current_price:
        missing.append("current_price")
    if not data.market_cap:
        missing.append("market_cap")
    if not data.shares_outstanding:
        missing.append("shares_outstanding")
    if missing:
        warnings.append(f"MISSING_FIELDS: {', '.join(missing)}")

    # ── Negative equity ────────────────────────────────────────────
    if data.total_equity is not None and data.total_equity < 0:
        warnings.append("NEGATIVE_EQUITY: RIM invalid; DCF weights may need adjustment")

    # ── Extreme P/E ────────────────────────────────────────────────
    if data.pe_ratio is not None:
        if data.pe_ratio > EXTREME_PE_UPPER:
            warnings.append(f"EXTREME_PE: P/E={data.pe_ratio:.1f} (>{EXTREME_PE_UPPER}); comps may be distorted")
        elif data.pe_ratio < EXTREME_PE_LOWER:
            warnings.append(f"NEGATIVE_PE: P/E={data.pe_ratio:.1f}; earnings are negative")

    # ── Negative FCF check ─────────────────────────────────────────
    if hasattr(data, "historical_fcf") and data.historical_fcf:
        negative_count = sum(1 for f in data.historical_fcf if f < 0)
        if negative_count >= 3:
            warnings.append(
                f"PERSISTENT_NEGATIVE_FCF: {negative_count} years of negative FCF; "
                f"DCF unreliable"
            )

    # ── No dividend ────────────────────────────────────────────────
    if not data.dividend_per_share or data.dividend_per_share <= 0:
        warnings.append("NO_DIVIDEND: GGM not applicable")

    # ── Extreme debt ───────────────────────────────────────────────
    if (data.total_debt and data.total_equity and data.total_equity > 0):
        de_ratio = data.total_debt / data.total_equity
        if de_ratio > 5.0:
            warnings.append(f"HIGH_LEVERAGE: Debt/Equity={de_ratio:.1f}; financial distress risk")

    return warnings


def validate_model_outputs(model_values: dict[str, float | None]) -> list[str]:
    """
    Validate combined model outputs for disagreement and coverage.
    """
    warnings = []
    valid = {k: v for k, v in model_values.items() if v is not None and v > 0}

    if len(valid) < MIN_VALID_MODELS:
        warnings.append(
            f"INSUFFICIENT_MODELS: Only {len(valid)} valid model(s); "
            f"need ≥{MIN_VALID_MODELS} for reliable signal"
        )

    if len(valid) >= 2:
        values = list(valid.values())
        mean_val = sum(values) / len(values)
        if mean_val > 0:
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            disagreement = std_val / mean_val
            if disagreement > MODEL_DISAGREEMENT_THRESHOLD:
                warnings.append(
                    f"HIGH_MODEL_DISAGREEMENT: {disagreement:.2%} — "
                    f"widen CI, reduce position sizing"
                )

    return warnings
