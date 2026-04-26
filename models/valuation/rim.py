"""
Residual Income Model (RIM / Edwards-Bell-Ohlson)

Intrinsic Value = Book Value + Σ [RI_t / (1+r)^t]
RI_t = Net Income_t - (r × Book Value_{t-1})

Works well when FCF is hard to estimate (banks, insurance).
Uses accounting data directly.

Reference: stock_valuation_ml_reference.md §2.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RIMResult:
    """Result of a Residual Income Model valuation."""
    intrinsic_value: float | None
    book_value_per_share: float | None
    pv_residual_income: float | None
    roe: float | None
    cost_of_equity: float
    forecast_years: int
    margin_of_safety: float | None
    valid: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "rim",
            "value": self.intrinsic_value,
            "inputs": {
                "book_value": self.book_value_per_share,
                "roe": self.roe,
                "r": self.cost_of_equity,
                "forecast_years": self.forecast_years,
            },
            "valid": self.valid,
            "margin_of_safety": self.margin_of_safety,
            "warnings": self.warnings,
        }


def compute_rim(
    book_value_per_share: float,
    earnings_per_share: float,
    cost_of_equity: float,
    forecast_years: int = 10,
    roe: float | None = None,
    roe_fade_to: float | None = None,
    current_price: float | None = None,
) -> RIMResult:
    """
    Compute intrinsic value using the Residual Income Model.

    V = BV + Σ (EPS - r×BV) / (1+r)^t

    For simplicity in Phase 1, we assume constant ROE and compute
    residual income as: RI = (ROE - r) × BV_per_share

    Over time, ROE fades toward cost of equity (competitive equilibrium).

    Args:
        book_value_per_share: Current book value per share
        earnings_per_share: Most recent EPS (trailing twelve months)
        cost_of_equity: Required return from CAPM (decimal)
        forecast_years: Number of years to project residual income
        roe: Return on equity (if None, computed from EPS/BV)
        roe_fade_to: Terminal ROE to fade toward (defaults to cost_of_equity)
        current_price: Current price for MOS calculation

    Returns:
        RIMResult with intrinsic value and diagnostics
    """
    warnings = []

    # ── Validity Check 1: Positive book value ──────────────────────
    if book_value_per_share is None or book_value_per_share <= 0:
        return RIMResult(
            intrinsic_value=None, book_value_per_share=book_value_per_share,
            pv_residual_income=None, roe=roe, cost_of_equity=cost_of_equity,
            forecast_years=forecast_years, margin_of_safety=None, valid=False,
            warnings=["NEGATIVE_EQUITY: Book value <= 0; RIM not applicable"],
        )

    # ── Validity Check 2: Cost of equity must be positive ──────────
    if cost_of_equity <= 0:
        return RIMResult(
            intrinsic_value=None, book_value_per_share=book_value_per_share,
            pv_residual_income=None, roe=roe, cost_of_equity=cost_of_equity,
            forecast_years=forecast_years, margin_of_safety=None, valid=False,
            warnings=["INVALID_COST_OF_EQUITY: r must be positive"],
        )

    # ── Compute ROE if not provided ────────────────────────────────
    if roe is None:
        if earnings_per_share is None:
            return RIMResult(
                intrinsic_value=None, book_value_per_share=book_value_per_share,
                pv_residual_income=None, roe=None, cost_of_equity=cost_of_equity,
                forecast_years=forecast_years, margin_of_safety=None, valid=False,
                warnings=["NO_EARNINGS: Cannot compute ROE without EPS"],
            )
        roe = earnings_per_share / book_value_per_share

    if roe_fade_to is None:
        roe_fade_to = cost_of_equity  # Competitive equilibrium

    # ── Project residual income with ROE fade ──────────────────────
    pv_ri_total = 0.0
    bv = book_value_per_share

    for t in range(1, forecast_years + 1):
        # Linear fade of ROE toward terminal ROE
        fade_progress = t / forecast_years
        current_roe = roe * (1 - fade_progress) + roe_fade_to * fade_progress

        # Residual income = (ROE - r) × BV
        ri = (current_roe - cost_of_equity) * bv

        # Discount back to present
        pv_ri = ri / (1.0 + cost_of_equity) ** t
        pv_ri_total += pv_ri

        # Update book value (BV grows by retained earnings)
        eps_t = current_roe * bv
        # Assume ~60% retention ratio for BV growth
        retention = 0.60
        bv = bv + eps_t * retention

    # ── Intrinsic value ────────────────────────────────────────────
    intrinsic_value = book_value_per_share + pv_ri_total

    if intrinsic_value <= 0:
        warnings.append("NEGATIVE_INTRINSIC: RIM yields negative value; ROE < cost of equity")

    # Handle edge cases
    if roe < cost_of_equity:
        warnings.append(
            f"VALUE_DESTROYER: ROE ({roe:.4f}) < cost of equity ({cost_of_equity:.4f}); "
            f"company destroys shareholder value"
        )

    if roe > 0.50:
        warnings.append(f"EXTREME_ROE: {roe:.4f}; may be unsustainable")

    # ── Margin of Safety ───────────────────────────────────────────
    mos = None
    if current_price and current_price > 0 and intrinsic_value and intrinsic_value > 0:
        mos = (intrinsic_value - current_price) / intrinsic_value

    logger.info(
        "RIM: BV=%.2f, ROE=%.4f, r=%.4f → IV=$%.2f",
        book_value_per_share, roe, cost_of_equity, intrinsic_value,
    )

    return RIMResult(
        intrinsic_value=round(intrinsic_value, 2) if intrinsic_value > 0 else None,
        book_value_per_share=book_value_per_share,
        pv_residual_income=round(pv_ri_total, 2),
        roe=round(roe, 4),
        cost_of_equity=cost_of_equity,
        forecast_years=forecast_years,
        margin_of_safety=round(mos, 4) if mos is not None else None,
        valid=intrinsic_value > 0,
        warnings=warnings,
    )
