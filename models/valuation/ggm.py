"""
Gordon Growth Model (GGM) — Dividend Discount Model

Intrinsic Value = D1 / (r - g)

Where:
    D1 = Next year's expected dividend = D0 × (1 + g)
    r  = Required rate of return (from CAPM)
    g  = Sustainable dividend growth rate

Applicability:
    - Company must pay dividends
    - Growth rate must be less than discount rate (g < r)
    - Best suited for mature, stable dividend-paying companies

Reference: stock_valuation_ml_reference.md §2.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GGMResult:
    """Result of a Gordon Growth Model valuation."""

    intrinsic_value: float | None
    d0: float
    d1: float | None
    growth_rate: float
    discount_rate: float
    margin_of_safety: float | None
    valid: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "ggm",
            "value": self.intrinsic_value,
            "inputs": {
                "d0": self.d0,
                "d1": self.d1,
                "g": self.growth_rate,
                "r": self.discount_rate,
            },
            "valid": self.valid,
            "margin_of_safety": self.margin_of_safety,
            "warnings": self.warnings,
        }


def compute_capm(
    risk_free_rate: float,
    beta: float,
    market_premium: float = 0.055,
) -> float:
    """
    Capital Asset Pricing Model — compute required rate of return.

    r = Rf + β × (Rm - Rf)

    Args:
        risk_free_rate: Risk-free rate (10Y US Treasury yield, decimal)
        beta: Stock beta vs market (S&P 500)
        market_premium: Equity risk premium Rm - Rf (default ~5.5%)

    Returns:
        Required rate of return as decimal (e.g., 0.089 for 8.9%)
    """
    if beta < 0:
        logger.warning("Negative beta (%.2f) — stock inversely correlated with market", beta)

    return risk_free_rate + beta * market_premium


def compute_sustainable_growth(
    roe: float,
    payout_ratio: float,
) -> float:
    """
    Sustainable dividend growth rate from fundamentals.

    g = ROE × Retention Ratio = ROE × (1 - Payout Ratio)

    Args:
        roe: Return on Equity (decimal, e.g., 0.15 for 15%)
        payout_ratio: Dividend Payout Ratio (decimal, e.g., 0.40 for 40%)

    Returns:
        Sustainable growth rate as decimal
    """
    retention_ratio = 1.0 - min(max(payout_ratio, 0.0), 1.0)
    g = roe * retention_ratio
    return max(g, 0.0)  # Growth rate cannot be negative in GGM context


def compute_ggm(
    d0: float,
    growth_rate: float,
    discount_rate: float,
    current_price: float | None = None,
) -> GGMResult:
    """
    Compute intrinsic value using the Gordon Growth Model.

    V = D1 / (r - g)  where D1 = D0 × (1 + g)

    Args:
        d0: Last annual dividend per share (must be > 0)
        growth_rate: Sustainable dividend growth rate (decimal)
        discount_rate: Required rate of return from CAPM (decimal)
        current_price: Current market price (optional, for MOS calculation)

    Returns:
        GGMResult with intrinsic value, validity, and warnings
    """
    warnings = []

    # ── Validity Check 1: Dividend must exist and be positive ───────
    if d0 is None or d0 <= 0:
        return GGMResult(
            intrinsic_value=None,
            d0=d0 or 0.0,
            d1=None,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            margin_of_safety=None,
            valid=False,
            warnings=["NO_DIVIDEND: Company does not pay dividends; GGM not applicable"],
        )

    # ── Validity Check 2: g must be less than r ────────────────────
    if growth_rate >= discount_rate:
        return GGMResult(
            intrinsic_value=None,
            d0=d0,
            d1=None,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            margin_of_safety=None,
            valid=False,
            warnings=[
                f"INVALID_GGM: Growth rate ({growth_rate:.4f}) >= "
                f"discount rate ({discount_rate:.4f}); model undefined"
            ],
        )

    # ── Validity Check 3: Rates must be reasonable ─────────────────
    if growth_rate < 0:
        warnings.append(f"NEGATIVE_GROWTH: g={growth_rate:.4f}; dividend may be declining")

    if discount_rate <= 0:
        return GGMResult(
            intrinsic_value=None,
            d0=d0,
            d1=None,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            margin_of_safety=None,
            valid=False,
            warnings=["INVALID_DISCOUNT_RATE: r must be positive"],
        )

    # ── Compute intrinsic value ────────────────────────────────────
    d1 = d0 * (1.0 + growth_rate)
    spread = discount_rate - growth_rate

    # Warn if spread is very narrow (value will be extremely sensitive)
    if spread < 0.01:
        warnings.append(
            f"NARROW_SPREAD: r-g={spread:.4f}; valuation highly sensitive to inputs"
        )

    intrinsic_value = d1 / spread

    # Sanity check: value shouldn't be astronomically high
    if current_price and intrinsic_value > current_price * 10:
        warnings.append(
            f"EXTREME_VALUE: GGM value (${intrinsic_value:.2f}) is "
            f">10x current price (${current_price:.2f}); inputs may be unreliable"
        )

    # ── Margin of Safety ───────────────────────────────────────────
    margin_of_safety = None
    if current_price and current_price > 0 and intrinsic_value:
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value

    logger.info(
        "GGM: D0=%.2f, D1=%.2f, g=%.4f, r=%.4f → IV=$%.2f",
        d0, d1, growth_rate, discount_rate, intrinsic_value,
    )

    return GGMResult(
        intrinsic_value=round(intrinsic_value, 2),
        d0=d0,
        d1=round(d1, 4),
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        margin_of_safety=round(margin_of_safety, 4) if margin_of_safety is not None else None,
        valid=True,
        warnings=warnings,
    )
