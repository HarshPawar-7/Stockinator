"""
Discounted Cash Flow (DCF) Model

Intrinsic Value = Σ [FCFt / (1+WACC)^t] + Terminal Value / (1+WACC)^n

Reference: stock_valuation_ml_reference.md §2.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import DCF_FORECAST_YEARS, TERMINAL_GROWTH_RATE, TERMINAL_GROWTH_MAX

logger = logging.getLogger(__name__)


@dataclass
class DCFResult:
    """Result of a Discounted Cash Flow valuation."""
    intrinsic_value: float | None
    intrinsic_value_per_share: float | None
    present_value_fcfs: float | None
    terminal_value: float | None
    pv_terminal_value: float | None
    wacc: float
    terminal_growth: float
    forecast_years: int
    fcf_forecasts: list[float]
    shares_outstanding: float | None
    margin_of_safety: float | None
    valid: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "dcf",
            "value": self.intrinsic_value_per_share,
            "inputs": {
                "wacc": self.wacc,
                "terminal_growth": self.terminal_growth,
                "forecast_years": self.forecast_years,
                "fcf_base": self.fcf_forecasts[0] if self.fcf_forecasts else None,
            },
            "enterprise_value": self.intrinsic_value,
            "valid": self.valid,
            "margin_of_safety": self.margin_of_safety,
            "warnings": self.warnings,
        }


def compute_wacc(market_cap, total_debt, cost_of_equity, cost_of_debt, tax_rate):
    """WACC = (E/V)×Re + (D/V)×Rd×(1-T)"""
    total_value = market_cap + total_debt
    if total_value <= 0:
        raise ValueError("Total enterprise value (E+D) must be positive")
    eq_w = market_cap / total_value
    debt_w = total_debt / total_value
    return max(eq_w * cost_of_equity + debt_w * cost_of_debt * (1.0 - tax_rate), 0.001)


def project_fcf(historical_fcf, forecast_years=DCF_FORECAST_YEARS, growth_cap=0.30):
    """Project future FCFs from historical data using median growth, capped."""
    if len(historical_fcf) < 2:
        raise ValueError("Need at least 2 years of historical FCF data")

    growth_rates = []
    for i in range(1, len(historical_fcf)):
        if historical_fcf[i - 1] > 0 and historical_fcf[i] > 0:
            growth_rates.append((historical_fcf[i] - historical_fcf[i - 1]) / historical_fcf[i - 1])

    if not growth_rates:
        positives = [f for f in historical_fcf if f > 0]
        if not positives:
            raise ValueError("No positive FCF in historical data")
        return [positives[-1]] * forecast_years, 0.0

    median_growth = float(np.median(growth_rates))
    capped = max(min(median_growth, growth_cap), -growth_cap)
    base = historical_fcf[-1]
    if base <= 0:
        positives = [f for f in historical_fcf if f > 0]
        if not positives:
            raise ValueError("No positive FCF in historical data")
        base = positives[-1]
        capped = min(capped, 0.05)

    projections = []
    for t in range(1, forecast_years + 1):
        decay = 1.0 - (t / (forecast_years + 1))
        yr_g = capped * decay + TERMINAL_GROWTH_RATE * (1 - decay)
        base = base * (1.0 + yr_g)
        projections.append(round(base, 2))

    return projections, capped


def compute_dcf(fcf_forecasts, wacc, terminal_growth=TERMINAL_GROWTH_RATE,
                shares_outstanding=None, net_debt=0.0, current_price=None):
    """Compute intrinsic value using DCF. Returns DCFResult."""
    warnings = []
    n = len(fcf_forecasts)

    if not fcf_forecasts or n == 0:
        return DCFResult(None, None, None, None, None, wacc, terminal_growth,
                         0, [], shares_outstanding, None, False,
                         ["NO_FCF_DATA: Cannot run DCF without FCF forecasts"])

    if terminal_growth >= wacc:
        return DCFResult(None, None, None, None, None, wacc, terminal_growth,
                         n, fcf_forecasts, shares_outstanding, None, False,
                         [f"INVALID_DCF: g_term ({terminal_growth:.4f}) >= WACC ({wacc:.4f})"])

    if terminal_growth > TERMINAL_GROWTH_MAX:
        warnings.append(f"CAPPED_TERMINAL_GROWTH: {terminal_growth:.4f} → {TERMINAL_GROWTH_MAX}")
        terminal_growth = TERMINAL_GROWTH_MAX

    if wacc < 0.02:
        warnings.append(f"LOW_WACC: {wacc:.4f}")
    if wacc > 0.25:
        warnings.append(f"HIGH_WACC: {wacc:.4f}")

    # Present value of projected FCFs
    pv_fcfs = sum(fcf / (1.0 + wacc) ** t for t, fcf in enumerate(fcf_forecasts, 1))

    # Terminal value (Gordon Growth)
    final_fcf = abs(fcf_forecasts[-1]) if fcf_forecasts[-1] <= 0 else fcf_forecasts[-1]
    if fcf_forecasts[-1] <= 0:
        warnings.append("NEGATIVE_TERMINAL_FCF: Terminal value may be unreliable")

    tv = final_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / (1.0 + wacc) ** n

    ev = pv_fcfs + pv_tv
    if ev > 0 and pv_tv / ev > 0.80:
        warnings.append(f"TV_DOMINANCE: Terminal value is {pv_tv/ev:.0%} of total")

    equity_value = ev - net_debt
    per_share = None
    if shares_outstanding and shares_outstanding > 0:
        per_share = equity_value / shares_outstanding
        if per_share < 0:
            warnings.append("NEGATIVE_EQUITY_VALUE")
            per_share = None

    mos = None
    v = per_share or None
    if current_price and current_price > 0 and v and v > 0:
        mos = (v - current_price) / v

    return DCFResult(
        intrinsic_value=round(ev, 2),
        intrinsic_value_per_share=round(per_share, 2) if per_share else None,
        present_value_fcfs=round(pv_fcfs, 2),
        terminal_value=round(tv, 2),
        pv_terminal_value=round(pv_tv, 2),
        wacc=wacc, terminal_growth=terminal_growth, forecast_years=n,
        fcf_forecasts=fcf_forecasts, shares_outstanding=shares_outstanding,
        margin_of_safety=round(mos, 4) if mos is not None else None,
        valid=ev > 0, warnings=warnings,
    )
