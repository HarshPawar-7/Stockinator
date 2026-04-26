"""
Comparable Company Analysis (Comps / Multiples)

Value = Median Peer Multiple × Company Metric

Supported multiples: EV/EBITDA, P/E, P/S, EV/Revenue, P/B

Reference: stock_valuation_ml_reference.md §2.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompsResult:
    """Result of a Comparable Company Analysis."""
    intrinsic_value: float | None
    multiple_used: str | None
    peer_median_multiple: float | None
    company_metric: float | None
    num_peers: int
    peer_tickers: list[str]
    margin_of_safety: float | None
    valid: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": "comps",
            "value": self.intrinsic_value,
            "inputs": {
                "multiple": self.multiple_used,
                "peer_median": self.peer_median_multiple,
                "company_metric": self.company_metric,
                "num_peers": self.num_peers,
            },
            "valid": self.valid,
            "margin_of_safety": self.margin_of_safety,
            "warnings": self.warnings,
        }


def select_best_multiple(company_data: dict) -> str:
    """
    Choose the most appropriate valuation multiple based on available data.

    Priority:
    1. EV/EBITDA — best general-purpose multiple
    2. P/E — if earnings are positive
    3. P/S — fallback for negative-earnings companies
    4. P/B — financials / banks
    """
    ebitda = company_data.get("ebitda", 0)
    net_income = company_data.get("net_income", 0)
    revenue = company_data.get("revenue", 0)
    sector = company_data.get("sector", "").lower()

    if "financ" in sector or "bank" in sector or "insurance" in sector:
        return "pb"
    if ebitda and ebitda > 0:
        return "ev_ebitda"
    if net_income and net_income > 0:
        return "pe"
    if revenue and revenue > 0:
        return "ps"
    return "ev_ebitda"  # default


def compute_multiple_value(
    company_data: dict,
    peer_multiples: list[float],
    multiple_type: str,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute implied value from peer median multiple.

    Returns: (implied_value_per_share, peer_median, company_metric)
    """
    if not peer_multiples:
        return None, None, None

    # Remove outliers: keep values within 1st-99th percentile
    arr = np.array([m for m in peer_multiples if m > 0 and np.isfinite(m)])
    if len(arr) == 0:
        return None, None, None

    p1, p99 = np.percentile(arr, [1, 99])
    filtered = arr[(arr >= p1) & (arr <= p99)]
    if len(filtered) == 0:
        filtered = arr

    median_multiple = float(np.median(filtered))
    shares = company_data.get("shares_outstanding", 0)

    if multiple_type == "ev_ebitda":
        ebitda = company_data.get("ebitda", 0)
        if not ebitda or ebitda <= 0:
            return None, median_multiple, None
        ev = median_multiple * ebitda
        net_debt = company_data.get("net_debt", 0)
        equity = ev - net_debt
        per_share = equity / shares if shares > 0 else None
        return per_share, median_multiple, ebitda

    elif multiple_type == "pe":
        eps = company_data.get("eps", 0)
        if not eps or eps <= 0:
            return None, median_multiple, None
        return round(median_multiple * eps, 2), median_multiple, eps

    elif multiple_type == "ps":
        revenue = company_data.get("revenue", 0)
        if not revenue or revenue <= 0:
            return None, median_multiple, None
        per_share = (median_multiple * revenue) / shares if shares > 0 else None
        return per_share, median_multiple, revenue

    elif multiple_type == "pb":
        book_value_ps = company_data.get("book_value_per_share", 0)
        if not book_value_ps or book_value_ps <= 0:
            return None, median_multiple, None
        return round(median_multiple * book_value_ps, 2), median_multiple, book_value_ps

    elif multiple_type == "ev_revenue":
        revenue = company_data.get("revenue", 0)
        if not revenue or revenue <= 0:
            return None, median_multiple, None
        ev = median_multiple * revenue
        net_debt = company_data.get("net_debt", 0)
        equity = ev - net_debt
        per_share = equity / shares if shares > 0 else None
        return per_share, median_multiple, revenue

    return None, median_multiple, None


def compute_comps(
    company_data: dict,
    peer_data: list[dict],
    multiple_type: str | None = None,
    current_price: float | None = None,
) -> CompsResult:
    """
    Compute intrinsic value using Comparable Company Analysis.

    Args:
        company_data: Dict with company fundamentals
            Required keys depend on multiple_type:
            - ev_ebitda: ebitda, net_debt, shares_outstanding
            - pe: eps
            - ps: revenue, shares_outstanding
            - pb: book_value_per_share
        peer_data: List of dicts with peer company data
            Each must have the relevant multiple field
        multiple_type: Which multiple to use (auto-selected if None)
        current_price: Current market price for MOS calculation

    Returns:
        CompsResult with implied valuation
    """
    warnings = []

    # Auto-select multiple if not specified
    if multiple_type is None:
        multiple_type = select_best_multiple(company_data)

    # Check if we have peers
    if not peer_data:
        return CompsResult(
            intrinsic_value=None, multiple_used=multiple_type,
            peer_median_multiple=None, company_metric=None,
            num_peers=0, peer_tickers=[],
            margin_of_safety=None, valid=False,
            warnings=["NO_PEERS: No comparable companies found"],
        )

    # Extract peer multiples
    multiple_key_map = {
        "ev_ebitda": "ev_ebitda_ratio",
        "pe": "pe_ratio",
        "ps": "ps_ratio",
        "pb": "pb_ratio",
        "ev_revenue": "ev_revenue_ratio",
    }
    key = multiple_key_map.get(multiple_type, f"{multiple_type}_ratio")

    peer_multiples = []
    peer_tickers = []
    for p in peer_data:
        val = p.get(key)
        if val and val > 0 and np.isfinite(val):
            peer_multiples.append(val)
            peer_tickers.append(p.get("ticker", "?"))

    if len(peer_multiples) < 3:
        warnings.append(f"FEW_PEERS: Only {len(peer_multiples)} comparable peers found")

    if not peer_multiples:
        return CompsResult(
            intrinsic_value=None, multiple_used=multiple_type,
            peer_median_multiple=None, company_metric=None,
            num_peers=0, peer_tickers=[],
            margin_of_safety=None, valid=False,
            warnings=["NO_VALID_MULTIPLES: Peers lack valid multiple data"],
        )

    # Compute implied value
    per_share, median_mult, metric = compute_multiple_value(
        company_data, peer_multiples, multiple_type
    )

    if per_share is None or per_share <= 0:
        return CompsResult(
            intrinsic_value=None, multiple_used=multiple_type,
            peer_median_multiple=median_mult, company_metric=metric,
            num_peers=len(peer_multiples), peer_tickers=peer_tickers,
            margin_of_safety=None, valid=False,
            warnings=warnings + ["INVALID_VALUE: Could not compute positive per-share value"],
        )

    per_share = round(per_share, 2)

    # Margin of Safety
    mos = None
    if current_price and current_price > 0 and per_share > 0:
        mos = round((per_share - current_price) / per_share, 4)

    logger.info(
        "Comps: %s median=%.2f, metric=%.2f → $%.2f/share (%d peers)",
        multiple_type, median_mult, metric or 0, per_share, len(peer_multiples),
    )

    return CompsResult(
        intrinsic_value=per_share,
        multiple_used=multiple_type,
        peer_median_multiple=round(median_mult, 2) if median_mult else None,
        company_metric=metric,
        num_peers=len(peer_multiples),
        peer_tickers=peer_tickers,
        margin_of_safety=mos,
        valid=True,
        warnings=warnings,
    )
