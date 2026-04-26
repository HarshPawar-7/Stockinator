"""
Batch Valuation Pipeline

Orchestrates: data fetch → validation → all 4 models → ensemble → output.
This is the main engine that ties everything together.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from config import TERMINAL_GROWTH_RATE
from models.valuation.ggm import compute_ggm, compute_capm, compute_sustainable_growth
from models.valuation.dcf import compute_dcf, project_fcf
from models.valuation.comps import compute_comps
from models.valuation.rim import compute_rim
from models.ensemble import weighted_ensemble
from pipeline.ingest import fetch_stock_data, fetch_peer_data, StockData
from pipeline.data_quality import validate_stock_data, validate_model_outputs

logger = logging.getLogger(__name__)


def valuate_single_stock(ticker: str, fetch_peers: bool = True) -> dict:
    """
    Run full valuation pipeline for a single stock.

    Steps:
    1. Fetch data via yfinance
    2. Run data quality checks
    3. Run GGM, DCF, Comps, RIM
    4. Combine via ensemble
    5. Return structured result

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        fetch_peers: Whether to fetch peer data for comps (slower)

    Returns:
        Dict with full valuation results
    """
    logger.info("=" * 60)
    logger.info("VALUATING: %s", ticker)
    logger.info("=" * 60)

    # ── Step 1: Fetch data ─────────────────────────────────────────
    data = fetch_stock_data(ticker)
    all_warnings = list(data.warnings)

    # ── Step 2: Data quality checks ────────────────────────────────
    quality_warnings = validate_stock_data(data, ticker)
    all_warnings.extend(quality_warnings)

    # ── Step 3a: Gordon Growth Model ───────────────────────────────
    ggm_result = None
    ggm_value = None
    if data.dividend_per_share and data.dividend_per_share > 0:
        r = data.cost_of_equity or 0.10
        g = data.sustainable_growth
        if g is None or g <= 0:
            # Fallback: use historical dividend growth or conservative estimate
            g = 0.03

        # Cap growth rate — companies with extreme ROE from buybacks
        # (e.g. AAPL ROE=150%) produce unsustainable growth estimates.
        # GGM is for mature dividend growers; cap g well below r.
        MAX_SUSTAINABLE_GROWTH = min(r - 0.01, 0.15)  # At most 15% or r-1%
        if g > MAX_SUSTAINABLE_GROWTH:
            all_warnings.append(
                f"GGM_GROWTH_CAPPED: g={g:.4f} capped to {MAX_SUSTAINABLE_GROWTH:.4f} "
                f"(likely inflated ROE from buybacks)"
            )
            g = MAX_SUSTAINABLE_GROWTH

        ggm_result = compute_ggm(
            d0=data.dividend_per_share,
            growth_rate=g,
            discount_rate=r,
            current_price=data.current_price,
        )
        if ggm_result.valid:
            ggm_value = ggm_result.intrinsic_value
        all_warnings.extend(ggm_result.warnings)
    else:
        all_warnings.append("GGM_SKIPPED: No dividend data")

    # ── Step 3b: Discounted Cash Flow ──────────────────────────────
    dcf_result = None
    dcf_value = None
    if data.historical_fcf and len(data.historical_fcf) >= 2:
        try:
            fcf_projections, implied_growth = project_fcf(data.historical_fcf)
            wacc = data.wacc or 0.10

            dcf_result = compute_dcf(
                fcf_forecasts=fcf_projections,
                wacc=wacc,
                terminal_growth=TERMINAL_GROWTH_RATE,
                shares_outstanding=data.shares_outstanding,
                net_debt=data.net_debt or 0,
                current_price=data.current_price,
            )
            if dcf_result.valid:
                dcf_value = dcf_result.intrinsic_value_per_share
            all_warnings.extend(dcf_result.warnings)
        except ValueError as e:
            all_warnings.append(f"DCF_ERROR: {e}")
    else:
        all_warnings.append("DCF_SKIPPED: Insufficient FCF history")

    # ── Step 3c: Comparable Company Analysis ───────────────────────
    comps_result = None
    comps_value = None
    if fetch_peers and data.sector:
        peer_data = fetch_peer_data(
            ticker=ticker,
            sector=data.sector,
            industry=data.industry,
            market_cap=data.market_cap,
        )
        if peer_data:
            company_dict = {
                "ebitda": data.ebitda,
                "eps": data.eps,
                "revenue": data.revenue,
                "book_value_per_share": data.book_value_per_share,
                "shares_outstanding": data.shares_outstanding,
                "net_debt": data.net_debt or 0,
                "sector": data.sector,
            }
            comps_result = compute_comps(
                company_data=company_dict,
                peer_data=peer_data,
                current_price=data.current_price,
            )
            if comps_result.valid:
                comps_value = comps_result.intrinsic_value
            all_warnings.extend(comps_result.warnings)
        else:
            all_warnings.append("COMPS_SKIPPED: No peers found")
    else:
        all_warnings.append("COMPS_SKIPPED: Sector unknown or peers disabled")

    # ── Step 3d: Residual Income Model ─────────────────────────────
    rim_result = None
    rim_value = None
    if data.book_value_per_share and data.book_value_per_share > 0:
        r = data.cost_of_equity or 0.10
        roe_input = data.roe
        # Cap extreme ROE (buyback-inflated, e.g. AAPL ~150%)
        MAX_ROE_FOR_RIM = 0.50
        if roe_input and roe_input > MAX_ROE_FOR_RIM:
            all_warnings.append(
                f"RIM_ROE_CAPPED: ROE={roe_input:.4f} capped to {MAX_ROE_FOR_RIM} "
                f"(likely inflated from buybacks/low equity)"
            )
            roe_input = MAX_ROE_FOR_RIM
        rim_result = compute_rim(
            book_value_per_share=data.book_value_per_share,
            earnings_per_share=data.eps or 0,
            cost_of_equity=r,
            roe=roe_input,
            current_price=data.current_price,
        )
        if rim_result.valid:
            rim_value = rim_result.intrinsic_value
        all_warnings.extend(rim_result.warnings)
    else:
        all_warnings.append("RIM_SKIPPED: No book value data")

    # ── Step 4: Model output validation ────────────────────────────
    model_values = {"ggm": ggm_value, "dcf": dcf_value, "comps": comps_value, "rim": rim_value}
    model_warnings = validate_model_outputs(model_values)
    all_warnings.extend(model_warnings)

    # ── Step 5: Ensemble ───────────────────────────────────────────
    ensemble = weighted_ensemble(
        ticker=ticker,
        ggm_value=ggm_value,
        dcf_value=dcf_value,
        comps_value=comps_value,
        rim_value=rim_value,
        current_price=data.current_price,
        all_warnings=all_warnings,
    )

    # ── Build structured result ────────────────────────────────────
    result = {
        "ticker": ticker,
        "company_name": data.company_name,
        "valuation_date": date.today().isoformat(),
        "market_price": data.current_price,
        "models": {
            "ggm": ggm_result.to_dict() if ggm_result else {"model": "ggm", "valid": False, "value": None, "warnings": ["SKIPPED"]},
            "dcf": dcf_result.to_dict() if dcf_result else {"model": "dcf", "valid": False, "value": None, "warnings": ["SKIPPED"]},
            "comps": comps_result.to_dict() if comps_result else {"model": "comps", "valid": False, "value": None, "warnings": ["SKIPPED"]},
            "rim": rim_result.to_dict() if rim_result else {"model": "rim", "valid": False, "value": None, "warnings": ["SKIPPED"]},
        },
        "ensemble": ensemble.to_dict(),
        "data_quality": {
            "fetch_date": data.fetch_date,
            "warnings": list(set(all_warnings)),  # Dedupe
            "warning_count": len(set(all_warnings)),
        },
        "inputs": {
            "risk_free_rate": data.risk_free_rate,
            "market_premium": data.market_premium,
            "beta": data.beta,
            "cost_of_equity": data.cost_of_equity,
            "wacc": data.wacc,
            "sector": data.sector,
            "industry": data.industry,
        },
    }

    logger.info(
        "RESULT [%s]: Ensemble=$%s | Signal=%s | MOS=%s | Models=%d/%d valid",
        ticker,
        ensemble.ensemble_value,
        ensemble.signal,
        f"{ensemble.margin_of_safety:.2%}" if ensemble.margin_of_safety is not None else "N/A",
        ensemble.valid_model_count, 4,
    )

    return result


def run_batch_valuation(
    tickers: list[str],
    fetch_peers: bool = True,
) -> list[dict]:
    """
    Run valuation pipeline on a list of tickers.

    Args:
        tickers: List of ticker symbols
        fetch_peers: Whether to fetch peer data for comps

    Returns:
        List of valuation result dicts
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info("Processing %d/%d: %s", i, total, ticker)
        try:
            result = valuate_single_stock(ticker, fetch_peers=fetch_peers)
            results.append(result)
        except Exception as e:
            logger.error("FAILED %s: %s", ticker, e, exc_info=True)
            results.append({
                "ticker": ticker,
                "valuation_date": date.today().isoformat(),
                "ensemble": {"signal": "ERROR", "ensemble_value": None},
                "error": str(e),
            })

    return results
