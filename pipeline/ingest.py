"""
Data Ingestion Pipeline — FMP + FRED

Fetches stock fundamentals, price data, and peer information
using Financial Modeling Prep (primary) and Alpha Vantage (fallback).

Enhanced with:
- Async I/O for non-blocking concurrent fetching
- Redis-backed caching to avoid redundant API calls
- FRED integration for live macro data
- Dynamic peer discovery via FMP /stock_peers
"""

from __future__ import annotations

import logging
import time
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE_DEFAULT, MARKET_PREMIUM_DEFAULT
from pipeline.cache import get_cached, set_cached

logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class StockData:
    """Comprehensive stock data for valuation models."""
    ticker: str
    fetch_date: str

    # Price
    current_price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None

    # Income Statement
    revenue: float | None = None
    net_income: float | None = None
    ebitda: float | None = None
    eps: float | None = None
    interest_expense: float | None = None

    # Balance Sheet
    total_debt: float | None = None
    total_equity: float | None = None
    book_value_per_share: float | None = None
    cash_and_equivalents: float | None = None

    # Cash Flow
    operating_cash_flow: float | None = None
    capital_expenditures: float | None = None
    free_cash_flow: float | None = None
    dividends_paid: float | None = None
    historical_fcf: list[float] = field(default_factory=list)

    # Derived Metrics
    roe: float | None = None
    roa: float | None = None
    dividend_per_share: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda_ratio: float | None = None
    net_debt: float | None = None
    effective_tax_rate: float | None = None
    cost_of_debt: float | None = None
    revenue_growth_yoy: float | None = None

    # CAPM / Required Returns
    risk_free_rate: float = RISK_FREE_RATE_DEFAULT
    market_premium: float = MARKET_PREMIUM_DEFAULT
    cost_of_equity: float | None = None
    wacc: float | None = None

    # Sustainable Growth
    sustainable_growth: float | None = None

    # Sector / Industry
    sector: str = ""
    industry: str = ""
    company_name: str = ""

    # Data quality
    warnings: list[str] = field(default_factory=list)

async def fetch_stock_data(ticker: str, use_fred: bool = True) -> StockData:
    """
    Fetch comprehensive stock data from reliable API sources.
    
    Returns StockData populated with all available fundamentals,
    derived metrics, and CAPM-based required returns.
    """
    cache_key = f"fetch_stock_data:{ticker}:{use_fred}"
    cached = await get_cached(cache_key)
    if cached:
        logger.debug("Cache hit for %s", ticker)
        return StockData(**cached)

    logger.info("Fetching data for %s...", ticker)
    
    rf_rate = RISK_FREE_RATE_DEFAULT
    mkt_premium = MARKET_PREMIUM_DEFAULT
    if use_fred:
        try:
            from pipeline.fred_api import fetch_macro_data
            macro = fetch_macro_data()
            rf_rate = macro.risk_free_rate
            logger.debug("Using FRED Rf=%.4f", rf_rate)
        except Exception as e:
            logger.debug("FRED unavailable, using defaults: %s", e)
            
    data = StockData(
        ticker=ticker,
        fetch_date=datetime.now().strftime("%Y-%m-%d"),
        risk_free_rate=rf_rate,
        market_premium=mkt_premium,
    )
    
    from pipeline.data_sources import fetch_fundamentals
    try:
        raw = await fetch_fundamentals(ticker)
    except Exception as e:
        data.warnings.append(f"FETCH_ERROR: {e}")
        logger.error(f"Failed to fetch {ticker}: {e}")
        return data

    source = raw.get("source")
    
    def _parse_float(val):
        try:
            if val is None or str(val).strip() == "" or val == "None":
                return None
            return float(val)
        except:
            return None
            
    if source == "fmp":
        p = raw.get("profile", {})
        i = raw.get("income", {})
        b = raw.get("balance", {})
        c = raw.get("cashflow", {})
        ch = raw.get("cashflow_history", [])
        
        data.company_name = p.get("companyName", ticker)
        data.sector = p.get("sector", "")
        data.industry = p.get("industry", "")
        data.current_price = _parse_float(p.get("price"))
        data.market_cap = _parse_float(p.get("mktCap"))
        data.beta = _parse_float(p.get("beta"))
        
        data.revenue = _parse_float(i.get("revenue"))
        data.net_income = _parse_float(i.get("netIncome"))
        data.ebitda = _parse_float(i.get("ebitda"))
        data.eps = _parse_float(i.get("eps"))
        data.interest_expense = abs(_parse_float(i.get("interestExpense")) or 0)
        
        tax_prov = _parse_float(i.get("incomeTaxExpense"))
        pretax = _parse_float(i.get("incomeBeforeTax"))
        if tax_prov and pretax and pretax > 0:
            data.effective_tax_rate = abs(tax_prov) / pretax
            
        data.total_debt = _parse_float(b.get("totalDebt"))
        data.total_equity = _parse_float(b.get("totalStockholdersEquity"))
        data.cash_and_equivalents = _parse_float(b.get("cashAndCashEquivalents"))
        
        data.operating_cash_flow = _parse_float(c.get("operatingCashFlow"))
        data.capital_expenditures = abs(_parse_float(c.get("capitalExpenditure")) or 0)
        data.free_cash_flow = _parse_float(c.get("freeCashFlow"))
        data.dividends_paid = abs(_parse_float(c.get("dividendsPaid")) or 0)
        
        data.historical_fcf = []
        for year_cf in ch:
            fcf = _parse_float(year_cf.get("freeCashFlow"))
            if fcf is not None:
                data.historical_fcf.append(fcf)
                
    elif source == "alpha_vantage":
        o = raw.get("overview", {})
        i = raw.get("income", {})
        b = raw.get("balance", {})
        c = raw.get("cashflow", {})
        ch = raw.get("cashflow_history", [])
        
        data.company_name = o.get("Name", ticker)
        data.sector = o.get("Sector", "")
        data.industry = o.get("Industry", "")
        data.market_cap = _parse_float(o.get("MarketCapitalization"))
        data.beta = _parse_float(o.get("Beta"))
        data.pe_ratio = _parse_float(o.get("PERatio"))
        data.pb_ratio = _parse_float(o.get("PriceToBookRatio"))
        data.ps_ratio = _parse_float(o.get("PriceToSalesRatioTTM"))
        data.dividend_yield = _parse_float(o.get("DividendYield"))
        data.eps = _parse_float(o.get("EPS"))
        data.roe = _parse_float(o.get("ReturnOnEquityTTM"))
        data.roa = _parse_float(o.get("ReturnOnAssetsTTM"))
        data.ev_ebitda_ratio = _parse_float(o.get("EVToEBITDA"))
        
        data.revenue = _parse_float(i.get("totalRevenue"))
        data.net_income = _parse_float(i.get("netIncome"))
        data.ebitda = _parse_float(i.get("ebitda"))
        data.interest_expense = abs(_parse_float(i.get("interestAndDebtExpense")) or 0)
        
        tax_prov = _parse_float(i.get("incomeTaxExpense"))
        pretax = _parse_float(i.get("incomeBeforeTax"))
        if tax_prov and pretax and pretax > 0:
            data.effective_tax_rate = abs(tax_prov) / pretax
            
        data.total_debt = _parse_float(b.get("shortLongTermDebtTotal")) or _parse_float(b.get("longTermDebtNoncurrent"))
        data.total_equity = _parse_float(b.get("totalShareholderEquity"))
        data.cash_and_equivalents = _parse_float(b.get("cashAndCashEquivalentsAtCarryingValue"))
        
        data.operating_cash_flow = _parse_float(c.get("operatingCashflow"))
        data.capital_expenditures = abs(_parse_float(c.get("capitalExpenditures")) or 0)
        data.free_cash_flow = _parse_float(c.get("operatingCashflow")) - data.capital_expenditures if data.operating_cash_flow is not None else None
        data.dividends_paid = abs(_parse_float(c.get("dividendPayout")) or 0)
        
        data.historical_fcf = []
        for year_cf in ch:
            ocf = _parse_float(year_cf.get("operatingCashflow"))
            cap = abs(_parse_float(year_cf.get("capitalExpenditures")) or 0)
            if ocf is not None:
                data.historical_fcf.append(ocf - cap)
                
        # To get current price for AV, we need to fetch it separately or calculate from MarketCap / Shares
        shares = _parse_float(o.get("SharesOutstanding"))
        if data.market_cap and shares and shares > 0:
            data.current_price = data.market_cap / shares
            data.shares_outstanding = shares
            
    # Derive missing info (shared logic)
    debt = data.total_debt or 0
    cash = data.cash_and_equivalents or 0
    data.net_debt = debt - cash

    if data.total_debt and data.total_debt > 0 and data.interest_expense:
        data.cost_of_debt = data.interest_expense / data.total_debt
    else:
        data.cost_of_debt = 0.04 

    if data.dividends_paid and data.shares_outstanding and data.shares_outstanding > 0:
        data.dividend_per_share = data.dividends_paid / data.shares_outstanding
    elif data.dividend_yield and data.current_price:
        data.dividend_per_share = data.dividend_yield * data.current_price

    if data.roe is None and data.net_income and data.total_equity and data.total_equity > 0:
        data.roe = data.net_income / data.total_equity

    if data.payout_ratio is None and data.dividends_paid and data.net_income and data.net_income > 0:
        data.payout_ratio = data.dividends_paid / data.net_income

    beta = data.beta if data.beta and data.beta > 0 else 1.0  
    data.cost_of_equity = data.risk_free_rate + beta * data.market_premium

    if data.market_cap and data.market_cap > 0:
        total_val = data.market_cap + (data.total_debt or 0)
        if total_val > 0:
            eq_w = data.market_cap / total_val
            debt_w = (data.total_debt or 0) / total_val
            tax = data.effective_tax_rate or 0.21  
            data.wacc = eq_w * data.cost_of_equity + debt_w * (data.cost_of_debt or 0.04) * (1 - tax)

    if data.roe and data.payout_ratio is not None:
        retention = 1.0 - min(max(data.payout_ratio, 0), 1.0)
        data.sustainable_growth = data.roe * retention

    logger.info("Fetched %s: price=$%s, mcap=%s, sector=%s",
                ticker, data.current_price, data.market_cap, data.sector)

    await set_cached(cache_key, asdict(data), CACHE_TTL_SECONDS)
    return data


async def fetch_peer_data(
    ticker: str,
    sector: str,
    industry: str,
    market_cap: float | None = None,
    max_peers: int = 10,
) -> list[dict]:
    """
    Fetch peer company data for comparable analysis.

    Uses FMP's /stock_peers endpoint for dynamic, accurate peer discovery.
    Results are cached in Redis for 24 hours.
    """
    from pipeline.peers import fetch_peers_from_fmp, fetch_peer_fundamentals

    cache_key = f"fetch_peer_data:{ticker}:{sector}:{max_peers}"
    cached = await get_cached(cache_key)
    if cached is not None:
        logger.debug("Cache hit for peer data: %s", ticker)
        return cached

    logger.info("Fetching peers for %s via FMP (sector=%s)", ticker, sector)

    peer_tickers = await fetch_peers_from_fmp(ticker, sector=sector)

    if not peer_tickers:
        logger.warning("No peers found from FMP for %s — comps unavailable", ticker)
        return []

    peer_results = await fetch_peer_fundamentals(
        peer_tickers=peer_tickers,
        base_market_cap=market_cap,
        max_peers=max_peers,
    )

    logger.info("Found %d peers for %s", len(peer_results), ticker)
    await set_cached(cache_key, peer_results, CACHE_TTL_SECONDS)
    return peer_results


def _safe_get(series: pd.Series, key: str) -> float | None:
    """Safely extract a value from a pandas Series."""
    try:
        val = series.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None
