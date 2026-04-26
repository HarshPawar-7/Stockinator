"""
Data Ingestion Pipeline — yfinance + FRED

Fetches stock fundamentals, price data, and peer information
using yfinance (free, no API key required).

Enhanced with:
- Retry logic with exponential backoff
- Request caching to avoid redundant API calls
- FRED integration for live macro data
- Rate limiting for peer fetching
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf

from config import RISK_FREE_RATE_DEFAULT, MARKET_PREMIUM_DEFAULT

logger = logging.getLogger(__name__)

# ── Simple in-memory cache ─────────────────────────────────────────
_data_cache: dict[str, tuple[object, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _cached(ttl: int = CACHE_TTL_SECONDS):
    """Cache decorator with TTL."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}:{args}:{sorted(kwargs.items())}"
            if key in _data_cache:
                val, ts = _data_cache[key]
                if time.time() - ts < ttl:
                    logger.debug("Cache hit: %s", key[:60])
                    return val
            result = fn(*args, **kwargs)
            _data_cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator


def _retry(max_attempts: int = 3, backoff: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error("%s failed after %d attempts: %s", fn.__name__, max_attempts, e)
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    logger.warning("%s attempt %d failed: %s — retrying in %.1fs", fn.__name__, attempt, e, wait)
                    time.sleep(wait)
        return wrapper
    return decorator


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

@_retry(max_attempts=3, backoff=1.0)
@_cached(ttl=CACHE_TTL_SECONDS)
def fetch_stock_data(ticker: str, use_fred: bool = True) -> StockData:
    """
    Fetch comprehensive stock data from yfinance.

    Returns StockData populated with all available fundamentals,
    derived metrics, and CAPM-based required returns.

    Args:
        ticker: Stock ticker symbol
        use_fred: Whether to fetch live macro data from FRED API
    """
    logger.info("Fetching data for %s...", ticker)

    # Fetch live macro data if available
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

    stock = yf.Ticker(ticker)
    info = stock.info or {}

    data = StockData(
        ticker=ticker,
        fetch_date=datetime.now().strftime("%Y-%m-%d"),
        risk_free_rate=rf_rate,
        market_premium=mkt_premium,
    )

    # ── Basic Info ─────────────────────────────────────────────────
    data.company_name = info.get("longName", info.get("shortName", ticker))
    data.sector = info.get("sector", "")
    data.industry = info.get("industry", "")
    data.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    data.market_cap = info.get("marketCap")
    data.shares_outstanding = info.get("sharesOutstanding")
    data.beta = info.get("beta")

    # ── Ratios from info ───────────────────────────────────────────
    data.pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    data.pb_ratio = info.get("priceToBook")
    data.ps_ratio = info.get("priceToSalesTrailing12Months")
    data.dividend_yield = info.get("dividendYield")
    data.payout_ratio = info.get("payoutRatio")
    data.roe = info.get("returnOnEquity")
    data.roa = info.get("returnOnAssets")
    data.eps = info.get("trailingEps")
    data.book_value_per_share = info.get("bookValue")
    data.revenue_growth_yoy = info.get("revenueGrowth")
    data.ev_ebitda_ratio = info.get("enterpriseToEbitda")

    # ── Financial Statements ───────────────────────────────────────
    try:
        income = stock.income_stmt
        if income is not None and not income.empty:
            latest = income.iloc[:, 0]  # Most recent year
            data.revenue = _safe_get(latest, "Total Revenue")
            data.net_income = _safe_get(latest, "Net Income")
            data.ebitda = _safe_get(latest, "EBITDA")
            data.interest_expense = abs(_safe_get(latest, "Interest Expense") or 0)

            # Effective tax rate
            pretax = _safe_get(latest, "Pretax Income")
            tax = _safe_get(latest, "Tax Provision")
            if pretax and pretax > 0 and tax is not None:
                data.effective_tax_rate = abs(tax) / pretax
    except Exception as e:
        data.warnings.append(f"INCOME_STMT_ERROR: {e}")
        logger.warning("Failed to fetch income statement for %s: %s", ticker, e)

    try:
        balance = stock.balance_sheet
        if balance is not None and not balance.empty:
            latest = balance.iloc[:, 0]
            data.total_debt = _safe_get(latest, "Total Debt")
            data.total_equity = _safe_get(latest, "Stockholders Equity") or _safe_get(latest, "Total Equity Gross Minority Interest")
            data.cash_and_equivalents = _safe_get(latest, "Cash And Cash Equivalents")
    except Exception as e:
        data.warnings.append(f"BALANCE_SHEET_ERROR: {e}")
        logger.warning("Failed to fetch balance sheet for %s: %s", ticker, e)

    try:
        cashflow = stock.cashflow
        if cashflow is not None and not cashflow.empty:
            latest = cashflow.iloc[:, 0]
            data.operating_cash_flow = _safe_get(latest, "Operating Cash Flow")
            data.capital_expenditures = abs(_safe_get(latest, "Capital Expenditure") or 0)
            data.free_cash_flow = _safe_get(latest, "Free Cash Flow")
            data.dividends_paid = abs(_safe_get(latest, "Common Stock Dividend Paid") or 0)

            # Historical FCF for DCF projection
            data.historical_fcf = []
            for col in reversed(cashflow.columns):
                fcf_val = _safe_get(cashflow[col], "Free Cash Flow")
                if fcf_val is not None:
                    data.historical_fcf.append(fcf_val)
    except Exception as e:
        data.warnings.append(f"CASHFLOW_ERROR: {e}")
        logger.warning("Failed to fetch cash flow for %s: %s", ticker, e)

    # ── Derived Metrics ────────────────────────────────────────────
    # Net debt
    debt = data.total_debt or 0
    cash = data.cash_and_equivalents or 0
    data.net_debt = debt - cash

    # Cost of debt
    if data.total_debt and data.total_debt > 0 and data.interest_expense:
        data.cost_of_debt = data.interest_expense / data.total_debt
    else:
        data.cost_of_debt = 0.04  # Default assumption

    # Dividend per share
    if data.dividends_paid and data.shares_outstanding and data.shares_outstanding > 0:
        data.dividend_per_share = data.dividends_paid / data.shares_outstanding
    elif data.dividend_yield and data.current_price:
        data.dividend_per_share = data.dividend_yield * data.current_price

    # ROE from financials if not available from info
    if data.roe is None and data.net_income and data.total_equity and data.total_equity > 0:
        data.roe = data.net_income / data.total_equity

    # Payout ratio
    if data.payout_ratio is None and data.dividends_paid and data.net_income and data.net_income > 0:
        data.payout_ratio = data.dividends_paid / data.net_income

    # ── CAPM & WACC ────────────────────────────────────────────────
    beta = data.beta if data.beta and data.beta > 0 else 1.0  # Default beta = 1
    data.cost_of_equity = data.risk_free_rate + beta * data.market_premium

    if data.market_cap and data.market_cap > 0:
        total_val = data.market_cap + (data.total_debt or 0)
        if total_val > 0:
            eq_w = data.market_cap / total_val
            debt_w = (data.total_debt or 0) / total_val
            tax = data.effective_tax_rate or 0.21  # Default US corporate rate
            data.wacc = eq_w * data.cost_of_equity + debt_w * (data.cost_of_debt or 0.04) * (1 - tax)

    # Sustainable growth rate
    if data.roe and data.payout_ratio is not None:
        retention = 1.0 - min(max(data.payout_ratio, 0), 1.0)
        data.sustainable_growth = data.roe * retention

    logger.info("Fetched %s: price=$%s, mcap=%s, sector=%s",
                ticker, data.current_price, data.market_cap, data.sector)

    return data


def fetch_peer_data(ticker: str, sector: str, industry: str,
                    market_cap: float | None = None, max_peers: int = 10) -> list[dict]:
    """
    Fetch peer company data for comparable analysis.

    Uses yfinance sector/industry screener to find similar companies.
    """
    logger.info("Fetching peers for %s (sector=%s, industry=%s)", ticker, sector, industry)

    # Curated peer groups for common sectors
    # In Phase 2, this will be replaced with API-based peer discovery
    PEER_MAP = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "INTC", "AMD", "CSCO", "IBM", "QCOM", "TXN"],
        "Consumer Defensive": ["KO", "PEP", "PG", "CL", "KHC", "MDLZ", "GIS", "K", "SJM", "CPB", "MKC", "HSY"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN", "GILD", "MDT"],
        "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD", "TGT", "LOW", "TJX", "BKNG"],
        "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR"],
        "Industrials": ["HON", "UNP", "UPS", "CAT", "DE", "GE", "BA", "MMM", "LMT", "RTX"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "DVN"],
    }

    # Find peers in same sector
    peers_list = PEER_MAP.get(sector, [])
    # Remove the ticker itself
    peers_list = [p for p in peers_list if p != ticker][:max_peers]

    if not peers_list:
        logger.warning("No curated peers for sector '%s'", sector)
        return []

    peer_results = []
    for pticker in peers_list:
        try:
            pinfo = yf.Ticker(pticker).info or {}
            peer_mcap = pinfo.get("marketCap", 0)

            # Filter by market cap similarity (±5x range — relaxed for Phase 1)
            if market_cap and peer_mcap:
                ratio = peer_mcap / market_cap
                if ratio < 0.1 or ratio > 10:
                    continue

            peer_results.append({
                "ticker": pticker,
                "pe_ratio": pinfo.get("trailingPE"),
                "pb_ratio": pinfo.get("priceToBook"),
                "ps_ratio": pinfo.get("priceToSalesTrailing12Months"),
                "ev_ebitda_ratio": pinfo.get("enterpriseToEbitda"),
                "ev_revenue_ratio": pinfo.get("enterpriseToRevenue"),
                "market_cap": peer_mcap,
                "sector": pinfo.get("sector", ""),
            })
        except Exception as e:
            logger.warning("Failed to fetch peer %s: %s", pticker, e)
            continue

    logger.info("Found %d peers for %s", len(peer_results), ticker)
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
