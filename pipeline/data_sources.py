import httpx
import os
import logging

logger = logging.getLogger(__name__)

async def fetch_fundamentals(ticker: str) -> dict:
    """Try sources in order. Return first successful result containing full financial profiles."""
    sources = [
        _fetch_from_fmp,
        _fetch_from_alpha_vantage,
    ]
    for source in sources:
        try:
            logger.info(f"Attempting to fetch {ticker} using {source.__name__}...")
            data = await source(ticker)
            if data:
                return data
        except Exception as e:
            logger.warning(f"[WARN] {source.__name__} failed for {ticker}: {e}")
            
    raise ValueError(f"All data sources exhausted for {ticker}")

async def _fetch_from_fmp(ticker: str) -> dict:
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        raise ValueError("FMP_API_KEY not set")
        
    base_url = "https://financialmodelingprep.com/api/v3"
    params = {"apikey": fmp_key}
    
    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Profile
        r_prof = await client.get(f"{base_url}/profile/{ticker}", params=params)
        r_prof.raise_for_status()
        prof_data = r_prof.json()
        if not prof_data:
            raise ValueError(f"FMP profile empty for {ticker}")
            
        # 2. Income Statement
        r_inc = await client.get(f"{base_url}/income-statement/{ticker}", params={"limit": 1, **params})
        r_inc.raise_for_status()
        inc_data = r_inc.json()
        
        # 3. Balance Sheet
        r_bal = await client.get(f"{base_url}/balance-sheet-statement/{ticker}", params={"limit": 1, **params})
        r_bal.raise_for_status()
        bal_data = r_bal.json()
        
        # 4. Cash Flow (fetch a few years for historical FCF)
        r_cf = await client.get(f"{base_url}/cash-flow-statement/{ticker}", params={"limit": 5, **params})
        r_cf.raise_for_status()
        cf_data = r_cf.json()

        return {
            "source": "fmp",
            "profile": prof_data[0] if prof_data else {},
            "income": inc_data[0] if inc_data else {},
            "balance": bal_data[0] if bal_data else {},
            "cashflow_history": cf_data,
            "cashflow": cf_data[0] if cf_data else {}
        }

async def _fetch_from_alpha_vantage(ticker: str) -> dict:
    av_key = os.environ.get("ALPHA_VANTAGE_KEY")
    if not av_key:
        raise ValueError("ALPHA_VANTAGE_KEY not set")
        
    base_url = "https://www.alphavantage.co/query"
    
    async with httpx.AsyncClient(timeout=10) as client:
        # Overview
        r_over = await client.get(base_url, params={"function": "OVERVIEW", "symbol": ticker, "apikey": av_key})
        r_over.raise_for_status()
        over_data = r_over.json()
        if "Symbol" not in over_data:
            raise ValueError(f"Alpha Vantage Overview failed for {ticker}")

        # Income Statement
        r_inc = await client.get(base_url, params={"function": "INCOME_STATEMENT", "symbol": ticker, "apikey": av_key})
        r_inc.raise_for_status()
        inc_data = r_inc.json()
        
        # Balance Sheet
        r_bal = await client.get(base_url, params={"function": "BALANCE_SHEET", "symbol": ticker, "apikey": av_key})
        r_bal.raise_for_status()
        bal_data = r_bal.json()
        
        # Cash Flow
        r_cf = await client.get(base_url, params={"function": "CASH_FLOW", "symbol": ticker, "apikey": av_key})
        r_cf.raise_for_status()
        cf_data = r_cf.json()

        return {
            "source": "alpha_vantage",
            "overview": over_data,
            "income": inc_data.get("annualReports", [{}])[0],
            "balance": bal_data.get("annualReports", [{}])[0],
            "cashflow_history": cf_data.get("annualReports", []),
            "cashflow": cf_data.get("annualReports", [{}])[0]
        }
