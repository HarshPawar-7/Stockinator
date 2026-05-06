"""
S&P 500 Universe Loader

Provides the list of S&P 500 constituent tickers for batch valuation.

Source priority (most to least reliable):
  1. Redis cache (24-hour TTL)
  2. FMP /sp500_constituent API
  3. Wikipedia scrape (pandas read_html)
  4. Hardcoded top-50 fallback
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import pandas as pd

from pipeline.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

SP500_CACHE_KEY = "sp500_tickers"
SP500_CACHE_TTL = 86400  # 24 hours

# Top 50 by market cap as hardcoded fallback
_FALLBACK_TOP50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST",
    "JNJ", "HD", "ABBV", "WMT", "NFLX", "BAC", "KO", "MRK", "CVX",
    "CRM", "AMD", "PEP", "TMO", "LIN", "CSCO", "ADBE", "ACN", "MCD",
    "ABT", "WFC", "IBM", "PM", "GE", "ISRG", "NOW", "QCOM", "CAT",
    "INTU", "TXN", "GS", "AMGN", "BKNG", "DHR",
]


async def load_sp500_tickers(source: str = "auto") -> list[str]:
    """
    Load S&P 500 ticker list, with Redis caching.

    Args:
        source: "auto" (cache → FMP → Wikipedia → fallback),
                "fmp", "wiki", "file", or "fallback"

    Returns:
        List of ticker symbols
    """
    if source == "fallback":
        return list(_FALLBACK_TOP50)
    if source == "file":
        return _load_from_file()

    # Always check cache first for non-explicit sources
    if source in ("auto", "fmp", "wiki"):
        cached = await get_cached(SP500_CACHE_KEY)
        if cached:
            logger.info("S&P 500 tickers loaded from cache (%d tickers)", len(cached))
            return cached

    if source in ("auto", "fmp"):
        tickers = await _load_from_fmp()
        if tickers:
            await set_cached(SP500_CACHE_KEY, tickers, SP500_CACHE_TTL)
            return tickers

    if source in ("auto", "wiki"):
        tickers = await _load_from_wikipedia()
        if tickers:
            await set_cached(SP500_CACHE_KEY, tickers, SP500_CACHE_TTL)
            return tickers

    logger.warning("All S&P 500 sources failed — using hardcoded fallback")
    return list(_FALLBACK_TOP50)


async def _load_from_fmp() -> list[str]:
    """Fetch S&P 500 constituents from FMP API."""
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        logger.debug("FMP_API_KEY not set, skipping FMP S&P 500 load")
        return []

    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={fmp_key}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        tickers = [item["symbol"] for item in data if "symbol" in item]
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info("Loaded %d S&P 500 tickers from FMP", len(tickers))
        return tickers

    except Exception as e:
        logger.warning("FMP S&P 500 load failed: %s", e)
        return []


async def _load_from_wikipedia() -> list[str]:
    """Scrape S&P 500 list from Wikipedia (sync via thread)."""
    import asyncio
    try:
        def _scrape():
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df["Symbol"].tolist()
            return [t.replace(".", "-") for t in tickers]

        tickers = await asyncio.to_thread(_scrape)
        logger.info("Loaded %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers

    except Exception as e:
        logger.warning("Wikipedia S&P 500 load failed: %s", e)
        return []


def _load_from_file() -> list[str]:
    """Load tickers from a local file."""
    filepath = Path(__file__).parent.parent / "data" / "raw" / "sp500_tickers.txt"
    if filepath.exists():
        tickers = [
            line.strip().upper()
            for line in filepath.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        logger.info("Loaded %d tickers from %s", len(tickers), filepath)
        return tickers

    logger.warning("No ticker file found at %s — using fallback", filepath)
    return list(_FALLBACK_TOP50)


async def get_sector_tickers(sector: str) -> list[str]:
    """
    Get tickers filtered by GICS sector.
    Tries FMP first, then Wikipedia.
    """
    fmp_key = os.environ.get("FMP_API_KEY")
    cache_key = f"sector_tickers:{sector}"

    cached = await get_cached(cache_key)
    if cached:
        return cached

    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={fmp_key}"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            tickers = [
                item["symbol"].replace(".", "-")
                for item in data
                if item.get("sector", "").lower() == sector.lower()
            ]
            if tickers:
                await set_cached(cache_key, tickers, SP500_CACHE_TTL)
                return tickers
        except Exception as e:
            logger.warning("FMP sector filter failed: %s", e)

    # Fallback: Wikipedia
    try:
        import asyncio
        def _scrape_sector():
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            filtered = df[df["GICS Sector"] == sector]["Symbol"].tolist()
            return [t.replace(".", "-") for t in filtered]

        tickers = await asyncio.to_thread(_scrape_sector)
        if tickers:
            await set_cached(cache_key, tickers, SP500_CACHE_TTL)
        return tickers
    except Exception:
        return []
