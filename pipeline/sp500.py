"""
S&P 500 Universe Loader

Provides the list of S&P 500 constituent tickers for batch valuation.
Uses Wikipedia's S&P 500 list page via pandas read_html, with a
hardcoded fallback for offline use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Top 50 by market cap as hardcoded fallback
_FALLBACK_TOP50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST",
    "JNJ", "HD", "ABBV", "WMT", "NFLX", "BAC", "KO", "MRK", "CVX",
    "CRM", "AMD", "PEP", "TMO", "LIN", "CSCO", "ADBE", "ACN", "MCD",
    "ABT", "WFC", "IBM", "PM", "GE", "ISRG", "NOW", "QCOM", "CAT",
    "INTU", "TXN", "GS", "AMGN", "BKNG", "DHR",
]


def load_sp500_tickers(source: str = "wiki") -> list[str]:
    """
    Load S&P 500 ticker list.

    Args:
        source: "wiki" (scrape Wikipedia), "file" (from local CSV), or "fallback"

    Returns:
        List of ticker symbols
    """
    if source == "wiki":
        return _load_from_wikipedia()
    elif source == "file":
        return _load_from_file()
    else:
        return list(_FALLBACK_TOP50)


def _load_from_wikipedia() -> list[str]:
    """Scrape S&P 500 list from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]

        # The ticker column is usually "Symbol"
        tickers = df["Symbol"].tolist()
        # Clean: replace dots with hyphens (BRK.B → BRK-B for yfinance)
        tickers = [t.replace(".", "-") for t in tickers]

        logger.info("Loaded %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers

    except Exception as e:
        logger.warning("Failed to load from Wikipedia: %s — using fallback", e)
        return list(_FALLBACK_TOP50)


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


def get_sector_tickers(sector: str) -> list[str]:
    """Get tickers filtered by sector (requires wiki load)."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        filtered = df[df["GICS Sector"] == sector]["Symbol"].tolist()
        return [t.replace(".", "-") for t in filtered]
    except Exception:
        return []
