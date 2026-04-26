"""
FRED API Integration — Federal Reserve Economic Data

Free API for macroeconomic data:
- Risk-free rate (10Y US Treasury yield — DGS10)
- Yield spread (10Y - 2Y Treasury — T10Y2Y)
- CPI (inflation)
- GDP growth
- Federal Funds Rate

Requires free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
Falls back to config defaults if key is not set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests

from config import RISK_FREE_RATE_DEFAULT, MARKET_PREMIUM_DEFAULT

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Cache to avoid hitting API repeatedly
_cache: dict[str, tuple[float, datetime]] = {}
CACHE_TTL_HOURS = 6


@dataclass
class MacroData:
    """Macroeconomic data snapshot."""
    risk_free_rate: float          # 10Y Treasury (DGS10)
    yield_spread_10_2: float | None  # 10Y - 2Y spread (T10Y2Y)
    federal_funds_rate: float | None  # FEDFUNDS
    cpi_yoy: float | None           # CPI year-over-year
    gdp_growth: float | None        # Real GDP growth
    market_premium: float           # Equity risk premium
    fetch_date: str
    source: str                     # "fred_api" or "defaults"

    def to_dict(self) -> dict:
        return {
            "risk_free_rate": self.risk_free_rate,
            "yield_spread_10_2": self.yield_spread_10_2,
            "federal_funds_rate": self.federal_funds_rate,
            "cpi_yoy": self.cpi_yoy,
            "gdp_growth": self.gdp_growth,
            "market_premium": self.market_premium,
            "fetch_date": self.fetch_date,
            "source": self.source,
        }


def _fetch_fred_series(series_id: str) -> float | None:
    """
    Fetch the latest observation for a FRED series.

    Uses caching to avoid redundant API calls within CACHE_TTL_HOURS.
    """
    if not FRED_API_KEY:
        return None

    # Check cache
    if series_id in _cache:
        value, cached_at = _cache[series_id]
        if datetime.now() - cached_at < timedelta(hours=CACHE_TTL_HOURS):
            logger.debug("FRED cache hit: %s = %s", series_id, value)
            return value

    try:
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "sort_order": "desc",
            "limit": 5,  # Get last 5 in case latest is missing
            "file_type": "json",
        }
        resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        for obs in observations:
            val_str = obs.get("value", ".")
            if val_str != ".":
                value = float(val_str)
                _cache[series_id] = (value, datetime.now())
                logger.info("FRED %s = %.4f (date: %s)", series_id, value, obs.get("date"))
                return value

        logger.warning("FRED %s: no valid observations found", series_id)
        return None

    except requests.RequestException as e:
        logger.warning("FRED API error for %s: %s", series_id, e)
        return None
    except (ValueError, KeyError) as e:
        logger.warning("FRED parse error for %s: %s", series_id, e)
        return None


def fetch_macro_data() -> MacroData:
    """
    Fetch current macroeconomic data from FRED API.

    Falls back to config defaults if API key is not set or API fails.

    Series used:
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - T10Y2Y: 10-Year minus 2-Year Treasury spread
    - FEDFUNDS: Federal Funds Effective Rate
    - CPIAUCSL: Consumer Price Index (for YoY calc)
    - A191RL1Q225SBEA: Real GDP growth (quarterly, annualized)
    """
    fetch_date = datetime.now().strftime("%Y-%m-%d")

    if not FRED_API_KEY:
        logger.info("No FRED_API_KEY set — using default macro values")
        return MacroData(
            risk_free_rate=RISK_FREE_RATE_DEFAULT,
            yield_spread_10_2=None,
            federal_funds_rate=None,
            cpi_yoy=None,
            gdp_growth=None,
            market_premium=MARKET_PREMIUM_DEFAULT,
            fetch_date=fetch_date,
            source="defaults",
        )

    # Fetch individual series
    dgs10 = _fetch_fred_series("DGS10")
    t10y2y = _fetch_fred_series("T10Y2Y")
    fedfunds = _fetch_fred_series("FEDFUNDS")
    gdp = _fetch_fred_series("A191RL1Q225SBEA")

    # Risk-free rate (convert from percentage to decimal)
    rf = dgs10 / 100.0 if dgs10 is not None else RISK_FREE_RATE_DEFAULT

    # Yield spread
    spread = t10y2y / 100.0 if t10y2y is not None else None

    # Fed funds rate
    ffr = fedfunds / 100.0 if fedfunds is not None else None

    # GDP growth (already in percentage form from FRED)
    gdp_growth = gdp / 100.0 if gdp is not None else None

    logger.info(
        "Macro data: Rf=%.4f, spread=%s, FFR=%s, GDP=%s",
        rf,
        f"{spread:.4f}" if spread else "N/A",
        f"{ffr:.4f}" if ffr else "N/A",
        f"{gdp_growth:.4f}" if gdp_growth else "N/A",
    )

    return MacroData(
        risk_free_rate=rf,
        yield_spread_10_2=spread,
        federal_funds_rate=ffr,
        cpi_yoy=None,  # Requires multi-observation calc
        gdp_growth=gdp_growth,
        market_premium=MARKET_PREMIUM_DEFAULT,
        fetch_date=fetch_date,
        source="fred_api",
    )


def get_risk_free_rate() -> float:
    """Quick helper — just get the current risk-free rate."""
    macro = fetch_macro_data()
    return macro.risk_free_rate
