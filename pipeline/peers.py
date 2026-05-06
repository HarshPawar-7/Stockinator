"""
Peer Discovery — FMP-Powered Dynamic Peer Grouping

Uses the FMP stable API to discover peers dynamically by:
1. Loading S&P 500 constituents filtered by sector (FMP /stable/sp500-constituent)
2. Fetching fundamentals for each candidate peer (FMP /stable/profile)
3. Filtering by market cap similarity

This is far superior to a hardcoded sector→ticker map since:
- Peers are pulled from the actual S&P 500 constituent list
- Any new addition or removal is automatically reflected
- Sector data comes from FMP's own classification, not Wikipedia
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from pipeline.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

PEER_CACHE_TTL = 86400       # 24 hours — peer groups don't change often
MAX_PEERS = 10
MARKET_CAP_RATIO_LIMIT = 10  # Filter peers within 10x market cap range
FMP_BASE = "https://financialmodelingprep.com/stable"


async def _get_sp500_by_sector(sector: str) -> list[str]:
    """
    Fetch S&P 500 tickers filtered by sector from FMP stable API.
    Cached in Redis for 24 hours.
    """
    cache_key = f"fmp_sector_peers:{sector}"
    cached = await get_cached(cache_key)
    if cached is not None:
        logger.debug("Cache hit for sector peers: %s", sector)
        return cached

    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{FMP_BASE}/sp500-constituent",
                params={"apikey": fmp_key}
            )
            resp.raise_for_status()
            data = resp.json()

        # Normalize sector comparison (FMP uses "Consumer Cyclical" etc.)
        sector_lower = sector.lower()
        tickers = [
            item["symbol"].replace(".", "-")
            for item in data
            if isinstance(item, dict)
            and item.get("sector", "").lower() == sector_lower
        ]

        logger.info("FMP S&P 500 sector '%s': %d tickers", sector, len(tickers))
        await set_cached(cache_key, tickers, PEER_CACHE_TTL)
        return tickers

    except Exception as e:
        logger.warning("FMP sector constituent fetch failed for '%s': %s", sector, e)
        return []


async def fetch_peers_from_fmp(ticker: str, sector: str = "") -> list[str]:
    """
    Fetch a list of peer tickers by looking up S&P 500 constituents in
    the same sector.

    Args:
        ticker: The subject stock ticker (e.g., "AAPL") — excluded from results
        sector: The sector to filter by (e.g., "Technology")

    Returns:
        List of peer ticker symbols (excluding the subject ticker).
    """
    if not sector:
        logger.warning("No sector provided for %s — cannot fetch sector peers", ticker)
        return []

    cache_key = f"fmp_peers:{ticker}:{sector}"
    cached = await get_cached(cache_key)
    if cached is not None:
        logger.debug("Cache hit for FMP peers: %s", ticker)
        return cached

    tickers = await _get_sp500_by_sector(sector)
    # Exclude the subject ticker itself
    peers = [t for t in tickers if t != ticker]

    logger.info("FMP returned %d sector peers for %s (sector=%s)", len(peers), ticker, sector)
    await set_cached(cache_key, peers, PEER_CACHE_TTL)
    return peers


async def fetch_peer_fundamentals(
    peer_tickers: list[str],
    base_market_cap: float | None = None,
    max_peers: int = MAX_PEERS,
) -> list[dict]:
    """
    Fetch key valuation multiples for a list of peer tickers using FMP stable/profile.

    Filters peers by market cap similarity (within MARKET_CAP_RATIO_LIMIT x).
    Uses asyncio.gather to fetch all peers concurrently.

    Args:
        peer_tickers: List of peer tickers to fetch
        base_market_cap: Market cap of the subject company for filtering
        max_peers: Maximum number of peers to return

    Returns:
        List of dicts with peer valuation metrics.
    """
    fmp_key = os.environ.get("FMP_API_KEY")

    async def _fetch_one_profile(pticker: str) -> dict | None:
        try:
            if not fmp_key:
                return None

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{FMP_BASE}/profile",
                    params={"symbol": pticker, "apikey": fmp_key}
                )
                resp.raise_for_status()
                data = resp.json()

            if not data:
                return None

            p = data[0] if isinstance(data, list) else data
            mcap = _parse_float(p.get("marketCap"))

            # Market cap filter: skip peers that are too different in size
            if base_market_cap and mcap:
                ratio = mcap / base_market_cap
                if ratio < (1 / MARKET_CAP_RATIO_LIMIT) or ratio > MARKET_CAP_RATIO_LIMIT:
                    logger.debug("Skipping peer %s: mcap ratio %.2fx out of range", pticker, ratio)
                    return None

            return {
                "ticker": pticker,
                "pe_ratio": _parse_float(p.get("pe")),
                "pb_ratio": _parse_float(p.get("priceToBook")),
                "ps_ratio": _parse_float(p.get("priceToSalesRatio")),
                "ev_ebitda_ratio": _parse_float(p.get("enterpriseValueOverEBITDA")),
                "ev_revenue_ratio": _parse_float(p.get("evToRevenue")),
                "market_cap": mcap,
                "sector": p.get("sector", ""),
            }
        except Exception as e:
            logger.warning("Failed to fetch profile for peer %s: %s", pticker, e)
            return None

    # Cap candidates to avoid excessive API calls
    candidates = peer_tickers[: max_peers * 3]
    tasks = [_fetch_one_profile(t) for t in candidates]
    raw_results = await asyncio.gather(*tasks)

    results = [r for r in raw_results if r is not None][:max_peers]
    logger.info(
        "Successfully fetched fundamentals for %d/%d peers",
        len(results), len(candidates)
    )
    return results


def _parse_float(val) -> float | None:
    """Safely parse a value to float."""
    try:
        if val is None or str(val).strip() in ("", "None", "N/A", "-"):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None
