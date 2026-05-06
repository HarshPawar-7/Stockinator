"""
PostgreSQL Database — Schema & Operations

Persistent storage for fundamentals, valuations, and macro data.
Uses asyncpg for high-performance async database operations.
"""

from __future__ import annotations

import json
import logging
import os
import asyncpg
from datetime import date

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          TEXT NOT NULL,
    report_date     DATE NOT NULL,
    revenue         REAL,
    net_income      REAL,
    ebitda          REAL,
    fcf             REAL,
    total_debt      REAL,
    total_equity    REAL,
    dividends_paid  REAL,
    capex           REAL,
    roe             REAL,
    eps             REAL,
    shares_outstanding REAL,
    market_cap      REAL,
    sector          TEXT,
    industry        TEXT,
    PRIMARY KEY (ticker, report_date)
);

CREATE TABLE IF NOT EXISTS valuations (
    ticker          TEXT NOT NULL,
    valuation_date  DATE NOT NULL,
    price           REAL,
    ggm_value       REAL,
    dcf_value       REAL,
    comps_value     REAL,
    rim_value       REAL,
    ensemble_value  REAL,
    ci_lower_95     REAL,
    ci_upper_95     REAL,
    margin_safety   REAL,
    signal          TEXT,
    model_flags     TEXT,
    raw_json        TEXT,
    PRIMARY KEY (ticker, valuation_date)
);

CREATE TABLE IF NOT EXISTS macro (
    date            DATE PRIMARY KEY,
    risk_free_rate  REAL,
    market_premium  REAL,
    cpi_yoy         REAL,
    gdp_growth      REAL
);
"""


async def get_db_pool() -> asyncpg.Pool:
    """Initialize database pool and create tables if needed."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
        
    logger.info("PostgreSQL Database pool initialized")
    return pool


async def save_valuation(pool: asyncpg.Pool, result: dict) -> None:
    """Save a single valuation result to the database."""
    ensemble = result.get("ensemble", {})
    models = result.get("models", {})

    query = """
        INSERT INTO valuations
        (ticker, valuation_date, price, ggm_value, dcf_value, comps_value,
         rim_value, ensemble_value, ci_lower_95, ci_upper_95,
         margin_safety, signal, model_flags, raw_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (ticker, valuation_date) DO UPDATE SET
            price = EXCLUDED.price,
            ggm_value = EXCLUDED.ggm_value,
            dcf_value = EXCLUDED.dcf_value,
            comps_value = EXCLUDED.comps_value,
            rim_value = EXCLUDED.rim_value,
            ensemble_value = EXCLUDED.ensemble_value,
            ci_lower_95 = EXCLUDED.ci_lower_95,
            ci_upper_95 = EXCLUDED.ci_upper_95,
            margin_safety = EXCLUDED.margin_safety,
            signal = EXCLUDED.signal,
            model_flags = EXCLUDED.model_flags,
            raw_json = EXCLUDED.raw_json
    """
    
    val_date = result.get("valuation_date")
    if isinstance(val_date, str):
        val_date = date.fromisoformat(val_date)

    ci = ensemble.get("ci_95") or [None, None]
    
    async with pool.acquire() as conn:
        await conn.execute(
            query,
            result.get("ticker"),
            val_date,
            result.get("market_price"),
            models.get("ggm", {}).get("value"),
            models.get("dcf", {}).get("value"),
            models.get("comps", {}).get("value"),
            models.get("rim", {}).get("value"),
            ensemble.get("ensemble_value"),
            ci[0],
            ci[1],
            ensemble.get("margin_of_safety"),
            ensemble.get("signal"),
            json.dumps(ensemble.get("warnings", [])),
            json.dumps(result),
        )


async def save_batch(pool: asyncpg.Pool, results: list[dict]) -> None:
    """Save a batch of valuation results."""
    for result in results:
        try:
            await save_valuation(pool, result)
        except Exception as e:
            logger.error("Failed to save %s: %s", result.get("ticker"), e)


async def get_latest_valuation(pool: asyncpg.Pool, ticker: str) -> dict | None:
    """Get most recent valuation for a ticker."""
    query = "SELECT raw_json FROM valuations WHERE ticker=$1 ORDER BY valuation_date DESC LIMIT 1"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, ticker)
        return json.loads(row["raw_json"]) if row else None


async def get_all_latest(pool: asyncpg.Pool) -> list[dict]:
    """Get the latest valuation for all tickers."""
    query = """
        SELECT v.raw_json FROM valuations v
        INNER JOIN (
            SELECT ticker, MAX(valuation_date) as max_date
            FROM valuations GROUP BY ticker
        ) latest ON v.ticker = latest.ticker AND v.valuation_date = latest.max_date
        ORDER BY v.ticker
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [json.loads(row["raw_json"]) for row in rows]


async def get_history(pool: asyncpg.Pool, ticker: str, limit: int = 30) -> list[dict]:
    """Get valuation history for a ticker."""
    query = "SELECT raw_json FROM valuations WHERE ticker=$1 ORDER BY valuation_date DESC LIMIT $2"
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, ticker, limit)
        return [json.loads(row["raw_json"]) for row in rows]
