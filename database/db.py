"""
SQLite Database — Schema & Operations

Persistent storage for fundamentals, valuations, and macro data.
SQLite used throughout (lightweight, zero-config, file-based).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

from config import DB_PATH

logger = logging.getLogger(__name__)


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


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Initialize database and create tables if needed."""
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Database initialized at %s", path)
    return conn


def save_valuation(conn: sqlite3.Connection, result: dict) -> None:
    """Save a single valuation result to the database."""
    ensemble = result.get("ensemble", {})
    models = result.get("models", {})

    conn.execute(
        """INSERT OR REPLACE INTO valuations
           (ticker, valuation_date, price, ggm_value, dcf_value, comps_value,
            rim_value, ensemble_value, ci_lower_95, ci_upper_95,
            margin_safety, signal, model_flags, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            result.get("ticker"),
            result.get("valuation_date"),
            result.get("market_price"),
            models.get("ggm", {}).get("value"),
            models.get("dcf", {}).get("value"),
            models.get("comps", {}).get("value"),
            models.get("rim", {}).get("value"),
            ensemble.get("ensemble_value"),
            ensemble.get("ci_95", [None, None])[0] if ensemble.get("ci_95") else None,
            ensemble.get("ci_95", [None, None])[1] if ensemble.get("ci_95") else None,
            ensemble.get("margin_of_safety"),
            ensemble.get("signal"),
            json.dumps(ensemble.get("warnings", [])),
            json.dumps(result),
        ),
    )
    conn.commit()


def save_batch(conn: sqlite3.Connection, results: list[dict]) -> None:
    """Save a batch of valuation results."""
    for result in results:
        try:
            save_valuation(conn, result)
        except Exception as e:
            logger.error("Failed to save %s: %s", result.get("ticker"), e)


def get_latest_valuation(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Get most recent valuation for a ticker."""
    cursor = conn.execute(
        "SELECT raw_json FROM valuations WHERE ticker=? ORDER BY valuation_date DESC LIMIT 1",
        (ticker,),
    )
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


def get_all_latest(conn: sqlite3.Connection) -> list[dict]:
    """Get the latest valuation for all tickers."""
    cursor = conn.execute("""
        SELECT raw_json FROM valuations v
        INNER JOIN (
            SELECT ticker, MAX(valuation_date) as max_date
            FROM valuations GROUP BY ticker
        ) latest ON v.ticker = latest.ticker AND v.valuation_date = latest.max_date
        ORDER BY v.ticker
    """)
    return [json.loads(row[0]) for row in cursor.fetchall()]


def get_history(conn: sqlite3.Connection, ticker: str, limit: int = 30) -> list[dict]:
    """Get valuation history for a ticker."""
    cursor = conn.execute(
        "SELECT raw_json FROM valuations WHERE ticker=? ORDER BY valuation_date DESC LIMIT ?",
        (ticker, limit),
    )
    return [json.loads(row[0]) for row in cursor.fetchall()]
