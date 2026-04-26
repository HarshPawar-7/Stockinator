"""
Scheduler — Simple batch job scheduling

Runs valuation pipeline on a schedule (daily, weekly, or on-demand).
Uses Python's sched module for simplicity — replace with Airflow/cron
for production.
"""

from __future__ import annotations

import logging
import sched
import time
from datetime import datetime

from config import TEST_TICKERS
from pipeline.batch_valuation import run_batch_valuation
from pipeline.sp500 import load_sp500_tickers
from reports.report_generator import save_reports
from database.db import init_db, save_batch

logger = logging.getLogger(__name__)


def run_scheduled_valuation(
    tickers: list[str] | None = None,
    universe: str = "test",
    fetch_peers: bool = True,
) -> None:
    """
    Run a single scheduled valuation batch.

    Args:
        tickers: Explicit ticker list (overrides universe)
        universe: "test" (4 stocks), "top50", or "sp500"
        fetch_peers: Whether to fetch peer data for comps
    """
    start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("SCHEDULED RUN: %s | Universe: %s", timestamp, universe)
    logger.info("=" * 60)

    # Resolve tickers
    if tickers is None:
        if universe == "sp500":
            tickers = load_sp500_tickers(source="wiki")
        elif universe == "top50":
            tickers = load_sp500_tickers(source="fallback")
        else:
            tickers = TEST_TICKERS

    logger.info("Processing %d tickers", len(tickers))

    # Run valuation
    results = run_batch_valuation(tickers=tickers, fetch_peers=fetch_peers)

    # Save to database
    try:
        conn = init_db()
        save_batch(conn, results)
        conn.close()
    except Exception as e:
        logger.error("Database save failed: %s", e)

    # Save reports
    try:
        save_reports(results)
    except Exception as e:
        logger.error("Report save failed: %s", e)

    elapsed = time.time() - start
    success = sum(1 for r in results if r.get("ensemble", {}).get("signal") != "ERROR")
    logger.info(
        "COMPLETED: %d/%d successful in %.1fs",
        success, len(results), elapsed,
    )


def start_scheduler(
    interval_hours: float = 24.0,
    universe: str = "test",
    fetch_peers: bool = True,
    max_runs: int | None = None,
) -> None:
    """
    Start a repeating scheduler.

    Args:
        interval_hours: Hours between runs
        universe: Ticker universe to process
        fetch_peers: Whether to fetch peers
        max_runs: Max number of runs (None = infinite)
    """
    scheduler = sched.scheduler(time.time, time.sleep)
    run_count = 0

    def _job():
        nonlocal run_count
        run_count += 1
        logger.info("Scheduler: Run %d", run_count)

        run_scheduled_valuation(universe=universe, fetch_peers=fetch_peers)

        if max_runs is None or run_count < max_runs:
            scheduler.enter(interval_hours * 3600, 1, _job)
            logger.info("Next run in %.1f hours", interval_hours)
        else:
            logger.info("Max runs (%d) reached. Stopping.", max_runs)

    # Run immediately, then schedule repeats
    scheduler.enter(0, 1, _job)
    logger.info(
        "Scheduler started: every %.1fh, universe=%s, max_runs=%s",
        interval_hours, universe, max_runs or "∞",
    )
    scheduler.run()
