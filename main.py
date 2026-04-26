"""
Stockinator — Main CLI Entry Point

ML-based stock valuation platform.
Determines intrinsic value using GGM, DCF, Comps, and RIM models
combined with ensemble averaging.

Usage:
    python main.py                        # Run on default test tickers
    python main.py AAPL MSFT GOOGL        # Run on specific tickers
    python main.py --no-peers AAPL        # Skip peer fetching (faster)
    python main.py --ticker-file sp500.txt # Read tickers from file
    python main.py --agent                # Interactive AI agent mode
    python main.py --universe sp500       # Run on S&P 500
    python main.py --schedule 24          # Run every 24 hours
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from config import TEST_TICKERS
from pipeline.batch_valuation import run_batch_valuation, valuate_single_stock
from reports.report_generator import save_reports, generate_markdown_report
from database.db import init_db, save_batch


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with colored output."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def print_summary(results: list[dict]) -> None:
    """Print a beautiful summary table to stdout."""
    print("\n" + "=" * 90)
    print("  📊 STOCKINATOR — Valuation Results")
    print("=" * 90)

    header = f"{'Ticker':<8} {'Price':>10} {'Ensemble':>10} {'MOS':>8} {'Signal':>12} {'Models':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        ticker = r.get("ticker", "?")
        price = r.get("market_price")
        e = r.get("ensemble", {})
        ev = e.get("ensemble_value")
        mos = e.get("margin_of_safety")
        signal = e.get("signal", "N/A")
        n_models = e.get("valid_model_count", 0)

        # Signal coloring (ANSI)
        signal_display = signal
        if "BUY" in signal:
            signal_display = f"\033[92m{signal}\033[0m"  # Green
        elif "SELL" in signal:
            signal_display = f"\033[91m{signal}\033[0m"  # Red
        elif signal == "HOLD":
            signal_display = f"\033[93m{signal}\033[0m"  # Yellow
        elif signal == "ERROR":
            signal_display = f"\033[91m{signal}\033[0m"

        price_str = f"${price:,.2f}" if price else "N/A"
        ev_str = f"${ev:,.2f}" if ev else "N/A"
        mos_str = f"{mos:+.1%}" if mos is not None else "N/A"

        print(f"{ticker:<8} {price_str:>10} {ev_str:>10} {mos_str:>8} {signal_display:>20} {n_models:>5}/4")

    print("=" * 90)

    # Warnings summary
    total_warnings = sum(
        len(r.get("data_quality", {}).get("warnings", []))
        for r in results
    )
    if total_warnings:
        print(f"\n⚠️  Total warnings: {total_warnings}")
        print("   Run with --verbose for details, or check reports/generated/ for full reports.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stockinator — ML-Based Stock Valuation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Default: AAPL, MSFT, KO, JNJ
  python main.py AAPL GOOGL NVDA     # Specific tickers
  python main.py --no-peers AAPL     # Skip peer fetching (faster)
  python main.py -v AAPL             # Verbose output
  python main.py --agent             # AI agent chat mode (needs GROQ_API_KEY)
  python main.py --universe top50    # Run on top 50 S&P 500 stocks
  python main.py --schedule 24       # Repeat every 24 hours
        """,
    )
    parser.add_argument(
        "tickers", nargs="*", default=None,
        help="Stock ticker symbols to valuate (default: AAPL MSFT KO JNJ)",
    )
    parser.add_argument(
        "--no-peers", action="store_true",
        help="Skip peer data fetching for comps (much faster)",
    )
    parser.add_argument(
        "--ticker-file", type=str,
        help="Read tickers from a text file (one per line)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save to database or generate report files",
    )
    parser.add_argument(
        "--agent", action="store_true",
        help="Launch interactive AI agent mode (requires GROQ_API_KEY in .env)",
    )
    parser.add_argument(
        "--universe", type=str, choices=["test", "top50", "sp500"],
        help="Use a predefined ticker universe instead of manual tickers",
    )
    parser.add_argument(
        "--schedule", type=float, metavar="HOURS",
        help="Run valuation on repeat every N hours",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("stockinator")

    # ── Agent Mode ─────────────────────────────────────────────────
    if args.agent:
        try:
            from agents.orchestrator import chat
            chat(verbose=args.verbose)
        except (ImportError, ValueError) as e:
            print(f"\n❌ {e}")
            print("   Install groq: pip install groq")
            print("   Set key: export GROQ_API_KEY=your_key_here\n")
            sys.exit(1)
        return

    # ── Scheduler Mode ─────────────────────────────────────────────
    if args.schedule:
        from pipeline.scheduler import start_scheduler
        universe = args.universe or "test"
        start_scheduler(
            interval_hours=args.schedule,
            universe=universe,
            fetch_peers=not args.no_peers,
        )
        return

    # ── Resolve Tickers ────────────────────────────────────────────
    tickers = args.tickers
    if args.ticker_file:
        with open(args.ticker_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.universe:
        from pipeline.sp500 import load_sp500_tickers
        if args.universe == "sp500":
            tickers = load_sp500_tickers(source="wiki")
        elif args.universe == "top50":
            tickers = load_sp500_tickers(source="fallback")
        else:
            tickers = TEST_TICKERS
    elif not tickers:
        tickers = TEST_TICKERS

    if not tickers:
        print("❌ No tickers specified. Use: python main.py AAPL MSFT GOOGL")
        sys.exit(1)

    print(f"\n🚀 Stockinator — Valuating {len(tickers)} stock(s): {', '.join(tickers[:10])}")
    if len(tickers) > 10:
        print(f"   ... and {len(tickers) - 10} more")
    print(f"   Peer comps: {'disabled' if args.no_peers else 'enabled'}\n")

    start_time = time.time()

    # ── Run Valuation ──────────────────────────────────────────────
    results = run_batch_valuation(
        tickers=tickers,
        fetch_peers=not args.no_peers,
    )

    elapsed = time.time() - start_time

    # ── Print Summary ──────────────────────────────────────────────
    print_summary(results)
    print(f"⏱️  Completed in {elapsed:.1f}s\n")

    # ── Save Results ───────────────────────────────────────────────
    if not args.no_save:
        # Database
        try:
            conn = init_db()
            save_batch(conn, results)
            conn.close()
            logger.info("Results saved to database")
        except Exception as e:
            logger.error("Database save failed: %s", e)

        # Reports
        try:
            saved_files = save_reports(results)
            print("📄 Reports saved:")
            for name, path in saved_files.items():
                print(f"   {name}: {path}")
        except Exception as e:
            logger.error("Report generation failed: %s", e)

    print("\n✅ Done! Not investment advice.\n")


if __name__ == "__main__":
    main()
