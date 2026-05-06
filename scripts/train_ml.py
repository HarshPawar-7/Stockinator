#!/usr/bin/env python3
"""
train_ml.py — Stockinator ML Training Script

Trains the valuation accuracy ML model on historical data.

Usage:
    python scripts/train_ml.py                      # Train on all DB history
    python scripts/train_ml.py --model xgboost      # Specify model type
    python scripts/train_ml.py --model ridge        # Lighter Ridge regression
    python scripts/train_ml.py --dry-run            # Validate setup only
    python scripts/train_ml.py --from-reports       # Use JSON report files

The model predicts: future_price_12m / current_ensemble_value
This ratio is used to ML-adjust live valuations in the pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def fetch_forward_price(ticker: str, valuation_date: str, months: int = 12) -> float | None:
    """
    Fetch the price approximately `months` after the valuation date from FMP.

    Uses FMP stable/historical-price-full with a narrow date window.
    """
    import httpx
    fmp_key = os.environ.get("FMP_API_KEY")
    if not fmp_key:
        return None

    try:
        val_dt = date.fromisoformat(valuation_date)
        target_dt = val_dt + timedelta(days=months * 30)
        from_dt = target_dt - timedelta(days=7)
        to_dt = target_dt + timedelta(days=7)

        url = f"https://financialmodelingprep.com/stable/historical-price-full/{ticker}"
        params = {
            "from": from_dt.isoformat(),
            "to": to_dt.isoformat(),
            "apikey": fmp_key,
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        history = data.get("historical", [])
        if history:
            return float(history[0]["close"])
        return None

    except Exception as e:
        logger.debug("Forward price fetch failed for %s: %s", ticker, e)
        return None


def load_from_reports() -> list[dict]:
    """Load historical valuation results from JSON report files."""
    reports_dir = Path(__file__).parent.parent / "reports" / "generated"
    if not reports_dir.exists():
        logger.warning("No reports directory found at %s", reports_dir)
        return []

    results = []
    for json_file in sorted(reports_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Handle both list and single result formats
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", json_file, e)

    logger.info("Loaded %d historical results from %d report files", len(results), len(list(reports_dir.glob("*.json"))))
    return results


async def load_from_db() -> list[dict]:
    """Load historical valuation results from PostgreSQL."""
    from database.db import get_db_pool, get_all_latest
    try:
        pool = await get_db_pool()
        results = await get_all_latest(pool)
        await pool.close()
        logger.info("Loaded %d historical results from database", len(results))
        return results
    except Exception as e:
        logger.warning("DB load failed: %s — falling back to report files", e)
        return []


async def main():
    parser = argparse.ArgumentParser(
        description="Train Stockinator ML valuation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", choices=["xgboost", "lightgbm", "ridge"], default="xgboost",
        help="Model type to train (default: xgboost)"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of time-series CV folds (default: 5)"
    )
    parser.add_argument(
        "--horizon", type=int, default=12,
        help="Forward price horizon in months (default: 12)"
    )
    parser.add_argument(
        "--from-reports", action="store_true",
        help="Use JSON report files instead of database"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate setup and data loading without training"
    )
    parser.add_argument(
        "--min-samples", type=int, default=20,
        help="Minimum samples required to proceed (default: 20)"
    )
    args = parser.parse_args()

    print("\n🤖 Stockinator ML Training Pipeline")
    print("=" * 50)

    # ── Load Historical Results ─────────────────────────────────
    if args.from_reports:
        historical = load_from_reports()
    else:
        historical = await load_from_db()
        if not historical:
            historical = load_from_reports()

    if not historical:
        print("❌ No historical valuation data found.")
        print("   Run some valuations first: python main.py AAPL MSFT KO JNJ --no-save")
        print("   Or use report files: python scripts/train_ml.py --from-reports")
        sys.exit(1)

    print(f"📊 Loaded {len(historical)} historical valuations")

    if args.dry_run:
        print("\n✅ Dry run — setup validated. Data loaded successfully.")
        print(f"   Would train: {args.model} with {args.folds}-fold TimeSeriesSplit")
        print(f"   Forward price horizon: {args.horizon} months")
        return

    # ── Fetch Forward Prices ────────────────────────────────────
    print(f"\n⏳ Fetching {args.horizon}M forward prices from FMP...")
    future_prices = {}
    tasks = []

    # Filter results that have ticker and date
    valid = [r for r in historical if r.get("ticker") and r.get("valuation_date")]

    # Only fetch for valuations old enough to have 12M of history
    cutoff = date.today() - timedelta(days=args.horizon * 30)
    eligible = [
        r for r in valid
        if date.fromisoformat(r["valuation_date"]) <= cutoff
    ]

    if len(eligible) < args.min_samples:
        print(f"\n⚠️  Only {len(eligible)} valuations are old enough for {args.horizon}M forward labels.")
        print(f"   Need at least {args.min_samples} samples to train meaningfully.")
        print("   Tip: Run valuations regularly over several months to build training data.")
        sys.exit(1)

    for r in eligible:
        tasks.append(fetch_forward_price(r["ticker"], r["valuation_date"], args.horizon))

    forward_prices_list = await asyncio.gather(*tasks)
    for r, fp in zip(eligible, forward_prices_list):
        if fp is not None:
            future_prices[r["ticker"]] = fp

    print(f"   Forward prices fetched: {len(future_prices)}/{len(eligible)}")

    if len(future_prices) < args.min_samples:
        print(f"\n❌ Insufficient labeled samples ({len(future_prices)} < {args.min_samples}).")
        print("   Check FMP_API_KEY and ensure historical price data is available.")
        sys.exit(1)

    # ── Build Training Dataset ──────────────────────────────────
    from models.ml.features import build_training_dataset
    X, y = build_training_dataset(eligible, future_prices)

    if len(X) < args.min_samples:
        print(f"\n❌ Feature matrix too small after filtering ({len(X)} rows).")
        sys.exit(1)

    print(f"\n🏋️  Training {args.model.upper()} on {len(X)} samples × {len(X.columns)} features")
    print(f"   Cross-validation: {args.folds}-fold temporal split")

    # ── Train ───────────────────────────────────────────────────
    from models.ml.train import train_model
    result = train_model(X, y, model_type=args.model, n_splits=args.folds)

    # ── Print Report ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📈 Training Results")
    print("=" * 50)
    m = result["avg_metrics"]
    print(f"  Avg MAE:  {m['avg_mae']:.4f}")
    print(f"  Avg RMSE: {m['avg_rmse']:.4f}")
    print(f"  Avg R²:   {m['avg_r2']:.4f}")

    print("\n🔑 Top Feature Importances:")
    for fi in result["feature_importance"][:8]:
        bar = "█" * int(fi["importance"] * 40)
        print(f"  {fi['feature']:<28} {bar}")

    print(f"\n✅ Model saved: {result['model_path']}")
    print(f"   Imputer saved: {result['imputer_path']}")
    print("\n💡 The model will now automatically enrich live valuations.")
    print("   View MLflow experiments with: mlflow ui\n")


if __name__ == "__main__":
    asyncio.run(main())
