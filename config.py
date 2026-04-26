"""
Stockinator — Configuration & Constants

Central configuration for all financial constants, thresholds,
and environment variable loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────
load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports" / "generated"
DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "stockinator.db")

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys (Phase 2+) ────────────────────────────────────────────
FMP_API_KEY = os.getenv("FMP_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Financial Constants ─────────────────────────────────────────────
# Market assumptions
RISK_FREE_RATE_DEFAULT = 0.043        # 10Y US Treasury (~4.3% as of 2025)
MARKET_PREMIUM_DEFAULT = 0.055        # Historical equity risk premium
EXPECTED_MARKET_RETURN = 0.10         # Long-run nominal (~10%)
TERMINAL_GROWTH_RATE = 0.025          # Long-run GDP growth (~2.5%)
TERMINAL_GROWTH_MAX = 0.04            # Cap terminal growth at 4%

# ── Valuation Model Weights ────────────────────────────────────────
MODEL_WEIGHTS = {
    "ggm": 0.20,
    "dcf": 0.35,
    "comps": 0.30,
    "rim": 0.15,
}

# ── Signal Thresholds (Margin of Safety) ────────────────────────────
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.30,     # MOS > 30%  → Strong BUY
    "buy": 0.20,            # MOS 20-30% → BUY
    "hold_upper": 0.20,     # MOS -20% to 20% → HOLD
    "sell": -0.20,          # MOS -20% to -30% → SELL
    "strong_sell": -0.30,   # MOS < -30% → Strong SELL
}

# Elevated thresholds when model disagreement is high
HIGH_DISAGREEMENT_BUY_THRESHOLD = 0.40

# ── Data Quality Thresholds ─────────────────────────────────────────
MAX_FUNDAMENTALS_AGE_DAYS = 120       # Max staleness before warning
MODEL_DISAGREEMENT_THRESHOLD = 0.40   # Std/Mean ratio triggering warning
MIN_VALID_MODELS = 2                  # Minimum models for a signal
EXTREME_PE_UPPER = 200                # P/E above this → warning
EXTREME_PE_LOWER = 0                  # Negative P/E → warning

# ── DCF Specific ────────────────────────────────────────────────────
DCF_FORECAST_YEARS = 5                # Default projection horizon
DCF_TERMINAL_METHOD = "gordon"        # "gordon" or "exit_multiple"

# ── Comps Specific ──────────────────────────────────────────────────
COMPS_REVENUE_TOLERANCE = 0.50        # ±50% revenue size for peer match
COMPS_GROWTH_TOLERANCE = 0.10         # ±10pp growth rate for peer match
DEFAULT_MULTIPLES = ["ev_ebitda", "pe", "ps"]

# ── Test Universe ───────────────────────────────────────────────────
TEST_TICKERS = ["AAPL", "MSFT", "KO", "JNJ"]

# Full S&P 500 universe (Phase 2+)
# SP500_TICKERS = [...]  # loaded from CSV or API
