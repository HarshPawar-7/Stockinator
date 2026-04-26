# 📊 Stockinator

**ML-Based Stock Valuation Platform**

Determines the **intrinsic value** of publicly traded stocks using multiple valuation models combined with machine learning to reduce estimation error. Outputs fair-value estimates with confidence intervals and margin-of-safety signals.

---

## 🧠 How It Works

Stockinator runs **4 independent valuation models** and combines them via an **ensemble** to produce a final intrinsic value:

| Model | Formula | Best For |
|-------|---------|----------|
| **GGM** (Gordon Growth) | V = D₁ / (r - g) | Mature dividend payers |
| **DCF** (Discounted Cash Flow) | V = Σ FCF/(1+WACC)ᵗ + TV | Cash-flow positive companies |
| **Comps** (Comparable Analysis) | V = Median Peer Multiple × Metric | Broad market context |
| **RIM** (Residual Income) | V = BV + Σ RI/(1+r)ᵗ | Banks, insurance, accounting-heavy |

The ensemble combines valid models with **configurable weights** (default: DCF 35%, Comps 30%, GGM 20%, RIM 15%), automatically redistributing when models are skipped.

### Signals

| Signal | Margin of Safety |
|--------|-----------------|
| 🟢 **STRONG BUY** | > 30% |
| 🟢 **BUY** | 20-30% |
| 🟡 **HOLD** | -20% to 20% |
| 🔴 **SELL** | -20% to -30% |
| 🔴 **STRONG SELL** | < -30% |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Valuation

```bash
# Default: AAPL, MSFT, KO, JNJ
python main.py

# Specific tickers
python main.py AAPL GOOGL NVDA TSLA

# Fast mode (skip peer comparison)
python main.py --no-peers AAPL

# Verbose debug output
python main.py -v AAPL

# Read tickers from file
python main.py --ticker-file sp500.txt
```

### 3. View Results

- **Terminal**: Color-coded summary table
- **CSV**: `data/outputs/valuations.csv`
- **JSON**: `data/outputs/valuations.json`
- **Reports**: `reports/generated/{TICKER}_valuation.md`
- **Database**: `stockinator.db` (SQLite)

---

## 📁 Project Structure

```
Stockinator/
├── models/
│   ├── valuation/
│   │   ├── ggm.py          # Gordon Growth Model
│   │   ├── dcf.py          # Discounted Cash Flow
│   │   ├── comps.py        # Comparable Company Analysis
│   │   └── rim.py          # Residual Income Model
│   ├── ensemble.py         # Weighted ensemble combiner
│   └── ml/
│       ├── features.py     # Feature engineering pipeline
│       ├── train.py        # XGBoost/LightGBM training
│       ├── predict.py      # ML-adjusted valuations
│       ├── conformal.py    # Conformal prediction CIs
│       └── backtest.py     # Backtesting framework
├── pipeline/
│   ├── ingest.py           # yfinance data ingestion (with retry/cache)
│   ├── fred_api.py         # FRED API for live macro data
│   ├── sp500.py            # S&P 500 universe loader
│   ├── scheduler.py        # Batch scheduling
│   ├── data_quality.py     # Validation gates
│   └── batch_valuation.py  # Batch processing engine
├── agents/
│   ├── orchestrator.py     # Groq-powered AI agent
│   ├── tools.py            # Tool definitions (6 tools)
│   └── prompts.py          # System prompts
├── database/
│   └── db.py               # SQLite persistence
├── reports/
│   ├── templates/          # Markdown report templates
│   └── report_generator.py # CSV, JSON, Markdown output
├── tests/                  # 9 test modules, 60+ test cases
├── config.py               # Central configuration
├── main.py                 # CLI entry point
└── requirements.txt
```

---

## 🤖 AI Agent Mode

Chat with an AI financial analyst powered by Groq (free tier):

```bash
# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Launch interactive mode
python main.py --agent
```

The agent can autonomously fetch data, run models, and explain results.

---

## 📅 Scheduling & Universes

```bash
# Run on top 50 S&P 500 stocks
python main.py --universe top50

# Run on full S&P 500
python main.py --universe sp500 --no-peers

# Schedule every 24 hours on test tickers
python main.py --schedule 24
```

---

## 🧪 Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## 📊 Data Sources

| Source | Phase | Cost | Data |
|--------|-------|------|------|
| yfinance | 1+ | Free | Fundamentals, prices, peers |
| FRED API | 2+ | Free | Risk-free rate, yield spread, GDP |
| Groq | 3+ | Free tier | AI analysis agent |

---

## ⚠️ Disclaimer

This is an automated valuation engine for **educational and research purposes**.

- Not investment advice
- Not a trading system
- Not suitable as sole basis for investment decisions
- Not adjusted for qualitative factors (management, moat, ESG)
- Not calibrated for illiquid or micro-cap stocks

**Always perform independent due diligence.**

---

## 🗺️ Roadmap

- [x] **Phase 1**: Core valuation engine (GGM, DCF, Comps, RIM) + yfinance
- [x] **Phase 2**: Enhanced pipeline (FRED API, retry/cache, S&P 500 universe)
- [x] **Phase 3**: AI Agent layer (Groq-powered autonomous valuation)
- [x] **Phase 4**: ML enhancement (XGBoost, feature engineering, backtesting)
- [ ] **Phase 5**: Vite web dashboard + production pipeline
