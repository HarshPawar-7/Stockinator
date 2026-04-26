# Stock Valuation with Machine Learning — Project Reference Document

> **Purpose:** Master reference for humans and AI agents building a stock valuation system.  
> **Scope:** Problem definition, valuation models, data acquisition approaches (4 methods), ML pipeline, output formats, risk controls, and implementation roadmap.  
> **Money is involved. Every section prioritizes correctness over convenience.**

---

## 1. Problem Statement

Determine the **intrinsic value** of a publicly traded stock using multiple valuation models combined with machine learning to reduce estimation error. Output a fair-value estimate with a confidence interval and margin-of-safety signal.

### Why Not Just One Model?

Single-model valuation has known failure modes:

| Model | Breaks When |
|---|---|
| Gordon Growth Model (GGM) | Company doesn't pay dividends or growth > discount rate |
| DCF | Terminal value assumptions dominate; sensitive to WACC |
| P/E Multiples | Earnings are negative or cyclically distorted |
| EV/EBITDA | Capital structure differences across peers |

**ML ensemble approach:** train on historical data where intrinsic value can be back-tested against future price convergence, then use model disagreement as uncertainty signal.

---

## 2. Valuation Models Implemented

### 2.1 Gordon Growth Model (GGM)

```
Intrinsic Value = D1 / (r - g)

D1  = next year dividend = D0 × (1 + g)
r   = required rate of return (CAPM or WACC)
g   = sustainable dividend growth rate
```

**CAPM for r:**
```
r = Rf + β × (Rm - Rf)

Rf  = risk-free rate (10Y US Treasury yield)
β   = stock beta vs S&P 500
Rm  = expected market return (historically ~10% nominal)
```

**Sustainable g (from fundamentals):**
```
g = ROE × Retention Ratio
  = ROE × (1 - Dividend Payout Ratio)
```

**Validity checks before running GGM:**
- Dividend exists and is positive
- g < r (model undefined otherwise)
- Company is mature (not high-growth startup)

---

### 2.2 Discounted Cash Flow (DCF)

```
Intrinsic Value = Σ [FCFt / (1+WACC)^t] + Terminal Value / (1+WACC)^n

FCFt = Free Cash Flow in year t = EBIT(1-T) + D&A - CapEx - ΔNWC
WACC = Weighted Average Cost of Capital
n    = forecast horizon (typically 5–10 years)

Terminal Value (Gordon approach):
TV = FCFn × (1 + g_terminal) / (WACC - g_terminal)
g_terminal ≤ long-run GDP growth (~2.5%)
```

**WACC formula:**
```
WACC = (E/V) × Re + (D/V) × Rd × (1 - T)

E   = Market cap
D   = Total debt
V   = E + D
Re  = Cost of equity (CAPM)
Rd  = Cost of debt (interest expense / total debt)
T   = Effective tax rate
```

---

### 2.3 Comparable Company Analysis (Comps / Multiples)

```
Value = Median Peer Multiple × Company Metric

Common multiples:
  EV/EBITDA  → Enterprise Value
  P/E        → Equity Value
  P/S        → Equity Value (for negative-earnings firms)
  EV/Revenue → Enterprise Value
  P/B        → Equity Value (financials/banks)
```

Peer selection criteria: same GICS sub-industry, similar revenue size (±50%), similar growth profile (±10pp).

---

### 2.4 Residual Income Model (RIM / Edwards-Bell-Ohlson)

```
Intrinsic Value = Book Value + Σ [RI_t / (1+r)^t]

RI_t = Net Income_t - (r × Book Value_{t-1})
     = Earnings that exceed the cost of equity
```

Works well when FCF is hard to estimate (banks, insurance). Uses accounting data directly.

---

### 2.5 ML Enhancement Layer

Models above produce **point estimates**. ML layer:

1. **Feature engineering** from fundamentals + macro
2. **Ensemble** GGM, DCF, Comps, RIM outputs as features
3. **Predict** price-to-intrinsic-value ratio 12 months forward
4. **Calibrate** uncertainty: output is value ± confidence interval
5. **Signal:** BUY (price < 0.8× IV), HOLD (0.8–1.2×), SELL (> 1.2×)

**Recommended ML models:**
- Gradient Boosting (XGBoost / LightGBM) — tabular financial data
- Random Forest — robust to missing features
- Ridge Regression — interpretable baseline
- Optional: LSTM for time-series macro features

**Training target:**
```
Label = Future_Price_12M / Current_IV_Estimate
# Train to predict this ratio; ratio ≈ 1 means model was accurate
```

**Avoid lookahead bias:** use only data available at prediction date. Enforce strict temporal train/test split (no random shuffle).

---

## 3. Data Acquisition — 4 Approaches

---

### Approach 1: CSV / Static Files

**Best for:** offline analysis, reproducible research, academic projects, no API key required.

**Sources:**
| Dataset | Content | URL |
|---|---|---|
| Simfin Bulk Download | Income stmt, balance sheet, cash flow, prices | simfin.com/en/bulk |
| Compustat via WRDS | Full fundamentals (institutional access) | wrds-web.wharton.upenn.edu |
| CRSP | Historical prices, returns, splits | crsp.org |
| Damodaran Datasets | Beta, WACC, risk premiums by sector | pages.stern.nyu.edu/~adamodar |
| Yahoo Finance Export | Manual OHLCV CSV download | finance.yahoo.com |
| FRED (St. Louis Fed) | Macro: Rf rate, GDP, inflation | fred.stlouisfed.org/graph/fredgraph.csv |

**CSV Pipeline:**
```python
import pandas as pd

# Load fundamentals
income   = pd.read_csv("simfin_income.csv", parse_dates=["Report Date"])
balance  = pd.read_csv("simfin_balance.csv", parse_dates=["Report Date"])
cashflow = pd.read_csv("simfin_cashflow.csv", parse_dates=["Report Date"])
prices   = pd.read_csv("simfin_prices.csv",  parse_dates=["Date"])

# Merge on ticker + nearest report date
df = income.merge(balance, on=["Ticker","Report Date"])
         .merge(cashflow, on=["Ticker","Report Date"])

# Compute core metrics
df["FCF"]        = df["Net Cash from Operating Activities"] - df["Capital Expenditures"]
df["ROE"]        = df["Net Income"] / df["Total Equity"]
df["Payout"]     = df["Dividends Paid"].abs() / df["Net Income"].clip(lower=1)
df["RetentionR"] = 1 - df["Payout"]
df["g_ggm"]      = df["ROE"] * df["RetentionR"]

# Risk-free rate from FRED CSV
fred = pd.read_csv("DGS10.csv", names=["Date","Rf"], skiprows=1, parse_dates=["Date"])
fred["Rf"] = pd.to_numeric(fred["Rf"], errors="coerce") / 100
```

**Pros:** Free, reproducible, no rate limits  
**Cons:** Stale (quarterly updates), manual download, no real-time

---

### Approach 2: REST API (Live Data)

**Best for:** production system, real-time valuation, automated pipeline.

**Recommended APIs:**
| Provider | Data | Free Tier | Paid |
|---|---|---|---|
| Financial Modeling Prep (FMP) | Fundamentals, DCF, ratios | 250 req/day | $15/mo |
| Alpha Vantage | Prices, earnings, macro | 25 req/day | $50/mo |
| Polygon.io | Prices, options, news | 5 req/min | $29/mo |
| FRED API | Risk-free rates, macro | Unlimited (key required) | Free |
| SEC EDGAR API | 10-K/10-Q filings (official) | Unlimited | Free |
| Yahoo Finance (yfinance) | Prices + basic fundamentals | Unofficial, may break | Free |

**FMP Fundamentals Pull:**
```python
import requests, os

FMP_KEY = os.getenv("FMP_API_KEY")  # Never hardcode keys

def get_income_statement(ticker: str, years: int = 5) -> list[dict]:
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
    r = requests.get(url, params={"limit": years * 4, "apikey": FMP_KEY})
    r.raise_for_status()
    return r.json()

def get_balance_sheet(ticker: str) -> list[dict]:
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}"
    r = requests.get(url, params={"limit": 20, "apikey": FMP_KEY})
    r.raise_for_status()
    return r.json()

def get_beta(ticker: str) -> float:
    url = f"https://financialmodelingprep.com/api/v3/company/profile/{ticker}"
    r = requests.get(url, params={"apikey": FMP_KEY})
    return r.json()["profile"]["beta"]

def get_risk_free_rate() -> float:
    """10Y US Treasury from FRED"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params={
        "series_id": "DGS10", "api_key": os.getenv("FRED_KEY"),
        "sort_order": "desc", "limit": 1, "file_type": "json"
    })
    return float(r.json()["observations"][0]["value"]) / 100
```

**Rate limit handling:**
```python
import time
from functools import wraps

def rate_limited(max_per_minute: int):
    min_interval = 60.0 / max_per_minute
    def decorator(fn):
        last_called = [0.0]
        @wraps(fn)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait = min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            result = fn(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

**Pros:** Fresh data, automatable, structured  
**Cons:** Cost at scale, API key management, rate limits, provider risk

---

### Approach 3: AI Agent (Autonomous Data + Valuation)

**Best for:** multi-stock screening, unstructured data (10-K text), earnings call analysis, research synthesis.

**Agent architecture:**
```
User Query: "Value AAPL"
    │
    ▼
Orchestrator Agent
    ├── Tool: fetch_fundamentals(ticker) → FMP/EDGAR API
    ├── Tool: fetch_prices(ticker)       → Polygon/Yahoo
    ├── Tool: fetch_macro()              → FRED API
    ├── Tool: fetch_filings(ticker)      → SEC EDGAR full-text
    ├── Tool: web_search(query)          → News, analyst targets
    └── Tool: run_valuation_models(data) → Python calculation engine
    │
    ▼
Valuation Agent
    ├── Runs GGM, DCF, Comps, RIM
    ├── Flags model invalidity (g > r, negative FCF, etc.)
    └── Returns: {ggm: x, dcf: y, comps: z, rim: w, mean: v, ci_95: [lo, hi]}
    │
    ▼
Risk & Sanity Check Agent
    ├── Checks: margin of safety, model disagreement, data quality
    ├── Flags: high debt, accounting anomalies, recent restatements
    └── Returns: signal + confidence + warnings[]
    │
    ▼
Output: Structured JSON + Markdown report
```

**Tool definitions (Claude / OpenAI function calling format):**
```json
{
  "name": "run_ggm_valuation",
  "description": "Compute Gordon Growth Model intrinsic value. Returns null if model conditions not met.",
  "parameters": {
    "type": "object",
    "properties": {
      "d0":    {"type": "number", "description": "Last annual dividend per share"},
      "g":     {"type": "number", "description": "Dividend growth rate (decimal, e.g. 0.05)"},
      "r":     {"type": "number", "description": "Required rate of return from CAPM (decimal)"},
      "price": {"type": "number", "description": "Current market price for margin of safety calc"}
    },
    "required": ["d0", "g", "r", "price"]
  }
}
```

**Agent implementation (Python + Anthropic SDK):**
```python
import anthropic, json

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "fetch_fundamentals",
        "description": "Fetch income statement, balance sheet, cash flow for ticker",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "years": {"type": "integer", "default": 5}},
            "required": ["ticker"]
        }
    },
    {
        "name": "run_dcf",
        "description": "Run DCF valuation given list of FCF forecasts and WACC",
        "input_schema": {
            "type": "object",
            "properties": {
                "fcf_forecasts": {"type": "array", "items": {"type": "number"}},
                "wacc": {"type": "number"},
                "terminal_growth": {"type": "number"}
            },
            "required": ["fcf_forecasts", "wacc", "terminal_growth"]
        }
    }
]

def run_valuation_agent(ticker: str) -> dict:
    messages = [{"role": "user", "content": f"Perform complete fundamental valuation of {ticker}. Use all available models. Flag any model that cannot be applied and explain why. Return structured JSON with intrinsic value estimates, confidence interval, and investment signal."}]
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return extract_valuation_json(response.content)
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })
        
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

**Pros:** Handles ambiguity, processes filings text, self-corrects, multi-stock  
**Cons:** Latency (seconds per stock), cost per run, hallucination risk on numbers — **always validate tool outputs**

---

### Approach 4: Automated Pipeline (Scheduled / Production)

**Best for:** running valuation on universe of stocks (e.g., S&P 500) on schedule, feeding dashboard or alerts.

**Stack:**
```
Data Sources → Ingestion → Feature Store → Valuation Engine → Output Store → Dashboard
```

**Component design:**

```
┌─────────────────────────────────────────────────────┐
│                  SCHEDULER (Airflow / cron)          │
│  - Daily: price update, macro rates                 │
│  - Quarterly: fundamentals update (earnings season) │
│  - Weekly: peer comps recalculation                 │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  DATA INGESTION  │
              │  FMP + FRED APIs │
              │  SEC EDGAR RSS   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  FEATURE STORE  │
              │  SQLite / DuckDB│
              │  (local)        │
              │  PostgreSQL     │
              │  (production)   │
              └────────┬────────┘
                       │
            ┌──────────▼──────────┐
            │  VALUATION ENGINE   │
            │  GGM / DCF / Comps  │
            │  RIM / ML Ensemble  │
            └──────────┬──────────┘
                       │
              ┌────────▼────────┐
              │  OUTPUT STORE   │
              │  valuation.csv  │
              │  valuation.json │
              │  SQLite table   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   DASHBOARD     │
              │  Streamlit /    │
              │  Grafana /      │
              │  Excel export   │
              └─────────────────┘
```

**Database schema (SQLite / PostgreSQL):**
```sql
CREATE TABLE fundamentals (
    ticker          TEXT NOT NULL,
    report_date     DATE NOT NULL,
    fiscal_year     INT,
    revenue         REAL,
    net_income      REAL,
    ebitda          REAL,
    fcf             REAL,
    total_debt      REAL,
    total_equity    REAL,
    dividends_paid  REAL,
    capex           REAL,
    roe             REAL,
    PRIMARY KEY (ticker, report_date)
);

CREATE TABLE valuations (
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
    margin_safety   REAL,  -- (ensemble_value - price) / ensemble_value
    signal          TEXT,  -- BUY / HOLD / SELL / INSUFFICIENT_DATA
    model_flags     TEXT,  -- JSON array of model warnings
    PRIMARY KEY (ticker, valuation_date)
);

CREATE TABLE macro (
    date            DATE PRIMARY KEY,
    risk_free_rate  REAL,  -- DGS10
    market_premium  REAL,  -- Rm - Rf (rolling 20Y)
    cpi_yoy         REAL,
    gdp_growth      REAL
);
```

**Batch valuation script:**
```python
import pandas as pd
import sqlite3
from datetime import date

def run_batch_valuation(tickers: list[str], db_path: str = "valuations.db"):
    conn = sqlite3.connect(db_path)
    results = []
    
    for ticker in tickers:
        try:
            data = load_fundamentals(conn, ticker)
            macro = load_latest_macro(conn)
            
            ggm   = compute_ggm(data, macro)   # returns float or None
            dcf   = compute_dcf(data, macro)   # returns float or None
            comps = compute_comps(data, ticker) # returns float or None
            rim   = compute_rim(data, macro)   # returns float or None

            valid = [v for v in [ggm, dcf, comps, rim] if v is not None]
            
            if len(valid) < 2:
                signal = "INSUFFICIENT_DATA"
                ensemble = None
            else:
                ensemble = weighted_ensemble(ggm, dcf, comps, rim)
                price    = get_latest_price(conn, ticker)
                mos      = (ensemble - price) / ensemble
                signal   = "BUY" if mos > 0.20 else "SELL" if mos < -0.20 else "HOLD"
            
            results.append({
                "ticker": ticker, "valuation_date": date.today(),
                "ggm_value": ggm, "dcf_value": dcf,
                "comps_value": comps, "rim_value": rim,
                "ensemble_value": ensemble, "signal": signal
            })
        except Exception as e:
            print(f"ERROR {ticker}: {e}")
            results.append({"ticker": ticker, "signal": "ERROR", "error": str(e)})
    
    pd.DataFrame(results).to_sql("valuations", conn, if_exists="append", index=False)
    conn.close()
    return results
```

**Pros:** Scalable, auditable, scheduled, dashboardable  
**Cons:** Infrastructure overhead, data pipeline failures, requires monitoring

---

## 4. Output Formats

### 4.1 CSV Output

```
ticker, date, price, ggm_value, dcf_value, comps_value, rim_value, ensemble_value, ci_lower_95, ci_upper_95, margin_of_safety, signal, flags
AAPL, 2025-01-15, 230.50, 198.20, 245.80, 221.40, 210.30, 218.93, 192.14, 245.72, -0.053, HOLD, "[]"
MSFT, 2025-01-15, 415.00, 380.10, 452.30, 421.80, 390.20, 411.10, 361.77, 460.43, 0.009, HOLD, "[]"
KO, 2025-01-15, 62.30, 71.80, 65.40, 63.20, 70.10, 67.63, 59.51, 75.74, 0.084, HOLD, "[]"
```

### 4.2 JSON Output (Agent-Readable)

```json
{
  "ticker": "AAPL",
  "valuation_date": "2025-01-15",
  "market_price": 230.50,
  "models": {
    "ggm": {
      "value": 198.20,
      "inputs": {"d0": 0.96, "g": 0.042, "r": 0.089},
      "valid": true,
      "warning": null
    },
    "dcf": {
      "value": 245.80,
      "inputs": {"wacc": 0.091, "terminal_growth": 0.025, "fcf_base": 102400000000},
      "valid": true,
      "warning": null
    },
    "comps": {
      "value": 221.40,
      "inputs": {"ev_ebitda_peer_median": 22.4, "ebitda": 125000000000},
      "valid": true,
      "warning": null
    },
    "rim": {
      "value": 210.30,
      "inputs": {"book_value": 67.00, "roe": 0.145, "r": 0.089},
      "valid": true,
      "warning": null
    }
  },
  "ensemble": {
    "value": 218.93,
    "weights": {"ggm": 0.20, "dcf": 0.35, "comps": 0.30, "rim": 0.15},
    "ci_95": [192.14, 245.72],
    "margin_of_safety": -0.053
  },
  "signal": "HOLD",
  "flags": [],
  "data_quality": {
    "fundamentals_age_days": 45,
    "price_age_days": 0,
    "missing_fields": []
  }
}
```

### 4.3 Markdown Report (Human-Readable)

Generated per stock. Template:

```markdown
# Valuation Report: {TICKER} — {DATE}

**Market Price:** ${price}  
**Intrinsic Value (Ensemble):** ${ensemble} [95% CI: ${ci_lo} – ${ci_hi}]  
**Margin of Safety:** {mos}%  
**Signal:** {signal}

## Model Results

| Model | Value | Applied | Notes |
|---|---|---|---|
| Gordon Growth | ${ggm} | ✅ | g=4.2%, r=8.9% |
| DCF | ${dcf} | ✅ | WACC=9.1%, TV=2.5% |
| Comps | ${comps} | ✅ | EV/EBITDA=22.4x |
| Residual Income | ${rim} | ✅ | ROE=14.5% |

## Risk Flags
{flags or "None identified"}

## Key Assumptions
- Risk-free rate: {rf}% (10Y Treasury, {date})
- Market premium: {mrp}%
- Beta: {beta}

---
*This report is generated by an automated model. Not investment advice.*
```

---

## 5. ML Model Pipeline

### 5.1 Feature Set

```python
FEATURES = {
    # Valuation model outputs (ensemble inputs)
    "ggm_value", "dcf_value", "comps_value", "rim_value",
    "model_disagreement",  # std(ggm, dcf, comps, rim) / mean(...)
    
    # Fundamentals
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe", "roa", "roic",
    "debt_to_equity", "interest_coverage",
    "gross_margin", "operating_margin", "net_margin",
    "revenue_growth_yoy", "earnings_growth_yoy", "fcf_growth_yoy",
    "payout_ratio", "dividend_yield",
    
    # Price-based
    "price_to_ggm",    # current price / ggm_value
    "52w_momentum",    # price / 52-week-low
    "price_vs_200ma",  # price / 200-day moving average
    
    # Macro
    "risk_free_rate", "yield_spread_10_2",  # 10Y - 2Y Treasury
    "vix",                                   # market volatility
    "sector_pe_premium",                     # stock P/E vs sector median P/E
}

TARGET = "price_12m_vs_iv"  
# = price 12 months later / ensemble_value at prediction date
# Value < 1: stock converged downward (overvalued signal confirmed)
# Value > 1: stock converged upward (undervalued signal confirmed)
```

### 5.2 Training Protocol

```python
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# CRITICAL: no random shuffle — respect time order
tscv = TimeSeriesSplit(n_splits=5, gap=252)  # 252 = 1 trading year gap

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
```

### 5.3 Confidence Interval

Use **conformal prediction** (distribution-free, valid coverage guarantee):

```python
from mapie.regression import MapieRegressor

mapie = MapieRegressor(estimator=model, method="plus", cv=5)
mapie.fit(X_train, y_train)
y_pred, y_ci = mapie.predict(X_test, alpha=0.05)  # 95% CI
```

---

## 6. Critical Risk Controls

> **These must be enforced before any signal is acted upon.**

### 6.1 Data Quality Gates

```python
def validate_data(data: dict, ticker: str) -> list[str]:
    warnings = []
    
    # Staleness
    if data["fundamentals_age_days"] > 120:
        warnings.append("STALE_FUNDAMENTALS: data older than one quarter")
    
    # Restatements
    if data.get("restatement_flag"):
        warnings.append("RESTATEMENT: recent financial restatement detected")
    
    # Negative book value
    if data["total_equity"] < 0:
        warnings.append("NEGATIVE_EQUITY: RIM invalid; DCF weights adjusted")
    
    # Extreme values
    if data["pe_ratio"] > 200 or data["pe_ratio"] < 0:
        warnings.append("EXTREME_PE: comps may be distorted")
    
    # Model disagreement > 40%
    values = [v for v in [data["ggm"], data["dcf"], data["comps"], data["rim"]] if v]
    if values and (max(values) - min(values)) / (sum(values)/len(values)) > 0.40:
        warnings.append("HIGH_MODEL_DISAGREEMENT: widen CI, reduce position sizing")
    
    return warnings
```

### 6.2 Model Invalidity Rules

| Condition | Action |
|---|---|
| No dividend history | Skip GGM; weight DCF + Comps + RIM |
| g ≥ r in GGM | Skip GGM; log warning |
| Negative FCF (3 consecutive years) | Skip DCF or use analyst estimates |
| Negative earnings | Skip P/E comps; use EV/Revenue or P/S |
| No comparable peers found | Skip Comps |
| Fewer than 2 valid models | Return INSUFFICIENT_DATA, no signal |

### 6.3 Margin of Safety Thresholds

```
MOS = (Intrinsic Value - Market Price) / Intrinsic Value

MOS > 30% → Strong BUY signal (high conviction)
MOS 20-30% → BUY signal
MOS -20% to 20% → HOLD
MOS -30% to -20% → SELL signal
MOS < -30% → Strong SELL signal

For HIGH model disagreement: require MOS > 40% for BUY signal
```

### 6.4 What This System Is NOT

- Not a trading system (no execution, no order management)
- Not a guarantee of returns (intrinsic value ≠ near-term price)
- Not suitable as sole basis for investment decisions
- Not adjusted for qualitative factors (management, moat, ESG)
- Not calibrated for illiquid or micro-cap stocks (< $100M market cap)

---

## 7. Implementation Roadmap

### Phase 1 — Proof of Concept (CSV Approach, Week 1-2)
- [ ] Download Simfin bulk data
- [ ] Implement GGM, DCF, Comps, RIM in Python
- [ ] Test on 10 well-known stocks (AAPL, MSFT, KO, JNJ, etc.)
- [ ] Compare outputs to known analyst targets
- [ ] Export CSV + Markdown report

### Phase 2 — API Integration (Week 3-4)
- [ ] Integrate FMP API for live fundamentals
- [ ] Integrate FRED API for real-time risk-free rate
- [ ] Add rate limiting, error handling, retry logic
- [ ] Automate valuation for S&P 500 universe

### Phase 3 — Agent Layer (Week 5-6)
- [ ] Build tool-calling agent with Anthropic/OpenAI SDK
- [ ] Add 10-K text parsing (SEC EDGAR)
- [ ] Add earnings call sentiment (optional)
- [ ] Structured JSON output with flags

### Phase 4 — ML Enhancement (Week 7-8)
- [ ] Build historical dataset (2010–2024)
- [ ] Feature engineering pipeline
- [ ] Train XGBoost ensemble
- [ ] Conformal prediction for CI
- [ ] Backtest: Sharpe ratio, max drawdown, hit rate

### Phase 5 — Production Pipeline (Week 9-10)
- [ ] Airflow / cron scheduler
- [ ] SQLite → PostgreSQL migration
- [ ] Streamlit dashboard
- [ ] Alert system (email/Slack on BUY signal)
- [ ] Monitoring: data freshness, model drift

---

## 8. File & Folder Structure

```
stock_valuation_ml/
├── data/
│   ├── raw/                  # Downloaded CSVs, API responses
│   ├── processed/            # Feature store, cleaned fundamentals
│   └── outputs/              # valuation results CSV/JSON
├── models/
│   ├── valuation/
│   │   ├── ggm.py
│   │   ├── dcf.py
│   │   ├── comps.py
│   │   └── rim.py
│   ├── ensemble.py
│   └── ml/
│       ├── train.py
│       ├── predict.py
│       └── conformal.py
├── agents/
│   ├── orchestrator.py
│   ├── tools.py
│   └── prompts.py
├── pipeline/
│   ├── ingest.py
│   ├── batch_valuation.py
│   └── scheduler.py
├── reports/
│   └── templates/
│       └── valuation_report.md
├── tests/
│   ├── test_ggm.py
│   ├── test_dcf.py
│   └── test_data_quality.py
├── config.py                 # API keys from env vars, thresholds
├── requirements.txt
└── README.md                 # This file (agent-readable)
```

---

## 9. Key Libraries

```
# requirements.txt
pandas>=2.0
numpy>=1.26
requests>=2.31
xgboost>=2.0
lightgbm>=4.0
scikit-learn>=1.3
mapie>=0.8              # conformal prediction
yfinance>=0.2           # unofficial Yahoo Finance
simfin>=0.9             # SimFin data
anthropic>=0.25         # Agent layer
python-dotenv>=1.0      # API key management
sqlalchemy>=2.0         # DB abstraction
airflow>=2.8            # Scheduling (Phase 5)
streamlit>=1.30         # Dashboard (Phase 5)
```

---

## 10. Quick Reference — Key Formulas

```
GGM:    V = D0(1+g) / (r - g)          | requires: dividend, g < r
DCF:    V = Σ FCFt/(1+WACC)^t + TV     | TV = FCFn(1+g) / (WACC - g_terminal)
CAPM:   r = Rf + β(Rm - Rf)            | Rf = 10Y Treasury, Rm-Rf ≈ 5.5%
WACC:   = (E/V)Re + (D/V)Rd(1-T)       | always use market-value weights
RIM:    V = BV + Σ (NI - r×BV) / (1+r)^t
MOS:    = (IV - Price) / IV             | buy signal threshold: 20-30%
g_sust: = ROE × (1 - payout ratio)     | sustainable growth rate
```

---

*Document version: 1.0 | Last updated: 2025 | Maintained for: humans + AI agents*  
*Do not act on signals without independent verification. Not investment advice.*
