# Stockinator: Solutions & Production Architecture

This document provides a comprehensive roadmap for resolving the problems identified in the Stockinator project. A major focus is placed on establishing a robust, multi-tiered data retrieval strategy encompassing Web Scraping, API Retrieval, Agentic AI, and Enterprise Vendor Data.

---

## 1. The 4-Tiered Data Retrieval Strategy

To replace the brittle `yfinance` integration, Stockinator should implement a cascaded data retrieval pipeline that balances cost, reliability, and unstructured data processing.

### Tier A: API Retrieval (The Core Foundation)
APIs should be the primary, default method for fetching structured financial data (Pricing, Income Statements, Balance Sheets, Cash Flows).
*   **Implementation:** Replace `yfinance` with a reliable financial API provider. Excellent choices include:
    *   **Financial Modeling Prep (FMP):** Offers extensive fundamental data, discounted cash flows, and peer lists.
    *   **Alpha Vantage / Polygon.io:** Excellent for real-time and historical pricing data.
*   **Fixes Addressed:** Resolves IP bans, unexpected structural breakages, and ensures institutional-grade data accuracy. You can query dynamic peers directly via API (e.g., fetching all tickers with the same GICS industry code).

### Tier B: Agentic AI (Unstructured Extraction)
For qualitative data, management sentiment, and non-standard metrics not found in typical APIs.
*   **Implementation:** Utilize Groq/Gemini combined with RAG (Retrieval-Augmented Generation).
    *   **SEC Filings:** Have the AI Agent download 10-Ks and 10-Qs, parse the "Management Discussion and Analysis" (MD&A) sections, and extract sentiment or hidden risks.
    *   **Earnings Calls:** Transcribe earnings calls and use the Agent to extract specific forward-looking guidance or planned CAPEX adjustments that the DCF model requires.
*   **Fixes Addressed:** Enhances the valuation engine beyond mere quantitative metrics, allowing for dynamic adjustments to growth rates based on management sentiment.

### Tier C: Vendor Data / Bloomberg (The Enterprise Tier)
If targeting institutional clients or a highly premium tier, direct integration with tier-1 vendors is required.
*   **Implementation:** Connect to Bloomberg Data License (via B-PIPE) or FactSet.
    *   This provides access to proprietary consensus estimates, institutional holdings, and highly scrubbed fundamental data.
*   **Fixes Addressed:** Provides the absolute highest standard of data reliability, effectively eliminating the need for "fallback default assumptions" like standard cost_of_debt.

### Tier D: Web Scraping (The Intelligent Fallback)
Scraping should strictly be a fallback of last resort, managed gracefully.
*   **Implementation:** If API data fails and Vendor data is unavailable, use headless browsers (e.g., Playwright) to scrape structured tables from macro-economic sites or SEC EDGAR. 
*   **Fixes Addressed:** Instead of relying on brittle `lxml` Wikipedia scraping, use Agentic web scraping tools (like Browser Use or Selenium) that can visually navigate and handle changing DOM elements gracefully.

---

## 2. Architectural & Scaling Solutions

### 2.1 Asynchronous I/O via `httpx` and `async def`
*   **Action:** Refactor `api/server.py` endpoints to use `async def`.
*   **Action:** Replace synchronous `requests` and `yfinance` calls with `httpx.AsyncClient`.
*   **Benefit:** FastAPI will leverage its native async event loop. A single server instance will easily handle thousands of concurrent valuation requests without thread starvation.

### 2.2 Migrate to PostgreSQL
*   **Action:** Replace SQLite with PostgreSQL for production.
*   **Action:** Update `database/db.py` to use an ORM like SQLAlchemy configured with a connection pool (e.g., PgBouncer).
*   **Benefit:** Enables horizontal scaling across multiple stateless Docker containers, ensures data durability, and allows for complex analytical queries on historical valuations.

### 2.3 Distributed Caching via Redis
*   **Action:** Replace the local dictionary `_data_cache` with a Redis instance.
*   **Action:** Implement tools like `FastAPI-Cache` backed by Redis.
*   **Benefit:** Ensures that all Uvicorn worker processes share the same cache. If Worker A fetches AAPL data, Worker B can instantly retrieve it from Redis, heavily reducing API costs and latency.

---

## 3. Security & API Hardening

### 3.1 Rate Limiting & Auth
*   **Action:** Implement API Gateway or middleware rate limiting (e.g., `slowapi`). Restrict users to X valuations per minute.
*   **Action:** Secure endpoints using JWT authentication or API Keys via FastAPI `Depends`.

### 3.2 Prompt Hardening for AI Agents
*   **Action:** Sandbox the AI Agent. If the agent needs to query the database, it should only have access to a read-only database user with strict row-level security.
*   **Action:** Implement semantic filtering on user inputs to detect and block Prompt Injection attempts before passing the text to the Groq/Gemini models.

---

## 4. DevOps & Codebase Maturity

### 4.1 Database Migrations with Alembic
*   **Action:** Integrate Alembic into the project. Generate migration scripts (`alembic revision --autogenerate`) instead of raw SQL `CREATE TABLE` commands.
*   **Benefit:** Allows seamless, version-controlled schema upgrades and rollbacks in production without data loss.

### 4.2 Dependency Management & Config Validation
*   **Action:** Migrate from `requirements.txt` to Poetry (`pyproject.toml`) or `uv` to lock exact dependency hashes.
*   **Action:** Use Pydantic `BaseSettings` for configuration. This forces the app to crash immediately upon boot if critical variables like `GROQ_API_KEY` or `DB_URL` are missing or malformed.

### 4.3 Structured Logging
*   **Action:** Replace `logging.basicConfig` with `structlog`. Output logs in JSON format for easy ingestion into Datadog, Grafana Loki, or AWS CloudWatch.

---

## 5. Financial Modeling & ML Enhancements

### 5.1 Dynamic Valuation Parameters
*   **Action:** Remove hardcoded caps (e.g., 15% growth caps). Instead, use statistical outlier detection (like IQR or Z-scores) based on the specific industry's historical distribution.
*   **Action:** Use the Tier B (Agentic AI) approach to dynamically assess risk and alter the Cost of Equity and Beta based on current news sentiment, rather than defaulting to generic macro numbers.

### 5.2 Correct Panel Data Time-Series CV
*   **Action:** Update `models/ml/train.py`. Since the data contains multiple tickers over time (Panel Data), use grouped time-series splits (e.g., Scikit-learn's `GroupKFold` mixed with temporal boundaries, or specialized libraries like `sktime`).
*   **Benefit:** Completely eliminates temporal leakage, ensuring the model's backtest accuracy accurately reflects real-world trading performance.

### 5.3 ML Model Registry
*   **Action:** Integrate `MLflow` or `Weights & Biases`.
*   **Benefit:** Every time `train.py` runs, it logs the hyperparameter configuration, the dataset version, and the resulting MAE/RMSE. The production API can dynamically pull the "champion" model from the registry rather than relying on a static `.pkl` file.
