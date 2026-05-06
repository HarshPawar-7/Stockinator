# Stockinator: Deep Problem Analysis

This document outlines the architectural, data, security, and modeling issues currently present in the Stockinator project. These issues prevent the application from being considered production-ready.

## 1. Data Sourcing & Reliability

### 1.1 Heavy Reliance on `yfinance`
Currently, the pipeline (`pipeline/ingest.py`) relies heavily on the `yfinance` library to fetch stock fundamentals, pricing, and historical cash flows. 
*   **The Problem:** `yfinance` is essentially a web scraper disguised as an API. It works by intercepting data from Yahoo Finance's frontend. It is highly brittle, subject to aggressive IP rate-limiting, and breaks without warning whenever Yahoo changes their HTML or API structure. It is unsuitable for any mission-critical financial application.

### 1.2 Brittle Wikipedia Scraping
For loading S&P 500 tickers, the project falls back to scraping Wikipedia tables using `lxml`.
*   **The Problem:** Wikipedia editors frequently update table layouts, change CSS classes, or alter column headers. A minor edit to the Wikipedia page will completely break the ticker ingestion pipeline.

### 1.3 Static Peer Mapping
Comparable company analysis (Comps) relies on a hardcoded dictionary (`PEER_MAP`) mapping broad sectors to a static list of large-cap tickers.
*   **The Problem:** This ignores thousands of mid-cap and small-cap stocks. It also fails to account for dynamic shifts in the market, delisted stocks, or new IPOs. Hardcoding financial relationships limits the valuation engine to only the most well-known tech and consumer giants.

## 2. Architecture & Scaling Limitations

### 2.1 Synchronous, Blocking I/O
The FastAPI backend (`api/server.py`) defines its core endpoints using synchronous `def` functions, and the ingestion pipeline uses synchronous `requests` and `yfinance`.
*   **The Problem:** When multiple users request valuations simultaneously, the blocking I/O calls will consume the entire thread pool. FastAPI will be unable to accept new connections, leading to severe bottlenecks, timeouts, and a degraded user experience.

### 2.2 Local SQLite Database
The project uses SQLite with `PRAGMA journal_mode=WAL` for persistence.
*   **The Problem:** While WAL mode improves local concurrency, SQLite is a file-bound database. If this application is deployed to a modern cloud environment (like Docker on AWS ECS, Kubernetes, or Vercel), the ephemeral file system means the database will be lost on every container restart. Furthermore, it prevents horizontal scaling across multiple server instances.

### 2.3 Process-Bound Caching
Data caching is implemented via an in-memory python dictionary (`_data_cache`).
*   **The Problem:** If you run this server with multiple Uvicorn workers (e.g., `workers=4`), each worker gets its own isolated cache. This leads to redundant API calls and cache inconsistency across the cluster.

## 3. Security & API Vulnerabilities

### 3.1 Open, Unprotected Endpoints
The `/api/valuate` endpoint is completely open to the public.
*   **The Problem:** Without Authentication (JWT, API Keys) or Authorization, anyone can access your backend resources. There is also no rate-limiting, allowing malicious actors to send massive arrays of tickers, effectively performing a Denial of Service (DoS) attack and instantly getting the server's IP banned by data providers.

### 3.2 AI Prompt Injection
The `/api/agent` endpoint blindly accepts user input (`req.message`) and feeds it directly into the LLM orchestrator.
*   **The Problem:** If the AI agent is ever granted access to execute code, query the database, or hit external APIs, it is highly vulnerable to Prompt Injection. A user could trick the LLM into dropping database tables, revealing internal API keys, or executing unauthorized actions.

## 4. DevOps & Codebase Standards

### 4.1 Lack of Database Migrations
Database schemas are defined as raw, hardcoded SQL strings in `db.py`.
*   **The Problem:** There is no safe way to alter the schema in production. If you need to add a new column for a new financial metric, you have to manually run `ALTER TABLE` scripts against your production database, which is extremely error-prone and risks data loss.

### 4.2 Unpinned Dependencies
The `requirements.txt` specifies loose version bounds (e.g., `xgboost>=2.0`, `pandas>=2.0`).
*   **The Problem:** This breaks the principle of deterministic, reproducible builds. A minor update to a sub-dependency could silently break the ML pipeline or the backend API in the future.

### 4.3 Silent Failures on Environment Variables
API keys (like `GROQ_API_KEY`) are fetched via `os.getenv` without startup validation.
*   **The Problem:** The server will start up successfully even if keys are missing. It will only crash later during runtime when a user actually tries to trigger an agent workflow, leading to a poor user experience.

### 4.4 Global Logging Configuration
The app uses `logging.basicConfig()` which overrides the root logger.
*   **The Problem:** In a production microservice, logs need to be structured (usually as JSON) so they can be parsed, filtered, and monitored by tools like Datadog, AWS CloudWatch, or ELK. Simple string logs make automated alerting nearly impossible.

## 5. Financial Modeling & ML Loopholes

### 5.1 Arbitrary Mathematical Caps
To prevent extreme ROE or buyback anomalies from breaking the math, the models employ hardcoded caps (e.g., capping Sustainable Growth at 15% and RIM ROE at 50%).
*   **The Problem:** While this acts as a safety net, it broadly penalizes hyper-growth companies and removes the nuance required for accurate high-growth tech valuations.

### 5.2 Dangerous Default Assumptions
The ingestion pipeline falls back to defaults (e.g., `beta=1.0`, `cost_of_debt=0.04`, `tax_rate=0.21`) when data is missing.
*   **The Problem:** These broad strokes will result in completely detached, highly inaccurate intrinsic valuations for companies where these averages do not apply.

### 5.3 ML Temporal Leakage
The `train.py` script uses `TimeSeriesSplit` for cross-validation on the feature matrix `X`.
*   **The Problem:** If the `X` dataframe contains a mix of multiple tickers (Panel Data) and isn't strictly grouped and sorted by Date across the entire universe, the split will blindly slice rows. Data from a future date for one ticker might leak into the training set predicting a past date for another ticker.

### 5.4 Missing ML Model Registry
Trained models are saved as raw `.pkl` files in the repository.
*   **The Problem:** There is no lineage tracking, performance monitoring, or hyperparameter versioning. If the model degrades (data drift), there is no structured way to roll back to a previous champion model.
