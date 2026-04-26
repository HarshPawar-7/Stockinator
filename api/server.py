"""
Stockinator REST API — FastAPI Backend

Exposes the valuation engine as a REST API for the web dashboard.

Endpoints:
    POST /api/valuate         — Run valuation on tickers
    GET  /api/valuate/{ticker} — Single stock valuation
    GET  /api/history         — Historical valuations from DB
    POST /api/agent           — Send message to AI agent
    GET  /api/macro           — Get macro data
    GET  /api/sp500           — Get S&P 500 ticker list
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stockinator API",
    description="ML-Based Stock Valuation Platform",
    version="1.0.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ────────────────────────────────────────

class ValuationRequest(BaseModel):
    tickers: list[str]
    include_peers: bool = True

class AgentRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stockinator"}


@app.post("/api/valuate")
def valuate_stocks(req: ValuationRequest):
    """Run valuation on multiple tickers."""
    from pipeline.batch_valuation import run_batch_valuation

    if not req.tickers:
        raise HTTPException(400, "No tickers provided")
    if len(req.tickers) > 20:
        raise HTTPException(400, "Max 20 tickers per request")

    # Clean tickers
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]

    try:
        results = run_batch_valuation(tickers=tickers, fetch_peers=req.include_peers)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("Valuation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Valuation error: {str(e)}")


@app.get("/api/valuate/{ticker}")
def valuate_single(ticker: str, include_peers: bool = True):
    """Run valuation on a single stock."""
    from pipeline.batch_valuation import valuate_single_stock

    ticker = ticker.strip().upper()
    try:
        result = valuate_single_stock(ticker, fetch_peers=include_peers)
        return result
    except Exception as e:
        logger.error("Valuation failed for %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, f"Valuation error: {str(e)}")


@app.get("/api/history")
def get_history(limit: int = 50):
    """Get recent valuations from the database."""
    from database.db import init_db
    import sqlite3
    import json

    try:
        db_path = Path(__file__).parent.parent / "stockinator.db"
        if not db_path.exists():
            return {"valuations": [], "count": 0}

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM valuations ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return {"valuations": rows, "count": len(rows)}
    except Exception as e:
        logger.error("History fetch failed: %s", e)
        return {"valuations": [], "count": 0, "error": str(e)}


@app.post("/api/agent")
def agent_chat(req: AgentRequest):
    """Send a message to the AI agent."""
    try:
        from agents.orchestrator import run_agent
        response = run_agent(req.message, verbose=False)
        return AgentResponse(response=response)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ImportError:
        raise HTTPException(501, "Groq not installed. Run: pip install groq")
    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True)
        raise HTTPException(500, f"Agent error: {str(e)}")


@app.get("/api/macro")
def get_macro():
    """Get current macroeconomic data."""
    try:
        from pipeline.fred_api import fetch_macro_data
        return fetch_macro_data().to_dict()
    except Exception as e:
        from config import RISK_FREE_RATE_DEFAULT, MARKET_PREMIUM_DEFAULT
        return {
            "risk_free_rate": RISK_FREE_RATE_DEFAULT,
            "market_premium": MARKET_PREMIUM_DEFAULT,
            "source": "defaults",
            "error": str(e),
        }


@app.get("/api/sp500")
def get_sp500(source: str = "fallback"):
    """Get S&P 500 ticker list."""
    from pipeline.sp500 import load_sp500_tickers
    tickers = load_sp500_tickers(source=source)
    return {"tickers": tickers, "count": len(tickers)}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
