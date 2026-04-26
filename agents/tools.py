"""
Agent Tool Definitions

Defines all tools the AI agent can call to perform stock valuation.
Each tool wraps a core pipeline function with proper error handling.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from pipeline.ingest import fetch_stock_data, fetch_peer_data
from pipeline.batch_valuation import valuate_single_stock
from pipeline.data_quality import validate_stock_data
from models.valuation.ggm import compute_ggm, compute_capm
from models.valuation.dcf import compute_dcf, compute_wacc
from models.valuation.comps import compute_comps
from models.valuation.rim import compute_rim
from models.ensemble import weighted_ensemble

logger = logging.getLogger(__name__)


# ── Tool Schema Definitions (for Groq/OpenAI function calling) ─────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_fundamentals",
            "description": "Fetch comprehensive fundamental data for a stock ticker including financials, ratios, and derived metrics. Returns all data needed for valuation models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_valuation",
            "description": "Run complete valuation pipeline on a stock: fetches data, runs GGM/DCF/Comps/RIM models, combines via ensemble, and returns intrinsic value with BUY/HOLD/SELL signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "include_peers": {
                        "type": "boolean",
                        "description": "Whether to include peer comparison (slower but more accurate). Default true.",
                        "default": True
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_ggm_valuation",
            "description": "Run Gordon Growth Model valuation. Returns intrinsic value based on dividend discount. Returns null if company doesn't pay dividends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "d0": {"type": "number", "description": "Last annual dividend per share"},
                    "g": {"type": "number", "description": "Dividend growth rate (decimal, e.g., 0.05 for 5%)"},
                    "r": {"type": "number", "description": "Required rate of return from CAPM (decimal)"},
                    "price": {"type": "number", "description": "Current market price for margin of safety calc"}
                },
                "required": ["d0", "g", "r", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_dcf_valuation",
            "description": "Run Discounted Cash Flow valuation with projected free cash flows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fcf_forecasts": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Projected Free Cash Flows for each forecast year"
                    },
                    "wacc": {"type": "number", "description": "Weighted Average Cost of Capital (decimal)"},
                    "terminal_growth": {"type": "number", "description": "Terminal growth rate (decimal, max 0.04)"},
                    "shares_outstanding": {"type": "number", "description": "Number of shares outstanding"},
                    "net_debt": {"type": "number", "description": "Total debt minus cash"},
                    "price": {"type": "number", "description": "Current market price"}
                },
                "required": ["fcf_forecasts", "wacc", "terminal_growth"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare valuation of multiple stocks side by side. Returns a comparison table with all models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ticker symbols to compare (max 10)"
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_data",
            "description": "Get current macroeconomic data: risk-free rate, yield spread, GDP growth.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
]


# ── Tool Implementations ───────────────────────────────────────────

def dispatch_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call to the appropriate implementation.

    Returns JSON string with the result.
    """
    try:
        if name == "fetch_fundamentals":
            return _tool_fetch_fundamentals(**arguments)
        elif name == "run_full_valuation":
            return _tool_run_valuation(**arguments)
        elif name == "run_ggm_valuation":
            return _tool_run_ggm(**arguments)
        elif name == "run_dcf_valuation":
            return _tool_run_dcf(**arguments)
        elif name == "compare_stocks":
            return _tool_compare(**arguments)
        elif name == "get_macro_data":
            return _tool_macro()
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return json.dumps({"error": str(e), "tool": name})


def _tool_fetch_fundamentals(ticker: str) -> str:
    data = fetch_stock_data(ticker)
    return json.dumps({
        "ticker": data.ticker,
        "company_name": data.company_name,
        "sector": data.sector,
        "industry": data.industry,
        "price": data.current_price,
        "market_cap": data.market_cap,
        "pe_ratio": data.pe_ratio,
        "pb_ratio": data.pb_ratio,
        "roe": data.roe,
        "eps": data.eps,
        "dividend_per_share": data.dividend_per_share,
        "dividend_yield": data.dividend_yield,
        "beta": data.beta,
        "free_cash_flow": data.free_cash_flow,
        "revenue": data.revenue,
        "net_income": data.net_income,
        "ebitda": data.ebitda,
        "total_debt": data.total_debt,
        "total_equity": data.total_equity,
        "book_value_per_share": data.book_value_per_share,
        "cost_of_equity": data.cost_of_equity,
        "wacc": data.wacc,
        "sustainable_growth": data.sustainable_growth,
        "warnings": data.warnings,
    }, default=str)


def _tool_run_valuation(ticker: str, include_peers: bool = True) -> str:
    result = valuate_single_stock(ticker, fetch_peers=include_peers)
    return json.dumps(result, default=str)


def _tool_run_ggm(d0: float, g: float, r: float, price: float) -> str:
    result = compute_ggm(d0=d0, growth_rate=g, discount_rate=r, current_price=price)
    return json.dumps(result.to_dict(), default=str)


def _tool_run_dcf(fcf_forecasts: list, wacc: float, terminal_growth: float,
                  shares_outstanding: float = None, net_debt: float = 0,
                  price: float = None) -> str:
    result = compute_dcf(
        fcf_forecasts=fcf_forecasts, wacc=wacc,
        terminal_growth=terminal_growth,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt, current_price=price,
    )
    return json.dumps(result.to_dict(), default=str)


def _tool_compare(tickers: list[str]) -> str:
    tickers = tickers[:10]  # Limit to 10
    from pipeline.batch_valuation import run_batch_valuation
    results = run_batch_valuation(tickers, fetch_peers=False)

    comparison = []
    for r in results:
        e = r.get("ensemble", {})
        comparison.append({
            "ticker": r.get("ticker"),
            "price": r.get("market_price"),
            "intrinsic_value": e.get("ensemble_value"),
            "margin_of_safety": e.get("margin_of_safety"),
            "signal": e.get("signal"),
            "valid_models": e.get("valid_model_count"),
        })
    return json.dumps(comparison, default=str)


def _tool_macro() -> str:
    try:
        from pipeline.fred_api import fetch_macro_data
        return json.dumps(fetch_macro_data().to_dict(), default=str)
    except Exception:
        from config import RISK_FREE_RATE_DEFAULT, MARKET_PREMIUM_DEFAULT
        return json.dumps({
            "risk_free_rate": RISK_FREE_RATE_DEFAULT,
            "market_premium": MARKET_PREMIUM_DEFAULT,
            "source": "defaults",
        })
