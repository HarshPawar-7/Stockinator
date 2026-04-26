"""
Orchestrator Agent — Groq-Powered Financial Analysis

Autonomous valuation agent that uses tool calling to:
1. Fetch real financial data
2. Run valuation models
3. Analyze and explain results

Uses Groq API (free tier) with Llama models for fast inference.
"""

from __future__ import annotations

import json
import logging
import os

from agents.tools import TOOL_SCHEMAS, dispatch_tool
from agents.prompts import SYSTEM_PROMPT, VALUATION_REQUEST_PROMPT, COMPARISON_PROMPT

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ROUNDS = 8  # Safety limit on tool-calling loops


def _get_groq_client():
    """Lazily import and create Groq client."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys\n"
            "Then add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        )


def run_agent(
    user_message: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> str:
    """
    Run the valuation agent with a user query.

    The agent will:
    1. Parse the user's request
    2. Call appropriate tools (fetch data, run models)
    3. Synthesize results into a clear analysis

    Args:
        user_message: Natural language query (e.g., "Value AAPL")
        model: Groq model to use
        verbose: Print intermediate tool calls

    Returns:
        Agent's final analysis as a string
    """
    client = _get_groq_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        logger.info("Agent round %d...", round_num)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.1,  # Low temp for financial accuracy
        )

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, we have the final response
        if not message.tool_calls:
            logger.info("Agent completed in %d rounds", round_num)
            return message.content or ""

        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  🔧 Tool: {fn_name}({json.dumps(fn_args, default=str)[:100]})")

            logger.info("Calling tool: %s(%s)", fn_name, fn_args)
            result = dispatch_tool(fn_name, fn_args)

            if verbose:
                # Truncate for display
                display = result[:200] + "..." if len(result) > 200 else result
                print(f"  📦 Result: {display}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    logger.warning("Agent hit max rounds (%d) — returning last response", MAX_TOOL_ROUNDS)
    return messages[-1].get("content", "Analysis incomplete — max tool rounds reached.")


def valuate_stock(ticker: str, verbose: bool = False) -> str:
    """
    Convenience function: run full valuation for a single stock.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        verbose: Print tool calls to stdout

    Returns:
        Formatted analysis string
    """
    prompt = VALUATION_REQUEST_PROMPT.format(ticker=ticker)
    return run_agent(prompt, verbose=verbose)


def compare_stocks(tickers: list[str], verbose: bool = False) -> str:
    """
    Convenience function: compare multiple stocks.

    Args:
        tickers: List of ticker symbols
        verbose: Print tool calls to stdout

    Returns:
        Formatted comparison analysis
    """
    prompt = COMPARISON_PROMPT.format(tickers=", ".join(tickers))
    return run_agent(prompt, verbose=verbose)


def chat(verbose: bool = False) -> None:
    """
    Interactive chat mode with the valuation agent.

    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 60)
    print("  🤖 Stockinator AI Agent")
    print("  Powered by Groq (free tier)")
    print("=" * 60)
    print("\nAsk me to value any stock! Examples:")
    print("  • Value AAPL")
    print("  • Compare MSFT GOOGL AMZN")
    print("  • What's the intrinsic value of KO?")
    print("  • Is JNJ undervalued?")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        print("\n🔄 Analyzing...\n")
        try:
            response = run_agent(user_input, verbose=verbose)
            print(f"\n📊 Stockinator:\n{response}\n")
        except ValueError as e:
            print(f"\n❌ {e}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logger.error("Agent error: %s", e, exc_info=True)
