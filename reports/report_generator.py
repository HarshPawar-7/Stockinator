"""
Report Generator

Produces Markdown, CSV, and JSON reports from valuation results.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from config import REPORTS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "valuation_report.md"


def _fmt_pct(val: float | None, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}"


def _fmt_dollar(val: float | None) -> str:
    """Format a value as dollar amount."""
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def _model_status(model_dict: dict) -> str:
    """Return ✅ or ❌ based on model validity."""
    return "✅" if model_dict.get("valid") else "❌"


def _model_inputs_str(model_dict: dict) -> str:
    """Format model inputs as a compact string."""
    inputs = model_dict.get("inputs", {})
    parts = []
    for k, v in inputs.items():
        if v is None:
            continue
        if isinstance(v, float):
            if abs(v) < 1:
                parts.append(f"{k}={v:.4f}")
            elif abs(v) > 1e6:
                parts.append(f"{k}={v/1e9:.1f}B")
            else:
                parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "—"


def generate_markdown_report(result: dict) -> str:
    """
    Generate a formatted Markdown report for a single stock valuation.
    """
    template = TEMPLATE_PATH.read_text()

    ensemble = result.get("ensemble", {})
    models = result.get("models", {})
    inputs = result.get("inputs", {})
    ci = ensemble.get("ci_95", [None, None]) or [None, None]

    # Risk flags
    warnings = result.get("data_quality", {}).get("warnings", [])
    if warnings:
        risk_flags = "\n".join(f"- ⚠️ {w}" for w in warnings)
    else:
        risk_flags = "✅ None identified"

    # Weights table
    weights = ensemble.get("weights", {})
    if weights:
        weights_table = "| Model | Weight |\n|-------|--------|\n"
        weights_table += "\n".join(f"| {k.upper()} | {v:.1%} |" for k, v in weights.items())
    else:
        weights_table = "N/A (insufficient models)"

    report = template.format(
        ticker=result.get("ticker", "?"),
        company_name=result.get("company_name", ""),
        valuation_date=result.get("valuation_date", ""),
        market_price=_fmt_dollar(result.get("market_price"))[1:],
        ensemble_value=_fmt_dollar(ensemble.get("ensemble_value"))[1:],
        ci_lower=_fmt_dollar(ci[0])[1:] if ci[0] else "N/A",
        ci_upper=_fmt_dollar(ci[1])[1:] if ci[1] else "N/A",
        margin_of_safety=_fmt_pct(ensemble.get("margin_of_safety")),
        signal=ensemble.get("signal", "N/A"),
        valid_models=ensemble.get("valid_model_count", 0),
        # GGM
        ggm_value=_fmt_dollar(models.get("ggm", {}).get("value")),
        ggm_status=_model_status(models.get("ggm", {})),
        ggm_inputs=_model_inputs_str(models.get("ggm", {})),
        # DCF
        dcf_value=_fmt_dollar(models.get("dcf", {}).get("value")),
        dcf_status=_model_status(models.get("dcf", {})),
        dcf_inputs=_model_inputs_str(models.get("dcf", {})),
        # Comps
        comps_value=_fmt_dollar(models.get("comps", {}).get("value")),
        comps_status=_model_status(models.get("comps", {})),
        comps_inputs=_model_inputs_str(models.get("comps", {})),
        # RIM
        rim_value=_fmt_dollar(models.get("rim", {}).get("value")),
        rim_status=_model_status(models.get("rim", {})),
        rim_inputs=_model_inputs_str(models.get("rim", {})),
        # Assumptions
        risk_free_rate=_fmt_pct(inputs.get("risk_free_rate")),
        market_premium=_fmt_pct(inputs.get("market_premium")),
        beta=f"{inputs.get('beta', 'N/A')}",
        cost_of_equity=_fmt_pct(inputs.get("cost_of_equity")),
        wacc=_fmt_pct(inputs.get("wacc")),
        # Flags
        risk_flags=risk_flags,
        weights_table=weights_table,
    )

    return report


def generate_csv_output(results: list[dict]) -> str:
    """Generate CSV output from a batch of valuation results."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "ticker", "date", "price", "ggm_value", "dcf_value", "comps_value",
        "rim_value", "ensemble_value", "ci_lower_95", "ci_upper_95",
        "margin_of_safety", "signal", "flags"
    ])

    for r in results:
        ensemble = r.get("ensemble", {})
        models = r.get("models", {})
        ci = ensemble.get("ci_95", [None, None]) or [None, None]
        warnings = r.get("data_quality", {}).get("warnings", [])

        writer.writerow([
            r.get("ticker"),
            r.get("valuation_date"),
            r.get("market_price"),
            models.get("ggm", {}).get("value"),
            models.get("dcf", {}).get("value"),
            models.get("comps", {}).get("value"),
            models.get("rim", {}).get("value"),
            ensemble.get("ensemble_value"),
            ci[0], ci[1],
            ensemble.get("margin_of_safety"),
            ensemble.get("signal"),
            json.dumps(warnings[:5]),  # Limit for CSV readability
        ])

    return output.getvalue()


def generate_json_output(results: list[dict]) -> str:
    """Generate pretty-printed JSON output."""
    return json.dumps(results, indent=2, default=str)


def save_reports(results: list[dict]) -> dict[str, Path]:
    """Save all report formats to disk."""
    saved = {}

    # CSV
    csv_path = OUTPUT_DIR / "valuations.csv"
    csv_path.write_text(generate_csv_output(results))
    saved["csv"] = csv_path

    # JSON
    json_path = OUTPUT_DIR / "valuations.json"
    json_path.write_text(generate_json_output(results))
    saved["json"] = json_path

    # Individual Markdown reports
    for result in results:
        ticker = result.get("ticker", "unknown")
        try:
            md = generate_markdown_report(result)
            md_path = REPORTS_DIR / f"{ticker}_valuation.md"
            md_path.write_text(md)
            saved[f"md_{ticker}"] = md_path
        except Exception as e:
            logger.error("Failed to generate MD report for %s: %s", ticker, e)

    logger.info("Reports saved: %s", {k: str(v) for k, v in saved.items()})
    return saved
