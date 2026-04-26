"""Tests for FRED API Integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.fred_api import MacroData, fetch_macro_data, get_risk_free_rate


class TestFREDFallback:
    def test_no_api_key_fallback(self):
        """Without FRED_API_KEY, should fall back to defaults."""
        import os
        original = os.environ.get("FRED_API_KEY")
        try:
            os.environ.pop("FRED_API_KEY", None)
            # Re-import to pick up env change
            import importlib
            import pipeline.fred_api as fred_mod
            importlib.reload(fred_mod)

            macro = fred_mod.fetch_macro_data()
            assert macro.source == "defaults"
            assert macro.risk_free_rate > 0
            assert macro.market_premium > 0
        finally:
            if original:
                os.environ["FRED_API_KEY"] = original

    def test_macro_data_structure(self):
        """MacroData should have all expected fields."""
        macro = MacroData(
            risk_free_rate=0.043,
            yield_spread_10_2=0.005,
            federal_funds_rate=0.05,
            cpi_yoy=0.03,
            gdp_growth=0.025,
            market_premium=0.055,
            fetch_date="2026-04-26",
            source="test",
        )
        d = macro.to_dict()
        assert d["risk_free_rate"] == 0.043
        assert d["source"] == "test"
        assert "yield_spread_10_2" in d

    def test_risk_free_rate_positive(self):
        """Risk-free rate should always be positive."""
        rf = get_risk_free_rate()
        assert rf > 0
        assert rf < 0.20  # Sanity check: not above 20%
