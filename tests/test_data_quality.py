"""Tests for Data Quality Validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data_quality import validate_stock_data, validate_model_outputs
from datetime import datetime


class MockStockData:
    """Mock stock data for testing."""
    def __init__(self, **kwargs):
        defaults = {
            "fetch_date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": 150.0,
            "market_cap": 2e12,
            "shares_outstanding": 15e9,
            "total_equity": 70e9,
            "total_debt": 100e9,
            "pe_ratio": 25.0,
            "historical_fcf": [80e9, 90e9, 100e9],
            "dividend_per_share": 0.96,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestValidateStockData:
    def test_clean_data(self):
        """Good data → no warnings."""
        data = MockStockData()
        warnings = validate_stock_data(data, "AAPL")
        # May have some minor warnings but nothing critical
        critical = [w for w in warnings if "STALE" in w or "NEGATIVE_EQUITY" in w]
        assert len(critical) == 0

    def test_negative_equity(self):
        """Negative equity → warning."""
        data = MockStockData(total_equity=-5e9)
        warnings = validate_stock_data(data, "TEST")
        assert any("NEGATIVE_EQUITY" in w for w in warnings)

    def test_extreme_pe(self):
        """P/E > 200 → warning."""
        data = MockStockData(pe_ratio=500.0)
        warnings = validate_stock_data(data, "TEST")
        assert any("EXTREME_PE" in w for w in warnings)

    def test_no_dividend(self):
        """No dividend → warning."""
        data = MockStockData(dividend_per_share=0.0)
        warnings = validate_stock_data(data, "TEST")
        assert any("NO_DIVIDEND" in w for w in warnings)

    def test_high_leverage(self):
        """Extreme D/E ratio → warning."""
        data = MockStockData(total_debt=500e9, total_equity=10e9)
        warnings = validate_stock_data(data, "TEST")
        assert any("HIGH_LEVERAGE" in w for w in warnings)


class TestValidateModelOutputs:
    def test_all_valid(self):
        """All models valid → minimal warnings."""
        warnings = validate_model_outputs({
            "ggm": 100, "dcf": 110, "comps": 105, "rim": 108
        })
        insufficient = [w for w in warnings if "INSUFFICIENT" in w]
        assert len(insufficient) == 0

    def test_insufficient_models(self):
        """Only 1 valid model → warning."""
        warnings = validate_model_outputs({
            "ggm": None, "dcf": 110, "comps": None, "rim": None
        })
        assert any("INSUFFICIENT" in w for w in warnings)

    def test_high_disagreement(self):
        """Models far apart → disagreement warning."""
        warnings = validate_model_outputs({
            "ggm": 50, "dcf": 200, "comps": 80, "rim": 180
        })
        assert any("DISAGREEMENT" in w for w in warnings)
