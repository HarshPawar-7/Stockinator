"""Tests for Residual Income Model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.valuation.rim import compute_rim


class TestRIM:
    def test_valid_rim(self):
        """Standard RIM with positive spread."""
        result = compute_rim(
            book_value_per_share=50.0,
            earnings_per_share=7.50,
            cost_of_equity=0.10,
            current_price=100.0,
        )
        assert result.valid is True
        assert result.intrinsic_value is not None
        assert result.intrinsic_value > 50.0  # Should exceed book value (ROE > r)

    def test_negative_book_value(self):
        """Negative BV → invalid."""
        result = compute_rim(
            book_value_per_share=-10.0,
            earnings_per_share=5.0,
            cost_of_equity=0.10,
        )
        assert result.valid is False
        assert "NEGATIVE_EQUITY" in result.warnings[0]

    def test_value_destroyer(self):
        """ROE < cost of equity → value destroyer warning."""
        result = compute_rim(
            book_value_per_share=50.0,
            earnings_per_share=2.0,  # ROE = 4% < 10%
            cost_of_equity=0.10,
            current_price=30.0,
        )
        assert any("VALUE_DESTROYER" in w for w in result.warnings)

    def test_roe_provided(self):
        """Explicit ROE should be used."""
        result = compute_rim(
            book_value_per_share=50.0,
            earnings_per_share=0.0,
            cost_of_equity=0.10,
            roe=0.15,
        )
        assert result.valid is True
        assert result.roe == 0.15

    def test_margin_of_safety(self):
        """MOS should be computed."""
        result = compute_rim(
            book_value_per_share=50.0,
            earnings_per_share=10.0,
            cost_of_equity=0.10,
            current_price=80.0,
        )
        if result.valid:
            assert result.margin_of_safety is not None
