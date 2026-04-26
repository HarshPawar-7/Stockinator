"""Tests for Discounted Cash Flow Model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.valuation.dcf import compute_dcf, compute_wacc, project_fcf


class TestWACC:
    def test_basic_wacc(self):
        """WACC with standard inputs."""
        wacc = compute_wacc(
            market_cap=2e12, total_debt=1e11,
            cost_of_equity=0.10, cost_of_debt=0.04, tax_rate=0.21
        )
        assert 0.05 < wacc < 0.15

    def test_no_debt(self):
        """All-equity firm → WACC = cost of equity."""
        wacc = compute_wacc(
            market_cap=1e9, total_debt=0,
            cost_of_equity=0.10, cost_of_debt=0.04, tax_rate=0.21
        )
        assert abs(wacc - 0.10) < 0.001


class TestProjectFCF:
    def test_basic_projection(self):
        """Project from growing FCF history."""
        historical = [80e9, 90e9, 100e9, 110e9]
        projections, growth = project_fcf(historical, forecast_years=5)
        assert len(projections) == 5
        assert all(p > 0 for p in projections)
        assert projections[0] > historical[-1]  # Should grow

    def test_insufficient_data(self):
        """Should raise with < 2 years."""
        try:
            project_fcf([100e9])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestDCF:
    def test_valid_dcf(self):
        """Standard DCF calculation."""
        result = compute_dcf(
            fcf_forecasts=[100e9, 105e9, 110e9, 115e9, 120e9],
            wacc=0.10, terminal_growth=0.025,
            shares_outstanding=15e9, net_debt=50e9,
            current_price=200.0,
        )
        assert result.valid is True
        assert result.intrinsic_value_per_share is not None
        assert result.intrinsic_value_per_share > 0

    def test_terminal_growth_exceeds_wacc(self):
        """g_term >= WACC → invalid."""
        result = compute_dcf(
            fcf_forecasts=[100e9], wacc=0.05, terminal_growth=0.06,
        )
        assert result.valid is False
        assert "INVALID_DCF" in result.warnings[0]

    def test_no_fcf(self):
        """Empty FCF list → invalid."""
        result = compute_dcf(fcf_forecasts=[], wacc=0.10)
        assert result.valid is False

    def test_margin_of_safety(self):
        """MOS should be computed when price is given."""
        result = compute_dcf(
            fcf_forecasts=[10e9, 11e9, 12e9, 13e9, 14e9],
            wacc=0.10, shares_outstanding=1e9,
            current_price=100.0,
        )
        if result.valid and result.intrinsic_value_per_share:
            assert result.margin_of_safety is not None
