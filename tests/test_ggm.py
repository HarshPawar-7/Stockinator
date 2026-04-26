"""Tests for Gordon Growth Model."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.valuation.ggm import compute_ggm, compute_capm, compute_sustainable_growth


class TestCAPM:
    def test_basic_capm(self):
        """CAPM with typical values."""
        r = compute_capm(risk_free_rate=0.043, beta=1.0, market_premium=0.055)
        assert abs(r - 0.098) < 0.001

    def test_high_beta(self):
        """High beta should increase required return."""
        r = compute_capm(risk_free_rate=0.043, beta=1.5, market_premium=0.055)
        assert r > 0.12

    def test_zero_beta(self):
        """Zero beta should equal risk-free rate."""
        r = compute_capm(risk_free_rate=0.043, beta=0.0, market_premium=0.055)
        assert abs(r - 0.043) < 0.001


class TestSustainableGrowth:
    def test_basic(self):
        """Standard sustainable growth calculation."""
        g = compute_sustainable_growth(roe=0.15, payout_ratio=0.40)
        assert abs(g - 0.09) < 0.001  # 15% * 60% = 9%

    def test_full_payout(self):
        """100% payout → zero growth."""
        g = compute_sustainable_growth(roe=0.20, payout_ratio=1.0)
        assert g == 0.0

    def test_no_payout(self):
        """Zero payout → growth = ROE."""
        g = compute_sustainable_growth(roe=0.12, payout_ratio=0.0)
        assert abs(g - 0.12) < 0.001


class TestGGM:
    def test_valid_ggm(self):
        """Standard GGM with KO-like inputs."""
        result = compute_ggm(d0=1.84, growth_rate=0.04, discount_rate=0.09, current_price=60.0)
        assert result.valid is True
        assert result.intrinsic_value is not None
        assert result.intrinsic_value > 0
        # D1 = 1.84 * 1.04 = 1.9136; V = 1.9136 / (0.09-0.04) = 38.27
        assert abs(result.intrinsic_value - 38.27) < 0.1

    def test_no_dividend(self):
        """No dividend → GGM invalid."""
        result = compute_ggm(d0=0.0, growth_rate=0.05, discount_rate=0.10)
        assert result.valid is False
        assert "NO_DIVIDEND" in result.warnings[0]

    def test_growth_exceeds_discount(self):
        """g >= r → GGM undefined."""
        result = compute_ggm(d0=2.0, growth_rate=0.12, discount_rate=0.10)
        assert result.valid is False
        assert "INVALID_GGM" in result.warnings[0]

    def test_growth_equals_discount(self):
        """g == r → GGM undefined."""
        result = compute_ggm(d0=2.0, growth_rate=0.10, discount_rate=0.10)
        assert result.valid is False

    def test_margin_of_safety(self):
        """MOS calculated correctly."""
        result = compute_ggm(d0=2.0, growth_rate=0.03, discount_rate=0.10, current_price=25.0)
        assert result.valid is True
        # V = 2.0*1.03 / (0.10-0.03) = 29.43
        assert result.margin_of_safety is not None
        assert result.margin_of_safety > 0  # Undervalued

    def test_narrow_spread_warning(self):
        """Very narrow r-g spread should warn."""
        result = compute_ggm(d0=2.0, growth_rate=0.095, discount_rate=0.10)
        assert result.valid is True
        assert any("NARROW_SPREAD" in w for w in result.warnings)
