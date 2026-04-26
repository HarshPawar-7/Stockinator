"""Tests for Comparable Company Analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.valuation.comps import compute_comps, select_best_multiple


class TestSelectMultiple:
    def test_financial_sector(self):
        """Financial companies should use P/B."""
        assert select_best_multiple({"sector": "Financial Services"}) == "pb"

    def test_positive_ebitda(self):
        """Positive EBITDA → EV/EBITDA."""
        assert select_best_multiple({"ebitda": 1e9, "sector": "Tech"}) == "ev_ebitda"

    def test_negative_ebitda_positive_income(self):
        """Negative EBITDA, positive income → P/E."""
        assert select_best_multiple({"ebitda": -1e9, "net_income": 5e8, "sector": "Tech"}) == "pe"


class TestComps:
    def test_valid_comps(self):
        """Standard comps with peer data."""
        company = {
            "ebitda": 100e9, "net_debt": 50e9,
            "shares_outstanding": 15e9, "sector": "Technology",
        }
        peers = [
            {"ticker": "MSFT", "ev_ebitda_ratio": 25.0},
            {"ticker": "GOOGL", "ev_ebitda_ratio": 20.0},
            {"ticker": "META", "ev_ebitda_ratio": 15.0},
            {"ticker": "NVDA", "ev_ebitda_ratio": 30.0},
        ]
        result = compute_comps(company, peers, multiple_type="ev_ebitda", current_price=200.0)
        assert result.valid is True
        assert result.intrinsic_value is not None
        assert result.num_peers == 4

    def test_no_peers(self):
        """No peers → invalid."""
        result = compute_comps({"ebitda": 1e9}, [], current_price=100.0)
        assert result.valid is False
        assert "NO_PEERS" in result.warnings[0]

    def test_pe_comps(self):
        """P/E-based comps."""
        company = {"eps": 6.50, "sector": "Technology"}
        peers = [
            {"ticker": "MSFT", "pe_ratio": 30.0},
            {"ticker": "GOOGL", "pe_ratio": 25.0},
            {"ticker": "META", "pe_ratio": 20.0},
        ]
        result = compute_comps(company, peers, multiple_type="pe", current_price=150.0)
        assert result.valid is True
        # Median P/E = 25, EPS = 6.50 → value = 162.50
        assert abs(result.intrinsic_value - 162.50) < 1.0
