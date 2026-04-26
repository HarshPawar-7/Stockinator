"""Tests for Ensemble Combiner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ensemble import weighted_ensemble


class TestEnsemble:
    def test_all_models_valid(self):
        """All four models producing valid results."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=100.0,
            dcf_value=120.0,
            comps_value=110.0,
            rim_value=105.0,
            current_price=95.0,
        )
        assert result.signal != "INSUFFICIENT_DATA"
        assert result.ensemble_value is not None
        assert result.valid_model_count == 4
        assert result.margin_of_safety is not None

    def test_two_models_valid(self):
        """Only two models valid — should still work."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=None,
            dcf_value=120.0,
            comps_value=110.0,
            rim_value=None,
            current_price=100.0,
        )
        assert result.valid_model_count == 2
        assert result.ensemble_value is not None
        assert result.signal != "INSUFFICIENT_DATA"

    def test_one_model_insufficient(self):
        """Only one model → INSUFFICIENT_DATA."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=None,
            dcf_value=120.0,
            comps_value=None,
            rim_value=None,
            current_price=100.0,
        )
        assert result.signal == "INSUFFICIENT_DATA"
        assert result.valid_model_count == 1

    def test_buy_signal(self):
        """Undervalued stock → BUY."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=150.0,
            dcf_value=160.0,
            comps_value=155.0,
            rim_value=145.0,
            current_price=100.0,
        )
        assert "BUY" in result.signal

    def test_sell_signal(self):
        """Overvalued stock → SELL."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=70.0,
            dcf_value=75.0,
            comps_value=72.0,
            rim_value=68.0,
            current_price=120.0,
        )
        assert "SELL" in result.signal

    def test_weight_redistribution(self):
        """Weights should sum to 1.0 even with missing models."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=None,
            dcf_value=120.0,
            comps_value=110.0,
            rim_value=105.0,
            current_price=100.0,
        )
        total_weight = sum(result.model_weights_used.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_model_disagreement(self):
        """High disagreement should be flagged."""
        result = weighted_ensemble(
            ticker="TEST",
            ggm_value=50.0,
            dcf_value=150.0,
            comps_value=80.0,
            rim_value=200.0,
            current_price=100.0,
        )
        assert result.model_disagreement is not None
        assert result.model_disagreement > 0.3  # Should flag
