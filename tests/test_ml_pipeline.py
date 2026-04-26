"""Tests for ML Feature Engineering Pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ml.features import build_features_from_result, build_feature_matrix


class TestFeatureExtraction:
    def _make_result(self, **overrides):
        """Create a mock valuation result dict."""
        result = {
            "ticker": "TEST",
            "valuation_date": "2026-04-26",
            "market_price": 150.0,
            "models": {
                "ggm": {"value": 120.0, "valid": True},
                "dcf": {"value": 160.0, "valid": True},
                "comps": {"value": 140.0, "valid": True},
                "rim": {"value": 130.0, "valid": True},
            },
            "ensemble": {
                "ensemble_value": 140.0,
                "margin_of_safety": -0.07,
                "signal": "HOLD",
                "valid_model_count": 4,
            },
            "inputs": {
                "risk_free_rate": 0.043,
                "beta": 1.1,
                "cost_of_equity": 0.10,
                "wacc": 0.09,
            },
        }
        result.update(overrides)
        return result

    def test_basic_extraction(self):
        """Features should be extracted from a valid result."""
        features = build_features_from_result(self._make_result())
        assert features["ticker"] == "TEST"
        assert features["ggm_value"] == 120.0
        assert features["dcf_value"] == 160.0
        assert features["ensemble_value"] == 140.0
        assert features["risk_free_rate"] == 0.043

    def test_model_disagreement(self):
        """Model disagreement should be computed."""
        features = build_features_from_result(self._make_result())
        assert features["model_disagreement"] is not None
        assert features["model_disagreement"] > 0

    def test_price_to_ensemble(self):
        """Price/ensemble ratio should be computed."""
        features = build_features_from_result(self._make_result())
        assert features["price_to_ensemble"] is not None
        # 150 / 140 ≈ 1.07
        assert abs(features["price_to_ensemble"] - 1.071) < 0.01

    def test_missing_models(self):
        """Should handle None model values gracefully."""
        result = self._make_result()
        result["models"]["ggm"]["value"] = None
        result["models"]["rim"]["value"] = None
        features = build_features_from_result(result)
        assert features["ggm_value"] is None
        assert features["rim_value"] is None
        assert features["dcf_value"] == 160.0

    def test_batch_matrix(self):
        """Feature matrix from multiple results."""
        results = [self._make_result(ticker="A"), self._make_result(ticker="B")]
        df = build_feature_matrix(results)
        assert len(df) == 2
        assert "ticker" in df.columns
        assert "ensemble_value" in df.columns
