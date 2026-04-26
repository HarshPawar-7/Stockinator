"""Tests for AI Agent Tools."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tools import dispatch_tool, TOOL_SCHEMAS


class TestToolSchemas:
    def test_schemas_are_valid(self):
        """All tool schemas should have required fields."""
        assert len(TOOL_SCHEMAS) >= 4
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_all_tools_named(self):
        """All tools should have unique names."""
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert len(names) == len(set(names))


class TestToolDispatch:
    def test_unknown_tool(self):
        """Unknown tool name should return error."""
        result = json.loads(dispatch_tool("nonexistent_tool", {}))
        assert "error" in result

    def test_macro_tool(self):
        """get_macro_data should return valid JSON."""
        result = json.loads(dispatch_tool("get_macro_data", {}))
        assert "risk_free_rate" in result
        assert result["risk_free_rate"] > 0

    def test_ggm_tool(self):
        """GGM tool with valid inputs."""
        result = json.loads(dispatch_tool("run_ggm_valuation", {
            "d0": 2.0, "g": 0.03, "r": 0.10, "price": 25.0
        }))
        assert result.get("valid") is True
        assert result.get("value") is not None
