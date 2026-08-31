"""Regression tests for the 2026-08-31 release-engineering review fixes
(agent-side). Each test failed against the pre-fix code — see
docs/plans/2026-08-31-release-engineering.md.
"""

from __future__ import annotations

from anchor.agent.agent import Agent


class TestBuildToolCallsMalformedArgs:
    """Finding 40: model-streamed args JSON parsed without a guard killed
    the whole turn; now it degrades to {} like the non-streaming path."""

    def test_malformed_json_degrades_to_empty_args(self):
        accs = {0: {"id": "c1", "name": "t", "args_json": '{"q": "x'}}
        calls = Agent._build_tool_calls(accs)
        assert calls[0].arguments == {}
        assert calls[0].name == "t"

    def test_non_object_json_degrades_to_empty_args(self):
        accs = {0: {"id": "c1", "name": "t", "args_json": '"just a string"'}}
        calls = Agent._build_tool_calls(accs)
        assert calls[0].arguments == {}

    def test_missing_id_gets_fallback(self):
        accs = {2: {"id": None, "name": "t", "args_json": "{}"}}
        calls = Agent._build_tool_calls(accs)
        assert calls[0].id == "call_2"

    def test_valid_args_unchanged(self):
        accs = {0: {"id": "c1", "name": "t", "args_json": '{"q": 1}'}}
        calls = Agent._build_tool_calls(accs)
        assert calls[0].arguments == {"q": 1}
