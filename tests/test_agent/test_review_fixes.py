"""Regression tests for the 2026-08-31 release-engineering review fixes
(agent-side). Each test failed against the pre-fix code — see
docs/plans/2026-08-31-release-engineering.md.
"""

from __future__ import annotations

import pytest

from anchor.agent.agent import Agent, _estimate_cost
from anchor.agent.models import AgentTool
from anchor.agent.skills.models import Skill
from anchor.llm.models import Message, Role
from tests.test_agent.test_agent import FakeLLMProvider


def _plain_tool(name: str) -> AgentTool:
    return AgentTool(
        name=name, description="d", input_schema={"type": "object"},
        fn=lambda: "ok",
    )


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


class TestCompactionKeepLastZero:
    """Finding: keep_last=0 made _split_for_compaction index one past the
    end (IndexError mid-turn on the first over-trigger round)."""

    def test_keep_last_zero_summarizes_everything(self):
        agent = Agent(llm=FakeLLMProvider([]))
        agent.with_compaction(1, keep_last=0)
        msgs = [
            Message(role=Role.USER, content="w1 w2 w3 w4 w5 w6 w7 w8"),
            Message(role=Role.ASSISTANT, content="a1 a2 a3 a4 a5 a6 a7 a8"),
        ]
        split = agent._split_for_compaction(msgs)  # pre-fix: IndexError
        if split is not None:
            head, tail = split
            assert head == msgs
            assert tail == []


class TestSkillMetaToolCollision:
    """Finding 22: skill meta-tools were created with no name check —
    two 'activate_skill' schemas reached the provider and first-match
    execution shadowed the meta-tool silently."""

    def test_user_tool_then_skill_raises(self):
        agent = Agent(llm=FakeLLMProvider([]))
        agent.with_tools([_plain_tool("activate_skill")])
        skill = Skill(
            name="db", description="d", activation="on_demand",
            tools=(_plain_tool("query_db"),),
        )
        with pytest.raises(ValueError, match="activate_skill"):
            agent.with_skill(skill)

    def test_skill_then_user_tool_raises(self):
        agent = Agent(llm=FakeLLMProvider([]))
        skill = Skill(
            name="db", description="d", activation="on_demand",
            tools=(_plain_tool("query_db"),),
        )
        agent.with_skill(skill)
        with pytest.raises(ValueError, match="activate_skill"):
            agent.with_tools([_plain_tool("activate_skill")])


class TestAstreamMemoryOrder:
    """Finding: astream persisted the user message BEFORE _ensure_mcp
    could raise — a retry after a connect failure duplicated the message
    in conversation memory (stream() orders its MCP guard first)."""

    def test_failed_mcp_connect_does_not_persist_message(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        agent = Agent(llm=FakeLLMProvider([]))
        memory = MagicMock()
        agent.with_memory(memory)

        async def run():
            with patch.object(
                Agent, "_ensure_mcp",
                AsyncMock(side_effect=RuntimeError("connect failed")),
            ):
                async for _ in agent.astream("hello"):
                    pass

        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(run())
        memory.add_user_message.assert_not_called()


class TestPriceMemoDoesNotShadowOverrides:
    """Finding 42: the unknown-model memo was consulted BEFORE the price
    table — a runtime MODEL_PRICING override never took effect."""

    def test_runtime_pricing_override_prices_after_first_miss(self):
        from anchor.llm.pricing import MODEL_PRICING

        model_id = "mock/review-fix-test-model"
        bare = "review-fix-test-model"  # the table is keyed sans provider
        try:
            assert _estimate_cost(model_id, 1000, 1000, 0, 0) == 0.0
            MODEL_PRICING[bare] = {"input": 1.0, "output": 2.0}
            assert _estimate_cost(model_id, 1000, 1000, 0, 0) > 0.0
        finally:
            MODEL_PRICING.pop(bare, None)
