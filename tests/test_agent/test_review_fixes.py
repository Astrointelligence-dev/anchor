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


class TestSubagentLastRoundValidation:
    """Finding 8: structured output validated the text accumulated across
    ALL rounds — narration in a tool round glued onto the final JSON and
    burned retries even when every final reply was schema-valid."""

    def test_narrating_tool_round_does_not_break_validation(self):
        import json as _json

        from anchor.llm.models import StopReason, StreamChunk, ToolCallDelta
        from pydantic import BaseModel
        from tests.test_agent.test_agent import (
            _Tok,
            _text_response,
            _tool_use_response,
        )

        class Answer(BaseModel):
            answer: str

        narrate_then_tool = [
            StreamChunk(content="Let me check. "),
            StreamChunk(tool_call_delta=ToolCallDelta(
                index=0, id="tc1", name="lookup",
            )),
            StreamChunk(tool_call_delta=ToolCallDelta(
                index=0, arguments_fragment=_json.dumps({}),
            )),
            StreamChunk(stop_reason=StopReason.TOOL_USE),
        ]
        sub_provider = FakeLLMProvider([
            narrate_then_tool,
            _text_response('{"answer": "42"}'),
        ])
        sub = Agent(llm=sub_provider, tokenizer=_Tok())
        sub.with_tools([_plain_tool("lookup")])

        orch_provider = FakeLLMProvider([
            _tool_use_response("tu_1", "researcher", {"task": "answer"}),
            _text_response("done"),
        ])
        orch = Agent(llm=orch_provider, tokenizer=_Tok())
        orch.with_tools([
            sub.as_tool("researcher", "d", output_model=Answer),
        ])

        assert "".join(orch.chat("Go")) == "done"
        # Exactly 2 child calls: narration + final JSON, NO retry burned.
        assert len(sub_provider.seen_messages) == 2
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(orch_provider, 1)
        assert result.is_error is False
        assert _json.loads(result.content) == {"answer": "42"}


class TestNoPhantomChildTurn:
    """Finding 2: a child failing BEFORE its turn started forwarded the
    previous turn's diagnostics as a duplicate ChildTurn."""

    def test_pre_start_failure_does_not_replay_old_diagnostics(self):
        from tests.test_agent.test_agent import (
            _Tok,
            _text_response,
            _tool_use_response,
        )

        sub_provider = FakeLLMProvider([_text_response("42")])
        sub = Agent(llm=sub_provider, tokenizer=_Tok())

        orch_provider = FakeLLMProvider([
            _tool_use_response("tu_1", "researcher", {"task": "a"}),
            _text_response("first done"),
            _tool_use_response("tu_2", "researcher", {"task": "b"}),
            _text_response("second done"),
        ])
        orch = Agent(llm=orch_provider, tokenizer=_Tok())
        orch.with_tools([sub.as_tool("researcher", "d")])

        assert "".join(orch.chat("Go")) == "first done"
        assert len(orch.last_turn.children) == 1

        # Make the child fail BEFORE _reset_turn_state: the sync MCP
        # guard raises at the top of stream().
        sub._mcp_configs.append("http://unreachable.example/mcp")
        assert "".join(orch.chat("Again")) == "second done"
        # Pre-fix: turn 1's diagnostics reappeared as a phantom child.
        assert orch.last_turn.children == ()


class TestSameChildSerialization:
    """Finding 1: two parallel task calls to the same child ran two
    turns concurrently on one Agent instance (pool disarm, output
    cross-talk). The runner now serializes per child."""

    @pytest.mark.asyncio
    async def test_parallel_same_child_calls_serialize(self):
        from tests.test_agent.test_agent import _Tok, _text_response
        from tests.test_agent.test_phase4_loop import (
            _multi_tool_use_response,
            _tool_results_of,
        )

        sub_provider = FakeLLMProvider([
            _text_response("answer-A"),
            _text_response("answer-B"),
        ])
        sub = Agent(llm=sub_provider, tokenizer=_Tok())

        orch_provider = FakeLLMProvider([
            _multi_tool_use_response([
                ("tu_1", "researcher", {"task": "a"}),
                ("tu_2", "researcher", {"task": "b"}),
            ]),
            _text_response("done"),
        ])
        orch = Agent(llm=orch_provider, tokenizer=_Tok())
        orch.with_tools([sub.as_tool("researcher", "d")])

        text = ""
        async for chunk in orch.achat("Go"):
            text += chunk
        assert text == "done"

        # The serialization lock was armed on the shared child (pre-fix:
        # the attribute did not exist), and both calls completed with
        # their own results in order.
        assert sub._turn_lock is not None
        results = _tool_results_of(orch_provider, 1)
        contents = {r.content for r in results}
        assert contents == {"answer-A", "answer-B"}
        assert len(orch.last_turn.children) == 2


class TestRetrySeesOwnReply:
    """Finding 9: a plain-text stop with structured output pending
    appended only the nudge — the retry ran without the model's own
    last answer in context, as consecutive user-role messages."""

    def test_nudge_round_carries_assistant_reply(self):
        from pydantic import BaseModel

        from tests.test_agent.test_agent import (
            _Tok,
            _text_response,
            _tool_use_response,
        )

        class Answer(BaseModel):
            answer: str

        provider = FakeLLMProvider([
            _text_response("I think it is 42."),   # plain text, no tool
            _tool_use_response("tu_1", "final_result", {"answer": "42"}),
        ])
        agent = Agent(llm=provider, tokenizer=_Tok())
        agent.with_output_model(Answer)

        result = agent.run("Question?")
        assert result.answer == "42"

        retry_messages = provider.seen_messages[1]
        roles = [m.role for m in retry_messages]
        # The model's own reply precedes the nudge — no user/user pair.
        assistant_texts = [
            str(m.content) for m in retry_messages
            if m.role == Role.ASSISTANT
        ]
        assert any("I think it is 42." in t for t in assistant_texts)
        assert [r for r in roles[-2:]] != [Role.USER, Role.USER]


class TestAgentEnablesPromptCaching:
    """Finding 10 (minimal leg): the Agent never enabled provider
    prompt caching — no cache_control ever reached the wire."""

    def test_create_provider_receives_prompt_caching(self):
        from unittest.mock import MagicMock, patch

        with patch(
            "anchor.agent.agent.create_provider",
            return_value=MagicMock(),
        ) as cp:
            Agent(model="claude-test")
        assert cp.call_args.kwargs.get("prompt_caching") is True


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
