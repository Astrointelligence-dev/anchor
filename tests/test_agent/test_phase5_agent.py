"""Phase 5 agent tests: context management, emulated compaction, cleanups."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from anchor._text import strip_markdown_fences
from anchor.agent.agent import Agent
from anchor.agent.subagent import SubagentDefinition
from anchor.llm.models import Role, StreamChunk
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import _echo_tool


def _agent(
    responses: list[list[StreamChunk]], *, max_rounds: int = 10, **kwargs,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, max_rounds=max_rounds, tokenizer=_Tok(), **kwargs)
    agent.with_system_prompt("You are helpful.")
    return agent, provider


# ---------------------------------------------------------------------------
# tool_choice on the final round
# ---------------------------------------------------------------------------


def test_final_round_forces_tool_choice_none():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "one"}),
        _tool_use_response("tu_2", "echo", {"x": "two"}),
    ]
    agent, provider = _agent(responses, max_rounds=2)
    agent.with_tools([_echo_tool()])
    list(agent.chat("Go"))

    assert "tool_choice" not in provider.seen_kwargs[0]
    assert provider.seen_kwargs[1]["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# Context management passthrough + raw-block round-trip
# ---------------------------------------------------------------------------


def test_context_management_passed_every_round():
    config = {"edits": [{"type": "clear_tool_uses_20250919"}]}
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "go"}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses)
    agent.with_tools([_echo_tool()]).with_context_management(config)
    list(agent.chat("Go"))

    assert all(
        kw.get("context_management") == config for kw in provider.seen_kwargs
    )


def test_compaction_raw_block_round_trips_verbatim():
    block = {"type": "compaction", "content": "earlier turns summarized"}
    round_one = [
        StreamChunk(raw_block=block),
        *_tool_use_response("tu_1", "echo", {"x": "go"}),
    ]
    responses = [round_one, _text_response("done")]
    agent, provider = _agent(responses)
    agent.with_tools([_echo_tool()])
    assert "".join(agent.chat("Go")) == "done"

    assistant = next(
        m
        for m in provider.seen_messages[1]
        if m.role == Role.ASSISTANT and m.tool_calls
    )
    assert assistant.raw_content == [block]


# ---------------------------------------------------------------------------
# Emulated (client-side) compaction — works with any provider
# ---------------------------------------------------------------------------


def test_emulated_compaction_replaces_old_messages():
    long_text = "word " * 40  # 40 tokens under _Tok
    responses = [
        _tool_use_response("tu_1", "echo", {"x": long_text}),
        _tool_use_response("tu_2", "echo", {"x": long_text}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses)
    agent.with_tools([_echo_tool()])
    agent.with_compaction(
        trigger_tokens=30, keep_last=2, compact_fn=lambda transcript: "SUMMARY",
    )
    assert "".join(agent.chat("Go")) == "done"

    # Round 3's request was compacted: a summary message replaced the head.
    final_messages = provider.seen_messages[2]
    contents = [str(m.content) for m in final_messages if m.content]
    assert any("[Conversation summary]\nSUMMARY" in c for c in contents)
    # The kept tail preserves the last tool_use/tool_result pair.
    assert any(m.role == Role.TOOL for m in final_messages)
    # Compacted request is shorter than it would have been (round 2 had
    # more raw messages than round 3 has after compaction).
    assert len(final_messages) <= len(provider.seen_messages[1])


def test_emulated_compaction_default_uses_agent_llm():
    long_text = "word " * 40
    responses = [
        _tool_use_response("tu_1", "echo", {"x": long_text}),
        _tool_use_response("tu_2", "echo", {"x": long_text}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses)
    agent.with_tools([_echo_tool()])
    agent.with_compaction(trigger_tokens=30, keep_last=2)
    # Default path summarizes via TierCompactor → agent's own llm.invoke.
    summary_response = MagicMock()
    summary_response.content = "LLM SUMMARY"
    provider.invoke = MagicMock(return_value=summary_response)  # type: ignore[method-assign]

    assert "".join(agent.chat("Go")) == "done"
    provider.invoke.assert_called_once()
    contents = [str(m.content) for m in provider.seen_messages[2] if m.content]
    assert any("[Conversation summary]\nLLM SUMMARY" in c for c in contents)


async def test_emulated_compaction_async_path():
    long_text = "word " * 40
    responses = [
        _tool_use_response("tu_1", "echo", {"x": long_text}),
        _tool_use_response("tu_2", "echo", {"x": long_text}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses)
    agent.with_tools([_echo_tool()])
    agent.with_compaction(
        trigger_tokens=30, keep_last=2, compact_fn=lambda transcript: "SUMMARY",
    )
    chunks = [c async for c in agent.achat("Go")]
    assert "".join(chunks) == "done"
    contents = [str(m.content) for m in provider.seen_messages[2] if m.content]
    assert any("[Conversation summary]\nSUMMARY" in c for c in contents)


def test_compaction_not_triggered_below_threshold():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "short"}),
        _text_response("done"),
    ]
    agent, _provider = _agent(responses)
    agent.with_tools([_echo_tool()])
    called = []
    agent.with_compaction(
        trigger_tokens=10_000,
        compact_fn=lambda t: called.append(t) or "S",
    )
    list(agent.chat("Go"))
    assert called == []


# ---------------------------------------------------------------------------
# Meta-tool name collision checks
# ---------------------------------------------------------------------------


def test_with_subagents_rejects_existing_task_tool():
    agent, _ = _agent([_text_response("hi")])
    agent.with_tools([_echo_tool("task")])
    with pytest.raises(ValueError, match="reserved for the subagent dispatcher"):
        agent.with_subagents([SubagentDefinition(name="w", description="d")])


def test_with_tools_rejects_task_after_subagents():
    agent, _ = _agent([_text_response("hi")])
    agent.with_subagents([SubagentDefinition(name="w", description="d")])
    with pytest.raises(ValueError, match="already registered as an agent meta-tool"):
        agent.with_tools([_echo_tool("task")])


def test_deferred_with_user_search_tools_collides():
    deferred = _echo_tool("obscure").model_copy(update={"defer_loading": True})
    agent, _ = _agent([_text_response("hi")])
    agent.with_tools([_echo_tool("search_tools"), deferred])
    with pytest.raises(ValueError, match="reserved for"):
        list(agent.chat("Go"))


# ---------------------------------------------------------------------------
# last_turn lifecycle
# ---------------------------------------------------------------------------


def test_last_turn_reset_at_turn_start():
    agent, _ = _agent([_text_response("one"), _text_response("two")])
    list(agent.chat("First"))
    assert agent.last_turn is not None

    generator = agent.chat("Second")
    next(generator)  # start the turn, then abandon it
    assert agent.last_turn is None
    generator.close()


# ---------------------------------------------------------------------------
# strip_markdown_fences (shared util + compactor regression)
# ---------------------------------------------------------------------------


def test_strip_markdown_fences_variants():
    assert strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    # Missing closing fence: last line must NOT be dropped.
    assert strip_markdown_fences('```json\n{"a": 1}') == '{"a": 1}'
    assert strip_markdown_fences('{"a": 1}') == '{"a": 1}'
    assert strip_markdown_fences("```json\n```") == ""


def test_compactor_parses_fenced_json_without_closing_fence():
    from anchor.memory.compactor import TierCompactor

    compactor = TierCompactor(MagicMock(), tokenizer=_Tok())
    raw = '```json\n[{"type": "decision", "content": "Use FastAPI"}]'
    facts = compactor._parse_facts(raw, source_tier=1)
    assert len(facts) == 1
    assert facts[0].content == "Use FastAPI"
