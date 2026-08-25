"""Event stream of the agent loop: Agent.stream / Agent.astream.

Message-level tests over FakeLLMProvider asserting the ordered event
sequence, tool_call_id correlation, live ToolFinished under parallel
execution, error-as-event semantics, the text projections, and the
try/finally bookkeeping on abandoned generators.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anchor.agent import Agent, AgentTool
from anchor.agent.events import (
    CompactionFinished,
    CompactionStarted,
    RoundFinished,
    RoundStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import (
    _echo_tool,
    _multi_tool_use_response,
    _tool_results_of,
)


def _agent(
    responses: list[list[Any]],
    *,
    tools: list[AgentTool] | None = None,
    max_rounds: int = 10,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, max_rounds=max_rounds, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    if tools:
        agent.with_tools(tools)
    return agent, provider


def _async_sleep_tool(name: str, delay: float, reply: str) -> AgentTool:
    tool = AgentTool(
        name=name,
        description="sleeps then replies",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: reply,
    )

    async def acall(_name: str, _input: dict[str, Any]) -> str:
        await asyncio.sleep(delay)
        return reply

    object.__setattr__(tool, "_anchor_async_caller", acall)
    return tool


class _Recorder:
    """Duck-typed AgentCallback recording every notification."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_tool_error(self, name: str, tool_input: dict, error: str) -> None:
        self.events.append(("tool_error", name, error))


# ---------------------------------------------------------------------------
# Ordered sequence and correlation ids
# ---------------------------------------------------------------------------


def test_stream_emits_ordered_events_single_round():
    agent, _ = _agent([_text_response("Hello!")])

    events = list(agent.stream("Hi"))

    assert [e.type for e in events] == [
        "turn_started", "round_started", "text_delta",
        "round_finished", "turn_finished",
    ]
    started = events[1]
    assert isinstance(started, RoundStarted)
    assert (started.round, started.max_rounds) == (0, 10)
    finished = events[-1]
    assert isinstance(finished, TurnFinished)
    assert finished.text == "Hello!"
    assert finished.diagnostics.stopped_by == "stop"
    assert finished.diagnostics is agent.last_turn
    assert all(e.parent_tool_call_id is None for e in events)


def test_stream_tool_round_events_carry_ids():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])

    events = list(agent.stream("Go"))

    assert [e.type for e in events] == [
        "turn_started",
        "round_started", "tool_started", "tool_finished", "round_finished",
        "round_started", "text_delta", "round_finished",
        "turn_finished",
    ]
    tool_started = next(e for e in events if isinstance(e, ToolStarted))
    tool_finished = next(e for e in events if isinstance(e, ToolFinished))
    assert tool_started.tool_call_id == "tu_1"
    assert tool_started.name == "echo"
    assert tool_started.tool_input == {"x": "hi"}
    assert tool_finished.tool_call_id == "tu_1"
    assert tool_finished.result == "echo:hi"
    assert tool_finished.is_error is False
    rounds = [e for e in events if isinstance(e, RoundFinished)]
    assert [r.round for r in rounds] == [0, 1]
    assert rounds[0].usage.round == 0


async def test_astream_parallel_tools_finish_live_in_completion_order():
    responses = [
        _multi_tool_use_response([
            ("tu_1", "slow", {}),
            ("tu_2", "fast", {}),
        ]),
        _text_response("done"),
    ]
    tools = [
        _async_sleep_tool("slow", 0.08, "A"),
        _async_sleep_tool("fast", 0.01, "B"),
    ]
    agent, provider = _agent(responses, tools=tools)

    events = [e async for e in agent.astream("Go")]

    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    # Starts in call order; finishes in completion order (fast first).
    assert [e.tool_call_id for e in started] == ["tu_1", "tu_2"]
    assert [e.tool_call_id for e in finished] == ["tu_2", "tu_1"]
    # The model still receives results in call order.
    results = _tool_results_of(provider, 1)
    assert [r.tool_call_id for r in results] == ["tu_1", "tu_2"]
    assert [r.content for r in results] == ["A", "B"]


async def test_astream_exception_escaping_call_becomes_error_event(monkeypatch):
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])
    recorder = _Recorder()
    agent.with_callbacks([recorder])

    async def boom(self: Agent, tc: Any) -> Any:
        msg = "kaput"
        raise RuntimeError(msg)

    monkeypatch.setattr(Agent, "_aexecute_call", boom)
    events = [e async for e in agent.astream("Go")]

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.is_error is True
    assert "RuntimeError: kaput" in finished.result
    assert ("tool_error", "echo", finished.result) in recorder.events


def test_final_round_tool_request_runs_no_tools():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "a"}),
        _tool_use_response("tu_2", "echo", {"x": "b"}),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()], max_rounds=2)

    events = list(agent.stream("Go"))

    started = [e for e in events if isinstance(e, ToolStarted)]
    assert [e.tool_call_id for e in started] == ["tu_1"]
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "max_rounds"


# ---------------------------------------------------------------------------
# Text projections
# ---------------------------------------------------------------------------


def test_chat_is_text_projection_of_stream():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}, text_before="Working. "),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])
    deltas = [
        e.text for e in agent.stream("Go") if isinstance(e, TextDelta)
    ]

    agent2, _ = _agent(responses, tools=[_echo_tool()])
    assert list(agent2.chat("Go")) == deltas


async def test_achat_is_text_projection_of_astream():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}, text_before="Working. "),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])
    deltas = [
        e.text async for e in agent.astream("Go") if isinstance(e, TextDelta)
    ]

    agent2, _ = _agent(responses, tools=[_echo_tool()])
    assert [c async for c in agent2.achat("Go")] == deltas


# ---------------------------------------------------------------------------
# try/finally bookkeeping
# ---------------------------------------------------------------------------


def test_abandoned_stream_still_persists_diagnostics():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])

    stream = agent.stream("Go")
    next(e for e in stream if isinstance(e, RoundFinished))
    stream.close()

    assert agent.last_turn is not None
    assert len(agent.last_turn.rounds) == 1


async def test_abandoned_astream_still_persists_diagnostics():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "hi"}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])

    stream = agent.astream("Go")
    async for event in stream:
        if isinstance(event, RoundFinished):
            break
    await stream.aclose()

    assert agent.last_turn is not None
    assert len(agent.last_turn.rounds) == 1


def test_provider_exception_still_persists_diagnostics():
    class _ExplodingProvider(FakeLLMProvider):
        def stream(self, *args: Any, **kwargs: Any):
            msg = "provider down"
            raise ConnectionError(msg)
            yield  # pragma: no cover

    provider = _ExplodingProvider([])
    agent = Agent(llm=provider, tokenizer=_Tok())

    with pytest.raises(ConnectionError):
        list(agent.stream("Go"))
    assert agent.last_turn is not None
    assert agent.last_turn.rounds == ()


# ---------------------------------------------------------------------------
# Compaction events
# ---------------------------------------------------------------------------


def test_compaction_events_in_stream():
    long_text = "word " * 40
    responses = [
        _tool_use_response("tu_1", "echo", {"x": long_text}),
        _tool_use_response("tu_2", "echo", {"x": long_text}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])
    agent.with_compaction(
        trigger_tokens=30, keep_last=2, compact_fn=lambda transcript: "SUMMARY",
    )

    events = list(agent.stream("Go"))

    started = [e for e in events if isinstance(e, CompactionStarted)]
    finished = [e for e in events if isinstance(e, CompactionFinished)]
    assert len(started) == 1
    assert len(finished) == 1
    assert finished[0].tokens_before > finished[0].tokens_after
