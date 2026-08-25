"""Phase 4 loop tests: tool hygiene, hooks, accounting, deferred tools."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from anchor.agent.agent import Agent
from anchor.agent.hooks import HookResult
from anchor.agent.models import AgentTool
from anchor.llm.models import (
    Role,
    StopReason,
    StreamChunk,
    ToolCallDelta,
    Usage,
)
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)


def _multi_tool_use_response(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> list[StreamChunk]:
    """Stream chunks for a round with multiple parallel tool calls."""
    chunks: list[StreamChunk] = []
    for index, (tool_id, name, arguments) in enumerate(calls):
        chunks.append(StreamChunk(
            tool_call_delta=ToolCallDelta(index=index, id=tool_id, name=name),
        ))
        chunks.append(StreamChunk(
            tool_call_delta=ToolCallDelta(
                index=index, arguments_fragment=json.dumps(arguments),
            ),
        ))
    chunks.append(StreamChunk(stop_reason=StopReason.TOOL_USE))
    return chunks


def _agent(
    responses: list[list[StreamChunk]],
    *,
    tools: list[AgentTool] | None = None,
    max_rounds: int = 10,
    **kwargs: Any,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, max_rounds=max_rounds, tokenizer=_Tok(), **kwargs)
    agent.with_system_prompt("You are helpful.")
    if tools:
        agent.with_tools(tools)
    return agent, provider


def _echo_tool(name: str = "echo") -> AgentTool:
    return AgentTool(
        name=name,
        description="echoes input",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=lambda x: f"echo:{x}",
    )


def _failing_tool() -> AgentTool:
    def fail(x: str) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    return AgentTool(
        name="fail",
        description="always fails",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=fail,
    )


def _tool_results_of(provider: FakeLLMProvider, call_index: int) -> list[Any]:
    """The ToolResult objects fed back to the model on a given LLM call."""
    return [
        m.tool_result
        for m in provider.seen_messages[call_index]
        if m.role == Role.TOOL and m.tool_result is not None
    ]


# ---------------------------------------------------------------------------
# Tool error forwarding
# ---------------------------------------------------------------------------


def test_tool_error_has_diagnostic_and_is_error_flag():
    responses = [
        _tool_use_response("tu_1", "fail", {"x": "go"}),
        _text_response("Recovered."),
    ]
    agent, provider = _agent(responses, tools=[_failing_tool()])
    assert "".join(agent.chat("Go")) == "Recovered."

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "RuntimeError: boom" in result.content
    assert "fail" in result.content


def test_unknown_tool_marked_error():
    responses = [
        _tool_use_response("tu_1", "nonexistent", {"q": "x"}),
        _text_response("OK"),
    ]
    agent, provider = _agent(responses, tools=[_echo_tool()])
    list(agent.chat("Test"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "Unknown tool: nonexistent" in result.content


# ---------------------------------------------------------------------------
# Parallel execution + timeout (async path)
# ---------------------------------------------------------------------------


def _async_sleep_tool(
    name: str, delay: float, reply: str, *, timeout: float | None = None,
) -> AgentTool:
    tool = AgentTool(
        name=name,
        description="sleeps then replies",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: reply,
        timeout=timeout,
    )

    async def acall(_name: str, _input: dict[str, Any]) -> str:
        await asyncio.sleep(delay)
        return reply

    object.__setattr__(tool, "_anchor_async_caller", acall)
    return tool


async def test_parallel_tool_calls_ordered_and_concurrent():
    responses = [
        _multi_tool_use_response([
            ("tu_1", "slow_a", {}),
            ("tu_2", "slow_b", {}),
        ]),
        _text_response("done"),
    ]
    tools = [
        _async_sleep_tool("slow_a", 0.05, "A"),
        _async_sleep_tool("slow_b", 0.05, "B"),
    ]
    agent, provider = _agent(responses, tools=tools)

    start = time.perf_counter()
    chunks = [c async for c in agent.achat("Go")]
    elapsed = time.perf_counter() - start

    assert "".join(chunks) == "done"
    results = _tool_results_of(provider, 1)
    assert [r.tool_call_id for r in results] == ["tu_1", "tu_2"]
    assert [r.content for r in results] == ["A", "B"]
    # Two 50ms sleeps run concurrently, not sequentially.
    assert elapsed < 0.09


async def test_async_tool_timeout_becomes_error_result():
    slow = _async_sleep_tool("slow", 0.5, "never", timeout=0.01)
    responses = [
        _tool_use_response("tu_1", "slow", {}),
        _text_response("moved on"),
    ]
    agent, provider = _agent(responses, tools=[slow])
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "moved on"
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "timed out after 0.01s" in result.content


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_pre_hook_deny_reason_reaches_model():
    executed = []

    def spy(x: str) -> str:
        executed.append(x)
        return "ran"

    tool = _echo_tool()
    tool = tool.model_copy(update={"fn": spy})
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "go"}),
        _text_response("understood"),
    ]
    agent, provider = _agent(responses, tools=[tool])
    agent.with_hooks(pre_tool_use=[
        lambda name, _input: HookResult(decision="deny", reason="not allowed here"),
    ])
    list(agent.chat("Go"))

    assert executed == []
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "denied: not allowed here" in result.content


def test_pre_hook_updates_input():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "original"}),
        _text_response("ok"),
    ]
    agent, provider = _agent(responses, tools=[_echo_tool()])
    agent.with_hooks(pre_tool_use=[
        lambda name, _input: HookResult(updated_input={"x": "rewritten"}),
    ])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.content == "echo:rewritten"


def test_pre_hook_exception_fails_closed():
    executed = []

    def spy(x: str) -> str:
        executed.append(x)
        return "ran"

    tool = _echo_tool().model_copy(update={"fn": spy})

    def broken_hook(name: str, tool_input: dict[str, Any]) -> HookResult:
        msg = "hook bug"
        raise ValueError(msg)

    responses = [
        _tool_use_response("tu_1", "echo", {"x": "go"}),
        _text_response("ok"),
    ]
    agent, provider = _agent(responses, tools=[tool])
    agent.with_hooks(pre_tool_use=[broken_hook])
    list(agent.chat("Go"))

    assert executed == []
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "pre-tool hook raised ValueError" in result.content


def test_post_hook_replaces_output():
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "go"}),
        _text_response("ok"),
    ]
    agent, provider = _agent(responses, tools=[_echo_tool()])
    agent.with_hooks(post_tool_use=[
        lambda name, _input, output: HookResult(updated_output=f"[redacted] {output}"),
    ])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.content == "[redacted] echo:go"
    assert result.is_error is False


def test_callbacks_fire_and_raising_callback_is_swallowed():
    events: list[tuple[str, Any]] = []

    class Recorder:
        def on_round_start(self, round_index: int) -> None:
            events.append(("round_start", round_index))

        def on_round_end(self, round_index: int) -> None:
            events.append(("round_end", round_index))

        def on_tool_start(self, name: str, tool_input: dict[str, Any]) -> None:
            events.append(("tool_start", name))

        def on_tool_end(
            self, name: str, tool_input: dict[str, Any], result: str,
        ) -> None:
            events.append(("tool_end", name))

    class Exploder:
        def on_tool_start(self, name: str, tool_input: dict[str, Any]) -> None:
            msg = "observer bug"
            raise RuntimeError(msg)

    responses = [
        _tool_use_response("tu_1", "echo", {"x": "go"}),
        _text_response("done"),
    ]
    agent, _ = _agent(responses, tools=[_echo_tool()])
    agent.with_callbacks([Recorder(), Exploder()])
    assert "".join(agent.chat("Go")) == "done"

    assert ("round_start", 0) in events
    assert ("tool_start", "echo") in events
    assert ("tool_end", "echo") in events
    assert ("round_end", 1) in events


def test_tool_error_callback_fires():
    events: list[str] = []

    class ErrorRecorder:
        def on_tool_error(
            self, name: str, tool_input: dict[str, Any], error: str,
        ) -> None:
            events.append(error)

    responses = [
        _tool_use_response("tu_1", "fail", {"x": "go"}),
        _text_response("ok"),
    ]
    agent, _ = _agent(responses, tools=[_failing_tool()])
    agent.with_callbacks([ErrorRecorder()])
    list(agent.chat("Go"))

    assert len(events) == 1
    assert "RuntimeError: boom" in events[0]


# ---------------------------------------------------------------------------
# Round accounting + round-limit signalling
# ---------------------------------------------------------------------------


def test_per_round_accounting_visible():
    tool_round = _tool_use_response("tu_1", "echo", {"x": "hello world"})
    # Provider-reported usage on the final round.
    final = [
        StreamChunk(content="All done."),
        StreamChunk(usage=Usage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )),
        StreamChunk(stop_reason=StopReason.STOP),
    ]
    agent, _ = _agent([tool_round, final], tools=[_echo_tool()])
    list(agent.chat("Go"))

    turn = agent.last_turn
    assert turn is not None
    assert turn.stopped_by == "stop"
    assert len(turn.rounds) == 2
    first, second = turn.rounds
    assert first.round == 0
    assert first.tool_schema_tokens > 0
    assert first.tool_result_tokens > 0  # "echo:hello world"
    assert second.prompt_tokens == 10
    assert second.completion_tokens == 5
    assert turn.total_prompt_tokens == 10


def test_stopped_by_max_tokens():
    responses = [[
        StreamChunk(content="truncated"),
        StreamChunk(stop_reason=StopReason.MAX_TOKENS),
    ]]
    agent, _ = _agent(responses)
    list(agent.chat("Go"))

    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "max_tokens"


def test_final_round_notice_injected_and_tools_not_executed():
    executed = []
    tool = _echo_tool().model_copy(
        update={"fn": lambda x: executed.append(x) or "ran"},
    )
    responses = [
        _tool_use_response("tu_1", "echo", {"x": "one"}),
        _tool_use_response("tu_2", "echo", {"x": "two"}),
    ]
    agent, provider = _agent(responses, tools=[tool], max_rounds=2)
    list(agent.chat("Go"))

    # Round 2 (the final round) carries the wrap-up notice.
    final_round_messages = provider.seen_messages[1]
    notices = [
        m for m in final_round_messages
        if m.role == Role.USER and m.content and "Final round" in str(m.content)
    ]
    assert len(notices) == 1
    # Only round 1's tool ran; the final round's call was not executed.
    assert executed == ["one"]
    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "max_rounds"


# ---------------------------------------------------------------------------
# Deferred tools + search_tools meta-tool
# ---------------------------------------------------------------------------


def _deferred_tool() -> AgentTool:
    return AgentTool(
        name="obscure_metric",
        description="Computes the obscure metric of a dataset",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=lambda x: f"metric:{x}",
        defer_loading=True,
    )


def test_deferred_tool_hidden_until_searched():
    responses = [
        _tool_use_response("tu_1", "search_tools", {"query": "obscure"}),
        _tool_use_response("tu_2", "obscure_metric", {"x": "data"}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[_echo_tool(), _deferred_tool()])
    assert "".join(agent.chat("Go")) == "done"

    first_round_names = {t.name for t in provider.seen_tools[0] or []}
    assert "echo" in first_round_names
    assert "search_tools" in first_round_names
    assert "obscure_metric" not in first_round_names

    (search_result,) = _tool_results_of(provider, 1)
    assert "obscure_metric" in search_result.content

    second_round_names = {t.name for t in provider.seen_tools[1] or []}
    assert "obscure_metric" in second_round_names

    metric_result = _tool_results_of(provider, 2)[-1]
    assert metric_result.content == "metric:data"


def test_search_tools_no_match_lists_deferred():
    responses = [
        _tool_use_response("tu_1", "search_tools", {"query": "zzz_nothing"}),
        _text_response("ok"),
    ]
    agent, provider = _agent(responses, tools=[_deferred_tool()])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert "No tools matching" in result.content
    assert "obscure_metric" in result.content


def test_input_examples_propagate_to_schema():
    tool = _echo_tool().model_copy(
        update={"input_examples": ({"x": "sample"},)},
    )
    schema = tool.to_tool_schema()
    assert schema.input_examples == ({"x": "sample"},)
