"""Inline human-in-the-loop approval seam (Agent.with_approval).

Triggered by AgentTool(requires_approval=True) or a pre-hook answering
"ask". The tool call pauses until the callback decides; deny becomes an
is_error tool result carrying the reason; no callback = fail closed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anchor.agent import (
    Agent,
    AgentTool,
    ApprovalDecision,
    ApprovalRequest,
    HookResult,
    tool,
)
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import (
    _multi_tool_use_response,
    _tool_results_of,
)


def _guarded_echo() -> AgentTool:
    @tool(requires_approval=True)
    def echo(x: str) -> str:
        """Echo the input."""
        return f"echo:{x}"

    return echo


def _agent(
    responses: list[list[Any]],
    *,
    tools: list[AgentTool] | None = None,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    if tools:
        agent.with_tools(tools)
    return agent, provider


def _turn_responses() -> list[list[Any]]:
    return [
        _tool_use_response("tu_1", "echo", {"x": "hi"}),
        _text_response("done"),
    ]


# ---------------------------------------------------------------------------
# requires_approval triggers
# ---------------------------------------------------------------------------


def test_approved_call_executes():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        seen.append(request)
        return ApprovalDecision(approved=True)

    agent.with_approval(approve)
    assert "".join(agent.chat("Go")) == "done"

    (request,) = seen
    assert request.tool_call_id == "tu_1"
    assert request.name == "echo"
    assert request.tool_input == {"x": "hi"}
    (result,) = _tool_results_of(provider, 1)
    assert result.content == "echo:hi"
    assert result.is_error is False


def test_denied_call_reason_reaches_model():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_approval(
        lambda request: ApprovalDecision(
            approved=False, reason="use the search tool instead",
        ),
    )

    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "denied" in result.content
    assert "use the search tool instead" in result.content


def test_approval_can_rewrite_input():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_approval(
        lambda request: ApprovalDecision(
            approved=True, updated_input={"x": "redacted"},
        ),
    )

    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.content == "echo:redacted"


def test_no_callback_fails_closed():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])

    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "no approval callback" in result.content


def test_raising_callback_fails_closed():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])

    def broken(request: ApprovalRequest) -> ApprovalDecision:
        msg = "approver crashed"
        raise RuntimeError(msg)

    agent.with_approval(broken)
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "RuntimeError" in result.content


# ---------------------------------------------------------------------------
# "ask" from a pre-hook
# ---------------------------------------------------------------------------


def test_pre_hook_ask_routes_to_callback():
    plain = tool(lambda x: f"echo:{x}", name="echo", description="echoes")
    agent, provider = _agent(_turn_responses(), tools=[plain])
    agent.with_hooks(
        pre_tool_use=[lambda name, tool_input: HookResult(decision="ask")],
    )
    decisions: list[str] = []

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        decisions.append(request.name)
        return ApprovalDecision(approved=True)

    agent.with_approval(approve)
    list(agent.chat("Go"))

    assert decisions == ["echo"]
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is False


def test_later_hook_deny_wins_over_ask():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_hooks(
        pre_tool_use=[
            lambda name, tool_input: HookResult(decision="ask"),
            lambda name, tool_input: HookResult(decision="deny", reason="nope"),
        ],
    )
    calls: list[str] = []
    agent.with_approval(
        lambda request: calls.append(request.name)
        or ApprovalDecision(approved=True),
    )

    list(agent.chat("Go"))

    assert calls == []  # deny short-circuits; callback never runs
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "nope" in result.content


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


async def test_async_callback_pauses_the_turn():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        await gate  # simulates a human taking their time
        return ApprovalDecision(approved=True)

    agent.with_approval(approve)

    async def _release() -> None:
        await asyncio.sleep(0.02)
        gate.set_result(None)

    release = asyncio.ensure_future(_release())
    chunks = [c async for c in agent.achat("Go")]
    await release

    assert "".join(chunks) == "done"
    (result,) = _tool_results_of(provider, 1)
    assert result.content == "echo:hi"


async def test_parallel_tools_with_distinct_decisions():
    allow = tool(
        lambda x: f"a:{x}", name="alpha", description="a",
        requires_approval=True,
    )
    deny = tool(
        lambda x: f"b:{x}", name="beta", description="b",
        requires_approval=True,
    )
    agent, provider = _agent(
        [
            _multi_tool_use_response([
                ("tu_1", "alpha", {"x": "1"}),
                ("tu_2", "beta", {"x": "2"}),
            ]),
            _text_response("done"),
        ],
        tools=[allow, deny],
    )

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            approved=request.name == "alpha", reason="beta is blocked",
        )

    agent.with_approval(approve)
    [c async for c in agent.achat("Go")]

    first, second = _tool_results_of(provider, 1)
    assert first.tool_call_id == "tu_1"
    assert first.is_error is False
    assert first.content == "a:1"
    assert second.tool_call_id == "tu_2"
    assert second.is_error is True
    assert "beta is blocked" in second.content


def test_mistyped_callback_return_fails_closed():
    agent, provider = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_approval(lambda request: True)  # type: ignore[arg-type,return-value]

    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "expected ApprovalDecision" in result.content


def test_non_coroutine_awaitable_on_sync_loop_raises_typeerror():
    class _Awaitable:
        def __await__(self):  # pragma: no cover - never driven
            yield

    agent, _ = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_approval(lambda request: _Awaitable())  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="astream"):
        list(agent.chat("Go"))


def test_async_callback_on_sync_loop_raises():
    agent, _ = _agent(_turn_responses(), tools=[_guarded_echo()])

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True)

    agent.with_approval(approve)
    with pytest.raises(TypeError, match="astream"):
        list(agent.chat("Go"))


def test_non_coroutine_awaitable_is_released_on_sync_loop():
    """A Future/Task gets cancelled, not abandoned pending."""
    released: list[bool] = []

    class _Future:
        def __await__(self):  # pragma: no cover - never driven
            yield

        def cancel(self) -> None:
            released.append(True)

    agent, _ = _agent(_turn_responses(), tools=[_guarded_echo()])
    agent.with_approval(lambda request: _Future())  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="astream"):
        list(agent.chat("Go"))
    assert released == [True]
