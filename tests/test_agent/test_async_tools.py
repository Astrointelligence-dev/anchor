"""``async def`` functions under ``@tool``/``AgentTool`` are awaited, not stringified."""

from __future__ import annotations

import asyncio

import pytest

from anchor.agent.tool_decorator import tool
from tests.test_agent.test_phase4_loop import (
    _agent,
    _text_response,
    _tool_results_of,
    _tool_use_response,
)


def _async_echo():
    @tool
    async def echo(text: str) -> str:
        """Echo asynchronously."""
        await asyncio.sleep(0)
        return f"Echo: {text}"

    return echo


async def test_async_tool_fn_is_awaited_on_the_async_path():
    responses = [_tool_use_response("tu_1", "echo", {"text": "hi"}), _text_response("done")]
    agent, provider = _agent(responses, tools=[_async_echo()])
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "done"
    (result,) = _tool_results_of(provider, 1)
    assert not result.is_error
    assert result.content == "Echo: hi"


def test_async_tool_fn_runs_from_sync_chat():
    responses = [_tool_use_response("tu_1", "echo", {"text": "hi"}), _text_response("done")]
    agent, provider = _agent(responses, tools=[_async_echo()])
    chunks = list(agent.chat("Go"))

    assert "".join(chunks) == "done"
    (result,) = _tool_results_of(provider, 1)
    assert not result.is_error
    assert result.content == "Echo: hi"


async def test_async_tool_fn_honors_timeout():
    @tool
    async def slow() -> str:
        """Sleeps past the timeout."""
        await asyncio.sleep(0.5)
        return "never"

    responses = [_tool_use_response("tu_1", "slow", {}), _text_response("moved on")]
    agent, provider = _agent(responses, tools=[slow.model_copy(update={"timeout": 0.01})])
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "moved on"
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert "timed out after 0.01s" in result.content


def test_sync_chat_with_async_tool_keeps_the_thread_loop():
    """asyncio.run would reset the thread's current loop; the private loop must not."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        responses = [_tool_use_response("tu_1", "echo", {"text": "hi"}), _text_response("done")]
        agent, _ = _agent(responses, tools=[_async_echo()])
        assert "".join(agent.chat("Go")) == "done"
        assert asyncio.get_event_loop() is loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def test_async_tool_under_a_running_loop_raises_to_the_caller():
    """The developer's mistake surfaces as TypeError, like an async approval under stream()."""
    responses = [_tool_use_response("tu_1", "echo", {"text": "hi"}), _text_response("done")]
    agent, _ = _agent(responses, tools=[_async_echo()])
    with pytest.raises(TypeError, match="achat"):
        list(agent.chat("Go"))


async def test_refused_future_is_cancelled():
    from anchor.agent.agent import _run_awaitable_blocking

    fut = asyncio.get_running_loop().create_future()
    with pytest.raises(TypeError):
        _run_awaitable_blocking(fut, "t")
    assert fut.cancelled()
