"""Usage limits enforced by the agent loop (Agent.with_usage_limits).

Breach semantics: graceful wrap-up — one final round with the
final-round notice + tool_choice="none", then the turn ends with
stopped_by="usage_limit". No exceptions.
"""

from __future__ import annotations

from typing import Any

from anchor.agent import Agent, AgentTool, UsageLimits
from anchor.agent.events import TurnFinished, UsageLimitReached
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import _echo_tool


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


def _looping_tool_responses(n: int) -> list[list[Any]]:
    """n tool rounds followed by a text answer."""
    rounds: list[list[Any]] = [
        _tool_use_response(f"tu_{i}", "echo", {"x": f"call {i}"})
        for i in range(n)
    ]
    rounds.append(_text_response("done"))
    return rounds


# ---------------------------------------------------------------------------
# total_tokens_limit
# ---------------------------------------------------------------------------


def test_total_tokens_breach_triggers_wrap_up_round():
    agent, provider = _agent(
        _looping_tool_responses(5), tools=[_echo_tool()],
    )
    # _Tok counts whitespace words; round 1 alone crosses this.
    agent.with_usage_limits(UsageLimits(total_tokens_limit=10))

    events = list(agent.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    assert breach.kind == "total_tokens"
    assert breach.used > breach.limit == 10
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    # Round 0 ran its tools; round 1 is the wrap-up: notice injected
    # and tool_choice forced, no tools executed.
    assert len(provider.seen_messages) == 2
    wrap_up_messages = provider.seen_messages[1]
    assert any("Final round" in str(m.content) for m in wrap_up_messages)
    assert provider.seen_kwargs[1]["tool_choice"] == "none"
    assert len(final.diagnostics.rounds) == 2
    assert final.diagnostics.rounds[1].tool_calls == 0


async def test_total_tokens_breach_async_mirror():
    agent, provider = _agent(
        _looping_tool_responses(5), tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=10))

    events = [e async for e in agent.astream("Go")]

    assert any(isinstance(e, UsageLimitReached) for e in events)
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    assert len(provider.seen_messages) == 2


def test_turn_below_limit_unaffected():
    agent, provider = _agent(
        _looping_tool_responses(1), tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=1_000_000))

    events = list(agent.stream("Go"))

    assert not any(isinstance(e, UsageLimitReached) for e in events)
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert final.text == "done"
    # No notice, no forced tool_choice on any request.
    for call_index in range(len(provider.seen_messages)):
        assert not any(
            "Final round" in str(m.content)
            for m in provider.seen_messages[call_index]
        )
        assert "tool_choice" not in provider.seen_kwargs[call_index]


def test_turn_that_ends_on_its_own_is_not_relabeled():
    # The last response crosses the limit, but the model already
    # stopped — the turn was not cut, so stopped_by stays "stop".
    agent, _ = _agent(
        [_text_response("a long answer with many many words here")],
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=2))

    events = list(agent.stream("Go"))

    assert not any(isinstance(e, UsageLimitReached) for e in events)
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"


def test_breach_on_penultimate_round_gets_no_extra_round():
    # max_rounds=2: round 0 breaches, round 1 is already the natural
    # final round — the wrap-up and max_rounds coincide.
    agent, provider = _agent(
        _looping_tool_responses(5), tools=[_echo_tool()], max_rounds=2,
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=10))

    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    assert len(provider.seen_messages) == 2


# ---------------------------------------------------------------------------
# tool_calls_limit
# ---------------------------------------------------------------------------


def test_tool_calls_breach_triggers_wrap_up():
    agent, provider = _agent(
        _looping_tool_responses(5), tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(tool_calls_limit=2))

    events = list(agent.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    assert breach.kind == "tool_calls"
    assert breach.used == 3
    assert breach.limit == 2
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    assert final.diagnostics.total_tool_calls == 3
    # Rounds 0-2 executed one call each; round 3 is the wrap-up.
    assert len(provider.seen_messages) == 4


def test_wrap_up_text_reaches_chat_projection():
    agent, _ = _agent(
        [
            _tool_use_response("tu_0", "echo", {"x": "call"}),
            _text_response("wrapped up"),
        ],
        tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(tool_calls_limit=0))

    # tool_calls_limit=0 still lets round 0's batch run (post-hoc
    # check), then wraps up; the projection sees the wrap-up text.
    text = "".join(agent.chat("Go"))
    assert text == "wrapped up"
    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "usage_limit"


# ---------------------------------------------------------------------------
# Estimation fallback (providers without stream usage)
# ---------------------------------------------------------------------------


def test_estimated_usage_without_provider_numbers():
    agent, _ = _agent(
        _looping_tool_responses(1), tools=[_echo_tool()],
    )

    list(agent.stream("Go"))

    turn = agent.last_turn
    assert turn is not None
    first = turn.rounds[0]
    # FakeLLMProvider reports no usage: both sides are estimated.
    assert first.prompt_tokens > 0
    assert first.completion_tokens > 0
    assert turn.total_tokens > 0


def test_estimated_prompt_includes_system_and_schemas():
    agent, _ = _agent([_text_response("hi")], tools=[_echo_tool()])

    list(agent.stream("Go"))

    turn = agent.last_turn
    assert turn is not None
    (only,) = turn.rounds
    # System prompt + user message + schema estimate.
    assert only.prompt_tokens >= only.tool_schema_tokens
    assert only.prompt_tokens > 3
