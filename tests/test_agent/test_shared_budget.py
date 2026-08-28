"""Shared usage pool across a multi-agent run (v0.2 #1).

The agent that starts a turn with ``UsageLimits`` creates a run-wide
pool; every subagent spawned during that turn debits the same pool, so
a child's spend counts against the orchestrator's budget. Breach keeps
the graceful-wrap-up contract at every level: the child wraps up and
returns a partial result to the parent; the parent wraps up on its own
next check. No exceptions.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest

import anchor.agent.agent as agent_mod
from anchor.agent import Agent, AgentTool, UsageLimits
from anchor.agent.events import RoundFinished, TurnFinished, UsageLimitReached
from anchor.agent.subagent import SubagentDefinition
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


def _burning_child(
    rounds: int, words: int = 40,
) -> tuple[Agent, FakeLLMProvider]:
    """A child that spends *rounds* tool rounds with fat payloads."""
    responses: list[list[Any]] = [
        _tool_use_response(f"sub_tu_{i}", "echo", {"x": "word " * words})
        for i in range(rounds)
    ]
    responses.append(_text_response("child done"))
    provider = FakeLLMProvider(responses)
    sub = Agent(llm=provider, tokenizer=_Tok())
    sub.with_system_prompt("You research.")
    sub.with_tools([_echo_tool()])
    return sub, provider


# ---------------------------------------------------------------------------
# The leak this front fixes: child spend must trip the parent's limit
# ---------------------------------------------------------------------------


def test_child_spend_trips_parent_limit():
    # The parent's own spend stays far below the limit; only the
    # child's spend can cross it. Before the shared pool, this turn
    # ran to completion with the limit never firing.
    sub, sub_provider = _burning_child(6)
    orch, _ = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig deep."}),
        _text_response("wrapped"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=400))

    events = list(orch.stream("Go"))

    breaches = [e for e in events if isinstance(e, UsageLimitReached)]
    assert breaches, "child spend never tripped the parent's limit"
    assert breaches[0].kind == "total_tokens"
    assert breaches[0].scope == "run"
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    # The child was cut mid-run (never reached its 7th canned response)
    # and wrapped up gracefully instead of raising.
    assert sub.last_turn is not None
    assert sub.last_turn.stopped_by == "usage_limit"
    assert len(sub_provider.seen_messages) < 7


async def test_child_spend_trips_parent_limit_async_mirror():
    sub, sub_provider = _burning_child(6)
    orch, _ = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig deep."}),
        _text_response("wrapped"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=400))

    events = [e async for e in orch.astream("Go")]

    breaches = [e for e in events if isinstance(e, UsageLimitReached)]
    assert breaches, "child spend never tripped the parent's limit"
    assert breaches[0].scope == "run"
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    assert sub.last_turn is not None
    assert sub.last_turn.stopped_by == "usage_limit"
    assert len(sub_provider.seen_messages) < 7


def test_tool_calls_shared_across_the_run():
    sub, _ = _burning_child(6, words=1)
    orch, _ = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Count."}),
        _text_response("wrapped"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])
    orch.with_usage_limits(UsageLimits(tool_calls_limit=3))

    events = list(orch.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    assert breach.kind == "tool_calls"
    assert breach.scope == "run"
    # The child tripped the pool on its own 4th call — the parent had
    # not even closed its round yet.
    assert breach.used == 4
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    # Parent's one subagent call + the child's four.
    assert final.diagnostics.run_total_tool_calls == 5


def test_second_child_starts_on_exhausted_pool_gets_one_wrap_up_round():
    # Sync tools run sequentially: child A drains the pool, child B
    # must go straight to its wrap-up round instead of burning more.
    sub_a, _ = _burning_child(6)
    sub_b, b_provider = _burning_child(6)
    orch, _ = _agent([
        _multi_tool_use_response([
            ("tu_a", "researcher_a", {"task": "Dig."}),
            ("tu_b", "researcher_b", {"task": "Dig too."}),
        ]),
        _text_response("wrapped"),
    ])
    orch.with_tools([
        sub_a.as_tool("researcher_a", "Researches"),
        sub_b.as_tool("researcher_b", "Researches"),
    ])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=300))

    list(orch.stream("Go"))

    assert sub_a.last_turn is not None
    assert sub_a.last_turn.stopped_by == "usage_limit"
    assert sub_b.last_turn is not None
    assert sub_b.last_turn.stopped_by == "usage_limit"
    # B made exactly one model call, already under the wrap-up notice.
    assert len(b_provider.seen_messages) == 1
    assert any(
        "Final round" in str(m.content) for m in b_provider.seen_messages[0]
    )


def test_partial_child_result_is_marked_for_the_parent():
    sub, _ = _burning_child(6)
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig."}),
        _text_response("wrapped"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=400))

    list(orch.stream("Go"))

    (result,) = _tool_results_of(orch_provider, 1)
    assert result.is_error is False
    assert "partial result" in result.content
    assert "usage limit" in result.content


def test_output_model_retry_suppressed_on_exhausted_pool():
    from pydantic import BaseModel

    class Finding(BaseModel):
        answer: str

    # The child's only reply is fat invalid prose that drains the pool;
    # a schema retry would just burn the wrap-up round again.
    provider = FakeLLMProvider([_text_response("not json " * 100)])
    sub = Agent(llm=provider, tokenizer=_Tok())
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig."}),
        _text_response("wrapped"),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=100))

    list(orch.stream("Go"))

    # One child turn only — no retry into the empty pool.
    assert len(provider.seen_messages) == 1
    (result,) = _tool_results_of(orch_provider, 1)
    assert result.is_error is True
    assert "usage limit reached" in result.content


# ---------------------------------------------------------------------------
# Narrowing: a child's own limits apply on top of the pool
# ---------------------------------------------------------------------------


def test_subagent_definition_limits_narrow_within_the_run():
    child_provider = FakeLLMProvider([
        _tool_use_response(f"sub_tu_{i}", "echo", {"x": "hi"})
        for i in range(4)
    ] + [_text_response("child done")])
    orch, _ = _agent([
        _tool_use_response("tu_1", "task", {
            "agent_name": "researcher", "task": "Dig.",
        }),
        _text_response("all good"),
    ])
    orch.with_subagents([
        SubagentDefinition(
            name="researcher",
            description="Researches",
            tools=(_echo_tool(),),
            usage_limits=UsageLimits(tool_calls_limit=1),
        ),
    ])
    # Wire the child to the canned provider (with_subagents builds it
    # from the orchestrator's provider otherwise).
    _, sub = orch._subagents["researcher"]
    sub._llm = child_provider
    orch.with_usage_limits(UsageLimits(total_tokens_limit=1_000_000))

    events = list(orch.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    # The child's own narrower limit tripped — not the run pool.
    assert breach.scope == "turn"
    assert breach.kind == "tool_calls"
    assert breach.parent_tool_call_id == "tu_1"
    assert sub.last_turn is not None
    assert sub.last_turn.stopped_by == "usage_limit"
    # The parent's own turn was never cut.
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"


# ---------------------------------------------------------------------------
# Aggregated diagnostics: TurnDiagnostics.children
# ---------------------------------------------------------------------------


def test_children_diagnostics_visible_on_parent_turn():
    sub, _ = _burning_child(2)
    orch, _ = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig."}),
        _text_response("done"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])

    events = list(orch.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    (child,) = final.diagnostics.children
    assert child.tool_call_id == "tu_1"
    assert child.name == "researcher"
    assert child.diagnostics.rounds
    assert child.diagnostics.stopped_by == "stop"
    assert (
        final.diagnostics.run_total_tokens
        > final.diagnostics.total_tokens
    )
    assert orch.last_turn is not None
    assert orch.last_turn.children == final.diagnostics.children


async def test_children_diagnostics_async_task_tool_resolves_agent_name():
    orch, _ = _agent([
        _tool_use_response("tu_1", "task", {
            "agent_name": "researcher", "task": "Dig.",
        }),
        _text_response("done"),
    ])
    orch.with_subagents([
        SubagentDefinition(name="researcher", description="Researches"),
    ])
    _, sub = orch._subagents["researcher"]
    sub._llm = FakeLLMProvider([_text_response("42")])

    events = [e async for e in orch.astream("Go")]

    final = events[-1]
    assert isinstance(final, TurnFinished)
    (child,) = final.diagnostics.children
    assert child.name == "researcher"
    assert child.tool_call_id == "tu_1"


# ---------------------------------------------------------------------------
# cost_limit (USD via genai-prices)
# ---------------------------------------------------------------------------


def test_cost_limit_trips_the_pool(monkeypatch):
    monkeypatch.setattr(agent_mod, "_estimate_cost", lambda *a: 1.0)
    responses: list[list[Any]] = [
        _tool_use_response(f"tu_{i}", "echo", {"x": "call"}) for i in range(5)
    ]
    responses.append(_text_response("wrapped"))
    agent, _ = _agent(responses, tools=[_echo_tool()])
    agent.with_usage_limits(UsageLimits(cost_limit=2.5))

    events = list(agent.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    assert breach.kind == "cost"
    assert breach.scope == "run"
    assert breach.used == 3.0
    assert breach.limit == 2.5
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    # Every round (including the wrap-up) was priced.
    assert final.diagnostics.rounds[0].cost_usd == 1.0
    assert final.diagnostics.total_cost_usd == 4.0
    assert final.diagnostics.run_total_cost_usd == 4.0


def test_cost_limit_without_genai_prices_warns_at_configuration(
    monkeypatch, caplog,
):
    monkeypatch.setitem(sys.modules, "genai_prices", None)
    agent, _ = _agent([_text_response("hi")])

    with caplog.at_level(logging.WARNING, logger="anchor.agent.agent"):
        agent.with_usage_limits(UsageLimits(cost_limit=1.0))

    # Soft gate: MODEL_PRICING and provider-reported costs still price
    # rounds, so configuration warns instead of raising.
    assert any("genai-prices" in r.getMessage() for r in caplog.records)


def test_unknown_model_warns_once_and_debits_zero_cost(
    monkeypatch, caplog,
):
    monkeypatch.setattr(agent_mod, "_PRICE_WARNED", set())
    agent, _ = _agent(
        [_text_response("hi"), _text_response("again")],
    )
    agent.with_usage_limits(UsageLimits(cost_limit=0.01))

    with caplog.at_level(logging.WARNING, logger="anchor.agent.agent"):
        list(agent.stream("Go"))
        list(agent.stream("Go again"))

    # genai-prices has no price for FakeLLMProvider's "test-model":
    # tokens still count, cost debits zero, and the limit never trips.
    warnings = [
        r for r in caplog.records if "no price data" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "stop"
    assert agent.last_turn.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Pool lifecycle: no ambient leakage (review round, judge finding 1)
# ---------------------------------------------------------------------------


def test_abandoned_stream_does_not_poison_later_agents():
    # Abandon a breached stream with the reference held: the pool must
    # not leak into the consumer's context and cut unrelated agents.
    burner, _ = _agent(
        [
            _tool_use_response("tu_0", "echo", {"x": "word " * 60}),
            _tool_use_response("tu_1", "echo", {"x": "word " * 60}),
            _text_response("never reached"),
        ],
        tools=[_echo_tool()],
    )
    burner.with_usage_limits(UsageLimits(total_tokens_limit=50))
    it = burner.stream("Go")
    for event in it:
        if isinstance(event, UsageLimitReached):
            break  # abandon mid-turn, generator alive, finally not run

    other, _ = _agent(
        [
            _tool_use_response("o_0", "echo", {"x": "hi"}),
            _text_response("done"),
        ],
        tools=[_echo_tool()],
    )
    events = list(other.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert len(final.diagnostics.rounds) == 2
    it.close()


def test_interleaved_streams_do_not_share_pools():
    # A second agent driven while the first's stream is mid-flight must
    # not debit the first agent's pool.
    limited, _ = _agent(
        [
            _tool_use_response("tu_0", "echo", {"x": "small"}),
            _text_response("done a"),
        ],
        tools=[_echo_tool()],
    )
    limited.with_usage_limits(UsageLimits(total_tokens_limit=150))
    it = limited.stream("Go")
    next(e for e in it if isinstance(e, RoundFinished))  # past round 0

    fat, _ = _agent([_text_response("word " * 200)])
    list(fat.stream("Say a lot"))

    rest = list(it)
    final = rest[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert not any(isinstance(e, UsageLimitReached) for e in rest)


def test_close_from_foreign_context_still_persists_diagnostics():
    # An abandoned generator may be finalized under a different context
    # (asyncio's asyncgen finalizer does this). The finally must still
    # persist diagnostics — nothing in it may raise cross-context.
    import contextvars

    agent, _ = _agent(
        [
            _tool_use_response("tu_0", "echo", {"x": "hi"}),
            _text_response("done"),
        ],
        tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=1_000_000))
    it = agent.stream("Go")
    next(e for e in it if isinstance(e, RoundFinished))

    contextvars.copy_context().run(it.close)

    assert agent.last_turn is not None
    assert agent.last_turn.rounds


async def test_aclose_from_fresh_task_still_persists_diagnostics():
    import asyncio

    agent, _ = _agent(
        [
            _tool_use_response("tu_0", "echo", {"x": "hi"}),
            _text_response("done"),
        ],
        tools=[_echo_tool()],
    )
    agent.with_usage_limits(UsageLimits(total_tokens_limit=1_000_000))
    stream = agent.astream("Go")
    async for event in stream:
        if isinstance(event, RoundFinished):
            break

    # A task gets its own context copy — the asyncgen-finalizer shape.
    await asyncio.create_task(stream.aclose())

    assert agent.last_turn is not None
    assert agent.last_turn.rounds


def test_prompted_run_budget_spans_retries():
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    provider = FakeLLMProvider([_text_response("junk " * 200)])
    agent = Agent(llm=provider, tokenizer=_Tok())
    agent.with_output_model(Out, mode="prompted", max_output_retries=2)
    agent.with_usage_limits(UsageLimits(total_tokens_limit=100))

    with pytest.raises(ValueError, match="usage limit reached"):
        agent.run("Go")

    # The first turn drained the budget: no retry turns were spawned.
    assert len(provider.seen_messages) == 1


def test_nesting_added_after_as_tool_is_caught_at_call_time():
    worker, worker_provider = _burning_child(2)
    wrapped = worker.as_tool("worker", "Does work")
    # Late registration: this bypasses the registration-time guards.
    worker.with_subagents([
        SubagentDefinition(name="grandchild", description="Nested"),
    ])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "worker", {"task": "Dig."}),
        _text_response("ok"),
    ])
    orch.with_tools([wrapped])

    list(orch.stream("Go"))

    (result,) = _tool_results_of(orch_provider, 1)
    assert result.is_error is True
    assert "nesting" in result.content
    # The 3-level tree never executed a single model round.
    assert len(worker_provider.seen_messages) == 0


def test_failed_child_still_appears_in_children():
    from anchor.llm.models import StopReason, StreamChunk

    class _DiesOnSecondCall(FakeLLMProvider):
        def stream(self, messages, **kwargs):
            if self._call_index >= 1:
                self.seen_messages.append(list(messages))
                msg = "provider blew up"
                raise RuntimeError(msg)
            return super().stream(messages, **kwargs)

    sub_provider = _DiesOnSecondCall([
        _tool_use_response("s_0", "echo", {"x": "hi"}),
        [StreamChunk(stop_reason=StopReason.STOP)],
    ])
    sub = Agent(llm=sub_provider, tokenizer=_Tok())
    sub.with_tools([_echo_tool()])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Dig."}),
        _text_response("recovered"),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches")])

    list(orch.stream("Go"))

    (result,) = _tool_results_of(orch_provider, 1)
    assert result.is_error is True
    # The child's debited round is visible in the parent's aggregation
    # even though its turn died mid-flight.
    assert orch.last_turn is not None
    (child,) = orch.last_turn.children
    assert child.tool_call_id == "tu_1"
    assert child.diagnostics.rounds


def test_provider_reported_cost_wins_over_the_table():
    from anchor.llm.models import StopReason, StreamChunk, Usage

    agent, _ = _agent([[
        StreamChunk(content="hi"),
        StreamChunk(
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                total_cost=0.5,
            ),
            stop_reason=StopReason.STOP,
        ),
    ]])

    list(agent.stream("Go"))

    assert agent.last_turn is not None
    assert agent.last_turn.rounds[0].cost_usd == 0.5
    assert agent.last_turn.total_cost_usd == 0.5


def test_child_own_limits_under_limitless_parent_report_turn_scope():
    # A subagent never roots a run pool: its own limits are per-turn
    # and must say so, regardless of the parent's configuration.
    child_provider = FakeLLMProvider(
        [
            _tool_use_response(f"s_{i}", "echo", {"x": "hi"})
            for i in range(3)
        ]
        + [_text_response("done")],
    )
    orch, _ = _agent([
        _tool_use_response("tu_1", "task", {
            "agent_name": "researcher", "task": "Dig.",
        }),
        _text_response("all good"),
    ])
    orch.with_subagents([
        SubagentDefinition(
            name="researcher",
            description="Researches",
            tools=(_echo_tool(),),
            usage_limits=UsageLimits(tool_calls_limit=1),
        ),
    ])
    _, sub = orch._subagents["researcher"]
    sub._llm = child_provider

    events = list(orch.stream("Go"))

    breach = next(e for e in events if isinstance(e, UsageLimitReached))
    assert breach.scope == "turn"
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"


# ---------------------------------------------------------------------------
# Concurrent children (async path) debit the same pool
# ---------------------------------------------------------------------------


async def test_concurrent_children_share_the_pool():
    sub_a, a_provider = _burning_child(6)
    sub_b, b_provider = _burning_child(6)
    orch, _ = _agent([
        _multi_tool_use_response([
            ("tu_a", "researcher_a", {"task": "Dig."}),
            ("tu_b", "researcher_b", {"task": "Dig too."}),
        ]),
        _text_response("wrapped"),
    ])
    orch.with_tools([
        sub_a.as_tool("researcher_a", "Researches"),
        sub_b.as_tool("researcher_b", "Researches"),
    ])
    orch.with_usage_limits(UsageLimits(total_tokens_limit=500))

    events = [e async for e in orch.astream("Go")]

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "usage_limit"
    assert {c.tool_call_id for c in final.diagnostics.children} == {
        "tu_a", "tu_b",
    }
    # Both children were cut short of their 6 canned tool rounds —
    # the pool is shared, not per-child.
    calls_made = len(a_provider.seen_messages) + len(b_provider.seen_messages)
    assert calls_made < 14
    for sub in (sub_a, sub_b):
        assert sub.last_turn is not None
        assert sub.last_turn.stopped_by == "usage_limit"
