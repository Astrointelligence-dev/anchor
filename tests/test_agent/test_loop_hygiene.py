"""Loop hygiene (v0.2 #2): result cap, read/write scheduling, stuck detection.

Items 2.1 and 2.2 are corrections — their core tests fail against the
pre-change HEAD by design (giant tool results entered the messages
whole; the async path ran every tool concurrently, writes included).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from anchor.agent.agent import Agent, _RoundState
from anchor.agent.events import TurnFinished
from anchor.agent.hooks import ApprovalDecision
from anchor.agent.models import AgentTool
from anchor.agent.subagent import _SUBAGENT_MARKER, _partial
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import (
    _agent,
    _echo_tool,
    _multi_tool_use_response,
    _tool_results_of,
)

# ---------------------------------------------------------------------------
# 2.1 — tool results are capped in the loop
# ---------------------------------------------------------------------------


def _big_payload(n_words: int) -> str:
    return "HEADSTART " + " ".join(f"w{i}" for i in range(n_words)) + " TAILEND"


def _big_tool(
    n_words: int = 30_000, *, max_result_tokens: int | None = None,
) -> AgentTool:
    payload = _big_payload(n_words)
    return AgentTool(
        name="big",
        description="returns a huge result",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: payload,
        max_result_tokens=max_result_tokens,
    )


def test_giant_tool_result_is_capped_in_messages():
    responses = [
        _tool_use_response("tu_1", "big", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[_big_tool()])
    assert "".join(agent.chat("Go")) == "done"

    (result,) = _tool_results_of(provider, 1)
    words = len(result.content.split())
    # Default cap (10k tokens) plus the truncation marker's own words.
    assert words <= 10_000 + 50
    assert "HEADSTART" in result.content
    assert "TAILEND" in result.content
    assert "truncated" in result.content


def test_per_tool_result_cap_overrides_default():
    responses = [
        _tool_use_response("tu_1", "big", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(
        responses, tools=[_big_tool(1_000, max_result_tokens=100)],
    )
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert len(result.content.split()) <= 100 + 25
    assert "truncated" in result.content


def test_result_cap_disabled_with_none():
    responses = [
        _tool_use_response("tu_1", "big", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(
        responses, tools=[_big_tool()], tool_result_max_tokens=None,
    )
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    # Uncapped: the full payload (30k words + head/tail sentinels).
    assert len(result.content.split()) == 30_002


def test_giant_error_result_is_capped():
    def explode(**_: Any) -> str:
        raise RuntimeError(_big_payload(30_000))

    bad = AgentTool(
        name="bad",
        description="raises a huge error",
        input_schema={"type": "object", "properties": {}},
        fn=explode,
    )
    responses = [
        _tool_use_response("tu_1", "bad", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[bad])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert len(result.content.split()) <= 10_000 + 50


def test_error_result_honors_per_tool_cap():
    def explode(**_: Any) -> str:
        raise RuntimeError(_big_payload(1_000))

    bad = AgentTool(
        name="bad",
        description="raises a big error",
        input_schema={"type": "object", "properties": {}},
        fn=explode,
        max_result_tokens=100,
    )
    responses = [
        _tool_use_response("tu_1", "bad", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[bad])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is True
    assert len(result.content.split()) <= 100 + 25


def test_token_dense_result_never_grows_when_capped():
    # 120 short words under a 100-token cap: the fallback slices must
    # not overlap (a "capped" result longer than the input, with the
    # middle duplicated).
    dense = "ab " * 120
    tool = AgentTool(
        name="dense",
        description="returns short dense words",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: dense,
        max_result_tokens=100,
    )
    responses = [
        _tool_use_response("tu_1", "dense", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[tool])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert len(result.content.split()) <= 100 + 25
    assert "truncated" in result.content


def test_single_line_blob_capped_despite_word_count():
    # A word-counting fallback tokenizer sees one "word" — the char
    # floor must still cut the blob.
    blob = "x" * 200_000
    tool = AgentTool(
        name="blob",
        description="returns a single-line blob",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: blob,
    )
    responses = [
        _tool_use_response("tu_1", "blob", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[tool])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert len(result.content) < 90_000
    assert "truncated" in result.content


def test_subagent_results_are_exempt_from_the_cap():
    big = _big_tool()
    object.__setattr__(big, _SUBAGENT_MARKER, True)
    responses = [
        _tool_use_response("tu_1", "big", {}),
        _text_response("done"),
    ]
    agent, provider = _agent(responses, tools=[big])
    list(agent.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert len(result.content.split()) == 30_002


def test_non_positive_caps_are_rejected():
    with pytest.raises(ValueError, match="tool_result_max_tokens"):
        Agent(llm=FakeLLMProvider([]), tokenizer=_Tok(), tool_result_max_tokens=0)
    with pytest.raises(ValidationError):
        AgentTool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            fn=lambda **_: "",
            max_result_tokens=0,
        )


# ---------------------------------------------------------------------------
# 2.2 — read-only tools fan out, writes run serially
# ---------------------------------------------------------------------------


def _sleepy_tool(
    name: str,
    tracker: dict[str, Any],
    *,
    delay: float = 0.03,
    read_only: bool = False,
    requires_approval: bool = False,
) -> AgentTool:
    """Async tool that records start/end order and peak concurrency."""
    t = AgentTool(
        name=name,
        description="sleeps then replies",
        input_schema={"type": "object", "properties": {}},
        fn=lambda **_: name,
        read_only=read_only,
        requires_approval=requires_approval,
    )

    async def acall(_n: str, _i: dict[str, Any]) -> str:
        tracker["active"] += 1
        tracker["max"] = max(tracker["max"], tracker["active"])
        tracker["log"].append(("start", name))
        await asyncio.sleep(delay)
        tracker["active"] -= 1
        tracker["log"].append(("end", name))
        return name

    object.__setattr__(t, "_anchor_async_caller", acall)
    return t


def _tracker() -> dict[str, Any]:
    return {"active": 0, "max": 0, "log": []}


async def test_write_tools_never_overlap():
    responses = [
        _multi_tool_use_response([("tu_1", "w1", {}), ("tu_2", "w2", {})]),
        _text_response("done"),
    ]
    tracker = _tracker()
    tools = [_sleepy_tool("w1", tracker), _sleepy_tool("w2", tracker)]
    agent, provider = _agent(responses, tools=tools)
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "done"
    # Unmarked tools are writes: they must run one at a time.
    assert tracker["max"] == 1
    results = _tool_results_of(provider, 1)
    assert [r.tool_call_id for r in results] == ["tu_1", "tu_2"]


async def test_read_only_tools_run_concurrently():
    responses = [
        _multi_tool_use_response([("tu_1", "r1", {}), ("tu_2", "r2", {})]),
        _text_response("done"),
    ]
    tracker = _tracker()
    tools = [
        _sleepy_tool("r1", tracker, read_only=True),
        _sleepy_tool("r2", tracker, read_only=True),
    ]
    agent, _ = _agent(responses, tools=tools)
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "done"
    assert tracker["max"] == 2


async def test_mixed_calls_batch_reads_and_serialize_writes():
    responses = [
        _multi_tool_use_response([
            ("tu_1", "r1", {}),
            ("tu_2", "r2", {}),
            ("tu_3", "w", {}),
            ("tu_4", "r3", {}),
        ]),
        _text_response("done"),
    ]
    tracker = _tracker()
    tools = [
        _sleepy_tool("r1", tracker, read_only=True),
        _sleepy_tool("r2", tracker, read_only=True),
        _sleepy_tool("w", tracker),
        _sleepy_tool("r3", tracker, read_only=True),
    ]
    agent, provider = _agent(responses, tools=tools)
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "done"
    log = tracker["log"]
    # r1/r2 overlap; the write starts only after both end; r3 only
    # after the write ends.
    assert log.index(("start", "w")) > log.index(("end", "r1"))
    assert log.index(("start", "w")) > log.index(("end", "r2"))
    assert log.index(("start", "r3")) > log.index(("end", "w"))
    assert log.index(("start", "r2")) < log.index(("end", "r1"))
    # Results keep call order regardless of scheduling.
    results = _tool_results_of(provider, 1)
    assert [r.tool_call_id for r in results] == ["tu_1", "tu_2", "tu_3", "tu_4"]


async def test_read_concurrency_capped():
    n = 12
    calls = [(f"tu_{i}", f"r{i}", {}) for i in range(n)]
    responses = [_multi_tool_use_response(calls), _text_response("done")]
    tracker = _tracker()
    tools = [
        _sleepy_tool(f"r{i}", tracker, read_only=True, delay=0.02)
        for i in range(n)
    ]
    agent, _ = _agent(responses, tools=tools)
    chunks = [c async for c in agent.achat("Go")]

    assert "".join(chunks) == "done"
    assert tracker["max"] <= 10


async def test_approval_waits_do_not_hold_execution_slots():
    # 11 approval-gated reads, and a callback that answers only after
    # every request arrived: if approvals held semaphore slots, request
    # 11 could never arrive and the turn would deadlock.
    n = 11
    calls = [(f"tu_{i}", f"r{i}", {}) for i in range(n)]
    responses = [_multi_tool_use_response(calls), _text_response("done")]
    tracker = _tracker()
    tools = [
        _sleepy_tool(
            f"r{i}", tracker, delay=0.01, read_only=True,
            requires_approval=True,
        )
        for i in range(n)
    ]
    agent, _ = _agent(responses, tools=tools)
    pending: list[Any] = []

    async def approve_when_all_arrived(request: Any) -> ApprovalDecision:
        pending.append(request)
        while len(pending) < n:
            await asyncio.sleep(0.005)
        return ApprovalDecision(approved=True)

    agent.with_approval(approve_when_all_arrived)

    async def _run() -> list[str]:
        return [c async for c in agent.achat("Go")]

    chunks = await asyncio.wait_for(_run(), timeout=5)
    assert "".join(chunks) == "done"
    assert len(pending) == n


# ---------------------------------------------------------------------------
# 2.3 — stuck detection: nudge, then graceful wrap-up
# ---------------------------------------------------------------------------


def _same_call_responses(n: int) -> list[list[Any]]:
    """n rounds of the identical echo call, then a text answer."""
    rounds: list[list[Any]] = [
        _tool_use_response(f"tu_{i}", "echo", {"x": "same"}) for i in range(n)
    ]
    rounds.append(_text_response("done"))
    return rounds


def _has_nudge(messages: list[Any]) -> bool:
    return any("times in a row" in str(m.content) for m in messages)


def test_identical_results_nudge_then_stuck():
    # Rounds 0-3: identical call+result → streak 4 → nudge. Round 4
    # repeats after the nudge → stuck → round 5 is the wrap-up.
    agent, provider = _agent(_same_call_responses(5), tools=[_echo_tool()])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stuck"
    assert final.text == "done"
    assert len(provider.seen_messages) == 6
    assert _has_nudge(provider.seen_messages[4])
    wrap_up = provider.seen_messages[5]
    assert any("Final round" in str(m.content) for m in wrap_up)
    assert any("no progress" in str(m.content) for m in wrap_up)
    assert provider.seen_kwargs[5]["tool_choice"] == "none"
    assert final.diagnostics.rounds[-1].tool_calls == 0


async def test_identical_results_stuck_async_mirror():
    agent, provider = _agent(_same_call_responses(5), tools=[_echo_tool()])
    events = [e async for e in agent.astream("Go")]

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stuck"
    assert len(provider.seen_messages) == 6
    assert _has_nudge(provider.seen_messages[4])
    assert provider.seen_kwargs[5]["tool_choice"] == "none"


def test_identical_errors_nudge_at_three():
    def fail(x: str) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    bad = _echo_tool().model_copy(update={"fn": fail})
    # Rounds 0-2: identical call+error → streak 3 → nudge. Round 3
    # repeats → stuck → round 4 wraps up.
    agent, provider = _agent(_same_call_responses(4), tools=[bad])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stuck"
    assert len(provider.seen_messages) == 5
    assert _has_nudge(provider.seen_messages[3])
    assert not _has_nudge(provider.seen_messages[2])


def test_polling_with_changing_results_is_not_stuck():
    # Same call, different observation each time — legitimate polling.
    ticks = iter(range(100))
    poll = _echo_tool().model_copy(
        update={"fn": lambda x: f"tick {next(ticks)}"},
    )
    agent, provider = _agent(_same_call_responses(6), tools=[poll])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert final.text == "done"
    assert not any(_has_nudge(msgs) for msgs in provider.seen_messages)


def test_polling_large_results_differing_in_the_middle_is_not_stuck():
    # Post-cap, these results are identical (same head, same tail, same
    # length) — the pre-cap digest must still tell them apart.
    ticks = iter(range(100))

    def poll(x: str) -> str:
        return "H " * 30_000 + f"mid{next(ticks):03d} " + "T " * 30_000

    tool = _echo_tool().model_copy(update={"fn": poll})
    agent, provider = _agent(_same_call_responses(6), tools=[tool])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert not any(_has_nudge(msgs) for msgs in provider.seen_messages)


def test_changed_arguments_reset_the_streak():
    rounds: list[list[Any]] = [
        _tool_use_response("tu_0", "echo", {"x": "same"}),
        _tool_use_response("tu_1", "echo", {"x": "same"}),
        _tool_use_response("tu_2", "echo", {"x": "same"}),
        _tool_use_response("tu_3", "echo", {"x": "other"}),
        _tool_use_response("tu_4", "echo", {"x": "same"}),
        _text_response("done"),
    ]
    agent, provider = _agent(rounds, tools=[_echo_tool()])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert not any(_has_nudge(msgs) for msgs in provider.seen_messages)


def test_companion_call_does_not_hide_the_repetition():
    # The identical call repeats every round alongside a varying one —
    # the per-key ledger must still catch it.
    rounds: list[list[Any]] = [
        _multi_tool_use_response([
            (f"tu_{i}a", "echo", {"x": "same"}),
            (f"tu_{i}b", "echo", {"x": f"vary {i}"}),
        ])
        for i in range(5)
    ]
    rounds.append(_text_response("done"))
    agent, provider = _agent(rounds, tools=[_echo_tool()])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stuck"
    assert _has_nudge(provider.seen_messages[4])


def test_duplicate_calls_in_one_round_count_once():
    # Four identical calls fanned out in a single round are one round
    # of repetition, not four.
    rounds: list[list[Any]] = [
        _multi_tool_use_response(
            [(f"tu_{i}", "echo", {"x": "same"}) for i in range(4)],
        ),
        _text_response("done"),
    ]
    agent, provider = _agent(rounds, tools=[_echo_tool()])
    events = list(agent.stream("Go"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.diagnostics.stopped_by == "stop"
    assert not any(_has_nudge(msgs) for msgs in provider.seen_messages)


def test_no_tool_round_resets_the_ledger():
    agent, _ = _agent([_text_response("hi")])
    key = ("echo", "{}", 0, False)
    agent._stuck_counts = {key: 3}
    agent._stuck_nudged = key

    wrap_up, stopped_by, breach = agent._post_round_cuts(
        _RoundState(), [], [], [], run_tools=False, stopped_by="max_rounds",
    )

    assert agent._stuck_counts == {}
    assert agent._stuck_nudged is None
    assert (wrap_up, stopped_by, breach) == (False, "max_rounds", None)


def test_partial_note_marks_stuck_children():
    assert "stuck" in _partial("x", "stuck")
    assert _partial("x", "stop") == "x"
