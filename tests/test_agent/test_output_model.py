"""Structured output on the main run (Agent.with_output_model + run/arun).

Tool mode (default): a synthetic final_result tool carries the schema,
tool_choice="any" keeps the model from stopping in plain text, and a
valid call ends the turn. Prompted mode reuses the subagent mechanic.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from anchor.agent import Agent, AgentTool, TurnFinished, tool
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import _echo_tool, _tool_results_of


class Finding(BaseModel):
    answer: str
    confidence: float


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


# ---------------------------------------------------------------------------
# Tool mode
# ---------------------------------------------------------------------------


def test_run_returns_validated_output():
    agent, provider = _agent([
        _tool_use_response(
            "tu_1", "final_result", {"answer": "42", "confidence": 0.9},
        ),
    ])
    agent.with_output_model(Finding)

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    assert result.answer == "42"
    assert agent.last_output == result.model_dump_json()
    # tool_choice="any" was sent; only one LLM call was needed.
    assert provider.seen_kwargs[0]["tool_choice"] == "any"
    assert len(provider.seen_messages) == 1


def test_turn_finished_carries_output():
    agent, _ = _agent([
        _tool_use_response(
            "tu_1", "final_result", {"answer": "42", "confidence": 0.9},
        ),
    ])
    agent.with_output_model(Finding)

    events = list(agent.stream("Question?"))

    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert final.output is not None
    assert Finding.model_validate_json(final.output).answer == "42"
    assert final.diagnostics.stopped_by == "stop"


def test_invalid_output_retries_via_error_result():
    agent, provider = _agent([
        _tool_use_response("tu_1", "final_result", {"answer": "42"}),
        _tool_use_response(
            "tu_2", "final_result", {"answer": "42", "confidence": 0.9},
        ),
    ])
    agent.with_output_model(Finding)

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    # The validation error went back to the model as an error result.
    (err,) = _tool_results_of(provider, 1)
    assert err.is_error is True
    assert "confidence" in err.content


def test_real_tools_run_before_the_final_result():
    agent, provider = _agent(
        [
            _tool_use_response("tu_1", "echo", {"x": "hi"}),
            _tool_use_response(
                "tu_2", "final_result", {"answer": "hi", "confidence": 1.0},
            ),
        ],
        tools=[_echo_tool()],
    )
    agent.with_output_model(Finding)

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    (echoed,) = _tool_results_of(provider, 1)
    assert echoed.content == "echo:hi"


async def test_arun_async_mirror():
    agent, _ = _agent([
        _tool_use_response(
            "tu_1", "final_result", {"answer": "42", "confidence": 0.9},
        ),
    ])
    agent.with_output_model(Finding)

    result = await agent.arun("Question?")
    assert isinstance(result, Finding)
    assert result.confidence == 0.9


def test_plain_text_stop_gets_nudged_then_succeeds():
    agent, provider = _agent([
        _text_response("The answer is 42."),
        _tool_use_response(
            "tu_1", "final_result", {"answer": "42", "confidence": 0.9},
        ),
    ])
    agent.with_output_model(Finding)

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    nudged = provider.seen_messages[1]
    assert any("final_result" in str(m.content) for m in nudged)


def test_retries_exhausted_raises():
    agent, _ = _agent([
        _text_response("plain"),
        _text_response("still plain"),
        _text_response("stubborn"),
    ])
    agent.with_output_model(Finding, max_output_retries=1)

    with pytest.raises(ValueError, match="final_result"):
        agent.run("Question?")


def test_final_result_collision_checked():
    clashing = tool(lambda: "x", name="final_result", description="clash")
    agent, _ = _agent([])
    agent.with_tools([clashing])
    with pytest.raises(ValueError, match="final_result"):
        agent.with_output_model(Finding)

    agent2, _ = _agent([])
    agent2.with_output_model(Finding)
    with pytest.raises(ValueError, match="final_result"):
        agent2.with_tools([clashing])


def test_run_without_output_model_raises():
    agent, _ = _agent([])
    with pytest.raises(ValueError, match="with_output_model"):
        agent.run("Question?")


def test_text_agents_unaffected():
    agent, provider = _agent([_text_response("hi")])
    assert "".join(agent.chat("Go")) == "hi"
    assert "tool_choice" not in provider.seen_kwargs[0]
    assert agent.last_output is None


# ---------------------------------------------------------------------------
# Final round / wrap-up (the P1 of the round review)
# ---------------------------------------------------------------------------


def test_final_round_forces_and_executes_final_result():
    # max_rounds=2: round 0 uses a real tool, round 1 (final) must
    # still record the answer — tool_choice names final_result and the
    # call executes even on the final round.
    agent, provider = _agent(
        [
            _tool_use_response("tu_1", "echo", {"x": "hi"}),
            _tool_use_response(
                "tu_2", "final_result", {"answer": "hi", "confidence": 1.0},
            ),
        ],
        tools=[_echo_tool()],
        max_rounds=2,
    )
    agent.with_output_model(Finding)

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    assert provider.seen_kwargs[1]["tool_choice"] == {
        "type": "tool", "name": "final_result",
    }
    # The output-aware notice asks for the call (not "do not call tools").
    final_messages = provider.seen_messages[1]
    assert any(
        "calling the final_result tool" in str(m.content)
        for m in final_messages
    )


def test_wrap_up_with_output_keeps_usage_limit_cause():
    from anchor.agent import UsageLimits

    agent, provider = _agent(
        [
            _tool_use_response("tu_1", "echo", {"x": "hi"}),
            _tool_use_response(
                "tu_2", "final_result", {"answer": "hi", "confidence": 1.0},
            ),
        ],
        tools=[_echo_tool()],
    )
    agent.with_output_model(Finding)
    agent.with_usage_limits(UsageLimits(tool_calls_limit=0))

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    # Output was delivered, but the budget cut stays visible.
    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "usage_limit"
    assert provider.seen_kwargs[1]["tool_choice"] == {
        "type": "tool", "name": "final_result",
    }


def test_mixed_batch_on_final_round_falls_back_to_error():
    # A provider that ignores tool_choice and mixes final_result with a
    # real tool on the final round: nothing executes, run() raises.
    from tests.test_agent.test_phase4_loop import _multi_tool_use_response

    agent, _ = _agent(
        [
            _multi_tool_use_response([
                ("tu_1", "final_result", {"answer": "hi", "confidence": 1.0}),
                ("tu_2", "echo", {"x": "side effect"}),
            ]),
        ],
        tools=[_echo_tool()],
        max_rounds=1,
    )
    agent.with_output_model(Finding)

    with pytest.raises(ValueError, match="without structured output"):
        agent.run("Question?")


def test_invalid_args_on_final_round_raises():
    agent, _ = _agent(
        [_tool_use_response("tu_1", "final_result", {"answer": "hi"})],
        max_rounds=1,
    )
    agent.with_output_model(Finding)

    with pytest.raises(ValueError, match="structured output"):
        agent.run("Question?")


# ---------------------------------------------------------------------------
# Reconfiguration + memory guard
# ---------------------------------------------------------------------------


def test_reconfigure_tool_to_prompted_drops_the_tool():
    agent, provider = _agent([
        _text_response('{"answer": "42", "confidence": 0.9}'),
    ])
    agent.with_output_model(Finding)  # tool mode first
    agent.with_output_model(Finding, mode="prompted")  # reconfigure

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    assert provider.seen_kwargs[0].get("tools") is None
    assert "tool_choice" not in provider.seen_kwargs[0]
    # And tool mode can be configured again without self-collision.
    agent.with_output_model(Finding)


def test_prompted_mode_with_memory_is_refused():
    from anchor.memory.manager import MemoryManager

    agent, _ = _agent([])
    agent.with_memory(MemoryManager())
    agent.with_output_model(Finding, mode="prompted")

    with pytest.raises(ValueError, match="without memory"):
        agent.run("Question?")


# ---------------------------------------------------------------------------
# Prompted mode
# ---------------------------------------------------------------------------


def test_prompted_mode_validates_and_retries():
    agent, provider = _agent([
        _text_response("not json"),
        _text_response('{"answer": "42", "confidence": 0.9}'),
    ])
    agent.with_output_model(Finding, mode="prompted")

    result = agent.run("Question?")

    assert isinstance(result, Finding)
    assert result.answer == "42"
    # The schema instruction was injected into the prompt.
    first_turn = provider.seen_messages[0]
    assert any("schema" in str(m.content) for m in first_turn)
    # No synthetic tool in prompted mode.
    assert provider.seen_kwargs[0].get("tools") is None
