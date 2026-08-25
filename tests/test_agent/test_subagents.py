"""Phase 4 subagent tests: as_tool primitive + declarative task tool."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from anchor.agent.agent import Agent
from anchor.agent.subagent import SubagentDefinition
from anchor.llm.models import Role, StreamChunk
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


class Finding(BaseModel):
    answer: str
    confidence: float


def _agent(
    responses: list[list[StreamChunk]], *, max_rounds: int = 10,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, max_rounds=max_rounds, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    return agent, provider


def _sub(responses: list[list[StreamChunk]]) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    sub = Agent(llm=provider, tokenizer=_Tok())
    sub.with_system_prompt("You are a researcher.")
    return sub, provider


# ---------------------------------------------------------------------------
# as_tool primitive
# ---------------------------------------------------------------------------


def test_as_tool_two_level_sync():
    sub, sub_provider = _sub([_text_response("42")])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Find the answer."}),
        _text_response("The answer is 42."),
    ])
    orch.with_tools([sub.as_tool("researcher", "Researches questions")])

    assert "".join(orch.chat("Go")) == "The answer is 42."

    # The subagent saw only the curated task, none of the parent context.
    sub_user_messages = [
        str(m.content)
        for m in sub_provider.seen_messages[0]
        if m.role == Role.USER
    ]
    assert any("Find the answer." in m for m in sub_user_messages)
    assert not any(m == "Go" for m in sub_user_messages)

    # The condensed result came back as the tool result.
    (result,) = _tool_results_of(orch_provider, 1)
    assert result.content == "42"
    assert result.is_error is False


def test_as_tool_output_model_returns_normalized_json():
    sub, sub_provider = _sub([
        _text_response('{"answer": "42", "confidence": 0.9}'),
    ])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Find it."}),
        _text_response("done"),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])
    list(orch.chat("Go"))

    # Schema instruction was appended to the task prompt.
    sub_user = [
        str(m.content)
        for m in sub_provider.seen_messages[0]
        if m.role == Role.USER
    ]
    assert any("Respond ONLY with a single JSON object" in m for m in sub_user)

    (result,) = _tool_results_of(orch_provider, 1)
    parsed = Finding.model_validate_json(result.content)
    assert parsed.answer == "42"
    assert parsed.confidence == 0.9


def test_as_tool_output_model_retries_with_validation_error():
    sub, sub_provider = _sub([
        _text_response("Sure! The answer is 42."),  # invalid JSON
        _text_response('{"answer": "42", "confidence": 0.9}'),
    ])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Find it."}),
        _text_response("done"),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])
    list(orch.chat("Go"))

    # Two subagent turns: original + one retry carrying the error.
    assert len(sub_provider.seen_messages) == 2
    retry_user = [
        str(m.content)
        for m in sub_provider.seen_messages[1]
        if m.role == Role.USER
    ]
    assert any("not valid JSON" in m for m in retry_user)
    assert any("Sure! The answer is 42." in m for m in retry_user)

    (result,) = _tool_results_of(orch_provider, 1)
    assert Finding.model_validate_json(result.content).answer == "42"


def test_as_tool_output_model_persistent_failure_is_error():
    sub, _ = _sub([
        _text_response("still not json"),
        _text_response("nope, not json either"),
    ])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Find it."}),
        _text_response("recovered"),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])
    assert "".join(orch.chat("Go")) == "recovered"

    (result,) = _tool_results_of(orch_provider, 1)
    assert result.is_error is True
    assert "failed schema validation" in result.content


def test_as_tool_fenced_json_accepted():
    fenced = '```json\n{"answer": "42", "confidence": 0.9}\n```'
    sub, _ = _sub([_text_response(fenced)])
    orch, orch_provider = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "Find it."}),
        _text_response("done"),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])
    list(orch.chat("Go"))

    (result,) = _tool_results_of(orch_provider, 1)
    assert Finding.model_validate_json(result.content).answer == "42"


def test_no_nesting_as_tool_raises():
    leaf, _ = _sub([_text_response("x")])
    middle, _ = _sub([_text_response("y")])
    middle.with_tools([leaf.as_tool("leaf", "leaf agent")])

    with pytest.raises(ValueError, match="no nesting"):
        middle.as_tool("middle", "middle agent")


def test_with_subagents_rejects_subagent_tools_in_definition():
    leaf, _ = _sub([_text_response("x")])
    orch, _ = _agent([_text_response("hi")])

    with pytest.raises(ValueError, match="no nesting"):
        orch.with_subagents([
            SubagentDefinition(
                name="bad",
                description="carries a subagent tool",
                tools=(leaf.as_tool("leaf", "leaf agent"),),
            ),
        ])


async def test_parallel_subagents_achat():
    sub_a, _ = _sub([_text_response("alpha")])
    sub_b, _ = _sub([_text_response("beta")])
    orch, orch_provider = _agent([
        _multi_tool_use_response([
            ("tu_1", "agent_a", {"task": "a"}),
            ("tu_2", "agent_b", {"task": "b"}),
        ]),
        _text_response("done"),
    ])
    orch.with_tools([
        sub_a.as_tool("agent_a", "Agent A"),
        sub_b.as_tool("agent_b", "Agent B"),
    ])

    chunks = [c async for c in orch.achat("Go")]
    assert "".join(chunks) == "done"

    results = _tool_results_of(orch_provider, 1)
    assert [r.content for r in results] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Declarative layer: with_subagents + task meta-tool
# ---------------------------------------------------------------------------


def test_with_subagents_task_tool_and_listing():
    # Subagents built with model=None share the orchestrator's provider,
    # so the canned responses are consumed in call order:
    # orchestrator round 1 → subagent turn → orchestrator round 2.
    responses = [
        _tool_use_response(
            "tu_1", "task", {"agent_name": "writer", "task": "Write a haiku."},
        ),
        _text_response("haiku text"),
        _text_response("Delivered."),
    ]
    orch, provider = _agent(responses)
    orch.with_subagents([
        SubagentDefinition(
            name="writer",
            description="Writes short texts",
            system_prompt="You are a poet.",
        ),
    ])

    assert "".join(orch.chat("Go")) == "Delivered."

    # Discovery listing is in the orchestrator's system message.
    system_msg = next(
        m for m in provider.seen_messages[0] if m.role == Role.SYSTEM
    )
    assert "Available subagents" in str(system_msg.content)
    assert "writer: Writes short texts" in str(system_msg.content)

    # The subagent turn (call index 1) used the poet system prompt.
    sub_system = next(
        m for m in provider.seen_messages[1] if m.role == Role.SYSTEM
    )
    assert "You are a poet." in str(sub_system.content)

    # Condensed return reached the orchestrator.
    (result,) = _tool_results_of(provider, 2)
    assert result.content == "haiku text"


def test_task_tool_unknown_subagent():
    responses = [
        _tool_use_response(
            "tu_1", "task", {"agent_name": "ghost", "task": "boo"},
        ),
        _text_response("ok"),
    ]
    orch, provider = _agent(responses)
    orch.with_subagents([
        SubagentDefinition(name="writer", description="Writes"),
    ])
    list(orch.chat("Go"))

    (result,) = _tool_results_of(provider, 1)
    assert "Unknown subagent: 'ghost'" in result.content
    assert "writer" in result.content


def test_with_subagents_duplicate_name_raises():
    orch, _ = _agent([_text_response("hi")])
    orch.with_subagents([SubagentDefinition(name="w", description="d")])
    with pytest.raises(ValueError, match="Duplicate subagent name"):
        orch.with_subagents([SubagentDefinition(name="w", description="d2")])


# ---------------------------------------------------------------------------
# Done-when: 2-level agent with per-round accounting visible
# ---------------------------------------------------------------------------


def test_e2e_orchestrator_subagent_with_accounting():
    sub, _ = _sub([
        _text_response(json.dumps({"answer": "blue", "confidence": 0.8})),
    ])
    orch, _ = _agent([
        _tool_use_response("tu_1", "researcher", {"task": "What color?"}),
        _text_response("It is blue."),
    ])
    orch.with_tools([
        sub.as_tool("researcher", "Researches", output_model=Finding),
    ])

    assert "".join(orch.chat("Go")) == "It is blue."

    # Orchestrator accounting: two rounds, both with tool schemas counted,
    # round 0 with the subagent's condensed return counted.
    turn = orch.last_turn
    assert turn is not None
    assert turn.stopped_by == "stop"
    assert len(turn.rounds) == 2
    assert turn.rounds[0].tool_schema_tokens > 0
    assert turn.rounds[0].tool_result_tokens > 0

    # Subagent kept its own accounting for its turn.
    assert sub.last_turn is not None
    assert len(sub.last_turn.rounds) == 1
