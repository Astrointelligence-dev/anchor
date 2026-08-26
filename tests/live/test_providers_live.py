"""Live provider smoke tests — real API calls, env-gated per key.

Run locally:

    export ANTHROPIC_API_KEY=...   # enables the Anthropic tests
    export OPENAI_API_KEY=...      # enables the OpenAI test
    export GEMINI_API_KEY=...      # enables the Gemini test
    uv run pytest tests/live -q

Everything here was previously verified only against call_kwargs; these
are the first tests that touch the real APIs (the Gemini streaming
tools bug lived exactly in that blind spot). Kept tiny and cheap:
haiku-class models, short prompts, low max_response_tokens.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from anchor.agent import Agent, TurnFinished, tool

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


class Sum(BaseModel):
    value: int


def _events_of(agent: Agent, message: str) -> list:
    return list(agent.stream(message))


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_event_stream_turn_with_tool() -> None:
    agent = Agent(
        model=_ANTHROPIC_MODEL, max_rounds=4, max_response_tokens=300,
    )
    agent.with_system_prompt(
        "Use the add tool for any arithmetic. Answer tersely.",
    )
    agent.with_tools([add])

    events = _events_of(agent, "What is 17 + 25? Use the tool.")

    assert events[0].type == "turn_started"
    final = events[-1]
    assert isinstance(final, TurnFinished)
    assert "42" in final.text
    finished = [e for e in events if e.type == "tool_finished"]
    assert any(e.result == "42" for e in finished)
    # Anthropic reports usage on the stream — no estimation needed.
    assert final.diagnostics.rounds[0].prompt_tokens > 0
    assert final.diagnostics.stopped_by == "stop"


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_structured_output_run() -> None:
    agent = Agent(
        model=_ANTHROPIC_MODEL, max_rounds=3, max_response_tokens=300,
    )
    result = agent.with_output_model(Sum).run("What is 6 * 7?")
    assert isinstance(result, Sum)
    assert result.value == 42


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_prompt_caching_flag_accepted() -> None:
    from anchor.llm.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model=_ANTHROPIC_MODEL, prompt_caching=True)
    agent = Agent(llm=provider, max_response_tokens=100)
    agent.with_system_prompt("Answer with one word.")

    text = "".join(agent.chat("Say the word hello."))

    assert text
    turn = agent.last_turn
    assert turn is not None
    assert turn.rounds[0].prompt_tokens > 0


# ---------------------------------------------------------------------------
# OpenAI / Gemini — one tool turn each (the class of bug the Gemini
# stream parser had is only visible here)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_tool_turn() -> None:
    agent = Agent(model="gpt-5-mini", max_rounds=4, max_response_tokens=300)
    agent.with_system_prompt(
        "Use the add tool for any arithmetic. Answer tersely.",
    )
    agent.with_tools([add])

    events = _events_of(agent, "What is 17 + 25? Use the tool.")

    finished = [e for e in events if e.type == "tool_finished"]
    assert any(e.result == "42" for e in finished)
    assert "42" in events[-1].text


@pytest.mark.skipif(not _GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_gemini_streaming_tool_turn() -> None:
    agent = Agent(
        model="gemini-2.5-flash", max_rounds=4, max_response_tokens=300,
    )
    agent.with_system_prompt(
        "Use the add tool for any arithmetic. Answer tersely.",
    )
    agent.with_tools([add])

    events = _events_of(agent, "What is 17 + 25? Use the tool.")

    # Before the 2026-08-25 fix, Gemini streaming silently never
    # executed tools (no TOOL_USE stop reason, no ids, str(dict) args).
    finished = [e for e in events if e.type == "tool_finished"]
    assert any(e.result == "42" for e in finished)
