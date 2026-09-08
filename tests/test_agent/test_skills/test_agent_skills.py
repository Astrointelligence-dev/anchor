"""Integration tests for Agent + skills system."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from anchor.agent.agent import Agent
from anchor.agent.skills.models import Skill
from anchor.agent.tools import AgentTool
from anchor.llm.models import (
    LLMResponse,
    Message,
    StopReason,
    StreamChunk,
    ToolCallDelta,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Fake LLM provider for skills tests
# ---------------------------------------------------------------------------


def _text_response(text: str) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    if text:
        chunks.append(StreamChunk(content=text))
    chunks.append(StreamChunk(stop_reason=StopReason.STOP))
    return chunks


def _tool_use_response(
    tool_id: str, name: str, arguments: dict[str, Any], *, text_before: str = "",
) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    if text_before:
        chunks.append(StreamChunk(content=text_before))
    args_json = json.dumps(arguments)
    chunks.append(StreamChunk(
        tool_call_delta=ToolCallDelta(index=0, id=tool_id, name=name),
    ))
    chunks.append(StreamChunk(
        tool_call_delta=ToolCallDelta(index=0, arguments_fragment=args_json),
    ))
    chunks.append(StreamChunk(stop_reason=StopReason.TOOL_USE))
    return chunks


class FakeLLMProvider:
    """Mock LLM provider that captures call kwargs for assertion."""

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = responses
        self._call_index = 0
        self.last_tools: list[ToolSchema] | None = None
        self.last_messages: list[Message] | None = None

    @property
    def model_id(self) -> str:
        return "mock/test-model"

    @property
    def provider_name(self) -> str:
        return "mock"

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        self.last_tools = tools
        self.last_messages = messages
        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
            yield from chunks
        else:
            yield StreamChunk(stop_reason=StopReason.STOP)

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        self.last_tools = tools
        self.last_messages = messages
        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
            for chunk in chunks:
                yield chunk
        else:
            yield StreamChunk(stop_reason=StopReason.STOP)

    def invoke(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "ok"


def _make_tool(name: str) -> AgentTool:
    return AgentTool(
        name=name, description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}}, fn=_noop,
    )


def _always_skill(name: str = "mem", tool_name: str = "save_fact") -> Skill:
    return Skill(
        name=name, description=f"{name} skill",
        tools=(_make_tool(tool_name),), activation="always",
    )


def _on_demand_skill(
    name: str = "rag", tool_name: str = "search_docs",
) -> Skill:
    return Skill(
        name=name, description=f"{name} skill",
        instructions=f"Use {tool_name} to search.",
        tools=(_make_tool(tool_name),), activation="on_demand",
    )


def _make_agent(
    responses: list[list[StreamChunk]], **kwargs: Any,
) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, **kwargs)
    agent.with_system_prompt("You are helpful.")
    return agent, provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlwaysLoadedSkill:
    def test_tools_available_immediately(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skill(_always_skill("mem", "save_fact"))

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "save_fact" in names

    def test_chat_includes_skill_tools(self) -> None:
        agent, provider = _make_agent([_text_response("Hello!")])
        agent.with_skill(_always_skill("mem", "save_fact"))

        result = "".join(agent.chat("Hi"))
        assert result == "Hello!"

        # Verify tools were sent to the provider
        assert provider.last_tools is not None
        tool_names = [t.name for t in provider.last_tools]
        assert "save_fact" in tool_names


class TestOnDemandSkill:
    def test_tools_not_available_initially(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        all_tools = agent._all_active_tools()
        tool_names = [t.name for t in all_tools]
        assert "search_docs" not in tool_names
        assert "activate_skill" in tool_names

    def test_activate_skill_meta_tool_injected(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skill(_on_demand_skill())

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "activate_skill" in names

    def test_no_activate_tool_when_only_always_skills(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skill(_always_skill())

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "activate_skill" not in names

    def test_activation_mid_conversation(self) -> None:
        """Simulate: round 1 = agent calls activate_skill, round 2 = agent uses new tool."""
        responses = [
            _tool_use_response(
                "tool_1", "activate_skill", {"skill_name": "rag"},
                text_before="Let me enable that.",
            ),
            _tool_use_response("tool_2", "search_docs", {}),
            _text_response("Here are the results."),
        ]
        agent, _ = _make_agent(responses)
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        result = "".join(agent.chat("Find docs about pipeline"))
        assert "results" in result

        # After activation, the skill should be active
        assert agent._skill_registry.is_active("rag")


class TestMixedToolsAndSkills:
    def test_direct_tools_and_skills_coexist(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        direct_tool = _make_tool("my_direct_tool")
        agent.with_tools([direct_tool])
        agent.with_skill(_always_skill("mem", "save_fact"))

        all_tools = agent._all_active_tools()
        names = [t.name for t in all_tools]
        assert "my_direct_tool" in names
        assert "save_fact" in names

    def test_with_tools_backward_compat(self) -> None:
        """The old with_tools() API still works unchanged."""
        agent, provider = _make_agent([_text_response("Hi")])
        tool = _make_tool("legacy_tool")
        agent.with_tools([tool])

        result = "".join(agent.chat("Hi"))
        assert result == "Hi"

        assert provider.last_tools is not None
        tool_names = [t.name for t in provider.last_tools]
        assert "legacy_tool" in tool_names


class TestWithSkillsFluent:
    def test_with_skills_registers_multiple(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skills([
            _always_skill("a", "tool_a"),
            _always_skill("b", "tool_b"),
        ])
        tools = agent._all_active_tools()
        names = [t.name for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_chaining_works(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        result = agent.with_skill(_always_skill("a", "t1")).with_skill(
            _on_demand_skill("b", "t2"),
        )
        assert result is agent


class TestDiscoveryPromptInSystem:
    def test_discovery_appended_when_on_demand_exists(self) -> None:
        agent, provider = _make_agent([_text_response("Hi")])
        agent.with_skill(_on_demand_skill("rag", "search_docs"))

        "".join(agent.chat("Hello"))

        # The discovery prompt should be in the messages sent to the provider
        assert provider.last_messages is not None
        system_msgs = [m for m in provider.last_messages if m.role.value == "system"]
        system_text = " ".join(
            m.content for m in system_msgs if isinstance(m.content, str)
        )
        assert "activate_skill" in system_text
        assert "rag" in system_text

    def test_no_discovery_when_only_always(self) -> None:
        agent, provider = _make_agent([_text_response("Hi")])
        agent.with_skill(_always_skill("mem", "save_fact"))

        "".join(agent.chat("Hello"))

        assert provider.last_messages is not None
        system_msgs = [m for m in provider.last_messages if m.role.value == "system"]
        system_text = " ".join(
            m.content for m in system_msgs if isinstance(m.content, str)
        )
        assert "activate_skill" not in system_text


class TestAsyncSkills:
    async def test_achat_with_always_skill(self) -> None:
        agent, provider = _make_agent([_text_response("Async hi")])
        agent.with_skill(_always_skill("mem", "save_fact"))

        chunks: list[str] = []
        async for chunk in agent.achat("Hello"):
            chunks.append(chunk)
        assert "".join(chunks) == "Async hi"

        assert provider.last_tools is not None
        tool_names = [t.name for t in provider.last_tools]
        assert "save_fact" in tool_names


class TestDuplicateSkillRegistration:
    def test_duplicate_raises(self) -> None:
        agent, _ = _make_agent([_text_response("Hi")])
        agent.with_skill(_always_skill("dup", "t1"))
        with pytest.raises(ValueError, match="already registered"):
            agent.with_skill(_always_skill("dup", "t2"))
