"""Tests for the Agent class."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from anchor.agent.agent import Agent
from anchor.agent.tools import AgentTool
from anchor.llm.models import (
    LLMResponse,
    Message,
    StopReason,
    StreamChunk,
    ToolCallDelta,
    ToolSchema,
)
from anchor.memory.manager import MemoryManager
from anchor.storage.json_memory_store import InMemoryEntryStore

# ---------------------------------------------------------------------------
# Fake LLM provider for testing (no SDK dependency needed)
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal tokenizer for tests."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _text_response(text: str) -> list[StreamChunk]:
    """Build stream chunks for a simple text response."""
    chunks: list[StreamChunk] = []
    if text:
        chunks.append(StreamChunk(content=text))
    chunks.append(StreamChunk(stop_reason=StopReason.STOP))
    return chunks


def _tool_use_response(
    tool_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    text_before: str = "",
) -> list[StreamChunk]:
    """Build stream chunks for a tool call response."""
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
    """Mock LLMProvider that returns canned StreamChunk sequences.

    Records the ``messages`` and ``tools`` passed to each call in
    ``seen_messages`` / ``seen_tools`` for loop-level assertions.
    """

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = responses
        self._call_index = 0
        self.seen_messages: list[list[Message]] = []
        self.seen_tools: list[list[ToolSchema] | None] = []

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
        self.seen_messages.append(list(messages))
        self.seen_tools.append(tools)
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
        self.seen_messages.append(list(messages))
        self.seen_tools.append(tools)
        if self._call_index < len(self._responses):
            chunks = self._responses[self._call_index]
            self._call_index += 1
            for chunk in chunks:
                yield chunk
        else:
            yield StreamChunk(stop_reason=StopReason.STOP)

    def invoke(
        self, messages: list[Message], **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    async def ainvoke(
        self, messages: list[Message], **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    responses: list[list[StreamChunk]],
    *,
    tools: list[AgentTool] | None = None,
    memory: MemoryManager | None = None,
    max_rounds: int = 10,
    system_prompt: str = "You are helpful.",
) -> Agent:
    """Create an Agent with a fake LLM provider."""
    provider = FakeLLMProvider(responses)
    agent = Agent(llm=provider, max_rounds=max_rounds)
    agent.with_system_prompt(system_prompt)
    if memory is not None:
        agent.with_memory(memory)
    if tools:
        agent.with_tools(tools)
    return agent


# ---------------------------------------------------------------------------
# Tests — sync chat
# ---------------------------------------------------------------------------


def test_basic_chat():
    """Agent returns streamed text for a simple response."""
    responses = [_text_response("Hello there!")]
    agent = _make_agent(responses)
    chunks = list(agent.chat("Hi"))
    assert "".join(chunks) == "Hello there!"


def test_memory_user_and_assistant_recorded():
    """User and assistant messages are recorded in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [_text_response("Hi!")]
    agent = _make_agent(responses, memory=memory)
    list(agent.chat("Hello"))

    turns = memory.conversation.turns
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hi!"


def test_tool_loop():
    """Tools are executed and results fed back for another response."""
    tool_calls: list[str] = []

    def echo_tool(text: str) -> str:
        tool_calls.append(text)
        return f"Echo: {text}"

    tool = AgentTool(
        name="echo",
        description="Echoes text back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=echo_tool,
    )

    responses = [
        _tool_use_response("tu_1", "echo", {"text": "hello"}),
        _text_response("Done!"),
    ]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Test tool"))

    assert "".join(chunks) == "Done!"
    assert tool_calls == ["hello"]


def test_tool_loop_with_text_before_tool():
    """Text before a tool call is still yielded."""
    def noop(x: str) -> str:
        return "ok"

    tool = AgentTool(
        name="noop", description="noop",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=noop,
    )

    responses = [
        _tool_use_response("tu_1", "noop", {"x": "1"}, text_before="Thinking..."),
        _text_response(" All done."),
    ]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Go"))
    assert "".join(chunks) == "Thinking... All done."


def test_max_rounds_stops_loop():
    """Agent stops after max_rounds even if model keeps calling tools."""
    call_count = [0]

    def counting_tool(x: str) -> str:
        call_count[0] += 1
        return "ok"

    tool = AgentTool(
        name="counter", description="counts",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=counting_tool,
    )

    # Always return tool_use (more responses than max_rounds)
    responses = [
        _tool_use_response(f"tu_{i}", "counter", {"x": "go"})
        for i in range(5)
    ]

    agent = _make_agent(responses, tools=[tool], max_rounds=3)
    list(agent.chat("Go"))

    # Rounds 0 and 1 execute tools; the final round's calls are not
    # executed (their results would never reach the model).
    assert call_count[0] == 2
    assert agent.last_turn is not None
    assert agent.last_turn.stopped_by == "max_rounds"


def test_unknown_tool_returns_error():
    """Calling an unknown tool returns an error message instead of crashing."""
    responses = [
        _tool_use_response("tu_1", "nonexistent", {"q": "x"}),
        _text_response("OK"),
    ]
    # No tools registered
    agent = _make_agent(responses)
    chunks = list(agent.chat("Test"))
    assert "".join(chunks) == "OK"


def test_tool_exception_handled():
    """A tool that raises an exception returns an error string."""
    def failing_tool(x: str) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    tool = AgentTool(
        name="fail", description="always fails",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        fn=failing_tool,
    )

    responses = [
        _tool_use_response("tu_1", "fail", {"x": "go"}),
        _text_response("Recovered."),
    ]
    agent = _make_agent(responses, tools=[tool])
    chunks = list(agent.chat("Go"))
    assert "".join(chunks) == "Recovered."


def test_no_memory_still_works():
    """Agent works without memory attached."""
    responses = [_text_response("Hi!")]
    agent = _make_agent(responses)
    assert agent.memory is None
    chunks = list(agent.chat("Hello"))
    assert "".join(chunks) == "Hi!"


def test_empty_response_no_memory_write():
    """Empty response text is not written to memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [_text_response("")]
    agent = _make_agent(responses, memory=memory)
    list(agent.chat("Hello"))

    turns = memory.conversation.turns
    # User message recorded, but empty assistant response is NOT
    assert len(turns) == 1
    assert turns[0].role == "user"


def test_with_tools_is_additive():
    """Calling with_tools multiple times adds tools, doesn't replace."""

    def t1() -> str:
        return "1"

    def t2() -> str:
        return "2"

    tool_a = AgentTool(
        name="a", description="a",
        input_schema={"type": "object"}, fn=t1,
    )
    tool_b = AgentTool(
        name="b", description="b",
        input_schema={"type": "object"}, fn=t2,
    )

    responses = [_text_response("ok")]
    agent = _make_agent(responses)
    agent.with_tools([tool_a])
    agent.with_tools([tool_b])

    # Access internal tools list
    assert len(agent._tools) == 2
    names = {t.name for t in agent._tools}
    assert names == {"a", "b"}


def test_pipeline_property():
    """Pipeline is accessible via property."""
    responses = [_text_response("ok")]
    agent = _make_agent(responses, system_prompt="Be helpful.")
    pipeline = agent.pipeline
    assert pipeline is not None
    assert pipeline.max_tokens == 16384


def test_memory_with_persistent_store():
    """Agent with persistent memory store records facts via tools."""
    memory = MemoryManager(
        conversation_tokens=2000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = [
        AgentTool(
            name="save_fact",
            description="Save a fact",
            input_schema={
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
            fn=lambda fact: (
                memory.add_fact(fact, tags=["auto"]) and ""  # type: ignore[func-returns-value]
            ) or "Saved",
        ),
    ]

    responses = [
        _tool_use_response("tu_1", "save_fact", {"fact": "User's name is Arthur"}),
        _text_response("Got it!"),
    ]
    agent = _make_agent(responses, tools=tools, memory=memory)
    list(agent.chat("My name is Arthur"))

    facts = memory.get_all_facts()
    assert len(facts) == 1
    assert "Arthur" in facts[0].content


# ---------------------------------------------------------------------------
# Tests — async chat (achat)
# ---------------------------------------------------------------------------


async def test_achat_basic():
    """Async chat returns text via async iteration."""
    responses = [_text_response("Hello async!")]
    agent = _make_agent(responses)
    chunks: list[str] = []
    async for chunk in agent.achat("Hi"):
        chunks.append(chunk)
    assert "".join(chunks) == "Hello async!"


async def test_achat_with_memory():
    """Async chat records user and assistant messages in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())
    responses = [_text_response("Hi!")]
    agent = _make_agent(responses, memory=memory)
    chunks: list[str] = []
    async for chunk in agent.achat("Hello"):
        chunks.append(chunk)

    turns = memory.conversation.turns
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hi!"


async def test_achat_tool_loop():
    """Async chat handles tool use loop."""
    tool_calls: list[str] = []

    def echo_tool(text: str) -> str:
        tool_calls.append(text)
        return f"Echo: {text}"

    tool = AgentTool(
        name="echo",
        description="Echoes text back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        fn=echo_tool,
    )

    responses = [
        _tool_use_response("tu_1", "echo", {"text": "hello"}),
        _text_response("Done!"),
    ]
    agent = _make_agent(responses, tools=[tool])
    chunks: list[str] = []
    async for chunk in agent.achat("Test tool"):
        chunks.append(chunk)

    assert "".join(chunks) == "Done!"
    assert tool_calls == ["hello"]


# ---------------------------------------------------------------------------
# Tests — tool call memory recording
# ---------------------------------------------------------------------------


def test_tool_calls_recorded_in_memory():
    """Tool calls are recorded as tool messages in memory."""
    memory = MemoryManager(conversation_tokens=2000, tokenizer=_Tok())

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = AgentTool(
        name="greet",
        description="Greet someone",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        fn=greet,
    )

    responses = [
        _tool_use_response("tu_1", "greet", {"name": "Alice"}),
        _text_response("Done greeting!"),
    ]
    agent = _make_agent(responses, tools=[tool], memory=memory)
    list(agent.chat("Greet Alice"))

    turns = memory.conversation.turns
    # Expect: user, tool, assistant
    assert len(turns) == 3
    assert turns[0].role == "user"
    assert turns[1].role == "tool"
    assert "[Tool: greet]" in turns[1].content
    assert "Alice" in turns[1].content
    assert "Hello, Alice!" in turns[1].content
    assert turns[2].role == "assistant"
    assert turns[2].content == "Done greeting!"
