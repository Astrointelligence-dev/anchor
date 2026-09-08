"""Tests for AgentTool, memory_tools, and rag_tools factories."""

from __future__ import annotations

import math

import pytest

from anchor.agent.tools import AgentTool, memory_tools, rag_tools
from anchor.memory.manager import MemoryManager
from anchor.models.context import ContextItem, SourceType
from anchor.retrieval.dense import DenseRetriever
from anchor.storage.json_memory_store import InMemoryEntryStore
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore


class _Tok:
    """Minimal tokenizer for tests."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _fake_embed(text: str) -> list[float]:
    """Deterministic embedding based on character sum."""
    seed = sum(ord(c) for c in text)
    raw = [math.sin(seed + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


# -- AgentTool --


def test_agent_tool_to_tool_schema_minimal():
    tool = AgentTool(
        name="my_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {}},
        fn=lambda: "ok",
    )
    schema = tool.to_tool_schema()
    assert schema.name == "my_tool"
    assert schema.description == "A test tool"
    assert "type" in schema.input_schema


def test_agent_tool_is_frozen():
    tool = AgentTool(
        name="t", description="d",
        input_schema={"type": "object"}, fn=lambda: "ok",
    )
    with pytest.raises((AttributeError, Exception)):
        tool.name = "x"  # type: ignore[misc]


# -- memory_tools --


def test_memory_tools_save_fact():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    save_tool = next(t for t in tools if t.name == "save_fact")

    result = save_tool.fn(fact="User's name is Arthur")
    assert "Saved" in result
    assert "Arthur" in result

    facts = memory.get_all_facts()
    assert len(facts) == 1
    assert "Arthur" in facts[0].content
    assert "auto" in facts[0].tags


def test_memory_tools_search_facts():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    memory.add_fact("User's name is Arthur")
    memory.add_fact("User prefers Python over JavaScript")

    tools = memory_tools(memory)
    search_tool = next(t for t in tools if t.name == "search_facts")

    result = search_tool.fn(query="name")
    # InMemoryEntryStore.search does substring matching
    assert "Arthur" in result


def test_memory_tools_search_facts_empty():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    search_tool = next(t for t in tools if t.name == "search_facts")

    result = search_tool.fn(query="anything")
    assert "No relevant facts found" in result


def test_memory_tools_save_fact_dedup():
    """Saving the same fact twice returns the existing entry (no duplicate)."""
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    save_tool = next(t for t in tools if t.name == "save_fact")

    result1 = save_tool.fn(fact="User's name is Arthur")
    result2 = save_tool.fn(fact="User's name is Arthur")

    # Both should succeed, but only one fact should exist
    assert "Saved" in result1
    assert "Saved" in result2
    facts = memory.get_all_facts()
    assert len(facts) == 1


def test_memory_tools_search_facts_returns_ids():
    """search_facts returns fact IDs for use with update/delete."""
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    entry = memory.add_fact("User's name is Arthur")
    tools = memory_tools(memory)
    search_tool = next(t for t in tools if t.name == "search_facts")

    result = search_tool.fn(query="name")
    assert entry.id[:8] in result


def test_memory_tools_update_fact():
    """update_fact changes content of an existing fact."""
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    entry = memory.add_fact("User is 23 years old")
    tools = memory_tools(memory)
    update_tool = next(t for t in tools if t.name == "update_fact")

    result = update_tool.fn(fact_id=entry.id, content="User is 24 years old")
    assert "Updated" in result

    facts = memory.get_all_facts()
    assert len(facts) == 1
    assert "24" in facts[0].content


def test_memory_tools_update_fact_not_found():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    update_tool = next(t for t in tools if t.name == "update_fact")

    result = update_tool.fn(fact_id="nonexistent-id", content="new content")
    assert "not found" in result


def test_memory_tools_delete_fact():
    """delete_fact removes an existing fact."""
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    entry = memory.add_fact("User's name is Arthur")
    tools = memory_tools(memory)
    delete_tool = next(t for t in tools if t.name == "delete_fact")

    result = delete_tool.fn(fact_id=entry.id)
    assert "Deleted" in result
    assert len(memory.get_all_facts()) == 0


def test_memory_tools_delete_fact_not_found():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    delete_tool = next(t for t in tools if t.name == "delete_fact")

    result = delete_tool.fn(fact_id="nonexistent-id")
    assert "not found" in result


def test_memory_tools_returns_four_tools():
    memory = MemoryManager(
        conversation_tokens=1000, tokenizer=_Tok(),
        persistent_store=InMemoryEntryStore(),
    )
    tools = memory_tools(memory)
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"save_fact", "search_facts", "update_fact", "delete_fact"}


# -- rag_tools --


def test_rag_tools_search_docs():
    tok = _Tok()
    retriever = DenseRetriever(
        vector_store=InMemoryVectorStore(),
        context_store=InMemoryContextStore(),
        embed_fn=_fake_embed,
        tokenizer=tok,
    )
    items = [
        ContextItem(
            content="The pipeline assembles context from multiple sources.",
            source=SourceType.RETRIEVAL, score=0.0, priority=5,
            token_count=tok.count_tokens("The pipeline assembles context from multiple sources."),
            metadata={"section": "Pipeline Overview"},
        ),
        ContextItem(
            content="Memory manager coordinates conversation and persistent facts.",
            source=SourceType.RETRIEVAL, score=0.0, priority=5,
            token_count=tok.count_tokens("Memory manager coordinates conversation and facts."),
            metadata={"section": "Memory"},
        ),
    ]
    retriever.index(items)

    tools = rag_tools(retriever, embed_fn=_fake_embed)
    assert len(tools) == 1
    search_tool = tools[0]
    assert search_tool.name == "search_docs"

    result = search_tool.fn(query="pipeline context")
    # Should return something (content from indexed items)
    assert len(result) > 0


def test_rag_tools_no_results():
    tok = _Tok()
    retriever = DenseRetriever(
        vector_store=InMemoryVectorStore(),
        context_store=InMemoryContextStore(),
        embed_fn=_fake_embed,
        tokenizer=tok,
    )
    # Empty index — no items
    tools = rag_tools(retriever, embed_fn=_fake_embed)
    result = tools[0].fn(query="anything")
    assert "No relevant documents found" in result


def test_rag_tools_includes_section_label():
    tok = _Tok()
    retriever = DenseRetriever(
        vector_store=InMemoryVectorStore(),
        context_store=InMemoryContextStore(),
        embed_fn=_fake_embed,
        tokenizer=tok,
    )
    items = [
        ContextItem(
            content="Token budgets control how much context fits.",
            source=SourceType.RETRIEVAL, score=0.0, priority=5,
            token_count=8,
            metadata={"section": "Token Budgets"},
        ),
    ]
    retriever.index(items)

    tools = rag_tools(retriever, embed_fn=_fake_embed)
    result = tools[0].fn(query="token budgets")
    assert "Token Budgets" in result


# -- validate_input --


def test_validate_input_valid():
    """Correct input passes validation."""
    tool = AgentTool(
        name="greet",
        description="Say hello",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        },
        fn=lambda name, age=0: f"Hello {name}",
    )
    valid, err = tool.validate_input({"name": "Alice", "age": 30})
    assert valid is True
    assert err == ""


def test_validate_input_missing_required():
    """Missing required field is detected."""
    tool = AgentTool(
        name="greet",
        description="Say hello",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        fn=lambda name: f"Hello {name}",
    )
    valid, err = tool.validate_input({})
    assert valid is False
    assert "name" in err
    assert "Missing required field" in err


def test_validate_input_wrong_type():
    """Type mismatch is detected."""
    tool = AgentTool(
        name="calc",
        description="Calculate",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        fn=lambda x: str(x),
    )
    valid, err = tool.validate_input({"x": "not_a_number"})
    assert valid is False
    assert "integer" in err
    assert "str" in err


def test_validate_input_extra_fields():
    """Extra fields are allowed (lenient mode)."""
    tool = AgentTool(
        name="greet",
        description="Say hello",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        fn=lambda name, **kw: f"Hello {name}",
    )
    valid, err = tool.validate_input({"name": "Alice", "extra": "ignored"})
    assert valid is True
    assert err == ""


def test_agent_tool_to_tool_schema():
    """Convert AgentTool to provider-agnostic ToolSchema."""
    from anchor.llm.models import ToolSchema

    tool = AgentTool(
        name="get_weather",
        description="Get weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        fn=lambda **kwargs: "sunny",
    )
    schema = tool.to_tool_schema()
    assert isinstance(schema, ToolSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get weather"
    assert schema.input_schema == tool.input_schema
