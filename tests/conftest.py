"""Shared fixtures for anchor tests."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from anchor.memory.manager import MemoryManager
from anchor.models.context import ContextItem
from anchor.models.memory import MemoryEntry, MemoryType
from anchor.models.query import QueryBundle
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore


def make_embedding(seed: int, dim: int = 128) -> list[float]:
    """Create a deterministic fake embedding using math.sin.

    Each seed produces a unique but reproducible unit-ish vector.
    The same seed always returns the same embedding, and different
    seeds produce different (approximately orthogonal) embeddings
    when dim is large enough relative to the number of seeds.
    """
    raw = [math.sin(seed * 1000 + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        return raw
    return [x / norm for x in raw]


class FakeTokenizer:
    """A simple tokenizer that splits on whitespace for testing.

    Satisfies the Tokenizer protocol without requiring tiktoken's
    network-downloaded encoding data.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens by splitting on whitespace."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token limit."""
        if max_tokens <= 0:
            return ""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


class FakeRetriever:
    """Fake retriever that returns pre-configured results for testing.

    Used across pipeline and retrieval tests as a stand-in for real retrievers.
    """

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._items[:top_k]


def make_memory_manager(conversation_tokens: int = 2000) -> MemoryManager:
    """Create a MemoryManager with FakeTokenizer for testing.

    Shared helper for both pipeline and memory manager tests.
    """
    return MemoryManager(conversation_tokens=conversation_tokens, tokenizer=FakeTokenizer())


def make_memory_entry(
    *,
    entry_id: str = "e1",
    content: str = "some memory content",
    relevance_score: float = 0.5,
    user_id: str | None = None,
    session_id: str | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    tags: list[str] | None = None,
    last_accessed: datetime | None = None,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryEntry:
    """Build a MemoryEntry with sensible test defaults.

    Shared helper extracted from test_entry_store, test_json_file_store,
    and test_memory_retriever to avoid duplication.
    """
    kwargs: dict = {
        "id": entry_id,
        "content": content,
        "relevance_score": relevance_score,
        "memory_type": memory_type,
    }
    if user_id is not None:
        kwargs["user_id"] = user_id
    if session_id is not None:
        kwargs["session_id"] = session_id
    if tags is not None:
        kwargs["tags"] = tags
    if last_accessed is not None:
        kwargs["last_accessed"] = last_accessed
    if created_at is not None:
        kwargs["created_at"] = created_at
    if expires_at is not None:
        kwargs["expires_at"] = expires_at
    return MemoryEntry(**kwargs)


@pytest.fixture
def counter() -> FakeTokenizer:
    """Return a FakeTokenizer instance for testing."""
    return FakeTokenizer()


@pytest.fixture
def context_store() -> InMemoryContextStore:
    """Return a fresh InMemoryContextStore."""
    return InMemoryContextStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Return a fresh InMemoryVectorStore."""
    return InMemoryVectorStore()




class FakeLLM:
    """A provider whose ``invoke`` answers with *content* (a str, or a list of str in order).

    Records every prompt in ``prompts``. Shared by the LLM-backed memory and
    graph tests.
    """

    def __init__(self, content: str | list[str] | None) -> None:
        self._answers = content if isinstance(content, list) else [content]
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        from types import SimpleNamespace

        self.prompts.append(messages[0].content)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return SimpleNamespace(content=answer)
