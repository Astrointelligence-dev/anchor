"""Tests for the rag_skill factory."""

from __future__ import annotations

import math

from anchor.agent.skills.rag import rag_skill
from anchor.models.context import ContextItem, SourceType
from anchor.retrieval.dense import DenseRetriever
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore


def _fake_embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text)
    raw = [math.sin(seed + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


class _Tok:
    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class TestRagSkill:
    def test_creates_valid_skill(self) -> None:
        ctx = InMemoryContextStore()
        vec = InMemoryVectorStore()
        retriever = DenseRetriever(
            context_store=ctx, vector_store=vec, embed_fn=_fake_embed, tokenizer=_Tok()
        )
        skill = rag_skill(retriever, _fake_embed)
        assert skill.name == "rag"
        assert skill.activation == "on_demand"
        assert len(skill.tools) == 1

    def test_tool_name(self) -> None:
        ctx = InMemoryContextStore()
        vec = InMemoryVectorStore()
        retriever = DenseRetriever(
            context_store=ctx, vector_store=vec, embed_fn=_fake_embed, tokenizer=_Tok()
        )
        skill = rag_skill(retriever, _fake_embed)
        assert skill.tools[0].name == "search_docs"

    def test_has_instructions(self) -> None:
        ctx = InMemoryContextStore()
        vec = InMemoryVectorStore()
        retriever = DenseRetriever(
            context_store=ctx, vector_store=vec, embed_fn=_fake_embed, tokenizer=_Tok()
        )
        skill = rag_skill(retriever, _fake_embed)
        assert skill.instructions != ""
        assert "search_docs" in skill.instructions

    def test_has_tags(self) -> None:
        ctx = InMemoryContextStore()
        vec = InMemoryVectorStore()
        retriever = DenseRetriever(
            context_store=ctx, vector_store=vec, embed_fn=_fake_embed, tokenizer=_Tok()
        )
        skill = rag_skill(retriever, _fake_embed)
        assert "retrieval" in skill.tags

    def test_tool_is_functional(self) -> None:
        ctx = InMemoryContextStore()
        vec = InMemoryVectorStore()
        tok = _Tok()
        retriever = DenseRetriever(
            context_store=ctx, vector_store=vec,
            embed_fn=_fake_embed, tokenizer=tok,
        )

        item = ContextItem(
            content="Python is a programming language.",
            source=SourceType.RETRIEVAL, score=0.0, priority=5,
            token_count=tok.count_tokens("Python is a programming language."),
            metadata={"section": "overview"},
        )
        retriever.index([item])

        skill = rag_skill(retriever, _fake_embed)
        search = skill.tools[0]
        result = search.fn(query="Python")
        assert "Python" in result
