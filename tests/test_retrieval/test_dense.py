"""Tests for anchor.retrieval.dense."""

from __future__ import annotations

import pytest

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import make_embedding
from tests.test_retrieval.conftest import make_dense_retriever


def _make_items() -> list[ContextItem]:
    """Create items for dense retrieval tests."""
    return [
        ContextItem(
            id=f"dense-{i}",
            content=text,
            source=SourceType.RETRIEVAL,
            token_count=10,
            priority=5,
        )
        for i, text in enumerate(
            [
                "Python programming language",
                "Java programming language",
                "Machine learning with neural networks",
                "Context engineering for LLMs",
                "Cooking pasta recipes",
            ]
        )
    ]


def _embed_fn(text: str) -> list[float]:
    """Deterministic embedding function: hashes text to a seed."""
    seed = sum(ord(c) for c in text) % 10000
    return make_embedding(seed)


class TestDenseRetrieverIndex:
    """DenseRetriever.index() stores items in both vector and context stores."""

    def test_index_returns_count(self) -> None:
        retriever = make_dense_retriever(embed_fn=_embed_fn)
        items = _make_items()
        count = retriever.index(items)
        assert count == 5

    def test_index_stores_in_context_store(self) -> None:
        cs = InMemoryContextStore()
        retriever = make_dense_retriever(cs=cs, embed_fn=_embed_fn)
        items = _make_items()
        retriever.index(items)
        assert cs.get("dense-0") is not None
        assert cs.get("dense-4") is not None

    def test_index_stores_in_vector_store(self) -> None:
        vs = InMemoryVectorStore()
        retriever = make_dense_retriever(vs=vs, embed_fn=_embed_fn)
        items = _make_items()
        retriever.index(items)
        # Vector store should have embeddings for all items
        results = vs.search(_embed_fn("anything"), top_k=10)
        assert len(results) == 5

    def test_index_without_embed_fn_raises(self) -> None:
        retriever = make_dense_retriever(embed_fn=None)
        with pytest.raises(RetrieverError, match="must be provided to index"):
            retriever.index(_make_items())


class TestDenseRetrieverRetrieve:
    """DenseRetriever.retrieve() returns scored ContextItems."""

    def test_retrieve_returns_scored_items(self) -> None:
        retriever = make_dense_retriever(embed_fn=_embed_fn)
        retriever.index(_make_items())

        query = QueryBundle(query_str="Python programming language")
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3
        # All results should have retrieval source
        for item in results:
            assert item.source == SourceType.RETRIEVAL
            assert 0.0 <= item.score <= 1.0
            assert item.metadata.get("retrieval_method") == "dense"

    def test_retrieve_without_embed_fn_and_no_query_embedding_raises(self) -> None:
        vs = InMemoryVectorStore()
        cs = InMemoryContextStore()
        # First index with an embed_fn
        retriever_for_index = make_dense_retriever(vs=vs, cs=cs, embed_fn=_embed_fn)
        retriever_for_index.index(_make_items())

        # Then create a retriever without embed_fn
        retriever = make_dense_retriever(vs=vs, cs=cs, embed_fn=None)
        query = QueryBundle(query_str="test")
        with pytest.raises(RetrieverError, match=r"Either provide query\.embedding"):
            retriever.retrieve(query)

    def test_retrieve_with_query_embedding_works(self) -> None:
        vs = InMemoryVectorStore()
        cs = InMemoryContextStore()
        retriever_for_index = make_dense_retriever(vs=vs, cs=cs, embed_fn=_embed_fn)
        retriever_for_index.index(_make_items())

        # Create retriever without embed_fn but provide embedding in query
        retriever = make_dense_retriever(vs=vs, cs=cs, embed_fn=None)
        query_embedding = _embed_fn("Python programming")
        query = QueryBundle(query_str="Python programming", embedding=query_embedding)
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3

    def test_retrieve_top_k_limits_output(self) -> None:
        retriever = make_dense_retriever(embed_fn=_embed_fn)
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2

    def test_retrieve_scores_are_normalized(self) -> None:
        retriever = make_dense_retriever(embed_fn=_embed_fn)
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming")
        results = retriever.retrieve(query, top_k=5)
        for item in results:
            assert 0.0 <= item.score <= 1.0
