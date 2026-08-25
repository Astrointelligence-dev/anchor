"""Tests for anchor.retrieval.sparse."""

from __future__ import annotations

import pytest

pytest.importorskip("rank_bm25", reason="rank-bm25 required for SparseRetriever tests")

from anchor.exceptions import RetrieverError  # noqa: E402
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from tests.test_retrieval.conftest import make_sparse_retriever


def _make_items() -> list[ContextItem]:
    """Create items for sparse retrieval tests."""
    texts = [
        "Python is a high-level programming language",
        "Java is an object-oriented programming language",
        "Machine learning uses neural networks for training",
        "Context engineering helps build better LLM prompts",
        "Cooking pasta requires boiling water and noodles",
    ]
    return [
        ContextItem(
            id=f"sparse-{i}",
            content=text,
            source=SourceType.RETRIEVAL,
            token_count=10,
            priority=5,
        )
        for i, text in enumerate(texts)
    ]


class TestSparseRetrieverIndex:
    """SparseRetriever.index() builds BM25 index."""

    def test_index_returns_count(self) -> None:
        retriever = make_sparse_retriever()
        items = _make_items()
        count = retriever.index(items)
        assert count == 5

    def test_index_builds_bm25_object(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())
        # bm25s backend when installed, rank_bm25 otherwise
        assert retriever._bm25s is not None or retriever._bm25 is not None

    def test_index_stores_items(self) -> None:
        retriever = make_sparse_retriever()
        items = _make_items()
        retriever.index(items)
        assert len(retriever._items) == 5


class TestSparseRetrieverRetrieve:
    """SparseRetriever.retrieve() returns relevant results."""

    def test_retrieve_returns_relevant_results(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming language")
        results = retriever.retrieve(query, top_k=3)
        assert len(results) > 0
        # The top results should be about programming
        contents = [item.content for item in results]
        assert any("programming" in c.lower() for c in contents)

    def test_retrieve_before_index_raises_runtime_error(self) -> None:
        retriever = make_sparse_retriever()
        query = QueryBundle(query_str="test")
        with pytest.raises(RetrieverError, match="Must call index"):
            retriever.retrieve(query)

    def test_score_normalization_to_zero_one(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming language")
        results = retriever.retrieve(query, top_k=5)
        for item in results:
            assert 0.0 <= item.score <= 1.0

    def test_top_result_score_is_one(self) -> None:
        """The top BM25 result should be normalized to 1.0."""
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming language")
        results = retriever.retrieve(query, top_k=5)
        if results:
            assert results[0].score == pytest.approx(1.0)

    def test_retrieve_irrelevant_query_returns_fewer_results(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        # Query with terms not in any document
        query = QueryBundle(query_str="quantum physics relativity")
        results = retriever.retrieve(query, top_k=5)
        # BM25 should return 0 results for completely irrelevant query
        assert len(results) == 0

    def test_retrieval_method_metadata(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming")
        results = retriever.retrieve(query, top_k=3)
        for item in results:
            assert item.metadata.get("retrieval_method") == "sparse_bm25"
            assert item.source == SourceType.RETRIEVAL

    def test_custom_tokenize_fn(self) -> None:
        """Custom tokenize function is used."""

        def custom_tokenize(text: str) -> list[str]:
            # Only keep words > 3 chars
            return [w.lower() for w in text.split() if len(w) > 3]

        retriever = make_sparse_retriever(tokenize_fn=custom_tokenize)
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming language")
        results = retriever.retrieve(query, top_k=5)
        # Should still find results, just with different tokenization
        assert len(results) > 0

    def test_top_k_limits_results(self) -> None:
        retriever = make_sparse_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming language python java machine")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) <= 2
