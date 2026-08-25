"""Regression tests for the Phase 0 correctness fixes.

Each test here pins a bug documented in
docs/research/2026-08-25-retrieval-stack-analysis.md §0:

1. Rerankers silently discarded an explicit ``top_k=10`` (sentinel bug).
2. Raw scores were destroyed by clamping/normalization (no thresholding possible).
3. RRF fusion was triplicated; hybrid retrievers now delegate to ``rrf_fuse``.
4. RecursiveCharacterChunker applied overlap at every recursion level.
5. SQLite vector store unpacked blobs using the *query* dimension.
6. Sync/async divergence: all-fail behavior and Cohere callback shape.
"""

from __future__ import annotations

import asyncio
import itertools
from unittest.mock import patch

import pytest

from anchor.exceptions import RetrieverError
from anchor.ingestion.chunkers import RecursiveCharacterChunker
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.retrieval.async_reranker import AsyncCohereReranker
from anchor.retrieval.async_retriever import AsyncDenseRetriever, AsyncHybridRetriever
from anchor.retrieval.hybrid import HybridRetriever
from anchor.retrieval.rerankers import (
    CohereReranker,
    CrossEncoderReranker,
    RerankerPipeline,
    RoundRobinReranker,
)
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import FakeTokenizer
from tests.test_retrieval.conftest import make_dense_retriever, make_sparse_retriever


def _items(n: int) -> list[ContextItem]:
    return [
        ContextItem(id=f"item-{i}", content=f"document number {i}", source=SourceType.RETRIEVAL)
        for i in range(n)
    ]


_QUERY = QueryBundle(query_str="test query")


class TestTopKSentinel:
    """Explicit top_k must be honored — including the literal value 10."""

    def test_explicit_top_k_10_is_honored(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=3)
        result = reranker.rerank(_QUERY, _items(12), top_k=10)
        assert len(result) == 10  # pre-fix: sentinel discarded 10, returned 3

    def test_explicit_top_k_overrides_constructor(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=3)
        assert len(reranker.rerank(_QUERY, _items(12), top_k=7)) == 7

    def test_none_falls_back_to_constructor(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=3)
        assert len(reranker.rerank(_QUERY, _items(12))) == 3

    def test_cohere_explicit_top_k_10(self) -> None:
        def rerank_fn(q: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(docs)))]

        reranker = CohereReranker(rerank_fn=rerank_fn, top_k=3)
        assert len(reranker.rerank(_QUERY, _items(12), top_k=10)) == 10

    def test_round_robin_explicit_top_k_10(self) -> None:
        reranker = RoundRobinReranker(top_k=3)
        assert len(reranker.rerank(_QUERY, _items(12), top_k=10)) == 10

    def test_pipeline_explicit_top_k_10(self) -> None:
        pipeline = RerankerPipeline(
            rerankers=[CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=3)],
            top_k=3,
        )
        assert len(pipeline.rerank(_QUERY, _items(12), top_k=10)) == 10


class TestRawScorePreserved:
    """The raw (unclamped, unnormalized) score must survive in metadata."""

    def test_cross_encoder_negative_score_preserved(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: -2.5)
        result = reranker.rerank(_QUERY, _items(2))
        assert result[0].score == 0.0  # field stays within [0, 1]
        assert result[0].metadata["raw_score"] == -2.5

    def test_cross_encoder_above_one_preserved(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 7.3)
        result = reranker.rerank(_QUERY, _items(1))
        assert result[0].score == 1.0
        assert result[0].metadata["raw_score"] == 7.3

    def test_dense_raw_cosine_preserved(self) -> None:
        vs, cs = InMemoryVectorStore(), InMemoryContextStore()
        retriever = make_dense_retriever(vs, cs, embed_fn=lambda text: [1.0, 0.0])
        item = ContextItem(id="a", content="anti-correlated", source=SourceType.RETRIEVAL)
        cs.add(item)
        vs.add_embedding("a", [-1.0, 0.0])
        result = retriever.retrieve(_QUERY, top_k=1)
        assert result[0].score == 0.0  # clamped for the field contract
        assert result[0].metadata["raw_score"] == pytest.approx(-1.0)

    def test_sparse_raw_bm25_preserved(self) -> None:
        pytest.importorskip("rank_bm25")
        retriever = make_sparse_retriever()
        # 3+ docs: with N=2/df=1 BM25Okapi's idf is exactly 0 and everything
        # scores 0, which the <= 0 filter drops.
        items = [
            ContextItem(id="a", content="cats and dogs", source=SourceType.RETRIEVAL),
            ContextItem(id="b", content="quantum physics", source=SourceType.RETRIEVAL),
            ContextItem(id="c", content="ancient history", source=SourceType.RETRIEVAL),
        ]
        retriever.index(items)
        result = retriever.retrieve(QueryBundle(query_str="cats"), top_k=2)
        assert result[0].score == 1.0  # max-normalized field
        assert result[0].metadata["raw_score"] > 0.0  # the actual BM25 score

    def test_dense_min_score_filters(self) -> None:
        vs, cs = InMemoryVectorStore(), InMemoryContextStore()
        with patch(
            "anchor.retrieval.dense.get_default_counter", return_value=FakeTokenizer()
        ):
            from anchor.retrieval.dense import DenseRetriever

            retriever = DenseRetriever(
                vs, cs, embed_fn=lambda text: [1.0, 0.0], min_score=0.5
            )
        for item_id, emb in [("close", [0.9, 0.1]), ("far", [-1.0, 0.0])]:
            cs.add(ContextItem(id=item_id, content=item_id, source=SourceType.RETRIEVAL))
            vs.add_embedding(item_id, emb)
        result = retriever.retrieve(_QUERY, top_k=10)
        assert [item.id for item in result] == ["close"]


class TestRRFDeduplication:
    """Hybrid retrievers delegate to the canonical rrf_fuse."""

    class _StaticRetriever:
        def __init__(self, items: list[ContextItem]) -> None:
            self._items = items

        def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
            return self._items[:top_k]

    def test_hybrid_uses_rrf_fuse_labelling(self) -> None:
        items = _items(3)
        hybrid = HybridRetriever(
            [self._StaticRetriever(items), self._StaticRetriever(list(reversed(items)))]
        )
        result = hybrid.retrieve(_QUERY, top_k=3)
        assert result, "fusion returned nothing"
        for item in result:
            assert item.metadata["retrieval_method"] == "hybrid_rrf"
            assert "rrf_raw_score" in item.metadata

    def test_hybrid_raw_rrf_scores_are_true_rrf(self) -> None:
        items = _items(2)
        hybrid = HybridRetriever([self._StaticRetriever(items)])
        result = hybrid.retrieve(_QUERY, top_k=2)
        # single list, weight 1.0, k=60: rank 1 -> 1/61, rank 2 -> 1/62
        raws = sorted((i.metadata["rrf_raw_score"] for i in result), reverse=True)
        assert raws[0] == pytest.approx(1 / 61)
        assert raws[1] == pytest.approx(1 / 62)


class TestAsyncSyncAlignment:
    def test_async_hybrid_raises_when_all_fail(self) -> None:
        class _Failing:
            async def aretrieve(
                self, query: QueryBundle, top_k: int = 10
            ) -> list[ContextItem]:
                msg = "boom"
                raise RuntimeError(msg)

        hybrid = AsyncHybridRetriever([_Failing()])  # type: ignore[list-item]
        with pytest.raises(RetrieverError):
            asyncio.run(hybrid.aretrieve(_QUERY))

    def test_async_cohere_matches_sync_callback_shape(self) -> None:
        async def rerank_fn(
            q: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return [(1, 0.9), (0, 0.4)]

        reranker = AsyncCohereReranker(rerank_fn=rerank_fn)
        result = asyncio.run(reranker.arerank(_QUERY, _items(2)))
        assert [item.id for item in result] == ["item-1", "item-0"]
        assert result[0].score == pytest.approx(0.9)
        assert result[0].metadata["raw_score"] == pytest.approx(0.9)

    def test_async_dense_raw_score_preserved(self) -> None:
        async def embed(text: str) -> list[float]:
            return [1.0, 0.0]

        retriever = AsyncDenseRetriever(embed_fn=embed)
        item = ContextItem(
            id="a",
            content="doc",
            source=SourceType.RETRIEVAL,
            metadata={"embedding": [-1.0, 0.0]},
        )
        retriever.index([item])
        result = asyncio.run(retriever.aretrieve(_QUERY, top_k=1))
        assert result[0].score == 0.0
        assert result[0].metadata["raw_score"] == pytest.approx(-1.0)


class TestOverlapSingleApplication:
    def test_split_produces_no_overlap(self) -> None:
        """_split (recursive) must not apply overlap; chunk() applies it once."""
        chunker = RecursiveCharacterChunker(
            chunk_size=8, overlap=3, tokenizer=FakeTokenizer()
        )
        text = "\n\n".join(
            " ".join(f"w{p}{i}" for i in range(6)) for p in range(4)
        )
        raw_chunks = chunker._split(text, 0)
        seen: set[str] = set()
        for chunk in raw_chunks:
            words = chunk.split()
            assert not seen.intersection(words), "overlap leaked into _split output"
            seen.update(words)

    def test_chunk_applies_overlap_exactly_once(self) -> None:
        chunker = RecursiveCharacterChunker(
            chunk_size=8, overlap=2, tokenizer=FakeTokenizer()
        )
        text = "\n\n".join(
            " ".join(f"w{p}{i}" for i in range(6)) for p in range(4)
        )
        chunks = chunker.chunk(text)
        for prev, cur in itertools.pairwise(chunks):
            cur_words = cur.split()
            prev_words = prev.split()
            # any overlap prefix must appear exactly once in the current chunk
            for word in prev_words:
                assert cur_words.count(word) <= 1


class TestSqliteDimensionValidation:
    @staticmethod
    def _make_store(tmp_path):
        from anchor.storage.sqlite import (
            SqliteConnectionManager,
            SqliteVectorStore,
            ensure_tables,
        )

        manager = SqliteConnectionManager(tmp_path / "vec.db")
        ensure_tables(manager.get_connection())
        return SqliteVectorStore(manager)

    def test_dimension_mismatch_raises(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        store.add_embedding("a", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimension"):
            store.search([1.0, 2.0], top_k=1)

    def test_matching_dimension_works(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        store.add_embedding("a", [1.0, 0.0])
        store.add_embedding("b", [0.0, 1.0])
        results = store.search([1.0, 0.0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)


class TestScoreRerankerRemoved:
    def test_not_exported(self) -> None:
        import anchor
        import anchor.retrieval

        assert not hasattr(anchor, "ScoreReranker")
        assert not hasattr(anchor.retrieval, "ScoreReranker")
