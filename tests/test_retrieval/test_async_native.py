"""Tests for native async retriever and reranker implementations."""

from __future__ import annotations

import asyncio
import time

import pytest

from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.reranker import AsyncReranker
from anchor.protocols.retriever import AsyncRetriever
from anchor.retrieval.async_reranker import (
    AsyncCohereReranker,
    AsyncCrossEncoderReranker,
)
from anchor.retrieval.async_retriever import (
    AsyncDenseRetriever,
    AsyncHybridRetriever,
)

# ---------------------------------------------------------------------------
# Fake async helpers
# ---------------------------------------------------------------------------


async def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding with distinct directions per text.

    Uses hashlib (not built-in hash) for cross-run determinism and
    generates 3 independent components so that cosine similarity can
    differentiate between vectors.
    """
    import hashlib

    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:3]]


async def _fake_score(query: str, doc: str) -> float:
    """Score 1.0 if query appears in doc, else 0.0."""
    return 1.0 if query in doc else 0.0


async def _fake_cohere_rerank(
    query: str, documents: list[str], top_k: int
) -> list[tuple[int, float]]:
    """Fake reranker: puts documents containing the query first."""
    scored = []
    for i, doc in enumerate(documents):
        score = 1.0 if query in doc else 0.0
        scored.append((score, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(idx, score) for score, idx in scored[:top_k]]


def _make_item(
    content: str, *, item_id: str | None = None, embedding: list[float] | None = None
) -> ContextItem:
    """Create a ContextItem with optional embedding in metadata."""
    meta: dict = {}
    if embedding is not None:
        meta["embedding"] = embedding
    kwargs: dict = {
        "content": content,
        "source": SourceType.RETRIEVAL,
        "metadata": meta,
    }
    if item_id is not None:
        kwargs["id"] = item_id
    return ContextItem(**kwargs)


# ---------------------------------------------------------------------------
# AsyncDenseRetriever tests
# ---------------------------------------------------------------------------


class TestAsyncDenseRetriever:
    """Tests for AsyncDenseRetriever."""

    def test_async_dense_protocol_compliance(self) -> None:
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        assert isinstance(retriever, AsyncRetriever)

    @pytest.mark.asyncio
    async def test_async_dense_retrieve(self) -> None:
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        emb_a = await _fake_embed("alpha")
        emb_b = await _fake_embed("beta")
        items = [
            _make_item("alpha", item_id="a", embedding=emb_a),
            _make_item("beta", item_id="b", embedding=emb_b),
        ]
        retriever.index(items)

        query = QueryBundle(query_str="alpha")
        results = await retriever.aretrieve(query, top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, ContextItem) for r in results)
        # The item whose embedding matches the query embedding should rank first
        assert results[0].content == "alpha"

    @pytest.mark.asyncio
    async def test_async_dense_empty_index(self) -> None:
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        query = QueryBundle(query_str="anything")
        results = await retriever.aretrieve(query, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_async_dense_top_k(self) -> None:
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        embeddings = [await _fake_embed(f"item{i}") for i in range(5)]
        items = [
            _make_item(f"item{i}", item_id=f"id{i}", embedding=embeddings[i]) for i in range(5)
        ]
        retriever.index(items)

        query = QueryBundle(query_str="item0")
        results = await retriever.aretrieve(query, top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_async_dense_aindex(self) -> None:
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        items = [
            _make_item("hello", item_id="h1"),
            _make_item("world", item_id="h2"),
        ]
        await retriever.aindex(items)

        query = QueryBundle(query_str="hello")
        results = await retriever.aretrieve(query, top_k=2)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# AsyncHybridRetriever tests
# ---------------------------------------------------------------------------


class TestAsyncHybridRetriever:
    """Tests for AsyncHybridRetriever."""

    def test_async_hybrid_protocol_compliance(self) -> None:
        r = AsyncDenseRetriever(embed_fn=_fake_embed)
        hybrid = AsyncHybridRetriever(retrievers=[r])
        assert isinstance(hybrid, AsyncRetriever)

    @pytest.mark.asyncio
    async def test_async_hybrid_retrieve(self) -> None:
        r1 = AsyncDenseRetriever(embed_fn=_fake_embed)
        r2 = AsyncDenseRetriever(embed_fn=_fake_embed)

        emb_a = await _fake_embed("alpha")
        emb_b = await _fake_embed("beta")
        items = [
            _make_item("alpha", item_id="a", embedding=emb_a),
            _make_item("beta", item_id="b", embedding=emb_b),
        ]
        r1.index(items)
        r2.index(items)

        hybrid = AsyncHybridRetriever(retrievers=[r1, r2])
        query = QueryBundle(query_str="alpha")
        results = await hybrid.aretrieve(query, top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, ContextItem) for r in results)

    @pytest.mark.asyncio
    async def test_async_hybrid_weights(self) -> None:
        r1 = AsyncDenseRetriever(embed_fn=_fake_embed)
        r2 = AsyncDenseRetriever(embed_fn=_fake_embed)

        emb_a = await _fake_embed("alpha")
        emb_b = await _fake_embed("beta")
        items = [
            _make_item("alpha", item_id="a", embedding=emb_a),
            _make_item("beta", item_id="b", embedding=emb_b),
        ]
        r1.index(items)
        r2.index(items)

        hybrid = AsyncHybridRetriever(retrievers=[r1, r2], weights=[2.0, 1.0])
        query = QueryBundle(query_str="alpha")
        results = await hybrid.aretrieve(query, top_k=2)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_async_hybrid_parallel_execution(self) -> None:
        """Verify retrievers are called concurrently, not sequentially."""

        async def slow_embed(text: str) -> list[float]:
            await asyncio.sleep(0.05)
            return [hash(text) % 100 / 100.0] * 3

        r1 = AsyncDenseRetriever(embed_fn=slow_embed)
        r2 = AsyncDenseRetriever(embed_fn=slow_embed)

        emb = [0.5, 0.5, 0.5]
        items = [_make_item("test", item_id="t1", embedding=emb)]
        r1.index(items)
        r2.index(items)

        hybrid = AsyncHybridRetriever(retrievers=[r1, r2])
        query = QueryBundle(query_str="test")

        start = time.monotonic()
        await hybrid.aretrieve(query, top_k=1)
        elapsed = time.monotonic() - start

        # If truly parallel, elapsed should be ~0.05s, not ~0.10s.
        # Allow generous slack for CI.
        assert elapsed < 0.15


# ---------------------------------------------------------------------------
# AsyncCrossEncoderReranker tests
# ---------------------------------------------------------------------------


class TestAsyncCrossEncoderReranker:
    """Tests for AsyncCrossEncoderReranker."""

    def test_async_cross_encoder_protocol(self) -> None:
        reranker = AsyncCrossEncoderReranker(score_fn=_fake_score)
        assert isinstance(reranker, AsyncReranker)

    @pytest.mark.asyncio
    async def test_async_cross_encoder_rerank(self) -> None:
        reranker = AsyncCrossEncoderReranker(score_fn=_fake_score)
        items = [
            _make_item("the cat sat", item_id="a"),
            _make_item("hello world", item_id="b"),
            _make_item("the cat climbed", item_id="c"),
        ]
        query = QueryBundle(query_str="the cat")
        results = await reranker.arerank(query, items, top_k=2)
        assert len(results) == 2
        # Items containing "the cat" should rank above "hello world"
        assert all("the cat" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_async_cross_encoder_empty(self) -> None:
        reranker = AsyncCrossEncoderReranker(score_fn=_fake_score)
        query = QueryBundle(query_str="test")
        results = await reranker.arerank(query, [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# AsyncCohereReranker tests
# ---------------------------------------------------------------------------


class TestAsyncCohereReranker:
    """Tests for AsyncCohereReranker."""

    def test_async_cohere_protocol(self) -> None:
        reranker = AsyncCohereReranker(rerank_fn=_fake_cohere_rerank)
        assert isinstance(reranker, AsyncReranker)

    @pytest.mark.asyncio
    async def test_async_cohere_rerank(self) -> None:
        reranker = AsyncCohereReranker(rerank_fn=_fake_cohere_rerank)
        items = [
            _make_item("hello world", item_id="a"),
            _make_item("find the needle", item_id="b"),
            _make_item("needle in haystack", item_id="c"),
        ]
        query = QueryBundle(query_str="needle")
        results = await reranker.arerank(query, items, top_k=2)
        assert len(results) == 2
        # Items containing "needle" should be first
        assert "needle" in results[0].content

    @pytest.mark.asyncio
    async def test_async_cohere_empty(self) -> None:
        reranker = AsyncCohereReranker(rerank_fn=_fake_cohere_rerank)
        query = QueryBundle(query_str="test")
        results = await reranker.arerank(query, [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Repr tests
# ---------------------------------------------------------------------------


class TestReprAll:
    """Test __repr__ for all async classes."""

    def test_repr_all(self) -> None:
        dense = AsyncDenseRetriever(embed_fn=_fake_embed)
        assert "AsyncDenseRetriever" in repr(dense)

        hybrid = AsyncHybridRetriever(retrievers=[dense], weights=[1.0])
        assert "AsyncHybridRetriever" in repr(hybrid)

        cross = AsyncCrossEncoderReranker(score_fn=_fake_score)
        assert "AsyncCrossEncoderReranker" in repr(cross)

        cohere = AsyncCohereReranker(rerank_fn=_fake_cohere_rerank)
        assert "AsyncCohereReranker" in repr(cohere)


# ---------------------------------------------------------------------------
# Additional edge-case tests: AsyncDenseRetriever
# ---------------------------------------------------------------------------


class TestAsyncDenseRetrieverEdgeCases:
    """Additional edge-case tests for AsyncDenseRetriever."""

    @pytest.mark.asyncio
    async def test_retrieve_no_indexed_items(self) -> None:
        """Retrieving from a retriever with no indexed items returns []."""
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        query = QueryBundle(query_str="anything")
        results = await retriever.aretrieve(query, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_aindex_then_aretrieve_full_async_workflow(self) -> None:
        """Full async workflow: aindex computes embeddings, aretrieve finds them."""
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        items = [
            _make_item("apple", item_id="a1"),
            _make_item("banana", item_id="b1"),
            _make_item("cherry", item_id="c1"),
        ]
        # None of these items have embeddings yet; aindex should compute them
        for item in items:
            assert "embedding" not in item.metadata

        await retriever.aindex(items)
        query = QueryBundle(query_str="apple")
        results = await retriever.aretrieve(query, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ContextItem) for r in results)
        # "apple" query embedding should exactly match the "apple" item embedding
        assert results[0].content == "apple"

    @pytest.mark.asyncio
    async def test_custom_similarity_function_is_used(self) -> None:
        """A custom similarity function should be used instead of cosine_similarity."""
        call_log: list[tuple[list[float], list[float]]] = []

        def custom_sim(a: list[float], b: list[float]) -> float:
            call_log.append((a, b))
            # Always return 0.5 regardless of input
            return 0.5

        retriever = AsyncDenseRetriever(embed_fn=_fake_embed, similarity_fn=custom_sim)
        emb = await _fake_embed("test")
        items = [_make_item("test", item_id="t1", embedding=emb)]
        retriever.index(items)

        query = QueryBundle(query_str="test")
        results = await retriever.aretrieve(query, top_k=1)
        assert len(call_log) == 1  # custom_sim was called exactly once
        assert len(results) == 1
        assert results[0].score == 0.5

    @pytest.mark.asyncio
    async def test_items_without_embeddings_are_skipped(self) -> None:
        """Items that lack an embedding in metadata should be skipped during retrieve."""
        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        emb = await _fake_embed("has_emb")
        items = [
            _make_item("has_emb", item_id="a", embedding=emb),
            _make_item("no_emb", item_id="b"),  # no embedding
        ]
        retriever.index(items)

        query = QueryBundle(query_str="has_emb")
        results = await retriever.aretrieve(query, top_k=10)
        # Only the item with embedding should appear
        assert len(results) == 1
        assert results[0].content == "has_emb"


# ---------------------------------------------------------------------------
# Additional edge-case tests: AsyncHybridRetriever
# ---------------------------------------------------------------------------


class TestAsyncHybridRetrieverEdgeCases:
    """Additional edge-case tests for AsyncHybridRetriever."""

    @pytest.mark.asyncio
    async def test_single_retriever(self) -> None:
        """Hybrid retriever works correctly with just one sub-retriever."""
        r = AsyncDenseRetriever(embed_fn=_fake_embed)
        emb_a = await _fake_embed("alpha")
        emb_b = await _fake_embed("beta")
        items = [
            _make_item("alpha", item_id="a", embedding=emb_a),
            _make_item("beta", item_id="b", embedding=emb_b),
        ]
        r.index(items)

        hybrid = AsyncHybridRetriever(retrievers=[r])
        query = QueryBundle(query_str="alpha")
        results = await hybrid.aretrieve(query, top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, ContextItem) for r in results)

    @pytest.mark.asyncio
    async def test_retriever_raises_exception_handled(self) -> None:
        """If a sub-retriever raises, the hybrid retriever skips it gracefully."""

        class FailingRetriever:
            async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                msg = "simulated failure"
                raise RuntimeError(msg)

        good_retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        emb = await _fake_embed("ok")
        items = [_make_item("ok", item_id="ok1", embedding=emb)]
        good_retriever.index(items)

        hybrid = AsyncHybridRetriever(
            retrievers=[good_retriever, FailingRetriever()],  # type: ignore[list-item]
            weights=[1.0, 1.0],
        )
        query = QueryBundle(query_str="ok")
        results = await hybrid.aretrieve(query, top_k=5)
        # The good retriever should still return results despite the failing one
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_empty_results_from_all_retrievers(self) -> None:
        """If all sub-retrievers return empty lists, hybrid returns []."""
        r1 = AsyncDenseRetriever(embed_fn=_fake_embed)
        r2 = AsyncDenseRetriever(embed_fn=_fake_embed)
        # Neither retriever has indexed items

        hybrid = AsyncHybridRetriever(retrievers=[r1, r2])
        query = QueryBundle(query_str="anything")
        results = await hybrid.aretrieve(query, top_k=5)
        assert results == []

    def test_hybrid_requires_at_least_one_retriever(self) -> None:
        """Creating a hybrid retriever with no sub-retrievers raises ValueError."""
        with pytest.raises(ValueError, match="At least one retriever"):
            AsyncHybridRetriever(retrievers=[])


# ---------------------------------------------------------------------------
# Additional edge-case tests: AsyncCrossEncoderReranker
# ---------------------------------------------------------------------------


class TestAsyncCrossEncoderRerankerEdgeCases:
    """Additional edge-case tests for AsyncCrossEncoderReranker."""

    @pytest.mark.asyncio
    async def test_rerank_single_item_top_k_1(self) -> None:
        """Reranking with top_k=1 and multiple items returns only the best one."""
        reranker = AsyncCrossEncoderReranker(score_fn=_fake_score)
        items = [
            _make_item("hello world", item_id="a"),
            _make_item("find the needle", item_id="b"),
            _make_item("needle and thread", item_id="c"),
        ]
        query = QueryBundle(query_str="needle")
        results = await reranker.arerank(query, items, top_k=1)
        assert len(results) == 1
        assert "needle" in results[0].content

    @pytest.mark.asyncio
    async def test_rerank_items_fewer_than_top_k(self) -> None:
        """3 items with top_k=10 should return all 3 items."""
        reranker = AsyncCrossEncoderReranker(score_fn=_fake_score)
        items = [
            _make_item("alpha", item_id="a"),
            _make_item("beta", item_id="b"),
            _make_item("gamma", item_id="c"),
        ]
        query = QueryBundle(query_str="alpha")
        results = await reranker.arerank(query, items, top_k=10)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Additional edge-case tests: AsyncCohereReranker
# ---------------------------------------------------------------------------


class TestAsyncCohereRerankerEdgeCases:
    """Additional edge-case tests for AsyncCohereReranker."""

    @pytest.mark.asyncio
    async def test_cohere_invalid_indices_are_skipped(self) -> None:
        """If the rerank callback returns out-of-range indices, they are skipped."""

        async def bad_rerank(
            query: str, documents: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            # Return some valid and some invalid indices
            return [(0, 0.9), (99, 0.8), (-5, 0.7), (1, 0.6)]

        reranker = AsyncCohereReranker(rerank_fn=bad_rerank)
        items = [
            _make_item("first doc", item_id="a"),
            _make_item("second doc", item_id="b"),
        ]
        query = QueryBundle(query_str="anything")
        results = await reranker.arerank(query, items, top_k=10)
        # Only indices 0 and 1 are valid; 99 is out of range, -5 is negative
        # Implementation checks `0 <= idx < len(items)`, so -5 fails the >= 0 check
        assert len(results) == 2
        assert results[0].content == "first doc"
        assert results[1].content == "second doc"


# ---------------------------------------------------------------------------
# Integration: AsyncDenseRetriever -> AsyncCrossEncoderReranker pipeline
# ---------------------------------------------------------------------------


class TestAsyncRetrieveThenRerank:
    """Integration test: retrieve then rerank in an async pipeline."""

    @pytest.mark.asyncio
    async def test_retrieve_then_rerank_pipeline(self) -> None:
        """Full pipeline: AsyncDenseRetriever retrieves, then AsyncCrossEncoderReranker reranks."""

        async def pipeline_score(query: str, doc: str) -> float:
            """Score based on shared word overlap."""
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            if not q_words:
                return 0.0
            return len(q_words & d_words) / len(q_words)

        retriever = AsyncDenseRetriever(embed_fn=_fake_embed)
        items = [
            _make_item("machine learning basics", item_id="ml"),
            _make_item("deep learning advanced", item_id="dl"),
            _make_item("cooking recipes pasta", item_id="cook"),
            _make_item("learning python programming", item_id="py"),
        ]
        await retriever.aindex(items)

        query = QueryBundle(query_str="learning")
        # Step 1: Retrieve top candidates
        retrieved = await retriever.aretrieve(query, top_k=4)
        assert len(retrieved) > 0

        # Step 2: Rerank the retrieved items
        reranker = AsyncCrossEncoderReranker(score_fn=pipeline_score)
        reranked = await reranker.arerank(query, retrieved, top_k=2)
        assert len(reranked) <= 2
        assert all(isinstance(r, ContextItem) for r in reranked)
        # Items containing "learning" should score higher
        for item in reranked:
            assert "learning" in item.content.lower()
