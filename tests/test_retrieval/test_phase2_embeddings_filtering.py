"""Phase 2 regression tests: embedding provider layer + filtered vector search.

Pins docs/plans/2026-08-25-sota-upgrade-plan.md Phase 2:
- EmbeddingProvider protocol with query/document asymmetry and batching
- CallableEmbeddingProvider unifying the legacy embed_fn shapes
- `where` metadata filter pushed down through VectorStore backends
- Batch indexing in DenseRetriever (one embed_documents call)
- AsyncDenseRetriever store-backed mode over AsyncVectorStore
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anchor.embeddings import CallableEmbeddingProvider
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.embeddings import EmbeddingProvider
from anchor.retrieval.async_retriever import AsyncDenseRetriever
from anchor.retrieval.dense import DenseRetriever
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import FakeTokenizer


def _embed(text: str) -> list[float]:
    """Deterministic toy embedding: [len % 7, vowels, 1]."""
    vowels = sum(text.lower().count(v) for v in "aeiou")
    return [float(len(text) % 7), float(vowels), 1.0]


def _items() -> list[ContextItem]:
    return [
        ContextItem(
            id="a", content="alpha doc", source=SourceType.RETRIEVAL,
            metadata={"tenant": "acme", "kind": "doc"},
        ),
        ContextItem(
            id="b", content="beta doc", source=SourceType.RETRIEVAL,
            metadata={"tenant": "acme", "kind": "note"},
        ),
        ContextItem(
            id="c", content="gamma doc", source=SourceType.RETRIEVAL,
            metadata={"tenant": "globex", "kind": "doc"},
        ),
    ]


class TestCallableEmbeddingProvider:
    def test_satisfies_protocol(self) -> None:
        provider = CallableEmbeddingProvider(_embed)
        assert isinstance(provider, EmbeddingProvider)

    def test_learns_dimensions(self) -> None:
        provider = CallableEmbeddingProvider(_embed)
        assert provider.dimensions == 0
        provider.embed_query("hello")
        assert provider.dimensions == 3

    def test_batch_falls_back_to_loop(self) -> None:
        provider = CallableEmbeddingProvider(_embed)
        vectors = provider.embed_documents(["one", "two"])
        assert vectors == [_embed("one"), _embed("two")]

    def test_native_batch_fn_used(self) -> None:
        calls: list[list[str]] = []

        def batch(texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [_embed(t) for t in texts]

        provider = CallableEmbeddingProvider(_embed, batch_fn=batch)
        provider.embed_documents(["one", "two", "three"])
        assert calls == [["one", "two", "three"]]

    def test_async_falls_back_to_thread(self) -> None:
        provider = CallableEmbeddingProvider(_embed)
        result = asyncio.run(provider.aembed_query("hello"))
        assert result == _embed("hello")

    def test_async_only_provider_rejects_sync(self) -> None:
        async def aembed(text: str) -> list[float]:
            return _embed(text)

        provider = CallableEmbeddingProvider(aembed_fn=aembed)
        with pytest.raises(TypeError, match="async"):
            provider.embed_query("hello")
        assert asyncio.run(provider.aembed_query("x")) == _embed("x")

    def test_requires_some_fn(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CallableEmbeddingProvider()


class TestBatchIndexing:
    def test_dense_index_uses_one_batch_call(self) -> None:
        calls: list[list[str]] = []

        class _BatchProvider:
            dimensions = 3

            def embed_query(self, text: str) -> list[float]:
                return _embed(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                calls.append(texts)
                return [_embed(t) for t in texts]

            async def aembed_query(self, text: str) -> list[float]:
                return _embed(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return [_embed(t) for t in texts]

        retriever = DenseRetriever(
            InMemoryVectorStore(),
            InMemoryContextStore(),
            tokenizer=FakeTokenizer(),
            embeddings=_BatchProvider(),
        )
        retriever.index(_items())
        assert len(calls) == 1  # one batch call, not one per item
        assert len(calls[0]) == 3

    def test_vector_count_mismatch_raises(self) -> None:
        class _BrokenProvider:
            dimensions = 3

            def embed_query(self, text: str) -> list[float]:
                return _embed(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [_embed(texts[0])]  # wrong count

            async def aembed_query(self, text: str) -> list[float]:
                return _embed(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return []

        from anchor.exceptions import RetrieverError

        retriever = DenseRetriever(
            InMemoryVectorStore(),
            InMemoryContextStore(),
            tokenizer=FakeTokenizer(),
            embeddings=_BrokenProvider(),
        )
        with pytest.raises(RetrieverError, match="vectors"):
            retriever.index(_items())


class TestWhereFilter:
    def _indexed_retriever(self) -> DenseRetriever:
        retriever = DenseRetriever(
            InMemoryVectorStore(),
            InMemoryContextStore(),
            embed_fn=_embed,
            tokenizer=FakeTokenizer(),
        )
        retriever.index(_items())
        return retriever

    def test_in_memory_where(self) -> None:
        retriever = self._indexed_retriever()
        results = retriever.retrieve(
            QueryBundle(query_str="doc"), top_k=10, where={"tenant": "acme"}
        )
        assert {item.id for item in results} == {"a", "b"}

    def test_where_multiple_keys(self) -> None:
        retriever = self._indexed_retriever()
        results = retriever.retrieve(
            QueryBundle(query_str="doc"),
            top_k=10,
            where={"tenant": "acme", "kind": "doc"},
        )
        assert [item.id for item in results] == ["a"]

    def test_no_where_returns_all(self) -> None:
        retriever = self._indexed_retriever()
        results = retriever.retrieve(QueryBundle(query_str="doc"), top_k=10)
        assert len(results) == 3

    def test_custom_store_without_where_still_works(self) -> None:
        """Pre-filter stores written before `where` existed must not break."""

        class _LegacyStore:
            def __init__(self) -> None:
                self._data: dict[str, list[float]] = {}

            def add_embedding(self, item_id, embedding, metadata=None) -> None:
                self._data[item_id] = embedding

            def search(self, query_embedding, top_k=10):  # no `where` param
                return [(item_id, 1.0) for item_id in list(self._data)[:top_k]]

            def delete(self, item_id) -> bool:
                return self._data.pop(item_id, None) is not None

        retriever = DenseRetriever(
            _LegacyStore(),
            InMemoryContextStore(),
            embed_fn=_embed,
            tokenizer=FakeTokenizer(),
        )
        retriever.index(_items())
        assert len(retriever.retrieve(QueryBundle(query_str="doc"), top_k=10)) == 3

    def test_sqlite_where(self, tmp_path: Path) -> None:
        pytest.importorskip("aiosqlite")
        from anchor.storage.sqlite import (
            SqliteConnectionManager,
            SqliteVectorStore,
            ensure_tables,
        )

        manager = SqliteConnectionManager(tmp_path / "vec.db")
        ensure_tables(manager.get_connection())
        store = SqliteVectorStore(manager)
        for item in _items():
            store.add_embedding(item.id, _embed(item.content), item.metadata)

        results = store.search(_embed("doc"), top_k=10, where={"tenant": "acme"})
        assert {r[0] for r in results} == {"a", "b"}

        results = store.search(
            _embed("doc"), top_k=10, where={"tenant": "acme", "kind": "doc"}
        )
        assert [r[0] for r in results] == ["a"]

    def test_async_sqlite_where(self, tmp_path: Path) -> None:
        pytest.importorskip("aiosqlite")
        from anchor.storage.sqlite import (
            AsyncSqliteVectorStore,
            SqliteConnectionManager,
            ensure_tables,
        )

        async def run() -> list[tuple[str, float]]:
            manager = SqliteConnectionManager(tmp_path / "avec.db")
            ensure_tables(manager.get_connection())
            store = AsyncSqliteVectorStore(manager)
            for item in _items():
                await store.add_embedding(item.id, _embed(item.content), item.metadata)
            return await store.search(_embed("doc"), top_k=10, where={"kind": "doc"})

        results = asyncio.run(run())
        assert {r[0] for r in results} == {"a", "c"}


class TestAsyncStoreBackedRetriever:
    def test_store_backed_end_to_end(self, tmp_path: Path) -> None:
        pytest.importorskip("aiosqlite")
        from anchor.storage.sqlite import (
            AsyncSqliteContextStore,
            AsyncSqliteVectorStore,
            SqliteConnectionManager,
            ensure_tables,
        )

        async def aembed(text: str) -> list[float]:
            return _embed(text)

        async def run() -> tuple[list[ContextItem], list[ContextItem]]:
            manager = SqliteConnectionManager(tmp_path / "e2e.db")
            ensure_tables(manager.get_connection())
            retriever = AsyncDenseRetriever(
                embed_fn=aembed,
                vector_store=AsyncSqliteVectorStore(manager),
                context_store=AsyncSqliteContextStore(manager),
            )
            await retriever.aindex(_items())
            all_results = await retriever.aretrieve(QueryBundle(query_str="doc"), top_k=10)
            filtered = await retriever.aretrieve(
                QueryBundle(query_str="doc"), top_k=10, where={"tenant": "globex"}
            )
            return all_results, filtered

        all_results, filtered = asyncio.run(run())
        assert len(all_results) == 3
        assert all("raw_score" in item.metadata for item in all_results)
        assert [item.id for item in filtered] == ["c"]

    def test_store_requires_context_store(self) -> None:
        with pytest.raises(ValueError, match="context_store"):
            AsyncDenseRetriever(vector_store=object())  # type: ignore[arg-type]


class TestPostgresSchema:
    def test_ensure_tables_creates_hnsw_and_requires_dim(self) -> None:
        pytest.importorskip("asyncpg")
        from anchor.storage.postgres._schema import ensure_tables

        executed: list[str] = []

        class _FakeConn:
            async def execute(self, sql: str, *args) -> None:
                executed.append(sql)

        asyncio.run(ensure_tables(_FakeConn(), embedding_dim=1024))
        joined = "\n".join(executed)
        assert "vector(1024)" in joined
        assert "USING hnsw" in joined
        assert "gin (metadata jsonb_path_ops)" in joined

        with pytest.raises(ValueError, match="embedding_dim"):
            asyncio.run(ensure_tables(_FakeConn(), embedding_dim=0))
