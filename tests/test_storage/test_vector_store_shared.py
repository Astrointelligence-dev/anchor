"""Shared behavioral tests for VectorStore implementations."""

from __future__ import annotations

import pytest

from tests.conftest import make_embedding


@pytest.fixture(params=["memory", "sqlite"])
def vector_store_impl(request, tmp_path):
    if request.param == "memory":
        from anchor.storage.memory_store import InMemoryVectorStore

        return InMemoryVectorStore()
    from anchor.storage.sqlite import (
        SqliteConnectionManager,
        SqliteVectorStore,
        ensure_tables,
    )

    mgr = SqliteConnectionManager(tmp_path / "test.db")
    ensure_tables(mgr.get_connection())
    return SqliteVectorStore(mgr)


class TestVectorStoreCRUD:
    """Core add_embedding / search / delete operations."""

    def test_add_and_search(self, vector_store_impl) -> None:
        emb = make_embedding(1)
        vector_store_impl.add_embedding("v1", emb)
        results = vector_store_impl.search(emb, top_k=5)
        assert len(results) >= 1
        ids = [item_id for item_id, _ in results]
        assert "v1" in ids

    def test_search_returns_ordered_by_similarity(
        self, vector_store_impl
    ) -> None:
        query = make_embedding(1)
        similar = make_embedding(1)  # identical to query -> highest sim
        dissimilar = make_embedding(99)  # very different

        vector_store_impl.add_embedding("similar", similar)
        vector_store_impl.add_embedding("dissimilar", dissimilar)

        results = vector_store_impl.search(query, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "similar"
        assert results[0][1] >= results[1][1]

    def test_search_respects_top_k(self, vector_store_impl) -> None:
        for i in range(10):
            vector_store_impl.add_embedding(
                f"item-{i}", make_embedding(i)
            )
        results = vector_store_impl.search(make_embedding(0), top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, vector_store_impl) -> None:
        results = vector_store_impl.search(make_embedding(1), top_k=5)
        assert results == []

    def test_delete_removes_embedding(self, vector_store_impl) -> None:
        emb = make_embedding(42)
        vector_store_impl.add_embedding("del-v", emb)
        assert vector_store_impl.delete("del-v") is True
        results = vector_store_impl.search(emb, top_k=5)
        ids = [item_id for item_id, _ in results]
        assert "del-v" not in ids

    def test_delete_returns_false_for_missing(
        self, vector_store_impl
    ) -> None:
        assert vector_store_impl.delete("nonexistent") is False
