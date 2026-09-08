"""Shared behavioral tests for DocumentStore implementations."""

from __future__ import annotations

import pytest


@pytest.fixture(params=["memory", "sqlite"])
def document_store_impl(request, tmp_path):
    if request.param == "memory":
        from anchor.storage.memory_store import InMemoryDocumentStore

        return InMemoryDocumentStore()
    from anchor.storage.sqlite import (
        SqliteConnectionManager,
        SqliteDocumentStore,
        ensure_tables,
    )

    mgr = SqliteConnectionManager(tmp_path / "test.db")
    ensure_tables(mgr.get_connection())
    return SqliteDocumentStore(mgr)


class TestDocumentStoreCRUD:
    """Core add_document / get_document / list / delete operations."""

    def test_add_and_get(self, document_store_impl) -> None:
        document_store_impl.add_document("doc-1", "some content")
        result = document_store_impl.get_document("doc-1")
        assert result == "some content"

    def test_get_returns_none_for_missing(
        self, document_store_impl
    ) -> None:
        assert document_store_impl.get_document("nonexistent") is None

    def test_list_documents_returns_all_ids(
        self, document_store_impl
    ) -> None:
        document_store_impl.add_document("a", "alpha")
        document_store_impl.add_document("b", "beta")
        document_store_impl.add_document("c", "gamma")
        ids = document_store_impl.list_documents()
        assert set(ids) == {"a", "b", "c"}

    def test_list_documents_empty_store(
        self, document_store_impl
    ) -> None:
        assert document_store_impl.list_documents() == []

    def test_delete_removes_document(
        self, document_store_impl
    ) -> None:
        document_store_impl.add_document("del-d", "bye")
        assert document_store_impl.delete_document("del-d") is True
        assert document_store_impl.get_document("del-d") is None

    def test_delete_returns_false_for_missing(
        self, document_store_impl
    ) -> None:
        assert document_store_impl.delete_document("nonexistent") is False

    def test_add_overwrites_existing(
        self, document_store_impl
    ) -> None:
        document_store_impl.add_document("dup-d", "original")
        document_store_impl.add_document("dup-d", "updated")
        result = document_store_impl.get_document("dup-d")
        assert result == "updated"
        assert len(document_store_impl.list_documents()) == 1
