"""Shared behavioral tests for ContextStore implementations.

Parametrized via fixtures to run against InMemoryContextStore and SqliteContextStore.
"""

from __future__ import annotations

import pytest

from anchor.models.context import ContextItem


@pytest.fixture
def context_store_impl(make_context_store):
    return make_context_store()


class TestContextStoreCRUD:
    """Core add / get / get_all / delete / clear operations."""

    def test_add_and_get(self, context_store_impl) -> None:
        item = ContextItem(
            id="ctx-1", content="hello world", source="user"
        )
        context_store_impl.add(item)
        result = context_store_impl.get("ctx-1")
        assert result is not None
        assert result.content == "hello world"
        assert result.source == "user"

    def test_get_returns_none_for_missing(self, context_store_impl) -> None:
        assert context_store_impl.get("nonexistent") is None

    def test_get_all_returns_all_items(self, context_store_impl) -> None:
        for i in range(3):
            context_store_impl.add(
                ContextItem(
                    id=f"item-{i}",
                    content=f"content {i}",
                    source="system",
                )
            )
        results = context_store_impl.get_all()
        assert len(results) == 3
        ids = {item.id for item in results}
        assert ids == {"item-0", "item-1", "item-2"}

    def test_get_all_empty_store(self, context_store_impl) -> None:
        assert context_store_impl.get_all() == []

    def test_delete_removes_item(self, context_store_impl) -> None:
        item = ContextItem(
            id="del-1", content="to be deleted", source="system"
        )
        context_store_impl.add(item)
        assert context_store_impl.delete("del-1") is True
        assert context_store_impl.get("del-1") is None

    def test_delete_returns_false_for_missing(
        self, context_store_impl
    ) -> None:
        assert context_store_impl.delete("nonexistent") is False

    def test_clear_removes_all(self, context_store_impl) -> None:
        for i in range(5):
            context_store_impl.add(
                ContextItem(
                    id=f"clr-{i}", content=f"data {i}", source="system"
                )
            )
        context_store_impl.clear()
        assert context_store_impl.get_all() == []

    def test_add_overwrites_existing(self, context_store_impl) -> None:
        context_store_impl.add(
            ContextItem(
                id="dup-1", content="original", source="system"
            )
        )
        context_store_impl.add(
            ContextItem(
                id="dup-1", content="updated", source="system"
            )
        )
        result = context_store_impl.get("dup-1")
        assert result is not None
        assert result.content == "updated"
        assert len(context_store_impl.get_all()) == 1
