"""Shared behavioral tests for InMemoryEntryStore and JsonFileMemoryStore.

These tests are parametrized via the ``entry_store`` fixture defined in
``conftest.py`` and run against **both** store implementations.  This
eliminates the DRY violation where search, filtering, and extra-method
tests were duplicated across test_entry_store.py and test_json_file_store.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from anchor.models.memory import MemoryType
from anchor.storage._base import BaseEntryStoreMixin
from tests.conftest import make_memory_entry as _make_entry

# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestSharedBasicCRUD:
    """Core add / search / list_all / delete / clear operations."""

    def test_add_stores_an_entry(self, entry_store: BaseEntryStoreMixin) -> None:
        entry = _make_entry(entry_id="a1", content="hello world")
        entry_store.add(entry)  # type: ignore[attr-defined]
        assert entry_store.get("a1") is not None
        assert entry_store.get("a1") == entry

    def test_search_finds_entries_by_substring(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(_make_entry(entry_id="s1", content="Python is great"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="s2", content="Java is verbose"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="s3", content="Python and Java"))  # type: ignore[attr-defined]

        results = entry_store.search("python")
        assert len(results) == 2
        ids = {e.id for e in results}
        assert ids == {"s1", "s3"}

    def test_search_returns_top_k_results(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        for i in range(10):
            entry_store.add(  # type: ignore[attr-defined]
                _make_entry(entry_id=f"t{i}", content=f"topic {i} about memory")
            )
        results = entry_store.search("memory", top_k=3)
        assert len(results) == 3

    def test_tagged_matches_beyond_top_k_rank_are_found(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        """Tags filtering must happen before top_k truncation.

        Regression (2026-08-31 review): the sqlite backend applied the SQL
        LIMIT first and the tags filter after, returning [] whenever
        higher-relevance untagged entries filled the limit.
        """
        for i in range(10):
            entry_store.add(  # type: ignore[attr-defined]
                _make_entry(
                    entry_id=f"plain{i}", content="memory note",
                    relevance_score=0.9,
                )
            )
        for i in range(5):
            entry_store.add(  # type: ignore[attr-defined]
                _make_entry(
                    entry_id=f"tagged{i}", content="memory note",
                    relevance_score=0.1, tags=["important"],
                )
            )

        results = entry_store.search_filtered(
            "memory", top_k=5, tags=["important"]
        )
        assert {e.id for e in results} == {f"tagged{i}" for i in range(5)}

    def test_search_sorts_by_relevance_score(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="lo", content="memory low", relevance_score=0.2)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="hi", content="memory high", relevance_score=0.9)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="mid", content="memory mid", relevance_score=0.5)
        )

        results = entry_store.search("memory")
        assert [e.id for e in results] == ["hi", "mid", "lo"]

    def test_search_filters_out_expired_entries(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="expired", content="memory old", expires_at=past)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="valid", content="memory new", expires_at=future)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="no_expiry", content="memory forever")
        )

        results = entry_store.search("memory")
        ids = {e.id for e in results}
        assert "expired" not in ids
        assert "valid" in ids
        assert "no_expiry" in ids

    def test_list_all_returns_all_non_expired_entries(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        entry_store.add(_make_entry(entry_id="a"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="b"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="c", expires_at=past))  # type: ignore[attr-defined]

        all_entries = entry_store.list_all()
        ids = {e.id for e in all_entries}
        assert ids == {"a", "b"}

    def test_delete_removes_an_entry(self, entry_store: BaseEntryStoreMixin) -> None:
        entry_store.add(_make_entry(entry_id="del1"))  # type: ignore[attr-defined]
        assert entry_store.delete("del1") is True  # type: ignore[attr-defined]
        assert entry_store.get("del1") is None

    def test_delete_returns_false_for_missing(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        assert entry_store.delete("nonexistent") is False  # type: ignore[attr-defined]

    def test_clear_removes_all_entries(self, entry_store: BaseEntryStoreMixin) -> None:
        for i in range(5):
            entry_store.add(_make_entry(entry_id=f"c{i}"))  # type: ignore[attr-defined]
        entry_store.clear()  # type: ignore[attr-defined]
        assert entry_store.list_all() == []


# ---------------------------------------------------------------------------
# Extra methods: get, list_all_unfiltered, delete_by_user
# ---------------------------------------------------------------------------


class TestSharedExtraMethods:
    """get(), list_all_unfiltered(), and delete_by_user() beyond the basic protocol."""

    def test_get_returns_entry_by_id(self, entry_store: BaseEntryStoreMixin) -> None:
        entry = _make_entry(entry_id="g1", content="retrievable")
        entry_store.add(entry)  # type: ignore[attr-defined]
        result = entry_store.get("g1")
        assert result is not None
        assert result.content == "retrievable"

    def test_get_returns_none_for_missing_id(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        assert entry_store.get("ghost") is None

    def test_list_all_unfiltered_includes_expired(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        entry_store.add(_make_entry(entry_id="ok"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="exp", expires_at=past))  # type: ignore[attr-defined]

        all_entries = entry_store.list_all_unfiltered()
        ids = {e.id for e in all_entries}
        assert ids == {"ok", "exp"}

    def test_delete_by_user_deletes_all_for_user(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(_make_entry(entry_id="u1", user_id="alice"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="u2", user_id="alice"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="u3", user_id="bob"))  # type: ignore[attr-defined]

        count = entry_store.delete_by_user("alice")
        assert count == 2
        assert entry_store.get("u1") is None
        assert entry_store.get("u2") is None
        assert entry_store.get("u3") is not None

    def test_delete_by_user_returns_count_of_deleted(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(_make_entry(entry_id="x1", user_id="carol"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="x2", user_id="carol"))  # type: ignore[attr-defined]
        entry_store.add(_make_entry(entry_id="x3", user_id="carol"))  # type: ignore[attr-defined]

        assert entry_store.delete_by_user("carol") == 3

    def test_delete_by_user_returns_zero_for_unknown_user(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(_make_entry(entry_id="y1", user_id="dave"))  # type: ignore[attr-defined]
        assert entry_store.delete_by_user("nobody") == 0


# ---------------------------------------------------------------------------
# search_filtered
# ---------------------------------------------------------------------------


class TestSharedSearchFiltered:
    """Filtered search with user_id, session_id, memory_type, tags, date range."""

    def test_filters_by_user_id(self, entry_store: BaseEntryStoreMixin) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="f1", content="data point", user_id="alice")
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="f2", content="data point", user_id="bob")
        )

        results = entry_store.search_filtered("data", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_filters_by_session_id(self, entry_store: BaseEntryStoreMixin) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="s1", content="info here", session_id="sess-a")
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="s2", content="info here", session_id="sess-b")
        )

        results = entry_store.search_filtered("info", session_id="sess-a")
        assert len(results) == 1
        assert results[0].id == "s1"

    def test_filters_by_memory_type(self, entry_store: BaseEntryStoreMixin) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="m1",
                content="procedure steps",
                memory_type=MemoryType.PROCEDURAL,
            )
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="m2",
                content="procedure memory",
                memory_type=MemoryType.SEMANTIC,
            )
        )

        results = entry_store.search_filtered(
            "procedure", memory_type=MemoryType.PROCEDURAL
        )
        assert len(results) == 1
        assert results[0].id == "m1"

    def test_filters_by_memory_type_as_string(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="m3",
                content="episodic event",
                memory_type=MemoryType.EPISODIC,
            )
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="m4",
                content="episodic story",
                memory_type=MemoryType.SEMANTIC,
            )
        )

        results = entry_store.search_filtered("episodic", memory_type="episodic")
        assert len(results) == 1
        assert results[0].id == "m3"

    def test_filters_by_tags_must_match_all(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="t1", content="tagged entry", tags=["python", "ml"])
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="t2", content="tagged entry", tags=["python"])
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="t3", content="tagged entry", tags=["ml", "deep"])
        )

        results = entry_store.search_filtered("tagged", tags=["python", "ml"])
        assert len(results) == 1
        assert results[0].id == "t1"

    def test_filters_by_created_after(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="old", content="historical note", created_at=old)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="new", content="historical note", created_at=recent)
        )

        results = entry_store.search_filtered("historical", created_after=cutoff)
        assert len(results) == 1
        assert results[0].id == "new"

    def test_filters_by_created_before(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="old", content="archival data", created_at=old)
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="new", content="archival data", created_at=recent)
        )

        results = entry_store.search_filtered("archival", created_before=cutoff)
        assert len(results) == 1
        assert results[0].id == "old"

    def test_combines_multiple_filters(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        ts = datetime(2025, 3, 15, tzinfo=UTC)

        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="hit",
                content="combined filter test",
                user_id="alice",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                tags=["important"],
                created_at=ts,
            )
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="miss_user",
                content="combined filter test",
                user_id="bob",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                tags=["important"],
                created_at=ts,
            )
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="miss_type",
                content="combined filter test",
                user_id="alice",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
                tags=["important"],
                created_at=ts,
            )
        )

        results = entry_store.search_filtered(
            "combined",
            user_id="alice",
            session_id="s1",
            memory_type=MemoryType.EPISODIC,
            tags=["important"],
            created_after=datetime(2025, 1, 1, tzinfo=UTC),
            created_before=datetime(2025, 12, 31, tzinfo=UTC),
        )
        assert len(results) == 1
        assert results[0].id == "hit"

    def test_search_filtered_excludes_expired_entries(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="expired",
                content="filtered content",
                user_id="alice",
                expires_at=past,
            )
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(
                entry_id="valid",
                content="filtered content",
                user_id="alice",
                expires_at=future,
            )
        )

        results = entry_store.search_filtered("filtered", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "valid"

    def test_search_filtered_empty_query_matches_all(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        """An empty query string should match all entries (no substring filtering)."""
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="a", content="alpha", user_id="alice")
        )
        entry_store.add(  # type: ignore[attr-defined]
            _make_entry(entry_id="b", content="beta", user_id="alice")
        )

        results = entry_store.search_filtered("", user_id="alice")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# _matches_filters static method (backwards-compat)
# ---------------------------------------------------------------------------


class TestSharedMatchesFilters:
    """Test the backwards-compatible _matches_filters static method."""

    def test_matches_filters_passes_matching_entry(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        if not hasattr(entry_store, "_matches_filters"):
            pytest.skip("SqliteEntryStore does filtering in SQL")
        entry = _make_entry(entry_id="mf1", user_id="alice")
        assert entry_store._matches_filters(
            entry,
            user_id="alice",
            session_id=None,
            memory_type_str=None,
            tags=None,
            created_after=None,
            created_before=None,
        )

    def test_matches_filters_rejects_non_matching_entry(
        self, entry_store: BaseEntryStoreMixin
    ) -> None:
        if not hasattr(entry_store, "_matches_filters"):
            pytest.skip("SqliteEntryStore does filtering in SQL")
        entry = _make_entry(entry_id="mf2", user_id="alice")
        assert not entry_store._matches_filters(
            entry,
            user_id="bob",
            session_id=None,
            memory_type_str=None,
            tags=None,
            created_after=None,
            created_before=None,
        )
