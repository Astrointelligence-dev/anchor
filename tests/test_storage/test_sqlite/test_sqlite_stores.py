"""SQLite-specific storage backend tests.

Tests WAL mode, connection management, schema creation, BLOB embedding
storage, and persistence across store instances.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta

import pytest

from anchor.models.context import ContextItem
from anchor.models.memory import MemoryType
from anchor.storage.sqlite import (
    SqliteConnectionManager,
    SqliteContextStore,
    SqliteDocumentStore,
    SqliteEntryStore,
    SqliteVectorStore,
    ensure_tables,
)
from tests.conftest import make_embedding, make_memory_entry

# ---------------------------------------------------------------------------
# TestSqliteConnectionManager
# ---------------------------------------------------------------------------


class TestSqliteConnectionManager:
    """Connection manager: WAL mode, thread-local connections, close."""

    def test_wal_mode_enabled(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "wal.db")
        conn = mgr.get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        mgr.close()

    def test_thread_local_returns_same_connection(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "tl.db")
        conn1 = mgr.get_connection()
        conn2 = mgr.get_connection()
        assert conn1 is conn2
        mgr.close()

    def test_close_clears_connection(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "close.db")
        conn1 = mgr.get_connection()
        mgr.close()
        # After close, a new call should return a fresh connection
        conn2 = mgr.get_connection()
        assert conn1 is not conn2
        mgr.close()

    def test_db_path_property(self, tmp_path):
        db_file = tmp_path / "prop.db"
        mgr = SqliteConnectionManager(db_file)
        assert mgr.db_path == db_file.resolve()

    def test_wal_mode_disabled(self, tmp_path):
        mgr = SqliteConnectionManager(
            tmp_path / "nowal.db", wal_mode=False
        )
        conn = mgr.get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        # Without WAL, SQLite defaults to "delete" journal mode
        assert mode.lower() != "wal"
        mgr.close()


# ---------------------------------------------------------------------------
# TestSqliteSchemaCreation
# ---------------------------------------------------------------------------


class TestSqliteSchemaCreation:
    """ensure_tables creates all four expected tables."""

    def test_all_tables_created(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "schema.db")
        conn = mgr.get_connection()
        ensure_tables(conn)

        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in rows}

        expected = {
            "context_items",
            "embeddings",
            "documents",
            "memory_entries",
        }
        assert expected.issubset(table_names)
        mgr.close()

    def test_idempotent(self, tmp_path):
        """Calling ensure_tables twice does not raise."""
        mgr = SqliteConnectionManager(tmp_path / "idem.db")
        conn = mgr.get_connection()
        ensure_tables(conn)
        ensure_tables(conn)  # should not raise
        mgr.close()


# ---------------------------------------------------------------------------
# TestSqliteContextStore
# ---------------------------------------------------------------------------


class TestSqliteContextStore:
    """Full CRUD for context items."""

    def test_add_and_get(self, sqlite_context_store: SqliteContextStore):
        item = ContextItem(
            id="ctx-1", content="hello", source="user"
        )
        sqlite_context_store.add(item)
        got = sqlite_context_store.get("ctx-1")
        assert got is not None
        assert got.content == "hello"
        assert got.id == "ctx-1"

    def test_get_missing_returns_none(
        self, sqlite_context_store: SqliteContextStore
    ):
        assert sqlite_context_store.get("nonexistent") is None

    def test_get_all(self, sqlite_context_store: SqliteContextStore):
        for i in range(3):
            sqlite_context_store.add(
                ContextItem(
                    id=f"c{i}", content=f"text-{i}", source="system"
                )
            )
        items = sqlite_context_store.get_all()
        assert len(items) == 3

    def test_delete(self, sqlite_context_store: SqliteContextStore):
        sqlite_context_store.add(
            ContextItem(id="del-1", content="bye", source="tool")
        )
        assert sqlite_context_store.delete("del-1") is True
        assert sqlite_context_store.get("del-1") is None

    def test_delete_missing_returns_false(
        self, sqlite_context_store: SqliteContextStore
    ):
        assert sqlite_context_store.delete("nope") is False

    def test_clear(self, sqlite_context_store: SqliteContextStore):
        for i in range(3):
            sqlite_context_store.add(
                ContextItem(
                    id=f"clr{i}", content="x", source="memory"
                )
            )
        sqlite_context_store.clear()
        assert sqlite_context_store.get_all() == []

    def test_overwrite_existing(
        self, sqlite_context_store: SqliteContextStore
    ):
        sqlite_context_store.add(
            ContextItem(id="ow-1", content="v1", source="user")
        )
        sqlite_context_store.add(
            ContextItem(id="ow-1", content="v2", source="user")
        )
        got = sqlite_context_store.get("ow-1")
        assert got is not None
        assert got.content == "v2"


# ---------------------------------------------------------------------------
# TestSqliteVectorStore
# ---------------------------------------------------------------------------


class TestSqliteVectorStore:
    """Embedding add, search ordering, delete, BLOB packing."""

    def test_add_and_search(
        self, sqlite_vector_store: SqliteVectorStore
    ):
        emb = make_embedding(1)
        sqlite_vector_store.add_embedding("v1", emb)
        results = sqlite_vector_store.search(emb, top_k=5)
        assert len(results) == 1
        assert results[0][0] == "v1"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_returns_ordered_by_similarity(
        self, sqlite_vector_store: SqliteVectorStore
    ):
        query = make_embedding(1)
        # Add several embeddings; seed=1 should be most similar to query
        for seed in range(1, 6):
            sqlite_vector_store.add_embedding(
                f"v{seed}", make_embedding(seed)
            )

        results = sqlite_vector_store.search(query, top_k=5)
        assert results[0][0] == "v1"
        # Scores should be in descending order
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete(self, sqlite_vector_store: SqliteVectorStore):
        sqlite_vector_store.add_embedding("vdel", make_embedding(10))
        assert sqlite_vector_store.delete("vdel") is True
        assert sqlite_vector_store.delete("vdel") is False

    def test_search_empty_returns_empty(
        self, sqlite_vector_store: SqliteVectorStore
    ):
        results = sqlite_vector_store.search(make_embedding(1))
        assert results == []

    def test_blob_packing_roundtrip(self):
        """Packed BLOB should roundtrip through struct pack/unpack."""
        original = make_embedding(42, dim=64)
        blob = struct.pack(f"{len(original)}f", *original)
        unpacked = list(struct.unpack(f"{len(original)}f", blob))
        assert len(unpacked) == 64
        for a, b in zip(original, unpacked, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_add_with_metadata(
        self, sqlite_vector_store: SqliteVectorStore
    ):
        emb = make_embedding(99)
        sqlite_vector_store.add_embedding(
            "vmeta", emb, metadata={"source": "test"}
        )
        results = sqlite_vector_store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "vmeta"


# ---------------------------------------------------------------------------
# TestSqliteDocumentStore
# ---------------------------------------------------------------------------


class TestSqliteDocumentStore:
    """Full CRUD for documents."""

    def test_add_and_get(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        sqlite_document_store.add_document("d1", "doc content")
        assert sqlite_document_store.get_document("d1") == "doc content"

    def test_get_missing_returns_none(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        assert sqlite_document_store.get_document("missing") is None

    def test_list_documents(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        sqlite_document_store.add_document("d1", "a")
        sqlite_document_store.add_document("d2", "b")
        ids = sqlite_document_store.list_documents()
        assert sorted(ids) == ["d1", "d2"]

    def test_delete_document(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        sqlite_document_store.add_document("d-del", "x")
        assert sqlite_document_store.delete_document("d-del") is True
        assert sqlite_document_store.get_document("d-del") is None

    def test_delete_missing_returns_false(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        assert sqlite_document_store.delete_document("nope") is False

    def test_overwrite(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        sqlite_document_store.add_document("d-ow", "v1")
        sqlite_document_store.add_document("d-ow", "v2")
        assert sqlite_document_store.get_document("d-ow") == "v2"

    def test_add_with_metadata(
        self, sqlite_document_store: SqliteDocumentStore
    ):
        sqlite_document_store.add_document(
            "d-meta", "content", metadata={"author": "test"}
        )
        assert (
            sqlite_document_store.get_document("d-meta") == "content"
        )


# ---------------------------------------------------------------------------
# TestSqliteEntryStorePersistence
# ---------------------------------------------------------------------------


class TestSqliteEntryStorePersistence:
    """Data survives creating a new store on the same db file."""

    def test_persistence_across_instances(self, tmp_path):
        db_path = tmp_path / "persist.db"

        # First instance: write data
        mgr1 = SqliteConnectionManager(db_path)
        ensure_tables(mgr1.get_connection())
        store1 = SqliteEntryStore(mgr1)
        entry = make_memory_entry(entry_id="p1", content="persistent")
        store1.add(entry)
        mgr1.close()

        # Second instance: data should still be there
        mgr2 = SqliteConnectionManager(db_path)
        ensure_tables(mgr2.get_connection())
        store2 = SqliteEntryStore(mgr2)
        got = store2.get("p1")
        assert got is not None
        assert got.content == "persistent"
        mgr2.close()

    def test_persistence_context_store(self, tmp_path):
        db_path = tmp_path / "persist_ctx.db"

        mgr1 = SqliteConnectionManager(db_path)
        ensure_tables(mgr1.get_connection())
        cs1 = SqliteContextStore(mgr1)
        cs1.add(ContextItem(id="pc1", content="ctx", source="user"))
        mgr1.close()

        mgr2 = SqliteConnectionManager(db_path)
        ensure_tables(mgr2.get_connection())
        cs2 = SqliteContextStore(mgr2)
        got = cs2.get("pc1")
        assert got is not None
        assert got.content == "ctx"
        mgr2.close()


# ---------------------------------------------------------------------------
# TestSqliteEntryStoreSearchFiltered
# ---------------------------------------------------------------------------


class TestSqliteEntryStoreSearchFiltered:
    """Exercise the SQL WHERE clause path with complex filters."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Return an SqliteEntryStore pre-loaded with varied entries."""
        mgr = SqliteConnectionManager(tmp_path / "filter.db")
        ensure_tables(mgr.get_connection())
        store = SqliteEntryStore(mgr)

        base_time = datetime(2025, 1, 15, tzinfo=UTC)

        store.add(make_memory_entry(
            entry_id="f1",
            content="alpha project note",
            user_id="alice",
            session_id="s1",
            memory_type=MemoryType.SEMANTIC,
            tags=["project", "alpha"],
            relevance_score=0.9,
            created_at=base_time,
        ))
        store.add(make_memory_entry(
            entry_id="f2",
            content="beta project plan",
            user_id="alice",
            session_id="s2",
            memory_type=MemoryType.EPISODIC,
            tags=["project", "beta"],
            relevance_score=0.7,
            created_at=base_time + timedelta(days=10),
        ))
        store.add(make_memory_entry(
            entry_id="f3",
            content="gamma meeting log",
            user_id="bob",
            session_id="s1",
            memory_type=MemoryType.SEMANTIC,
            tags=["meeting"],
            relevance_score=0.5,
            created_at=base_time + timedelta(days=20),
        ))
        store.add(make_memory_entry(
            entry_id="f4",
            content="delta project review",
            user_id="bob",
            session_id="s3",
            memory_type=MemoryType.PROCEDURAL,
            tags=["project", "review"],
            relevance_score=0.8,
            created_at=base_time + timedelta(days=30),
        ))
        return store

    def test_filter_by_user_id(self, populated_store):
        results = populated_store.search_filtered(
            "", top_k=10, user_id="alice"
        )
        assert all(e.user_id == "alice" for e in results)
        assert len(results) == 2

    def test_filter_by_session_id(self, populated_store):
        results = populated_store.search_filtered(
            "", top_k=10, session_id="s1"
        )
        assert all(e.session_id == "s1" for e in results)
        assert len(results) == 2

    def test_filter_by_memory_type(self, populated_store):
        results = populated_store.search_filtered(
            "", top_k=10, memory_type=MemoryType.SEMANTIC
        )
        assert all(
            e.memory_type == MemoryType.SEMANTIC for e in results
        )
        assert len(results) == 2

    def test_filter_by_tags(self, populated_store):
        results = populated_store.search_filtered(
            "", top_k=10, tags=["project"]
        )
        assert all("project" in e.tags for e in results)
        assert len(results) == 3

    def test_filter_by_created_after(self, populated_store):
        cutoff = datetime(2025, 1, 20, tzinfo=UTC)
        results = populated_store.search_filtered(
            "", top_k=10, created_after=cutoff
        )
        assert all(e.created_at > cutoff for e in results)

    def test_filter_by_created_before(self, populated_store):
        cutoff = datetime(2025, 1, 20, tzinfo=UTC)
        results = populated_store.search_filtered(
            "", top_k=10, created_before=cutoff
        )
        assert all(e.created_at < cutoff for e in results)

    def test_combined_filters(self, populated_store):
        """Multiple filters narrow results via SQL AND clauses."""
        results = populated_store.search_filtered(
            "project",
            top_k=10,
            user_id="alice",
            memory_type=MemoryType.SEMANTIC,
        )
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_query_text_filter(self, populated_store):
        results = populated_store.search_filtered(
            "meeting", top_k=10
        )
        assert len(results) == 1
        assert results[0].id == "f3"

    def test_results_ordered_by_relevance(self, populated_store):
        results = populated_store.search_filtered(
            "project", top_k=10
        )
        scores = [e.relevance_score for e in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, populated_store):
        results = populated_store.search_filtered("", top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestAsyncSqliteSmoke
# ---------------------------------------------------------------------------


aiosqlite = pytest.importorskip("aiosqlite")


class TestAsyncSqliteSmoke:
    """Basic smoke tests for async SQLite store variants."""

    @pytest.fixture
    def async_conn_manager(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "async_smoke.db")
        ensure_tables(mgr.get_connection())
        return mgr

    @pytest.mark.asyncio
    async def test_async_entry_add_and_get(self, async_conn_manager):
        from anchor.storage.sqlite._entry_store import AsyncSqliteEntryStore

        store = AsyncSqliteEntryStore(async_conn_manager)
        entry = make_memory_entry(entry_id="ae1", content="async entry")
        await store.add(entry)
        got = await store.get("ae1")
        assert got is not None
        assert got.content == "async entry"
        await async_conn_manager.aclose()

    @pytest.mark.asyncio
    async def test_async_context_add_and_get(self, async_conn_manager):
        from anchor.storage.sqlite._context_store import AsyncSqliteContextStore

        store = AsyncSqliteContextStore(async_conn_manager)
        item = ContextItem(id="ac1", content="async ctx", source="user")
        await store.add(item)
        got = await store.get("ac1")
        assert got is not None
        assert got.content == "async ctx"
        await async_conn_manager.aclose()

    @pytest.mark.asyncio
    async def test_async_document_add_and_get(self, async_conn_manager):
        from anchor.storage.sqlite._document_store import AsyncSqliteDocumentStore

        store = AsyncSqliteDocumentStore(async_conn_manager)
        await store.add_document("ad1", "async doc content")
        got = await store.get_document("ad1")
        assert got == "async doc content"
        await async_conn_manager.aclose()

    @pytest.mark.asyncio
    async def test_async_vector_add_and_search(self, async_conn_manager):
        from anchor.storage.sqlite._vector_store import AsyncSqliteVectorStore

        store = AsyncSqliteVectorStore(async_conn_manager)
        emb = make_embedding(dim=4, seed=42)
        await store.add_embedding("av1", emb)
        results = await store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "av1"
        await async_conn_manager.aclose()
