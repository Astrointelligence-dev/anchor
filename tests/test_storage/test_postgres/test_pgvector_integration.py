"""pgvector integration tests — run only against a real database.

Set ``ANCHOR_TEST_POSTGRES_DSN`` to enable, e.g.::

    export ANCHOR_TEST_POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/anchor_test

The database needs the pgvector extension available. Tables are created via
``ensure_tables`` and dropped afterwards.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import RetrievalScope

pytest.importorskip("asyncpg")

DSN = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="ANCHOR_TEST_POSTGRES_DSN not set"
)

_DIM = 3
_TABLES = ("embeddings", "context_items")

# Pre-vault shape (fa934c7): bare-id primary keys, no scope columns.
_LEGACY_DDL = (
    """CREATE TABLE context_items (
        id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
        score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        priority INTEGER NOT NULL DEFAULT 5,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL)""",
    f"""CREATE TABLE embeddings (
        item_id TEXT PRIMARY KEY, embedding vector({_DIM}),
        metadata JSONB NOT NULL DEFAULT '{{}}')""",
    "CREATE INDEX idx_embeddings_scope ON embeddings (item_id text_pattern_ops)",
)


def _embed(text: str) -> list[float]:
    vowels = sum(text.lower().count(v) for v in "aeiou")
    return [float(len(text) % 7), float(vowels), 1.0]


async def _with_manager(fn, *, legacy: bool = False):
    """Run *fn(manager)* on freshly created (or legacy-then-migrated) tables."""
    import asyncpg

    from anchor.storage.postgres import PostgresConnectionManager, ensure_tables

    conn = await asyncpg.connect(DSN)
    try:
        for table in _TABLES:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if legacy:
            for ddl in _LEGACY_DDL:
                await conn.execute(ddl)
            await conn.execute(
                "INSERT INTO context_items (id, content, source, created_at) "
                "VALUES ('old1', 'legacy', 'retrieval', now())"
            )
        await ensure_tables(conn, embedding_dim=_DIM)
        manager = PostgresConnectionManager(DSN, min_size=1, max_size=2)
        await manager.initialize()
        try:
            return await fn(manager, conn)
        finally:
            await manager.close()
    finally:
        for table in _TABLES:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.close()


class TestPgvectorEndToEnd:
    def test_upsert_search_filter_delete(self) -> None:
        from anchor.storage.postgres import PostgresVectorStore

        async def scenario(manager, _conn):
            store = PostgresVectorStore(manager)
            await store.add_embedding("a", _embed("alpha doc"), {"tenant": "acme"})
            await store.add_embedding("b", _embed("beta doc"), {"tenant": "acme"})
            await store.add_embedding("c", _embed("gamma doc"), {"tenant": "globex"})

            all_results = await store.search(_embed("doc"), top_k=10)
            assert len(all_results) == 3

            filtered = await store.search(
                _embed("doc"), top_k=10, where={"tenant": "globex"}
            )
            assert [r[0] for r in filtered] == ["c"]

            # upsert overwrites
            await store.add_embedding("a", _embed("alpha doc"), {"tenant": "globex"})
            filtered = await store.search(
                _embed("doc"), top_k=10, where={"tenant": "globex"}
            )
            assert {r[0] for r in filtered} == {"a", "c"}

            assert await store.delete("a") is True
            assert await store.delete("a") is False
            return True

        assert asyncio.run(_with_manager(scenario)) is True


class TestLegacyUpgrade:
    """Ritual xhigh #2/#5: migrated tables must take writes from any vault
    and filter namespaces byte-wise regardless of the database collation."""

    def test_two_vaults_write_after_migration(self) -> None:
        from anchor.storage.postgres import PostgresContextStore, PostgresVectorStore

        async def scenario(manager, conn):
            for vault, text in (("a", "A"), ("b", "B")):
                await PostgresContextStore(manager, vault=vault).add(
                    ContextItem(id="d1", content=text, source=SourceType.RETRIEVAL)
                )
                await PostgresVectorStore(manager, vault=vault).add_embedding(
                    "d1", _embed(text)
                )
            a = await PostgresContextStore(manager, vault="a").get("d1")
            b = await PostgresContextStore(manager, vault="b").get("d1")
            old = await PostgresContextStore(manager).get("old1")
            pk = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'context_items_pkey'"
            )
            scope_idx = await conn.fetchval(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_embeddings_scope'"
            )
            collation = await conn.fetchval(
                "SELECT collname FROM pg_attribute a JOIN pg_collation c "
                "ON c.oid = a.attcollation WHERE a.attrelid = 'embeddings'::regclass "
                "AND a.attname = 'namespace'"
            )
            return a.content, b.content, old.content, pk, scope_idx, collation

        a, b, old, pk, scope_idx, collation = asyncio.run(
            _with_manager(scenario, legacy=True)
        )
        assert (a, b, old) == ("A", "B", "legacy")
        assert "(vault, id)" in pk
        assert "text_pattern_ops" not in scope_idx
        assert collation == "C"

    def test_namespace_scope_is_boundary_aware(self) -> None:
        from anchor.storage.postgres import PostgresVectorStore

        async def scenario(manager, _conn):
            store = PostgresVectorStore(manager)
            rows = {
                "root": "/campanha-1",
                "child": "/campanha-1/sessoes",
                "deep": "/campanha-1/sessoes/2",
                "sibling": "/campanha-10",
                "other": "/outra",
            }
            for item_id, ns in rows.items():
                await store.add_embedding(item_id, _embed(item_id), namespace=ns)
            q = _embed("root")
            inc = await store.search(
                q, top_k=10, scope=RetrievalScope(include=("/campanha-1",))
            )
            exc = await store.search(
                q, top_k=10, scope=RetrievalScope(exclude=("/campanha-1",))
            )
            return {i for i, _ in inc}, {i for i, _ in exc}

        included, excluded = asyncio.run(_with_manager(scenario))
        assert included == {"root", "child", "deep"}
        assert excluded == {"sibling", "other"}


class TestSearchSemantics:
    """Ritual xhigh #6/#12/#13 on the real backend."""

    def test_where_semantics_match_the_reference(self) -> None:
        from anchor.storage.postgres import PostgresVectorStore

        async def scenario(manager, _conn):
            store = PostgresVectorStore(manager)
            rows = {
                "tagged": {"status": "active", "year": 2024},
                "later": {"year": 2025},
                "untagged": {},
                "odd": {"year": "unknown"},
            }
            for item_id, meta in rows.items():
                await store.add_embedding(item_id, _embed(item_id), meta)
            q = _embed("tagged")

            async def ids(where):
                return {i for i, _ in await store.search(q, top_k=10, where=where)}

            return (
                await ids({"status": {"$ne": "archived"}}),
                await ids({"year": {"$nin": [2025]}}),
                await ids({"year": {"$gt": 2024}}),
                await ids({"year": {"$gte": "unknown"}}),
            )

        ne, nin, gt, gte_str = asyncio.run(_with_manager(scenario))
        assert ne == {"tagged", "later", "untagged", "odd"}
        assert nin == {"tagged", "untagged", "odd"}
        assert gt == {"later"}
        assert gte_str == {"odd"}

    def test_iterative_scan_is_a_session_default_that_survives_release(self) -> None:
        async def scenario(manager, _conn):
            seen = []
            for _ in range(2):
                async with manager.acquire() as conn:
                    seen.append(await conn.fetchval("SHOW hnsw.iterative_scan"))
            return seen

        assert asyncio.run(_with_manager(scenario)) == ["relaxed_order"] * 2

