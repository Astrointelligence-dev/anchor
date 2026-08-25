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

pytest.importorskip("asyncpg")

DSN = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="ANCHOR_TEST_POSTGRES_DSN not set"
)

_DIM = 3


def _embed(text: str) -> list[float]:
    vowels = sum(text.lower().count(v) for v in "aeiou")
    return [float(len(text) % 7), float(vowels), 1.0]


async def _with_store(fn):
    import asyncpg

    from anchor.storage.postgres._schema import ensure_tables
    from anchor.storage.postgres._vector_store import PostgresVectorStore

    conn = await asyncpg.connect(DSN)
    try:
        await conn.execute("DROP TABLE IF EXISTS embeddings")
        await ensure_tables(conn, embedding_dim=_DIM)

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return None

        class _Manager:
            def acquire(self):
                return _Ctx()

        store = PostgresVectorStore(_Manager())
        return await fn(store)
    finally:
        await conn.execute("DROP TABLE IF EXISTS embeddings")
        await conn.close()


class TestPgvectorEndToEnd:
    def test_upsert_search_filter_delete(self) -> None:
        async def scenario(store):
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

        assert asyncio.run(_with_store(scenario)) is True
