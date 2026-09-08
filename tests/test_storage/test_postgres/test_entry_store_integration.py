"""PostgresEntryStore against a real database (``ANCHOR_TEST_POSTGRES_DSN``).

The soft-delete contract of roadmap #5 rests on every backend hiding
expired entries from ``search``/``list_all`` while keeping them readable
unfiltered; this is the Postgres leg of that contract.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from anchor.models.memory import MemoryEntry

pytest.importorskip("asyncpg")

DSN = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="ANCHOR_TEST_POSTGRES_DSN not set")


async def _run() -> None:
    import asyncpg

    from anchor.storage.postgres import (
        PostgresConnectionManager,
        PostgresEntryStore,
        ensure_tables,
    )

    conn = await asyncpg.connect(DSN)
    try:
        await conn.execute("DROP TABLE IF EXISTS memory_entries")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await ensure_tables(conn, embedding_dim=3)
        manager = PostgresConnectionManager(DSN, min_size=1, max_size=2)
        await manager.initialize()
        try:
            store = PostgresEntryStore(manager)
            past = datetime.now(UTC) - timedelta(hours=1)
            await store.add(MemoryEntry(id="old", content="memory old", expires_at=past))
            await store.add(MemoryEntry(id="live", content="memory live"))
            await store.add(MemoryEntry(id="soon", content="memory soon").invalidate(by="live"))

            assert [e.id for e in await store.search("memory")] == ["live"]
            assert [e.id for e in await store.list_all()] == ["live"]
            unfiltered = {e.id: e for e in await store.list_all_unfiltered()}
            assert set(unfiltered) == {"old", "live", "soon"}
            assert unfiltered["soon"].metadata == {"invalidated_by": "live"}
        finally:
            await manager.close()
    finally:
        await conn.execute("DROP TABLE IF EXISTS memory_entries")
        await conn.close()


def test_expired_entries_are_hidden_but_kept_unfiltered() -> None:
    asyncio.run(_run())
