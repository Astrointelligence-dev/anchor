"""PostgreSQL-backed MemoryEntryStore implementation.

Note: This module does NOT reuse ``anchor.storage._serialization`` because
asyncpg returns ``asyncpg.Record`` objects which already parse JSONB columns
into native Python dicts/lists. The SQLite serialization helpers assume
plain strings that need ``json.loads()`` calls, making them incompatible
with asyncpg's automatic deserialization.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from anchor.models.memory import MemoryEntry, MemoryType
from anchor.storage._serialization import escape_like

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresEntryStore:
    """Async PostgreSQL-backed memory entry store.

    Implements ``AsyncMemoryEntryStore`` and ``AsyncGarbageCollectableStore``
    protocols. All filtering happens at the SQL level using PostgreSQL-native
    JSONB queries.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add(self, entry: MemoryEntry) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                """INSERT INTO memory_entries
                   (id, content, relevance_score, access_count, last_accessed,
                    created_at, updated_at, tags, metadata, memory_type,
                    user_id, session_id, expires_at, content_hash,
                    source_turns, links)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                           $11,$12,$13,$14,$15,$16)
                   ON CONFLICT (id) DO UPDATE SET
                       content = EXCLUDED.content,
                       relevance_score = EXCLUDED.relevance_score,
                       access_count = EXCLUDED.access_count,
                       last_accessed = EXCLUDED.last_accessed,
                       updated_at = EXCLUDED.updated_at,
                       tags = EXCLUDED.tags,
                       metadata = EXCLUDED.metadata,
                       memory_type = EXCLUDED.memory_type,
                       user_id = EXCLUDED.user_id,
                       session_id = EXCLUDED.session_id,
                       expires_at = EXCLUDED.expires_at,
                       content_hash = EXCLUDED.content_hash,
                       source_turns = EXCLUDED.source_turns,
                       links = EXCLUDED.links""",
                entry.id,
                entry.content,
                entry.relevance_score,
                entry.access_count,
                entry.last_accessed,
                entry.created_at,
                entry.updated_at,
                json.dumps(entry.tags),
                json.dumps(entry.metadata, default=str),
                str(entry.memory_type),
                entry.user_id,
                entry.session_id,
                entry.expires_at,
                entry.content_hash,
                json.dumps(entry.source_turns),
                json.dumps(entry.links),
            )

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        now = datetime.now(UTC)
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM memory_entries
                   WHERE (expires_at IS NULL OR expires_at > $1)
                     AND content ILIKE $2 ESCAPE '\\'
                   ORDER BY relevance_score DESC
                   LIMIT $3""",
                now,
                f"%{escape_like(query)}%",
                top_k,
            )
            return [_row_to_entry(r) for r in rows]

    async def list_all(self) -> list[MemoryEntry]:
        now = datetime.now(UTC)
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM memory_entries
                   WHERE expires_at IS NULL OR expires_at > $1""",
                now,
            )
            return [_row_to_entry(r) for r in rows]

    async def delete(self, entry_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_entries WHERE id = $1", entry_id
            )
            # asyncpg returns "DELETE N" where N is rows affected
            return int(result.split()[-1]) > 0

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM memory_entries")

    async def list_all_unfiltered(self) -> list[MemoryEntry]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM memory_entries")
            return [_row_to_entry(r) for r in rows]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_entries WHERE id = $1", entry_id
            )
            if row is None:
                return None
            return _row_to_entry(row)

    async def search_filtered(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[MemoryEntry]:
        now = datetime.now(UTC)
        clauses = ["(expires_at IS NULL OR expires_at > $1)"]
        params: list[Any] = [now]
        idx = 2

        if query:
            clauses.append(f"content ILIKE ${idx} ESCAPE '\\'")

            params.append(f"%{escape_like(query)}%")
            idx += 1
        if user_id is not None:
            clauses.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if session_id is not None:
            clauses.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if memory_type is not None:
            clauses.append(f"memory_type = ${idx}")
            params.append(str(memory_type))
            idx += 1
        if tags:
            for tag in tags:
                clauses.append(f"tags @> ${idx}::jsonb")
                params.append(json.dumps([tag]))
                idx += 1
        if created_after is not None:
            clauses.append(f"created_at > ${idx}")
            params.append(created_after)
            idx += 1
        if created_before is not None:
            clauses.append(f"created_at < ${idx}")
            params.append(created_before)
            idx += 1

        where = " AND ".join(clauses)
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM memory_entries WHERE {where} "  # noqa: S608
                f"ORDER BY relevance_score DESC LIMIT ${idx}",
                *params,
                top_k,
            )
            return [_row_to_entry(r) for r in rows]

    async def delete_by_user(self, user_id: str) -> int:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_entries WHERE user_id = $1", user_id
            )
            # asyncpg returns "DELETE N"
            return int(result.split()[-1])

    async def export_user_entries(self, user_id: str) -> list[MemoryEntry]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memory_entries WHERE user_id = $1", user_id
            )
            return [_row_to_entry(r) for r in rows]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _row_to_entry(row: Any) -> MemoryEntry:
    """Convert an asyncpg Record to a MemoryEntry."""
    tags = row["tags"]
    if isinstance(tags, str):
        tags = json.loads(tags)
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    source_turns = row["source_turns"]
    if isinstance(source_turns, str):
        source_turns = json.loads(source_turns)
    links = row["links"]
    if isinstance(links, str):
        links = json.loads(links)

    return MemoryEntry(
        id=row["id"],
        content=row["content"],
        relevance_score=row["relevance_score"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        tags=tags,
        metadata=metadata,
        memory_type=row["memory_type"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        expires_at=row["expires_at"],
        content_hash=row["content_hash"],
        source_turns=source_turns,
        links=links,
    )
