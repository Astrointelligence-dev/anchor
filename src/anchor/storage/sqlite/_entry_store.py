"""SQLite-backed MemoryEntryStore implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from anchor.models.memory import MemoryEntry, MemoryType
from anchor.storage._serialization import escape_like, memory_entry_to_row, row_to_memory_entry

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

_INSERT_SQL = (
    "INSERT OR REPLACE INTO memory_entries "
    "(id, content, relevance_score, access_count, last_accessed, created_at, "
    "updated_at, tags_json, metadata_json, memory_type, user_id, session_id, "
    "expires_at, content_hash, source_turns_json, links_json) "
    "VALUES (:id, :content, :relevance_score, :access_count, :last_accessed, "
    ":created_at, :updated_at, :tags_json, :metadata_json, :memory_type, "
    ":user_id, :session_id, :expires_at, :content_hash, "
    ":source_turns_json, :links_json)"
)

_NON_EXPIRED_CLAUSE = "(expires_at IS NULL OR expires_at > ?)"


class SqliteEntryStore:
    """SQLite-backed memory entry store.

    Implements ``MemoryEntryStore`` and ``GarbageCollectableStore`` protocols.
    Does NOT inherit from ``BaseEntryStoreMixin`` -- all filtering and search
    happens at the SQL level for efficiency.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        row = memory_entry_to_row(entry)
        conn = self._conn_manager.get_connection()
        conn.execute(_INSERT_SQL, row)
        conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        now = datetime.now(UTC).isoformat()
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            f"SELECT * FROM memory_entries WHERE {_NON_EXPIRED_CLAUSE} "  # noqa: S608
            "AND content LIKE ? ESCAPE '\\' ORDER BY relevance_score DESC LIMIT ?",
            (now, f"%{escape_like(query)}%", top_k),
        ).fetchall()
        return [row_to_memory_entry(r) for r in rows]

    def list_all(self) -> list[MemoryEntry]:
        now = datetime.now(UTC).isoformat()
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            f"SELECT * FROM memory_entries WHERE {_NON_EXPIRED_CLAUSE}",  # noqa: S608
            (now,),
        ).fetchall()
        return [row_to_memory_entry(r) for r in rows]

    def delete(self, entry_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM memory_entries WHERE id = ?", (entry_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM memory_entries")
        conn.commit()

    # ------------------------------------------------------------------
    # GarbageCollectableStore protocol
    # ------------------------------------------------------------------

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT * FROM memory_entries").fetchall()
        return [row_to_memory_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Extra methods (match BaseEntryStoreMixin interface)
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> MemoryEntry | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return row_to_memory_entry(row)

    def search_filtered(
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
        now = datetime.now(UTC).isoformat()
        clauses = [_NON_EXPIRED_CLAUSE]
        params: list[Any] = [now]

        if query:
            clauses.append("content LIKE ? ESCAPE '\\'")
            params.append(f"%{escape_like(query)}%")
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(str(memory_type))
        if created_after is not None:
            clauses.append("created_at > ?")
            params.append(created_after.isoformat())
        if created_before is not None:
            clauses.append("created_at < ?")
            params.append(created_before.isoformat())

        where = " AND ".join(clauses)
        conn = self._conn_manager.get_connection()
        # Tags are filtered in Python (they live in a JSON array column),
        # so the SQL LIMIT may only apply when there is no tags filter —
        # otherwise tagged matches ranked below the first top_k rows are
        # silently dropped (can return [] while matches exist).
        limit_sql = "" if tags else " LIMIT ?"
        if not tags:
            params.append(top_k)
        rows = conn.execute(
            f"SELECT * FROM memory_entries WHERE {where} "  # noqa: S608
            f"ORDER BY relevance_score DESC{limit_sql}",
            tuple(params),
        ).fetchall()

        entries = [row_to_memory_entry(r) for r in rows]

        if tags:
            entries = [
                e for e in entries if all(t in e.tags for t in tags)
            ]

        return entries[:top_k]

    def delete_by_user(self, user_id: str) -> int:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM memory_entries WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount

    def export_user_entries(self, user_id: str) -> list[MemoryEntry]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT * FROM memory_entries WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [row_to_memory_entry(r) for r in rows]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteEntryStore:
    """Async SQLite-backed memory entry store.

    Implements ``AsyncMemoryEntryStore`` and
    ``AsyncGarbageCollectableStore`` protocols.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add(self, entry: MemoryEntry) -> None:
        row = memory_entry_to_row(entry)
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(_INSERT_SQL, row)
        await conn.commit()

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        now = datetime.now(UTC).isoformat()
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            f"SELECT * FROM memory_entries WHERE {_NON_EXPIRED_CLAUSE} "  # noqa: S608
            "AND content LIKE ? ESCAPE '\\' ORDER BY relevance_score DESC LIMIT ?",
            (now, f"%{escape_like(query)}%", top_k),
        )
        rows = await cursor.fetchall()
        return [row_to_memory_entry(r) for r in rows]

    async def list_all(self) -> list[MemoryEntry]:
        now = datetime.now(UTC).isoformat()
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            f"SELECT * FROM memory_entries WHERE {_NON_EXPIRED_CLAUSE}",  # noqa: S608
            (now,),
        )
        rows = await cursor.fetchall()
        return [row_to_memory_entry(r) for r in rows]

    async def delete(self, entry_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM memory_entries WHERE id = ?", (entry_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM memory_entries")
        await conn.commit()

    async def list_all_unfiltered(self) -> list[MemoryEntry]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT * FROM memory_entries")
        rows = await cursor.fetchall()
        return [row_to_memory_entry(r) for r in rows]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row_to_memory_entry(row)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
