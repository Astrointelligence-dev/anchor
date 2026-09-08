"""SQLite-backed DocumentStore implementations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager


class SqliteDocumentStore:
    """SQLite-backed document store. Implements the DocumentStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, content, metadata_json) "
            "VALUES (?, ?, ?)",
            (doc_id, content, json.dumps(metadata or {}, default=str)),
        )
        conn.commit()

    def get_document(self, doc_id: str) -> str | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT content FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return row["content"]

    def list_documents(self) -> list[str]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT doc_id FROM documents").fetchall()
        return [row["doc_id"] for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteDocumentStore:
    """Async SQLite-backed document store.

    Implements the AsyncDocumentStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, content, metadata_json) "
            "VALUES (?, ?, ?)",
            (doc_id, content, json.dumps(metadata or {}, default=str)),
        )
        await conn.commit()

    async def get_document(self, doc_id: str) -> str | None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT content FROM documents WHERE doc_id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row["content"]

    async def list_documents(self) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT doc_id FROM documents")
        rows = await cursor.fetchall()
        return [row["doc_id"] for row in rows]

    async def delete_document(self, doc_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
