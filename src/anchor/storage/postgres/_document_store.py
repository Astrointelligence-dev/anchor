"""PostgreSQL-backed DocumentStore implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresDocumentStore:
    """Async PostgreSQL-backed document store.

    Implements the AsyncDocumentStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                """INSERT INTO documents (doc_id, content, metadata)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (doc_id) DO UPDATE SET
                       content = EXCLUDED.content,
                       metadata = EXCLUDED.metadata""",
                doc_id,
                content,
                json.dumps(metadata or {}, default=str),
            )

    async def get_document(self, doc_id: str) -> str | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content FROM documents WHERE doc_id = $1", doc_id
            )
            if row is None:
                return None
            return row["content"]

    async def list_documents(self) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT doc_id FROM documents")
            return [row["doc_id"] for row in rows]

    async def delete_document(self, doc_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE doc_id = $1", doc_id
            )
            # asyncpg returns "DELETE N" where N is rows affected
            return int(result.split()[-1]) > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
