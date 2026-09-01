"""SQLite-backed ContextStore implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anchor.models.context import ContextItem
from anchor.models.scope import DEFAULT_VAULT
from anchor.storage._serialization import context_item_to_row, row_to_context_item

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager


class SqliteContextStore:
    """SQLite-backed context store. Implements the ContextStore protocol."""

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self, conn_manager: SqliteConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    def add(self, item: ContextItem) -> None:
        # The mount owns the vault: whatever comes in is stored under it.
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        row = context_item_to_row(item)
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO context_items "
            "(id, content, source, score, priority, token_count, "
            "metadata_json, created_at, vault, namespace) "
            "VALUES (:id, :content, :source, :score, :priority, "
            ":token_count, :metadata_json, :created_at, :vault, :namespace)",
            row,
        )
        conn.commit()

    def get(self, item_id: str) -> ContextItem | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT * FROM context_items WHERE id = ? AND vault = ?",
            (item_id, self._vault),
        ).fetchone()
        if row is None:
            return None
        return row_to_context_item(row)

    def get_all(self) -> list[ContextItem]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT * FROM context_items WHERE vault = ?", (self._vault,)
        ).fetchall()
        return [row_to_context_item(r) for r in rows]

    def delete(self, item_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM context_items WHERE id = ? AND vault = ?",
            (item_id, self._vault),
        )
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute(
            "DELETE FROM context_items WHERE vault = ?", (self._vault,)
        )
        conn.commit()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteContextStore:
    """Async SQLite-backed context store.

    Implements the AsyncContextStore protocol.
    """

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self, conn_manager: SqliteConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    async def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        row = context_item_to_row(item)
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(
            "INSERT OR REPLACE INTO context_items "
            "(id, content, source, score, priority, token_count, "
            "metadata_json, created_at, vault, namespace) "
            "VALUES (:id, :content, :source, :score, :priority, "
            ":token_count, :metadata_json, :created_at, :vault, :namespace)",
            row,
        )
        await conn.commit()

    async def get(self, item_id: str) -> ContextItem | None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT * FROM context_items WHERE id = ? AND vault = ?",
            (item_id, self._vault),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row_to_context_item(row)

    async def get_all(self) -> list[ContextItem]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT * FROM context_items WHERE vault = ?", (self._vault,)
        )
        rows = await cursor.fetchall()
        return [row_to_context_item(r) for r in rows]

    async def delete(self, item_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM context_items WHERE id = ? AND vault = ?",
            (item_id, self._vault),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(
            "DELETE FROM context_items WHERE vault = ?", (self._vault,)
        )
        await conn.commit()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
