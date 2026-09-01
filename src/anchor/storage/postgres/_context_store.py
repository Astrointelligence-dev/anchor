"""PostgreSQL-backed ContextStore implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import DEFAULT_VAULT, ROOT_NAMESPACE

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresContextStore:
    """Async PostgreSQL-backed context store.

    Implements the AsyncContextStore protocol.
    """

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self,
        conn_manager: PostgresConnectionManager,
        *,
        vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    async def add(self, item: ContextItem) -> None:
        # The mount owns the vault: whatever comes in is stored under it.
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                """INSERT INTO context_items
                   (id, content, source, score, priority,
                    token_count, metadata, created_at, vault, namespace)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                   ON CONFLICT (vault, id) DO UPDATE SET
                       content = EXCLUDED.content,
                       source = EXCLUDED.source,
                       score = EXCLUDED.score,
                       priority = EXCLUDED.priority,
                       token_count = EXCLUDED.token_count,
                       metadata = EXCLUDED.metadata,
                       created_at = EXCLUDED.created_at,
                       namespace = EXCLUDED.namespace""",
                item.id,
                item.content,
                str(item.source),
                item.score,
                item.priority,
                item.token_count,
                json.dumps(item.metadata, default=str),
                item.created_at,
                item.vault,
                item.namespace,
            )

    async def get(self, item_id: str) -> ContextItem | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM context_items WHERE id = $1 AND vault = $2",
                item_id,
                self._vault,
            )
            if row is None:
                return None
            return _row_to_context_item(row)

    async def get_all(self) -> list[ContextItem]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM context_items WHERE vault = $1", self._vault
            )
            return [_row_to_context_item(r) for r in rows]

    async def delete(self, item_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM context_items WHERE id = $1 AND vault = $2",
                item_id,
                self._vault,
            )
            # asyncpg returns "DELETE N" where N is rows affected
            return int(result.split()[-1]) > 0

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                "DELETE FROM context_items WHERE vault = $1", self._vault
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vault={self._vault!r})"


def _row_to_context_item(row: Any) -> ContextItem:
    """Convert an asyncpg Record to a ContextItem."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return ContextItem(
        id=row["id"],
        content=row["content"],
        source=SourceType(row["source"]),
        score=row["score"],
        priority=row["priority"],
        token_count=row["token_count"],
        metadata=metadata,
        created_at=row["created_at"],
        vault=_get(row, "vault") or DEFAULT_VAULT,
        namespace=_get(row, "namespace") or ROOT_NAMESPACE,
    )


def _get(row: Any, key: str) -> Any:
    """asyncpg Records raise on missing columns (pre-migration rows)."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None
