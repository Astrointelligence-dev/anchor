"""PostgreSQL-backed VectorStore using pgvector."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresVectorStore:
    """Async PostgreSQL-backed vector store using pgvector.

    Uses the ``<=>`` cosine distance operator for similarity search.
    Similarity scores are computed as ``1 - distance``.

    Implements the AsyncVectorStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                """INSERT INTO embeddings (item_id, embedding, metadata)
                   VALUES ($1, $2::vector, $3)
                   ON CONFLICT (item_id) DO UPDATE SET
                       embedding = EXCLUDED.embedding,
                       metadata = EXCLUDED.metadata""",
                item_id,
                vec_str,
                json.dumps(metadata or {}),
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        async with self._conn_manager.acquire() as conn:
            if where:
                # JSONB containment: metadata must include every filter pair.
                rows = await conn.fetch(
                    """SELECT item_id, 1 - (embedding <=> $1::vector) AS score
                       FROM embeddings
                       WHERE metadata @> $3::jsonb
                       ORDER BY embedding <=> $1::vector
                       LIMIT $2""",
                    vec_str,
                    top_k,
                    json.dumps(where),
                )
            else:
                rows = await conn.fetch(
                    """SELECT item_id, 1 - (embedding <=> $1::vector) AS score
                       FROM embeddings
                       ORDER BY embedding <=> $1::vector
                       LIMIT $2""",
                    vec_str,
                    top_k,
                )
            return [(row["item_id"], row["score"]) for row in rows]

    async def delete(self, item_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM embeddings WHERE item_id = $1", item_id
            )
            # asyncpg returns "DELETE N" where N is rows affected
            return int(result.split()[-1]) > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
