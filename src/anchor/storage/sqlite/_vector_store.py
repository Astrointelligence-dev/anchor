"""SQLite-backed VectorStore with brute-force cosine similarity."""

from __future__ import annotations

import heapq
import json
import logging
import struct
from typing import TYPE_CHECKING, Any

from anchor._math import cosine_similarity

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

logger = logging.getLogger(__name__)


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float list.

    The dimension is derived from the blob length (4 bytes per float32),
    never from the query vector — a query/stored dimension mismatch then
    surfaces as a clear ``ValueError`` from cosine_similarity instead of
    silently mis-unpacking.
    """
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _build_search_query(where: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Build the embeddings SELECT with optional metadata equality filters.

    Filtering happens in SQL via ``json_extract`` so non-matching rows are
    never unpacked or scored.
    """
    sql = "SELECT item_id, embedding_blob FROM embeddings"
    params: list[Any] = []
    if where:
        clauses = []
        for key, value in where.items():
            clauses.append("json_extract(metadata_json, ?) = ?")
            params.extend([f"$.{key}", value])
        sql += " WHERE " + " AND ".join(clauses)
    return sql, params


class SqliteVectorStore:
    """SQLite-backed vector store with brute-force cosine similarity.

    Embeddings are stored as packed float BLOBs. Search loads all embeddings
    and computes cosine similarity in Python -- suitable for small-to-medium
    datasets (< 50k vectors). For larger datasets, use a dedicated vector
    database (Qdrant, Chroma, pgvector).

    Implements the VectorStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        blob = _pack_embedding(embedding)
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(item_id, embedding_blob, metadata_json) "
            "VALUES (?, ?, ?)",
            (item_id, blob, json.dumps(metadata or {})),
        )
        conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        conn = self._conn_manager.get_connection()
        sql, params = _build_search_query(where)
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        results: list[tuple[str, float]] = []
        for row in rows:
            emb = _unpack_embedding(row["embedding_blob"])
            score = cosine_similarity(query_embedding, emb)
            results.append((row["item_id"], score))
        return heapq.nlargest(top_k, results, key=lambda x: x[1])

    def delete(self, item_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE item_id = ?", (item_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteVectorStore:
    """Async SQLite-backed vector store.

    Implements the AsyncVectorStore protocol.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: SqliteConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        blob = _pack_embedding(embedding)
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(item_id, embedding_blob, metadata_json) "
            "VALUES (?, ?, ?)",
            (item_id, blob, json.dumps(metadata or {})),
        )
        await conn.commit()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        conn = await self._conn_manager.get_async_connection()
        sql, params = _build_search_query(where)
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        if not rows:
            return []

        results: list[tuple[str, float]] = []
        for row in rows:
            emb = _unpack_embedding(row["embedding_blob"])
            score = cosine_similarity(query_embedding, emb)
            results.append((row["item_id"], score))
        return heapq.nlargest(top_k, results, key=lambda x: x[1])

    async def delete(self, item_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM embeddings WHERE item_id = ?", (item_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
