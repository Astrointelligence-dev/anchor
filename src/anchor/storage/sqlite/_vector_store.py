"""SQLite-backed VectorStore with brute-force cosine similarity."""

from __future__ import annotations

import heapq
import json
import logging
import struct
from typing import TYPE_CHECKING, Any

from anchor._math import cosine_similarity
from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
)
from anchor.storage._where import scope_sql_clauses, sql_where_clauses

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


def _meta_expr(key: str) -> tuple[str, list[Any]]:
    return "json_extract(metadata_json, ?)", [f"$.{key}"]


def _build_search_query(
    vault: str,
    where: dict[str, Any] | None,
    scope: RetrievalScope | None,
) -> tuple[str, list[Any]]:
    """Build the embeddings SELECT: vault bind + operators + scope ranges.

    Filtering happens in SQL (``json_extract`` for metadata, boundary-aware
    prefix ranges for the namespace column) so non-matching rows are never
    unpacked or scored. Shared compiler: :mod:`anchor.storage._where`.
    """
    sql = "SELECT item_id, embedding_blob FROM embeddings WHERE vault = ?"
    params: list[Any] = [vault]
    w_clauses, w_params = sql_where_clauses(where, _meta_expr)
    if w_clauses:
        sql += " AND " + " AND ".join(w_clauses)
        params.extend(w_params)
    s_clauses, s_params = scope_sql_clauses(scope, "namespace")
    if s_clauses:
        sql += " AND " + " AND ".join(s_clauses)
        params.extend(s_params)
    return sql, params


class SqliteVectorStore:
    """SQLite-backed vector store with brute-force cosine similarity.

    Embeddings are stored as packed float BLOBs. Search loads all embeddings
    and computes cosine similarity in Python -- suitable for small-to-medium
    datasets (< 50k vectors). For larger datasets, use a dedicated vector
    database (Qdrant, Chroma, pgvector).

    Implements the VectorStore protocol.
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

    def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        *,
        namespace: str = ROOT_NAMESPACE,
    ) -> None:
        blob = _pack_embedding(embedding)
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(item_id, embedding_blob, metadata_json, vault, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                item_id, blob, json.dumps(metadata or {}),
                self._vault, normalize_namespace(namespace),
            ),
        )
        conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        conn = self._conn_manager.get_connection()
        sql, params = _build_search_query(self._vault, where, scope)
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
            "DELETE FROM embeddings WHERE item_id = ? AND vault = ?",
            (item_id, self._vault),
        )
        conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"


class AsyncSqliteVectorStore:
    """Async SQLite-backed vector store.

    Implements the AsyncVectorStore protocol.
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

    async def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        *,
        namespace: str = ROOT_NAMESPACE,
    ) -> None:
        blob = _pack_embedding(embedding)
        conn = await self._conn_manager.get_async_connection()
        await conn.execute(
            "INSERT OR REPLACE INTO embeddings "
            "(item_id, embedding_blob, metadata_json, vault, namespace) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                item_id, blob, json.dumps(metadata or {}),
                self._vault, normalize_namespace(namespace),
            ),
        )
        await conn.commit()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        conn = await self._conn_manager.get_async_connection()
        sql, params = _build_search_query(self._vault, where, scope)
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
            "DELETE FROM embeddings WHERE item_id = ? AND vault = ?",
            (item_id, self._vault),
        )
        await conn.commit()
        return cursor.rowcount > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s})"
