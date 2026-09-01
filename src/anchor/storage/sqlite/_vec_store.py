"""sqlite-vec backed VectorStore: real vector search in a single file.

Requires the ``sqlite-vec`` extra: ``pip install astro-anchor[sqlite-vec]``.

Unlike :class:`SqliteVectorStore` (which materializes every embedding into
Python and scores with a pure-Python cosine loop), this store runs KNN
inside SQLite via the ``vec0`` virtual table (C implementation, cosine
distance, metadata pre-filtering through a companion table).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
)
from anchor.storage._where import scope_sql_clauses, sql_where_clauses

logger = logging.getLogger(__name__)


def _meta_expr(key: str) -> tuple[str, list[Any]]:
    return "json_extract(metadata_json, ?)", [f"$.{key}"]


class SqliteVecVectorStore:
    """Vector store on the sqlite-vec ``vec0`` virtual table.

    Parameters
    ----------
    db_path:
        SQLite database file (``:memory:`` works for tests).
    dimensions:
        Embedding dimensionality — declared up front, validated on every
        add and search.

    Implements the ``VectorStore`` protocol (including ``where``
    metadata filtering, executed as a rowid pre-filter inside the KNN
    query).
    """

    __slots__ = ("_conn", "_dimensions", "_vault")

    def __init__(
        self,
        db_path: str | Path,
        dimensions: int,
        *,
        vault: str = DEFAULT_VAULT,
    ) -> None:
        if dimensions <= 0:
            msg = f"dimensions must be positive, got {dimensions}"
            raise ValueError(msg)
        try:
            import sqlite_vec
        except ImportError as e:
            msg = (
                "sqlite-vec is required for SqliteVecVectorStore. "
                "Install it with: pip install astro-anchor[sqlite-vec]"
            )
            raise ImportError(msg) from e

        self._dimensions = dimensions
        self._vault = vault
        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_items ("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
            "item_id TEXT NOT NULL, "
            "metadata_json TEXT NOT NULL DEFAULT '{}', "
            "vault TEXT NOT NULL DEFAULT '__default__', "
            "namespace TEXT NOT NULL DEFAULT '/', "
            "UNIQUE (vault, item_id))"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vec_items_scope "
            "ON vec_items(vault, namespace)"
        )
        # vault as vec0 PARTITION KEY: same-vault vectors are collocated
        # in chunks and other vaults' chunks are never scanned (verified
        # pre-filter; the 2026-09-01 research report has the evidence).
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0("
            f"vault TEXT PARTITION KEY, "
            f"embedding float[{dimensions}] distance_metric=cosine)"
        )
        self._conn.commit()

    @property
    def vault(self) -> str:
        return self._vault

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(dimensions={self._dimensions}, vault={self._vault!r})"
        )

    def _check_dim(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimensions:
            msg = (
                f"Embedding has {len(embedding)} dimensions, "
                f"store is declared with {self._dimensions}"
            )
            raise ValueError(msg)

    def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        *,
        namespace: str = ROOT_NAMESPACE,
    ) -> None:
        import sqlite_vec

        self._check_dim(embedding)
        blob = sqlite_vec.serialize_float32(embedding)
        meta_json = json.dumps(metadata or {})
        ns = normalize_namespace(namespace)

        cursor = self._conn.execute(
            "SELECT rowid FROM vec_items WHERE vault = ? AND item_id = ?",
            (self._vault, item_id),
        )
        row = cursor.fetchone()
        if row is not None:
            rowid = row[0]
            self._conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "UPDATE vec_items SET metadata_json = ?, namespace = ? "
                "WHERE rowid = ?",
                (meta_json, ns, rowid),
            )
        else:
            cursor = self._conn.execute(
                "INSERT INTO vec_items (item_id, metadata_json, vault, namespace) "
                "VALUES (?, ?, ?, ?)",
                (item_id, meta_json, self._vault, ns),
            )
            rowid = cursor.lastrowid
        self._conn.execute(
            "INSERT INTO vec_index (rowid, vault, embedding) VALUES (?, ?, ?)",
            (rowid, self._vault, blob),
        )
        self._conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        import sqlite_vec

        self._check_dim(query_embedding)
        blob = sqlite_vec.serialize_float32(query_embedding)

        rowid_filter = ""
        params: list[Any] = [blob, self._vault, top_k]
        if where or scope is not None:
            # Everything beyond the vault goes through a rowid subquery on
            # the companion table: the subquery pushes down as a true
            # pre-filter, and OR (multi-include scopes) is safe inside it —
            # an OR written directly into the vec0 KNN WHERE silently
            # degrades to a post-filter (verified 2026-09-01).
            sub_clauses = ["vault = ?"]
            sub_params: list[Any] = [self._vault]
            w_clauses, w_params = sql_where_clauses(where, _meta_expr)
            sub_clauses += w_clauses
            sub_params += w_params
            s_clauses, s_params = scope_sql_clauses(scope, "namespace")
            sub_clauses += s_clauses
            sub_params += s_params
            rowid_filter = (
                " AND rowid IN (SELECT rowid FROM vec_items WHERE "  # noqa: S608 -- fixed templates, values parameterized
                + " AND ".join(sub_clauses)
                + ")"
            )
            params = [blob, self._vault, *sub_params, top_k]

        # rowid_filter contains only "?" placeholders; values are parameterized.
        knn_sql = (
            "SELECT rowid, distance FROM vec_index "  # noqa: S608
            f"WHERE embedding MATCH ? AND vault = ?{rowid_filter} AND k = ? "
            "ORDER BY distance"
        )
        rows = self._conn.execute(knn_sql, params).fetchall()
        if not rows:
            return []

        # Placeholders only; rowids are bound as parameters.
        ids_sql = (
            "SELECT rowid, item_id FROM vec_items WHERE rowid IN "  # noqa: S608
            f"({','.join('?' for _ in rows)})"
        )
        id_map = {
            r[0]: r[1]
            for r in self._conn.execute(ids_sql, [r[0] for r in rows]).fetchall()
        }
        # cosine distance in [0, 2] -> similarity score = 1 - distance
        return [
            (id_map[rowid], 1.0 - float(distance))
            for rowid, distance in rows
            if rowid in id_map
        ]

    def delete(self, item_id: str) -> bool:
        cursor = self._conn.execute(
            "SELECT rowid FROM vec_items WHERE vault = ? AND item_id = ?",
            (self._vault, item_id),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        rowid = row[0]
        self._conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
        self._conn.commit()
        return True

    def count(self) -> int:
        """Number of stored embeddings in this store's vault."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM vec_items WHERE vault = ?", (self._vault,)
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
