"""PostgreSQL-backed VectorStore using pgvector."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
)
from anchor.storage._where import SET_OPS, SQL_OPS, check_operator, is_op_dict

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


class _Params:
    """Numbered-placeholder accumulator ($1, $2, ...) for asyncpg."""

    def __init__(self, *initial: Any) -> None:
        self.values: list[Any] = list(initial)

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"


def _pg_where_clauses(where: dict[str, Any] | None, p: _Params) -> list[str]:
    """Compile the where dict for Postgres (JSONB metadata).

    Semantics match :func:`anchor.storage._where.matches_where` — including
    absent-key behavior ($nin passes on absent keys, ranges do not).
    """
    clauses: list[str] = []
    for key, cond in (where or {}).items():
        if not is_op_dict(cond):
            clauses.append(f"metadata @> {p.add(json.dumps({key: cond}))}::jsonb")
            continue
        for op, operand in cond.items():
            check_operator(op)
            if op == "$eq":
                clauses.append(
                    f"metadata @> {p.add(json.dumps({key: operand}))}::jsonb"
                )
            elif op == "$ne":
                clauses.append(
                    f"NOT metadata @> {p.add(json.dumps({key: operand}))}::jsonb"
                )
            elif op in SET_OPS:
                values = [json.dumps(v) for v in operand]
                if not values:
                    clauses.append("1=0" if op == "$in" else "1=1")
                    continue
                arr = p.add(values)
                if op == "$in":
                    clauses.append(f"metadata->{p.add(key)} = ANY({arr}::jsonb[])")
                else:
                    # Absent key passes $nin (mirror of the Python evaluator).
                    kp = p.add(key)
                    clauses.append(
                        f"(NOT metadata ? {kp} "
                        f"OR metadata->{kp} <> ALL({arr}::jsonb[]))"
                    )
            else:
                sym = SQL_OPS[op]
                kp = p.add(key)
                if isinstance(operand, bool) or not isinstance(
                    operand, (int, float)
                ):
                    clauses.append(f"metadata->>{kp} {sym} {p.add(str(operand))}")
                else:
                    clauses.append(
                        f"(metadata->>{kp})::numeric {sym} {p.add(operand)}"
                    )
    return clauses


def _pg_scope_clauses(scope: RetrievalScope | None, p: _Params) -> list[str]:
    if scope is None:
        return []

    def prefix_range(prefix: str) -> str:
        if prefix == ROOT_NAMESPACE:
            return "1=1"
        a = p.add(prefix)
        b = p.add(prefix + "/")
        c = p.add(prefix + "0")  # '0' = char after '/'
        return f"(namespace = {a} OR (namespace >= {b} AND namespace < {c}))"

    clauses: list[str] = []
    if scope.include:
        clauses.append(
            "(" + " OR ".join(prefix_range(x) for x in scope.include) + ")"
        )
    clauses.extend(f"NOT {prefix_range(x)}" for x in scope.exclude)
    return clauses


class PostgresVectorStore:
    """Async PostgreSQL-backed vector store using pgvector.

    Uses the ``<=>`` cosine distance operator for similarity search.
    Similarity scores are computed as ``1 - distance``.

    Implements the AsyncVectorStore protocol.
    """

    __slots__ = ("_conn_manager", "_iterative_scan", "_vault")

    def __init__(
        self,
        conn_manager: PostgresConnectionManager,
        *,
        vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault
        # None = capability not probed yet; set on first filtered search.
        self._iterative_scan: bool | None = None

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
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._conn_manager.acquire() as conn:
            await conn.execute(
                """INSERT INTO embeddings
                       (item_id, embedding, metadata, vault, namespace)
                   VALUES ($1, $2::vector, $3, $4, $5)
                   ON CONFLICT (vault, item_id) DO UPDATE SET
                       embedding = EXCLUDED.embedding,
                       metadata = EXCLUDED.metadata,
                       namespace = EXCLUDED.namespace""",
                item_id,
                vec_str,
                json.dumps(metadata or {}),
                self._vault,
                normalize_namespace(namespace),
            )

    async def _enable_iterative_scan(self, conn: Any) -> None:
        """Best-effort pgvector >= 0.8 iterative scan for filtered recall.

        Filtering over HNSW is a post-filter; iterative scans keep
        scanning until LIMIT is satisfied. On older pgvector the SET
        fails and we degrade to the classic behavior (recall bounded by
        ef_search). The pool RESETs settings on release, so this never
        leaks across acquirers.
        """
        if self._iterative_scan is False:
            return
        try:
            await conn.execute("SET hnsw.iterative_scan = relaxed_order")
            self._iterative_scan = True
        except Exception:
            if self._iterative_scan is None:
                logger.info(
                    "pgvector iterative scans unavailable (needs >= 0.8.0); "
                    "filtered-search recall is bounded by hnsw.ef_search",
                )
            self._iterative_scan = False

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        p = _Params(vec_str, top_k)
        clauses = [f"vault = {p.add(self._vault)}"]
        clauses += _pg_where_clauses(where, p)
        clauses += _pg_scope_clauses(scope, p)
        # relaxed_order returns slightly out-of-order rows: re-sort in an
        # outer query so callers keep exact descending-score ordering.
        sql = (
            "SELECT item_id, score FROM ("  # noqa: S608 -- fixed templates, values as $n parameters
            "SELECT item_id, 1 - (embedding <=> $1::vector) AS score "
            "FROM embeddings WHERE " + " AND ".join(clauses) + " "
            "ORDER BY embedding <=> $1::vector LIMIT $2"
            ") sub ORDER BY score DESC"
        )
        async with self._conn_manager.acquire() as conn:
            if where or scope is not None:
                await self._enable_iterative_scan(conn)
            rows = await conn.fetch(sql, *p.values)
            return [(row["item_id"], row["score"]) for row in rows]

    async def delete(self, item_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM embeddings WHERE item_id = $1 AND vault = $2",
                item_id,
                self._vault,
            )
            # asyncpg returns "DELETE N" where N is rows affected
            return int(result.split()[-1]) > 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vault={self._vault!r})"
