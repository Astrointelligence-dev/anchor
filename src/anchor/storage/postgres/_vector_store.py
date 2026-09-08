"""PostgreSQL-backed VectorStore using pgvector."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
    validate_vault,
)
from anchor.storage._where import (
    SET_OPS,
    SQL_OPS,
    check_operator,
    is_op_dict,
    scope_sql_clauses,
)

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
                # Ranges compare only like with like; a CASE guard is the
                # documented way to keep the cast from ever seeing a
                # non-number (evaluation order is otherwise undefined).
                sym = SQL_OPS[op]
                kp = p.add(key)
                if isinstance(operand, bool) or not isinstance(
                    operand, (int, float)
                ):
                    clauses.append(
                        f"(CASE WHEN jsonb_typeof(metadata->{kp}) = 'string' "
                        f"THEN metadata->>{kp} END) {sym} {p.add(str(operand))}"
                    )
                else:
                    clauses.append(
                        f"(CASE WHEN jsonb_typeof(metadata->{kp}) = 'number' "
                        f"THEN (metadata->>{kp})::numeric END) {sym} {p.add(operand)}"
                    )
    return clauses


def _numbered(clauses: list[str], params: list[Any], p: _Params) -> list[str]:
    """Renumber the shared compiler's ``?`` placeholders as ``$n``.

    Scope clauses only: the JSONB where-compiler above uses ``?`` as the
    key-exists operator, so it keeps its own numbering.
    """
    values = iter(params)
    return [re.sub(r"\?", lambda _m: p.add(next(values)), c) for c in clauses]


class PostgresVectorStore:
    """Async PostgreSQL-backed vector store using pgvector.

    Uses the ``<=>`` cosine distance operator for similarity search.
    Similarity scores are computed as ``1 - distance``.

    Implements the AsyncVectorStore protocol.
    """

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self,
        conn_manager: PostgresConnectionManager,
        *,
        vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = validate_vault(vault)

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
        # Every search is a filtered HNSW search (the vault bind), so the
        # pool sets hnsw.iterative_scan once per session (pgvector >= 0.8;
        # see PostgresConnectionManager) — recall is not bounded by
        # ef_search for a small vault sharing the table with a big one.
        clauses = [f"vault = {p.add(self._vault)}"]
        clauses += _pg_where_clauses(where, p)
        clauses += _numbered(*scope_sql_clauses(scope, "namespace"), p)
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
