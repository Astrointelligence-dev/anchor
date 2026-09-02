"""PostgreSQL-backed GraphStore (roadmap #4, phase C). Async, asyncpg.

Same tables and the same Python visibility rules as the SQLite backend
(:mod:`anchor.storage._graph_sql`); ``namespace`` is ``COLLATE "C"`` so the
boundary-aware prefix ranges hold under any database collation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from anchor.models.graph import GraphEdge, GraphNode, Subgraph
from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
    validate_vault,
)
from anchor.storage._graph_sql import (
    merge_edge,
    merge_node,
    row_to_edge,
    row_to_node,
    visible_edges,
    visible_nodes,
)
from anchor.storage._where import scope_sql_clauses
from anchor.storage.postgres._vector_store import _numbered, _Params

if TYPE_CHECKING:
    import asyncpg

    from anchor.storage.postgres._connection import PostgresConnectionManager

_NODE_SELECT = (
    "SELECT id, label, aliases AS aliases_json, metadata AS metadata_json FROM graph_nodes"
)
_EDGE_SELECT = (
    "SELECT id, source, target, relation, fact, provenance, confidence, valid_from, "
    "valid_to, created_at, invalidated_at, metadata AS metadata_json FROM graph_edges"
)


class PostgresGraphStore:
    """Async PostgreSQL-backed knowledge graph. Implements AsyncGraphStore."""

    # ponytail: `version` is an in-process write counter (derived caches
    # live in-process); a graph_meta row when another process must see it.
    __slots__ = ("_conn_manager", "_vault", "_version")

    def __init__(
        self, conn_manager: PostgresConnectionManager, *, vault: str = DEFAULT_VAULT
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = validate_vault(vault)
        self._version = 0

    @property
    def vault(self) -> str:
        return self._vault

    @property
    def version(self) -> int:
        return self._version

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vault={self._vault!r})"

    # -- writes ---------------------------------------------------------

    async def upsert_node(self, node: GraphNode) -> GraphNode:
        async with self._conn_manager.acquire() as conn:
            current = await self._get_node(conn, node.id)
            stored = node if current is None else merge_node(current, node)
            await conn.execute(
                "INSERT INTO graph_nodes (vault, id, label, aliases, metadata) "
                "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb) "
                "ON CONFLICT (vault, id) DO UPDATE SET label = EXCLUDED.label, "
                "aliases = EXCLUDED.aliases, metadata = EXCLUDED.metadata",
                self._vault,
                stored.id,
                stored.label,
                json.dumps(list(stored.aliases)),
                json.dumps(stored.metadata, default=str),
            )
        self._version += 1
        return stored

    async def _get_node(self, conn: asyncpg.Connection, node_id: str) -> GraphNode | None:
        row = await conn.fetchrow(
            _NODE_SELECT + " WHERE vault = $1 AND id = $2", self._vault, node_id
        )
        return None if row is None else row_to_node(row)

    async def get_node(self, node_id: str) -> GraphNode | None:
        async with self._conn_manager.acquire() as conn:
            return await self._get_node(conn, node_id)

    async def add_edge(self, edge: GraphEdge) -> GraphEdge:
        async with self._conn_manager.acquire() as conn, conn.transaction():
            if edge.evidence:
                known = {
                    r["item_id"]
                    for r in await conn.fetch(
                        "SELECT item_id FROM graph_items WHERE vault = $1 AND item_id = ANY($2)",
                        self._vault,
                        list(edge.evidence),
                    )
                }
                missing = [i for i in edge.evidence if i not in known]
                if missing:
                    msg = (
                        f"evidence items are not linked to the graph: {missing} "
                        "(call link_item first)"
                    )
                    raise ValueError(msg)
            for node_id in (edge.source, edge.target):
                await conn.execute(
                    "INSERT INTO graph_nodes (vault, id, label) VALUES ($1, $2, $2) "
                    "ON CONFLICT DO NOTHING",
                    self._vault,
                    node_id,
                )
            live = await conn.fetchrow(
                _EDGE_SELECT + " WHERE vault = $1 AND source = $2 AND relation = $3 "
                "AND target = $4 AND invalidated_at IS NULL",
                self._vault,
                edge.source,
                edge.relation,
                edge.target,
            )
            if live is not None:
                evidence = (await self._evidence_of(conn, [live["id"]])).get(live["id"], [])
                stored = merge_edge(row_to_edge(live, evidence), edge)
                await conn.execute(
                    "UPDATE graph_edges SET confidence = $1, provenance = $2, fact = $3, "
                    "metadata = $4::jsonb WHERE vault = $5 AND id = $6",
                    stored.confidence,
                    stored.provenance,
                    stored.fact,
                    json.dumps(stored.metadata, default=str),
                    self._vault,
                    stored.id,
                )
            else:
                exists = await conn.fetchval(
                    "SELECT 1 FROM graph_edges WHERE vault = $1 AND id = $2", self._vault, edge.id
                )
                if exists is not None:
                    msg = f"edge id '{edge.id}' already exists"
                    raise ValueError(msg)
                stored = edge
                await conn.execute(
                    "INSERT INTO graph_edges (vault, id, source, target, relation, fact, "
                    "provenance, confidence, valid_from, valid_to, created_at, "
                    "invalidated_at, metadata) VALUES "
                    "($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)",
                    self._vault,
                    edge.id,
                    edge.source,
                    edge.target,
                    edge.relation,
                    edge.fact,
                    edge.provenance,
                    edge.confidence,
                    edge.valid_from,
                    edge.valid_to,
                    edge.created_at,
                    edge.invalidated_at,
                    json.dumps(edge.metadata, default=str),
                )
            for item_id in stored.evidence:
                await conn.execute(
                    "INSERT INTO graph_edge_items (vault, edge_id, item_id) VALUES ($1, $2, $3) "
                    "ON CONFLICT DO NOTHING",
                    self._vault,
                    stored.id,
                    item_id,
                )
        self._version += 1
        return stored

    async def get_edge(self, edge_id: str) -> GraphEdge | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                _EDGE_SELECT + " WHERE vault = $1 AND id = $2", self._vault, edge_id
            )
            if row is None:
                return None
            evidence = (await self._evidence_of(conn, [edge_id])).get(edge_id, [])
            return row_to_edge(row, evidence)

    async def invalidate_edge(self, edge_id: str, *, at: datetime | None = None) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "UPDATE graph_edges SET invalidated_at = $1 "
                "WHERE vault = $2 AND id = $3 AND invalidated_at IS NULL",
                at if at is not None else datetime.now(UTC),
                self._vault,
                edge_id,
            )
        done = int(result.split()[-1]) > 0
        if done:
            self._version += 1
        return done

    async def remove_node(self, node_id: str) -> bool:
        async with self._conn_manager.acquire() as conn, conn.transaction():
            if await self._get_node(conn, node_id) is None:
                return False
            await conn.execute(
                "UPDATE graph_edges SET invalidated_at = $1 WHERE vault = $2 "
                "AND (source = $3 OR target = $3) AND invalidated_at IS NULL",
                datetime.now(UTC),
                self._vault,
                node_id,
            )
            await conn.execute(
                "DELETE FROM graph_node_items WHERE vault = $1 AND node_id = $2",
                self._vault,
                node_id,
            )
            await conn.execute(
                "DELETE FROM graph_nodes WHERE vault = $1 AND id = $2", self._vault, node_id
            )
        self._version += 1
        return True

    async def link_item(self, node_id: str, item_id: str, namespace: str = ROOT_NAMESPACE) -> None:
        async with self._conn_manager.acquire() as conn, conn.transaction():
            if await self._get_node(conn, node_id) is None:
                msg = f"node '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            ns = normalize_namespace(namespace)
            known = await conn.fetchval(
                "SELECT namespace FROM graph_items WHERE vault = $1 AND item_id = $2",
                self._vault,
                item_id,
            )
            if known is not None and known != ns:
                msg = f"item '{item_id}' is already linked under namespace '{known}', not '{ns}'"
                raise ValueError(msg)
            await conn.execute(
                "INSERT INTO graph_items (vault, item_id, namespace) VALUES ($1, $2, $3) "
                "ON CONFLICT DO NOTHING",
                self._vault,
                item_id,
                ns,
            )
            await conn.execute(
                "INSERT INTO graph_node_items (vault, node_id, item_id) VALUES ($1, $2, $3) "
                "ON CONFLICT DO NOTHING",
                self._vault,
                node_id,
                item_id,
            )
        self._version += 1

    async def unlink_item(self, item_id: str) -> int:
        async with self._conn_manager.acquire() as conn, conn.transaction():
            result = await conn.execute(
                "DELETE FROM graph_items WHERE vault = $1 AND item_id = $2", self._vault, item_id
            )
            if int(result.split()[-1]) == 0:
                return 0
            await conn.execute(
                "DELETE FROM graph_node_items WHERE vault = $1 AND item_id = $2",
                self._vault,
                item_id,
            )
            edge_ids = [
                r["edge_id"]
                for r in await conn.fetch(
                    "SELECT edge_id FROM graph_edge_items WHERE vault = $1 AND item_id = $2",
                    self._vault,
                    item_id,
                )
            ]
            await conn.execute(
                "DELETE FROM graph_edge_items WHERE vault = $1 AND item_id = $2",
                self._vault,
                item_id,
            )
            invalidated = 0
            now = datetime.now(UTC)
            for edge_id in edge_ids:
                left = await conn.fetchval(
                    "SELECT 1 FROM graph_edge_items WHERE vault = $1 AND edge_id = $2 LIMIT 1",
                    self._vault,
                    edge_id,
                )
                if left is None:
                    done = await conn.execute(
                        "UPDATE graph_edges SET invalidated_at = $1 "
                        "WHERE vault = $2 AND id = $3 AND invalidated_at IS NULL",
                        now,
                        self._vault,
                        edge_id,
                    )
                    invalidated += int(done.split()[-1])
        self._version += 1
        return invalidated

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn, conn.transaction():
            for table in (
                "graph_nodes",
                "graph_edges",
                "graph_items",
                "graph_node_items",
                "graph_edge_items",
            ):
                await conn.execute(f"DELETE FROM {table} WHERE vault = $1", self._vault)  # noqa: S608 -- fixed table names
        self._version += 1

    # -- reads ----------------------------------------------------------

    async def _visible(
        self, conn: asyncpg.Connection, scope: RetrievalScope | None
    ) -> set[str] | None:
        if scope is None:
            return None
        p = _Params(self._vault)
        clauses, params = scope_sql_clauses(scope, "namespace")
        sql = "SELECT item_id FROM graph_items WHERE vault = $1" + "".join(  # noqa: S608 -- compiled clauses
            f" AND {c}" for c in _numbered(clauses, params, p)
        )
        return {r["item_id"] for r in await conn.fetch(sql, *p.values)}

    async def _node_items_of(
        self, conn: asyncpg.Connection, node_ids: Iterable[str] | None
    ) -> dict[str, list[str]]:
        if node_ids is None:
            rows = await conn.fetch(
                "SELECT node_id, item_id FROM graph_node_items WHERE vault = $1 ORDER BY seq",
                self._vault,
            )
        else:
            ids = list(dict.fromkeys(node_ids))
            if not ids:
                return {}
            rows = await conn.fetch(
                "SELECT node_id, item_id FROM graph_node_items "
                "WHERE vault = $1 AND node_id = ANY($2) ORDER BY seq",
                self._vault,
                ids,
            )
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["node_id"], []).append(r["item_id"])
        return out

    async def _evidence_of(
        self, conn: asyncpg.Connection, edge_ids: Iterable[str] | None
    ) -> dict[str, list[str]]:
        if edge_ids is None:
            rows = await conn.fetch(
                "SELECT edge_id, item_id FROM graph_edge_items WHERE vault = $1 ORDER BY seq",
                self._vault,
            )
        else:
            ids = list(dict.fromkeys(edge_ids))
            if not ids:
                return {}
            rows = await conn.fetch(
                "SELECT edge_id, item_id FROM graph_edge_items "
                "WHERE vault = $1 AND edge_id = ANY($2) ORDER BY seq",
                self._vault,
                ids,
            )
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["edge_id"], []).append(r["item_id"])
        return out

    async def node_items(self, node_id: str, *, scope: RetrievalScope | None = None) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            visible = await self._visible(conn, scope)
            items = (await self._node_items_of(conn, [node_id])).get(node_id, [])
            return [i for i in items if visible is None or i in visible]

    async def edges_of(
        self,
        node_id: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        async with self._conn_manager.acquire() as conn:
            where = {
                "out": "source = $2",
                "in": "target = $2",
                "both": "(source = $2 OR target = $2)",
            }[direction]
            order = "seq" if direction != "both" else "CASE WHEN source = $2 THEN 0 ELSE 1 END, seq"
            rows = await conn.fetch(
                f"{_EDGE_SELECT} WHERE vault = $1 AND {where} AND invalidated_at IS NULL "
                f"ORDER BY {order}",
                self._vault,
                node_id,
            )
            evidence = await self._evidence_of(conn, [r["id"] for r in rows])
            edges = [row_to_edge(r, evidence.get(r["id"], [])) for r in rows]
            visible = await self._visible(conn, scope)
            node_items: dict[str, list[str]] = {}
            if visible is not None:
                ends = [e.source for e in edges] + [e.target for e in edges]
                node_items = await self._node_items_of(conn, ends)
            return visible_edges(edges, node_items, visible, as_of)

    async def subgraph(
        self,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> Subgraph:
        async with self._conn_manager.acquire() as conn:
            visible = await self._visible(conn, scope)
            node_items = await self._node_items_of(conn, None)
            all_nodes = [
                row_to_node(r)
                for r in await conn.fetch(
                    _NODE_SELECT + " WHERE vault = $1 ORDER BY seq", self._vault
                )
            ]
            kept = set(visible_nodes([n.id for n in all_nodes], node_items, visible))
            nodes = {n.id: n for n in all_nodes if n.id in kept}
            rows = await conn.fetch(
                _EDGE_SELECT + " WHERE vault = $1 AND invalidated_at IS NULL ORDER BY seq",
                self._vault,
            )
            evidence = await self._evidence_of(conn, None)
            edges = [row_to_edge(r, evidence.get(r["id"], [])) for r in rows]
            edges = [e for e in edges if e.source in nodes and e.target in nodes]
            items = {
                n: tuple(i for i in node_items.get(n, []) if visible is None or i in visible)
                for n in nodes
            }
            return Subgraph(
                nodes=nodes, edges=visible_edges(edges, node_items, visible, as_of), items=items
            )
