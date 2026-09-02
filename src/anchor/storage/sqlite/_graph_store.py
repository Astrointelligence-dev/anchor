"""SQLite-backed GraphStore (roadmap #4, phase C).

Cheap indexed queries fetch rows; the visibility rules run in Python
through :mod:`anchor.storage._graph_sql`, the same code every SQL
backend shares — so a scope means exactly the same thing here as on the
in-memory reference. Edges are invalidated in place; the live
``(source, relation, target)`` is unique through a partial index.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from anchor.models.graph import GraphEdge, GraphNode, Subgraph
from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
    validate_vault,
)
from anchor.storage._graph_sql import (
    EDGE_COLUMNS,
    edge_to_row,
    merge_edge,
    merge_node,
    node_to_row,
    row_to_edge,
    row_to_node,
    visible_edges,
    visible_nodes,
)
from anchor.storage._where import scope_sql_clauses

if TYPE_CHECKING:
    import sqlite3

    from anchor.storage.sqlite._connection import SqliteConnectionManager

_EDGE_SELECT = "SELECT " + ", ".join(EDGE_COLUMNS) + " FROM graph_edges"  # noqa: S608 -- fixed column names
_EDGE_INSERT = (
    "INSERT INTO graph_edges (vault, "  # noqa: S608 -- fixed column names, placeholders only
    + ", ".join(EDGE_COLUMNS)
    + ") VALUES (?, "
    + ", ".join("?" * len(EDGE_COLUMNS))
    + ")"
)


def _marks(n: int) -> str:
    return ", ".join("?" * n)


class SqliteGraphStore:
    """SQLite-backed knowledge graph. Implements the GraphStore protocol.

    Thread-safe within a process (one lock around every operation — the
    manager's connections are per-thread, the lock keeps multi-statement
    writes atomic against each other).
    """

    __slots__ = ("_conn_manager", "_lock", "_vault")

    def __init__(
        self, conn_manager: SqliteConnectionManager, *, vault: str = DEFAULT_VAULT
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = validate_vault(vault)
        self._lock = threading.RLock()

    @property
    def vault(self) -> str:
        return self._vault

    @property
    def version(self) -> int:
        row = (
            self._conn()
            .execute("SELECT version FROM graph_meta WHERE vault = ?", (self._vault,))
            .fetchone()
        )
        return int(row[0]) if row is not None else 0

    def _conn(self) -> sqlite3.Connection:
        return self._conn_manager.get_connection()

    def _bump(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO graph_meta (vault, version) VALUES (?, 1) "
            "ON CONFLICT(vault) DO UPDATE SET version = version + 1",
            (self._vault,),
        )
        conn.commit()

    # -- writes ---------------------------------------------------------

    def upsert_node(self, node: GraphNode) -> GraphNode:
        with self._lock:
            conn = self._conn()
            current = self.get_node(node.id)
            stored = node if current is None else merge_node(current, node)
            row = node_to_row(stored)
            conn.execute(
                "INSERT OR REPLACE INTO graph_nodes "
                "(vault, id, label, aliases_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (self._vault, row["id"], row["label"], row["aliases_json"], row["metadata_json"]),
            )
            self._bump(conn)
            return stored

    def get_node(self, node_id: str) -> GraphNode | None:
        row = (
            self._conn()
            .execute(
                "SELECT id, label, aliases_json, metadata_json FROM graph_nodes "
                "WHERE vault = ? AND id = ?",
                (self._vault, node_id),
            )
            .fetchone()
        )
        return None if row is None else row_to_node(row)

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        with self._lock:
            conn = self._conn()
            if edge.evidence:
                known = {
                    r[0]
                    for r in conn.execute(
                        "SELECT item_id FROM graph_items WHERE vault = ? "  # noqa: S608 -- placeholders only
                        f"AND item_id IN ({_marks(len(edge.evidence))})",
                        (self._vault, *edge.evidence),
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
                conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (vault, id, label) VALUES (?, ?, ?)",
                    (self._vault, node_id, node_id),
                )
            live = conn.execute(
                _EDGE_SELECT + " WHERE vault = ? AND source = ? AND relation = ? "
                "AND target = ? AND invalidated_at IS NULL",
                (self._vault, edge.source, edge.relation, edge.target),
            ).fetchone()
            if live is not None:
                evidence = self._evidence_of(conn, [live["id"]]).get(live["id"], [])
                current = row_to_edge(live, evidence)
                stored = merge_edge(current, edge)
                conn.execute(
                    "UPDATE graph_edges SET confidence = ?, provenance = ?, fact = ?, "
                    "metadata_json = ? WHERE vault = ? AND id = ?",
                    (
                        stored.confidence,
                        stored.provenance,
                        stored.fact,
                        edge_to_row(stored)["metadata_json"],
                        self._vault,
                        stored.id,
                    ),
                )
            else:
                if self.get_edge(edge.id) is not None:
                    msg = f"edge id '{edge.id}' already exists"
                    raise ValueError(msg)
                stored = edge
                row = edge_to_row(edge)
                conn.execute(_EDGE_INSERT, (self._vault, *[row[c] for c in EDGE_COLUMNS]))
            conn.executemany(
                "INSERT OR IGNORE INTO graph_edge_items (vault, edge_id, item_id) VALUES (?, ?, ?)",
                [(self._vault, stored.id, i) for i in stored.evidence],
            )
            self._bump(conn)
            return stored

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        conn = self._conn()
        row = conn.execute(
            _EDGE_SELECT + " WHERE vault = ? AND id = ?", (self._vault, edge_id)
        ).fetchone()
        if row is None:
            return None
        return row_to_edge(row, self._evidence_of(conn, [edge_id]).get(edge_id, []))

    def invalidate_edge(self, edge_id: str, *, at: datetime | None = None) -> bool:
        with self._lock:
            conn = self._conn()
            stamp = (at if at is not None else datetime.now(UTC)).isoformat()
            cursor = conn.execute(
                "UPDATE graph_edges SET invalidated_at = ? "
                "WHERE vault = ? AND id = ? AND invalidated_at IS NULL",
                (stamp, self._vault, edge_id),
            )
            if cursor.rowcount == 0:
                conn.commit()
                return False
            self._bump(conn)
            return True

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            conn = self._conn()
            if self.get_node(node_id) is None:
                return False
            conn.execute(
                "UPDATE graph_edges SET invalidated_at = ? WHERE vault = ? "
                "AND (source = ? OR target = ?) AND invalidated_at IS NULL",
                (datetime.now(UTC).isoformat(), self._vault, node_id, node_id),
            )
            conn.execute(
                "DELETE FROM graph_node_items WHERE vault = ? AND node_id = ?",
                (self._vault, node_id),
            )
            conn.execute(
                "DELETE FROM graph_nodes WHERE vault = ? AND id = ?", (self._vault, node_id)
            )
            self._bump(conn)
            return True

    def link_item(self, node_id: str, item_id: str, namespace: str = ROOT_NAMESPACE) -> None:
        with self._lock:
            conn = self._conn()
            if self.get_node(node_id) is None:
                msg = f"node '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            ns = normalize_namespace(namespace)
            known = conn.execute(
                "SELECT namespace FROM graph_items WHERE vault = ? AND item_id = ?",
                (self._vault, item_id),
            ).fetchone()
            if known is not None and known[0] != ns:
                msg = f"item '{item_id}' is already linked under namespace '{known[0]}', not '{ns}'"
                raise ValueError(msg)
            conn.execute(
                "INSERT OR IGNORE INTO graph_items (vault, item_id, namespace) VALUES (?, ?, ?)",
                (self._vault, item_id, ns),
            )
            conn.execute(
                "INSERT OR IGNORE INTO graph_node_items (vault, node_id, item_id) VALUES (?, ?, ?)",
                (self._vault, node_id, item_id),
            )
            self._bump(conn)

    def unlink_item(self, item_id: str) -> int:
        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                "DELETE FROM graph_items WHERE vault = ? AND item_id = ?", (self._vault, item_id)
            )
            if cursor.rowcount == 0:
                conn.commit()
                return 0
            conn.execute(
                "DELETE FROM graph_node_items WHERE vault = ? AND item_id = ?",
                (self._vault, item_id),
            )
            edge_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT edge_id FROM graph_edge_items WHERE vault = ? AND item_id = ?",
                    (self._vault, item_id),
                )
            ]
            conn.execute(
                "DELETE FROM graph_edge_items WHERE vault = ? AND item_id = ?",
                (self._vault, item_id),
            )
            invalidated = 0
            stamp = datetime.now(UTC).isoformat()
            for edge_id in edge_ids:
                left = conn.execute(
                    "SELECT 1 FROM graph_edge_items WHERE vault = ? AND edge_id = ? LIMIT 1",
                    (self._vault, edge_id),
                ).fetchone()
                if left is None:
                    done = conn.execute(
                        "UPDATE graph_edges SET invalidated_at = ? "
                        "WHERE vault = ? AND id = ? AND invalidated_at IS NULL",
                        (stamp, self._vault, edge_id),
                    )
                    invalidated += done.rowcount
            self._bump(conn)
            return invalidated

    def clear(self) -> None:
        with self._lock:
            conn = self._conn()
            for table in (
                "graph_nodes",
                "graph_edges",
                "graph_items",
                "graph_node_items",
                "graph_edge_items",
            ):
                conn.execute(f"DELETE FROM {table} WHERE vault = ?", (self._vault,))  # noqa: S608 -- fixed table names
            self._bump(conn)

    # -- reads ----------------------------------------------------------

    def _visible(self, conn: sqlite3.Connection, scope: RetrievalScope | None) -> set[str] | None:
        if scope is None:
            return None
        clauses, params = scope_sql_clauses(scope, "namespace")
        sql = "SELECT item_id FROM graph_items WHERE vault = ?" + "".join(  # noqa: S608 -- compiled clauses, placeholders only
            f" AND {c}" for c in clauses
        )
        return {r[0] for r in conn.execute(sql, (self._vault, *params))}

    def _node_items_of(
        self, conn: sqlite3.Connection, node_ids: Iterable[str] | None
    ) -> dict[str, list[str]]:
        sql = "SELECT node_id, item_id FROM graph_node_items WHERE vault = ?"
        params: list[Any] = [self._vault]
        if node_ids is not None:
            ids = list(dict.fromkeys(node_ids))
            if not ids:
                return {}
            sql += f" AND node_id IN ({_marks(len(ids))})"
            params.extend(ids)
        out: dict[str, list[str]] = {}
        for node_id, item_id in conn.execute(sql + " ORDER BY rowid", params):
            out.setdefault(node_id, []).append(item_id)
        return out

    def _evidence_of(
        self, conn: sqlite3.Connection, edge_ids: Iterable[str] | None
    ) -> dict[str, list[str]]:
        sql = "SELECT edge_id, item_id FROM graph_edge_items WHERE vault = ?"
        params: list[Any] = [self._vault]
        if edge_ids is not None:
            ids = list(dict.fromkeys(edge_ids))
            if not ids:
                return {}
            sql += f" AND edge_id IN ({_marks(len(ids))})"
            params.extend(ids)
        out: dict[str, list[str]] = {}
        for edge_id, item_id in conn.execute(sql + " ORDER BY rowid", params):
            out.setdefault(edge_id, []).append(item_id)
        return out

    def node_items(self, node_id: str, *, scope: RetrievalScope | None = None) -> list[str]:
        conn = self._conn()
        visible = self._visible(conn, scope)
        items = self._node_items_of(conn, [node_id]).get(node_id, [])
        return [i for i in items if visible is None or i in visible]

    def edges_of(
        self,
        node_id: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        conn = self._conn()
        where = {
            "out": "source = ?",
            "in": "target = ?",
            "both": "(source = ? OR target = ?)",
        }[direction]
        # Outgoing edges first, then incoming, insertion order inside each:
        # the reference (in-memory) order, so every backend walks alike.
        params: tuple[Any, ...] = (self._vault, node_id)
        order = "rowid"
        if direction == "both":
            params = (self._vault, node_id, node_id, node_id)
            order = "CASE WHEN source = ? THEN 0 ELSE 1 END, rowid"
        rows = conn.execute(
            f"{_EDGE_SELECT} WHERE vault = ? AND {where} AND invalidated_at IS NULL "
            f"ORDER BY {order}",
            params,
        ).fetchall()
        evidence = self._evidence_of(conn, [r["id"] for r in rows])
        edges = [row_to_edge(r, evidence.get(r["id"], [])) for r in rows]
        visible = self._visible(conn, scope)
        node_items: dict[str, list[str]] = {}
        if visible is not None:
            ends = [e.source for e in edges] + [e.target for e in edges]
            node_items = self._node_items_of(conn, ends)
        return visible_edges(edges, node_items, visible, as_of)

    def subgraph(
        self,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> Subgraph:
        conn = self._conn()
        visible = self._visible(conn, scope)
        node_items = self._node_items_of(conn, None)
        all_nodes = [
            row_to_node(r)
            for r in conn.execute(
                "SELECT id, label, aliases_json, metadata_json FROM graph_nodes "
                "WHERE vault = ? ORDER BY rowid",
                (self._vault,),
            )
        ]
        kept = set(visible_nodes([n.id for n in all_nodes], node_items, visible))
        nodes = {n.id: n for n in all_nodes if n.id in kept}
        rows = conn.execute(
            _EDGE_SELECT + " WHERE vault = ? AND invalidated_at IS NULL ORDER BY rowid",
            (self._vault,),
        ).fetchall()
        evidence = self._evidence_of(conn, None)
        edges = [row_to_edge(r, evidence.get(r["id"], [])) for r in rows]
        edges = [e for e in edges if e.source in nodes and e.target in nodes]
        items = {
            n: tuple(i for i in node_items.get(n, []) if visible is None or i in visible)
            for n in nodes
        }
        return Subgraph(
            nodes=nodes, edges=visible_edges(edges, node_items, visible, as_of), items=items
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db={self._conn_manager.db_path!s}, vault={self._vault!r})"


class AsyncSqliteGraphStore:
    """Async twin of :class:`SqliteGraphStore`. Implements AsyncGraphStore.

    Runs the sync store in a worker thread (``asyncio.to_thread``) — which
    is what aiosqlite does underneath anyway, and the connection manager
    hands every thread its own connection.
    """

    # ponytail: a native aiosqlite twin when the graph sits on a hot async path.
    __slots__ = ("_sync",)

    def __init__(
        self, conn_manager: SqliteConnectionManager, *, vault: str = DEFAULT_VAULT
    ) -> None:
        self._sync = SqliteGraphStore(conn_manager, vault=vault)

    @property
    def vault(self) -> str:
        return self._sync.vault

    @property
    def version(self) -> int:
        return self._sync.version

    async def upsert_node(self, node: GraphNode) -> GraphNode:
        return await asyncio.to_thread(self._sync.upsert_node, node)

    async def get_node(self, node_id: str) -> GraphNode | None:
        return await asyncio.to_thread(self._sync.get_node, node_id)

    async def add_edge(self, edge: GraphEdge) -> GraphEdge:
        return await asyncio.to_thread(self._sync.add_edge, edge)

    async def get_edge(self, edge_id: str) -> GraphEdge | None:
        return await asyncio.to_thread(self._sync.get_edge, edge_id)

    async def invalidate_edge(self, edge_id: str, *, at: datetime | None = None) -> bool:
        return await asyncio.to_thread(self._sync.invalidate_edge, edge_id, at=at)

    async def remove_node(self, node_id: str) -> bool:
        return await asyncio.to_thread(self._sync.remove_node, node_id)

    async def link_item(self, node_id: str, item_id: str, namespace: str = ROOT_NAMESPACE) -> None:
        await asyncio.to_thread(self._sync.link_item, node_id, item_id, namespace)

    async def unlink_item(self, item_id: str) -> int:
        return await asyncio.to_thread(self._sync.unlink_item, item_id)

    async def node_items(self, node_id: str, *, scope: RetrievalScope | None = None) -> list[str]:
        return await asyncio.to_thread(self._sync.node_items, node_id, scope=scope)

    async def edges_of(
        self,
        node_id: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        return await asyncio.to_thread(
            self._sync.edges_of, node_id, direction=direction, scope=scope, as_of=as_of
        )

    async def subgraph(
        self,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> Subgraph:
        return await asyncio.to_thread(self._sync.subgraph, scope=scope, as_of=as_of)

    async def clear(self) -> None:
        await asyncio.to_thread(self._sync.clear)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._sync!r})"
