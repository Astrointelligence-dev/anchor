"""In-memory storage implementations for development and testing.

These are the default backends -- no external dependencies needed.
Production users provide their own implementations (Redis, Postgres, etc.)
that satisfy the storage protocols.
"""

from __future__ import annotations

import heapq
import logging
import threading
from datetime import UTC, datetime
from typing import Any, Literal

from anchor._math import cosine_similarity
from anchor.models.context import ContextItem
from anchor.models.graph import GraphEdge, GraphNode, Subgraph, require_aware
from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
    validate_vault,
)
from anchor.storage._graph_sql import (
    check_new_edge,
    merge_edge,
    merge_node,
    visible_edges,
    visible_nodes,
)
from anchor.storage._where import matches_where

logger = logging.getLogger(__name__)


class InMemoryContextStore:
    """Dict-backed context store. Implements ContextStore protocol.

    Bound to one vault at construction (front #3): items are stamped
    with the store's vault on write, and only that vault is visible —
    the mount is the isolation boundary, not the caller's queries.
    """

    __slots__ = ("_items", "_lock", "_vault")

    def __init__(self, *, vault: str = DEFAULT_VAULT) -> None:
        self._items: dict[str, ContextItem] = {}
        self._lock = threading.Lock()
        self._vault = validate_vault(vault)

    @property
    def vault(self) -> str:
        return self._vault

    def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> ContextItem | None:
        with self._lock:
            return self._items.get(item_id)

    def get_all(self) -> list[ContextItem]:
        with self._lock:
            return list(self._items.values())

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(items={len(self._items)})"


class InMemoryVectorStore:
    """Brute-force cosine similarity vector store.

    For development/testing only. Production use should provide
    FAISS, Chroma, Qdrant, etc. via the VectorStore protocol.
    """

    __slots__ = (
        "_embeddings",
        "_large_store_warned",
        "_lock",
        "_metadata",
        "_namespaces",
        "_vault",
    )

    _LARGE_STORE_THRESHOLD: int = 5000

    def __init__(self, *, vault: str = DEFAULT_VAULT) -> None:
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._namespaces: dict[str, str] = {}
        self._large_store_warned: bool = False
        self._lock = threading.Lock()
        self._vault = validate_vault(vault)

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
        with self._lock:
            self._embeddings[item_id] = embedding
            self._namespaces[item_id] = normalize_namespace(namespace)
            if metadata:
                self._metadata[item_id] = metadata

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._embeddings:
                return []
            n = len(self._embeddings)
            if n > self._LARGE_STORE_THRESHOLD and not self._large_store_warned:
                logger.warning(
                    "InMemoryVectorStore has %d embeddings. Consider using a dedicated "
                    "vector database (FAISS, Chroma) for better performance.",
                    n,
                )
                self._large_store_warned = True
            results: list[tuple[str, float]] = []
            for item_id, emb in self._embeddings.items():
                if scope is not None and not scope.matches(
                    self._namespaces.get(item_id, ROOT_NAMESPACE)
                ):
                    continue
                if where is not None and not matches_where(self._metadata.get(item_id, {}), where):
                    continue
                score = cosine_similarity(query_embedding, emb)
                results.append((item_id, score))
            return heapq.nlargest(top_k, results, key=lambda x: x[1])

    def delete(self, item_id: str) -> bool:
        with self._lock:
            removed = self._embeddings.pop(item_id, None) is not None
            self._metadata.pop(item_id, None)
            self._namespaces.pop(item_id, None)
            return removed

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors without numpy.

        Delegates to :func:`anchor._math.cosine_similarity`.
        Kept for backwards compatibility with code that calls this static method.
        """
        return cosine_similarity(a, b)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embeddings={len(self._embeddings)})"


class InMemoryDocumentStore:
    """Dict-backed document store. Implements DocumentStore protocol."""

    __slots__ = ("_documents", "_lock", "_metadata")

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._documents[doc_id] = content
            if metadata:
                self._metadata[doc_id] = metadata

    def get_document(self, doc_id: str) -> str | None:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self) -> list[str]:
        with self._lock:
            return list(self._documents.keys())

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(doc_id, None) is not None
            self._metadata.pop(doc_id, None)
            return removed

    def __repr__(self) -> str:
        return f"{type(self).__name__}(documents={len(self._documents)})"


class InMemoryGraphStore:
    """Dict-backed knowledge graph. Implements GraphStore protocol.

    The reference the SQL backends are tested against: the visibility and
    merge rules come from :mod:`anchor.storage._graph_sql`, so every backend
    applies literally the same code to its rows. Edges are invalidated in
    place and never deleted; a live ``(source, relation, target)`` is unique
    — adding it again reinforces the existing edge.
    """

    __slots__ = (
        "_edges",
        "_in",
        "_item_edges",
        "_item_nodes",
        "_item_ns",
        "_live",
        "_lock",
        "_node_items",
        "_nodes",
        "_out",
        "_vault",
        "_version",
    )

    def __init__(self, *, vault: str = DEFAULT_VAULT) -> None:
        self._vault = validate_vault(vault)
        self._lock = threading.RLock()
        self._version = 0
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._live: dict[tuple[str, str, str], str] = {}
        # insertion-ordered "sets" (dict keys) so reads are deterministic
        self._out: dict[str, dict[str, None]] = {}
        self._in: dict[str, dict[str, None]] = {}
        self._item_ns: dict[str, str] = {}
        self._item_nodes: dict[str, dict[str, None]] = {}
        self._item_edges: dict[str, dict[str, None]] = {}
        self._node_items: dict[str, dict[str, None]] = {}

    @property
    def vault(self) -> str:
        return self._vault

    @property
    def version(self) -> int:
        return self._version

    # -- writes ---------------------------------------------------------

    def upsert_node(self, node: GraphNode) -> GraphNode:
        with self._lock:
            current = self._nodes.get(node.id)
            stored = node if current is None else merge_node(current, node)
            self._nodes[node.id] = stored
            self._version += 1
            return stored

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def _link(self, node_id: str, item_id: str) -> None:
        self._node_items.setdefault(node_id, {})[item_id] = None
        self._item_nodes.setdefault(item_id, {})[node_id] = None

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        with self._lock:
            check_new_edge(edge)
            missing = [i for i in edge.evidence if i not in self._item_ns]
            if missing:
                msg = (
                    f"evidence items are not linked to the graph: {missing} (call link_item first)"
                )
                raise ValueError(msg)
            for node_id in (edge.source, edge.target):
                if node_id not in self._nodes:
                    self._nodes[node_id] = GraphNode(id=node_id)
                for item_id in edge.evidence:  # the item mentions both endpoints
                    self._link(node_id, item_id)
            key = (edge.source, edge.relation, edge.target)
            live_id = self._live.get(key)
            if live_id is not None:
                stored = merge_edge(self._edges[live_id], edge)
                self._edges[live_id] = stored
            else:
                if edge.id in self._edges:
                    msg = f"edge id '{edge.id}' already exists"
                    raise ValueError(msg)
                stored = edge
                self._edges[edge.id] = edge
                self._live[key] = edge.id
                self._out.setdefault(edge.source, {})[edge.id] = None
                self._in.setdefault(edge.target, {})[edge.id] = None
            for item_id in stored.evidence:
                self._item_edges.setdefault(item_id, {})[stored.id] = None
            self._version += 1
            return stored

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        return self._edges.get(edge_id)

    def invalidate_edge(self, edge_id: str, *, at: datetime | None = None) -> bool:
        with self._lock:
            stamp = require_aware(at, "at") if at is not None else datetime.now(UTC)
            assert stamp is not None  # noqa: S101 -- narrowed above
            return self._invalidate(edge_id, stamp)

    def _invalidate(self, edge_id: str, at: datetime) -> bool:
        edge = self._edges.get(edge_id)
        if edge is None or edge.invalidated_at is not None:
            return False
        self._edges[edge_id] = edge.model_copy(update={"invalidated_at": at})
        self._live.pop((edge.source, edge.relation, edge.target), None)
        self._version += 1
        return True

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            now = datetime.now(UTC)
            for edge_id in (*self._out.get(node_id, ()), *self._in.get(node_id, ())):
                self._invalidate(edge_id, now)
            for item_id in self._node_items.pop(node_id, {}):
                self._item_nodes.get(item_id, {}).pop(node_id, None)
            del self._nodes[node_id]
            self._version += 1
            return True

    def link_item(self, node_id: str, item_id: str, namespace: str = ROOT_NAMESPACE) -> None:
        with self._lock:
            if node_id not in self._nodes:
                msg = f"node '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            # The latest link decides the item's namespace — re-indexing a
            # moved document must move its evidence, as the ContextStore does.
            self._item_ns[item_id] = normalize_namespace(namespace)
            self._link(node_id, item_id)
            self._version += 1

    def unlink_item(self, item_id: str) -> int:
        with self._lock:
            if self._item_ns.pop(item_id, None) is None:
                return 0
            for node_id in self._item_nodes.pop(item_id, {}):
                self._node_items.get(node_id, {}).pop(item_id, None)
            invalidated = 0
            now = datetime.now(UTC)
            for edge_id in self._item_edges.pop(item_id, {}):
                edge = self._edges[edge_id]
                remaining = tuple(i for i in edge.evidence if i != item_id)
                self._edges[edge_id] = edge.model_copy(update={"evidence": remaining})
                if not remaining and self._invalidate(edge_id, now):
                    invalidated += 1
            self._version += 1
            return invalidated

    def clear(self) -> None:
        with self._lock:
            for table in (
                self._nodes,
                self._edges,
                self._live,
                self._out,
                self._in,
                self._item_ns,
                self._item_nodes,
                self._item_edges,
                self._node_items,
            ):
                table.clear()
            self._version += 1

    # -- reads ----------------------------------------------------------

    def _visible(self, scope: RetrievalScope | None) -> tuple[set[str] | None, bool]:
        """(visible item ids or None when unscoped, whether the root namespace is visible)."""
        if scope is None:
            return None, True
        return {i for i, ns in self._item_ns.items() if scope.matches(ns)}, scope.matches(
            ROOT_NAMESPACE
        )

    def node_items(self, node_id: str, *, scope: RetrievalScope | None = None) -> list[str]:
        with self._lock:
            visible, _ = self._visible(scope)
            items = self._node_items.get(node_id, ())
            return [i for i in items if visible is None or i in visible]

    def edges_of(
        self,
        node_id: str,
        *,
        direction: Literal["out", "in", "both"] = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        with self._lock:
            ids: dict[str, None] = {}
            if direction in ("out", "both"):
                ids.update(self._out.get(node_id, {}))
            if direction in ("in", "both"):
                ids.update(self._in.get(node_id, {}))
            visible, root = self._visible(scope)
            edges = (self._edges[i] for i in ids)
            return visible_edges(edges, self._node_items, visible, as_of, root)

    def subgraph(
        self,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> Subgraph:
        with self._lock:
            visible, root = self._visible(scope)
            kept = visible_nodes(self._nodes, self._node_items, visible, root)
            nodes = {n: self._nodes[n] for n in kept}
            edges = [e for e in self._edges.values() if e.source in nodes and e.target in nodes]
            items = {
                n: tuple(i for i in self._node_items.get(n, ()) if visible is None or i in visible)
                for n in nodes
            }
            return Subgraph(
                nodes=nodes,
                edges=visible_edges(edges, self._node_items, visible, as_of, root),
                items=items,
            )

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(vault={self._vault!r}, nodes={len(self._nodes)}, "
            f"edges={len(self._live)})"
        )
