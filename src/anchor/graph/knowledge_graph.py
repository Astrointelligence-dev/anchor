"""KnowledgeGraph: the navigation API over a ``GraphStore`` (roadmap #4).

``neighbors`` / ``backlinks`` / ``path`` / ``explain`` walk the store one
node at a time; ``query`` (personalized PageRank over the entity-item
bipartite graph, HippoRAG-2 style) and ``hubs`` load the visible subgraph
once. Zero LLM calls on the read path.

Every read takes ``scope`` (namespace navigation, a ``RetrievalScope``)
and ``as_of`` (world time). The graph never reads the agent's published
scope itself — the caller (``GraphRetriever``, tools) resolves
``effective_scope`` and passes it, the same contract the stores have.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import datetime
from itertools import pairwise
from typing import Any, Literal

from anchor.graph.algorithms import adjacency, degree, personalized_pagerank
from anchor.models.graph import GraphEdge, GraphNode, Provenance, normalize_key
from anchor.models.scope import ROOT_NAMESPACE, RetrievalScope
from anchor.protocols.storage import GraphStore
from anchor.storage.memory_store import InMemoryGraphStore

Direction = Literal["out", "in", "both"]


class KnowledgeGraph:
    """Entities, notes and the items that evidence them, navigable under a scope.

    Example::

        graph = KnowledgeGraph()                      # in-memory store
        graph.add_node("Alice", metadata={"type": "person"})
        graph.link_item("Alice", "mem-001")            # item id = the currency
        graph.add_edge("Alice", "works on", "Project X", evidence=["mem-001"])

        graph.neighbors("alice", max_depth=2)          # ["project_x"]
        graph.related_items("Project X")               # ["mem-001"]
        graph.query(["alice"], top_k=5)                # [("mem-001", score)]
    """

    __slots__ = ("_store",)

    def __init__(self, store: GraphStore | None = None) -> None:
        self._store: GraphStore = store if store is not None else InMemoryGraphStore()

    @property
    def store(self) -> GraphStore:
        return self._store

    @property
    def vault(self) -> str:
        return self._store.vault

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        *,
        label: str | None = None,
        aliases: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> GraphNode:
        """Upsert a node; metadata and aliases merge, the first label wins."""
        node = GraphNode(
            id=name,
            label=label or name.strip(),
            aliases=tuple(aliases),
            metadata=dict(metadata or {}),
        )
        return self._store.upsert_node(node)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        evidence: Iterable[str] = (),
        fact: str | None = None,
        provenance: Provenance = "extracted",
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Add (or reinforce) a directed edge; endpoint nodes are created as needed.

        Evidence items must already be linked with :meth:`link_item` — an
        edge cannot cite an item the graph does not know the namespace of.
        """
        self.add_node(source)
        self.add_node(target)
        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            fact=fact,
            provenance=provenance,
            confidence=confidence,
            evidence=tuple(evidence),
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=dict(metadata or {}),
        )
        return self._store.add_edge(edge)

    def link_item(self, name: str, item_id: str, namespace: str = ROOT_NAMESPACE) -> None:
        """Record that *item_id* (a ``ContextItem`` / ``MemoryEntry`` id) evidences the node."""
        self._store.link_item(normalize_key(name), item_id, namespace)

    def unlink_item(self, item_id: str) -> int:
        """Forget an item; edges left without evidence are invalidated. Returns that count."""
        return self._store.unlink_item(item_id)

    def invalidate_edge(self, edge_id: str, *, at: datetime | None = None) -> bool:
        return self._store.invalidate_edge(edge_id, at=at)

    def remove_node(self, name: str) -> bool:
        """Drop a node; its edges are invalidated (history kept), its links dropped."""
        return self._store.remove_node(normalize_key(name))

    def clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def node(self, name: str) -> GraphNode | None:
        return self._store.get_node(normalize_key(name))

    def nodes(self, *, scope: RetrievalScope | None = None) -> list[str]:
        """Visible node ids, sorted."""
        return sorted(self._store.subgraph(scope=scope).nodes)

    def items(self, name: str, *, scope: RetrievalScope | None = None) -> list[str]:
        """Visible evidence item ids of a node."""
        return self._store.node_items(normalize_key(name), scope=scope)

    def edges(
        self,
        name: str,
        *,
        direction: Direction = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        """Visible live edges touching a node."""
        return self._store.edges_of(
            normalize_key(name), direction=direction, scope=scope, as_of=as_of
        )

    def backlinks(
        self,
        name: str,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        """Visible live edges pointing INTO the node."""
        return self.edges(name, direction="in", scope=scope, as_of=as_of)

    def neighbors(
        self,
        name: str,
        *,
        max_depth: int = 1,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Node ids within *max_depth* hops (both directions), BFS order, start excluded.

        Walks only visible edges, so an excluded node is never reached and
        never crossed.
        """
        start = normalize_key(name)
        if not self._exists(start, scope):
            return []
        visited = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        out: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._store.edges_of(current, scope=scope, as_of=as_of):
                nxt = edge.target if edge.source == current else edge.source
                if nxt not in visited:
                    visited.add(nxt)
                    out.append(nxt)
                    queue.append((nxt, depth + 1))
        return out

    def related_items(
        self,
        name: str,
        *,
        max_depth: int = 2,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Evidence items of the node and of its neighborhood, deduplicated, node order."""
        start = normalize_key(name)
        seen: dict[str, None] = {}
        for node_id in (
            start,
            *self.neighbors(start, max_depth=max_depth, scope=scope, as_of=as_of),
        ):
            for item_id in self._store.node_items(node_id, scope=scope):
                seen[item_id] = None
        return list(seen)

    def path(
        self,
        a: str,
        b: str,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[str] | None:
        """Fewest-hops path of node ids from *a* to *b*, or ``None``.

        Never crosses a node hidden by *scope* — a wall, not a bridge.
        """
        start, goal = normalize_key(a), normalize_key(b)
        if not (self._exists(start, scope) and self._exists(goal, scope)):
            return None
        if start == goal:
            return [start]
        parent: dict[str, str] = {start: start}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for edge in self._store.edges_of(current, scope=scope, as_of=as_of):
                nxt = edge.target if edge.source == current else edge.source
                if nxt in parent:
                    continue
                parent[nxt] = current
                if nxt == goal:
                    trail = [goal]
                    while trail[-1] != start:
                        trail.append(parent[trail[-1]])
                    return trail[::-1]
                queue.append(nxt)
        return None

    def explain(
        self,
        a: str,
        b: str,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        """The edges along :meth:`path` — relation, fact, provenance and evidence per hop."""
        trail = self.path(a, b, scope=scope, as_of=as_of)
        if not trail:
            return []
        out: list[GraphEdge] = []
        for u, v in pairwise(trail):
            for edge in self._store.edges_of(u, scope=scope, as_of=as_of):
                if {edge.source, edge.target} == {u, v}:
                    out.append(edge)
                    break
        return out

    def query(
        self,
        seeds: Iterable[str],
        *,
        top_k: int = 10,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
        damping: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Items ranked by personalized PageRank from the seed nodes.

        The walk runs over entities AND items (an item is adjacent to every
        node it evidences), so the score lands directly on items — the
        ``TokenBudget`` cuts the tail. Seeds that are unknown or hidden by
        *scope* are ignored; no visible seed → ``[]``.
        """
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        seed_keys = {("n", k) for k in (normalize_key(s) for s in seeds) if k in sub.nodes}
        if not seed_keys:
            return []
        pairs: list[tuple[tuple[str, str], tuple[str, str]]] = [
            (("n", e.source), ("n", e.target)) for e in sub.edges
        ]
        pairs.extend(
            (("n", node_id), ("i", item_id))
            for node_id, item_ids in sub.items.items()
            for item_id in item_ids
        )
        adj = adjacency(pairs)
        for node_id in sub.nodes:
            adj.setdefault(("n", node_id), [])
        scores = personalized_pagerank(adj, dict.fromkeys(seed_keys, 1.0), damping=damping)
        ranked = sorted(
            ((key[1], score) for key, score in scores.items() if key[0] == "i" and score > 0),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ranked[:top_k]

    def hubs(
        self,
        *,
        k: int = 10,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[tuple[str, int]]:
        """The *k* best-connected visible nodes as ``(node_id, degree)``."""
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        adj = adjacency((e.source, e.target) for e in sub.edges)
        deg = degree(adj)
        ranked = sorted(((n, deg.get(n, 0)) for n in sub.nodes), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def _exists(self, node_id: str, scope: RetrievalScope | None) -> bool:
        if self._store.get_node(node_id) is None:
            return False
        return scope is None or bool(self._store.node_items(node_id, scope=scope))

    def __len__(self) -> int:
        return len(self._store.subgraph().nodes)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(store={self._store!r})"
