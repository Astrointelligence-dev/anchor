"""KnowledgeGraph: the navigation API over a ``GraphStore`` (roadmap #4).

Every read loads the visible subgraph once (``store.subgraph(scope, as_of)``)
and walks it with the pure-Python algorithms — one scoped load instead of
one query per visited node, the shape ``query``/``hubs``/``communities``
share. Zero LLM calls on the read path.

Every read takes ``scope`` (namespace navigation, a ``RetrievalScope``)
and ``as_of`` (world time). The graph never reads the agent's published
scope itself — the caller (``GraphRetriever``, tools) resolves
``effective_scope`` and passes it, the same contract the stores have.
The visibility rule is stated in :mod:`anchor.models.graph`.
"""

from __future__ import annotations

import inspect
import itertools
import re
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

from anchor.graph.algorithms import (
    adjacency,
    bfs,
    degree,
    louvain,
    personalized_pagerank,
    shortest_path,
)
from anchor.models.graph import GraphEdge, GraphNode, Provenance, Subgraph, normalize_key
from anchor.models.scope import ROOT_NAMESPACE, RetrievalScope
from anchor.protocols.storage import GraphStore
from anchor.storage.memory_store import InMemoryGraphStore

Direction = Literal["out", "in", "both"]
# (cache key, partition, expiry when the key carries as_of=None)
_CommunityCache = tuple[tuple[Any, ...], dict[str, int], datetime | None]


def _leiden(adj: dict[str, list[str]]) -> dict[str, int] | None:
    """Leiden via igraph when installed (``pip install astro-anchor[graph]``), else ``None``."""
    try:
        import igraph
    except ImportError:
        return None
    nodes = sorted(adj)
    index = {n: i for i, n in enumerate(nodes)}
    edges = sorted({(index[a], index[b]) for a in nodes for b in adj[a] if index[a] < index[b]})
    graph = igraph.Graph(n=len(nodes), edges=edges)
    membership = graph.community_leiden(objective_function="modularity", n_iterations=-1).membership
    renumber: dict[int, int] = {}
    return {n: renumber.setdefault(int(membership[index[n]]), len(renumber)) for n in nodes}


def _key(name: str) -> str | None:
    """``normalize_key`` for READ paths: an unusable name is an unknown node, not an error."""
    try:
        return normalize_key(name)
    except ValueError:
        return None


def _match_key(text: str) -> str:
    """Tokenizer for mention matching: any non-word run collapses to ``_``.

    Looser than ``normalize_key`` on purpose — ``"Checkout?"`` in a question
    must still meet the node ``checkout``.
    """
    return "_".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _alias_index(sub: Subgraph) -> dict[str, str]:
    """match-key of every id and alias → node id (first wins, insertion order)."""
    index: dict[str, str] = {}
    for node_id, node in sub.nodes.items():
        for spelling in (node_id, *node.aliases):
            key = _match_key(spelling)
            if key:  # a punctuation-only alias must never match the empty n-gram
                index.setdefault(key, node_id)
    return index


def _resolve(name: str, sub: Subgraph) -> str | None:
    """A visible node id for *name*: its key, else an alias of a visible node."""
    key = _key(name)
    if key is not None and key in sub.nodes:
        return key
    return _alias_index(sub).get(_match_key(name)) if name.strip() else None


def _adjacency(sub: Subgraph) -> dict[str, list[str]]:
    adj = adjacency((e.source, e.target) for e in sub.edges)
    for node_id in sub.nodes:
        adj.setdefault(node_id, [])
    return adj


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

    __slots__ = ("_communities_cache", "_store")

    def __init__(self, store: GraphStore | None = None) -> None:
        if store is not None and inspect.iscoroutinefunction(getattr(store, "get_node", None)):
            msg = (
                f"{type(store).__name__} is an async store; KnowledgeGraph is synchronous "
                "(use SqliteGraphStore/InMemoryGraphStore, or drive the async store directly)"
            )
            raise TypeError(msg)
        self._store: GraphStore = store if store is not None else InMemoryGraphStore()
        # (scope, as_of, store.version) → partition; derived data is never
        # persisted — recomputed lazily when the graph changes (roadmap #4).
        self._communities_cache: _CommunityCache | None = None

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
        """Upsert a node; metadata and aliases merge, the first real label wins."""
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
        The evidence then evidences both endpoints too.
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
        """The node under *name* — its key, or an alias of it."""
        key = _key(name)
        found = None if key is None else self._store.get_node(key)
        if found is not None:
            return found
        resolved = _resolve(name, self._store.subgraph())
        return None if resolved is None else self._store.get_node(resolved)

    def nodes(self, *, scope: RetrievalScope | None = None) -> list[str]:
        """Visible node ids, sorted."""
        return sorted(self._store.subgraph(scope=scope).nodes)

    def items(self, name: str, *, scope: RetrievalScope | None = None) -> list[str]:
        """Visible evidence item ids of a node (name or alias)."""
        node_id = _resolve(name, self._store.subgraph(scope=scope))
        return [] if node_id is None else self._store.node_items(node_id, scope=scope)

    def edges(
        self,
        name: str,
        *,
        direction: Direction = "both",
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        """Visible live edges touching a node: outgoing first, then incoming."""
        node_id = _resolve(name, self._store.subgraph(scope=scope, as_of=as_of))
        if node_id is None:
            return []
        return self._store.edges_of(node_id, direction=direction, scope=scope, as_of=as_of)

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

        Walks only the visible subgraph, so an excluded node is never reached
        and never crossed.
        """
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        start = _resolve(name, sub)
        if start is None:
            return []
        return bfs(_adjacency(sub), start, max_depth=max_depth)

    def related_items(
        self,
        name: str,
        *,
        max_depth: int = 2,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Evidence items of the node and of its neighborhood, deduplicated, node order."""
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        start = _resolve(name, sub)
        if start is None:
            return []
        seen: dict[str, None] = {}
        for node_id in (start, *bfs(_adjacency(sub), start, max_depth=max_depth)):
            for item_id in sub.items.get(node_id, ()):
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
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        start, goal = _resolve(a, sub), _resolve(b, sub)
        if start is None or goal is None:
            return None
        return shortest_path(_adjacency(sub), start, goal)

    def explain(
        self,
        a: str,
        b: str,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> list[GraphEdge]:
        """The edges along :meth:`path` — relation, fact, provenance and evidence per hop."""
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        start, goal = _resolve(a, sub), _resolve(b, sub)
        if start is None or goal is None:
            return []
        trail = shortest_path(_adjacency(sub), start, goal)
        if not trail or len(trail) < 2:
            return []
        by_pair: dict[frozenset[str], GraphEdge] = {}
        for edge in sub.edges:
            by_pair.setdefault(frozenset((edge.source, edge.target)), edge)
        return [by_pair[frozenset(pair)] for pair in itertools.pairwise(trail)]

    def mentions(
        self,
        text: str,
        *,
        scope: RetrievalScope | None = None,
        max_words: int = 4,
        subgraph: Subgraph | None = None,
    ) -> list[str]:
        """Visible nodes whose id or alias appears in *text* (word n-grams, longest first).

        The zero-cost seeder: no model, just the vocabulary the graph already
        has. ``"Who is on call for Checkout?"`` → ``["checkout_service"]`` when
        ``Checkout`` is an alias of that note. Pass *subgraph* to reuse a load.
        """
        sub = subgraph if subgraph is not None else self._store.subgraph(scope=scope)
        index = _alias_index(sub)
        words = [w for w in _match_key(text).split("_") if w]
        found: dict[str, None] = {}
        for n in range(max_words, 0, -1):
            for i in range(len(words) - n + 1):
                hit = index.get("_".join(words[i : i + n]))
                if hit is not None:
                    found[hit] = None
        return list(found)

    def query(
        self,
        seeds: Iterable[str],
        *,
        item_seeds: Iterable[str] = (),
        top_k: int = 10,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
        damping: float = 0.5,
        subgraph: Subgraph | None = None,
    ) -> list[tuple[str, float]]:
        """Items ranked by personalized PageRank from the seed nodes (and items).

        The walk runs over entities AND items (an item is adjacent to every
        node it evidences), so the score lands directly on items — the
        ``TokenBudget`` cuts the tail. *item_seeds* lets a dense search seed
        the walk with passages (HippoRAG-2). Seeds that are unknown or
        hidden by *scope* are ignored; no visible seed → ``[]``. Pass
        *subgraph* to reuse a load.
        """
        sub = subgraph if subgraph is not None else self._store.subgraph(scope=scope, as_of=as_of)
        visible_items = {item_id for ids in sub.items.values() for item_id in ids}
        seed_keys = {("n", k) for k in map(_key, seeds) if k is not None and k in sub.nodes}
        seed_keys |= {("i", i) for i in item_seeds if i in visible_items}
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
        deg = degree(adjacency((e.source, e.target) for e in sub.edges))
        ranked = sorted(((n, deg.get(n, 0)) for n in sub.nodes), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def communities(
        self,
        *,
        scope: RetrievalScope | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, int]:
        """``{node_id: community_id}`` over the visible subgraph, cached by graph version.

        Engine: igraph's Leiden when the ``[graph]`` extra is installed,
        else the built-in deterministic Louvain. Isolated nodes get their own
        community.
        """
        now = datetime.now(UTC) if as_of is None else as_of
        key = (scope, as_of, self._store.version)
        cached = self._communities_cache
        if cached is not None and cached[0] == key and (cached[2] is None or now < cached[2]):
            return dict(cached[1])
        sub = self._store.subgraph(scope=scope, as_of=as_of)
        adj = _adjacency(sub)
        partition = _leiden(adj) or louvain(adj)
        # With as_of=None the partition depends on the wall clock: it expires
        # when the first live edge reaches its valid_to. (An edge whose
        # valid_from lies ahead is not in the live subgraph — ponytail: the
        # cache misses that birth until the next write.)
        expires = None
        if as_of is None:
            ends = [e.valid_to for e in sub.edges if e.valid_to is not None and e.valid_to > now]
            expires = min(ends) if ends else None
        self._communities_cache = (key, partition, expires)
        return dict(partition)

    def __len__(self) -> int:
        return len(self._store.subgraph().nodes)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(store={self._store!r})"
