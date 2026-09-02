"""Pure-Python graph algorithms (no dependency).

Sized for the graphs anchor builds from a vault (10^3-10^4 nodes): PPR by
power iteration runs well under a second there. The ``[graph]`` extra
(igraph) is the engine swap when that stops being true.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def adjacency(pairs: Iterable[tuple[K, K]]) -> dict[K, list[K]]:
    """Undirected adjacency lists from ``(a, b)`` pairs — insertion-ordered, deduplicated."""
    adj: dict[K, dict[K, None]] = {}
    for a, b in pairs:
        adj.setdefault(a, {})[b] = None
        adj.setdefault(b, {})[a] = None
    return {k: list(v) for k, v in adj.items()}


def bfs(adj: Mapping[K, Iterable[K]], start: K, *, max_depth: int) -> list[K]:
    """Nodes within *max_depth* hops of *start*, in BFS order, *start* excluded."""
    visited = {start}
    queue: deque[tuple[K, int]] = deque([(start, 0)])
    out: list[K] = []
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in adj.get(current, ()):
            if nxt not in visited:
                visited.add(nxt)
                out.append(nxt)
                queue.append((nxt, depth + 1))
    return out


def shortest_path(adj: Mapping[K, Iterable[K]], a: K, b: K) -> list[K] | None:
    """Fewest-hops path from *a* to *b* (inclusive), or ``None`` when unreachable."""
    if a == b:
        return [a] if a in adj else None
    parent: dict[K, K] = {a: a}
    queue: deque[K] = deque([a])
    while queue:
        current = queue.popleft()
        for nxt in adj.get(current, ()):
            if nxt in parent:
                continue
            parent[nxt] = current
            if nxt == b:
                path = [b]
                while path[-1] != a:
                    path.append(parent[path[-1]])
                return path[::-1]
            queue.append(nxt)
    return None


def degree(adj: Mapping[K, Iterable[K]]) -> dict[K, int]:
    """Degree per node."""
    return {k: len(list(v)) for k, v in adj.items()}


def personalized_pagerank(
    adj: Mapping[K, Iterable[K]],
    seeds: Mapping[K, float],
    *,
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[K, float]:
    """Personalized PageRank by power iteration: ``r ← (1-d)·p + d·M·r``.

    *seeds* is the restart vector (weights normalized here; unknown seeds
    ignored). ``damping=0.5`` (HippoRAG's choice) keeps the mass near the
    seeds; 0.85 explores further. Dangling nodes restart at the seeds.
    Returns ``{}`` when no seed is in the graph.
    """
    total = sum(w for k, w in seeds.items() if k in adj)
    if total <= 0:
        return {}
    p = {k: w / total for k, w in seeds.items() if k in adj}
    nbrs = {k: list(v) for k, v in adj.items()}
    r: dict[K, float] = {k: 0.0 for k in nbrs}
    r.update(p)
    for _ in range(max_iter):
        nxt = {k: (1.0 - damping) * p.get(k, 0.0) for k in nbrs}
        for u, out in nbrs.items():
            mass = damping * r[u]
            if not out:
                for k, pk in p.items():
                    nxt[k] += mass * pk
                continue
            share = mass / len(out)
            for v in out:
                nxt[v] += share
        delta = sum(abs(nxt[k] - r[k]) for k in nbrs)
        r = nxt
        if delta < tol:
            break
    return r
