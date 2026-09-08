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


def louvain(
    adj: Mapping[K, Iterable[K]], *, resolution: float = 1.0, max_levels: int = 10
) -> dict[K, int]:
    """Communities by the Louvain method (local moving + aggregation), deterministic.

    Nodes are visited in a fixed order and a node moves only for a strictly
    positive modularity gain, so the same graph always yields the same
    partition. Unweighted; multi-edges from the aggregation carry weights
    internally. Returns ``{node: community_id}``, ids renumbered 0..k-1 by
    first appearance in sorted node order. Isolated nodes get their own
    community. Sized for a vault (10^3-10^4 nodes); the ``[graph]`` extra
    (igraph, Leiden) is the engine swap beyond that.
    """
    nodes = sorted(adj, key=repr)
    # weighted graph over integer ids: w[u][v]
    index = {n: i for i, n in enumerate(nodes)}
    w: list[dict[int, float]] = [{} for _ in nodes]
    for n in nodes:
        for m in adj[n]:
            w[index[n]][index[m]] = w[index[n]].get(index[m], 0.0) + 1.0
    membership = list(range(len(nodes)))  # original node -> current community
    for _ in range(max_levels):
        moved, comm = _louvain_level(w, resolution)
        membership = [comm[c] for c in membership]
        if not moved:
            break
        w = _aggregate(w, comm)
    renumber: dict[int, int] = {}
    return {n: renumber.setdefault(membership[index[n]], len(renumber)) for n in nodes}


def _louvain_level(w: list[dict[int, float]], resolution: float) -> tuple[bool, list[int]]:
    """One local-moving phase. Returns (anything moved, node -> community)."""
    n = len(w)
    k = [sum(nbrs.values()) for nbrs in w]
    m2 = sum(k) or 1.0  # 2m
    comm = list(range(n))
    tot = k[:]  # total degree per community
    moved_any = False
    while True:
        moved = False
        for u in range(n):
            links: dict[int, float] = {}
            for v, wt in w[u].items():
                if v != u:
                    links[comm[v]] = links.get(comm[v], 0.0) + wt
            current = comm[u]
            tot[current] -= k[u]
            best, best_gain = (
                current,
                links.get(current, 0.0) - resolution * tot[current] * k[u] / m2,
            )
            for c in sorted(links):
                gain = links[c] - resolution * tot[c] * k[u] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            tot[best] += k[u]
            if best != current:
                comm[u] = best
                moved = moved_any = True
        if not moved:
            break
    renumber: dict[int, int] = {}
    return moved_any, [renumber.setdefault(c, len(renumber)) for c in comm]


def _aggregate(w: list[dict[int, float]], comm: list[int]) -> list[dict[int, float]]:
    """Collapse communities into nodes, summing edge weights (self-loops kept)."""
    size = max(comm) + 1 if comm else 0
    out: list[dict[int, float]] = [{} for _ in range(size)]
    for u, nbrs in enumerate(w):
        for v, wt in nbrs.items():
            cu, cv = comm[u], comm[v]
            out[cu][cv] = out[cu].get(cv, 0.0) + wt
    return out


def modularity(adj: Mapping[K, Iterable[K]], communities: Mapping[K, int]) -> float:
    """Newman modularity of a partition over an undirected graph.

    ``Q = sum_c [ L_c / m - (D_c / 2m)^2 ]`` with ``L_c`` intra-community
    edges and ``D_c`` the degree sum of the community. For tests and tuning.
    """
    degree_of = {n: len(list(v)) for n, v in adj.items()}
    m2 = float(sum(degree_of.values()))  # 2m
    if m2 == 0:
        return 0.0
    intra: dict[int, float] = {}
    total: dict[int, float] = {}
    for n, nbrs in adj.items():
        c = communities[n]
        total[c] = total.get(c, 0.0) + degree_of[n]
        for v in nbrs:
            if communities.get(v) == c:
                intra[c] = intra.get(c, 0.0) + 1.0  # counted twice (both ends)
    return sum(intra.get(c, 0.0) / m2 - (total[c] / m2) ** 2 for c in total)
