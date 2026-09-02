"""Pure-Python graph algorithms."""

from __future__ import annotations

import pytest

from anchor.graph.algorithms import adjacency, bfs, degree, personalized_pagerank, shortest_path


def _chain() -> dict[str, list[str]]:
    return adjacency([("a", "b"), ("b", "c"), ("c", "d"), ("b", "a")])


class TestAdjacency:
    def test_undirected_deduplicated_insertion_ordered(self) -> None:
        adj = _chain()
        assert adj["a"] == ["b"]
        assert adj["b"] == ["a", "c"]
        assert adj["d"] == ["c"]

    def test_empty(self) -> None:
        assert adjacency([]) == {}


class TestBfs:
    def test_depth_limits_and_excludes_start(self) -> None:
        adj = _chain()
        assert bfs(adj, "a", max_depth=1) == ["b"]
        assert bfs(adj, "a", max_depth=2) == ["b", "c"]
        assert bfs(adj, "a", max_depth=10) == ["b", "c", "d"]

    def test_unknown_start(self) -> None:
        assert bfs(_chain(), "zzz", max_depth=3) == []


class TestShortestPath:
    def test_path(self) -> None:
        assert shortest_path(_chain(), "a", "d") == ["a", "b", "c", "d"]

    def test_prefers_fewest_hops(self) -> None:
        adj = adjacency([("a", "b"), ("b", "c"), ("a", "c")])
        assert shortest_path(adj, "a", "c") == ["a", "c"]

    def test_unreachable(self) -> None:
        adj = adjacency([("a", "b"), ("x", "y")])
        assert shortest_path(adj, "a", "y") is None

    def test_same_node(self) -> None:
        adj = _chain()
        assert shortest_path(adj, "a", "a") == ["a"]
        assert shortest_path(adj, "zzz", "zzz") is None


class TestDegree:
    def test_degree(self) -> None:
        assert degree(_chain()) == {"a": 1, "b": 2, "c": 2, "d": 1}


class TestPersonalizedPagerank:
    def test_mass_sums_to_one_and_decays_from_seed(self) -> None:
        scores = personalized_pagerank(_chain(), {"a": 1.0})
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)
        assert scores["a"] > scores["b"] > scores["c"] > scores["d"] > 0

    def test_unknown_seed_is_empty(self) -> None:
        assert personalized_pagerank(_chain(), {"zzz": 1.0}) == {}

    def test_disconnected_component_gets_no_mass(self) -> None:
        adj = adjacency([("a", "b"), ("x", "y")])
        scores = personalized_pagerank(adj, {"a": 1.0})
        assert scores["x"] == 0.0
        assert scores["y"] == 0.0

    def test_dangling_node_restarts_at_seeds(self) -> None:
        adj = {"a": ["b"], "b": []}  # directed-looking: b is dangling
        scores = personalized_pagerank(adj, {"a": 1.0})
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)
        assert scores["a"] > scores["b"] > 0

    def test_higher_damping_explores_further(self) -> None:
        near = personalized_pagerank(_chain(), {"a": 1.0}, damping=0.3)
        far = personalized_pagerank(_chain(), {"a": 1.0}, damping=0.9)
        assert far["d"] > near["d"]


class TestCommunities:
    def _two_cliques(self) -> dict[str, list[str]]:
        pairs = [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z"), ("c", "x")]
        return adjacency(pairs)

    def test_louvain_finds_the_two_cliques(self) -> None:
        from anchor.graph.algorithms import louvain, modularity

        adj = self._two_cliques()
        adj["lonely"] = []
        part = louvain(adj)
        assert part["a"] == part["b"] == part["c"]
        assert part["x"] == part["y"] == part["z"]
        assert part["a"] != part["x"]
        assert part["lonely"] not in (part["a"], part["x"])
        assert sorted(set(part.values())) == [0, 1, 2]  # renumbered by first appearance
        assert modularity(adj, part) > 0.3
        assert louvain(adj) == part  # deterministic

    def test_modularity_matches_networkx(self) -> None:
        nx = pytest.importorskip("networkx")
        from anchor.graph.algorithms import louvain, modularity

        adj = self._two_cliques()
        part = louvain(adj)
        g = nx.Graph()
        for a, nbrs in adj.items():
            for b in nbrs:
                g.add_edge(a, b)
        groups: dict[int, set[str]] = {}
        for n, c in part.items():
            groups.setdefault(c, set()).add(n)
        assert modularity(adj, part) == pytest.approx(
            nx.community.modularity(g, list(groups.values())), abs=1e-9
        )

    def test_empty_graph(self) -> None:
        from anchor.graph.algorithms import louvain, modularity

        assert louvain({}) == {}
        assert modularity({}, {}) == 0.0
