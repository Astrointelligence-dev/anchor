"""KnowledgeGraph + InMemoryGraphStore: model, navigation, temporal validity, scope."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from anchor.graph import KnowledgeGraph
from anchor.models.graph import GraphEdge, GraphNode, normalize_key
from anchor.models.scope import RetrievalScope
from anchor.protocols.storage import GraphStore
from anchor.storage.memory_store import InMemoryGraphStore

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def _basic() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node("Alice", metadata={"type": "person"})
    g.add_node("Project X", metadata={"type": "project"})
    g.add_edge("Alice", "works on", "Project X")
    return g


def _campaign(g: KnowledgeGraph | None = None) -> KnowledgeGraph:
    """hero -knows-> villain -rules-> city; hero -lives_in-> city;
    hero -allied_with-> ally -lives_in-> city.

    The villain is evidenced only by a /secret item — the decoy every
    scope test hides. Pass a graph to build the same world on any store.
    """
    g = g if g is not None else KnowledgeGraph()
    for name in ("hero", "villain", "city", "ally"):
        g.add_node(name)
    g.link_item("hero", "doc-1", "/public")
    g.link_item("city", "doc-1", "/public")
    g.link_item("villain", "doc-2", "/secret")
    g.link_item("ally", "doc-3", "/public/allies")
    g.link_item("city", "doc-3", "/public/allies")
    g.add_edge("hero", "knows", "villain", evidence=["doc-2"], fact="the hero knows the villain")
    g.add_edge("villain", "rules", "city", evidence=["doc-2"])
    g.add_edge("hero", "lives_in", "city", evidence=["doc-1"], fact="the hero lives in the city")
    g.add_edge("hero", "allied_with", "ally", evidence=["doc-3"])
    g.add_edge("ally", "lives_in", "city", evidence=["doc-3"])
    return g


NO_SECRET = RetrievalScope(exclude=("/secret",))


# ===========================================================================
# Models
# ===========================================================================


class TestModels:
    def test_normalize_key_converges_spellings(self) -> None:
        assert normalize_key("Project X") == normalize_key(" project   x ") == "project_x"
        assert normalize_key("Ｃａｆé") == "café"  # NFKC folds full-width

    def test_normalize_key_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            normalize_key("   ")

    def test_node_label_defaults_to_raw_name(self) -> None:
        node = GraphNode(id="Project X")
        assert node.id == "project_x"
        assert node.label == "Project X"

    def test_edge_normalizes_endpoints_and_relation(self) -> None:
        edge = GraphEdge(source="Alice", target="Project X", relation="Works On")
        assert (edge.source, edge.relation, edge.target) == ("alice", "works_on", "project_x")
        assert edge.provenance == "extracted"
        assert edge.confidence == 1.0
        assert edge.invalidated_at is None

    def test_edge_evidence_deduplicated_and_weight(self) -> None:
        edge = GraphEdge(source="a", target="b", relation="r", evidence=("i1", "i1", "i2"))
        assert edge.evidence == ("i1", "i2")

    def test_edge_confidence_bounded(self) -> None:
        with pytest.raises(ValueError):
            GraphEdge(source="a", target="b", relation="r", confidence=1.5)

    def test_is_live_temporal_window(self) -> None:
        edge = GraphEdge(
            source="a",
            target="b",
            relation="r",
            valid_from=NOW,
            valid_to=NOW + timedelta(days=1),
        )
        assert not edge.is_live(NOW - timedelta(seconds=1))
        assert edge.is_live(NOW)
        assert not edge.is_live(NOW + timedelta(days=1))
        assert not edge.model_copy(update={"invalidated_at": NOW}).is_live(NOW)


# ===========================================================================
# Store contract
# ===========================================================================


class TestStore:
    def test_store_satisfies_protocol_and_validates_vault(self) -> None:
        store = InMemoryGraphStore(vault="dnd")
        assert isinstance(store, GraphStore)
        assert store.vault == "dnd"
        assert KnowledgeGraph(store).vault == "dnd"
        with pytest.raises(ValueError):
            InMemoryGraphStore(vault="a/b")

    def test_version_increments_on_writes_only(self) -> None:
        g = _basic()
        v = g.store.version
        g.neighbors("alice")
        assert g.store.version == v
        g.link_item("alice", "mem-1")
        assert g.store.version == v + 1

    def test_add_edge_requires_linked_evidence(self) -> None:
        g = _basic()
        with pytest.raises(ValueError, match="link_item"):
            g.add_edge("alice", "knows", "bob", evidence=["unknown-item"])

    def test_link_item_unknown_node_raises(self) -> None:
        g = _basic()
        with pytest.raises(KeyError):
            g.link_item("nobody", "mem-1")

    def test_latest_link_decides_the_items_namespace(self) -> None:
        # A re-indexed (moved) document moves its evidence, as the ContextStore does.
        g = _basic()
        g.link_item("alice", "mem-1", "/a")
        g.link_item("project x", "mem-1", "/b")
        assert g.items("alice", scope=RetrievalScope(include=("/a",))) == []
        assert g.items("alice", scope=RetrievalScope(include=("/b",))) == ["mem-1"]

    def test_duplicate_edge_id_raises(self) -> None:
        g = _basic()
        edge = g.add_edge("alice", "knows", "bob")
        clone = edge.model_copy(update={"relation": "hates"})
        with pytest.raises(ValueError, match="already exists"):
            g.store.add_edge(clone)

    def test_clear(self) -> None:
        g = _basic()
        g.link_item("alice", "mem-1")
        g.clear()
        assert len(g) == 0
        assert g.items("alice") == []
        assert "nodes=0" in repr(g.store)


# ===========================================================================
# Nodes and edges
# ===========================================================================


class TestNodesAndEdges:
    def test_add_node_merges_metadata_and_aliases_keeps_first_label(self) -> None:
        g = KnowledgeGraph()
        g.add_node("Alice", metadata={"type": "person"}, aliases=["ali"])
        node = g.add_node("ALICE", metadata={"role": "engineer"}, aliases=["ali", "al"])
        assert node.label == "Alice"
        assert node.metadata == {"type": "person", "role": "engineer"}
        assert node.aliases == ("ali", "al")
        assert g.node("alice") == node
        assert g.node("nobody") is None

    def test_names_are_normalized_everywhere(self) -> None:
        g = _basic()
        assert g.neighbors("ALICE") == ["project_x"]
        assert g.neighbors("Project X") == ["alice"]
        g.link_item("PROJECT  X", "doc-1")
        assert g.items("project_x") == ["doc-1"]

    def test_add_edge_creates_missing_nodes_with_readable_labels(self) -> None:
        g = KnowledgeGraph()
        edge = g.add_edge("Alice", "knows", "Bob Smith")
        assert edge.relation == "knows"
        assert g.node("bob smith") is not None
        assert g.node("bob smith").label == "Bob Smith"  # type: ignore[union-attr]
        assert sorted(g.nodes()) == ["alice", "bob_smith"]

    def test_readding_live_edge_reinforces_instead_of_duplicating(self) -> None:
        g = KnowledgeGraph()
        g.add_node("a")
        g.link_item("a", "i1", "/x")
        g.link_item("a", "i2", "/y")
        first = g.add_edge("a", "r", "b", evidence=["i1"], provenance="inferred", confidence=0.6)
        second = g.add_edge(
            "a", "r", "b", evidence=["i2"], provenance="extracted", confidence=0.9, fact="a r b"
        )
        assert second.id == first.id
        assert second.evidence == ("i1", "i2")
        assert second.provenance == "extracted"
        assert second.confidence == 0.9
        assert second.fact == "a r b"
        assert len(g.edges("a")) == 1
        assert g.store.get_edge(first.id) == second

    def test_edges_direction_and_backlinks(self) -> None:
        g = _campaign()
        assert [e.relation for e in g.edges("city", direction="in")] == [
            "rules",
            "lives_in",
            "lives_in",
        ]
        assert g.edges("city", direction="out") == []
        assert {e.source for e in g.backlinks("city")} == {"villain", "hero", "ally"}

    def test_remove_node_invalidates_edges_and_drops_links(self) -> None:
        g = _campaign()
        edge_ids = [e.id for e in g.edges("villain")]
        assert g.remove_node("villain") is True
        assert g.remove_node("villain") is False
        assert g.node("villain") is None
        for edge_id in edge_ids:
            stored = g.store.get_edge(edge_id)
            assert stored is not None
            assert stored.invalidated_at is not None  # history kept
        assert "villain" not in g.neighbors("hero", max_depth=3)
        assert g.items("villain") == []

    def test_len_and_repr(self) -> None:
        g = _basic()
        assert len(g) == 2
        assert "KnowledgeGraph" in repr(g)


# ===========================================================================
# Items and incremental maintenance
# ===========================================================================


class TestItems:
    def test_items_and_related_items_order_and_dedupe(self) -> None:
        g = _campaign()
        # city is evidenced by its own links AND by the edges that mention it
        assert g.items("city") == ["doc-1", "doc-3", "doc-2"]
        assert g.related_items("hero", max_depth=1) == ["doc-1", "doc-2", "doc-3"]
        assert g.related_items("villain", max_depth=0) == ["doc-2"]

    def test_related_items_unknown_node(self) -> None:
        assert _campaign().related_items("nobody") == []

    def test_unlink_item_strips_evidence_and_invalidates_orphans(self) -> None:
        g = KnowledgeGraph()
        g.add_node("a")
        g.link_item("a", "i1")
        g.link_item("a", "i2")
        shared = g.add_edge("a", "r", "b", evidence=["i1", "i2"])
        lonely = g.add_edge("a", "s", "c", evidence=["i1"])
        assert g.unlink_item("i1") == 1
        assert g.store.get_edge(shared.id).evidence == ("i2",)  # type: ignore[union-attr]
        assert g.store.get_edge(lonely.id).invalidated_at is not None  # type: ignore[union-attr]
        assert [e.relation for e in g.edges("a")] == ["r"]
        assert g.items("a") == ["i2"]
        assert g.unlink_item("i1") == 0

    def test_reindex_same_item_is_idempotent(self) -> None:
        g = KnowledgeGraph()
        for _ in range(3):
            g.add_node("a")
            g.link_item("a", "doc", "/n")
            g.add_edge("a", "r", "b", evidence=["doc"])
        assert len(g.edges("a")) == 1
        assert g.items("a") == ["doc"]
        assert g.store.get_edge(g.edges("a")[0].id).evidence == ("doc",)  # type: ignore[union-attr]


# ===========================================================================
# Navigation
# ===========================================================================


class TestNavigation:
    def test_neighbors_depth_and_direction(self) -> None:
        g = _campaign()
        assert g.neighbors("villain", max_depth=1) == ["hero", "city"]  # edge insertion order
        assert set(g.neighbors("ally", max_depth=1)) == {"hero", "city"}
        assert set(g.neighbors("ally", max_depth=2)) == {"hero", "city", "villain"}
        assert g.neighbors("nobody") == []

    def test_path_and_explain(self) -> None:
        g = _campaign()
        assert g.path("villain", "ally") == ["villain", "hero", "ally"]
        hops = g.explain("villain", "ally")
        assert [(e.source, e.relation, e.target) for e in hops] == [
            ("hero", "knows", "villain"),
            ("hero", "allied_with", "ally"),
        ]
        assert g.path("hero", "hero") == ["hero"]
        assert g.explain("hero", "hero") == []
        assert g.path("hero", "nobody") is None
        assert g.explain("hero", "nobody") == []

    def test_path_unreachable(self) -> None:
        g = _campaign()
        g.add_node("island")
        assert g.path("hero", "island") is None

    def test_hubs_by_degree(self) -> None:
        g = _campaign()
        assert g.hubs(k=2) == [("city", 3), ("hero", 3)]
        assert g.hubs(k=10)[2:] == [("ally", 2), ("villain", 2)]

    def test_query_ranks_items_near_the_seed_first(self) -> None:
        g = _campaign()
        ranked = g.query(["villain"], top_k=10)
        ids = [item for item, _ in ranked]
        assert ids[0] == "doc-2"
        assert set(ids) == {"doc-1", "doc-2", "doc-3"}
        assert all(a >= b for (_, a), (_, b) in pairwise(ranked))

    def test_query_top_k_and_unknown_seed(self) -> None:
        g = _campaign()
        assert len(g.query(["hero"], top_k=1)) == 1
        assert g.query(["nobody"]) == []
        assert g.query([]) == []


# ===========================================================================
# Temporal validity
# ===========================================================================


class TestTemporal:
    def test_invalidated_edge_hidden_from_reads_but_kept(self) -> None:
        g = _campaign()
        edge = next(e for e in g.edges("hero") if e.relation == "knows")
        assert g.invalidate_edge(edge.id, at=NOW) is True
        assert g.invalidate_edge(edge.id) is False
        assert "villain" not in g.neighbors("hero")
        assert g.path("hero", "villain") == ["hero", "city", "villain"]
        assert g.store.get_edge(edge.id).invalidated_at == NOW  # type: ignore[union-attr]

    def test_as_of_respects_world_time(self) -> None:
        g = KnowledgeGraph()
        g.add_edge("alice", "lives_in", "sp", valid_to=NOW)
        g.add_edge("alice", "lives_in", "rio", valid_from=NOW)
        assert g.neighbors("alice", as_of=NOW - timedelta(days=1)) == ["sp"]
        assert g.neighbors("alice", as_of=NOW) == ["rio"]
        assert g.neighbors("alice", as_of=NOW + timedelta(days=30)) == ["rio"]
        assert g.hubs(k=1, as_of=NOW - timedelta(days=1)) == [("alice", 1)]

    def test_subgraph_filters_by_time(self) -> None:
        g = KnowledgeGraph()
        g.add_edge("a", "r", "b", valid_from=NOW)
        assert g.store.subgraph(as_of=NOW - timedelta(days=1)).edges == []
        assert len(g.store.subgraph(as_of=NOW).edges) == 1


# ===========================================================================
# Scope — the decoy tests: the excluded node must not exist by any path
# ===========================================================================


class TestScope:
    def test_excluded_node_does_not_exist(self) -> None:
        g = _campaign()
        assert "villain" in g.nodes()
        assert "villain" not in g.nodes(scope=NO_SECRET)
        assert g.neighbors("villain", scope=NO_SECRET) == []
        assert g.items("villain", scope=NO_SECRET) == []
        assert g.edges("villain", scope=NO_SECRET) == []

    def test_excluded_node_unreachable_and_never_crossed(self) -> None:
        g = _campaign()
        assert "villain" not in g.neighbors("hero", max_depth=5, scope=NO_SECRET)
        assert g.path("hero", "villain", scope=NO_SECRET) is None
        assert g.path("villain", "city", scope=NO_SECRET) is None
        assert g.explain("hero", "villain", scope=NO_SECRET) == []
        assert {e.source for e in g.backlinks("city", scope=NO_SECRET)} == {"hero", "ally"}
        assert all(name != "villain" for name, _ in g.hubs(scope=NO_SECRET))
        # city loses the villain's edge: every survivor ties at degree 2
        assert g.hubs(k=3, scope=NO_SECRET) == [("ally", 2), ("city", 2), ("hero", 2)]

    def test_wall_not_bridge(self) -> None:
        # hero —knows→ villain —rules→ city is the ONLY route once the direct
        # and allied edges are gone; under the scope the route must vanish
        # instead of the topology betraying the hidden node.
        g = _campaign()
        for edge in g.edges("hero"):
            if edge.relation in ("lives_in", "allied_with"):
                g.invalidate_edge(edge.id)
        assert g.path("hero", "city") == ["hero", "villain", "city"]
        assert g.path("hero", "city", scope=NO_SECRET) is None
        assert g.related_items("hero", max_depth=3, scope=NO_SECRET) == ["doc-1", "doc-3"]

    def test_query_never_returns_hidden_items(self) -> None:
        g = _campaign()
        ranked = g.query(["hero"], top_k=10, scope=NO_SECRET)
        assert ranked
        assert "doc-2" not in {item for item, _ in ranked}
        assert g.query(["villain"], scope=NO_SECRET) == []

    def test_include_scope_and_edge_with_hidden_evidence(self) -> None:
        g = _campaign()
        only_allies = RetrievalScope(include=("/public/allies",))
        # city and hero are evidenced by doc-3 (/public/allies) through the ally
        # edges: they survive, but the hero→city edge (evidence doc-1 only) does not.
        assert g.nodes(scope=only_allies) == ["ally", "city", "hero"]
        assert {e.relation for e in g.edges("city", scope=only_allies)} == {"lives_in"}
        assert g.items("city", scope=only_allies) == ["doc-3"]
        assert g.path("hero", "city", scope=only_allies) == ["hero", "ally", "city"]
        assert g.hubs(k=1, scope=only_allies) == [("ally", 2)]

    def test_node_without_evidence_lives_at_the_root(self) -> None:
        g = _campaign()
        g.add_edge("hero", "seeks", "artifact")  # manual edge, no evidence
        assert "artifact" in g.neighbors("hero")
        assert "artifact" in g.neighbors("hero", scope=RetrievalScope())  # identity
        assert "artifact" in g.nodes(scope=NO_SECRET)
        assert "artifact" in g.nodes(scope=RetrievalScope(include=("/",)))
        assert "artifact" not in g.nodes(scope=RetrievalScope(include=("/public",)))
        assert "artifact" not in g.nodes(scope=RetrievalScope(exclude=("/",)))

    def test_empty_scope_is_the_identity(self) -> None:
        g = _campaign()
        g.add_edge("hero", "seeks", "artifact")
        empty = RetrievalScope()
        assert g.nodes(scope=empty) == g.nodes()
        for name in g.nodes():
            assert g.neighbors(name, max_depth=2, scope=empty) == g.neighbors(name, max_depth=2)
            assert g.items(name, scope=empty) == g.items(name)
        assert g.path("villain", "artifact", scope=empty) == g.path("villain", "artifact")
        assert g.query(["hero"], scope=empty) == g.query(["hero"])
        assert g.hubs(scope=empty) == g.hubs()

    def test_subagent_scope_only_narrows(self) -> None:
        g = _campaign()
        parent = RetrievalScope(include=("/public",))
        child = RetrievalScope(include=("/public/allies",), exclude=("/secret",))
        widen = RetrievalScope(include=("/",))
        assert g.nodes(scope=parent.intersect(child)) == ["ally", "city", "hero"]
        # a child asking for "/" cannot see more than the parent
        assert g.nodes(scope=parent.intersect(widen)) == g.nodes(scope=parent)
        assert "villain" not in g.nodes(scope=parent.intersect(widen))


class TestCommunities:
    def test_communities_follow_scope_and_cache_by_version(self) -> None:
        g = _campaign()
        g.add_edge("far_a", "r", "far_b")
        part = g.communities()
        assert part["hero"] == part["city"]
        assert part["far_a"] == part["far_b"] != part["hero"]
        assert g.communities() is not part  # a copy each call...
        assert g.communities() == part  # ...but served from the cache
        assert "villain" not in g.communities(scope=NO_SECRET)
        # the graph changed → recomputed from the current subgraph, not the cache
        from anchor.graph.algorithms import adjacency, louvain

        for a, b in (("far_a", "hero"), ("far_a", "city"), ("far_b", "hero"), ("far_b", "city")):
            g.add_edge(a, "r", b)
        sub = g.store.subgraph()
        fresh = louvain(adjacency((e.source, e.target) for e in sub.edges))
        assert g.communities() == fresh
        assert len(g.edges("far_a")) == 3

    def test_igraph_engine_when_installed(self) -> None:
        pytest.importorskip("igraph")
        from anchor.graph.knowledge_graph import _leiden

        g = _campaign()
        sub = g.store.subgraph()
        from anchor.graph.algorithms import adjacency

        adj = adjacency((e.source, e.target) for e in sub.edges)
        part = _leiden(adj)
        assert part is not None
        assert set(part) == set(adj)


# ===========================================================================
# Ritual xhigh (session 11) — regressions pinned by the review
# ===========================================================================


class TestReviewRegressions:
    def test_unusable_names_are_unknown_nodes_not_errors(self) -> None:
        g = _campaign()
        assert g.node("   ") is None
        assert g.items("") == []
        assert g.edges(" ") == []
        assert g.neighbors("") == []
        assert g.related_items("  ") == []
        assert g.path("", "hero") is None
        assert g.explain("hero", "") == []
        # one bad seed never costs the good ones
        assert [i for i, _ in g.query(["", "villain"], top_k=1)] == ["doc-2"]

    def test_async_store_is_refused_at_construction(self) -> None:
        pytest.importorskip("asyncpg")
        from anchor.storage.postgres import PostgresGraphStore

        with pytest.raises(TypeError, match="async store"):
            KnowledgeGraph(PostgresGraphStore(object(), vault="v"))  # type: ignore[arg-type]

    def test_punctuation_only_alias_never_matches(self) -> None:
        g = KnowledgeGraph()
        g.add_node("weird", aliases=["—"])
        g.add_node("checkout-service", aliases=["Checkout"])
        assert g.mentions("???") == []
        assert g.mentions("") == []
        assert g.mentions("who owns Checkout?") == ["checkout_service"]

    def test_reinforcing_with_valid_to_ends_the_fact(self) -> None:
        g = KnowledgeGraph()
        first = g.add_edge("alice", "lives_in", "sp", valid_from=NOW - timedelta(days=9))
        ended = g.add_edge(
            "alice", "lives_in", "sp", valid_to=NOW, valid_from=NOW - timedelta(days=30)
        )
        assert ended.id == first.id
        assert ended.valid_from == NOW - timedelta(days=30)  # earliest known
        assert ended.valid_to == NOW
        assert g.neighbors("alice", as_of=NOW - timedelta(days=1)) == ["sp"]
        assert g.neighbors("alice", as_of=NOW) == []

    def test_communities_cache_expires_with_the_wall_clock(self) -> None:
        import time

        g = KnowledgeGraph()
        soon = datetime.now(UTC) + timedelta(seconds=0.2)
        g.add_edge("a", "r", "b", valid_to=soon)
        g.add_edge("c", "r", "d")
        before = g.communities()
        assert before["a"] == before["b"]
        time.sleep(0.3)
        after = g.communities()  # no write happened: the cache must still expire
        assert after["a"] != after["b"]
        assert g.hubs(k=4)[0][1] == 1  # c-d is the only live edge now

    def test_placeholder_label_is_replaced_by_the_first_real_name(self) -> None:
        g = KnowledgeGraph()
        g.add_edge("project x", "r", "y")  # via KnowledgeGraph the raw name is the label
        assert g.node("project x").label == "project x"  # type: ignore[union-attr]
        g.store.add_edge(GraphEdge(source="bare", target="y", relation="r"))
        assert g.node("bare").label == "bare"  # store-level auto-create: key as placeholder
        g.add_node("BARE", label="Bare Thing")
        assert g.node("bare").label == "Bare Thing"  # type: ignore[union-attr]


class TestReviewRegressionsBatchTwo:
    def test_naive_datetimes_are_rejected_at_the_boundary(self) -> None:
        g = _campaign()
        naive = datetime(2030, 1, 1)
        with pytest.raises(ValueError, match="timezone-aware"):
            g.add_edge("a", "r", "b", valid_to=naive)
        with pytest.raises(ValueError, match="timezone-aware"):
            GraphEdge(source="a", target="b", relation="r", created_at=naive)
        edge = g.edges("hero")[0]
        with pytest.raises(ValueError, match="timezone-aware"):
            g.invalidate_edge(edge.id, at=naive)
        with pytest.raises(ValueError, match="timezone-aware"):
            g.neighbors("hero", as_of=naive)
        assert g.neighbors("hero", as_of=NOW)  # aware is fine

    def test_add_edge_refuses_a_dead_edge(self) -> None:
        g = _campaign()
        with pytest.raises(ValueError, match="invalidate_edge"):
            g.store.add_edge(GraphEdge(source="a", target="b", relation="r", invalidated_at=NOW))

    def test_incoming_valid_from_corrects_the_start(self) -> None:
        g = KnowledgeGraph()
        g.add_edge("a", "r", "b", valid_from=NOW - timedelta(days=10))
        later = g.add_edge("a", "r", "b", valid_from=NOW - timedelta(days=2))
        assert later.valid_from == NOW - timedelta(days=2)
        assert g.add_edge("a", "r", "b").valid_from == NOW - timedelta(days=2)  # None keeps current

    def test_aliases_resolve_in_every_read(self) -> None:
        g = _campaign()
        g.add_node("checkout-service", aliases=["Checkout"])
        g.link_item("checkout-service", "doc-9", "/public")
        g.add_edge("checkout-service", "links_to", "hero", evidence=["doc-9"])
        assert g.node("Checkout") is not None
        assert g.node("Checkout").id == "checkout_service"  # type: ignore[union-attr]
        assert g.path("Checkout", "city") == ["checkout_service", "hero", "city"]
        assert g.explain("Checkout", "hero")[0].relation == "links_to"
        assert [e.source for e in g.backlinks("hero")][-1] == "checkout_service"
        assert g.items("Checkout") == ["doc-9"]
        assert "hero" in g.neighbors("Checkout")
        # an alias of a node hidden by the scope resolves to nothing
        assert g.node("Villain") is not None
        assert g.path("hero", "villain", scope=NO_SECRET) is None

    def test_separators_are_one_identity(self) -> None:
        assert normalize_key("auth-service") == normalize_key("Auth Service") == "auth_service"
        assert normalize_key("AUTH_SERVICE") == "auth_service"
        assert normalize_key("C++") == "c++"
        assert normalize_key(".NET") == ".net"
        assert normalize_key("--x--") == "x"
        g = KnowledgeGraph()
        g.add_node("auth-service")
        g.add_node("Auth Service", aliases=["auth"])
        assert g.nodes() == ["auth_service"]
        assert g.node("auth_service").aliases == ("auth",)  # type: ignore[union-attr]
