"""GraphStore contract across backends — the in-memory store is the reference.

Every backend must give the same answers on the campaign world (scope
decoy, wall-not-bridge, temporal validity, incremental maintenance) and
isolate vaults sharing one database file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from anchor.graph import KnowledgeGraph
from anchor.models.scope import DEFAULT_VAULT, RetrievalScope
from anchor.protocols.storage import AsyncGraphStore, GraphStore
from anchor.storage.memory_store import InMemoryGraphStore
from anchor.storage.sqlite import (
    AsyncSqliteGraphStore,
    SqliteConnectionManager,
    SqliteGraphStore,
    ensure_tables,
)
from tests.test_graph.test_knowledge_graph import NO_SECRET, _campaign

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def make_graph_store(request, tmp_path):
    """Factory: make_graph_store(vault) → store; sqlite stores share ONE file."""
    if request.param == "memory":

        def maker(vault: str = DEFAULT_VAULT):
            return InMemoryGraphStore(vault=vault)

        return maker
    mgr = SqliteConnectionManager(tmp_path / "graph.db")
    ensure_tables(mgr.get_connection())

    def maker(vault: str = DEFAULT_VAULT):
        return SqliteGraphStore(mgr, vault=vault)

    return maker


class TestContract:
    def test_protocol_and_vault(self, make_graph_store) -> None:
        store = make_graph_store("dnd")
        assert isinstance(store, GraphStore)
        assert store.vault == "dnd"
        with pytest.raises(ValueError):
            make_graph_store("a/b")

    def test_navigation_matches_reference(self, make_graph_store) -> None:
        g = _campaign(KnowledgeGraph(make_graph_store()))
        ref = _campaign()
        for name in ("hero", "villain", "city", "ally"):
            assert g.neighbors(name, max_depth=2) == ref.neighbors(name, max_depth=2)
            assert g.items(name) == ref.items(name)
            assert [e.relation for e in g.backlinks(name)] == [
                e.relation for e in ref.backlinks(name)
            ]
        assert g.path("villain", "ally") == ["villain", "city", "ally"]
        assert [
            (e.source, e.relation, e.target, e.evidence) for e in g.explain("villain", "ally")
        ] == [
            ("villain", "rules", "city", ("doc-2",)),
            ("ally", "lives_in", "city", ("doc-3",)),
        ]
        assert g.hubs(k=2) == [("city", 3), ("hero", 3)]
        assert [i for i, _ in g.query(["villain"], top_k=3)] == [
            i for i, _ in ref.query(["villain"], top_k=3)
        ]
        assert g.related_items("hero", max_depth=1) == ["doc-1", "doc-2", "doc-3"]

    def test_scope_decoy_and_wall(self, make_graph_store) -> None:
        g = _campaign(KnowledgeGraph(make_graph_store()))
        assert "villain" not in g.nodes(scope=NO_SECRET)
        assert g.neighbors("villain", scope=NO_SECRET) == []
        assert "villain" not in g.neighbors("hero", max_depth=5, scope=NO_SECRET)
        assert g.path("hero", "villain", scope=NO_SECRET) is None
        assert {e.source for e in g.backlinks("city", scope=NO_SECRET)} == {"hero", "ally"}
        assert g.hubs(k=3, scope=NO_SECRET) == [("ally", 2), ("city", 2), ("hero", 2)]
        assert "doc-2" not in {i for i, _ in g.query(["hero"], top_k=10, scope=NO_SECRET)}
        for edge in g.edges("hero"):
            if edge.relation in ("lives_in", "allied_with"):
                g.invalidate_edge(edge.id)
        assert g.path("hero", "city") == ["hero", "villain", "city"]
        assert g.path("hero", "city", scope=NO_SECRET) is None
        only_allies = RetrievalScope(include=("/public/allies",))
        assert g.nodes(scope=only_allies) == ["ally", "city"]
        assert g.items("city", scope=only_allies) == ["doc-3"]

    def test_node_without_evidence_exists_only_unscoped(self, make_graph_store) -> None:
        g = _campaign(KnowledgeGraph(make_graph_store()))
        g.add_edge("hero", "seeks", "artifact")
        assert "artifact" in g.neighbors("hero")
        assert "artifact" not in g.neighbors("hero", scope=RetrievalScope())

    def test_temporal(self, make_graph_store) -> None:
        g = KnowledgeGraph(make_graph_store())
        g.add_edge("alice", "lives_in", "sp", valid_to=NOW)
        rio = g.add_edge("alice", "lives_in", "rio", valid_from=NOW)
        assert g.neighbors("alice", as_of=NOW - timedelta(days=1)) == ["sp"]
        assert g.neighbors("alice", as_of=NOW) == ["rio"]
        assert g.invalidate_edge(rio.id, at=NOW) is True
        assert g.invalidate_edge(rio.id) is False
        assert g.neighbors("alice", as_of=NOW) == []
        stored = g.store.get_edge(rio.id)
        assert stored is not None
        assert stored.invalidated_at == NOW  # history kept, timezone intact
        assert stored.valid_from == NOW

    def test_reinforce_merge_and_ids(self, make_graph_store) -> None:
        g = KnowledgeGraph(make_graph_store())
        g.add_node("a", metadata={"k": 1}, aliases=["A1"])
        node = g.add_node("A", metadata={"j": 2}, aliases=["A2"])
        assert (node.label, node.aliases, node.metadata) == ("a", ("A1", "A2"), {"k": 1, "j": 2})
        g.link_item("a", "i1", "/x")
        g.link_item("a", "i2", "/y")
        first = g.add_edge("a", "r", "b", evidence=["i1"], provenance="inferred", confidence=0.6)
        second = g.add_edge("a", "r", "b", evidence=["i2"], confidence=0.9, fact="a r b")
        assert second.id == first.id
        assert (second.evidence, second.provenance, second.confidence, second.fact) == (
            ("i1", "i2"),
            "extracted",
            0.9,
            "a r b",
        )
        assert g.store.get_edge(first.id) == second
        assert len(g.edges("a")) == 1
        with pytest.raises(ValueError, match="link_item"):
            g.add_edge("a", "s", "c", evidence=["unknown"])
        with pytest.raises(KeyError):
            g.link_item("nobody", "i1")
        with pytest.raises(ValueError, match="namespace"):
            g.link_item("b", "i1", "/other")
        with pytest.raises(ValueError, match="already exists"):
            g.store.add_edge(first.model_copy(update={"relation": "t"}))

    def test_unlink_and_remove(self, make_graph_store) -> None:
        g = KnowledgeGraph(make_graph_store())
        g.add_node("a")
        g.link_item("a", "i1")
        g.link_item("a", "i2")
        shared = g.add_edge("a", "r", "b", evidence=["i1", "i2"])
        lonely = g.add_edge("a", "s", "c", evidence=["i1"])
        assert g.unlink_item("i1") == 1
        assert g.unlink_item("i1") == 0
        assert g.store.get_edge(shared.id).evidence == ("i2",)  # type: ignore[union-attr]
        assert g.store.get_edge(lonely.id).invalidated_at is not None  # type: ignore[union-attr]
        assert g.items("a") == ["i2"]
        assert g.remove_node("b") is True
        assert g.remove_node("b") is False
        assert g.edges("a") == []
        assert g.store.get_edge(shared.id).invalidated_at is not None  # type: ignore[union-attr]

    def test_version_and_clear(self, make_graph_store) -> None:
        g = KnowledgeGraph(make_graph_store())
        v0 = g.store.version
        g.add_node("a")
        assert g.store.version == v0 + 1
        g.nodes()
        assert g.store.version == v0 + 1
        g.clear()
        assert g.nodes() == []
        assert g.store.version == v0 + 2

    def test_vault_isolation_on_one_backing(self, make_graph_store) -> None:
        a = KnowledgeGraph(make_graph_store("a"))
        b = KnowledgeGraph(make_graph_store("b"))
        a.add_node("shared", metadata={"from": "a"})
        b.add_node("shared", metadata={"from": "b"})
        a.link_item("shared", "doc", "/x")
        b.link_item("shared", "doc", "/y")  # same item id, other vault, other namespace
        a.add_edge("shared", "r", "only-a", evidence=["doc"])
        assert a.node("shared").metadata == {"from": "a"}  # type: ignore[union-attr]
        assert b.node("shared").metadata == {"from": "b"}  # type: ignore[union-attr]
        assert b.neighbors("shared") == []
        assert b.items("shared", scope=RetrievalScope(include=("/y",))) == ["doc"]
        assert a.items("shared", scope=RetrievalScope(include=("/y",))) == []
        assert b.unlink_item("doc") == 0
        assert a.edges("shared") != []
        b.clear()
        assert a.node("shared") is not None
        assert a.store.version != b.store.version


class TestSqlitePersistence:
    def test_survives_reopen(self, tmp_path) -> None:
        db = tmp_path / "g.db"
        mgr = SqliteConnectionManager(db)
        ensure_tables(mgr.get_connection())
        _campaign(KnowledgeGraph(SqliteGraphStore(mgr)))
        mgr.close()
        mgr2 = SqliteConnectionManager(db)
        ensure_tables(mgr2.get_connection())  # idempotent on an existing graph
        g = KnowledgeGraph(SqliteGraphStore(mgr2))
        assert g.path("villain", "ally") == ["villain", "city", "ally"]
        assert g.items("city") == ["doc-1", "doc-3"]
        assert "villain" not in g.nodes(scope=NO_SECRET)


class TestAsyncTwin:
    async def test_async_store(self, tmp_path) -> None:
        mgr = SqliteConnectionManager(tmp_path / "g.db")
        ensure_tables(mgr.get_connection())
        store = AsyncSqliteGraphStore(mgr, vault="v")
        assert isinstance(store, AsyncGraphStore)
        assert store.vault == "v"
        from anchor.models.graph import GraphEdge, GraphNode

        await store.upsert_node(GraphNode(id="a"))
        await store.link_item("a", "i1", "/x")
        edge = await store.add_edge(
            GraphEdge(source="a", target="b", relation="r", evidence=("i1",))
        )
        assert (await store.get_edge(edge.id)) == edge
        assert [e.id for e in await store.edges_of("b")] == [edge.id]
        assert await store.node_items("a", scope=RetrievalScope(include=("/x",))) == ["i1"]
        assert (await store.subgraph(scope=RetrievalScope(exclude=("/x",)))).nodes == {}
        assert await store.unlink_item("i1") == 1
        assert await store.invalidate_edge(edge.id) is False
        assert await store.remove_node("a") is True
        await store.clear()
        assert store.version > 0
