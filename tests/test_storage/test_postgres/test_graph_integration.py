"""PostgresGraphStore against a real database (``ANCHOR_TEST_POSTGRES_DSN``).

The contract: the same world built on ``InMemoryGraphStore`` (the
reference) and on Postgres must answer identically — subgraph, edges_of and
node_items under several scopes and times — and vaults sharing the
database must not see each other.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from anchor.models.graph import GraphEdge, GraphNode
from anchor.models.scope import RetrievalScope
from anchor.protocols.storage import AsyncGraphStore
from anchor.storage.memory_store import InMemoryGraphStore

pytest.importorskip("asyncpg")

DSN = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="ANCHOR_TEST_POSTGRES_DSN not set")

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
_TABLES = (
    "graph_nodes",
    "graph_edges",
    "graph_items",
    "graph_node_items",
    "graph_edge_items",
    "graph_meta",
)
SCOPES = (
    None,
    RetrievalScope(exclude=("/secret",)),
    RetrievalScope(include=("/public/allies",)),
    RetrievalScope(include=("/public",), exclude=("/public/allies",)),
)


async def _build(store) -> None:
    """The campaign world, through the store API (works sync and async)."""

    async def call(fn, *a, **k):
        r = fn(*a, **k)
        return await r if asyncio.iscoroutine(r) else r

    for name in ("hero", "villain", "city", "ally"):
        await call(store.upsert_node, GraphNode(id=name))
    for node, item, ns in (
        ("hero", "doc-1", "/public"),
        ("city", "doc-1", "/public"),
        ("villain", "doc-2", "/secret"),
        ("ally", "doc-3", "/public/allies"),
        ("city", "doc-3", "/public/allies"),
    ):
        await call(store.link_item, node, item, ns)
    edges = [
        ("hero", "knows", "villain", ("doc-2",), {}),
        ("villain", "rules", "city", ("doc-2",), {}),
        ("hero", "lives_in", "city", ("doc-1",), {"fact": "the hero lives in the city"}),
        ("hero", "allied_with", "ally", ("doc-3",), {}),
        ("ally", "lives_in", "city", ("doc-3",), {"valid_from": NOW}),
        ("hero", "seeks", "artifact", (), {"valid_to": NOW}),
    ]
    for s, r, t, ev, extra in edges:
        await call(store.add_edge, GraphEdge(source=s, target=t, relation=r, evidence=ev, **extra))
    # reinforce one edge from another item, then drop an item
    await call(
        store.add_edge,
        GraphEdge(
            source="hero", target="city", relation="lives_in", evidence=("doc-3",), confidence=0.5
        ),
    )
    await call(store.unlink_item, "doc-2")


def _shape(sub):
    return (
        sorted(sub.nodes),
        sorted(
            (e.source, e.relation, e.target, e.evidence, e.confidence, e.fact) for e in sub.edges
        ),
        {n: tuple(i) for n, i in sub.items.items()},
    )


async def _with_store(fn):
    import asyncpg

    from anchor.storage.postgres import PostgresConnectionManager, PostgresGraphStore, ensure_tables

    conn = await asyncpg.connect(DSN)
    try:
        for table in _TABLES:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await ensure_tables(conn, embedding_dim=3)
        await ensure_tables(conn, embedding_dim=3)  # idempotent
        manager = PostgresConnectionManager(DSN, min_size=1, max_size=2)
        await manager.initialize()
        try:
            return await fn(lambda vault="__default__": PostgresGraphStore(manager, vault=vault))
        finally:
            await manager.close()
    finally:
        for table in _TABLES:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.close()


def test_matches_in_memory_reference():
    async def run(make):
        pg = make()
        assert isinstance(pg, AsyncGraphStore)
        ref = InMemoryGraphStore()
        await _build(pg)
        await _build(ref)
        for scope in SCOPES:
            for as_of in (None, NOW - timedelta(days=1), NOW + timedelta(days=1)):
                assert _shape(await pg.subgraph(scope=scope, as_of=as_of)) == _shape(
                    ref.subgraph(scope=scope, as_of=as_of)
                ), (scope, as_of)
                for node in ("hero", "villain", "city", "ally", "artifact"):
                    got = await pg.edges_of(node, scope=scope, as_of=as_of)
                    want = ref.edges_of(node, scope=scope, as_of=as_of)
                    assert [(e.source, e.relation, e.target, e.evidence) for e in got] == [
                        (e.source, e.relation, e.target, e.evidence) for e in want
                    ], (node, scope, as_of)
                    assert await pg.node_items(node, scope=scope) == ref.node_items(
                        node, scope=scope
                    )
        assert await pg.edges_of("villain") == []  # doc-2 was the only evidence → invalidated
        hero_city = next(
            e
            for e in (await pg.subgraph()).edges
            if e.relation == "lives_in" and e.source == "hero"
        )
        stored = await pg.get_edge(hero_city.id)
        assert stored is not None
        assert stored.evidence == ("doc-1", "doc-3")  # reinforced, timezone-aware datetimes intact
        assert stored.created_at.tzinfo is not None
        with pytest.raises(ValueError, match="link_item"):
            await pg.add_edge(GraphEdge(source="a", target="b", relation="r", evidence=("nope",)))
        with pytest.raises(KeyError):
            await pg.link_item("nobody", "x")
        await pg.link_item("hero", "doc-1", "/elsewhere")  # the latest link moves the item
        assert await pg.node_items("hero", scope=RetrievalScope(include=("/elsewhere",))) == [
            "doc-1"
        ]
        moved = await pg.node_items("city", scope=RetrievalScope(include=("/public",)))
        assert "doc-1" not in moved  # it left /public with the relink

    asyncio.run(_with_store(run))


def test_vault_isolation_and_clear():
    async def run(make):
        a, b = make("a"), make("b")
        await a.upsert_node(GraphNode(id="shared", metadata={"from": "a"}))
        await b.upsert_node(GraphNode(id="shared", metadata={"from": "b"}))
        await a.link_item("shared", "doc", "/x")
        await b.link_item("shared", "doc", "/y")
        await a.add_edge(
            GraphEdge(source="shared", target="only-a", relation="r", evidence=("doc",))
        )
        assert (await a.get_node("shared")).metadata == {"from": "a"}
        assert (await b.get_node("shared")).metadata == {"from": "b"}
        assert await b.edges_of("shared") == []
        assert await a.node_items("shared", scope=RetrievalScope(include=("/y",))) == []
        assert await b.node_items("shared", scope=RetrievalScope(include=("/y",))) == ["doc"]
        assert await b.unlink_item("doc") == 0
        assert await b.remove_node("shared") is True
        assert (await a.get_node("shared")) is not None
        await a.clear()
        assert (await a.subgraph()).nodes == {}
        assert a.version >= 4  # in-process write counter: upsert, link, edge, clear
        assert b.version >= 3

    asyncio.run(_with_store(run))
