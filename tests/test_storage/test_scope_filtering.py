"""Front #3: vault isolation + namespace scope filtering in vector stores.

Written failing-first. The decoy pattern (an out-of-scope item whose
embedding IS the query) proves pre-filtering: a post-filter or a naive
OR pushed into the KNN loses a top-k slot to the decoy and returns
fewer/wrong rows.
"""

from __future__ import annotations

import pytest

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import DEFAULT_VAULT, RetrievalScope
from tests.conftest import make_embedding


class TestVaultIsolation:
    def test_other_vault_is_invisible(self, make_vector_store):
        a = make_vector_store("vault-a")
        b = make_vector_store("vault-b")
        emb = make_embedding(1)
        a.add_embedding("v1", emb)

        assert [i for i, _ in a.search(emb, top_k=5)] == ["v1"]
        assert b.search(emb, top_k=5) == []

    def test_same_id_in_two_vaults_coexists(self, make_vector_store):
        a = make_vector_store("vault-a")
        b = make_vector_store("vault-b")
        a.add_embedding("shared-id", make_embedding(1))
        b.add_embedding("shared-id", make_embedding(2))

        assert [i for i, _ in a.search(make_embedding(1), top_k=5)] == [
            "shared-id"
        ]
        assert [i for i, _ in b.search(make_embedding(2), top_k=5)] == [
            "shared-id"
        ]


class TestNamespaceScope:
    def _seed(self, store):
        store.add_embedding(
            "regras", make_embedding(1), namespace="/regras/combate",
        )
        store.add_embedding(
            "sessao", make_embedding(2), namespace="/campanha-1/sessoes",
        )
        store.add_embedding(
            "spoiler", make_embedding(3), namespace="/campanha-1/spoilers",
        )

    def test_no_scope_sees_all(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        ids = {i for i, _ in store.search(make_embedding(1), top_k=10)}
        assert ids == {"regras", "sessao", "spoiler"}

    def test_include_prefix(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        scope = RetrievalScope(include=("/regras",))
        ids = [i for i, _ in store.search(
            make_embedding(1), top_k=10, scope=scope,
        )]
        assert ids == ["regras"]

    def test_exclude_wins_even_when_nearest(self, make_vector_store):
        """The spoilers test: the excluded item's embedding IS the query."""
        store = make_vector_store()
        self._seed(store)
        scope = RetrievalScope(
            include=("/campanha-1",), exclude=("/campanha-1/spoilers",),
        )
        # Query with the spoiler's own embedding — globally nearest.
        results = store.search(make_embedding(3), top_k=2, scope=scope)
        ids = [i for i, _ in results]
        assert "spoiler" not in ids
        assert "sessao" in ids  # the slot was not eaten by the decoy

    def test_prefix_is_boundary_aware(self, make_vector_store):
        store = make_vector_store()
        store.add_embedding("c1", make_embedding(1), namespace="/campanha-1")
        store.add_embedding("c10", make_embedding(2), namespace="/campanha-10")
        scope = RetrievalScope(include=("/campanha-1",))
        ids = [i for i, _ in store.search(
            make_embedding(1), top_k=10, scope=scope,
        )]
        assert ids == ["c1"]

    def test_multi_include_with_decoy(self, make_vector_store):
        """Two include prefixes + nearest-item decoy outside both.

        A naive OR pushed into the KNN (or a post-filter) loses top-k
        slots to the decoy — verified failure mode on vec0.
        """
        store = make_vector_store()
        store.add_embedding("a", make_embedding(1), namespace="/alpha/x")
        store.add_embedding("b", make_embedding(2), namespace="/beta/y")
        store.add_embedding("decoy", make_embedding(9), namespace="/fora")
        scope = RetrievalScope(include=("/alpha", "/beta"))

        # Query with the decoy's own embedding, k=2: both in-scope items
        # must come back; the decoy must not eat a slot.
        results = store.search(make_embedding(9), top_k=2, scope=scope)
        assert {i for i, _ in results} == {"a", "b"}


class TestWhereOperators:
    def _seed(self, store):
        store.add_embedding("d1", make_embedding(1), {"year": 2024, "kind": "lei"})
        store.add_embedding("d2", make_embedding(2), {"year": 2025, "kind": "contrato"})
        store.add_embedding("d3", make_embedding(3), {"year": 2026, "kind": "contrato"})

    def test_equality_still_works(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10, where={"kind": "contrato"},
        )}
        assert ids == {"d2", "d3"}

    def test_in_operator(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10, where={"year": {"$in": [2024, 2026]}},
        )}
        assert ids == {"d1", "d3"}

    def test_ne_and_nin(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10, where={"kind": {"$ne": "lei"}},
        )}
        assert ids == {"d2", "d3"}
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10, where={"year": {"$nin": [2025]}},
        )}
        assert ids == {"d1", "d3"}

    def test_range_operators(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10, where={"year": {"$gt": 2024}},
        )}
        assert ids == {"d2", "d3"}
        ids = {i for i, _ in store.search(
            make_embedding(1), top_k=10,
            where={"year": {"$gte": 2024, "$lt": 2026}},
        )}
        assert ids == {"d1", "d2"}

    def test_unknown_operator_raises(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        with pytest.raises(ValueError, match="\\$regex"):
            store.search(
                make_embedding(1), top_k=10, where={"kind": {"$regex": ".*"}},
            )


@pytest.fixture(params=["memory", "sqlite"])
def make_context_store(request, tmp_path):
    if request.param == "memory":
        from anchor.storage.memory_store import InMemoryContextStore

        def maker(vault: str = DEFAULT_VAULT):
            return InMemoryContextStore(vault=vault)

        return maker

    from anchor.storage.sqlite import (
        SqliteConnectionManager,
        SqliteContextStore,
        ensure_tables,
    )

    mgr = SqliteConnectionManager(tmp_path / "ctx.db")
    ensure_tables(mgr.get_connection())

    def maker(vault: str = DEFAULT_VAULT):
        return SqliteContextStore(mgr, vault=vault)

    return maker


class TestContextStoreScopePersistence:
    def test_namespace_round_trips(self, make_context_store):
        store = make_context_store()
        item = ContextItem(
            id="i1", content="x", source=SourceType.RETRIEVAL,
            namespace="/contratos/2026",
        )
        store.add(item)
        got = store.get("i1")
        assert got is not None
        assert got.namespace == "/contratos/2026"

    def test_store_vault_is_stamped(self, make_context_store):
        store = make_context_store("juridico")
        store.add(ContextItem(id="i1", content="x", source=SourceType.RETRIEVAL))
        got = store.get("i1")
        assert got is not None
        assert got.vault == "juridico"

    def test_get_by_id_does_not_cross_vaults(self, make_context_store):
        a = make_context_store("vault-a")
        b = make_context_store("vault-b")
        a.add(ContextItem(id="i1", content="x", source=SourceType.RETRIEVAL))
        assert a.get("i1") is not None
        assert b.get("i1") is None


class TestWhereSemanticsAgreeAcrossBackends:
    """Ritual xhigh #12/#13: the same where dict, the same answer on every
    backend — absent keys pass $ne/$nin, ranges ignore values of another type."""

    def _seed(self, store):
        store.add_embedding("tagged", make_embedding(1), {"status": "active", "year": 2024})
        store.add_embedding("later", make_embedding(2), {"year": 2025})
        store.add_embedding("untagged", make_embedding(3), {})
        store.add_embedding("odd", make_embedding(4), {"year": "unknown"})

    def _ids(self, store, where):
        return {i for i, _ in store.search(make_embedding(1), top_k=10, where=where)}

    def test_absent_key_passes_ne_and_nin(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        assert self._ids(store, {"status": {"$ne": "archived"}}) == {
            "tagged", "later", "untagged", "odd",
        }
        assert self._ids(store, {"year": {"$nin": [2025]}}) == {
            "tagged", "untagged", "odd",
        }

    def test_ranges_ignore_values_of_another_type(self, make_vector_store):
        store = make_vector_store()
        self._seed(store)
        assert self._ids(store, {"year": {"$gt": 2024}}) == {"later"}
        assert self._ids(store, {"year": {"$lt": 2025}}) == {"tagged"}
        assert self._ids(store, {"year": {"$gte": "unknown"}}) == {"odd"}

