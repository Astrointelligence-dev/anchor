"""GraphRetriever: seeding, scoring, scope, fusion-friendly ids."""

from __future__ import annotations

import pytest

from anchor.exceptions import RetrieverError
from anchor.graph import KnowledgeGraph
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope
from anchor.protocols.retriever import Retriever
from anchor.retrieval import GraphRetriever, HybridRetriever
from anchor.storage.memory_store import (
    InMemoryContextStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from tests.conftest import FakeTokenizer


def _world() -> tuple[KnowledgeGraph, InMemoryContextStore]:
    """checkout —links_to→ payments —links_to→ bruno; each note has one chunk."""
    graph = KnowledgeGraph()
    store = InMemoryContextStore()
    for node, item_id, ns, text in (
        ("checkout-service", "c1", "/services", "Checkout hands cards to payments."),
        ("payments-service", "p1", "/services", "Payments is on-call Bruno."),
        ("bruno-costa", "b1", "/people", "Bruno Costa, payments on-call."),
    ):
        graph.add_node(node)
        graph.link_item(node, item_id, ns)
        store.add(ContextItem(id=item_id, content=text, source=SourceType.RETRIEVAL, namespace=ns))
    graph.add_node("checkout-service", aliases=["Checkout"])
    graph.add_edge("checkout-service", "links_to", "payments-service", evidence=["c1"])
    graph.add_edge("payments-service", "links_to", "bruno-costa", evidence=["p1"])
    return graph, store


class TestGraphRetriever:
    def test_is_a_retriever(self) -> None:
        graph, store = _world()
        assert isinstance(GraphRetriever(graph, store), Retriever)

    def test_mentions_seed_and_scores_normalize(self) -> None:
        graph, store = _world()
        r = GraphRetriever(graph, store, tokenizer=FakeTokenizer())
        items = r.retrieve(QueryBundle(query_str="Who is on call for Checkout?"), top_k=3)
        assert [i.id for i in items] == ["c1", "p1", "b1"]
        assert items[0].score == 1.0
        assert items[0].score > items[1].score > items[2].score > 0
        assert items[0].source == SourceType.RETRIEVAL
        assert items[0].metadata["retrieval_method"] == "graph"
        assert items[0].metadata["graph_seeds"] == ["checkout_service"]
        assert items[0].token_count > 0

    def test_custom_entity_extractor(self) -> None:
        graph, store = _world()
        r = GraphRetriever(graph, store, entity_extractor=lambda q: ["bruno-costa"])
        # bruno is evidenced by b1 (linked) and p1 (the payments edge mentions him)
        assert {i.id for i in r.retrieve(QueryBundle(query_str="anything"), top_k=2)} == {
            "b1",
            "p1",
        }

    def test_no_seed_is_empty(self) -> None:
        graph, store = _world()
        assert GraphRetriever(graph, store).retrieve(QueryBundle(query_str="nothing here")) == []

    def test_scope_hides_items_and_walls_the_walk(self) -> None:
        graph, store = _world()
        r = GraphRetriever(graph, store)
        items = r.retrieve(
            QueryBundle(query_str="Checkout"), top_k=5, scope=RetrievalScope(exclude=("/people",))
        )
        assert [i.id for i in items] == ["c1", "p1"]
        assert (
            r.retrieve(
                QueryBundle(query_str="Checkout"), scope=RetrievalScope(include=("/people",))
            )
            == []
        )

    def test_dense_seeding(self) -> None:
        graph, store = _world()
        vectors = InMemoryVectorStore()
        vectors.add_embedding("b1", [1.0, 0.0], namespace="/people")
        vectors.add_embedding("c1", [0.0, 1.0], namespace="/services")
        r = GraphRetriever(graph, store, vector_store=vectors, seed_k=1)
        items = r.retrieve(
            QueryBundle(query_str="no node named here", embedding=[1.0, 0.0]), top_k=3
        )
        assert items[0].id == "b1"  # the passage seed wins, the walk adds the rest
        assert {i.id for i in items} == {"b1", "p1", "c1"}
        with pytest.raises(RetrieverError, match="embeddings"):
            r.retrieve(QueryBundle(query_str="no embedding"))

    def test_items_missing_from_context_store_are_skipped(self) -> None:
        graph, store = _world()
        store.delete("p1")
        items = GraphRetriever(graph, store).retrieve(QueryBundle(query_str="Checkout"), top_k=3)
        assert [i.id for i in items] == ["c1", "b1"]

    def test_vault_mismatch_refused(self) -> None:
        graph = KnowledgeGraph(InMemoryGraphStore(vault="a"))
        with pytest.raises(ValueError, match="vault"):
            GraphRetriever(graph, InMemoryContextStore(vault="b"))

    def test_fuses_with_rrf_by_canonical_id(self) -> None:
        graph, store = _world()

        class Lexical:
            def retrieve(
                self, query: QueryBundle, top_k: int = 10, *, scope=None
            ) -> list[ContextItem]:
                item = store.get("p1")
                assert item is not None
                return [item.model_copy(update={"score": 1.0})]

        fused = HybridRetriever([Lexical(), GraphRetriever(graph, store)]).retrieve(
            QueryBundle(query_str="Checkout"), top_k=3
        )
        assert fused[0].id == "p1"  # found by both lists → top after fusion
        assert len({i.id for i in fused}) == len(fused) == 3


class TestReviewRegressions:
    def test_scores_normalize_against_the_best_resolved_item(self) -> None:
        graph, store = _world()
        graph.add_node("memory-only")
        graph.link_item("memory-only", "mem-1")  # not in the ContextStore
        graph.add_edge("memory-only", "links_to", "checkout-service", evidence=["mem-1"])
        r = GraphRetriever(graph, store, entity_extractor=lambda q: ["memory-only"])
        items = r.retrieve(QueryBundle(query_str="x"), top_k=5)
        assert items[0].score == 1.0
        assert "mem-1" not in {i.id for i in items}

    def test_memory_only_graph_yields_nothing_quietly(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node("m")
        graph.link_item("m", "mem-1")
        assert (
            GraphRetriever(graph, InMemoryContextStore()).retrieve(QueryBundle(query_str="m")) == []
        )

    def test_bad_seed_from_extractor_is_ignored(self) -> None:
        graph, store = _world()
        r = GraphRetriever(graph, store, entity_extractor=lambda q: ["", "  ", "checkout-service"])
        assert [i.id for i in r.retrieve(QueryBundle(query_str="x"), top_k=1)] == ["c1"]
