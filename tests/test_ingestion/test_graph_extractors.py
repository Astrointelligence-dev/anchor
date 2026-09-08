"""Deterministic graph extraction: wikilinks, structure, indexer."""

from __future__ import annotations

import pytest

from anchor.graph import KnowledgeGraph
from anchor.ingestion.graph_extractors import (
    Extraction,
    GraphExtractor,
    GraphIndexer,
    StructureExtractor,
    Wikilink,
    WikilinkExtractor,
    note_name,
    parse_wikilink,
    wikilinks,
)
from anchor.models.context import ContextItem, SourceType
from anchor.models.graph import GraphEdge
from anchor.models.scope import RetrievalScope
from anchor.storage.memory_store import InMemoryGraphStore


def _item(
    content: str,
    *,
    item_id: str = "c1",
    filename: str | None = "checkout-service.md",
    namespace: str = "/",
    vault: str = "__default__",
    **doc_meta: object,
) -> ContextItem:
    metadata: dict[str, object] = {"parent_doc_id": "doc-1", **doc_meta}
    if filename is not None:
        metadata["doc_filename"] = filename
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        metadata=metadata,
        namespace=namespace,
        vault=vault,
    )


class TestWikilinkGrammar:
    @pytest.mark.parametrize(
        ("inner", "expected"),
        [
            ("Auth", Wikilink("Auth")),
            ("Auth|the auth service", Wikilink("Auth", alias="the auth service")),
            ("Auth#Tokens", Wikilink("Auth", anchor="Tokens")),
            ("Auth#^blk|alias", Wikilink("Auth", anchor="^blk", alias="alias")),
            ("folder/sub/Auth.md", Wikilink("Auth")),
            ("  Auth  ", Wikilink("Auth")),
        ],
    )
    def test_parse(self, inner: str, expected: Wikilink) -> None:
        assert parse_wikilink(inner) == expected

    def test_same_note_anchor_is_not_a_link(self) -> None:
        assert parse_wikilink("#Heading") is None
        assert parse_wikilink("|alias only") is None

    def test_embed_and_order(self) -> None:
        links = wikilinks("see ![[diagram.png]] and [[a]] then [[b|B]] and [[]] end")
        assert links == [
            Wikilink("diagram.png", embed=True),
            Wikilink("a"),
            Wikilink("b", alias="B"),
        ]


class TestStructureExtractor:
    def test_note_node_with_aliases_and_tags(self) -> None:
        item = _item("body", doc_aliases=["Checkout", "cart"], doc_tags="service, #payments")
        found = StructureExtractor().extract(item)
        assert [n.id for n in found.nodes] == ["checkout_service", "service", "payments"]
        note = found.nodes[0]
        assert note.aliases == ("Checkout", "cart")
        assert note.metadata == {"kind": "note", "doc_id": "doc-1"}
        assert [(e.source, e.relation, e.target) for e in found.edges] == [
            ("checkout_service", "tagged", "service"),
            ("checkout_service", "tagged", "payments"),
        ]

    def test_title_fallback_and_no_identity(self) -> None:
        assert note_name(_item("x", filename=None, doc_title="My Note")) == "My Note"
        assert note_name(_item("x", filename=None)) is None
        assert StructureExtractor().extract(_item("x", filename=None)).nodes == []


class TestWikilinkExtractor:
    def test_links_to_edges_from_the_chunks_note(self) -> None:
        text = (
            "uses [[payments-service]] and [[fraud-scoring#thresholds|fraud]] "
            "twice [[payments-service]]"
        )
        found = WikilinkExtractor().extract(_item(text))
        assert [n.id for n in found.nodes] == [
            "payments_service",
            "fraud_scoring",
            "payments_service",
        ]
        assert found.nodes[1].aliases == ("fraud",)
        edges = [(e.source, e.relation, e.target) for e in found.edges]
        assert edges == [
            ("checkout_service", "links_to", "payments_service"),
            ("checkout_service", "links_to", "fraud_scoring"),
        ]
        assert found.edges[1].metadata == {"extractor": "wikilink", "anchor": "thresholds"}
        assert found.edges[0].evidence == ()  # the indexer stamps the item

    def test_self_link_skipped_and_mentions_without_identity(self) -> None:
        found = WikilinkExtractor().extract(_item("[[Checkout-Service]] links itself"))
        assert found.nodes == []
        assert found.edges == []
        found = WikilinkExtractor().extract(_item("[[a]] [[b]]", filename=None))
        assert [n.id for n in found.nodes] == ["a", "b"]
        assert found.edges == []

    def test_satisfies_protocol(self) -> None:
        assert isinstance(WikilinkExtractor(), GraphExtractor)
        assert isinstance(StructureExtractor(), GraphExtractor)


class TestGraphIndexer:
    def test_indexes_notes_links_and_evidence(self) -> None:
        graph = KnowledgeGraph()
        items = [
            _item(
                "Checkout hands cards to [[payments-service]].", item_id="c1", namespace="/services"
            ),
            _item(
                "Payments is on-call [[bruno-costa]].",
                item_id="p1",
                filename="payments-service.md",
                namespace="/services",
            ),
            _item(
                "Bruno maintains [[ledger-service]].",
                item_id="b1",
                filename="bruno-costa.md",
                namespace="/people",
            ),
        ]
        stats = GraphIndexer(graph).index(items)
        assert (stats.items, stats.nodes, stats.edges) == (3, 4, 3)  # distinct, not emissions
        assert graph.path("checkout_service", "ledger_service") == [
            "checkout_service",
            "payments_service",
            "bruno_costa",
            "ledger_service",
        ]
        hop = graph.explain("checkout_service", "payments_service")[0]
        assert hop.evidence == ("c1",)
        assert hop.provenance == "extracted"
        # the target note is a mention of the chunk that links to it
        assert graph.items("payments_service") == ["c1", "p1"]
        assert graph.items("bruno_costa", scope=RetrievalScope(exclude=("/people",))) == ["p1"]

    def test_reindex_is_idempotent(self) -> None:
        graph = KnowledgeGraph()
        indexer = GraphIndexer(graph)
        item = _item("see [[payments-service]]")
        indexer.index([item])
        version = graph.store.version
        indexer.index([item])
        assert graph.store.version > version  # writes happened...
        assert len(graph.edges("checkout_service")) == 1  # ...but nothing duplicated
        assert graph.explain("checkout_service", "payments_service")[0].evidence == ("c1",)
        assert graph.items("payments_service") == ["c1"]

    def test_unlink_item_is_the_removal_path(self) -> None:
        graph = KnowledgeGraph()
        GraphIndexer(graph).index([_item("see [[payments-service]]")])
        assert graph.unlink_item("c1") == 1
        assert graph.edges("checkout_service") == []
        # no evidence left: the nodes fall back to the root namespace
        assert graph.nodes(scope=RetrievalScope(include=("/x",))) == []
        assert graph.nodes(scope=RetrievalScope()) == ["checkout_service", "payments_service"]

    def test_refuses_items_from_another_vault(self) -> None:
        graph = KnowledgeGraph(InMemoryGraphStore(vault="dnd"))
        with pytest.raises(ValueError, match="vault"):
            GraphIndexer(graph).index([_item("x", vault="other")])

    def test_custom_extractor_list(self) -> None:
        graph = KnowledgeGraph()
        indexer = GraphIndexer(graph, extractors=[WikilinkExtractor()])
        assert indexer.extractors == (WikilinkExtractor(),) or len(indexer.extractors) == 1
        indexer.index([_item("[[a]]", doc_tags=["t"])])
        assert "t" not in graph.nodes()


class TestReviewRegressions:
    def test_bare_hash_tag_is_ignored(self) -> None:
        found = StructureExtractor().extract(_item("x", doc_tags=["#", "#todo", " "]))
        assert [n.id for n in found.nodes] == ["checkout_service", "todo"]
        found = StructureExtractor().extract(_item("x", doc_tags="#todo, #"))
        assert [n.id for n in found.nodes] == ["checkout_service", "todo"]

    def test_edge_only_extractor_gets_its_endpoints_linked(self) -> None:
        class EdgesOnly:
            def extract(self, item: ContextItem) -> Extraction:
                return Extraction(edges=[GraphEdge(source="Left", target="Right", relation="r")])

        graph = KnowledgeGraph()
        stats = GraphIndexer(graph, extractors=[EdgesOnly()]).index([_item("x", namespace="/z")])
        assert (stats.nodes, stats.edges) == (2, 1)
        scope = RetrievalScope(include=("/z",))
        assert graph.nodes(scope=scope) == ["left", "right"]
        assert graph.path("left", "right", scope=scope) == ["left", "right"]

    def test_reindex_under_another_namespace_moves_the_evidence(self) -> None:
        graph = KnowledgeGraph()
        indexer = GraphIndexer(graph)
        indexer.index([_item("see [[payments-service]]", namespace="/a")])
        indexer.index([_item("see [[payments-service]]", namespace="/b")])
        assert graph.nodes(scope=RetrievalScope(include=("/a",))) == []
        assert graph.nodes(scope=RetrievalScope(include=("/b",))) == [
            "checkout_service",
            "payments_service",
        ]
