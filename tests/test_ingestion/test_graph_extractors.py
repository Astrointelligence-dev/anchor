"""Deterministic graph extraction: wikilinks, structure, indexer."""

from __future__ import annotations

import pytest

from anchor.graph import KnowledgeGraph
from anchor.ingestion.graph_extractors import (
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
        assert [n.id for n in found.nodes] == ["checkout-service", "service", "payments"]
        note = found.nodes[0]
        assert note.aliases == ("Checkout", "cart")
        assert note.metadata == {"kind": "note", "doc_id": "doc-1"}
        assert [(e.source, e.relation, e.target) for e in found.edges] == [
            ("checkout-service", "tagged", "service"),
            ("checkout-service", "tagged", "payments"),
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
            "payments-service",
            "fraud-scoring",
            "payments-service",
        ]
        assert found.nodes[1].aliases == ("fraud",)
        edges = [(e.source, e.relation, e.target) for e in found.edges]
        assert edges == [
            ("checkout-service", "links_to", "payments-service"),
            ("checkout-service", "links_to", "fraud-scoring"),
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
        assert (stats.items, stats.nodes, stats.edges) == (3, 6, 3)
        assert graph.path("checkout-service", "ledger-service") == [
            "checkout-service",
            "payments-service",
            "bruno-costa",
            "ledger-service",
        ]
        hop = graph.explain("checkout-service", "payments-service")[0]
        assert hop.evidence == ("c1",)
        assert hop.provenance == "extracted"
        # the target note is a mention of the chunk that links to it
        assert graph.items("payments-service") == ["c1", "p1"]
        assert graph.items("bruno-costa", scope=RetrievalScope(exclude=("/people",))) == ["p1"]

    def test_reindex_is_idempotent(self) -> None:
        graph = KnowledgeGraph()
        indexer = GraphIndexer(graph)
        item = _item("see [[payments-service]]")
        indexer.index([item])
        version = graph.store.version
        indexer.index([item])
        assert graph.store.version > version  # writes happened...
        assert len(graph.edges("checkout-service")) == 1  # ...but nothing duplicated
        assert graph.explain("checkout-service", "payments-service")[0].evidence == ("c1",)
        assert graph.items("payments-service") == ["c1"]

    def test_unlink_item_is_the_removal_path(self) -> None:
        graph = KnowledgeGraph()
        GraphIndexer(graph).index([_item("see [[payments-service]]")])
        assert graph.unlink_item("c1") == 1
        assert graph.edges("checkout-service") == []
        assert graph.nodes(scope=RetrievalScope()) == []  # no evidence left anywhere

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
