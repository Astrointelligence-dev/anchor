"""MemoryManager(graph=...): the knowledge graph follows the persistent store."""

from __future__ import annotations

from anchor.graph import KnowledgeGraph
from anchor.ingestion.graph_extractors import Extraction, GraphIndexer
from anchor.memory.manager import MemoryManager
from anchor.models.context import ContextItem
from anchor.models.graph import GraphEdge, GraphNode
from anchor.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer


class WordPairExtractor:
    """Deterministic stand-in for the LLM: first two words become a `knows` edge."""

    def extract(self, item: ContextItem) -> Extraction:
        words = item.content.split()
        if len(words) < 2:
            return Extraction()
        return Extraction(
            nodes=[GraphNode(id=words[0]), GraphNode(id=words[1])],
            edges=[GraphEdge(source=words[0], target=words[1], relation="knows")],
        )


def _manager() -> tuple[MemoryManager, KnowledgeGraph]:
    graph = KnowledgeGraph()
    manager = MemoryManager(
        tokenizer=FakeTokenizer(),
        persistent_store=InMemoryEntryStore(),
        graph=GraphIndexer(graph, extractors=[WordPairExtractor()]),
    )
    return manager, graph


class TestManagerGraph:
    def test_add_update_delete_clear_keep_the_graph_in_step(self) -> None:
        manager, graph = _manager()
        entry = manager.add_fact("alice bob")
        assert graph.items("alice") == [entry.id]
        assert graph.neighbors("alice") == ["bob"]

        manager.update_fact(entry.id, "alice carol")
        assert graph.neighbors("alice") == ["carol"]  # the old edge lost its only evidence
        assert graph.items("bob") == []
        assert graph.items("carol") == [entry.id]

        assert manager.delete_fact(entry.id) is True
        assert graph.neighbors("alice") == []
        assert graph.items("alice") == []

        manager.add_fact("dave erin")
        manager.clear()
        assert graph.neighbors("dave") == []

    def test_duplicate_fact_is_not_reindexed(self) -> None:
        manager, graph = _manager()
        manager.add_fact("alice bob")
        version = graph.store.version
        manager.add_fact("alice bob")  # content-hash dedupe returns the existing entry
        assert graph.store.version == version

    def test_without_graph_nothing_changes(self) -> None:
        manager = MemoryManager(tokenizer=FakeTokenizer(), persistent_store=InMemoryEntryStore())
        entry = manager.add_fact("alice bob")
        assert manager.update_fact(entry.id, "x y") is not None
        assert manager.delete_fact(entry.id) is True
