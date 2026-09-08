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

    def test_remember_keeps_the_graph_in_step(self) -> None:
        from anchor.models.memory import MemoryEntry
        from anchor.protocols.memory import MemoryOperation

        class LastTurnExtractor:
            def extract(self, turns):
                return [MemoryEntry(content=turns[-1].content)]

        class Replace:
            def consolidate(self, new_entries, existing):
                gone = [(MemoryOperation.DELETE, e) for e in existing]
                return gone + [(MemoryOperation.ADD, e) for e in new_entries]

        graph = KnowledgeGraph()
        manager = MemoryManager(
            tokenizer=FakeTokenizer(),
            persistent_store=InMemoryEntryStore(),
            graph=GraphIndexer(graph, extractors=[WordPairExtractor()]),
            extractor=LastTurnExtractor(),
            consolidator=Replace(),
        )
        manager.add_user_message("alice bob")
        manager.remember()
        assert graph.neighbors("alice") == ["bob"]

        manager.add_user_message("alice carol")
        manager.remember()
        assert graph.neighbors("alice") == ["carol"]  # the invalidated fact took its evidence
        assert graph.items("bob") == []
        assert [e.content for e in manager.get_all_facts()] == ["alice carol"]

    def test_without_graph_nothing_changes(self) -> None:
        manager = MemoryManager(tokenizer=FakeTokenizer(), persistent_store=InMemoryEntryStore())
        entry = manager.add_fact("alice bob")
        assert manager.update_fact(entry.id, "x y") is not None
        assert manager.delete_fact(entry.id) is True


class TestGraphIndexingEntryStore:
    def test_every_writer_keeps_the_graph_in_step(self) -> None:
        from datetime import UTC, datetime, timedelta

        from anchor.ingestion import GraphIndexingEntryStore
        from anchor.memory.gc import MemoryGarbageCollector
        from anchor.models.memory import MemoryEntry
        from anchor.pipeline.memory_steps import _store_with_consolidation

        graph = KnowledgeGraph()
        inner = InMemoryEntryStore()
        store = GraphIndexingEntryStore(
            inner, GraphIndexer(graph, extractors=[WordPairExtractor()])
        )
        # the consolidation step writes straight to the store
        _store_with_consolidation([MemoryEntry(id="e1", content="alice bob")], store, None)
        assert graph.neighbors("alice") == ["bob"]
        # unchanged content: no re-extraction (version stays)
        version = graph.store.version
        store.add(MemoryEntry(id="e1", content="alice bob"))
        assert graph.store.version == version
        # changed content: old evidence goes, new comes
        store.add(MemoryEntry(id="e1", content="alice carol"))
        assert graph.neighbors("alice") == ["carol"]
        assert graph.items("bob") == []
        # a soft delete (same text, now expired) takes its evidence with it
        past = datetime.now(UTC) - timedelta(days=1)
        store.add(MemoryEntry(id="e2", content="dave erin"))
        assert graph.items("dave") == ["e2"]
        store.add(MemoryEntry(id="e2", content="dave erin", expires_at=past))
        assert graph.items("dave") == []
        assert {e.id for e in store.list_all_unfiltered()} == {"e1", "e2"}  # history kept
        # an entry arriving already expired never evidences the graph
        store.add(MemoryEntry(id="e3", content="fay gus", expires_at=past))
        assert graph.items("fay") == []
        # the garbage collector deletes straight on the store
        MemoryGarbageCollector(store).collect_expired()
        assert {e.id for e in store.list_all_unfiltered()} == {"e1"}
        # clear unlinks everything
        store.clear()
        assert graph.items("alice") == []
        assert store.list_all() == []
        assert store.inner is inner
        assert store.search("x") == []
        assert store.list_all_unfiltered() == []  # forwarded

    def test_manager_wraps_the_store(self) -> None:
        from anchor.ingestion import GraphIndexingEntryStore

        manager, graph = _manager()
        assert isinstance(manager.persistent_store, GraphIndexingEntryStore)
        assert manager.persistent_store.graph is graph
