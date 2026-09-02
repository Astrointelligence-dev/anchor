"""Tests for anchor.pipeline.memory_steps."""

from __future__ import annotations

from typing import Any

from anchor.graph import KnowledgeGraph
from anchor.memory.extractor import CallbackExtractor
from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import ConversationTurn, MemoryEntry
from anchor.models.query import QueryBundle
from anchor.pipeline.memory_steps import (
    auto_promotion_step,
    create_eviction_promoter,
    graph_retrieval_step,
)
from anchor.pipeline.pipeline import ContextPipeline
from anchor.pipeline.step import retriever_step
from anchor.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeRetriever, FakeTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_and_store() -> tuple[KnowledgeGraph, InMemoryEntryStore]:
    """Build a small graph with two entities and linked memory entries."""
    graph = KnowledgeGraph()
    graph.add_node("alice", metadata={"type": "person"})
    graph.add_node("project-x", metadata={"type": "project"})
    graph.add_edge("alice", "works_on", "project-x")

    store = InMemoryEntryStore()
    entry_a = MemoryEntry(id="mem-a", content="Alice is an engineer")
    entry_b = MemoryEntry(id="mem-b", content="Project X uses Python")
    store.add(entry_a)
    store.add(entry_b)

    graph.link_item("alice", "mem-a")
    graph.link_item("project-x", "mem-b")

    return graph, store


def _simple_entity_extractor(query: str) -> list[str]:
    """Trivial entity extractor that maps known keywords to entity IDs."""
    entities: list[str] = []
    lower = query.lower()
    if "alice" in lower:
        entities.append("alice")
    if "project" in lower:
        entities.append("project-x")
    return entities


def _empty_entity_extractor(query: str) -> list[str]:
    return []


def _make_query(text: str = "Tell me about Alice") -> QueryBundle:
    return QueryBundle(query_str=text)


def _make_memory_items(n: int = 2) -> list[ContextItem]:
    """Create MEMORY-source ContextItems that look like conversation turns."""
    return [
        ContextItem(
            id=f"mem-item-{i}",
            content=f"User said something interesting {i}",
            source=SourceType.MEMORY,
            score=0.8,
            priority=7,
            token_count=5,
            metadata={"role": "user"},
        )
        for i in range(n)
    ]


def _simple_extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Extract one memory per turn."""
    return [{"content": f"Extracted: {t.content}"} for t in turns]


# ===========================================================================
# TestGraphRetrievalStepBasic
# ===========================================================================


class TestGraphRetrievalStepBasic:
    """graph_retrieval_step: basic entity extraction -> graph traversal -> memory retrieval."""

    def test_retrieves_linked_memories(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        result = step.execute([], _make_query("Tell me about Alice"))
        assert len(result) > 0
        contents = [item.content for item in result]
        assert "Alice is an engineer" in contents

    def test_items_have_memory_source_and_priority_6(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        result = step.execute([], _make_query("Tell me about Alice"))
        for item in result:
            assert item.source == SourceType.MEMORY
            assert item.priority == 6

    def test_metadata_includes_graph_retrieval_source(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        result = step.execute([], _make_query("Tell me about Alice"))
        for item in result:
            assert item.metadata["source"] == "graph_retrieval"
            assert "memory_id" in item.metadata

    def test_appends_to_existing_items(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        existing = [
            ContextItem(
                content="existing item",
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=5,
            )
        ]
        result = step.execute(existing, _make_query("Tell me about Alice"))
        assert result[0].content == "existing item"
        assert len(result) > 1

    def test_traverses_graph_relationships(self) -> None:
        """Querying 'alice' should also find project-x memories via relationship."""
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
            max_depth=2,
        )
        result = step.execute([], _make_query("Tell me about Alice"))
        contents = [item.content for item in result]
        # Alice's direct memory
        assert "Alice is an engineer" in contents
        # Project-x memory found via alice -> works_on -> project-x
        assert "Project X uses Python" in contents


# ===========================================================================
# TestGraphRetrievalStepNoEntities
# ===========================================================================


class TestGraphRetrievalStepNoEntities:
    """graph_retrieval_step: no entities found -> returns existing items unchanged."""

    def test_no_entities_returns_items_unchanged(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_empty_entity_extractor,
        )
        existing = [
            ContextItem(
                content="keep me",
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=5,
            )
        ]
        result = step.execute(existing, _make_query("no entities here"))
        assert result == existing

    def test_empty_query_returns_empty_items(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_empty_entity_extractor,
        )
        result = step.execute([], _make_query(""))
        assert result == []


# ===========================================================================
# TestGraphRetrievalStepEntityNotInGraph
# ===========================================================================


class TestGraphRetrievalStepEntityNotInGraph:
    """graph_retrieval_step: entity not in graph -> skipped gracefully."""

    def test_unknown_entity_returns_items_unchanged(self) -> None:
        graph, store = _make_graph_and_store()

        def unknown_extractor(query: str) -> list[str]:
            return ["unknown-entity"]

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=unknown_extractor,
        )
        result = step.execute([], _make_query("test"))
        assert result == []

    def test_mix_of_known_and_unknown_entities(self) -> None:
        graph, store = _make_graph_and_store()

        def mixed_extractor(query: str) -> list[str]:
            return ["alice", "unknown-entity"]

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=mixed_extractor,
        )
        result = step.execute([], _make_query("test"))
        assert len(result) > 0
        contents = [item.content for item in result]
        assert "Alice is an engineer" in contents


# ===========================================================================
# TestGraphRetrievalStepMaxDepth
# ===========================================================================


class TestGraphRetrievalStepMaxDepth:
    """graph_retrieval_step: max_depth controls traversal depth."""

    def test_depth_1_only_direct_neighbors(self) -> None:
        graph = KnowledgeGraph()
        graph.add_edge("alice", "knows", "bob")
        graph.add_edge("bob", "knows", "carol")
        graph.add_node("carol")

        store = InMemoryEntryStore()
        entry_bob = MemoryEntry(id="mem-bob", content="Bob info")
        entry_carol = MemoryEntry(id="mem-carol", content="Carol info")
        store.add(entry_bob)
        store.add(entry_carol)

        graph.link_item("bob", "mem-bob")
        graph.link_item("carol", "mem-carol")

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=lambda q: ["alice"],
            max_depth=1,
        )
        result = step.execute([], _make_query("alice"))
        contents = [item.content for item in result]
        assert "Bob info" in contents
        assert "Carol info" not in contents

    def test_depth_2_includes_hop_2(self) -> None:
        graph = KnowledgeGraph()
        graph.add_edge("alice", "knows", "bob")
        graph.add_edge("bob", "knows", "carol")
        graph.add_node("carol")

        store = InMemoryEntryStore()
        entry_carol = MemoryEntry(id="mem-carol", content="Carol info")
        store.add(entry_carol)
        graph.link_item("carol", "mem-carol")

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=lambda q: ["alice"],
            max_depth=2,
        )
        result = step.execute([], _make_query("alice"))
        contents = [item.content for item in result]
        assert "Carol info" in contents


# ===========================================================================
# TestGraphRetrievalStepMaxItems
# ===========================================================================


class TestGraphRetrievalStepMaxItems:
    """graph_retrieval_step: max_items limits results."""

    def test_max_items_caps_output(self) -> None:
        graph = KnowledgeGraph()
        store = InMemoryEntryStore()

        graph.add_node("hub")
        for i in range(10):
            entity_id = f"node-{i}"
            mem_id = f"mem-{i}"
            graph.add_node(entity_id)
            graph.add_edge("hub", "links_to", entity_id)
            entry = MemoryEntry(id=mem_id, content=f"Memory {i}")
            store.add(entry)
            graph.link_item(entity_id, mem_id)

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=lambda q: ["hub"],
            max_items=3,
        )
        result = step.execute([], _make_query("hub"))
        assert len(result) == 3

    def test_default_max_items_is_5(self) -> None:
        graph = KnowledgeGraph()
        store = InMemoryEntryStore()

        graph.add_node("hub")
        for i in range(10):
            entity_id = f"node-{i}"
            mem_id = f"mem-{i}"
            graph.add_node(entity_id)
            graph.add_edge("hub", "links_to", entity_id)
            entry = MemoryEntry(id=mem_id, content=f"Memory {i}")
            store.add(entry)
            graph.link_item(entity_id, mem_id)

        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=lambda q: ["hub"],
        )
        result = step.execute([], _make_query("hub"))
        assert len(result) == 5


# ===========================================================================
# TestGraphRetrievalStepOnError
# ===========================================================================


class TestGraphRetrievalStepOnError:
    """graph_retrieval_step: on_error='skip' handles errors gracefully."""

    def test_on_error_skip_is_default(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        assert step.on_error == "skip"

    def test_on_error_raise_can_be_set(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
            on_error="raise",
        )
        assert step.on_error == "raise"

    def test_step_name_defaults_to_graph_retrieval(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
        )
        assert step.name == "graph_retrieval"

    def test_custom_step_name(self) -> None:
        graph, store = _make_graph_and_store()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=_simple_entity_extractor,
            name="custom_graph",
        )
        assert step.name == "custom_graph"


# ===========================================================================
# TestAutoPromotionStepBasic
# ===========================================================================


class TestAutoPromotionStepBasic:
    """auto_promotion_step: extracts memories and stores them."""

    def test_extracts_and_stores_memories(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store)

        items = _make_memory_items(2)
        step.execute(items, _make_query())

        stored = store.list_all()
        assert len(stored) == 2
        contents = [e.content for e in stored]
        assert any("Extracted:" in c for c in contents)

    def test_stores_entries_from_extractor_output(self) -> None:
        store = InMemoryEntryStore()

        def specific_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "User likes Python", "tags": ["preference"]}]

        extractor = CallbackExtractor(extract_fn=specific_fn)
        step = auto_promotion_step(extractor=extractor, store=store)

        step.execute(_make_memory_items(1), _make_query())

        stored = store.list_all()
        assert len(stored) == 1
        assert stored[0].content == "User likes Python"
        assert "preference" in stored[0].tags


# ===========================================================================
# TestAutoPromotionStepConsolidation
# ===========================================================================


class TestAutoPromotionStepConsolidation:
    """auto_promotion_step: with consolidator deduplicates."""

    def test_consolidator_prevents_duplicates(self) -> None:
        store = InMemoryEntryStore()
        existing = MemoryEntry(id="existing-1", content="User likes Python")
        store.add(existing)

        def extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "User likes Python"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)

        # Create a mock consolidator that returns "none" for duplicates
        class DedupConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                results: list[tuple[str, MemoryEntry | None]] = []
                existing_contents = {e.content for e in existing}
                for entry in new_entries:
                    if entry.content in existing_contents:
                        results.append(("none", None))
                    else:
                        results.append(("add", entry))
                return results

        step = auto_promotion_step(
            extractor=extractor,
            store=store,
            consolidator=DedupConsolidator(),  # type: ignore[arg-type]
        )
        step.execute(_make_memory_items(1), _make_query())

        # Should still only have the original entry
        stored = store.list_all()
        assert len(stored) == 1
        assert stored[0].id == "existing-1"

    def test_consolidator_updates_existing(self) -> None:
        store = InMemoryEntryStore()
        existing = MemoryEntry(id="existing-1", content="User likes Python")
        store.add(existing)

        def extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "User likes Python and JavaScript"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)

        class UpdateConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                # Always "update" with the new entry
                return [("update", entry) for entry in new_entries]

        step = auto_promotion_step(
            extractor=extractor,
            store=store,
            consolidator=UpdateConsolidator(),  # type: ignore[arg-type]
        )
        step.execute(_make_memory_items(1), _make_query())

        stored = store.list_all()
        # Should have 2: original + updated
        assert len(stored) == 2


# ===========================================================================
# TestAutoPromotionStepPassthrough
# ===========================================================================


class TestAutoPromotionStepPassthrough:
    """auto_promotion_step: returns original items unchanged."""

    def test_returns_original_items_unchanged(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store)

        items = _make_memory_items(3)
        result = step.execute(items, _make_query())

        assert result is items
        assert len(result) == 3

    def test_non_memory_items_pass_through(self) -> None:
        store = InMemoryEntryStore()

        def extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "extracted"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store)

        retrieval_items = [
            ContextItem(
                content="retrieval content",
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=5,
            )
        ]
        result = step.execute(retrieval_items, _make_query())

        assert result is retrieval_items
        # No memory items -> no extraction
        assert len(store.list_all()) == 0

    def test_empty_items_returns_empty(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store)

        result = step.execute([], _make_query())
        assert result == []
        assert len(store.list_all()) == 0


# ===========================================================================
# TestAutoPromotionStepErrorHandling
# ===========================================================================


class TestAutoPromotionStepErrorHandling:
    """auto_promotion_step: on_error='skip' handles extraction errors."""

    def test_on_error_skip_is_default(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store)
        assert step.on_error == "skip"

    def test_on_error_raise_can_be_set(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        step = auto_promotion_step(extractor=extractor, store=store, on_error="raise")
        assert step.on_error == "raise"


# ===========================================================================
# TestCreateEvictionPromoter
# ===========================================================================


class TestCreateEvictionPromoter:
    """create_eviction_promoter: callback extracts and stores on eviction."""

    def test_callback_extracts_and_stores(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        turns = [
            ConversationTurn(
                role="user",  # type: ignore[arg-type]
                content="I prefer dark mode",
                token_count=4,
            ),
        ]
        promoter(turns)

        stored = store.list_all()
        assert len(stored) == 1
        assert "Extracted:" in stored[0].content

    def test_callback_stores_multiple_entries(self) -> None:
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        turns = [
            ConversationTurn(
                role="user",  # type: ignore[arg-type]
                content="Turn A",
                token_count=2,
            ),
            ConversationTurn(
                role="assistant",  # type: ignore[arg-type]
                content="Turn B",
                token_count=2,
            ),
        ]
        promoter(turns)

        stored = store.list_all()
        assert len(stored) == 2


# ===========================================================================
# TestCreateEvictionPromoterConsolidation
# ===========================================================================


class TestCreateEvictionPromoterConsolidation:
    """create_eviction_promoter: with consolidator performs dedup."""

    def test_consolidator_deduplicates(self) -> None:
        store = InMemoryEntryStore()
        existing = MemoryEntry(id="old-1", content="Extracted: I prefer dark mode")
        store.add(existing)

        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)

        class SkipDupConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                existing_contents = {e.content for e in existing}
                results: list[tuple[str, MemoryEntry | None]] = []
                for entry in new_entries:
                    if entry.content in existing_contents:
                        results.append(("none", None))
                    else:
                        results.append(("add", entry))
                return results

        promoter = create_eviction_promoter(
            extractor=extractor,
            store=store,
            consolidator=SkipDupConsolidator(),  # type: ignore[arg-type]
        )

        turns = [
            ConversationTurn(
                role="user",  # type: ignore[arg-type]
                content="I prefer dark mode",
                token_count=4,
            ),
        ]
        promoter(turns)

        # Should still only have the original entry (duplicate skipped)
        stored = store.list_all()
        assert len(stored) == 1
        assert stored[0].id == "old-1"


# ===========================================================================
# TestCreateEvictionPromoterErrorHandling
# ===========================================================================


class TestCreateEvictionPromoterErrorHandling:
    """create_eviction_promoter: handles extractor errors gracefully."""

    def test_extractor_error_does_not_propagate(self) -> None:
        store = InMemoryEntryStore()

        def failing_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            msg = "extractor exploded"
            raise RuntimeError(msg)

        extractor = CallbackExtractor(extract_fn=failing_fn)
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        turns = [
            ConversationTurn(
                role="user",  # type: ignore[arg-type]
                content="test",
                token_count=1,
            ),
        ]
        # Should not raise
        promoter(turns)
        assert len(store.list_all()) == 0

    def test_store_error_does_not_propagate(self) -> None:
        class FailingStore:
            def add(self, entry: MemoryEntry) -> None:
                msg = "store exploded"
                raise RuntimeError(msg)

            def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
                return []

            def list_all(self) -> list[MemoryEntry]:
                return []

            def delete(self, entry_id: str) -> bool:
                return False

            def clear(self) -> None:
                pass

        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)
        promoter = create_eviction_promoter(
            extractor=extractor,
            store=FailingStore(),  # type: ignore[arg-type]
        )

        turns = [
            ConversationTurn(
                role="user",  # type: ignore[arg-type]
                content="test",
                token_count=1,
            ),
        ]
        # Should not raise
        promoter(turns)


# ===========================================================================
# TestIntegrationGraphRetrievalWithPipeline
# ===========================================================================


class TestIntegrationGraphRetrievalWithPipeline:
    """Integration: full pipeline with graph_retrieval_step + memory."""

    def test_graph_step_in_pipeline(self) -> None:
        graph, store = _make_graph_and_store()
        tokenizer = FakeTokenizer()

        pipeline = ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
        pipeline.add_system_prompt("You are helpful.")
        pipeline.add_step(
            graph_retrieval_step(
                graph=graph,
                store=store,
                entity_extractor=_simple_entity_extractor,
            )
        )

        result = pipeline.build(QueryBundle(query_str="Tell me about Alice"))
        contents = [item.content for item in result.window.items]
        assert "You are helpful." in contents
        assert "Alice is an engineer" in contents

    def test_graph_step_with_regular_retriever(self) -> None:
        graph, store = _make_graph_and_store()
        tokenizer = FakeTokenizer()

        retrieval_items = [
            ContextItem(
                id="r1",
                content="Retrieved document",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=tokenizer.count_tokens("Retrieved document"),
            ),
        ]

        pipeline = ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
        pipeline.add_step(retriever_step("search", FakeRetriever(retrieval_items)))
        pipeline.add_step(
            graph_retrieval_step(
                graph=graph,
                store=store,
                entity_extractor=_simple_entity_extractor,
            )
        )

        result = pipeline.build(QueryBundle(query_str="Tell me about Alice"))
        sources = [item.source for item in result.window.items]
        assert SourceType.RETRIEVAL in sources
        assert SourceType.MEMORY in sources

    def test_auto_promotion_step_in_pipeline(self) -> None:
        """auto_promotion_step stores memories as side-effect during pipeline build."""
        tokenizer = FakeTokenizer()
        store = InMemoryEntryStore()
        extractor = CallbackExtractor(extract_fn=_simple_extract_fn)

        from anchor.memory.manager import MemoryManager

        memory = MemoryManager(conversation_tokens=2000, tokenizer=tokenizer)
        memory.add_user_message("I prefer dark mode")
        memory.add_assistant_message("Noted, dark mode it is!")

        pipeline = ContextPipeline(max_tokens=8192, tokenizer=tokenizer)
        pipeline.with_memory(memory)
        pipeline.add_step(auto_promotion_step(extractor=extractor, store=store))

        result = pipeline.build(QueryBundle(query_str="test"))

        # Conversation items should be in the result
        assert any(item.source == SourceType.CONVERSATION for item in result.window.items)

        # Side-effect: memories should be extracted and stored
        stored = store.list_all()
        assert len(stored) > 0


# ===========================================================================
# TestGraphRetrievalStepScope
# ===========================================================================


class TestGraphRetrievalStepScope:
    """graph_retrieval_step: memory reached through the graph honors the published scope."""

    def _graph_with_secret(self) -> tuple[KnowledgeGraph, InMemoryEntryStore]:
        graph = KnowledgeGraph()
        graph.add_edge("alice", "knows", "villain")
        store = InMemoryEntryStore()
        store.add(MemoryEntry(id="mem-alice", content="Alice is an engineer"))
        store.add(MemoryEntry(id="mem-villain", content="The villain's secret plan"))
        graph.link_item("alice", "mem-alice", "/public")
        graph.link_item("villain", "mem-villain", "/secret")
        return graph, store

    def test_static_scope_hides_secret_memory(self) -> None:
        from anchor.models.scope import RetrievalScope

        graph, store = self._graph_with_secret()
        step = graph_retrieval_step(
            graph=graph,
            store=store,
            entity_extractor=lambda q: ["alice"],
            scope=RetrievalScope(exclude=("/secret",)),
        )
        contents = [item.content for item in step.execute([], _make_query("alice"))]
        assert contents == ["Alice is an engineer"]

    def test_published_scope_narrows_the_walk(self) -> None:
        from anchor.models.scope import ACTIVE_SCOPE, RetrievalScope

        graph, store = self._graph_with_secret()
        step = graph_retrieval_step(graph=graph, store=store, entity_extractor=lambda q: ["alice"])
        unscoped = [item.content for item in step.execute([], _make_query("alice"))]
        assert "The villain's secret plan" in unscoped

        token = ACTIVE_SCOPE.set(RetrievalScope(exclude=("/secret",)))
        try:
            scoped = [item.content for item in step.execute([], _make_query("alice"))]
        finally:
            ACTIVE_SCOPE.reset(token)
        assert scoped == ["Alice is an engineer"]
