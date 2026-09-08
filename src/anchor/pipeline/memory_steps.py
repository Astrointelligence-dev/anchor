"""Pipeline step factories for memory-related operations.

Provides step factories that integrate graph-based entity lookup and
automatic memory promotion into the context assembly pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import RetrievalScope, effective_scope
from anchor.pipeline.step import PipelineStep
from anchor.protocols.memory import MemoryOperation

if TYPE_CHECKING:
    from anchor.graph.knowledge_graph import KnowledgeGraph
    from anchor.models.memory import ConversationTurn, MemoryEntry
    from anchor.models.query import QueryBundle
    from anchor.protocols.memory import MemoryConsolidator, MemoryExtractor
    from anchor.protocols.storage import MemoryEntryStore

logger = logging.getLogger(__name__)


def _store_with_consolidation(
    entries: list[MemoryEntry],
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None,
) -> None:
    """Persist entries, optionally consolidating via a consolidator.

    With a *consolidator*, each ``(operation, entry)`` it returns is applied
    to the store: ``ADD`` and ``UPDATE`` write the entry, ``DELETE``
    invalidates it (``MemoryEntry.invalidate`` — a soft delete hidden from
    ``search``/``list_all`` and kept as history until the garbage
    collector's retention elapses); ``NONE`` and a target-less ``DELETE``
    do nothing. Without a consolidator every entry is added directly.
    """
    if consolidator is None:
        for entry in entries:
            store.add(entry)
        return
    for action, target in consolidator.consolidate(entries, store.list_all()):
        if target is None:
            continue
        if action in (MemoryOperation.ADD, MemoryOperation.UPDATE):
            store.add(target)
        elif action == MemoryOperation.DELETE:
            store.add(target if target.is_expired else target.invalidate())


def graph_retrieval_step(
    graph: KnowledgeGraph,
    store: MemoryEntryStore,
    entity_extractor: Callable[[str], list[str]],
    max_depth: int = 2,
    max_items: int = 5,
    name: str = "graph_retrieval",
    on_error: Literal["raise", "skip"] = "skip",
    *,
    scope: RetrievalScope | None = None,
) -> PipelineStep:
    """Create a pipeline step that retrieves memory entries linked to graph entities.

    Flow:
        1. Extract entity names from the query using *entity_extractor*.
        2. For each entity, walk the graph up to *max_depth* hops.
        3. Collect the item ids evidencing those nodes — for memory, the
           item id **is** ``MemoryEntry.id`` (link entries with
           ``graph.link_item(entity, entry.id)``).
        4. Fetch the corresponding ``MemoryEntry`` objects from the *store*.
        5. Convert to ``ContextItem`` objects with ``source_type=MEMORY``,
           ``priority=6``.

    The walk honors the scope published by the running agent turn
    intersected with the static *scope* (the ``retriever_step`` doctrine:
    pipeline retrieval can only narrow).

    Parameters:
        graph: The ``KnowledgeGraph`` to walk.
        store: A ``MemoryEntryStore`` implementation that holds persistent
            ``MemoryEntry`` objects.
        entity_extractor: User-provided callable that maps a query string
            to a list of entity names.
        max_depth: Maximum traversal depth (default 2).
        max_items: Maximum number of ``ContextItem`` objects to return.
        name: Step name for diagnostics.
        on_error: Error policy -- ``"skip"`` (default) or ``"raise"``.
        scope: Static namespace scope for the walk.

    Returns:
        A ``PipelineStep`` suitable for ``pipeline.add_step()``.
    """

    def _retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        entity_ids = entity_extractor(query.query_str)
        if not entity_ids:
            return items

        active = effective_scope(scope)
        # Collect memory IDs from all extracted entities and their neighbors
        seen_memory_ids: set[str] = set()
        ordered_memory_ids: list[str] = []
        for entity_id in entity_ids:
            related_ids = graph.related_items(entity_id, max_depth=max_depth, scope=active)
            for mid in related_ids:
                if mid not in seen_memory_ids:
                    seen_memory_ids.add(mid)
                    ordered_memory_ids.append(mid)

        if not ordered_memory_ids:
            return items

        # Fetch entries from the store and convert to ContextItems
        all_entries = store.list_all()
        entry_map: dict[str, MemoryEntry] = {e.id: e for e in all_entries}

        new_items: list[ContextItem] = []
        for mid in ordered_memory_ids:
            if len(new_items) >= max_items:
                break
            entry = entry_map.get(mid)
            if entry is None:
                continue
            new_items.append(
                ContextItem(
                    id=entry.id,  # the item id IS the memory id (one currency)
                    content=entry.content,
                    source=SourceType.MEMORY,
                    score=entry.relevance_score,
                    priority=6,
                    metadata={
                        "memory_id": entry.id,
                        "memory_type": str(entry.memory_type),
                        "tags": list(entry.tags),
                        "source": "graph_retrieval",
                    },
                )
            )

        return items + new_items

    return PipelineStep(
        name=name,
        fn=_retrieve,
        on_error=on_error,
    )


def auto_promotion_step(
    extractor: MemoryExtractor,
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None = None,
    name: str = "auto_promotion",
    on_error: Literal["raise", "skip"] = "skip",
) -> PipelineStep:
    """Create a pipeline step that extracts and stores memories from context.

    This is a post-processor-style step that runs **after** retrieval.  It
    inspects the memory-typed items currently in the pipeline, extracts
    structured ``MemoryEntry`` objects via the *extractor*, and persists
    them in *store*.

    If a *consolidator* is provided it is used to deduplicate against
    entries already present in the store.

    The step returns the original items unchanged -- it is side-effect only.

    Parameters:
        extractor: A ``MemoryExtractor`` implementation.
        store: A ``MemoryEntryStore`` for persistence.
        consolidator: Optional ``MemoryConsolidator`` for deduplication.
        name: Step name for diagnostics.
        on_error: Error policy -- ``"skip"`` (default) or ``"raise"``.

    Returns:
        A ``PipelineStep`` suitable for ``pipeline.add_step()``.
    """

    def _promote(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        # Gather memory-source items and convert to ConversationTurn-like objects
        from anchor.models.memory import ConversationTurn

        memory_items = [
            item
            for item in items
            if item.source in (SourceType.MEMORY, SourceType.CONVERSATION)
        ]
        if not memory_items:
            return items

        # Build ConversationTurn objects from memory items
        turns: list[ConversationTurn] = []
        for item in memory_items:
            role = item.metadata.get("role", "user")
            turns.append(
                ConversationTurn(
                    role=role,
                    content=item.content,
                    token_count=item.token_count,
                    timestamp=item.created_at,
                )
            )

        new_entries = extractor.extract(turns)
        if not new_entries:
            return items

        _store_with_consolidation(new_entries, store, consolidator)

        return items

    return PipelineStep(
        name=name,
        fn=_promote,
        on_error=on_error,
    )


def create_eviction_promoter(
    extractor: MemoryExtractor,
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None = None,
) -> Callable[[list[ConversationTurn]], None]:
    """Create an ``on_evict`` callback that promotes evicted turns to long-term memory.

    The returned callback is designed for use with
    ``SlidingWindowMemory(on_evict=...)``.  When turns are evicted from
    the sliding window the callback:

    1. Calls ``extractor.extract(turns)`` to produce ``MemoryEntry`` objects.
    2. If *consolidator* is provided, consolidates against existing entries.
    3. Stores new/updated entries in *store*.

    Errors are logged but **never** propagated -- a failing promoter must
    not crash the memory pipeline.

    Usage::

        promoter = create_eviction_promoter(extractor, store, consolidator)
        memory = SlidingWindowMemory(max_tokens=4096, on_evict=promoter)

    Parameters:
        extractor: A ``MemoryExtractor`` implementation.
        store: A ``MemoryEntryStore`` for persistence.
        consolidator: Optional ``MemoryConsolidator`` for deduplication.

    Returns:
        A callable matching the ``on_evict`` signature.
    """

    def _on_evict(turns: list[ConversationTurn]) -> None:
        try:
            new_entries = extractor.extract(turns)
            if not new_entries:
                return

            _store_with_consolidation(new_entries, store, consolidator)
        except Exception:
            logger.exception(
                "eviction promoter failed — ignoring to protect pipeline"
            )

    return _on_evict
