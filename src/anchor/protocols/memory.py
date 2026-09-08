"""Protocols for memory extensibility.

All memory extension points are defined as ``@runtime_checkable`` Protocols
(PEP 544).  Any object with the matching method signatures satisfies the
protocol -- no inheritance required.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from anchor.models.context import ContextItem
    from anchor.models.memory import ConversationTurn, MemoryEntry


class MemoryOperation(StrEnum):
    """Actions that a MemoryConsolidator can prescribe."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


@runtime_checkable
class CompactionStrategy(Protocol):
    """Compacts evicted conversation turns into a summary."""

    def compact(self, turns: list[ConversationTurn]) -> str:
        """Compact a sequence of conversation turns into a summary string.

        Called by ``SummaryBufferMemory`` when evicted turns need to be
        summarised before being discarded from the sliding window.

        Parameters:
            turns: The conversation turns to compact.  Ordered
                chronologically (oldest first).

        Returns:
            A concise summary string that preserves the essential
            information from the turns.
        """
        ...


@runtime_checkable
class AsyncCompactionStrategy(Protocol):
    """Async variant of CompactionStrategy."""

    async def compact(self, turns: list[ConversationTurn]) -> str:
        """Asynchronously compact conversation turns into a summary string.

        Async counterpart of ``CompactionStrategy.compact``, intended for
        implementations that call an external LLM or network service.

        Parameters:
            turns: The conversation turns to compact.  Ordered
                chronologically (oldest first).

        Returns:
            A concise summary string that preserves the essential
            information from the turns.
        """
        ...


@runtime_checkable
class MemoryExtractor(Protocol):
    """Extracts structured memories from conversation turns."""

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        """Extract structured memory entries from conversation turns.

        Implementations analyse the provided turns and produce zero or
        more ``MemoryEntry`` objects representing facts, preferences, or
        other persistent knowledge worth remembering.

        Parameters:
            turns: The conversation turns to analyse.  Ordered
                chronologically (oldest first).

        Returns:
            A list of newly created ``MemoryEntry`` objects.  Returns an
            empty list when no extractable memories are found.
        """
        ...


@runtime_checkable
class AsyncMemoryExtractor(Protocol):
    """Async variant of MemoryExtractor."""

    async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        """Asynchronously extract structured memory entries from conversation turns.

        Async counterpart of ``MemoryExtractor.extract``, intended for
        implementations that call an external LLM or network service.

        Parameters:
            turns: The conversation turns to analyse.  Ordered
                chronologically (oldest first).

        Returns:
            A list of newly created ``MemoryEntry`` objects.  Returns an
            empty list when no extractable memories are found.
        """
        ...


@runtime_checkable
class MemoryConsolidator(Protocol):
    """Decides how to merge new memories with existing ones.

    Returns a list of ``(MemoryOperation, entry | None)`` tuples describing
    the actions to take.
    """

    def consolidate(
        self, new_entries: list[MemoryEntry], existing: list[MemoryEntry]
    ) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
        """Consolidate new memory entries against existing ones.

        For each new entry, the implementation decides whether to add it,
        update an existing entry, invalidate an existing entry, or take no
        action.

        Parameters:
            new_entries: Newly extracted ``MemoryEntry`` objects to be
                consolidated.
            existing: Candidate entries already in the store — all of
                them, or a similarity-selected subset.

        Returns:
            A list of ``(MemoryOperation, entry)`` tuples. The ``entry``
            slot names the target of the operation:

            - ``ADD`` — the new entry to store.
            - ``UPDATE`` — the existing entry rewritten **under its own
              id** (``store.add`` overwrites by id; a fresh id is a
              silent ADD).
            - ``DELETE`` — the existing entry to invalidate. The applier
              soft-deletes it (``MemoryEntry.invalidate``: hidden from
              ``search``/``list_all``, kept as history until the garbage
              collector's retention elapses). ``(DELETE, None)`` has no
              target and is a no-op.
            - ``NONE`` — ``None``; nothing to do.
        """
        ...


@runtime_checkable
class EvictionPolicy(Protocol):
    """Decides which turns to evict when memory is full.

    Returns indices of turns to evict from the provided list.
    """

    def select_for_eviction(
        self, turns: list[ConversationTurn], tokens_to_free: int
    ) -> list[int]:
        """Select conversation turns for eviction.

        Called when the memory's token budget is exceeded and turns must
        be removed to make room.

        Parameters:
            turns: All conversation turns currently held in memory.
            tokens_to_free: The minimum number of tokens that must be
                freed to accommodate the new turn.

        Returns:
            A list of zero-based indices into *turns* identifying the
            turns to evict.  The indices should free at least
            ``tokens_to_free`` tokens in total.
        """
        ...


@runtime_checkable
class MemoryDecay(Protocol):
    """Computes a retention score for a memory entry.

    Returns a float from 0.0 (forget) to 1.0 (perfect retention).
    """

    def compute_retention(self, entry: MemoryEntry) -> float:
        """Compute the retention score for a memory entry.

        The score reflects how well the memory is retained over time,
        typically decaying as the entry ages.

        Parameters:
            entry: The memory entry to score.

        Returns:
            A float from 0.0 (completely forgotten) to 1.0 (perfectly
            retained).  Used by ``MemoryGarbageCollector`` to prune
            entries below a retention threshold.
        """
        ...


@runtime_checkable
class MemoryQueryEnricher(Protocol):
    """Enriches a query with memory context before retrieval."""

    def enrich(self, query: str, memory_items: list[MemoryEntry]) -> str:
        """Enrich a query string using relevant memory entries.

        Called before retrieval steps to augment the raw user query with
        context from persistent memory.

        Parameters:
            query: The original user query string.
            memory_items: Relevant ``MemoryEntry`` objects retrieved from
                the persistent store.

        Returns:
            An enriched query string that incorporates memory context.
            If no enrichment is possible, the original query should be
            returned unchanged.
        """
        ...


QueryEnricher = MemoryQueryEnricher  # Deprecated: use MemoryQueryEnricher


@runtime_checkable
class RecencyScorer(Protocol):
    """Computes recency scores for memory items.

    ``index=0`` is the oldest item, ``index=total-1`` is the newest.
    Returns a float from 0.0 to 1.0.
    """

    def score(self, index: int, total: int) -> float:
        """Compute a recency score for an item at the given position.

        Parameters:
            index: Zero-based position of the item in the chronologically
                sorted list (0 = oldest, ``total - 1`` = newest).
            total: Total number of items in the list.

        Returns:
            A float from 0.0 (oldest / least recent) to 1.0 (newest /
            most recent).  The mapping function (linear, exponential,
            etc.) is implementation-defined.
        """
        ...


@runtime_checkable
class ConversationMemory(Protocol):
    """Protocol for conversation memory implementations.

    Both ``SlidingWindowMemory`` and ``SummaryBufferMemory`` satisfy this
    protocol.  Use this type instead of concrete classes when only read-side
    access to conversation memory is needed.
    """

    @property
    def turns(self) -> list[ConversationTurn]: ...

    @property
    def total_tokens(self) -> int: ...

    def to_context_items(self, priority: int = 7) -> list[ContextItem]: ...

    def clear(self) -> None: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for objects that provide context items from memory.

    ``ContextPipeline.with_memory()`` accepts any object satisfying this
    protocol.  ``MemoryManager`` is the canonical implementation.
    """

    def get_context_items(self, priority: int = 7) -> list[ContextItem]: ...
