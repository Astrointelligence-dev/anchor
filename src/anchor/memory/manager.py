"""Memory manager that coordinates conversation memory."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from anchor.exceptions import StorageError
from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import (
    ConversationTurn,
    MemoryEntry,
    MemoryType,
    Role,
    _compute_content_hash,
)
from anchor.protocols.memory import ConversationMemory, MemoryOperation
from anchor.protocols.storage import MemoryEntryStore
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

from .callbacks import MemoryCallback, _fire_memory_callback
from .consolidator import apply_consolidation
from .progressive import ProgressiveSummarizationMemory
from .sliding_window import SlidingWindowMemory
from .summary_buffer import SummaryBufferMemory

if TYPE_CHECKING:
    from anchor.ingestion.graph_extractors import GraphIndexer
    from anchor.protocols.memory import MemoryConsolidator, MemoryExtractor

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates different memory strategies and produces context items.

    Wraps ``SlidingWindowMemory`` or ``SummaryBufferMemory`` for conversation
    history and optionally integrates a persistent ``MemoryEntryStore`` for
    long-term facts.

    When *conversation_memory* is provided it is used directly as the
    conversation backend, and *conversation_tokens*, *tokenizer*, and
    *on_evict* are ignored.  Otherwise a new ``SlidingWindowMemory`` is
    created from those parameters (backwards-compatible default).

    *graph* (opt-in, roadmap #4) keeps a knowledge graph in step with the
    persistent store by wrapping it in a ``GraphIndexingEntryStore``: every
    entry written is indexed (the item id is the entry id), a rewritten
    entry is re-extracted, a deleted or cleared entry takes its evidence
    with it — whoever the writer is (this manager, the pipeline's
    consolidation step, the garbage collector). Pass a ``GraphIndexer``
    whose extractors suit memory — typically ``[LLMGraphExtractor(llm)]``,
    since facts carry no wikilinks. Entries that already exist when the
    graph is attached are not backfilled: run
    ``indexer.index_entries(store.list_all())`` once.

    *extractor* and *consolidator* (opt-in, roadmap #5) turn the
    conversation into long-term facts: ``remember()`` runs the extractor
    over the turns not yet remembered (at most *extract_window* of them —
    the newest) and applies the consolidator's ADD / UPDATE / DELETE / NONE
    to the persistent store — the two-phase update of the mem0 paper, with
    ``LLMExtractor(agent.llm)`` and ``LLMConsolidator(agent.llm)``, or the
    cheap ``SimilarityConsolidator``. An ``Agent`` calls ``after_turn()``
    once per completed turn, which runs ``remember()`` every
    *remember_every* turns (``0`` = only when you call it). *callbacks*
    observe ``on_extraction`` and ``on_consolidation``. Hand
    ``manager.persistent_store`` (not the raw store) to a
    ``MemoryGarbageCollector`` so the graph, when attached, sees its
    deletes too.
    """

    __slots__ = (
        "_callbacks",
        "_consolidator",
        "_conversation",
        "_extract_window",
        "_extractor",
        "_last_remembered",
        "_persistent_store",
        "_remember_every",
        "_tokenizer",
        "_turns_since_remember",
    )

    def __init__(
        self,
        conversation_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
        on_evict: Callable[[list[ConversationTurn]], None] | None = None,
        persistent_store: MemoryEntryStore | None = None,
        conversation_memory: ConversationMemory | None = None,
        graph: GraphIndexer | None = None,
        extractor: MemoryExtractor | None = None,
        consolidator: MemoryConsolidator | None = None,
        extract_window: int = 10,
        remember_every: int = 1,
        callbacks: list[MemoryCallback] | None = None,
    ) -> None:
        if extract_window <= 0:
            msg = "extract_window must be a positive integer"
            raise ValueError(msg)
        if remember_every < 0:
            msg = "remember_every must be >= 0 (0 = only on an explicit remember())"
            raise ValueError(msg)
        self._extractor = extractor
        self._consolidator = consolidator
        self._extract_window = extract_window
        self._remember_every = remember_every
        self._turns_since_remember = 0
        self._last_remembered: ConversationTurn | None = None
        self._callbacks = callbacks or []
        self._tokenizer = tokenizer or get_default_counter()
        if graph is not None and persistent_store is not None:
            from anchor.ingestion.graph_extractors import GraphIndexingEntryStore

            persistent_store = GraphIndexingEntryStore(persistent_store, graph)
        if conversation_memory is not None:
            self._conversation: ConversationMemory = conversation_memory
        else:
            if conversation_tokens <= 0:
                msg = "conversation_tokens must be a positive integer"
                raise ValueError(msg)
            self._conversation = SlidingWindowMemory(
                max_tokens=conversation_tokens,
                tokenizer=self._tokenizer,
                on_evict=on_evict,
            )
        self._persistent_store = persistent_store

    def __repr__(self) -> str:
        has_store = self._persistent_store is not None
        return (
            f"MemoryManager(conversation={self._conversation!r}, "
            f"persistent_store={'yes' if has_store else 'none'})"
        )

    @property
    def conversation(self) -> ConversationMemory:
        """Access the underlying conversation memory."""
        return self._conversation

    @property
    def conversation_type(self) -> str:
        """Return the type of the underlying conversation memory.

        Returns ``"sliding_window"``, ``"summary_buffer"``,
        ``"progressive_summarization"``, or the class name for custom
        ``ConversationMemory`` implementations.
        """
        if isinstance(self._conversation, ProgressiveSummarizationMemory):
            return "progressive_summarization"
        if isinstance(self._conversation, SummaryBufferMemory):
            return "summary_buffer"
        if isinstance(self._conversation, SlidingWindowMemory):
            return "sliding_window"
        return type(self._conversation).__name__

    @property
    def persistent_store(self) -> MemoryEntryStore | None:
        """Access the underlying persistent memory store, if any."""
        return self._persistent_store

    # ---- Conversation helpers ----

    def _add_message(self, role: Role, content: str) -> None:
        """Add a message to the conversation backend (works with both types)."""
        if isinstance(self._conversation, (ProgressiveSummarizationMemory, SummaryBufferMemory)):
            self._conversation.add_message(role, content)
        elif isinstance(self._conversation, SlidingWindowMemory):
            self._conversation.add_turn(role, content)
        else:
            msg = (
                f"ConversationMemory implementation {type(self._conversation).__name__!r} "
                "does not support add_turn() or add_message()"
            )
            raise TypeError(msg)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self._add_message("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation history."""
        self._add_message("system", content)

    def add_tool_message(self, content: str) -> None:
        """Add a tool message to the conversation history."""
        self._add_message("tool", content)

    # ---- Persistent fact management ----

    def add_fact(
        self,
        content: str,
        tags: list[str] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create and store a persistent memory entry.

        Performs content-hash deduplication: if an entry with the same
        content already exists, the existing entry is returned instead
        of creating a duplicate.

        Parameters:
            content: The textual content of the memory.
            tags: Optional classification tags.
            memory_type: The cognitive type of the memory.
            metadata: Arbitrary key-value metadata.

        Returns:
            The newly created ``MemoryEntry``, or the existing one if
            a duplicate was detected.

        Raises:
            StorageError: If no persistent store has been configured.
        """
        if self._persistent_store is None:
            msg = "No persistent_store configured. Pass a MemoryEntryStore to MemoryManager."
            raise StorageError(msg)

        # Content-hash deduplication: check for existing entry with same content
        content_hash = _compute_content_hash(content)
        for existing in self._persistent_store.list_all():
            if existing.content_hash == content_hash:
                return existing

        entry = MemoryEntry(
            content=content,
            tags=tags or [],
            memory_type=memory_type,
            metadata=metadata or {},
        )
        self._persistent_store.add(entry)
        return entry

    def get_relevant_facts(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search the persistent store for entries relevant to *query*.

        Returns an empty list when no persistent store is configured.
        """
        if self._persistent_store is None:
            return []
        return self._persistent_store.search(query, top_k=top_k)

    def get_all_facts(self) -> list[MemoryEntry]:
        """Return every entry in the persistent store.

        Returns an empty list when no persistent store is configured.
        """
        if self._persistent_store is None:
            return []
        return self._persistent_store.list_all()

    def delete_fact(self, entry_id: str) -> bool:
        """Delete a persistent memory entry by id.

        Returns ``False`` when no persistent store is configured or the
        entry does not exist.
        """
        if self._persistent_store is None:
            return False
        return self._persistent_store.delete(entry_id)

    def update_fact(self, entry_id: str, content: str) -> MemoryEntry | None:
        """Update the content of an existing persistent memory entry.

        Parameters:
            entry_id: The id of the entry to update.
            content: The new content text.

        Returns:
            The updated ``MemoryEntry``, or ``None`` if the entry was
            not found or no persistent store is configured.
        """
        if self._persistent_store is None:
            return None
        existing: MemoryEntry | None = None
        for entry in self._persistent_store.list_all():
            if entry.id == entry_id:
                existing = entry
                break
        if existing is None:
            return None
        updated = existing.model_copy(
            update={
                "content": content,
                "content_hash": _compute_content_hash(content),
                "updated_at": datetime.now(UTC),
            }
        )
        self._persistent_store.add(updated)
        return updated

    # ---- Long-term memory from the conversation (roadmap #5) ----

    def remember(self) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
        """Extract facts from the turns not yet remembered and consolidate them.

        Reads the conversation after the last turn a previous call saw (at
        most *extract_window* turns, the newest), so nothing is extracted
        twice and a turn that the window skips is gone for good. Returns the
        operations applied — empty without an extractor, a persistent store
        or new turns. ``DELETE`` is the soft delete of
        ``MemoryEntry.invalidate``. Fires ``on_extraction`` once and
        ``on_consolidation`` per operation, and resets the ``after_turn``
        counter. Errors propagate: this is the explicit call.
        """
        self._turns_since_remember = 0
        if self._extractor is None or self._persistent_store is None:
            return []
        turns = self._fresh_turns()
        if not turns:
            return []
        entries = self._extractor.extract(turns)
        self._last_remembered = turns[-1]  # only once extracted: a failed call retries them
        _fire_memory_callback(self._callbacks, "on_extraction", turns, entries)
        if not entries:
            return []
        return apply_consolidation(
            entries, self._persistent_store, self._consolidator, callbacks=self._callbacks
        )

    def _fresh_turns(self) -> list[ConversationTurn]:
        """The turns after the last remembered one — eviction-proof.

        The cursor is matched by identity, then by equality, so a backend
        that rebuilds ``ConversationTurn`` objects on every ``turns`` read
        still finds it. A cursor no longer in the window means everything
        there is new.
        """
        # Tool turns are the agent's own bookkeeping, never facts about the user:
        # they neither reach the extractor nor eat the window.
        turns = [t for t in self._conversation.turns if t.role != Role.TOOL]
        last = self._last_remembered
        start = 0
        if last is not None:
            start = next((i + 1 for i, t in enumerate(turns) if t is last or t == last), 0)
        return turns[start:][-self._extract_window :]

    def after_turn(self) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
        """Count a completed turn; run ``remember()`` every *remember_every* turns.

        Called by ``Agent`` after each turn. Like the eviction promoter, a
        failure here is logged and never propagates — the answer has
        already been delivered.
        """
        if self._remember_every == 0 or self._extractor is None:
            return []
        self._turns_since_remember += 1
        if self._turns_since_remember < self._remember_every:
            return []
        try:
            return self.remember()
        except Exception:
            logger.exception("remember() failed after the turn — ignoring to protect the agent")
            return []

    # ---- Context assembly ----

    def get_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Get all memory as context items for pipeline assembly.

        Persistent memory facts are included at priority 8 (between
        system=10 and conversation=7).  Conversation turns use the
        caller-supplied *priority* (default 7).
        """
        items: list[ContextItem] = []

        # Persistent facts first (higher priority)
        if self._persistent_store is not None:
            for entry in self._persistent_store.list_all():
                if entry.is_expired:
                    continue
                token_count = self._tokenizer.count_tokens(entry.content)
                item = ContextItem(
                    content=entry.content,
                    source=SourceType.MEMORY,
                    score=entry.relevance_score,
                    priority=8,
                    token_count=token_count,
                    metadata={
                        "memory_entry_id": entry.id,
                        "memory_type": str(entry.memory_type),
                        "tags": entry.tags,
                    },
                    created_at=entry.created_at,
                )
                items.append(item)

        # Conversation turns
        items.extend(self._conversation.to_context_items(priority=priority))
        return items

    def clear(self) -> None:
        """Clear conversation history and persistent store (if present)."""
        self._conversation.clear()
        if self._persistent_store is not None:
            self._persistent_store.clear()
