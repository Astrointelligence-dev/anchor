"""Memory manager that coordinates conversation memory."""

from __future__ import annotations

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
from anchor.protocols.memory import ConversationMemory
from anchor.protocols.storage import MemoryEntryStore
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

from .progressive import ProgressiveSummarizationMemory
from .sliding_window import SlidingWindowMemory
from .summary_buffer import SummaryBufferMemory

if TYPE_CHECKING:
    from anchor.ingestion.graph_extractors import GraphIndexer


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
    """

    __slots__ = ("_conversation", "_persistent_store", "_tokenizer")

    def __init__(
        self,
        conversation_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
        on_evict: Callable[[list[ConversationTurn]], None] | None = None,
        persistent_store: MemoryEntryStore | None = None,
        conversation_memory: ConversationMemory | None = None,
        graph: GraphIndexer | None = None,
    ) -> None:
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
        if isinstance(self._conversation, ProgressiveSummarizationMemory) or isinstance(
            self._conversation, SummaryBufferMemory
        ):
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
