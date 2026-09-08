"""Callback protocol for memory operation observability.

Provides a ``MemoryCallback`` protocol that mirrors the approach used by
``PipelineCallback`` in ``pipeline/callbacks.py``.  All methods have
default no-op implementations so that partial implementations are valid.

The ``_fire_memory_callback`` helper iterates over a list of callbacks,
calling the named method on each one and swallowing any exception so that
a buggy callback can never crash a memory operation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from anchor._callbacks import fire_callbacks

if TYPE_CHECKING:
    from anchor.models.memory import ConversationTurn, KeyFact, MemoryEntry

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryCallback(Protocol):
    """Protocol for observing memory operations.

    All methods have default no-op implementations.  Implementers
    only need to override the methods they care about.
    """

    def on_eviction(
        self, turns: list[ConversationTurn], remaining_tokens: int
    ) -> None:
        """Called when conversation turns are evicted from the sliding window.

        Parameters:
            turns: The conversation turns that were evicted.
            remaining_tokens: Token count remaining in the window after eviction.
        """
        ...

    def on_compaction(
        self,
        evicted_turns: list[ConversationTurn],
        summary: str,
        previous_summary: str | None,
    ) -> None:
        """Called when evicted turns are compacted into a summary.

        Parameters:
            evicted_turns: The turns that were compacted.
            summary: The resulting summary text.
            previous_summary: The previous summary, or ``None`` if this is
                the first compaction.
        """
        ...

    def on_extraction(
        self, turns: list[ConversationTurn], entries: list[MemoryEntry]
    ) -> None:
        """Called when memories are extracted from conversation turns.

        Parameters:
            turns: The source conversation turns.
            entries: The extracted memory entries.
        """
        ...

    def on_consolidation(
        self,
        action: str,
        new_entry: MemoryEntry | None,
        existing_entry: MemoryEntry | None,
    ) -> None:
        """Called for each consolidation decision (add/update/delete/none).

        Parameters:
            action: One of ``"add"``, ``"update"``, ``"delete"``, ``"none"``.
            new_entry: The new entry being considered, or ``None``.
            existing_entry: The existing entry being compared against, or ``None``.
        """
        ...

    def on_decay_prune(
        self, pruned_entries: list[MemoryEntry], threshold: float
    ) -> None:
        """Called when entries are pruned due to low retention scores.

        Parameters:
            pruned_entries: The entries that were (or will be) pruned.
            threshold: The retention threshold below which entries are pruned.
        """
        ...

    def on_expiry_prune(self, pruned_entries: list[MemoryEntry]) -> None:
        """Called when expired entries are removed.

        Parameters:
            pruned_entries: The entries that were (or will be) removed.
        """
        ...


@runtime_checkable
class ProgressiveSummarizationCallback(Protocol):
    """Callback protocol for progressive summarization events.

    All methods have default no-op implementations.  Implementers
    only need to override the methods they care about.
    """

    def on_tier_cascade(
        self, from_tier: int, to_tier: int, tokens_in: int, tokens_out: int
    ) -> None:
        """Called when content cascades from one tier to a lower tier."""
        ...

    def on_facts_extracted(self, facts: list[KeyFact], source_tier: int) -> None:
        """Called when key facts are extracted during a tier transition."""
        ...

    def on_compaction_error(self, tier: int, error: Exception) -> None:
        """Called when compaction fails and falls back."""
        ...


def _fire_memory_callback(
    callbacks: list[MemoryCallback],
    method: str,
    *args: object,
    **kwargs: object,
) -> None:
    """Fire a callback method on all registered callbacks, swallowing errors.

    This mirrors the error-swallowing pattern from
    ``pipeline/callbacks.py`` -- a buggy callback must never crash a
    memory operation.

    Parameters:
        callbacks: The list of callback instances to notify.
        method: The name of the method to invoke on each callback.
        *args: Positional arguments forwarded to the callback method.
        **kwargs: Keyword arguments forwarded to the callback method.
    """
    fire_callbacks(callbacks, method, *args, logger=logger, log_level=logging.WARNING, **kwargs)
