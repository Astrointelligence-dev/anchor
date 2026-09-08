"""Garbage collection for memory entries.

Provides ``MemoryGarbageCollector`` which prunes expired and decayed
memory entries from a store.  The collector fires ``MemoryCallback``
hooks so that external systems can observe what was pruned.

The store must implement ``GarbageCollectableStore`` which extends the
base ``MemoryEntryStore`` protocol with a ``list_all_unfiltered`` method
that returns *all* entries -- including expired ones -- so that the
garbage collector can identify and delete them.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anchor.memory.callbacks import MemoryCallback, _fire_memory_callback
from anchor.protocols.storage import GarbageCollectableStore

if TYPE_CHECKING:
    from datetime import timedelta

    from anchor.models.memory import MemoryEntry
    from anchor.protocols.memory import MemoryDecay

logger = logging.getLogger(__name__)


class GCStats:
    """Statistics from a garbage collection run.

    Attributes:
        expired_pruned: Number of entries pruned because they were expired.
        decayed_pruned: Number of entries pruned because their retention
            score fell below the threshold.
        total_remaining: Number of entries still in the store after collection.
        dry_run: Whether this was a dry run (no entries actually deleted).
    """

    __slots__ = ("decayed_pruned", "dry_run", "expired_pruned", "total_remaining")

    def __init__(
        self,
        expired_pruned: int,
        decayed_pruned: int,
        total_remaining: int,
        dry_run: bool,
    ) -> None:
        self.expired_pruned = expired_pruned
        self.decayed_pruned = decayed_pruned
        self.total_remaining = total_remaining
        self.dry_run = dry_run

    @property
    def total_pruned(self) -> int:
        """Total number of entries pruned (expired + decayed)."""
        return self.expired_pruned + self.decayed_pruned

    def __repr__(self) -> str:
        mode = "dry_run" if self.dry_run else "applied"
        return (
            f"GCStats({mode}: expired_pruned={self.expired_pruned}, "
            f"decayed_pruned={self.decayed_pruned}, "
            f"total_remaining={self.total_remaining})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GCStats):
            return NotImplemented
        return (
            self.expired_pruned == other.expired_pruned
            and self.decayed_pruned == other.decayed_pruned
            and self.total_remaining == other.total_remaining
            and self.dry_run == other.dry_run
        )


class MemoryGarbageCollector:
    """Prunes expired and decayed memory entries from a store.

    Usage::

        gc = MemoryGarbageCollector(store, decay=EbbinghausDecay())
        stats = gc.collect(retention_threshold=0.1)
        # stats = GCStats(expired_pruned=3, decayed_pruned=5, total_remaining=42)

    The collector works in two phases:

    1. **Expiry phase** -- remove entries whose ``is_expired`` property
       returns ``True`` (and, with *retention*, whose expiry is older than
       the retention window).
    2. **Decay phase** -- if a ``MemoryDecay`` function is provided,
       compute the retention score of every remaining entry and remove
       those whose score falls below ``retention_threshold``.

    Both phases fire the appropriate ``MemoryCallback`` hooks.

    *retention* keeps every expired entry around for that long before the
    expiry phase deletes it — a TTL that lapsed as much as an invalidated
    memory (a consolidator's ``DELETE``, see ``MemoryEntry.invalidate``) —
    so it stays readable through ``list_all_unfiltered`` as history and
    counts in ``GCStats.total_remaining`` meanwhile. ``None`` (default)
    deletes on the first collection. A TTL entry indexed in a knowledge
    graph keeps its evidence until it is actually deleted.
    """

    __slots__ = ("_callbacks", "_decay", "_retention", "_store")

    def __init__(
        self,
        store: GarbageCollectableStore,
        decay: MemoryDecay | None = None,
        callbacks: list[MemoryCallback] | None = None,
        *,
        retention: timedelta | None = None,
    ) -> None:
        self._store = store
        self._decay = decay
        self._callbacks = callbacks or []
        self._retention = retention

    def _collectable(self, entry: MemoryEntry) -> bool:
        """Expired — and past the retention window when one is configured."""
        if entry.expires_at is None or not entry.is_expired:
            return False
        if self._retention is None:
            return True
        return entry.expires_at + self._retention <= datetime.now(UTC)

    def collect(
        self,
        retention_threshold: float = 0.1,
        dry_run: bool = False,
    ) -> GCStats:
        """Run garbage collection.

        1. Remove all expired entries (``is_expired is True``).
        2. If a decay function is configured, compute retention for
           remaining entries and remove those below *retention_threshold*.
        3. Fire callbacks for pruned entries.
        4. Return stats.

        Parameters:
            retention_threshold: Retention score below which entries are
                pruned.  Only used when a decay function is configured.
            dry_run: If ``True``, identify entries that *would* be pruned
                without actually deleting them.

        Returns:
            A ``GCStats`` instance summarising what was (or would be) pruned.
        """
        all_entries = self._store.list_all_unfiltered()
        expired = self.collect_expired(dry_run=dry_run, _entries=all_entries)
        decayed = self.collect_decayed(
            retention_threshold=retention_threshold, dry_run=dry_run, _entries=all_entries
        ) if self._decay is not None else []

        expired_ids = {e.id for e in expired}
        decayed_ids = {e.id for e in decayed}
        total_remaining = len(all_entries) - len(expired_ids) - len(decayed_ids)
        return GCStats(
            expired_pruned=len(expired),
            decayed_pruned=len(decayed),
            total_remaining=total_remaining,
            dry_run=dry_run,
        )

    def collect_expired(
        self,
        dry_run: bool = False,
        _entries: list[MemoryEntry] | None = None,
    ) -> list[MemoryEntry]:
        """Remove only expired entries past retention (simpler, no decay scoring).

        Parameters:
            dry_run: If ``True``, identify but do not delete entries.
            _entries: Pre-fetched entries list (internal optimisation).

        Returns:
            The list of entries that were (or would be) pruned.
        """
        all_entries = _entries if _entries is not None else self._store.list_all_unfiltered()
        expired = [e for e in all_entries if self._collectable(e)]

        if expired and not dry_run:
            for entry in expired:
                self._store.delete(entry.id)

        if expired:
            _fire_memory_callback(
                self._callbacks, "on_expiry_prune", expired
            )

        return expired

    def collect_decayed(
        self,
        retention_threshold: float = 0.1,
        dry_run: bool = False,
        _entries: list[MemoryEntry] | None = None,
    ) -> list[MemoryEntry]:
        """Remove only decayed entries (requires decay function).

        Parameters:
            retention_threshold: Retention score below which entries are
                pruned.
            dry_run: If ``True``, identify but do not delete entries.
            _entries: Pre-fetched entries list (internal optimisation).

        Returns:
            The list of entries that were (or would be) pruned.

        Raises:
            ValueError: If no decay function was configured.
        """
        if self._decay is None:
            msg = "Cannot collect decayed entries without a decay function"
            raise ValueError(msg)

        all_entries = _entries if _entries is not None else self._store.list_all_unfiltered()
        # Only consider non-expired entries for decay scoring
        candidates = [e for e in all_entries if not e.is_expired]

        decayed: list[MemoryEntry] = []
        for entry in candidates:
            retention = self._decay.compute_retention(entry)
            if retention < retention_threshold:
                decayed.append(entry)

        if decayed and not dry_run:
            for entry in decayed:
                self._store.delete(entry.id)

        if decayed:
            _fire_memory_callback(
                self._callbacks, "on_decay_prune", decayed, retention_threshold
            )

        return decayed
