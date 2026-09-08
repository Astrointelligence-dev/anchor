"""Tests for MemoryGarbageCollector and GCStats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from anchor.memory.gc import (
    GarbageCollectableStore,
    GCStats,
    MemoryGarbageCollector,
)
from anchor.models.memory import MemoryEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleStore:
    """Minimal store implementing GarbageCollectableStore for testing."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def delete(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None


class FixedDecay:
    """Decay that returns a fixed score for each entry id."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def compute_retention(self, entry: MemoryEntry) -> float:
        return self._scores.get(entry.id, 1.0)


class AlwaysLowDecay:
    """Decay that always returns a very low score."""

    def __init__(self, score: float = 0.01) -> None:
        self._score = score

    def compute_retention(self, entry: MemoryEntry) -> float:
        return self._score


class AlwaysHighDecay:
    """Decay that always returns a high score."""

    def __init__(self, score: float = 0.99) -> None:
        self._score = score

    def compute_retention(self, entry: MemoryEntry) -> float:
        return self._score


class RecordingGCCallback:
    """Records GC-relevant callbacks for assertion."""

    def __init__(self) -> None:
        self.expiry_calls: list[list[MemoryEntry]] = []
        self.decay_calls: list[tuple[list[MemoryEntry], float]] = []

    def on_eviction(self, turns: Any, remaining_tokens: int) -> None: ...

    def on_compaction(
        self, evicted_turns: Any, summary: str, previous_summary: str | None
    ) -> None: ...

    def on_extraction(self, turns: Any, entries: Any) -> None: ...

    def on_consolidation(
        self, action: str, new_entry: Any, existing_entry: Any
    ) -> None: ...

    def on_decay_prune(
        self, pruned_entries: list[MemoryEntry], threshold: float
    ) -> None:
        self.decay_calls.append((pruned_entries, threshold))

    def on_expiry_prune(self, pruned_entries: list[MemoryEntry]) -> None:
        self.expiry_calls.append(pruned_entries)


def _make_entry(
    entry_id: str = "e1",
    content: str = "test",
    *,
    expired: bool = False,
    expires_in_hours: float | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry with optional expiry."""
    expires_at: datetime | None = None
    if expired:
        expires_at = datetime.now(UTC) - timedelta(hours=1)
    elif expires_in_hours is not None:
        expires_at = datetime.now(UTC) + timedelta(hours=expires_in_hours)
    return MemoryEntry(id=entry_id, content=content, expires_at=expires_at)


# ---------------------------------------------------------------------------
# Tests: GCStats
# ---------------------------------------------------------------------------


class TestGCStats:
    def test_total_pruned(self) -> None:
        stats = GCStats(expired_pruned=3, decayed_pruned=5, total_remaining=42, dry_run=False)
        assert stats.total_pruned == 8

    def test_total_pruned_zero(self) -> None:
        stats = GCStats(expired_pruned=0, decayed_pruned=0, total_remaining=10, dry_run=False)
        assert stats.total_pruned == 0

    def test_repr_applied(self) -> None:
        stats = GCStats(expired_pruned=2, decayed_pruned=3, total_remaining=10, dry_run=False)
        r = repr(stats)
        assert "applied" in r
        assert "expired_pruned=2" in r
        assert "decayed_pruned=3" in r
        assert "total_remaining=10" in r

    def test_repr_dry_run(self) -> None:
        stats = GCStats(expired_pruned=1, decayed_pruned=0, total_remaining=5, dry_run=True)
        r = repr(stats)
        assert "dry_run" in r

    def test_equality(self) -> None:
        s1 = GCStats(expired_pruned=1, decayed_pruned=2, total_remaining=3, dry_run=False)
        s2 = GCStats(expired_pruned=1, decayed_pruned=2, total_remaining=3, dry_run=False)
        assert s1 == s2

    def test_inequality(self) -> None:
        s1 = GCStats(expired_pruned=1, decayed_pruned=2, total_remaining=3, dry_run=False)
        s2 = GCStats(expired_pruned=0, decayed_pruned=2, total_remaining=3, dry_run=False)
        assert s1 != s2

    def test_inequality_with_different_type(self) -> None:
        stats = GCStats(expired_pruned=0, decayed_pruned=0, total_remaining=0, dry_run=False)
        assert stats != "not a GCStats"


# ---------------------------------------------------------------------------
# Tests: GarbageCollectableStore protocol
# ---------------------------------------------------------------------------


class TestGarbageCollectableStoreProtocol:
    def test_simple_store_satisfies_protocol(self) -> None:
        store = SimpleStore()
        assert isinstance(store, GarbageCollectableStore)

    def test_object_without_methods_does_not_satisfy(self) -> None:
        assert not isinstance("not a store", GarbageCollectableStore)


# ---------------------------------------------------------------------------
# Tests: collect() -- expired entries
# ---------------------------------------------------------------------------


class TestCollectExpired:
    def test_removes_expired_entries(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "old", expired=True))
        store.add(_make_entry("e2", "still fresh"))

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 1
        assert stats.total_remaining == 1
        # Verify entry was actually deleted
        remaining_ids = {e.id for e in store.list_all_unfiltered()}
        assert "e1" not in remaining_ids
        assert "e2" in remaining_ids

    def test_removes_multiple_expired_entries(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired1", expired=True))
        store.add(_make_entry("e2", "expired2", expired=True))
        store.add(_make_entry("e3", "fresh"))

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 2
        assert stats.total_remaining == 1

    def test_no_expired_entries(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "fresh"))
        store.add(_make_entry("e2", "also fresh"))

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 0
        assert stats.total_remaining == 2


# ---------------------------------------------------------------------------
# Tests: collect() -- decayed entries
# ---------------------------------------------------------------------------


class TestCollectDecayed:
    def test_removes_decayed_entries_below_threshold(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "weak memory"))
        store.add(_make_entry("e2", "strong memory"))

        decay = FixedDecay({"e1": 0.05, "e2": 0.9})
        gc = MemoryGarbageCollector(store, decay=decay)
        stats = gc.collect(retention_threshold=0.1)

        assert stats.decayed_pruned == 1
        assert stats.total_remaining == 1
        remaining_ids = {e.id for e in store.list_all_unfiltered()}
        assert "e1" not in remaining_ids
        assert "e2" in remaining_ids

    def test_no_decay_function_skips_decay_phase(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "some memory"))

        gc = MemoryGarbageCollector(store, decay=None)
        stats = gc.collect(retention_threshold=0.1)

        assert stats.decayed_pruned == 0
        assert stats.total_remaining == 1

    def test_expired_entries_not_scored_for_decay(self) -> None:
        """Expired entries are removed in the expiry phase and should
        not be considered for decay scoring."""
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "strong"))

        # If e1 were scored, AlwaysLowDecay would tag it.
        # But it should be removed by expiry first.
        decay = AlwaysHighDecay(score=0.99)
        gc = MemoryGarbageCollector(store, decay=decay)
        stats = gc.collect(retention_threshold=0.5)

        assert stats.expired_pruned == 1
        assert stats.decayed_pruned == 0
        assert stats.total_remaining == 1


# ---------------------------------------------------------------------------
# Tests: collect() -- dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_does_not_delete_expired(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "fresh"))

        gc = MemoryGarbageCollector(store)
        stats = gc.collect(dry_run=True)

        assert stats.dry_run is True
        assert stats.expired_pruned == 1
        # Entry should still be in the store
        assert len(store.list_all_unfiltered()) == 2

    def test_dry_run_does_not_delete_decayed(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "weak"))

        decay = AlwaysLowDecay(score=0.01)
        gc = MemoryGarbageCollector(store, decay=decay)
        stats = gc.collect(retention_threshold=0.1, dry_run=True)

        assert stats.dry_run is True
        assert stats.decayed_pruned == 1
        # Entry should still be in the store
        assert len(store.list_all_unfiltered()) == 1

    def test_dry_run_still_fires_callbacks(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))

        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, callbacks=[cb])
        gc.collect(dry_run=True)

        assert len(cb.expiry_calls) == 1


# ---------------------------------------------------------------------------
# Tests: collect() fires callbacks
# ---------------------------------------------------------------------------


class TestCollectCallbacks:
    def test_fires_expiry_callback(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))

        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, callbacks=[cb])
        gc.collect()

        assert len(cb.expiry_calls) == 1
        assert len(cb.expiry_calls[0]) == 1
        assert cb.expiry_calls[0][0].id == "e1"

    def test_fires_decay_callback(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "low retention"))

        decay = AlwaysLowDecay(score=0.01)
        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, decay=decay, callbacks=[cb])
        gc.collect(retention_threshold=0.1)

        assert len(cb.decay_calls) == 1
        pruned, threshold = cb.decay_calls[0]
        assert len(pruned) == 1
        assert pruned[0].id == "e1"
        assert threshold == 0.1

    def test_fires_both_callbacks(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "weak"))

        decay = AlwaysLowDecay(score=0.01)
        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, decay=decay, callbacks=[cb])
        gc.collect(retention_threshold=0.5)

        assert len(cb.expiry_calls) == 1
        assert len(cb.decay_calls) == 1

    def test_no_callback_when_nothing_pruned(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "healthy"))

        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, callbacks=[cb])
        gc.collect()

        assert len(cb.expiry_calls) == 0
        assert len(cb.decay_calls) == 0


# ---------------------------------------------------------------------------
# Tests: collect_expired()
# ---------------------------------------------------------------------------


class TestCollectExpiredMethod:
    def test_only_removes_expired(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "fresh"))

        gc = MemoryGarbageCollector(store)
        pruned = gc.collect_expired()

        assert len(pruned) == 1
        assert pruned[0].id == "e1"
        assert len(store.list_all_unfiltered()) == 1

    def test_dry_run_does_not_delete(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))

        gc = MemoryGarbageCollector(store)
        pruned = gc.collect_expired(dry_run=True)

        assert len(pruned) == 1
        assert len(store.list_all_unfiltered()) == 1  # Still there

    def test_empty_store(self) -> None:
        store = SimpleStore()
        gc = MemoryGarbageCollector(store)
        pruned = gc.collect_expired()
        assert pruned == []


# ---------------------------------------------------------------------------
# Tests: collect_decayed()
# ---------------------------------------------------------------------------


class TestCollectDecayedMethod:
    def test_only_removes_low_retention(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "weak"))
        store.add(_make_entry("e2", "strong"))

        decay = FixedDecay({"e1": 0.05, "e2": 0.8})
        gc = MemoryGarbageCollector(store, decay=decay)
        pruned = gc.collect_decayed(retention_threshold=0.1)

        assert len(pruned) == 1
        assert pruned[0].id == "e1"
        assert len(store.list_all_unfiltered()) == 1

    def test_raises_without_decay_function(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "something"))

        gc = MemoryGarbageCollector(store, decay=None)

        with pytest.raises(ValueError, match="decay function"):
            gc.collect_decayed()

    def test_dry_run_does_not_delete(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "weak"))

        decay = AlwaysLowDecay(score=0.01)
        gc = MemoryGarbageCollector(store, decay=decay)
        pruned = gc.collect_decayed(retention_threshold=0.1, dry_run=True)

        assert len(pruned) == 1
        assert len(store.list_all_unfiltered()) == 1  # Still there

    def test_all_above_threshold(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "solid memory"))

        decay = AlwaysHighDecay(score=0.9)
        gc = MemoryGarbageCollector(store, decay=decay)
        pruned = gc.collect_decayed(retention_threshold=0.5)

        assert pruned == []
        assert len(store.list_all_unfiltered()) == 1

    def test_skips_expired_entries(self) -> None:
        """Expired entries should not be scored for decay."""
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "fresh and strong"))

        decay = AlwaysHighDecay(score=0.99)
        gc = MemoryGarbageCollector(store, decay=decay)
        pruned = gc.collect_decayed(retention_threshold=0.5)

        # e1 is expired -> skipped, e2 is above threshold -> not pruned
        assert pruned == []


# ---------------------------------------------------------------------------
# Tests: Empty store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    def test_collect_on_empty_store(self) -> None:
        store = SimpleStore()
        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 0
        assert stats.decayed_pruned == 0
        assert stats.total_remaining == 0
        assert stats.total_pruned == 0
        assert stats.dry_run is False

    def test_collect_expired_on_empty_store(self) -> None:
        store = SimpleStore()
        gc = MemoryGarbageCollector(store)
        assert gc.collect_expired() == []

    def test_collect_decayed_on_empty_store(self) -> None:
        store = SimpleStore()
        decay = AlwaysLowDecay()
        gc = MemoryGarbageCollector(store, decay=decay)
        assert gc.collect_decayed() == []


# ---------------------------------------------------------------------------
# Tests: Preserves high-retention, non-expired entries
# ---------------------------------------------------------------------------


class TestPreservation:
    def test_preserves_non_expired_high_retention(self) -> None:
        store = SimpleStore()
        store.add(_make_entry("e1", "good memory"))
        store.add(_make_entry("e2", "another good one"))
        store.add(_make_entry("e3", "future expiry", expires_in_hours=24))

        decay = AlwaysHighDecay(score=0.99)
        gc = MemoryGarbageCollector(store, decay=decay)
        stats = gc.collect(retention_threshold=0.5)

        assert stats.total_pruned == 0
        assert stats.total_remaining == 3

    def test_preserves_entries_with_no_expiry(self) -> None:
        store = SimpleStore()
        store.add(MemoryEntry(id="e1", content="permanent"))
        store.add(MemoryEntry(id="e2", content="also permanent"))

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 0
        assert stats.total_remaining == 2


# ---------------------------------------------------------------------------
# Tests: Integration scenario
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_mixed_entries_gc(self) -> None:
        """Add entries with various expiry and decay states, run GC,
        and verify correct pruning."""
        store = SimpleStore()

        # Expired entries (should be removed)
        store.add(_make_entry("expired-1", "old news", expired=True))
        store.add(_make_entry("expired-2", "stale data", expired=True))

        # Low-retention entries (should be removed by decay)
        store.add(_make_entry("weak-1", "fading memory"))
        store.add(_make_entry("weak-2", "barely remembered"))

        # High-retention entries (should survive)
        store.add(_make_entry("strong-1", "important fact"))
        store.add(_make_entry("strong-2", "core knowledge"))

        # Future-expiry entry (should survive)
        store.add(_make_entry("future-1", "upcoming", expires_in_hours=48))

        decay = FixedDecay({
            "weak-1": 0.02,
            "weak-2": 0.05,
            "strong-1": 0.95,
            "strong-2": 0.88,
            "future-1": 0.75,
        })
        cb = RecordingGCCallback()
        gc = MemoryGarbageCollector(store, decay=decay, callbacks=[cb])
        stats = gc.collect(retention_threshold=0.1)

        # Verify stats
        assert stats.expired_pruned == 2
        assert stats.decayed_pruned == 2
        assert stats.total_pruned == 4
        assert stats.total_remaining == 3
        assert stats.dry_run is False

        # Verify remaining entries
        remaining_ids = {e.id for e in store.list_all_unfiltered()}
        assert remaining_ids == {"strong-1", "strong-2", "future-1"}

        # Verify callbacks fired
        assert len(cb.expiry_calls) == 1
        assert len(cb.decay_calls) == 1
        expired_ids = {e.id for e in cb.expiry_calls[0]}
        assert expired_ids == {"expired-1", "expired-2"}
        decayed_ids = {e.id for e in cb.decay_calls[0][0]}
        assert decayed_ids == {"weak-1", "weak-2"}

    def test_integration_dry_run_then_apply(self) -> None:
        """Dry run first to inspect, then apply."""
        store = SimpleStore()
        store.add(_make_entry("e1", "expired", expired=True))
        store.add(_make_entry("e2", "weak"))
        store.add(_make_entry("e3", "strong"))

        decay = FixedDecay({"e2": 0.01, "e3": 0.99})
        gc = MemoryGarbageCollector(store, decay=decay)

        # Dry run -- nothing should be deleted
        dry_stats = gc.collect(retention_threshold=0.1, dry_run=True)
        assert dry_stats.dry_run is True
        assert dry_stats.total_pruned == 2
        assert len(store.list_all_unfiltered()) == 3  # All still there

        # Apply -- entries should be deleted
        stats = gc.collect(retention_threshold=0.1, dry_run=False)
        assert stats.dry_run is False
        assert stats.total_pruned == 2
        assert stats.total_remaining == 1
        assert len(store.list_all_unfiltered()) == 1
        assert store.list_all_unfiltered()[0].id == "e3"


# ---------------------------------------------------------------------------
# Retention (history of invalidated entries)
# ---------------------------------------------------------------------------


class TestRetention:
    def test_expired_entries_survive_until_retention_elapses(self) -> None:
        store = SimpleStore()
        now = datetime.now(UTC)
        store.add(MemoryEntry(id="fresh", content="a", expires_at=now - timedelta(minutes=1)))
        store.add(MemoryEntry(id="stale", content="b", expires_at=now - timedelta(hours=2)))
        store.add(MemoryEntry(id="live", content="c"))

        stats = MemoryGarbageCollector(store, retention=timedelta(hours=1)).collect()

        assert stats == GCStats(
            expired_pruned=1, decayed_pruned=0, total_remaining=2, dry_run=False
        )
        assert {e.id for e in store.list_all_unfiltered()} == {"fresh", "live"}

    def test_invalidate_is_a_soft_delete_the_collector_honours(self) -> None:
        store = SimpleStore()
        old = MemoryEntry(id="old", content="lives in SP")
        store.add(old.invalidate(by="new"))

        kept = store.list_all_unfiltered()[0]
        assert kept.is_expired
        assert kept.metadata == {"invalidated_by": "new"}
        assert kept.content_hash == old.content_hash  # content untouched
        assert MemoryGarbageCollector(store, retention=timedelta(days=1)).collect_expired() == []
        assert [e.id for e in MemoryGarbageCollector(store).collect_expired()] == ["old"]
