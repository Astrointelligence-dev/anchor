"""Tests for anchor.memory.consolidator."""

from __future__ import annotations

import math
from collections.abc import Callable

from anchor.memory.consolidator import SimilarityConsolidator
from anchor.models.memory import MemoryEntry
from anchor.protocols.memory import MemoryOperation


def _fake_embed(text: str) -> list[float]:
    """Deterministic embedding: hash-based 8-dim vector for testing."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(8)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        return raw
    return [x / norm for x in raw]


def _identical_embed(_text: str) -> list[float]:
    """Always returns the same embedding -- everything looks identical."""
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_orthogonal_embed() -> Callable[[str], list[float]]:
    """Factory that returns an embed function with a fresh index per call."""
    index: dict[str, int] = {}

    def _embed(text: str) -> list[float]:
        dim = 128
        if text not in index:
            index[text] = len(index)
        idx = index[text] % dim
        vec = [0.0] * dim
        vec[idx] = 1.0
        return vec

    return _embed


class TestSimilarityConsolidatorDedup:
    """Exact content-hash deduplication returns 'none'."""

    def test_exact_duplicate_returns_none(self) -> None:
        consolidator = SimilarityConsolidator(embed_fn=_fake_embed)
        existing = MemoryEntry(content="User likes Python")
        new_entry = MemoryEntry(content="User likes Python")
        # Same content -> same content_hash
        assert existing.content_hash == new_entry.content_hash

        results = consolidator.consolidate([new_entry], [existing])
        assert len(results) == 1
        action, entry = results[0]
        assert action == MemoryOperation.NONE
        assert entry is None

    def test_different_content_not_deduped(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=make_orthogonal_embed(),
            similarity_threshold=0.99,
        )
        existing = MemoryEntry(content="User likes Python")
        new_entry = MemoryEntry(content="User likes JavaScript")

        results = consolidator.consolidate([new_entry], [existing])
        assert len(results) == 1
        action, _entry = results[0]
        assert action != MemoryOperation.NONE


class TestSimilarityConsolidatorUpdate:
    """High similarity returns 'update' with merged entry."""

    def test_high_similarity_returns_update(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.8,
        )
        existing = MemoryEntry(
            content="User likes coding",
            tags=["preference"],
            relevance_score=0.5,
        )
        new_entry = MemoryEntry(
            content="User really likes coding a lot",
            tags=["user-pref"],
            relevance_score=0.7,
        )
        # Different content -> different hash -> not exact duplicate
        # But _identical_embed returns same vector -> cosine sim = 1.0 > threshold

        results = consolidator.consolidate([new_entry], [existing])
        assert len(results) == 1
        action, merged = results[0]
        assert action == MemoryOperation.UPDATE
        assert merged is not None

    def test_merged_entry_preserves_longer_content(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="short")
        new_entry = MemoryEntry(content="this is a longer content string")

        results = consolidator.consolidate([new_entry], [existing])
        action, merged = results[0]
        assert action == MemoryOperation.UPDATE
        assert merged is not None
        assert merged.content == "this is a longer content string"

    def test_merged_entry_keeps_existing_when_longer(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="existing content that is much longer")
        new_entry = MemoryEntry(content="short new")

        results = consolidator.consolidate([new_entry], [existing])
        action, merged = results[0]
        assert action == MemoryOperation.UPDATE
        assert merged is not None
        assert merged.content == "existing content that is much longer"

    def test_merged_entry_combines_tags(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="content A", tags=["tag1", "tag2"])
        new_entry = MemoryEntry(content="content B slightly longer", tags=["tag2", "tag3"])

        results = consolidator.consolidate([new_entry], [existing])
        _, merged = results[0]
        assert merged is not None
        # Tags should be merged with deduplication
        assert "tag1" in merged.tags
        assert "tag2" in merged.tags
        assert "tag3" in merged.tags
        # No duplicates
        assert len(merged.tags) == 3

    def test_merged_entry_increments_access_count(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="content", access_count=3)
        new_entry = MemoryEntry(content="slightly different content text")

        results = consolidator.consolidate([new_entry], [existing])
        _, merged = results[0]
        assert merged is not None
        assert merged.access_count == 4  # existing.access_count + 1

    def test_merged_entry_keeps_max_relevance_score(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="a", relevance_score=0.3)
        new_entry = MemoryEntry(content="ab", relevance_score=0.9)

        results = consolidator.consolidate([new_entry], [existing])
        _, merged = results[0]
        assert merged is not None
        assert merged.relevance_score == 0.9


class TestSimilarityConsolidatorAdd:
    """Low similarity returns 'add'."""

    def test_low_similarity_returns_add(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=make_orthogonal_embed(),
            similarity_threshold=0.85,
        )
        existing = MemoryEntry(content="User likes Python")
        new_entry = MemoryEntry(content="The weather is sunny today")

        results = consolidator.consolidate([new_entry], [existing])
        assert len(results) == 1
        action, entry = results[0]
        assert action == MemoryOperation.ADD
        assert entry is new_entry

    def test_empty_existing_all_entries_are_add(self) -> None:
        consolidator = SimilarityConsolidator(embed_fn=_fake_embed)
        new_entries = [
            MemoryEntry(content="Fact one"),
            MemoryEntry(content="Fact two"),
            MemoryEntry(content="Fact three"),
        ]

        results = consolidator.consolidate(new_entries, existing=[])
        assert len(results) == 3
        for action, entry in results:
            assert action == MemoryOperation.ADD
            assert entry is not None


class TestSimilarityConsolidatorMultiple:
    """Consolidating multiple entries at once."""

    def test_mixed_actions(self) -> None:
        consolidator = SimilarityConsolidator(
            embed_fn=_identical_embed,
            similarity_threshold=0.5,
        )
        existing = MemoryEntry(content="known fact")
        duplicate = MemoryEntry(content="known fact")  # exact duplicate
        similar = MemoryEntry(content="related but different fact text")  # will be similar

        results = consolidator.consolidate([duplicate, similar], [existing])
        assert len(results) == 2
        # First entry is exact duplicate -> none
        assert results[0][0] == MemoryOperation.NONE
        # Second entry has same embedding (identical_embed) -> update
        assert results[1][0] == MemoryOperation.UPDATE

    def test_empty_new_entries(self) -> None:
        consolidator = SimilarityConsolidator(embed_fn=_fake_embed)
        existing = MemoryEntry(content="existing")
        results = consolidator.consolidate([], [existing])
        assert results == []


class TestSimilarityConsolidatorValidation:
    """Constructor validation."""

    def test_invalid_threshold_above_one(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="similarity_threshold"):
            SimilarityConsolidator(embed_fn=_fake_embed, similarity_threshold=1.5)

    def test_invalid_threshold_below_zero(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="similarity_threshold"):
            SimilarityConsolidator(embed_fn=_fake_embed, similarity_threshold=-0.1)


class TestMergedEntryHash:
    def test_merge_recomputes_content_hash(self) -> None:
        from anchor.memory.consolidator import SimilarityConsolidator
        from anchor.models.memory import MemoryEntry, _compute_content_hash

        existing = MemoryEntry(id="e1", content="alice likes tea")
        newer = MemoryEntry(id="e2", content="alice likes tea and biscuits")
        merged = SimilarityConsolidator._merge_entries(newer, existing)
        assert merged.content == "alice likes tea and biscuits"
        assert merged.content_hash == _compute_content_hash(merged.content)
        assert merged.content_hash != existing.content_hash
