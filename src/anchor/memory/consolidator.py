"""Memory consolidation via embedding similarity and content hashing.

Determines whether new memory entries should be added, merged with
existing entries, or skipped entirely based on content hash
deduplication and cosine similarity of embeddings.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anchor._math import cosine_similarity as _cosine_similarity
from anchor.models.memory import _compute_content_hash
from anchor.protocols.memory import MemoryOperation

if TYPE_CHECKING:
    from anchor.models.memory import MemoryEntry


class SimilarityConsolidator:
    """Consolidates memories using embedding cosine similarity and content hashing.

    For each new entry the consolidator:

    1. Checks ``content_hash`` against existing entries -- exact duplicates
       are skipped (``"none"``).
    2. Embeds the new entry and compares it against cached embeddings of
       existing entries using cosine similarity.
    3. If similarity exceeds the threshold the entries are merged
       (``"update"``).
    4. Otherwise the new entry is added as-is (``"add"``).

    The library never calls an LLM. The user-provided ``embed_fn``
    handles all embedding logic.
    """

    __slots__ = ("_embed_fn", "_embedding_cache", "_max_cache_size", "_similarity_threshold")

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            msg = "similarity_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold
        self._max_cache_size = max_cache_size
        self._embedding_cache: dict[str, list[float]] = {}

    def _get_embedding(self, entry: MemoryEntry) -> list[float]:
        """Return a cached embedding for the entry, computing if necessary."""
        if entry.id not in self._embedding_cache:
            if len(self._embedding_cache) >= self._max_cache_size:
                self._embedding_cache.clear()
            self._embedding_cache[entry.id] = self._embed_fn(entry.content)
        return self._embedding_cache[entry.id]

    @staticmethod
    def _merge_entries(new_entry: MemoryEntry, existing: MemoryEntry) -> MemoryEntry:
        """Merge a new entry into an existing one, preserving the richer metadata."""
        merged_tags = list(dict.fromkeys([*existing.tags, *new_entry.tags]))
        merged_links = list(dict.fromkeys([*existing.links, *new_entry.links]))
        merged_source_turns = list(
            dict.fromkeys([*existing.source_turns, *new_entry.source_turns])
        )
        merged_metadata = {**existing.metadata, **new_entry.metadata}

        # Keep the longer or newer content
        content = (
            new_entry.content
            if len(new_entry.content) >= len(existing.content)
            else existing.content
        )

        return existing.model_copy(
            update={
                "content": content,
                # model_copy skips validators: recompute the hash or the merged
                # entry keeps the OLD one and dedupe misfires against it.
                "content_hash": _compute_content_hash(content),
                "tags": merged_tags,
                "links": merged_links,
                "source_turns": merged_source_turns,
                "metadata": merged_metadata,
                "access_count": existing.access_count + 1,
                "updated_at": datetime.now(UTC),
                "relevance_score": max(existing.relevance_score, new_entry.relevance_score),
            }
        )

    def consolidate(
        self,
        new_entries: list[MemoryEntry],
        existing: list[MemoryEntry],
    ) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
        """Determine how each new entry relates to the existing memory store.

        Parameters:
            new_entries: Freshly extracted memory entries.
            existing: Already-stored memory entries.

        Returns:
            A list of ``(action, entry)`` tuples where *action* is one of:

            - ``MemoryOperation.ADD``  -- entry is new, append to store.
            - ``MemoryOperation.UPDATE`` -- entry is similar to an existing one,
              replace with the merged result.
            - ``MemoryOperation.NONE`` -- exact duplicate, skip.
        """
        existing_hashes = {e.content_hash for e in existing}
        existing_embeddings = [(e, self._get_embedding(e)) for e in existing]

        results: list[tuple[MemoryOperation, MemoryEntry | None]] = []

        for new_entry in new_entries:
            # 1. Exact content-hash deduplication
            if new_entry.content_hash in existing_hashes:
                results.append((MemoryOperation.NONE, None))
                continue

            # 2. Semantic similarity check
            new_emb = self._embed_fn(new_entry.content)
            self._embedding_cache[new_entry.id] = new_emb  # Cache for future consolidation rounds
            best_sim = 0.0
            best_existing: MemoryEntry | None = None

            for ex_entry, ex_emb in existing_embeddings:
                sim = _cosine_similarity(new_emb, ex_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_existing = ex_entry

            # 3. Merge or add
            if best_sim >= self._similarity_threshold and best_existing is not None:
                merged = self._merge_entries(new_entry, best_existing)
                results.append((MemoryOperation.UPDATE, merged))
            else:
                results.append((MemoryOperation.ADD, new_entry))

        return results
