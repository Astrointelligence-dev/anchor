"""Memory consolidation: embedding similarity and content hashing, or an LLM.

``SimilarityConsolidator`` (the default) decides by content hash and cosine
similarity and never calls a model. ``LLMConsolidator`` is the opt-in update
phase of the mem0 paper: cheap gates first, then one model call deciding
ADD / UPDATE / DELETE per new fact against the most similar memories.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from anchor._math import cosine_similarity as _cosine_similarity
from anchor._text import strip_markdown_fences
from anchor.llm.models import Message, Role
from anchor.models.memory import _compute_content_hash
from anchor.protocols.memory import MemoryOperation

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.models.memory import MemoryEntry

logger = logging.getLogger(__name__)

Outcome = tuple[MemoryOperation, "MemoryEntry | None"]


class _EmbeddingCache:
    """Embeddings keyed by content hash; wiped wholesale past *max_size*."""

    __slots__ = ("_cache", "_embed_fn", "_max_size")

    def __init__(self, embed_fn: Callable[[str], list[float]], max_size: int) -> None:
        self._embed_fn = embed_fn
        self._max_size = max_size
        self._cache: dict[str, list[float]] = {}

    def get(self, entry: MemoryEntry) -> list[float]:
        if entry.content_hash not in self._cache:
            if len(self._cache) >= self._max_size:
                self._cache.clear()
            self._cache[entry.content_hash] = self._embed_fn(entry.content)
        return self._cache[entry.content_hash]


def _merge_entries(
    new_entry: MemoryEntry, existing: MemoryEntry, *, content: str | None = None
) -> MemoryEntry:
    """Merge *new_entry* into *existing* under the existing id, keeping the richer metadata.

    *content* is the text of the result — by default the longer of the two.
    """
    merged_tags = list(dict.fromkeys([*existing.tags, *new_entry.tags]))
    merged_links = list(dict.fromkeys([*existing.links, *new_entry.links]))
    merged_source_turns = list(dict.fromkeys([*existing.source_turns, *new_entry.source_turns]))
    merged_metadata = {**existing.metadata, **new_entry.metadata}

    if content is None:
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

    __slots__ = ("_cache", "_similarity_threshold")

    _merge_entries = staticmethod(_merge_entries)

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            msg = "similarity_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)
        self._similarity_threshold = similarity_threshold
        self._cache = _EmbeddingCache(embed_fn, max_cache_size)

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
        existing_embeddings = [(e, self._cache.get(e)) for e in existing]

        results: list[tuple[MemoryOperation, MemoryEntry | None]] = []

        for new_entry in new_entries:
            # 1. Exact content-hash deduplication
            if new_entry.content_hash in existing_hashes:
                results.append((MemoryOperation.NONE, None))
                continue

            # 2. Semantic similarity check
            new_emb = self._cache.get(new_entry)
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


_DECISION_PROMPT = (
    "You maintain a user's long-term memory. For each NEW FACT, compare it with the "
    "EXISTING MEMORIES and choose one operation:\n"
    '- "add": new information — record it.\n'
    '- "update": the fact is about the same thing as an existing memory and changes or '
    "enriches it (a move, a changed preference, more detail): give \"target\" (the memory "
    'index) and the full rewritten "content" that replaces it — the current fact only; '
    "the memory keeps its previous text itself, so never restate what changed.\n"
    '- "delete": an existing memory stopped being true and nothing replaces it: give '
    '"target".\n'
    '- "none": the fact is already captured by an existing memory.\n'
    "Never update or delete on a guess: a fact about a different person, time or thing is "
    'an "add". Prefer "update" over "delete" whenever the new fact supersedes the old one. '
    "A fact you leave out is recorded as-is.\n"
    "Return ONLY a JSON array, one object per decision:\n"
    '[{{"fact": 0, "op": "update", "target": 2, "content": "User lives in Rio de Janeiro"}}]'
    "\n\nEXISTING MEMORIES:\n{existing}\n\nNEW FACTS:\n{facts}\n"
)


class _Decision(BaseModel):
    """One line of the model's answer."""

    fact: int
    op: Literal["add", "update", "delete", "none"]
    target: int | None = None
    content: str | None = None


def _decision(item: Any) -> _Decision | None:
    try:
        return _Decision.model_validate(item)
    except ValidationError as exc:
        logger.warning("LLM consolidation returned an unusable decision %r: %s", item, exc)
        return None


class LLMConsolidator:
    """The mem0-paper update phase: one model call decides ADD/UPDATE/DELETE per fact.

    Opt-in; the provider is injected (the agent's own). Cheap gates run
    first: an exact content-hash match is ``NONE`` without a call, and an
    empty store makes everything ``ADD``. With *embed_fn*, a fact whose best
    cosine against the store is below *new_threshold* is ``ADD`` without a
    call; the rest reach the model with the *top_k* most similar memories
    per fact — the *max_candidates* most recent ones when there is no
    embed_fn. High similarity never decides ``NONE`` on its own: cosine
    cannot tell a contradiction from a paraphrase (MemStrata, 2026), so
    that band is exactly where the model decides.

    Targets are indices into the candidates and are validated: an unknown
    ``update`` target degrades to ``ADD``, an unknown ``delete`` target is
    ignored, each with a warning; a fact the model leaves out is ``ADD``.
    ``UPDATE`` keeps the target's id, recomputes the content hash and
    records ``metadata["previous_content"]``; ``DELETE`` returns the target
    invalidated (``MemoryEntry.invalidate``). A malformed or failed response
    falls back to ``ADD`` for every fact — the behaviour without a
    consolidator — and logs a warning.
    """

    __slots__ = ("_cache", "_llm", "_max_candidates", "_new_threshold", "_top_k")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
        new_threshold: float = 0.3,
        top_k: int = 5,
        max_candidates: int = 20,
        max_cache_size: int = 1000,
    ) -> None:
        if not 0.0 <= new_threshold <= 1.0:
            msg = "new_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)
        if top_k <= 0 or max_candidates <= 0:
            msg = "top_k and max_candidates must be positive"
            raise ValueError(msg)
        self._llm = llm
        self._cache = _EmbeddingCache(embed_fn, max_cache_size) if embed_fn is not None else None
        self._new_threshold = new_threshold
        self._top_k = top_k
        self._max_candidates = max_candidates

    def consolidate(
        self,
        new_entries: list[MemoryEntry],
        existing: list[MemoryEntry],
    ) -> list[tuple[MemoryOperation, MemoryEntry | None]]:
        decided: dict[int, list[Outcome]] = {}
        hashes = {e.content_hash for e in existing}
        pending: list[int] = []
        for i, entry in enumerate(new_entries):
            if entry.content_hash in hashes:
                decided[i] = [(MemoryOperation.NONE, None)]
            elif not existing:
                decided[i] = [(MemoryOperation.ADD, entry)]
            else:
                pending.append(i)
        if pending:
            asked, candidates = self._select(new_entries, pending, existing)
            for i in pending:
                if i not in asked:
                    decided[i] = [(MemoryOperation.ADD, new_entries[i])]
            if asked:
                decided.update(self._ask(new_entries, asked, candidates))
        return [op for i in range(len(new_entries)) for op in decided[i]]

    def _select(
        self, new_entries: list[MemoryEntry], pending: list[int], existing: list[MemoryEntry]
    ) -> tuple[list[int], list[MemoryEntry]]:
        """Which pending facts need the model, and against which candidates."""
        if self._cache is None:
            recent = sorted(existing, key=lambda e: e.updated_at, reverse=True)
            return pending, recent[: self._max_candidates]
        vectors = [(e, self._cache.get(e)) for e in existing]
        asked: list[int] = []
        chosen: dict[str, MemoryEntry] = {}
        for i in pending:
            vec = self._cache.get(new_entries[i])
            ranked = sorted(
                ((_cosine_similarity(vec, ev), e) for e, ev in vectors),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if ranked[0][0] < self._new_threshold:
                continue  # nothing similar in the store: a plain ADD, no model call
            asked.append(i)
            for _, entry in ranked[: self._top_k]:
                chosen.setdefault(entry.id, entry)
        return asked, list(chosen.values())[: self._max_candidates]

    def _ask(
        self, new_entries: list[MemoryEntry], asked: list[int], candidates: list[MemoryEntry]
    ) -> dict[int, list[Outcome]]:
        facts = [new_entries[i] for i in asked]
        prompt = _DECISION_PROMPT.format(
            existing="\n".join(f"[{k}] {e.content}" for k, e in enumerate(candidates)),
            facts="\n".join(f"[{k}] {f.content}" for k, f in enumerate(facts)),
        )
        try:
            response = self._llm.invoke([Message(role=Role.USER, content=prompt)])
            data = json.loads(strip_markdown_fences(response.content or ""))
            if not isinstance(data, list):
                msg = "response is not a JSON array"
                raise TypeError(msg)
        except Exception as exc:
            logger.warning(
                "LLM consolidation failed, adding %d fact(s) as-is: %s", len(facts), exc
            )
            return {i: [(MemoryOperation.ADD, new_entries[i])] for i in asked}
        by_fact: dict[int, list[_Decision]] = {}
        for item in data:
            decision = _decision(item)
            if decision is None:
                continue
            if 0 <= decision.fact < len(facts):
                by_fact.setdefault(decision.fact, []).append(decision)
            else:
                logger.warning(
                    "LLM consolidation named fact %d of %d: ignored", decision.fact, len(facts)
                )
        return {
            i: self._apply(facts[k], by_fact.get(k, []), candidates) for k, i in enumerate(asked)
        }

    @staticmethod
    def _apply(
        fact: MemoryEntry, decisions: list[_Decision], candidates: list[MemoryEntry]
    ) -> list[Outcome]:
        if not decisions:
            return [(MemoryOperation.ADD, fact)]
        replaced = any(d.op in ("add", "update") for d in decisions)
        ops: list[Outcome] = []
        for d in decisions:
            if d.op == "none":
                ops.append((MemoryOperation.NONE, None))
                continue
            if d.op == "add":
                ops.append((MemoryOperation.ADD, fact))
                continue
            in_range = d.target is not None and 0 <= d.target < len(candidates)
            if not in_range:
                logger.warning(
                    "LLM consolidation %s names unknown memory %r: %s",
                    d.op,
                    d.target,
                    "added as-is" if d.op == "update" else "ignored",
                )
                if d.op == "update":
                    ops.append((MemoryOperation.ADD, fact))
                continue
            target = candidates[d.target]  # type: ignore[index]
            if d.op == "update":
                history = fact.model_copy(
                    update={"metadata": {**fact.metadata, "previous_content": target.content}}
                )
                merged = _merge_entries(history, target, content=d.content or fact.content)
                ops.append((MemoryOperation.UPDATE, merged))
            else:
                invalidated = target.invalidate(by=fact.id if replaced else None)
                ops.append((MemoryOperation.DELETE, invalidated))
        return ops

    def __repr__(self) -> str:
        return f"LLMConsolidator(llm={self._llm!r}, top_k={self._top_k})"
