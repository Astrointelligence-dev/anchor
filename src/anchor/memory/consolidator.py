"""Memory consolidation: embedding similarity and content hashing, or an LLM.

``SimilarityConsolidator`` (the default) decides by content hash and cosine
similarity and never calls a model. ``LLMConsolidator`` is the opt-in update
phase of the mem0 paper: cheap gates first, then one model call deciding
ADD / UPDATE / DELETE per new fact against the most similar memories.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from anchor._math import cosine_similarity as _cosine_similarity
from anchor._text import ask_json
from anchor.memory.callbacks import MemoryCallback, _fire_memory_callback
from anchor.models.memory import _compute_content_hash
from anchor.protocols.memory import MemoryOperation

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.models.memory import MemoryEntry
    from anchor.protocols.memory import MemoryConsolidator
    from anchor.protocols.storage import MemoryEntryStore

logger = logging.getLogger(__name__)

Outcome = tuple[MemoryOperation, "MemoryEntry | None"]

EmbedFn = Callable[[str], list[float]]


def apply_consolidation(
    entries: list[MemoryEntry],
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None,
    *,
    callbacks: Sequence[MemoryCallback] = (),
) -> list[Outcome]:
    """Persist entries, optionally consolidating them against the store.

    With a *consolidator*, each ``(operation, entry)`` it returns is applied:
    ``ADD`` and ``UPDATE`` write the entry, ``DELETE`` invalidates it
    (``MemoryEntry.invalidate`` — a soft delete hidden from
    ``search``/``list_all`` and kept as history until the garbage collector's
    retention elapses), ``NONE`` does nothing. A ``DELETE`` whose target is
    not in the store is ignored with a warning — a target-less ``DELETE`` is
    a no-op. Without a consolidator every entry is added. Fires
    ``on_consolidation(action, new_entry, existing_entry)`` on *callbacks*
    and returns the operations applied.
    """
    if consolidator is None:
        for entry in entries:
            store.add(entry)
        return [(MemoryOperation.ADD, entry) for entry in entries]
    existing = store.list_all()
    by_id = {e.id: e for e in existing}
    applied: list[Outcome] = []
    for action, target in consolidator.consolidate(entries, existing):
        previous = by_id.get(target.id) if target is not None else None
        if target is not None and action in (MemoryOperation.ADD, MemoryOperation.UPDATE):
            store.add(target)
        elif target is not None and action == MemoryOperation.DELETE:
            if previous is None:
                logger.warning(
                    "consolidation DELETE names an entry not in the store: %s", target.id
                )
                continue
            if not target.is_expired:
                target = target.invalidate()
            store.add(target)
            previous, target = target, None
        applied.append((action, target if target is not None else previous))
        _fire_memory_callback(list(callbacks), "on_consolidation", str(action), target, previous)
    return applied


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

    __slots__ = ("_embed", "_similarity_threshold")

    _merge_entries = staticmethod(_merge_entries)

    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            msg = "similarity_threshold must be in [0.0, 1.0]"
            raise ValueError(msg)
        self._similarity_threshold = similarity_threshold
        self._embed = functools.lru_cache(maxsize=max_cache_size)(embed_fn)  # keyed by text

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
        existing_embeddings = [(e, self._embed(e.content)) for e in existing]

        results: list[tuple[MemoryOperation, MemoryEntry | None]] = []

        for new_entry in new_entries:
            # 1. Exact content-hash deduplication
            if new_entry.content_hash in existing_hashes:
                results.append((MemoryOperation.NONE, None))
                continue

            # 2. Semantic similarity check
            new_emb = self._embed(new_entry.content)
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


def _overlap(query: str, content: str) -> float:
    """Share of the query's words found in the content — the ScoredMemoryRetriever scorer."""
    terms = set(query.lower().split())
    if not terms:
        return 0.0
    lowered = content.lower()
    return sum(1 for term in terms if term in lowered) / len(terms)


class LLMConsolidator:
    """The mem0-paper update phase: one model call decides ADD/UPDATE/DELETE per fact.

    Opt-in; the provider is injected (the agent's own). Cheap gates run
    first: an exact content-hash match is ``NONE`` without a call, and an
    empty store makes everything ``ADD``. With *embed_fn*, a fact whose best
    cosine against the store is below *new_threshold* is ``ADD`` without a
    call; the rest reach the model with the *top_k* most similar memories
    per fact — ranked by keyword overlap (recency breaking ties) when there
    is no embed_fn — capped at *max_candidates* in total; a fact none of
    whose candidates made the cap is ``ADD`` without a call rather than
    asked blind. High similarity never decides ``NONE`` on its own: cosine
    cannot tell a contradiction from a paraphrase (MemStrata, 2026), so that
    band is exactly where the model decides.

    Targets are indices into the candidates and are validated: an unknown
    ``update`` target degrades to ``ADD``, an unknown ``delete`` target is
    ignored, each with a warning; a fact the model leaves out is ``ADD``.
    Decisions are applied in order against a working copy of the candidates,
    so two facts updating one memory chain, and a memory deleted in this
    batch is not touched again. ``UPDATE`` keeps the target's id, recomputes
    the content hash and records ``metadata["previous_content"]``;
    ``DELETE`` returns the target invalidated (``MemoryEntry.invalidate``)
    with ``invalidated_by`` = the id actually written for the superseding
    fact, when there is one. A malformed or failed response falls back to
    ``ADD`` for every fact — the behaviour without a consolidator — and
    logs a warning.
    """

    __slots__ = ("_embed", "_llm", "_max_candidates", "_new_threshold", "_top_k")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        embed_fn: EmbedFn | None = None,
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
        self._embed: EmbedFn | None = (
            functools.lru_cache(maxsize=max_cache_size)(embed_fn) if embed_fn is not None else None
        )
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

    def _rank(
        self, fact: MemoryEntry, existing: list[MemoryEntry]
    ) -> list[tuple[float, MemoryEntry]]:
        """Existing memories by similarity to *fact*, best first (recency breaks ties)."""
        if self._embed is None:
            score = functools.partial(_overlap, fact.content)
            scored = [(score(e.content), e) for e in existing]
        else:
            vec = self._embed(fact.content)
            scored = [(_cosine_similarity(vec, self._embed(e.content)), e) for e in existing]
        return sorted(scored, key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)

    def _select(
        self, new_entries: list[MemoryEntry], pending: list[int], existing: list[MemoryEntry]
    ) -> tuple[list[int], list[MemoryEntry]]:
        """Which pending facts need the model, and against which candidates."""
        top: dict[int, list[MemoryEntry]] = {}
        for i in pending:
            ranked = self._rank(new_entries[i], existing)
            if self._embed is not None and ranked[0][0] < self._new_threshold:
                continue  # nothing similar in the store: a plain ADD, no model call
            top[i] = [entry for _, entry in ranked[: self._top_k]]
        chosen: dict[str, MemoryEntry] = {}
        for entries in top.values():
            for entry in entries:
                if len(chosen) >= self._max_candidates:
                    break
                chosen.setdefault(entry.id, entry)
        asked = [i for i, entries in top.items() if any(e.id in chosen for e in entries)]
        return asked, list(chosen.values())

    def _ask(
        self, new_entries: list[MemoryEntry], asked: list[int], candidates: list[MemoryEntry]
    ) -> dict[int, list[Outcome]]:
        facts = [new_entries[i] for i in asked]
        prompt = _DECISION_PROMPT.format(
            existing="\n".join(f"[{k}] {e.content}" for k, e in enumerate(candidates)),
            facts="\n".join(f"[{k}] {f.content}" for k, f in enumerate(facts)),
        )
        data = ask_json(self._llm, prompt, log=logger, what="LLM consolidation failed")
        if data is None:
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
        batch = _Batch(candidates)  # working copies: decisions chain in order
        return {i: batch.apply(facts[k], by_fact.get(k, [])) for k, i in enumerate(asked)}

    def __repr__(self) -> str:
        return f"LLMConsolidator(llm={self._llm!r}, top_k={self._top_k})"


class _Batch:
    """One model answer applied decision by decision against live copies of the candidates.

    Two facts updating one memory chain; a memory deleted earlier in the
    batch is not touched again; ``invalidated_by`` names the id the
    superseding fact was actually written under.
    """

    __slots__ = ("_current", "_deleted")

    def __init__(self, candidates: list[MemoryEntry]) -> None:
        self._current = dict(enumerate(candidates))
        self._deleted: set[int] = set()

    def _target(self, d: _Decision) -> int | None:
        if d.target is None or d.target not in self._current or d.target in self._deleted:
            logger.warning(
                "LLM consolidation %s names unknown or deleted memory %r: %s",
                d.op,
                d.target,
                "added as-is" if d.op == "update" else "ignored",
            )
            return None
        return d.target

    def apply(self, fact: MemoryEntry, decisions: list[_Decision]) -> list[Outcome]:
        if not decisions:
            return [(MemoryOperation.ADD, fact)]
        ops: list[Outcome] = []
        survivor: str | None = None  # the id the fact ends up stored under, if any
        for d in decisions:
            if d.op == "none":
                ops.append((MemoryOperation.NONE, None))
            elif d.op == "add":
                ops.append((MemoryOperation.ADD, fact))
                survivor = survivor or fact.id
            elif d.op == "update":
                written = self._update(fact, d)
                ops.append(written)
                survivor = survivor or written[1].id  # type: ignore[union-attr]
        for d in decisions:
            if d.op == "delete" and (t := self._target(d)) is not None:
                self._deleted.add(t)
                ops.append((MemoryOperation.DELETE, self._current[t].invalidate(by=survivor)))
        return ops

    def _update(self, fact: MemoryEntry, d: _Decision) -> Outcome:
        t = self._target(d)
        if t is None:
            return (MemoryOperation.ADD, fact)
        target = self._current[t]
        history = fact.model_copy(
            update={"metadata": {**fact.metadata, "previous_content": target.content}}
        )
        merged = _merge_entries(history, target, content=d.content or fact.content)
        self._current[t] = merged
        return (MemoryOperation.UPDATE, merged)
