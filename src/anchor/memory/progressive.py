"""Multi-tier progressive summarization memory.

``ProgressiveSummarizationMemory`` implements a 4-tier cascading compression
system with LLM-driven compaction and key-fact extraction. It satisfies the
``ConversationMemory`` protocol and integrates with ``MemoryManager``.

Tiers:
    0 — Verbatim recent turns (``SlidingWindowMemory``)
    1 — Detailed summary (~500 tokens)
    2 — Compact summary (~100 tokens)
    3 — Ultra-compact summary (~20 tokens)

Key facts are extracted at each tier transition and stored in a sidecar.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from anchor.memory.compactor import TierCompactor
from anchor.memory.sliding_window import SlidingWindowMemory
from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import (
    ConversationTurn,
    KeyFact,
    Role,
    SummaryTier,
    TierConfig,
)
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.memory.callbacks import ProgressiveSummarizationCallback
    from anchor.protocols.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_DEFAULT_TIER_CONFIG = [
    TierConfig(level=0, max_tokens=4096, target_tokens=0, priority=7),
    TierConfig(level=1, max_tokens=1024, target_tokens=500, priority=6),
    TierConfig(level=2, max_tokens=256, target_tokens=100, priority=5),
    TierConfig(level=3, max_tokens=64, target_tokens=20, priority=4),
]


def _derive_tier_config(max_tokens: int) -> list[TierConfig]:
    """Derive tier config proportionally from a total token budget."""
    # 50% verbatim, 25% detailed, 12.5% compact, 6.25% ultra + ~6.25% buffer
    return [
        TierConfig(level=0, max_tokens=max_tokens // 2, target_tokens=0, priority=7),
        TierConfig(level=1, max_tokens=max_tokens // 4, target_tokens=max_tokens // 8, priority=6),
        TierConfig(level=2, max_tokens=max_tokens // 8, target_tokens=max_tokens // 16, priority=5),
        TierConfig(level=3, max_tokens=max_tokens // 16, target_tokens=max_tokens // 32, priority=4),
    ]


def _serialize_turns(turns: list[ConversationTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


class ProgressiveSummarizationMemory:
    """A 4-tier progressive summarization memory.

    Satisfies the ``ConversationMemory`` protocol. Pluggable into
    ``MemoryManager`` and ``ContextPipeline``.
    """

    __slots__ = (
        "_aevict_lock",
        "_callbacks",
        "_compactor",
        "_fact_token_budget",
        "_facts",
        "_lock",
        "_max_facts",
        "_tier_configs",
        "_tiers",
        "_tokenizer",
        "_window",
    )

    def __init__(
        self,
        max_tokens: int = 8192,
        llm: LLMProvider | str = "anthropic/claude-haiku-4-5-20251001",
        tier_config: list[TierConfig] | None = None,
        max_facts: int = 50,
        fact_token_budget: int = 500,
        tokenizer: Tokenizer | None = None,
        callbacks: list[ProgressiveSummarizationCallback] | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)

        self._tokenizer: Tokenizer = tokenizer or get_default_counter()
        self._max_facts = max_facts
        self._fact_token_budget = fact_token_budget
        self._facts: list[KeyFact] = []
        self._callbacks: list[ProgressiveSummarizationCallback] = callbacks or []
        self._lock = threading.RLock()
        # Serializes the async eviction cascade: its read-summarize-write
        # spans awaits, so without this two concurrent evictions read the
        # same tier state and the second write erases the first's turns.
        self._aevict_lock = asyncio.Lock()

        # Resolve LLM provider
        if isinstance(llm, str):
            from anchor.llm.registry import create_provider
            resolved_llm = create_provider(llm)
        else:
            resolved_llm = llm

        self._compactor = TierCompactor(llm=resolved_llm, tokenizer=self._tokenizer)

        # Configure tiers
        self._tier_configs: dict[int, TierConfig] = {}
        configs = tier_config or _derive_tier_config(max_tokens)
        for cfg in configs:
            self._tier_configs[cfg.level] = cfg

        # Tier 0 = SlidingWindowMemory
        tier0_config = self._tier_configs.get(0)
        tier0_tokens = tier0_config.max_tokens if tier0_config else max_tokens // 2
        self._window = SlidingWindowMemory(
            max_tokens=tier0_tokens,
            tokenizer=self._tokenizer,
            on_evict=self._handle_eviction,
        )

        # Tiers 1-3 stored as SummaryTier or None
        self._tiers: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}

    def __repr__(self) -> str:
        tier_summary = {k: (v.token_count if v else 0) for k, v in self._tiers.items()}
        return (
            f"ProgressiveSummarizationMemory("
            f"window_tokens={self._window.total_tokens}, "
            f"tiers={tier_summary}, "
            f"facts={len(self._facts)})"
        )

    # -- ConversationMemory protocol --

    @property
    def turns(self) -> list[ConversationTurn]:
        return self._window.turns

    @property
    def total_tokens(self) -> int:
        return self._window.total_tokens

    def to_context_items(self, priority: int = 7) -> list[ContextItem]:
        with self._lock:
            items: list[ContextItem] = []

            # Tier 3 (lowest priority summary)
            for tier_level in (3, 2, 1):
                tier = self._tiers.get(tier_level)
                if tier is not None:
                    # tier 1 = priority-1, tier 2 = priority-2, tier 3 = priority-3
                    tier_priority = priority - tier_level
                    items.append(
                        ContextItem(
                            content=tier.content,
                            source=SourceType.CONVERSATION,
                            score=0.5,
                            priority=tier_priority,
                            token_count=tier.token_count,
                            metadata={"tier": tier_level, "summary": True},
                        )
                    )

            # Tier 0 verbatim turns
            items.extend(self._window.to_context_items(priority=priority))

            # Key facts (highest priority)
            fact_priority = priority + 1
            for fact in self._facts:
                items.append(
                    ContextItem(
                        content=fact.content,
                        source=SourceType.MEMORY,
                        score=0.8,
                        priority=fact_priority,
                        token_count=fact.token_count,
                        metadata={
                            "fact_type": fact.fact_type.value,
                            "source_tier": fact.source_tier,
                        },
                    )
                )

            return items

    def clear(self) -> None:
        with self._lock:
            self._window.clear()
            self._tiers = {1: None, 2: None, 3: None}
            self._facts = []

    # -- Public methods --

    def add_message(
        self, role: Role | str, content: str, **metadata: object
    ) -> ConversationTurn:
        with self._lock:
            return self._window.add_turn(role, content, **metadata)

    def add_turn(self, turn: ConversationTurn) -> None:
        with self._lock:
            self._window.add_turn(
                role=turn.role, content=turn.content, **turn.metadata
            )

    async def aadd_message(
        self, role: Role | str, content: str, **metadata: object
    ) -> ConversationTurn:
        """Async add: collects evicted turns, then runs async LLM compaction."""
        evicted: list[ConversationTurn] = []
        # Temporarily swap callback to capture evicted turns instead of sync
        # compaction. The WHOLE swap-add-restore sequence holds the RLock —
        # with the swap outside it, a concurrent caller could read the
        # temporary lambda as its "original" and restore it permanently,
        # silently killing all future evictions. Still couples to
        # SlidingWindowMemory._on_evict (a private attribute).
        with self._lock:
            original_cb = self._window._on_evict
            self._window._on_evict = lambda turns: evicted.extend(turns)
            try:
                turn = self._window.add_turn(role, content, **metadata)
            finally:
                self._window._on_evict = original_cb
        # Perform async cascade outside the lock
        if evicted:
            await self._handle_eviction_async(evicted)
        return turn

    async def aadd_turn(self, turn: ConversationTurn) -> None:
        """Async add_turn: delegates to :meth:`aadd_message`."""
        await self.aadd_message(turn.role, turn.content, **turn.metadata)

    # -- Introspection --

    @property
    def tiers(self) -> dict[int, SummaryTier | None]:
        return dict(self._tiers)

    @property
    def facts(self) -> list[KeyFact]:
        return list(self._facts)

    @property
    def tier_tokens(self) -> dict[int, int]:
        result: dict[int, int] = {0: self._window.total_tokens}
        for level, tier in self._tiers.items():
            result[level] = tier.token_count if tier is not None else 0
        return result

    @property
    def summary(self) -> str | None:
        tier1 = self._tiers.get(1)
        return tier1.content if tier1 is not None else None

    # -- Eviction handling (called from within SlidingWindowMemory's lock) --

    def _handle_eviction(self, evicted_turns: list[ConversationTurn]) -> None:
        """Callback invoked by SlidingWindowMemory when turns are evicted.

        This runs inside the outer RLock (acquired by add_message/add_turn).
        It must NOT call back into the SlidingWindowMemory.
        """
        serialized = _serialize_turns(evicted_turns)
        turn_count = len(evicted_turns)

        # Summarize into Tier 1
        existing_t1 = self._tiers.get(1)
        existing_summary = existing_t1.content if existing_t1 else None
        existing_count = existing_t1.source_turn_count if existing_t1 else 0

        t1_config = self._tier_configs.get(1, TierConfig(level=1, max_tokens=1024, target_tokens=500))

        try:
            new_summary = self._compactor.summarize(
                serialized,
                target_tier=1,
                target_tokens=t1_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception as exc:
            logger.exception("Tier 1 compaction failed; using raw content")
            new_summary = serialized
            self._fire_callback("on_compaction_error", 1, exc)

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        self._tiers[1] = SummaryTier(
            level=1,
            content=new_summary,
            token_count=summary_tokens,
            source_turn_count=existing_count + turn_count,
            created_at=existing_t1.created_at if existing_t1 else now,
            updated_at=now,
        )

        self._fire_callback(
            "on_tier_cascade", 0, 1, self._tokenizer.count_tokens(serialized), summary_tokens
        )

        # Extract facts
        try:
            new_facts = self._compactor.extract_facts(serialized, source_tier=0)
            if new_facts:
                self._add_facts(new_facts)
                self._fire_callback("on_facts_extracted", new_facts, 0)
        except Exception:
            logger.warning("Fact extraction failed during tier 0→1 cascade")

        # Check if Tier 1 needs to cascade to Tier 2
        if summary_tokens > t1_config.max_tokens:
            self._cascade_tier(1, 2)

    def _cascade_tier(self, from_level: int, to_level: int) -> None:
        """Cascade content from one tier to the next."""
        source_tier = self._tiers.get(from_level)
        if source_tier is None:
            return

        to_config = self._tier_configs.get(
            to_level, TierConfig(level=to_level, max_tokens=64, target_tokens=20)
        )
        existing_target = self._tiers.get(to_level)
        existing_summary = existing_target.content if existing_target else None
        existing_count = existing_target.source_turn_count if existing_target else 0

        try:
            new_summary = self._compactor.summarize(
                source_tier.content,
                target_tier=to_level,
                target_tokens=to_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception as exc:
            logger.exception("Tier %d→%d compaction failed", from_level, to_level)
            new_summary = source_tier.content
            self._fire_callback("on_compaction_error", to_level, exc)

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        self._tiers[to_level] = SummaryTier(
            level=to_level,
            content=new_summary,
            token_count=summary_tokens,
            source_turn_count=existing_count + source_tier.source_turn_count,
            created_at=existing_target.created_at if existing_target else now,
            updated_at=now,
        )

        # Extract facts from the source tier content (not for tier 2→3)
        if from_level < 2:
            try:
                new_facts = self._compactor.extract_facts(source_tier.content, source_tier=from_level)
                if new_facts:
                    self._add_facts(new_facts)
                    self._fire_callback("on_facts_extracted", new_facts, from_level)
            except Exception:
                logger.warning("Fact extraction failed during tier %d→%d cascade", from_level, to_level)

        # Clear the source tier after cascading
        self._tiers[from_level] = None

        self._fire_callback(
            "on_tier_cascade", from_level, to_level, source_tier.token_count, summary_tokens
        )

        # Check if target needs to cascade further
        if to_level < 3 and summary_tokens > to_config.max_tokens:
            self._cascade_tier(to_level, to_level + 1)

    def _add_facts(self, new_facts: list[KeyFact]) -> None:
        """Add facts to the store, respecting max_facts and token budget."""
        for fact in new_facts:
            self._facts.append(fact)

        # FIFO eviction if over max_facts
        while len(self._facts) > self._max_facts:
            self._facts.pop(0)

        # Token budget enforcement
        total_fact_tokens = sum(f.token_count for f in self._facts)
        while self._facts and total_fact_tokens > self._fact_token_budget:
            removed = self._facts.pop(0)
            total_fact_tokens -= removed.token_count

    def _fire_callback(self, method: str, *args: object) -> None:
        """Fire a callback method on all registered callbacks."""
        from anchor._callbacks import fire_callbacks
        fire_callbacks(self._callbacks, method, *args, logger=logger, log_level=logging.WARNING)

    # -- Async eviction handling (used by aadd_message / aadd_turn) --

    async def _handle_eviction_async(self, evicted_turns: list[ConversationTurn]) -> None:
        """Async variant of _handle_eviction using async LLM calls.

        The whole cascade is serialized by ``_aevict_lock``: the
        read-summarize-write sequence spans awaits, and unserialized
        concurrent evictions lose turns (both read the same tier state,
        second write wins). ``_cascade_tier_async`` recursion runs inside
        the held lock — it must never re-acquire it.
        """
        # ponytail: sync evictions (RLock only) can still interleave with an
        # async cascade mid-await; single-mode usage is the supported shape.
        async with self._aevict_lock:
            await self._handle_eviction_async_locked(evicted_turns)

    async def _handle_eviction_async_locked(
        self, evicted_turns: list[ConversationTurn],
    ) -> None:
        serialized = _serialize_turns(evicted_turns)
        turn_count = len(evicted_turns)

        with self._lock:
            existing_t1 = self._tiers.get(1)
            existing_summary = existing_t1.content if existing_t1 else None
            existing_count = existing_t1.source_turn_count if existing_t1 else 0
            t1_config = self._tier_configs.get(1, TierConfig(level=1, max_tokens=1024, target_tokens=500))

        try:
            new_summary = await self._compactor.asummarize(
                serialized, target_tier=1, target_tokens=t1_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception as exc:
            logger.exception("Async tier 1 compaction failed; using raw content")
            new_summary = serialized
            self._fire_callback("on_compaction_error", 1, exc)

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        with self._lock:
            self._tiers[1] = SummaryTier(
                level=1, content=new_summary, token_count=summary_tokens,
                source_turn_count=existing_count + turn_count,
                created_at=existing_t1.created_at if existing_t1 else now, updated_at=now,
            )
        self._fire_callback("on_tier_cascade", 0, 1, self._tokenizer.count_tokens(serialized), summary_tokens)

        try:
            new_facts = await self._compactor.aextract_facts(serialized, source_tier=0)
            if new_facts:
                with self._lock:
                    self._add_facts(new_facts)
                self._fire_callback("on_facts_extracted", new_facts, 0)
        except Exception:
            logger.warning("Async fact extraction failed during tier 0→1 cascade")

        if summary_tokens > t1_config.max_tokens:
            await self._cascade_tier_async(1, 2)

    async def _cascade_tier_async(self, from_level: int, to_level: int) -> None:
        """Async variant of _cascade_tier."""
        with self._lock:
            source_tier = self._tiers.get(from_level)
            if source_tier is None:
                return

            to_config = self._tier_configs.get(to_level, TierConfig(level=to_level, max_tokens=64, target_tokens=20))
            existing_target = self._tiers.get(to_level)
            existing_summary = existing_target.content if existing_target else None
            existing_count = existing_target.source_turn_count if existing_target else 0

        try:
            new_summary = await self._compactor.asummarize(
                source_tier.content, target_tier=to_level, target_tokens=to_config.target_tokens,
                existing_summary=existing_summary,
            )
        except Exception as exc:
            logger.exception("Async tier %d→%d compaction failed", from_level, to_level)
            new_summary = source_tier.content
            self._fire_callback("on_compaction_error", to_level, exc)

        summary_tokens = self._tokenizer.count_tokens(new_summary)
        now = datetime.now(UTC)

        with self._lock:
            self._tiers[to_level] = SummaryTier(
                level=to_level, content=new_summary, token_count=summary_tokens,
                source_turn_count=existing_count + source_tier.source_turn_count,
                created_at=existing_target.created_at if existing_target else now, updated_at=now,
            )

        if from_level < 2:
            try:
                new_facts = await self._compactor.aextract_facts(source_tier.content, source_tier=from_level)
                if new_facts:
                    with self._lock:
                        self._add_facts(new_facts)
                    self._fire_callback("on_facts_extracted", new_facts, from_level)
            except Exception:
                logger.warning("Async fact extraction failed during tier %d→%d cascade", from_level, to_level)

        with self._lock:
            self._tiers[from_level] = None
        self._fire_callback("on_tier_cascade", from_level, to_level, source_tier.token_count, summary_tokens)

        if to_level < 3 and summary_tokens > to_config.max_tokens:
            await self._cascade_tier_async(to_level, to_level + 1)
