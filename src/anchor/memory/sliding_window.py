"""Token-aware sliding window memory for conversation history."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import ConversationTurn, Role
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from anchor.protocols.memory import EvictionPolicy, RecencyScorer

logger = logging.getLogger(__name__)


class SlidingWindowMemory:
    """Maintains a rolling window of conversation turns within a token budget.

    When adding turns that would exceed the token limit, oldest turns are
    evicted first (FIFO) unless a custom ``eviction_policy`` is provided.
    Similarly, recency scores default to linear 0.5-1.0 unless a custom
    ``recency_scorer`` is supplied.
    """

    __slots__ = (
        "_eviction_policy",
        "_lock",
        "_max_tokens",
        "_on_evict",
        "_recency_scorer",
        "_tokenizer",
        "_total_tokens",
        "_turns",
    )

    def __init__(
        self,
        max_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
        on_evict: Callable[[list[ConversationTurn]], None] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        recency_scorer: RecencyScorer | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer or get_default_counter()
        self._on_evict = on_evict
        self._eviction_policy = eviction_policy
        self._recency_scorer = recency_scorer
        self._turns: deque[ConversationTurn] = deque()
        self._total_tokens: int = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(turns={len(self._turns)}, "
            f"tokens={self._total_tokens}/{self._max_tokens})"
        )

    @property
    def turns(self) -> list[ConversationTurn]:
        with self._lock:
            return list(self._turns)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def add_turn(self, role: Role, content: str, **metadata: object) -> ConversationTurn:
        """Add a conversation turn, evicting old turns if necessary."""
        token_count = self._tokenizer.count_tokens(content)
        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=token_count,
            metadata=dict(metadata),
        )

        # If this single turn exceeds the budget, truncate it
        if token_count > self._max_tokens:
            truncated = self._tokenizer.truncate_to_tokens(content, self._max_tokens)
            content = truncated
            token_count = self._max_tokens
            turn = ConversationTurn(
                role=role,
                content=content,
                token_count=token_count,
                metadata={**dict(metadata), "truncated": True},
            )

        with self._lock:
            # Evict turns until the new turn fits
            evicted_turns: list[ConversationTurn] = []
            if (
                self._eviction_policy is not None
                and self._turns
                and (self._total_tokens + turn.token_count > self._max_tokens)
            ):
                tokens_to_free = (self._total_tokens + turn.token_count) - self._max_tokens
                indices = self._eviction_policy.select_for_eviction(
                    list(self._turns), tokens_to_free
                )
                # Sort descending so we can pop from highest index first
                # to avoid index shifting
                for idx in sorted(indices, reverse=True):
                    evicted = self._turns[idx]
                    del self._turns[idx]
                    self._total_tokens -= evicted.token_count
                    evicted_turns.append(evicted)
                # Reverse so evicted_turns is in original order (oldest first)
                evicted_turns.reverse()
            else:
                # Default FIFO eviction
                while self._turns and (
                    self._total_tokens + turn.token_count > self._max_tokens
                ):
                    evicted = self._turns.popleft()
                    self._total_tokens -= evicted.token_count
                    evicted_turns.append(evicted)

            self._turns.append(turn)
            self._total_tokens += turn.token_count

        # Invoked OUTSIDE the lock: the callback may do slow work (LLM
        # summarization) or read this memory back — under the non-reentrant
        # lock that read self-deadlocked and the slow work serialized every
        # other thread on the window.
        if evicted_turns and self._on_evict is not None:
            try:
                self._on_evict(evicted_turns)
            except Exception:
                logger.exception(
                    "on_evict callback failed — ignoring to protect pipeline"
                )
        return turn

    def to_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Convert conversation turns to ContextItems for the pipeline.

        The content is stored as-is without a role prefix. The role is
        available in the item's metadata under the ``"role"`` key so that
        downstream formatters (Anthropic, OpenAI) can set the role via
        their own message structure and avoid the double-role bug
        (e.g. ``{"role": "user", "content": "user: Hello"}``).
        """
        with self._lock:
            items: list[ContextItem] = []
            num_turns = len(self._turns)
            for i, turn in enumerate(self._turns):
                # Recency-weighted score: use custom scorer or default linear 0.5-1.0
                if self._recency_scorer is not None:
                    recency_score = self._recency_scorer.score(i, num_turns)
                else:
                    recency_score = 0.5 + 0.5 * (i / max(1, num_turns - 1))
                item = ContextItem(
                    content=turn.content,
                    source=SourceType.CONVERSATION,
                    score=round(recency_score, 4),
                    priority=priority,
                    token_count=turn.token_count,
                    metadata={"role": turn.role, **turn.metadata},
                    created_at=turn.timestamp,
                )
                items.append(item)
            return items

    def clear(self) -> None:
        self._turns.clear()
        self._total_tokens = 0
