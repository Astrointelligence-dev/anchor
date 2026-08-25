"""Late interaction retrieval (ColBERT-style MaxSim scoring).

Provides token-level scoring and a two-stage retrieval pipeline that
uses a first-stage retriever for candidate generation followed by
fine-grained token-level re-scoring via MaxSim or a custom score function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from anchor._math import cosine_similarity as _cosine_sim
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.protocols.late_interaction import TokenLevelEncoder
from anchor.protocols.retriever import Retriever

logger = logging.getLogger(__name__)


class MaxSimScorer:
    """ColBERT-style MaxSim scorer for token-level embeddings.

    For each query token, finds the maximum cosine similarity across all
    document tokens and sums these maxima to produce the final score.
    """

    __slots__ = ()

    def score(
        self,
        query_tokens: list[list[float]],
        doc_tokens: list[list[float]],
    ) -> float:
        """Score a query-document pair using MaxSim.

        Parameters:
            query_tokens: Per-token embeddings for the query.
            doc_tokens: Per-token embeddings for the document.

        Returns:
            The sum of per-query-token maximum cosine similarities.
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        total = 0.0
        for q_tok in query_tokens:
            max_sim = max(_cosine_sim(q_tok, d_tok) for d_tok in doc_tokens)
            total += max_sim
        return total

    def __repr__(self) -> str:
        return "MaxSimScorer()"


class LateInteractionScorer:
    """Configurable late interaction scorer.

    Wraps a scoring function that operates on token-level embeddings.
    Defaults to ``MaxSimScorer`` when no custom function is provided.

    Parameters:
        score_fn: Optional callable taking (query_tokens, doc_tokens)
            and returning a float score.  Defaults to MaxSim.
    """

    __slots__ = ("_score_fn",)

    def __init__(
        self,
        score_fn: Callable[[list[list[float]], list[list[float]]], float] | None = None,
    ) -> None:
        self._score_fn = score_fn if score_fn is not None else MaxSimScorer().score

    def score(
        self,
        query_tokens: list[list[float]],
        doc_tokens: list[list[float]],
    ) -> float:
        """Score a query-document pair using the configured scoring function.

        Parameters:
            query_tokens: Per-token embeddings for the query.
            doc_tokens: Per-token embeddings for the document.

        Returns:
            A relevance score (higher is more relevant).
        """
        return self._score_fn(query_tokens, doc_tokens)

    def __repr__(self) -> str:
        return f"LateInteractionScorer(score_fn={self._score_fn!r})"


class LateInteractionRetriever:
    """Two-stage retriever with late interaction re-scoring.

    Uses a first-stage retriever to generate candidates, then re-scores
    each candidate using token-level embeddings (e.g. ColBERT MaxSim).

    Implements the ``Retriever`` protocol.

    Parameters:
        first_stage: A retriever used for candidate generation.
        encoder: A token-level encoder for producing per-token embeddings.
        scorer: Optional late interaction scorer.  Defaults to MaxSim.
        first_stage_k: Number of candidates to retrieve from the first stage.
    """

    __slots__ = ("_encoder", "_first_stage", "_first_stage_k", "_scorer")

    def __init__(
        self,
        first_stage: Retriever,
        encoder: TokenLevelEncoder,
        scorer: LateInteractionScorer | None = None,
        first_stage_k: int = 100,
    ) -> None:
        self._first_stage = first_stage
        self._encoder = encoder
        self._scorer = scorer if scorer is not None else LateInteractionScorer()
        self._first_stage_k = first_stage_k

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve and re-score items using late interaction.

        Parameters:
            query: The query bundle containing the user's query text.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects re-scored by token-level
            similarity, sorted by score descending.
        """
        candidates = self._first_stage.retrieve(query, top_k=self._first_stage_k)
        if not candidates:
            return []

        query_tokens = self._encoder.encode_tokens(query.query_str)

        scored: list[tuple[float, ContextItem]] = []
        for candidate in candidates:
            doc_tokens = self._encoder.encode_tokens(candidate.content)
            score = self._scorer.score(query_tokens, doc_tokens)
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def __repr__(self) -> str:
        return (
            f"LateInteractionRetriever(first_stage={self._first_stage!r}, "
            f"encoder={self._encoder!r}, scorer={self._scorer!r}, "
            f"first_stage_k={self._first_stage_k})"
        )
