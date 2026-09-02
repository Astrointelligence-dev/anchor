"""Hybrid retrieval combining multiple retrievers with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope, scope_kwargs
from anchor.protocols.retriever import Retriever
from anchor.retrieval._rrf import rrf_fuse

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF).

    RRF score for document d across N ranking lists:
        RRF(d) = sum(weight_i / (rrf_k + rank_i(d))) for each list i

    The smoothing constant rrf_k (default 60) prevents top-ranked items
    from dominating excessively. This is the standard value from
    the original RRF paper.

    Implements the Retriever protocol.
    """

    __slots__ = ("_retrievers", "_rrf_k", "_weights")

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = 60,
        weights: list[float] | None = None,
    ) -> None:
        if not retrievers:
            msg = "At least one retriever is required"
            raise ValueError(msg)
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        if weights is not None:
            if len(weights) != len(retrievers):
                msg = "weights must have same length as retrievers"
                raise ValueError(msg)
            self._weights = weights
        else:
            self._weights = [1.0] * len(retrievers)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(retrievers={len(self._retrievers)}, "
            f"rrf_k={self._rrf_k}, weights={self._weights})"
        )

    def retrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Retrieve from all sub-retrievers (scope forwarded) and fuse with RRF."""
        all_rankings: list[list[ContextItem]] = []
        successful_weights: list[float] = []
        for retriever, weight in zip(self._retrievers, self._weights, strict=True):
            try:
                results = retriever.retrieve(
                    query, top_k=top_k, **scope_kwargs(scope),
                )
                all_rankings.append(results)
                successful_weights.append(weight)
            except Exception as exc:
                logger.warning(
                    "Sub-retriever %r failed, skipping: %s",
                    retriever, exc,
                )
                continue

        if not all_rankings:
            msg = "All sub-retrievers failed during hybrid retrieval"
            raise RetrieverError(msg)

        return rrf_fuse(
            all_rankings,
            weights=successful_weights,
            k=self._rrf_k,
            top_k=top_k,
            retrieval_method="hybrid_rrf",
        )
