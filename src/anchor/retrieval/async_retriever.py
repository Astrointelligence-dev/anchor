"""Native async retriever implementations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from anchor._math import cosine_similarity
from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.retrieval._rrf import rrf_fuse

logger = logging.getLogger(__name__)


class AsyncDenseRetriever:
    """Async embedding-based retriever using cosine similarity.

    Uses a user-provided async embedding function to compute query embeddings
    and scores indexed items by cosine similarity against their stored
    embeddings. Items must have an ``"embedding"`` key in their metadata.

    Implements the ``AsyncRetriever`` protocol.

    Parameters:
        embed_fn: Async callable that takes a text string and returns
            a list of floats representing the embedding.
        similarity_fn: Optional callable for computing similarity between
            two embedding vectors. Defaults to cosine similarity.
    """

    __slots__ = ("_embed_fn", "_items", "_similarity_fn")

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        similarity_fn: Callable[[list[float], list[float]], float] | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._items: list[ContextItem] = []
        self._similarity_fn = similarity_fn or cosine_similarity

    def __repr__(self) -> str:
        return (
            f"AsyncDenseRetriever(items={len(self._items)}, "
            f"embed_fn={'set' if self._embed_fn is not None else 'None'})"
        )

    def index(self, items: list[ContextItem]) -> None:
        """Store items for retrieval. Items must have embeddings in metadata.

        Parameters:
            items: Context items to index. Each should have an ``"embedding"``
                key in its metadata containing the embedding vector.
        """
        self._items = list(items)

    async def aindex(self, items: list[ContextItem]) -> None:
        """Async index: embed items using embed_fn and store them.

        Parameters:
            items: Context items to index. Embeddings will be computed
                via the configured ``embed_fn`` and stored in metadata.
        """
        indexed: list[ContextItem] = []
        for item in items:
            if "embedding" not in item.metadata:
                embedding = await self._embed_fn(item.content)
                item = item.model_copy(
                    update={"metadata": {**item.metadata, "embedding": embedding}}
                )
            indexed.append(item)
        self._items = indexed

    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Asynchronously retrieve items most similar to the query.

        Parameters:
            query: The query bundle containing the user's query text.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects ranked by similarity
            (most similar first).
        """
        if not self._items:
            return []

        embedding = await self._embed_fn(query.query_str)

        scored: list[tuple[float, ContextItem]] = []
        for item in self._items:
            item_embedding = item.metadata.get("embedding")
            if item_embedding is None:
                continue
            score = self._similarity_fn(embedding, item_embedding)
            clamped = max(0.0, min(1.0, score))
            updated = item.model_copy(
                update={
                    "source": SourceType.RETRIEVAL,
                    "score": clamped,
                    "metadata": {
                        **item.metadata,
                        "retrieval_method": "async_dense",
                        "raw_score": score,
                    },
                }
            )
            scored.append((score, updated))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


class AsyncHybridRetriever:
    """Async hybrid retriever combining multiple async retrievers with RRF.

    Fans out to all sub-retrievers concurrently via ``asyncio.gather``
    and fuses results using Reciprocal Rank Fusion (RRF).

    Implements the ``AsyncRetriever`` protocol.

    Parameters:
        retrievers: List of async retrievers to combine.
        weights: Optional per-retriever weights for RRF scoring.
            Defaults to equal weights.
        k: RRF smoothing constant (default 60).
    """

    __slots__ = ("_k", "_retrievers", "_weights")

    def __init__(
        self,
        retrievers: list[AsyncDenseRetriever],
        weights: list[float] | None = None,
        k: int = 60,
    ) -> None:
        if not retrievers:
            msg = "At least one retriever is required"
            raise ValueError(msg)
        self._retrievers = retrievers
        self._k = k
        if weights is not None:
            if len(weights) != len(retrievers):
                msg = "weights must have same length as retrievers"
                raise ValueError(msg)
            self._weights = weights
        else:
            self._weights = [1.0] * len(retrievers)

    def __repr__(self) -> str:
        return (
            f"AsyncHybridRetriever(retrievers={len(self._retrievers)}, "
            f"k={self._k}, weights={self._weights})"
        )

    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Fan out to all retrievers concurrently and fuse with RRF.

        Parameters:
            query: The query bundle containing the user's query text.
            top_k: Maximum number of items to return.

        Returns:
            A fused list of ``ContextItem`` objects ranked by RRF score.
        """
        tasks = [r.aretrieve(query, top_k=top_k) for r in self._retrievers]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_rankings: list[list[ContextItem]] = []
        successful_weights: list[float] = []

        for result, weight in zip(all_results, self._weights, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Async sub-retriever failed, skipping: %s", result)
                continue
            all_rankings.append(result)
            successful_weights.append(weight)

        if not all_rankings:
            msg = "All sub-retrievers failed during hybrid retrieval"
            raise RetrieverError(msg)

        return rrf_fuse(
            all_rankings,
            weights=successful_weights,
            k=self._k,
            top_k=top_k,
            retrieval_method="async_hybrid_rrf",
        )
