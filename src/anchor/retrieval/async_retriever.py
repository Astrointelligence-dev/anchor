"""Native async retriever implementations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from anchor._math import cosine_similarity
from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.embeddings import EmbeddingProvider
from anchor.protocols.storage import AsyncContextStore, AsyncVectorStore
from anchor.retrieval._rrf import rrf_fuse

logger = logging.getLogger(__name__)


class AsyncDenseRetriever:
    """Async embedding-based retriever.

    Two operating modes:

    * **Store-backed** (pass ``vector_store`` + ``context_store``): search
      is delegated to an :class:`AsyncVectorStore` (e.g. pgvector via
      ``PostgresVectorStore``, or ``AsyncSqliteVectorStore``), including
      metadata ``where`` filtering.
    * **In-process** (default): items are held in a Python list with
      embeddings in their metadata and scored by cosine similarity.

    Implements the ``AsyncRetriever`` protocol.

    Parameters:
        embed_fn: Async callable ``(text) -> embedding``. Alternative to
            ``embeddings``.
        similarity_fn: Optional similarity for the in-process mode.
            Defaults to cosine similarity.
        embeddings: An :class:`EmbeddingProvider`; preferred over embed_fn.
        vector_store: Optional :class:`AsyncVectorStore` backend.
        context_store: Optional :class:`AsyncContextStore` used to resolve
            ids returned by the vector store. Required with vector_store.
        min_score: Optional raw-score threshold.
    """

    __slots__ = (
        "_context_store",
        "_embed_fn",
        "_embeddings",
        "_items",
        "_min_score",
        "_similarity_fn",
        "_vector_store",
    )

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        similarity_fn: Callable[[list[float], list[float]], float] | None = None,
        *,
        embeddings: EmbeddingProvider | None = None,
        vector_store: AsyncVectorStore | None = None,
        context_store: AsyncContextStore | None = None,
        min_score: float | None = None,
    ) -> None:
        if vector_store is not None and context_store is None:
            msg = "context_store is required when vector_store is provided"
            raise ValueError(msg)
        self._embed_fn = embed_fn
        self._embeddings = embeddings
        self._items: list[ContextItem] = []
        self._similarity_fn = similarity_fn or cosine_similarity
        self._vector_store = vector_store
        self._context_store = context_store
        self._min_score = min_score

    def __repr__(self) -> str:
        return (
            f"AsyncDenseRetriever(items={len(self._items)}, "
            f"store={'set' if self._vector_store is not None else 'None'})"
        )

    async def _embed_query(self, text: str) -> list[float]:
        if self._embeddings is not None:
            return await self._embeddings.aembed_query(text)
        if self._embed_fn is not None:
            return await self._embed_fn(text)
        msg = "Configure embeddings or embed_fn to embed queries"
        raise RetrieverError(msg)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._embeddings is not None:
            return await self._embeddings.aembed_documents(texts)
        if self._embed_fn is not None:
            return [await self._embed_fn(t) for t in texts]
        msg = "Configure embeddings or embed_fn to embed documents"
        raise RetrieverError(msg)

    def index(self, items: list[ContextItem]) -> None:
        """Store items for in-process retrieval (embeddings in metadata).

        Parameters:
            items: Context items to index. Each should have an ``"embedding"``
                key in its metadata containing the embedding vector.
        """
        self._items = list(items)

    async def aindex(self, items: list[ContextItem]) -> None:
        """Async index: embed items and store them.

        In store-backed mode, documents are embedded in one batch and
        written to the vector + context stores. In in-process mode,
        embeddings are stored in item metadata.
        """
        if self._vector_store is not None and self._context_store is not None:
            if not items:
                return
            vectors = await self._embed_documents([item.content for item in items])
            for item, embedding in zip(items, vectors, strict=True):
                await self._vector_store.add_embedding(
                    item.id, embedding, item.metadata
                )
                await self._context_store.add(item)
            return

        indexed: list[ContextItem] = []
        for item in items:
            if "embedding" not in item.metadata:
                embedding = (await self._embed_documents([item.content]))[0]
                item = item.model_copy(
                    update={"metadata": {**item.metadata, "embedding": embedding}}
                )
            indexed.append(item)
        self._items = indexed

    async def aretrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[ContextItem]:
        """Asynchronously retrieve items most similar to the query.

        Parameters:
            query: The query bundle containing the user's query text.
            top_k: Maximum number of items to return.
            where: Optional metadata equality filter (store-backed mode).

        Returns:
            A list of ``ContextItem`` objects ranked by similarity
            (most similar first).
        """
        if self._vector_store is not None and self._context_store is not None:
            return await self._aretrieve_from_store(query, top_k, where)

        if not self._items:
            return []

        embedding = (
            query.embedding
            if query.embedding is not None
            else await self._embed_query(query.query_str)
        )

        scored: list[tuple[float, ContextItem]] = []
        for item in self._items:
            item_embedding = item.metadata.get("embedding")
            if item_embedding is None:
                continue
            if where is not None and any(
                item.metadata.get(k) != v for k, v in where.items()
            ):
                continue
            score = self._similarity_fn(embedding, item_embedding)
            if self._min_score is not None and score < self._min_score:
                continue
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

    async def _aretrieve_from_store(
        self,
        query: QueryBundle,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[ContextItem]:
        if self._vector_store is None or self._context_store is None:  # pragma: no cover
            msg = "store-backed retrieval requires vector_store and context_store"
            raise RetrieverError(msg)
        embedding = (
            query.embedding
            if query.embedding is not None
            else await self._embed_query(query.query_str)
        )
        if where is not None:
            results = await self._vector_store.search(
                embedding, top_k=top_k, where=where
            )
        else:
            results = await self._vector_store.search(embedding, top_k=top_k)

        items: list[ContextItem] = []
        for item_id, score in results:
            if self._min_score is not None and score < self._min_score:
                continue
            item = await self._context_store.get(item_id)
            if item is not None:
                items.append(
                    item.model_copy(
                        update={
                            "source": SourceType.RETRIEVAL,
                            "score": min(1.0, max(0.0, score)),
                            "metadata": {
                                **item.metadata,
                                "retrieval_method": "async_dense",
                                "raw_score": score,
                            },
                        }
                    )
                )
        return items


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
