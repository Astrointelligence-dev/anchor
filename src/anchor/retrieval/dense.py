"""Dense (embedding-based) retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anchor.embeddings._base import as_embedding_provider
from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.models.scope import ROOT_NAMESPACE, RetrievalScope, same_vault
from anchor.protocols.embeddings import EmbeddingProvider
from anchor.protocols.storage import ContextStore, VectorStore
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter


class DenseRetriever:
    """Retrieves context items via embedding similarity search.

    Requires a VectorStore backend and a ContextStore to resolve IDs to items.
    Embeddings come from an :class:`EmbeddingProvider` (or a bare callable,
    wrapped automatically) — anchor never calls an LLM directly.

    Implements the Retriever protocol.
    """

    __slots__ = (
        "_context_store",
        "_embeddings",
        "_min_score",
        "_tokenizer",
        "_vector_store",
    )

    def __init__(
        self,
        vector_store: VectorStore,
        context_store: ContextStore,
        embed_fn: Callable[[str], list[float]] | None = None,
        tokenizer: Tokenizer | None = None,
        min_score: float | None = None,
        *,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        same_vault(vector_store, context_store)
        self._vector_store = vector_store
        self._context_store = context_store
        self._embeddings = embeddings or as_embedding_provider(embed_fn)
        self._tokenizer = tokenizer or get_default_counter()
        self._min_score = min_score

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(vector_store={self._vector_store!r}, "
            f"context_store={self._context_store!r}, "
            f"embeddings={'set' if self._embeddings is not None else 'None'})"
        )

    def index(self, items: list[ContextItem]) -> int:
        """Index items into vector and context stores. Returns count indexed.

        Documents are embedded in one batch call, so API-backed providers
        make a single request instead of one per item.
        """
        if self._embeddings is None:
            msg = "An embedding provider (or embed_fn) must be provided to index items"
            raise RetrieverError(msg)
        if not items:
            return 0
        vectors = self._embeddings.embed_documents([item.content for item in items])
        if len(vectors) != len(items):
            msg = (
                f"Embedding provider returned {len(vectors)} vectors "
                f"for {len(items)} documents"
            )
            raise RetrieverError(msg)
        for item, embedding in zip(items, vectors, strict=True):
            # namespace only when set, so custom VectorStore implementations
            # written before front #3 keep working unchanged.
            if item.namespace != ROOT_NAMESPACE:
                self._vector_store.add_embedding(
                    item.id, embedding, item.metadata, namespace=item.namespace,
                )
            else:
                self._vector_store.add_embedding(item.id, embedding, item.metadata)
            self._context_store.add(item)
        return len(items)

    def retrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        where: dict[str, object] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Retrieve items most similar to the query embedding.

        Parameters:
            query: The query bundle (uses ``query.embedding`` when present).
            top_k: Maximum number of items to return.
            where: Optional metadata filter (equality or operator dicts),
                pushed down to the vector store (pre-filtering).
            scope: Optional namespace scope (include/exclude prefixes,
                exclude wins), pushed down to the vector store.
        """
        if query.embedding is not None:
            query_embedding = query.embedding
        elif self._embeddings is not None:
            query_embedding = self._embeddings.embed_query(query.query_str)
        else:
            msg = "Either provide query.embedding or configure an embedding provider"
            raise RetrieverError(msg)

        # Pass `where`/`scope` only when set, so custom VectorStore
        # implementations written before the filters existed keep working.
        kwargs: dict[str, Any] = {}
        if where is not None:
            kwargs["where"] = where
        if scope is not None:
            kwargs["scope"] = scope
        results = self._vector_store.search(query_embedding, top_k=top_k, **kwargs)
        items: list[ContextItem] = []
        for item_id, score in results:
            if self._min_score is not None and score < self._min_score:
                continue
            item = self._context_store.get(item_id)
            if item is not None:
                scored_item = item.model_copy(update={
                    "source": SourceType.RETRIEVAL,
                    "score": min(1.0, max(0.0, score)),
                    "token_count": item.token_count or self._tokenizer.count_tokens(item.content),
                    "metadata": {
                        **item.metadata,
                        "retrieval_method": "dense",
                        "raw_score": score,
                    },
                })
                items.append(scored_item)
        return items
