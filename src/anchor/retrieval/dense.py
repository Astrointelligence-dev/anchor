"""Dense (embedding-based) retrieval."""

from __future__ import annotations

from collections.abc import Callable

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.storage import ContextStore, VectorStore
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter


class DenseRetriever:
    """Retrieves context items via embedding similarity search.

    Requires a VectorStore backend and a ContextStore to resolve IDs to items.
    The embed_fn is user-provided -- anchor never calls an LLM directly.

    Implements the Retriever protocol.
    """

    __slots__ = ("_context_store", "_embed_fn", "_min_score", "_tokenizer", "_vector_store")

    def __init__(
        self,
        vector_store: VectorStore,
        context_store: ContextStore,
        embed_fn: Callable[[str], list[float]] | None = None,
        tokenizer: Tokenizer | None = None,
        min_score: float | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._context_store = context_store
        self._embed_fn = embed_fn
        self._tokenizer = tokenizer or get_default_counter()
        self._min_score = min_score

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(vector_store={self._vector_store!r}, "
            f"context_store={self._context_store!r}, "
            f"embed_fn={'set' if self._embed_fn is not None else 'None'})"
        )

    def index(self, items: list[ContextItem]) -> int:
        """Index items into vector and context stores. Returns count indexed."""
        if self._embed_fn is None:
            msg = "embed_fn must be provided to index items"
            raise RetrieverError(msg)
        for item in items:
            embedding = self._embed_fn(item.content)
            self._vector_store.add_embedding(item.id, embedding, item.metadata)
            self._context_store.add(item)
        return len(items)

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items most similar to the query embedding."""
        if query.embedding is not None:
            query_embedding = query.embedding
        elif self._embed_fn is not None:
            query_embedding = self._embed_fn(query.query_str)
        else:
            msg = "Either provide query.embedding or set embed_fn on the retriever"
            raise RetrieverError(msg)

        results = self._vector_store.search(query_embedding, top_k=top_k)
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
