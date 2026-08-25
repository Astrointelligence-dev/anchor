"""Sparse (BM25) retrieval.

Requires the 'bm25' extra: pip install anchor[bm25]
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import TYPE_CHECKING

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi


class SparseRetriever:
    """BM25-based retrieval over tokenized documents.

    Implements the Retriever protocol.
    """

    __slots__ = ("_bm25", "_items", "_min_score", "_tokenize_fn", "_tokenizer")

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]] | None = None,
        tokenizer: Tokenizer | None = None,
        min_score: float | None = None,
    ) -> None:
        self._tokenize_fn = tokenize_fn or self._default_tokenize
        self._bm25: BM25Okapi | None = None
        self._items: list[ContextItem] = []
        self._tokenizer = tokenizer or get_default_counter()
        self._min_score = min_score

    def __repr__(self) -> str:
        return (
            f"SparseRetriever(indexed_items={len(self._items)}, "
            f"bm25_ready={self._bm25 is not None})"
        )

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def index(self, items: list[ContextItem]) -> int:
        """Build the BM25 index from context items."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            msg = (
                "rank-bm25 is required for SparseRetriever. "
                "Install it with: pip install anchor[bm25]"
            )
            raise RetrieverError(msg) from e

        self._items = list(items)
        tokenized_corpus = [self._tokenize_fn(item.content) for item in self._items]
        self._bm25 = BM25Okapi(tokenized_corpus)
        return len(self._items)

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items using BM25 scoring."""
        if self._bm25 is None:
            msg = "Must call index() before retrieve()"
            raise RetrieverError(msg)

        tokenized_query = self._tokenize_fn(query.query_str)
        scores = self._bm25.get_scores(tokenized_query)

        if len(scores) == 0:
            return []

        raw_max = max(scores)
        max_score = raw_max if raw_max > 0 else 1.0

        # Use heapq.nlargest for O(N log K) instead of full sort O(N log N)
        # score field is max-normalized (ContextItem.score is [0,1]); the raw
        # BM25 score is preserved in metadata["raw_score"] for thresholding.
        scored_indices = [(float(s / max_score), float(s), i) for i, s in enumerate(scores)]
        top_entries = heapq.nlargest(top_k, scored_indices)

        items: list[ContextItem] = []
        for score, raw_score, idx in top_entries:
            if score <= 0:
                continue
            if self._min_score is not None and raw_score < self._min_score:
                continue
            item = self._items[idx]
            scored_item = item.model_copy(update={
                "source": SourceType.RETRIEVAL,
                "score": score,
                "token_count": item.token_count or self._tokenizer.count_tokens(item.content),
                "metadata": {
                    **item.metadata,
                    "retrieval_method": "sparse_bm25",
                    "raw_score": raw_score,
                },
            })
            items.append(scored_item)
        return items
