"""Advanced reranker implementations for two-stage retrieval.

Provides multiple reranking strategies:
- CrossEncoderReranker: User-provided cross-encoder scoring callback.
- CohereReranker: Batch reranking via user-provided API callback.
- FlashRankReranker: Local reranking via flashrank (optional dependency).
- RoundRobinReranker: Merges results from multiple retrievers round-robin.
- RerankerPipeline: Chains multiple rerankers sequentially.

All implementations conform to the ``Reranker`` protocol defined in
``anchor.protocols.reranker``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.protocols.reranker import Reranker

if TYPE_CHECKING:
    from flashrank import Ranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks context items using a user-provided cross-encoder scoring function.

    The scoring function receives ``(query_str, document_content)`` and returns
    a relevance score (higher = more relevant).  Items are sorted by score
    descending and truncated to ``top_k``.

    Usage::

        def my_scorer(query: str, doc: str) -> float:
            return cross_encoder.predict(query, doc)

        reranker = CrossEncoderReranker(score_fn=my_scorer, top_k=10)
        results = reranker.rerank(query, items)

    Parameters:
        score_fn: Callable that takes (query_str, doc_content) and returns a float.
        top_k: Maximum number of items to return.
    """

    __slots__ = ("_score_fn", "_top_k")

    def __init__(
        self,
        score_fn: Callable[[str, str], float],
        top_k: int = 10,
    ) -> None:
        self._score_fn = score_fn
        self._top_k = top_k

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_k={self._top_k})"

    def rerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Rerank items using the cross-encoder scoring function.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return. Overrides the
                constructor ``top_k`` when provided.

        Returns:
            Reranked list of context items, truncated to ``top_k``.
        """
        if not items:
            return []

        effective_top_k = top_k if top_k is not None else self._top_k

        scored: list[tuple[float, ContextItem]] = []
        for item in items:
            score = self._score_fn(query.query_str, item.content)
            clamped = max(0.0, min(1.0, score))
            updated = item.model_copy(update={
                "score": clamped,
                "metadata": {**item.metadata, "raw_score": score},
            })
            scored.append((score, updated))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:effective_top_k]]


class CohereReranker:
    """Reranks context items using a batch reranking callback.

    The callback receives ``(query_str, documents, top_k)`` and returns
    a list of ``(original_index, score)`` tuples representing the ranked
    results.  The user provides the actual API call; this class handles
    mapping back to ``ContextItem`` objects.

    Usage::

        def cohere_rerank(query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
            response = co.rerank(query=query, documents=docs, top_n=top_k)
            return [(r.index, r.relevance_score) for r in response.results]

        reranker = CohereReranker(rerank_fn=cohere_rerank, top_k=10)
        results = reranker.rerank(query, items)

    Parameters:
        rerank_fn: Callable that takes (query, documents, top_k) and returns
            list of (index, score) tuples.
        top_k: Maximum number of items to return.
    """

    __slots__ = ("_rerank_fn", "_top_k")

    def __init__(
        self,
        rerank_fn: Callable[[str, list[str], int], list[tuple[int, float]]],
        top_k: int = 10,
    ) -> None:
        self._rerank_fn = rerank_fn
        self._top_k = top_k

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_k={self._top_k})"

    def rerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Rerank items using the batch reranking callback.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            Reranked list of context items with updated scores.
        """
        if not items:
            return []

        effective_top_k = top_k if top_k is not None else self._top_k
        documents = [item.content for item in items]

        ranked_results = self._rerank_fn(query.query_str, documents, effective_top_k)

        result: list[ContextItem] = []
        for idx, score in ranked_results:
            if 0 <= idx < len(items):
                clamped = max(0.0, min(1.0, score))
                updated = items[idx].model_copy(update={
                    "score": clamped,
                    "metadata": {**items[idx].metadata, "raw_score": score},
                })
                result.append(updated)

        return result[:effective_top_k]


class FlashRankReranker:
    """Reranks context items using flashrank for local cross-encoder inference.

    Requires the ``flashrank`` optional dependency:
    ``pip install anchor[flashrank]``

    The flashrank model is lazily loaded on first call to ``rerank()``.

    Parameters:
        model_name: The flashrank model to use.
        top_k: Maximum number of items to return.
    """

    __slots__ = ("_model_name", "_ranker", "_top_k")

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        top_k: int = 10,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._ranker: Ranker | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_name={self._model_name!r}, "
            f"top_k={self._top_k})"
        )

    def _get_ranker(self) -> Ranker:
        """Lazily initialize the flashrank Ranker.

        Returns:
            The flashrank Ranker instance.

        Raises:
            RetrieverError: If flashrank is not installed.
        """
        if self._ranker is not None:
            return self._ranker
        try:
            from flashrank import Ranker
        except ImportError as e:
            msg = (
                "flashrank is required for FlashRankReranker. "
                "Install it with: pip install anchor[flashrank]"
            )
            raise RetrieverError(msg) from e
        self._ranker = Ranker(model_name=self._model_name)
        return self._ranker

    def rerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Rerank items using the flashrank local cross-encoder.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            Reranked list of context items with updated scores.
        """
        if not items:
            return []

        try:
            from flashrank import RerankRequest
        except ImportError as e:
            msg = (
                "flashrank is required for FlashRankReranker. "
                "Install it with: pip install anchor[flashrank]"
            )
            raise RetrieverError(msg) from e

        effective_top_k = top_k if top_k is not None else self._top_k
        ranker = self._get_ranker()

        # Build passages with item IDs for mapping back
        passages: list[dict[str, Any]] = [
            {"id": item.id, "text": item.content}
            for item in items
        ]

        rerank_request = RerankRequest(query=query.query_str, passages=passages)
        ranked = ranker.rerank(rerank_request)

        # Build a lookup from item ID to original item
        item_map = {item.id: item for item in items}

        result: list[ContextItem] = []
        for entry in ranked[:effective_top_k]:
            item_id = entry["id"]
            score = float(entry["score"])
            if item_id in item_map:
                clamped = max(0.0, min(1.0, score))
                updated = item_map[item_id].model_copy(update={
                    "score": clamped,
                    "metadata": {**item_map[item_id].metadata, "raw_score": score},
                })
                result.append(updated)

        return result


class RoundRobinReranker:
    """Merges results from multiple result sets in round-robin fashion.

    Takes interleaved items from each result set, deduplicating by item ID.
    Also implements the standard ``rerank()`` as a pass-through that
    re-sorts items by existing score.

    Parameters:
        top_k: Maximum number of items to return.
    """

    __slots__ = ("_top_k",)

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_k={self._top_k})"

    def rerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Re-sort items by existing score descending.

        This is the standard Reranker protocol method.  For round-robin
        merging of multiple result sets, use ``rerank_multiple()``.

        Parameters:
            query: The query bundle (unused for simple re-sorting).
            items: Context items to re-sort by score.
            top_k: Maximum number of items to return.

        Returns:
            Items sorted by score descending, truncated to ``top_k``.
        """
        if not items:
            return []

        effective_top_k = top_k if top_k is not None else self._top_k
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
        return sorted_items[:effective_top_k]

    def rerank_multiple(
        self,
        query: QueryBundle,
        result_sets: list[list[ContextItem]],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Merge multiple result sets in round-robin order.

        Iterates through each result set, taking one item at a time from
        each in turn.  Duplicates (by item ID) are skipped.

        Parameters:
            query: The query bundle (available for future use).
            result_sets: List of result lists to merge.
            top_k: Maximum number of items to return. Defaults to
                the constructor ``top_k`` if ``None``.

        Returns:
            Merged list of context items in round-robin order.
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        if not result_sets:
            return []

        seen: set[str] = set()
        merged: list[ContextItem] = []
        max_len = max(len(rs) for rs in result_sets) if result_sets else 0

        for i in range(max_len):
            if len(merged) >= effective_top_k:
                break
            for result_set in result_sets:
                if len(merged) >= effective_top_k:
                    break
                if i < len(result_set):
                    item = result_set[i]
                    if item.id not in seen:
                        seen.add(item.id)
                        merged.append(item)

        return merged


class RerankerPipeline:
    """Chains multiple rerankers sequentially.

    Each reranker's output feeds into the next.  The final ``top_k``
    is applied at the end.

    Usage::

        pipeline = RerankerPipeline(
            rerankers=[cross_encoder, cohere_reranker],
            top_k=5,
        )
        results = pipeline.rerank(query, items)

    Parameters:
        rerankers: Ordered list of rerankers to apply.
        top_k: Maximum number of items to return from the final stage.
    """

    __slots__ = ("_rerankers", "_top_k")

    def __init__(
        self,
        rerankers: list[Reranker],
        top_k: int = 10,
    ) -> None:
        if not rerankers:
            msg = "At least one reranker is required for RerankerPipeline"
            raise ValueError(msg)
        self._rerankers = rerankers
        self._top_k = top_k

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rerankers={len(self._rerankers)}, "
            f"top_k={self._top_k})"
        )

    def rerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int | None = None,
    ) -> list[ContextItem]:
        """Rerank items through the chain of rerankers.

        Intermediate stages pass all items through (no truncation).
        The final ``top_k`` is applied at the end.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            Reranked list of context items from the final stage.
        """
        if not items:
            return []

        effective_top_k = top_k if top_k is not None else self._top_k
        current = items

        for reranker in self._rerankers:
            # Pass a large top_k to intermediate stages to avoid premature truncation
            current = reranker.rerank(query, current, top_k=len(current))

        return current[:effective_top_k]
