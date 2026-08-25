"""Reranker protocol definitions.

Any object with a ``rerank`` (or ``arerank``) method matching these
signatures can be used as a reranker in the pipeline -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle


@runtime_checkable
class Reranker(Protocol):
    """Protocol for synchronous reranking of retrieved results.

    Rerankers take a query and a list of candidate context items, score
    them for relevance, and return the top-k most relevant items in
    ranked order.
    """

    def rerank(
        self, query: QueryBundle, items: list[ContextItem], top_k: int | None = None
    ) -> list[ContextItem]:
        """Rerank context items by relevance to the query.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects ranked by relevance
            (most relevant first).  May return fewer than ``top_k``
            items if the input contains fewer candidates.
        """
        ...


@runtime_checkable
class AsyncReranker(Protocol):
    """Protocol for asynchronous reranking of retrieved results.

    Async rerankers are used with ``ContextPipeline.abuild()`` for
    non-blocking I/O during cross-encoder inference, API calls, etc.
    """

    async def arerank(
        self, query: QueryBundle, items: list[ContextItem], top_k: int | None = None
    ) -> list[ContextItem]:
        """Asynchronously rerank context items by relevance to the query.

        This is the async counterpart of ``Reranker.rerank``, intended
        for use with ``ContextPipeline.abuild()`` to enable non-blocking
        I/O during cross-encoder inference or external API calls.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects ranked by relevance
            (most relevant first).  May return fewer than ``top_k``
            items if the input contains fewer candidates.
        """
        ...
