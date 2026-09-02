"""Retriever protocol definitions.

Any object with a `retrieve` (or `aretrieve`) method matching these
signatures can be used as a retriever in the pipeline -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope


@runtime_checkable
class Retriever(Protocol):
    """Protocol for synchronous retrieval strategies."""

    def retrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Retrieve the most relevant context items for a query.

        Parameters:
            query: The query bundle containing the user's query text and
                any associated metadata used for retrieval.
            top_k: Maximum number of items to return.
            scope: Namespace scope (include/exclude prefixes, exclude
                wins). Only items whose namespace is visible under it may
                be returned; a retriever that cannot honor it must raise.

        Returns:
            A list of ``ContextItem`` objects ranked by relevance
            (most relevant first).  May return fewer than ``top_k``
            items if the store contains fewer candidates.
        """
        ...


@runtime_checkable
class AsyncRetriever(Protocol):
    """Protocol for asynchronous retrieval strategies.

    Async retrievers are used with ``ContextPipeline.abuild()`` for
    non-blocking I/O during embedding lookups, database queries, etc.
    """

    async def aretrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Asynchronously retrieve the most relevant context items for a query.

        This is the async counterpart of ``Retriever.retrieve``, intended
        for use with ``ContextPipeline.abuild()`` to enable non-blocking
        I/O during embedding lookups, database queries, or API calls.

        Parameters:
            query: The query bundle containing the user's query text and
                any associated metadata used for retrieval.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects ranked by relevance
            (most relevant first).  May return fewer than ``top_k``
            items if the store contains fewer candidates.
        """
        ...
