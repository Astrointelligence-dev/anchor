"""Query routers and routed retriever for directing queries to backends."""

from __future__ import annotations

import logging
from collections.abc import Callable

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope, scope_kwargs
from anchor.protocols.retriever import Retriever

logger = logging.getLogger(__name__)

__all__ = [
    "CallbackRouter",
    "KeywordRouter",
    "MetadataRouter",
    "RoutedRetriever",
]


class KeywordRouter:
    """Routes queries based on keyword matching.

    Checks if the query contains any of the configured keywords for
    each route. First match wins. Falls back to default route.

    Parameters:
        routes: Mapping of retriever name to list of trigger keywords.
        default: Default retriever name when no keywords match.
        case_sensitive: Whether keyword matching is case-sensitive.
    """

    __slots__ = ("_case_sensitive", "_default", "_routes")

    def __init__(
        self,
        routes: dict[str, list[str]],
        default: str,
        *,
        case_sensitive: bool = False,
    ) -> None:
        self._routes = routes
        self._default = default
        self._case_sensitive = case_sensitive

    def route(self, query: QueryBundle) -> str:
        """Determine which retriever should handle this query.

        Parameters:
            query: The query to route.

        Returns:
            The name/key of the target retriever.
        """
        text = query.query_str if self._case_sensitive else query.query_str.lower()
        for name, keywords in self._routes.items():
            for kw in keywords:
                check_kw = kw if self._case_sensitive else kw.lower()
                if check_kw in text:
                    logger.debug("KeywordRouter matched keyword %r -> route %r", kw, name)
                    return name
        logger.debug("KeywordRouter no match, using default %r", self._default)
        return self._default

    def __repr__(self) -> str:
        return (
            f"KeywordRouter(routes={len(self._routes)}, "
            f"default={self._default!r}, "
            f"case_sensitive={self._case_sensitive})"
        )


class CallbackRouter:
    """Routes queries using a user-provided callback function.

    Parameters:
        callback: A callable that takes a QueryBundle and returns a route name.
        default: Default route when callback returns None.
    """

    __slots__ = ("_callback", "_default")

    def __init__(
        self,
        callback: Callable[[QueryBundle], str | None],
        default: str = "default",
    ) -> None:
        self._callback = callback
        self._default = default

    def route(self, query: QueryBundle) -> str:
        """Determine which retriever should handle this query.

        Parameters:
            query: The query to route.

        Returns:
            The name/key of the target retriever.
        """
        result = self._callback(query)
        if result is None:
            logger.debug("CallbackRouter returned None, using default %r", self._default)
            return self._default
        return result

    def __repr__(self) -> str:
        return f"CallbackRouter(callback={self._callback!r}, default={self._default!r})"


class MetadataRouter:
    """Routes queries based on query metadata fields.

    Inspects ``query.metadata`` for a specified key and uses its value
    as the route name.

    Parameters:
        metadata_key: Key to look up in query.metadata. Default "route".
        default: Default route when key is missing.
    """

    __slots__ = ("_default", "_metadata_key")

    def __init__(
        self,
        metadata_key: str = "route",
        default: str = "default",
    ) -> None:
        self._metadata_key = metadata_key
        self._default = default

    def route(self, query: QueryBundle) -> str:
        """Determine which retriever should handle this query.

        Parameters:
            query: The query to route.

        Returns:
            The name/key of the target retriever.
        """
        value = query.metadata.get(self._metadata_key)
        if value is None:
            logger.debug(
                "MetadataRouter key %r not found, using default %r",
                self._metadata_key,
                self._default,
            )
            return self._default
        return str(value)

    def __repr__(self) -> str:
        return (
            f"MetadataRouter(metadata_key={self._metadata_key!r}, "
            f"default={self._default!r})"
        )


class RoutedRetriever:
    """Retriever that delegates to different backends based on query routing.

    Wraps a ``QueryRouter`` and a mapping of named retrievers.
    For each query, the router selects which retriever to use.

    Implements the ``Retriever`` protocol.

    Parameters:
        router: A QueryRouter to determine the target retriever.
        retrievers: Mapping of route names to Retriever implementations.
        default_retriever: Optional fallback retriever name if route not found.
    """

    __slots__ = ("_default_retriever", "_retrievers", "_router")

    def __init__(
        self,
        router: KeywordRouter | CallbackRouter | MetadataRouter,
        retrievers: dict[str, Retriever],
        default_retriever: str | None = None,
    ) -> None:
        self._router = router
        self._retrievers = retrievers
        self._default_retriever = default_retriever

    def retrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Route the query and delegate to the selected retriever (scope forwarded).

        Parameters:
            query: The query to route and retrieve for.
            top_k: Maximum number of items to return.

        Returns:
            A list of context items from the selected retriever.

        Raises:
            RetrieverError: If the route maps to an unknown retriever
                and no default is configured.
        """
        route_name = self._router.route(query)
        retriever = self._retrievers.get(route_name)

        if retriever is None and self._default_retriever is not None:
            logger.warning(
                "Route %r not found, falling back to default %r",
                route_name,
                self._default_retriever,
            )
            retriever = self._retrievers.get(self._default_retriever)

        if retriever is None:
            msg = (
                f"Route {route_name!r} maps to unknown retriever. "
                f"Available routes: {list(self._retrievers)}"
            )
            raise RetrieverError(msg)

        logger.debug("RoutedRetriever using route %r", route_name)
        return retriever.retrieve(query, top_k=top_k, **scope_kwargs(scope))

    def __repr__(self) -> str:
        return (
            f"RoutedRetriever(router={self._router!r}, "
            f"routes={list(self._retrievers)}, "
            f"default_retriever={self._default_retriever!r})"
        )
