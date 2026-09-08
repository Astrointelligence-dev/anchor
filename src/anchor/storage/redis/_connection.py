"""Redis connection manager with sync and async support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import redis
    import redis.asyncio

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """Manages Redis connections for both sync and async operations.

    Uses redis-py's built-in connection pooling. The ``prefix`` parameter
    is used to namespace all keys, allowing multiple applications to share
    a single Redis instance.

    Example::

        mgr = RedisConnectionManager("redis://localhost:6379")
        client = mgr.get_client()
        client.set("key", "value")
        mgr.close()
    """

    __slots__ = ("_async_client", "_client", "_prefix", "_url")

    def __init__(self, url: str = "redis://localhost:6379", *, prefix: str = "anchor:") -> None:
        self._url = url
        self._prefix = prefix
        self._client: redis.Redis[Any] | None = None
        self._async_client: redis.asyncio.Redis[Any] | None = None

    @property
    def prefix(self) -> str:
        """Return the key prefix for this connection."""
        return self._prefix

    def get_client(self) -> redis.Redis[Any]:
        """Return a sync Redis client, creating one if needed."""
        if self._client is None:
            try:
                import redis as _redis
            except ImportError as e:
                msg = (
                    "redis is required for the Redis backend. "
                    "Install it with: pip install astro-anchor[redis]"
                )
                raise ImportError(msg) from e
            self._client = _redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    def get_async_client(self) -> redis.asyncio.Redis[Any]:
        """Return an async Redis client, creating one if needed."""
        if self._async_client is None:
            try:
                import redis.asyncio as _aredis
            except ImportError as e:
                msg = (
                    "redis is required for the Redis backend. "
                    "Install it with: pip install astro-anchor[redis]"
                )
                raise ImportError(msg) from e
            self._async_client = _aredis.Redis.from_url(self._url, decode_responses=True)
        return self._async_client

    def close(self) -> None:
        """Close sync client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close async client connection."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._prefix!r})"
