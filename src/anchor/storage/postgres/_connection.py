"""PostgreSQL connection manager using asyncpg connection pool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Manages an asyncpg connection pool for PostgreSQL.

    The pool must be initialized before use by calling ``initialize()``.
    Always call ``close()`` when done to release connections.

    Example::

        mgr = PostgresConnectionManager("postgresql://user:pass@localhost/db")
        await mgr.initialize()
        async with mgr.acquire() as conn:
            await conn.fetch("SELECT 1")
        await mgr.close()
    """

    __slots__ = ("_dsn", "_max_size", "_min_size", "_pool")

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]

    async def initialize(self, **kwargs: Any) -> None:
        """Create the connection pool. Must be called before any operations."""
        try:
            import asyncpg as _asyncpg
        except ImportError as e:
            msg = (
                "asyncpg is required for the PostgreSQL backend. "
                "Install it with: pip install astro-anchor[postgres]"
            )
            raise ImportError(msg) from e

        # Session default, not a per-query SET: the pool's RESET ALL on
        # release restores exactly the startup-packet settings. Filtered
        # HNSW searches (every vault-bound search) keep scanning until
        # LIMIT is met instead of stopping at ef_search candidates
        # (pgvector >= 0.8; older versions leave the placeholder unused).
        server_settings = {
            "hnsw.iterative_scan": "relaxed_order",
            **(kwargs.pop("server_settings", None) or {}),
        }
        self._pool = await _asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            server_settings=server_settings,
            **kwargs,
        )
        logger.info(
            "PostgreSQL pool initialized (min=%d, max=%d)",
            self._min_size,
            self._max_size,
        )

    def _get_pool(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        """Return the pool, raising if not initialized."""
        if self._pool is None:
            msg = "Connection pool not initialized. Call initialize() first."
            raise RuntimeError(msg)
        return self._pool

    def acquire(self) -> asyncpg.pool.PoolAcquireContext:  # type: ignore[type-arg]
        """Acquire a connection from the pool (use as async context manager)."""
        return self._get_pool().acquire()

    async def close(self) -> None:
        """Close the connection pool and release all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dsn=***)"
