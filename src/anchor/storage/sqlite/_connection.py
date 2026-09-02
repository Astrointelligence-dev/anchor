"""SQLite connection manager with WAL mode and thread-local connections."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SqliteConnectionManager:
    """Manages SQLite connections with WAL mode and thread-local storage.

    Each thread gets its own connection to allow concurrent reads.
    WAL mode enables readers to proceed without blocking writers.

    The async connection is cached (one per manager instance) and shared
    across ``await get_async_connection()`` calls.  Call :meth:`aclose`
    to release it.

    Example::

        mgr = SqliteConnectionManager("data.db")
        conn = mgr.get_connection()
        conn.execute("SELECT 1")
        mgr.close()
    """

    __slots__ = ("_async_conn", "_db_path", "_local", "_wal_mode")

    def __init__(self, db_path: str | Path, *, wal_mode: bool = True) -> None:
        self._db_path = Path(db_path).resolve()
        self._wal_mode = wal_mode
        self._local = threading.local()
        self._async_conn: aiosqlite.Connection | None = None

    @property
    def db_path(self) -> Path:
        """Return the resolved database file path."""
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            if self._wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            logger.debug(
                "Opened SQLite connection to %s (thread=%s)",
                self._db_path,
                threading.current_thread().name,
            )
        return conn

    async def get_async_connection(self, _attempts: int = 3) -> aiosqlite.Connection:
        """Return a cached aiosqlite connection, creating one on first call.

        The connection is reused across calls.  Call :meth:`aclose` to
        release it when done.

        Raises ``ImportError`` if ``aiosqlite`` is not installed.
        """
        if self._async_conn is not None:
            return self._async_conn

        try:
            import aiosqlite as _aiosqlite
        except ImportError as e:
            msg = (
                "aiosqlite is required for async SQLite operations. "
                "Install it with: pip install astro-anchor[sqlite]"
            )
            raise ImportError(msg) from e

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await _aiosqlite.connect(str(self._db_path))
        try:
            conn.row_factory = _aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA foreign_keys=ON")
            if self._wal_mode:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
        except BaseException:
            # Never leak the worker thread. "database is locked" here is
            # the concurrent first-call race (the winner is switching the
            # file to WAL): back off and pick up the winner's connection.
            await conn.close()
            if _attempts <= 1:
                raise
            await asyncio.sleep(0)
            return await self.get_async_connection(_attempts - 1)
        if self._async_conn is not None:
            # Lost a concurrent first-call race: keep the winner and close
            # this connection, or its non-daemon worker thread leaks and
            # blocks interpreter shutdown.
            await conn.close()
            return self._async_conn
        self._async_conn = conn
        return conn

    def close(self) -> None:
        """Close the current thread's connection if it exists.

        .. note::
            This only closes the calling thread's connection.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    async def aclose(self) -> None:
        """Close the cached async connection if it exists."""
        if self._async_conn is not None:
            await self._async_conn.close()
            self._async_conn = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(db_path={self._db_path!s})"
