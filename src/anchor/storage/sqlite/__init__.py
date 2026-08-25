"""SQLite persistent storage backend.

Install with: pip install astro-anchor[sqlite]
"""

from anchor.storage.sqlite._connection import SqliteConnectionManager
from anchor.storage.sqlite._context_store import (
    AsyncSqliteContextStore,
    SqliteContextStore,
)
from anchor.storage.sqlite._document_store import (
    AsyncSqliteDocumentStore,
    SqliteDocumentStore,
)
from anchor.storage.sqlite._entry_store import AsyncSqliteEntryStore, SqliteEntryStore
from anchor.storage.sqlite._schema import ensure_tables, ensure_tables_async
from anchor.storage.sqlite._vec_store import SqliteVecVectorStore
from anchor.storage.sqlite._vector_store import (
    AsyncSqliteVectorStore,
    SqliteVectorStore,
)

__all__ = [
    "AsyncSqliteContextStore",
    "AsyncSqliteDocumentStore",
    "AsyncSqliteEntryStore",
    "AsyncSqliteVectorStore",
    "SqliteConnectionManager",
    "SqliteContextStore",
    "SqliteDocumentStore",
    "SqliteEntryStore",
    "SqliteVecVectorStore",
    "SqliteVectorStore",
    "ensure_tables",
    "ensure_tables_async",
]
