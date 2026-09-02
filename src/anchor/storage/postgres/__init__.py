"""PostgreSQL persistent storage backend.

Async-first using asyncpg. Supports pgvector for vector similarity search.

Install with: pip install astro-anchor[postgres]
"""

from anchor.storage.postgres._connection import PostgresConnectionManager
from anchor.storage.postgres._context_store import PostgresContextStore
from anchor.storage.postgres._document_store import PostgresDocumentStore
from anchor.storage.postgres._entry_store import PostgresEntryStore
from anchor.storage.postgres._graph_store import PostgresGraphStore
from anchor.storage.postgres._schema import ensure_tables
from anchor.storage.postgres._vector_store import PostgresVectorStore

__all__ = [
    "PostgresConnectionManager",
    "PostgresContextStore",
    "PostgresDocumentStore",
    "PostgresEntryStore",
    "PostgresGraphStore",
    "PostgresVectorStore",
    "ensure_tables",
]
