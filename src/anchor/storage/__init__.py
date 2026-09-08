"""Built-in storage implementations.

In-memory and JSON-file stores ship with the core package.
Persistent backends (SQLite, PostgreSQL, Redis) are available as optional extras::

    pip install astro-anchor[sqlite]    # SQLite with WAL mode
    pip install astro-anchor[postgres]  # PostgreSQL with pgvector
    pip install astro-anchor[redis]     # Redis caching backend
"""

from .json_file_store import JsonFileMemoryStore
from .json_memory_store import InMemoryEntryStore
from .memory_store import (
    InMemoryContextStore,
    InMemoryDocumentStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)

__all__ = [
    "InMemoryContextStore",
    "InMemoryDocumentStore",
    "InMemoryEntryStore",
    "InMemoryGraphStore",
    "InMemoryVectorStore",
    "JsonFileMemoryStore",
]
