"""SQLite schema definitions and table creation."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

_TABLES: list[str] = [
    """CREATE TABLE IF NOT EXISTS context_items (
        id          TEXT NOT NULL,
        content     TEXT NOT NULL,
        source      TEXT NOT NULL,
        score       REAL NOT NULL DEFAULT 0.0,
        priority    INTEGER NOT NULL DEFAULT 5,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at  TEXT NOT NULL,
        vault       TEXT NOT NULL DEFAULT '__default__',
        namespace   TEXT NOT NULL DEFAULT '/',
        PRIMARY KEY (vault, id)
    )""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        item_id       TEXT NOT NULL,
        embedding_blob BLOB NOT NULL,
        metadata_json  TEXT NOT NULL DEFAULT '{}',
        vault         TEXT NOT NULL DEFAULT '__default__',
        namespace     TEXT NOT NULL DEFAULT '/',
        PRIMARY KEY (vault, item_id)
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_id        TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS memory_entries (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        relevance_score REAL NOT NULL DEFAULT 0.5,
        access_count    INTEGER NOT NULL DEFAULT 0,
        last_accessed   TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        tags_json       TEXT NOT NULL DEFAULT '[]',
        metadata_json   TEXT NOT NULL DEFAULT '{}',
        memory_type     TEXT NOT NULL DEFAULT 'semantic',
        user_id         TEXT,
        session_id      TEXT,
        expires_at      TEXT,
        content_hash    TEXT NOT NULL DEFAULT '',
        source_turns_json TEXT NOT NULL DEFAULT '[]',
        links_json      TEXT NOT NULL DEFAULT '[]'
    )""",
]

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_context_items_scope ON context_items(vault, namespace)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_scope ON embeddings(vault, namespace)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_user_id ON memory_entries(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_session_id ON memory_entries(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_memory_type ON memory_entries(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON memory_entries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_expires_at ON memory_entries(expires_at)",
]


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create all storage tables and indexes if they do not exist."""
    for ddl in _TABLES:
        conn.execute(ddl)
    for ddl in _INDEXES:
        conn.execute(ddl)
    conn.commit()


async def ensure_tables_async(conn: aiosqlite.Connection) -> None:
    """Async variant of :func:`ensure_tables`."""
    for ddl in _TABLES:
        await conn.execute(ddl)
    for ddl in _INDEXES:
        await conn.execute(ddl)
    await conn.commit()
