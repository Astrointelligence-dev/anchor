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


# Front #3 migration: pre-vault tables gain the scope columns in place
# (ALTER ADD COLUMN with constant defaults — cheap and idempotent, runs
# before the scope indexes that need the columns). The composite
# (vault, id) PRIMARY KEY exists only on freshly created tables: SQLite
# cannot ALTER a PK, and the full-table rebuild is deferred until
# multi-vault-in-one-legacy-db actually needs it.
_SCOPE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "context_items": [
        ("vault", "TEXT NOT NULL DEFAULT '__default__'"),
        ("namespace", "TEXT NOT NULL DEFAULT '/'"),
    ],
    "embeddings": [
        ("vault", "TEXT NOT NULL DEFAULT '__default__'"),
        ("namespace", "TEXT NOT NULL DEFAULT '/'"),
    ],
}


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tables/indexes if missing and migrate pre-vault shapes."""
    for ddl in _TABLES:
        conn.execute(ddl)
    for table, columns in _SCOPE_COLUMNS.items():
        existing = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    for ddl in _INDEXES:
        conn.execute(ddl)
    conn.commit()


async def ensure_tables_async(conn: aiosqlite.Connection) -> None:
    """Async variant of :func:`ensure_tables`."""
    for ddl in _TABLES:
        await conn.execute(ddl)
    for table, columns in _SCOPE_COLUMNS.items():
        cursor = await conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        for name, decl in columns:
            if name not in existing:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    for ddl in _INDEXES:
        await conn.execute(ddl)
    await conn.commit()
