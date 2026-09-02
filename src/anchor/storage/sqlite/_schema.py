"""SQLite schema definitions, table creation and the vault migration."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from anchor.models.scope import DEFAULT_VAULT, ROOT_NAMESPACE

if TYPE_CHECKING:
    import aiosqlite

_TABLES: dict[str, str] = {
    "context_items": f"""CREATE TABLE IF NOT EXISTS context_items (
        id          TEXT NOT NULL,
        content     TEXT NOT NULL,
        source      TEXT NOT NULL,
        score       REAL NOT NULL DEFAULT 0.0,
        priority    INTEGER NOT NULL DEFAULT 5,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{{}}',
        created_at  TEXT NOT NULL,
        vault       TEXT NOT NULL DEFAULT '{DEFAULT_VAULT}',
        namespace   TEXT NOT NULL DEFAULT '{ROOT_NAMESPACE}',
        PRIMARY KEY (vault, id)
    )""",
    "embeddings": f"""CREATE TABLE IF NOT EXISTS embeddings (
        item_id       TEXT NOT NULL,
        embedding_blob BLOB NOT NULL,
        metadata_json  TEXT NOT NULL DEFAULT '{{}}',
        vault         TEXT NOT NULL DEFAULT '{DEFAULT_VAULT}',
        namespace     TEXT NOT NULL DEFAULT '{ROOT_NAMESPACE}',
        PRIMARY KEY (vault, item_id)
    )""",
    "documents": """CREATE TABLE IF NOT EXISTS documents (
        doc_id        TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    "memory_entries": """CREATE TABLE IF NOT EXISTS memory_entries (
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
}

# Tables keyed by (vault, id) since front #3 — the ones the migration rebuilds.
_SCOPED_TABLES = ("context_items", "embeddings")

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_context_items_scope ON context_items(vault, namespace)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_scope ON embeddings(vault, namespace)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_user_id ON memory_entries(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_session_id ON memory_entries(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_memory_type ON memory_entries(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON memory_entries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entries_expires_at ON memory_entries(expires_at)",
]


def _rebuild_plan(table: str, old_cols: list[str], new_cols: list[str]) -> list[str]:
    """Copy every column both shapes share (rowid included — vec0 points
    at it), then swap the tables. The new DDL fills vault/namespace with
    their defaults for legacy rows."""
    cols = ", ".join(["rowid", *[c for c in new_cols if c in old_cols]])
    return [
        f"INSERT INTO {table}__new ({cols}) SELECT {cols} FROM {table}",  # noqa: S608 -- identifiers come from PRAGMA, never from callers
        f"DROP TABLE {table}",
        f"ALTER TABLE {table}__new RENAME TO {table}",
    ]


def _has_vault_key(index_lists: list[tuple[str, list[str]]]) -> bool:
    """True when some UNIQUE/PRIMARY KEY index on the table includes vault."""
    return any("vault" in cols for _, cols in index_lists)


def rebuild_with_scope_key(conn: sqlite3.Connection, table: str, ddl: str) -> bool:
    """Rebuild *table* under *ddl* when no unique key on it includes ``vault``.

    A pre-vault table keys rows on the bare id, so a write from one vault
    would replace another vault's row (``INSERT OR REPLACE``) or be
    rejected (``UNIQUE``). SQLite cannot alter a key: the rows are copied
    into a fresh table and swapped in, in one transaction, rowids kept.
    Idempotent — returns True only when a rebuild happened.
    """
    uniques = [
        (row[1], [r[2] for r in conn.execute(f"PRAGMA index_info({row[1]})")])
        for row in conn.execute(f"PRAGMA index_list({table})")
        if row[2]  # unique
    ]
    if _has_vault_key(uniques):
        return False
    old_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}__new")
        conn.execute(ddl.replace(table, f"{table}__new", 1))
        new_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table}__new)")]
        for sql in _rebuild_plan(table, old_cols, new_cols):
            conn.execute(sql)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return True


async def rebuild_with_scope_key_async(
    conn: aiosqlite.Connection, table: str, ddl: str,
) -> bool:
    """Async twin of :func:`rebuild_with_scope_key`."""
    uniques = []
    for row in await (await conn.execute(f"PRAGMA index_list({table})")).fetchall():
        if row[2]:
            info = await (await conn.execute(f"PRAGMA index_info({row[1]})")).fetchall()
            uniques.append((row[1], [r[2] for r in info]))
    if _has_vault_key(uniques):
        return False
    old_rows = await (await conn.execute(f"PRAGMA table_info({table})")).fetchall()
    old_cols = [r[1] for r in old_rows]
    if conn.in_transaction:
        await conn.commit()
    await conn.execute("BEGIN")
    try:
        await conn.execute(f"DROP TABLE IF EXISTS {table}__new")
        await conn.execute(ddl.replace(table, f"{table}__new", 1))
        new_rows = await (await conn.execute(f"PRAGMA table_info({table}__new)")).fetchall()
        for sql in _rebuild_plan(table, old_cols, [r[1] for r in new_rows]):
            await conn.execute(sql)
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise
    return True


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tables/indexes if missing and migrate pre-vault shapes."""
    for ddl in _TABLES.values():
        conn.execute(ddl)
    for table in _SCOPED_TABLES:
        rebuild_with_scope_key(conn, table, _TABLES[table])
    for ddl in _INDEXES:
        conn.execute(ddl)
    conn.commit()


async def ensure_tables_async(conn: aiosqlite.Connection) -> None:
    """Async variant of :func:`ensure_tables`."""
    for ddl in _TABLES.values():
        await conn.execute(ddl)
    for table in _SCOPED_TABLES:
        await rebuild_with_scope_key_async(conn, table, _TABLES[table])
    for ddl in _INDEXES:
        await conn.execute(ddl)
    await conn.commit()
