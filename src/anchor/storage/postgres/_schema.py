"""PostgreSQL schema definitions and table creation with pgvector support."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg


async def ensure_tables(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    *,
    embedding_dim: int,
) -> None:
    """Create all storage tables and indexes if they do not exist.

    Parameters:
        conn: An asyncpg connection.
        embedding_dim: Dimension of embedding vectors for the pgvector
            column. Required — it must match your embedding provider's
            ``dimensions`` (e.g. 1536 for text-embedding-3-small, 1024 for
            voyage-3.5 or BGE-M3).
    """
    if embedding_dim <= 0:
        msg = f"embedding_dim must be positive, got {embedding_dim}"
        raise ValueError(msg)
    # Enable pgvector extension (requires superuser or CREATE privilege)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS context_items (
            id          TEXT NOT NULL,
            content     TEXT NOT NULL,
            source      TEXT NOT NULL,
            score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            priority    INTEGER NOT NULL DEFAULT 5,
            token_count INTEGER NOT NULL DEFAULT 0,
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL,
            vault       TEXT NOT NULL DEFAULT '__default__',
            namespace   TEXT COLLATE "C" NOT NULL DEFAULT '/',
            PRIMARY KEY (vault, id)
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            item_id   TEXT NOT NULL,
            embedding vector({embedding_dim}),
            metadata  JSONB NOT NULL DEFAULT '{{}}',
            vault     TEXT NOT NULL DEFAULT '__default__',
            namespace TEXT COLLATE "C" NOT NULL DEFAULT '/',
            PRIMARY KEY (vault, item_id)
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id   TEXT PRIMARY KEY,
            content  TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_entries (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
            access_count    INTEGER NOT NULL DEFAULT 0,
            last_accessed   TIMESTAMPTZ NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL,
            tags            JSONB NOT NULL DEFAULT '[]',
            metadata        JSONB NOT NULL DEFAULT '{}',
            memory_type     TEXT NOT NULL DEFAULT 'semantic',
            user_id         TEXT,
            session_id      TEXT,
            expires_at      TIMESTAMPTZ,
            content_hash    TEXT NOT NULL DEFAULT '',
            source_turns    JSONB NOT NULL DEFAULT '[]',
            links           JSONB NOT NULL DEFAULT '[]'
        )
    """)

    # Indexes
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_user_id ON memory_entries(user_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_session_id ON memory_entries(session_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_memory_type ON memory_entries(memory_type)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_created_at ON memory_entries(created_at)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_me_expires_at ON memory_entries(expires_at)"
    )

    # pgvector HNSW index (pgvector >= 0.5). Unlike IVFFlat it needs no
    # training data, so it can be created on an empty table. Defaults
    # m=16, ef_construction=64 are the pgvector recommendations.
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    # GIN index so JSONB containment filters (search where=...) stay fast.
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_metadata ON embeddings "
        "USING gin (metadata jsonb_path_ops)"
    )
    # Front #3 migration of pre-vault tables: scope columns, the composite
    # (vault, id) key the upserts rely on (ON CONFLICT needs a matching
    # unique index), and a byte-wise namespace so the prefix ranges hold
    # under any database collation. Every step probes the catalog first —
    # ALTER TABLE takes ACCESS EXCLUSIVE even when IF NOT EXISTS is a no-op.
    for table, id_col in (("context_items", "id"), ("embeddings", "item_id")):
        await _migrate_scoped_table(conn, table, id_col)

    # Scope btree: vault bind + boundary-aware namespace prefix ranges.
    # Plain opclass on a COLLATE "C" column — text_pattern_ops only serves
    # LIKE, never the >=/< ranges the compiler emits.
    for table in ("embeddings", "context_items"):
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_scope ON {table} "
            "(vault, namespace)"
        )


async def _migrate_scoped_table(
    conn: asyncpg.Connection,
    table: str,
    id_col: str,
) -> None:
    cols = {
        r["attname"]
        for r in await conn.fetch(
            "SELECT attname FROM pg_attribute WHERE attrelid = $1::regclass "
            "AND attnum > 0 AND NOT attisdropped",
            table,
        )
    }
    if "vault" not in cols:
        await conn.execute(
            f"ALTER TABLE {table} ADD COLUMN vault TEXT NOT NULL "
            "DEFAULT '__default__'"
        )
    if "namespace" not in cols:
        await conn.execute(
            f"ALTER TABLE {table} ADD COLUMN namespace TEXT COLLATE \"C\" "
            "NOT NULL DEFAULT '/'"
        )
    # The pre-fix index was built with text_pattern_ops: drop it before the
    # collation change rebuilds it for nothing (recreated plain by the caller).
    indexdef = await conn.fetchval(
        "SELECT indexdef FROM pg_indexes WHERE indexname = $1",
        f"idx_{table}_scope",
    )
    if indexdef and "text_pattern_ops" in indexdef:
        await conn.execute(f"DROP INDEX idx_{table}_scope")
    collation = await conn.fetchval(
        "SELECT c.collname FROM pg_attribute a "
        "JOIN pg_collation c ON c.oid = a.attcollation "
        "WHERE a.attrelid = $1::regclass AND a.attname = 'namespace'",
        table,
    )
    if collation != "C":
        await conn.execute(
            f"ALTER TABLE {table} ALTER COLUMN namespace TYPE TEXT COLLATE \"C\""
        )
    pk = await conn.fetchrow(
        "SELECT c.conname, array_agg(a.attname ORDER BY k.ord) AS cols "
        "FROM pg_constraint c "
        "JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord) "
        "ON true "
        "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum "
        "WHERE c.conrelid = $1::regclass AND c.contype = 'p' "
        "GROUP BY c.conname",
        table,
    )
    if pk is None or "vault" not in pk["cols"]:
        drop = f"DROP CONSTRAINT {pk['conname']}, " if pk is not None else ""
        await conn.execute(
            f"ALTER TABLE {table} {drop}ADD PRIMARY KEY (vault, {id_col})"
        )
