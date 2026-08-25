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
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            source      TEXT NOT NULL,
            score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            priority    INTEGER NOT NULL DEFAULT 5,
            token_count INTEGER NOT NULL DEFAULT 0,
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL
        )
    """)

    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            item_id   TEXT PRIMARY KEY,
            embedding vector({embedding_dim}),
            metadata  JSONB NOT NULL DEFAULT '{{}}'
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
