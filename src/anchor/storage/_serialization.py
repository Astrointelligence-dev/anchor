"""Shared serialization helpers for SQL-backed storage backends."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import MemoryEntry


def context_item_to_row(item: ContextItem) -> dict[str, Any]:
    """Convert a frozen ContextItem to a flat dict suitable for SQL INSERT."""
    return {
        "id": item.id,
        "content": item.content,
        "source": str(item.source),
        "score": item.score,
        "priority": item.priority,
        "token_count": item.token_count,
        "metadata_json": json.dumps(item.metadata, default=str),
        "created_at": item.created_at.isoformat(),
        "vault": item.vault,
        "namespace": item.namespace,
    }


def row_to_context_item(row: dict[str, Any] | Any) -> ContextItem:
    """Reconstruct a ContextItem from a database row."""
    # Handle both dict and sqlite3.Row
    r = dict(row) if not isinstance(row, dict) else row
    return ContextItem(
        id=r["id"],
        content=r["content"],
        source=SourceType(r["source"]),
        score=r["score"],
        priority=r["priority"],
        token_count=r["token_count"],
        metadata=json.loads(r["metadata_json"]),
        created_at=datetime.fromisoformat(r["created_at"]),
        vault=r["vault"],
        namespace=r["namespace"],
    )


def memory_entry_to_row(entry: MemoryEntry) -> dict[str, Any]:
    """Convert a MemoryEntry to a flat dict for SQL INSERT."""
    return {
        "id": entry.id,
        "content": entry.content,
        "relevance_score": entry.relevance_score,
        "access_count": entry.access_count,
        "last_accessed": entry.last_accessed.isoformat(),
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "tags_json": json.dumps(entry.tags),
        "metadata_json": json.dumps(entry.metadata, default=str),
        "memory_type": str(entry.memory_type),
        "user_id": entry.user_id,
        "session_id": entry.session_id,
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        "content_hash": entry.content_hash,
        "source_turns_json": json.dumps(entry.source_turns),
        "links_json": json.dumps(entry.links),
    }


def row_to_memory_entry(row: dict[str, Any] | Any) -> MemoryEntry:
    """Reconstruct a MemoryEntry from a database row."""
    r = dict(row) if not isinstance(row, dict) else row
    expires_at = None
    if r.get("expires_at"):
        expires_at = datetime.fromisoformat(r["expires_at"])

    return MemoryEntry(
        id=r["id"],
        content=r["content"],
        relevance_score=r["relevance_score"],
        access_count=r["access_count"],
        last_accessed=datetime.fromisoformat(r["last_accessed"]),
        created_at=datetime.fromisoformat(r["created_at"]),
        updated_at=datetime.fromisoformat(r["updated_at"]),
        tags=json.loads(r["tags_json"]),
        metadata=json.loads(r["metadata_json"]),
        memory_type=r["memory_type"],
        user_id=r.get("user_id"),
        session_id=r.get("session_id"),
        expires_at=expires_at,
        content_hash=r["content_hash"],
        source_turns=json.loads(r["source_turns_json"]),
        links=json.loads(r["links_json"]),
    )


def escape_like(query: str) -> str:
    """Escape special characters (``%``, ``_``, ``\\``) for SQL LIKE patterns.

    The caller must append ``ESCAPE '\\\\'`` to the SQL statement.
    """
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
