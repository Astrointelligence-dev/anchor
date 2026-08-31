"""Redis-backed MemoryEntryStore implementations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from anchor.models.memory import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from anchor.storage.redis._connection import RedisConnectionManager


class RedisEntryStore:
    """Redis-backed memory entry store. Implements MemoryEntryStore protocol.

    Entries are stored as JSON strings. Search is done client-side via
    substring matching after loading candidates. Secondary indexes on
    user_id and session_id enable efficient filtered lookups.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: RedisConnectionManager) -> None:
        self._conn_manager = conn_manager

    def _key(self, entry_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:{entry_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}mem:_ids"

    def _user_key(self, user_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:user:{user_id}"

    def _session_key(self, session_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:sess:{session_id}"

    def add(self, entry: MemoryEntry) -> None:
        client = self._conn_manager.get_client()
        data = entry.model_dump_json()
        # Upsert: drop index memberships of a previous version whose
        # user/session changed — a stale set would keep claiming the id
        # and delete_by_user would destroy the re-owned entry.
        old = self._get_entry(entry.id)
        pipe = client.pipeline()
        if old is not None:
            if old.user_id and old.user_id != entry.user_id:
                pipe.srem(self._user_key(old.user_id), entry.id)
            if old.session_id and old.session_id != entry.session_id:
                pipe.srem(self._session_key(old.session_id), entry.id)
        pipe.set(self._key(entry.id), data)
        pipe.sadd(self._ids_key(), entry.id)
        if entry.user_id:
            pipe.sadd(self._user_key(entry.user_id), entry.id)
        if entry.session_id:
            pipe.sadd(self._session_key(entry.session_id), entry.id)
        pipe.execute()

    def _get_entry(self, entry_id: str) -> MemoryEntry | None:
        client = self._conn_manager.get_client()
        data = client.get(self._key(entry_id))
        if data is None:
            return None
        return MemoryEntry.model_validate_json(data)

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._get_entry(entry_id)

    def _load_entries(self, entry_ids: set[Any]) -> list[MemoryEntry]:
        """Load multiple entries by ID, filtering out None values."""
        if not entry_ids:
            return []
        client = self._conn_manager.get_client()
        keys = [self._key(eid) for eid in entry_ids]
        values = client.mget(keys)
        entries = []
        for v in values:
            if v is not None:
                entries.append(MemoryEntry.model_validate_json(v))
        return entries

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        entries = self._load_entries(ids)

        query_lower = query.lower()
        results = [
            e
            for e in entries
            if not e.is_expired and query_lower in e.content.lower()
        ]
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    def list_all(self) -> list[MemoryEntry]:
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        entries = self._load_entries(ids)
        return [e for e in entries if not e.is_expired]

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        return self._load_entries(ids)

    def delete(self, entry_id: str) -> bool:
        entry = self._get_entry(entry_id)
        if entry is None:
            return False
        client = self._conn_manager.get_client()
        pipe = client.pipeline()
        pipe.delete(self._key(entry_id))
        pipe.srem(self._ids_key(), entry_id)
        if entry.user_id:
            pipe.srem(self._user_key(entry.user_id), entry_id)
        if entry.session_id:
            pipe.srem(self._session_key(entry.session_id), entry_id)
        pipe.execute()
        return True

    def clear(self) -> None:
        """Remove all entries.

        .. warning::
            Not atomic — entries added between ``smembers`` and ``execute``
            may be partially cleared. For strict atomicity use a Lua script.

        .. note::
            If entry data keys were evicted by Redis but the ID tracking set
            still has entries, orphaned ``user:`` and ``sess:`` index keys
            may remain. A periodic ``SCAN``-based cleanup is recommended for
            long-lived deployments.
        """
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        if not ids:
            return
        # Load entries to clean up secondary indexes
        entries = self._load_entries(ids)
        pipe = client.pipeline()
        for entry in entries:
            pipe.delete(self._key(entry.id))
            if entry.user_id:
                pipe.srem(self._user_key(entry.user_id), entry.id)
            if entry.session_id:
                pipe.srem(self._session_key(entry.session_id), entry.id)
        pipe.delete(self._ids_key())
        pipe.execute()

    def search_filtered(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[MemoryEntry]:
        client = self._conn_manager.get_client()

        # Use secondary indexes to narrow candidates
        if user_id is not None:
            candidate_ids = client.smembers(self._user_key(user_id))
        elif session_id is not None:
            candidate_ids = client.smembers(self._session_key(session_id))
        else:
            candidate_ids = client.smembers(self._ids_key())

        entries = self._load_entries(candidate_ids)
        query_lower = query.lower()
        memory_type_str = str(memory_type) if memory_type is not None else None

        results = [
            e for e in entries
            if self._matches(
                e, query_lower, user_id, session_id,
                memory_type_str, tags, created_after, created_before,
            )
        ]
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _matches(
        e: MemoryEntry,
        query_lower: str,
        user_id: str | None,
        session_id: str | None,
        memory_type_str: str | None,
        tags: list[str] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> bool:
        if e.is_expired:
            return False
        if query_lower and query_lower not in e.content.lower():
            return False
        if user_id is not None and e.user_id != user_id:
            return False
        if session_id is not None and e.session_id != session_id:
            return False
        if memory_type_str is not None and str(e.memory_type) != memory_type_str:
            return False
        if tags is not None and not all(t in e.tags for t in tags):
            return False
        if created_after is not None and e.created_at <= created_after:
            return False
        return not (created_before is not None and e.created_at >= created_before)

    def delete_by_user(self, user_id: str) -> int:
        client = self._conn_manager.get_client()
        entry_ids = client.smembers(self._user_key(user_id))
        if not entry_ids:
            return 0

        entries = self._load_entries(entry_ids)
        pipe = client.pipeline()
        deleted = 0
        for entry in entries:
            if entry.user_id != user_id:
                # Stale index membership (entry re-owned since): the index
                # set is wiped below, but the entry itself must survive.
                continue
            pipe.delete(self._key(entry.id))
            pipe.srem(self._ids_key(), entry.id)
            if entry.session_id:
                pipe.srem(self._session_key(entry.session_id), entry.id)
            deleted += 1
        pipe.delete(self._user_key(user_id))
        pipe.execute()
        return deleted

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._conn_manager.prefix!r})"


class AsyncRedisEntryStore:
    """Async Redis-backed memory entry store. Implements AsyncMemoryEntryStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: RedisConnectionManager) -> None:
        self._conn_manager = conn_manager

    def _key(self, entry_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:{entry_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}mem:_ids"

    def _user_key(self, user_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:user:{user_id}"

    def _session_key(self, session_id: str) -> str:
        return f"{self._conn_manager.prefix}mem:sess:{session_id}"

    async def add(self, entry: MemoryEntry) -> None:
        client = self._conn_manager.get_async_client()
        data = entry.model_dump_json()
        # Upsert: same stale-index cleanup as the sync store.
        raw = await client.get(self._key(entry.id))
        old = MemoryEntry.model_validate_json(raw) if raw is not None else None
        pipe = client.pipeline()
        if old is not None:
            if old.user_id and old.user_id != entry.user_id:
                pipe.srem(self._user_key(old.user_id), entry.id)
            if old.session_id and old.session_id != entry.session_id:
                pipe.srem(self._session_key(old.session_id), entry.id)
        pipe.set(self._key(entry.id), data)
        pipe.sadd(self._ids_key(), entry.id)
        if entry.user_id:
            pipe.sadd(self._user_key(entry.user_id), entry.id)
        if entry.session_id:
            pipe.sadd(self._session_key(entry.session_id), entry.id)
        await pipe.execute()

    async def get(self, entry_id: str) -> MemoryEntry | None:
        client = self._conn_manager.get_async_client()
        data = await client.get(self._key(entry_id))
        if data is None:
            return None
        return MemoryEntry.model_validate_json(data)

    async def _load_entries(self, entry_ids: set[Any]) -> list[MemoryEntry]:
        if not entry_ids:
            return []
        client = self._conn_manager.get_async_client()
        keys = [self._key(eid) for eid in entry_ids]
        values = await client.mget(keys)
        return [MemoryEntry.model_validate_json(v) for v in values if v is not None]

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        entries = await self._load_entries(ids)

        query_lower = query.lower()
        results = [
            e
            for e in entries
            if not e.is_expired and query_lower in e.content.lower()
        ]
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    async def list_all(self) -> list[MemoryEntry]:
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        entries = await self._load_entries(ids)
        return [e for e in entries if not e.is_expired]

    async def list_all_unfiltered(self) -> list[MemoryEntry]:
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        return await self._load_entries(ids)

    async def delete(self, entry_id: str) -> bool:
        client = self._conn_manager.get_async_client()
        data = await client.get(self._key(entry_id))
        if data is None:
            return False
        entry = MemoryEntry.model_validate_json(data)
        pipe = client.pipeline()
        pipe.delete(self._key(entry_id))
        pipe.srem(self._ids_key(), entry_id)
        if entry.user_id:
            pipe.srem(self._user_key(entry.user_id), entry_id)
        if entry.session_id:
            pipe.srem(self._session_key(entry.session_id), entry_id)
        await pipe.execute()
        return True

    async def clear(self) -> None:
        """Remove all entries.

        .. warning::
            Not atomic — entries added between ``smembers`` and ``execute``
            may be partially cleared. For strict atomicity use a Lua script.

        .. note::
            If entry data keys were evicted by Redis but the ID tracking set
            still has entries, orphaned ``user:`` and ``sess:`` index keys
            may remain.
        """
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        if not ids:
            return
        entries = await self._load_entries(ids)
        pipe = client.pipeline()
        for entry in entries:
            pipe.delete(self._key(entry.id))
            if entry.user_id:
                pipe.srem(self._user_key(entry.user_id), entry.id)
            if entry.session_id:
                pipe.srem(self._session_key(entry.session_id), entry.id)
        pipe.delete(self._ids_key())
        await pipe.execute()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._conn_manager.prefix!r})"
