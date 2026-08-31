"""Regression tests for the 2026-08-31 release-engineering review fixes
(storage side). Each test failed against the pre-fix code — see
docs/plans/2026-08-31-release-engineering.md, findings 18-20.
"""

from __future__ import annotations

import asyncio

import pytest

from anchor.storage.redis._entry_store import RedisEntryStore
from anchor.storage.sqlite._connection import SqliteConnectionManager
from tests.conftest import make_memory_entry as _make_entry

pytest.importorskip("aiosqlite")


class TestAsyncConnectionRace:
    """Finding 19: concurrent first calls both connected; the loser
    connection leaked its non-daemon worker thread and hung shutdown."""

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_share_one_connection(self, tmp_path):
        mgr = SqliteConnectionManager(tmp_path / "race.db")
        try:
            c1, c2 = await asyncio.gather(
                mgr.get_async_connection(), mgr.get_async_connection(),
            )
            assert c1 is c2
            assert mgr._async_conn is c1
        finally:
            await mgr.aclose()


# ---------------------------------------------------------------------------
# Minimal in-process fake of the redis client surface the entry store uses
# (the repo has no redis test infra — these bugs shipped unobserved).
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, store: _FakeRedis) -> None:
        self._store = store
        self._ops: list[tuple] = []

    def set(self, key, value):
        self._ops.append(("set", key, value))

    def delete(self, key):
        self._ops.append(("delete", key))

    def sadd(self, key, member):
        self._ops.append(("sadd", key, member))

    def srem(self, key, member):
        self._ops.append(("srem", key, member))

    def execute(self):
        for op in self._ops:
            kind, key = op[0], op[1]
            if kind == "set":
                self._store.kv[key] = op[2]
            elif kind == "delete":
                self._store.kv.pop(key, None)
                self._store.sets.pop(key, None)
            elif kind == "sadd":
                self._store.sets.setdefault(key, set()).add(op[2])
            elif kind == "srem":
                self._store.sets.get(key, set()).discard(op[2])
        self._ops = []


class _FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def get(self, key):
        return self.kv.get(key)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def pipeline(self):
        return _FakePipeline(self)


class _FakeConnManager:
    prefix = ""

    def __init__(self) -> None:
        self._client = _FakeRedis()

    def get_client(self):
        return self._client


class TestRedisReownedEntry:
    """Finding 20: upsert left the id in the old owner's index set, and
    delete_by_user destroyed the re-owned entry without rechecking."""

    def test_delete_by_user_spares_reowned_entry(self):
        store = RedisEntryStore(_FakeConnManager())
        store.add(_make_entry(entry_id="e1", user_id="alice"))
        store.add(_make_entry(entry_id="e1", user_id="bob"))

        deleted = store.delete_by_user("alice")

        assert deleted == 0
        assert store.get("e1") is not None
        assert store.get("e1").user_id == "bob"

    def test_upsert_moves_index_membership(self):
        store = RedisEntryStore(_FakeConnManager())
        store.add(_make_entry(entry_id="e1", user_id="alice"))
        store.add(_make_entry(entry_id="e1", user_id="bob"))

        client = store._conn_manager.get_client()
        assert "e1" not in client.smembers("mem:user:alice")
        assert "e1" in client.smembers("mem:user:bob")

    def test_upsert_moves_session_membership(self):
        store = RedisEntryStore(_FakeConnManager())
        store.add(_make_entry(entry_id="e1", session_id="s1"))
        store.add(_make_entry(entry_id="e1", session_id="s2"))

        client = store._conn_manager.get_client()
        assert "e1" not in client.smembers("mem:sess:s1")
        assert "e1" in client.smembers("mem:sess:s2")
