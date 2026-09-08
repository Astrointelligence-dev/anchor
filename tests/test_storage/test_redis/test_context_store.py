"""Redis context stores over fakeredis: vault layout + pre-vault migration.

Ritual xhigh #7/#8: the ``__default__`` store must SEE legacy keys in
every operation (not only ``get``), and no key can be spelled from
another vault — ``:`` in an item id, an id named ``_ids``, or a legacy
``ctx:`` key can never alias a ``ctxv:`` one.
"""

from __future__ import annotations

import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis")

from anchor.models.context import ContextItem, SourceType  # noqa: E402
from anchor.storage.redis import RedisConnectionManager  # noqa: E402
from anchor.storage.redis._context_store import (  # noqa: E402
    AsyncRedisContextStore,
    RedisContextStore,
)

PREFIX = "anchor:"


def _item(item_id: str, content: str = "x") -> ContextItem:
    return ContextItem(id=item_id, content=content, source=SourceType.RETRIEVAL)


@pytest.fixture
def manager() -> RedisConnectionManager:
    server = fakeredis.FakeServer()
    mgr = RedisConnectionManager("redis://fake", prefix=PREFIX)
    mgr._client = fakeredis.FakeRedis(server=server, decode_responses=True)
    mgr._async_client = fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True,
    )
    return mgr


def _seed_legacy(manager: RedisConnectionManager, *ids: str) -> None:
    client = manager.get_client()
    for id_ in ids:
        client.set(f"{PREFIX}ctx:{id_}", _item(id_, f"legacy {id_}").model_dump_json())
        client.sadd(f"{PREFIX}ctx:_ids", id_)


class TestVaultLayout:
    def test_crud_is_per_vault(self, manager):
        a = RedisContextStore(manager, vault="a")
        b = RedisContextStore(manager, vault="b")
        a.add(_item("d1", "A"))
        b.add(_item("d1", "B"))

        assert a.get("d1").content == "A"
        assert b.get("d1").content == "B"
        assert [i.id for i in a.get_all()] == ["d1"]
        assert a.delete("d1") is True
        assert a.get("d1") is None
        assert b.get("d1").content == "B"
        b.clear()
        assert b.get_all() == []

    def test_colon_in_item_id_cannot_cross_vaults(self, manager):
        RedisContextStore(manager, vault="a").add(_item("b:c", "A"))
        RedisContextStore(manager, vault="ab").add(_item("c", "AB"))
        assert RedisContextStore(manager, vault="a").get("b:c").content == "A"
        assert RedisContextStore(manager, vault="ab").get("c").content == "AB"
        assert RedisContextStore(manager).get("a:b:c") is None

    def test_item_named_ids_does_not_alias_the_index(self, manager):
        store = RedisContextStore(manager)
        store.add(_item("_ids", "sneaky"))
        store.add(_item("d1"))
        assert {i.id for i in store.get_all()} == {"_ids", "d1"}

    def test_vault_with_colon_is_refused(self, manager):
        with pytest.raises(ValueError, match="':'"):
            RedisContextStore(manager, vault="a:b")


class TestLegacyMigration:
    def test_default_store_migrates_legacy_keys_on_first_use(self, manager):
        _seed_legacy(manager, "old1", "old2")
        store = RedisContextStore(manager)

        assert {i.id for i in store.get_all()} == {"old1", "old2"}
        assert store.get("old1").content == "legacy old1"
        assert store.delete("old1") is True
        assert store.get("old1") is None  # no zombie
        store.add(_item("old2", "new"))
        assert store.delete("old2") is True
        assert store.get("old2") is None
        assert manager.get_client().keys(f"{PREFIX}ctx:*") == []

    def test_migration_never_clobbers_a_newer_write(self, manager):
        RedisContextStore(manager).add(_item("old1", "new"))  # written post-upgrade
        _seed_legacy(manager, "old1")
        assert RedisContextStore(manager).get("old1").content == "new"

    def test_other_vaults_leave_legacy_keys_alone(self, manager):
        _seed_legacy(manager, "old1")
        assert RedisContextStore(manager, vault="other").get_all() == []
        assert manager.get_client().exists(f"{PREFIX}ctx:old1") == 1

    def test_async_twin_migrates_too(self, manager):
        _seed_legacy(manager, "old1")

        async def scenario():
            store = AsyncRedisContextStore(manager)
            ids = {i.id for i in await store.get_all()}
            deleted = await store.delete("old1")
            gone = await store.get("old1")
            return ids, deleted, gone

        assert asyncio.run(scenario()) == ({"old1"}, True, None)
