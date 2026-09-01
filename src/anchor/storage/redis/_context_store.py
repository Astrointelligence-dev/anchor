"""Redis-backed ContextStore implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anchor.models.context import ContextItem
from anchor.models.scope import DEFAULT_VAULT

if TYPE_CHECKING:
    from anchor.storage.redis._connection import RedisConnectionManager


class RedisContextStore:
    """Redis-backed context store. Implements the ContextStore protocol.

    Bound to one vault at construction (front #3): the vault is part of
    the key path (``{prefix}ctx:{vault}:{id}``), so cross-vault access is
    structurally impossible — the same pattern as the connection-manager
    prefix. Pre-vault keys (``ctx:{id}``) belong to the ``__default__``
    vault and are found via a legacy-key fallback on read.
    """

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self, conn_manager: RedisConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    def _key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctx:{self._vault}:{item_id}"

    def _legacy_key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctx:{item_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}ctx:{self._vault}:_ids"

    def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        client = self._conn_manager.get_client()
        data = item.model_dump_json()
        pipe = client.pipeline()
        pipe.set(self._key(item.id), data)
        pipe.sadd(self._ids_key(), item.id)
        pipe.execute()

    def get(self, item_id: str) -> ContextItem | None:
        client = self._conn_manager.get_client()
        data = client.get(self._key(item_id))
        if data is None and self._vault == DEFAULT_VAULT:
            data = client.get(self._legacy_key(item_id))
        if data is None:
            return None
        return ContextItem.model_validate_json(data)

    def get_all(self) -> list[ContextItem]:
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        if not ids:
            return []
        keys = [self._key(id_) for id_ in ids]
        values = client.mget(keys)
        return [ContextItem.model_validate_json(v) for v in values if v is not None]

    def delete(self, item_id: str) -> bool:
        client = self._conn_manager.get_client()
        pipe = client.pipeline()
        pipe.delete(self._key(item_id))
        pipe.srem(self._ids_key(), item_id)
        results = pipe.execute()
        return results[0] > 0

    def clear(self) -> None:
        """Remove all items.

        .. warning::
            Not atomic — items added between ``smembers`` and ``delete``
            may be partially cleared. For strict atomicity use a Lua script.
        """
        client = self._conn_manager.get_client()
        ids = client.smembers(self._ids_key())
        if ids:
            keys = [self._key(id_) for id_ in ids]
            keys.append(self._ids_key())
            client.delete(*keys)
        else:
            client.delete(self._ids_key())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._conn_manager.prefix!r})"


class AsyncRedisContextStore:
    """Async Redis-backed context store (vault-bound, same keys as sync)."""

    __slots__ = ("_conn_manager", "_vault")

    def __init__(
        self, conn_manager: RedisConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    def _key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctx:{self._vault}:{item_id}"

    def _legacy_key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctx:{item_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}ctx:{self._vault}:_ids"

    async def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        client = self._conn_manager.get_async_client()
        data = item.model_dump_json()
        pipe = client.pipeline()
        pipe.set(self._key(item.id), data)
        pipe.sadd(self._ids_key(), item.id)
        await pipe.execute()

    async def get(self, item_id: str) -> ContextItem | None:
        client = self._conn_manager.get_async_client()
        data = await client.get(self._key(item_id))
        if data is None and self._vault == DEFAULT_VAULT:
            data = await client.get(self._legacy_key(item_id))
        if data is None:
            return None
        return ContextItem.model_validate_json(data)

    async def get_all(self) -> list[ContextItem]:
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        if not ids:
            return []
        keys = [self._key(id_) for id_ in ids]
        values = await client.mget(keys)
        return [ContextItem.model_validate_json(v) for v in values if v is not None]

    async def delete(self, item_id: str) -> bool:
        client = self._conn_manager.get_async_client()
        pipe = client.pipeline()
        pipe.delete(self._key(item_id))
        pipe.srem(self._ids_key(), item_id)
        results = await pipe.execute()
        return results[0] > 0

    async def clear(self) -> None:
        """Remove all items.

        .. warning::
            Not atomic — items added between ``smembers`` and ``delete``
            may be partially cleared. For strict atomicity use a Lua script.
        """
        client = self._conn_manager.get_async_client()
        ids = await client.smembers(self._ids_key())
        if ids:
            keys = [self._key(id_) for id_ in ids]
            keys.append(self._ids_key())
            await client.delete(*keys)
        else:
            await client.delete(self._ids_key())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._conn_manager.prefix!r})"
