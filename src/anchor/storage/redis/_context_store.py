"""Redis-backed ContextStore implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anchor.models.context import ContextItem
from anchor.models.scope import DEFAULT_VAULT, validate_vault

if TYPE_CHECKING:
    from anchor.storage.redis._connection import RedisConnectionManager

# Pre-vault layout: ``{prefix}ctx:{id}`` + the ``{prefix}ctx:_ids`` set.
_LEGACY_IDS = "ctx:_ids"


def _migration_steps(
    prefix: str,
    legacy_ids: Any,
    key_of: Callable[[str], str],
    ids_key: str,
) -> list[tuple[Any, ...]]:
    """Move every legacy key under the vault layout, then drop the old set.

    ``RENAMENX`` never clobbers an item already written under the new
    layout; the whole plan runs once per store instance and is a no-op
    afterwards (the legacy set is gone).
    """
    steps: list[tuple[Any, ...]] = []
    for id_ in legacy_ids:
        steps.append(("renamenx", f"{prefix}ctx:{id_}", key_of(id_)))
        steps.append(("sadd", ids_key, id_))
    steps.append(("delete", f"{prefix}{_LEGACY_IDS}"))
    return steps


class RedisContextStore:
    """Redis-backed context store. Implements the ContextStore protocol.

    Bound to one vault at construction (front #3): the vault is part of
    the key path — ``{prefix}ctxv:{vault}:{id}``, ids in
    ``{prefix}ctxv-ids:{vault}`` — so cross-vault access is structurally
    impossible: a vault name never contains ``:`` (validated), the ids
    set lives under its own prefix (an item id ``_ids`` cannot alias it),
    and the pre-vault ``ctx:`` keys can never spell a ``ctxv:`` one. The
    ``__default__`` store migrates pre-vault keys in place on first use.
    """

    __slots__ = ("_conn_manager", "_migrated", "_vault")

    def __init__(
        self, conn_manager: RedisConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = validate_vault(vault)
        self._migrated = vault != DEFAULT_VAULT

    @property
    def vault(self) -> str:
        return self._vault

    def _key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctxv:{self._vault}:{item_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}ctxv-ids:{self._vault}"

    def _client(self) -> Any:
        client = self._conn_manager.get_client()
        if not self._migrated:
            self._migrated = True
            prefix = self._conn_manager.prefix
            legacy_ids = client.smembers(f"{prefix}{_LEGACY_IDS}")
            if legacy_ids:
                pipe = client.pipeline()
                for op, *args in _migration_steps(
                    prefix, legacy_ids, self._key, self._ids_key(),
                ):
                    getattr(pipe, op)(*args)
                pipe.execute(raise_on_error=False)
        return client

    def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        client = self._client()
        data = item.model_dump_json()
        pipe = client.pipeline()
        pipe.set(self._key(item.id), data)
        pipe.sadd(self._ids_key(), item.id)
        pipe.execute()

    def get(self, item_id: str) -> ContextItem | None:
        data = self._client().get(self._key(item_id))
        if data is None:
            return None
        return ContextItem.model_validate_json(data)

    def get_all(self) -> list[ContextItem]:
        client = self._client()
        ids = client.smembers(self._ids_key())
        if not ids:
            return []
        keys = [self._key(id_) for id_ in ids]
        values = client.mget(keys)
        return [ContextItem.model_validate_json(v) for v in values if v is not None]

    def delete(self, item_id: str) -> bool:
        pipe = self._client().pipeline()
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
        client = self._client()
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

    __slots__ = ("_conn_manager", "_migrated", "_vault")

    def __init__(
        self, conn_manager: RedisConnectionManager, *, vault: str = DEFAULT_VAULT,
    ) -> None:
        self._conn_manager = conn_manager
        self._vault = validate_vault(vault)
        self._migrated = vault != DEFAULT_VAULT

    @property
    def vault(self) -> str:
        return self._vault

    def _key(self, item_id: str) -> str:
        return f"{self._conn_manager.prefix}ctxv:{self._vault}:{item_id}"

    def _ids_key(self) -> str:
        return f"{self._conn_manager.prefix}ctxv-ids:{self._vault}"

    async def _client(self) -> Any:
        client = self._conn_manager.get_async_client()
        if not self._migrated:
            self._migrated = True
            prefix = self._conn_manager.prefix
            legacy_ids = await client.smembers(f"{prefix}{_LEGACY_IDS}")
            if legacy_ids:
                pipe = client.pipeline()
                for op, *args in _migration_steps(
                    prefix, legacy_ids, self._key, self._ids_key(),
                ):
                    getattr(pipe, op)(*args)
                await pipe.execute(raise_on_error=False)
        return client

    async def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        client = await self._client()
        data = item.model_dump_json()
        pipe = client.pipeline()
        pipe.set(self._key(item.id), data)
        pipe.sadd(self._ids_key(), item.id)
        await pipe.execute()

    async def get(self, item_id: str) -> ContextItem | None:
        client = await self._client()
        data = await client.get(self._key(item_id))
        if data is None:
            return None
        return ContextItem.model_validate_json(data)

    async def get_all(self) -> list[ContextItem]:
        client = await self._client()
        ids = await client.smembers(self._ids_key())
        if not ids:
            return []
        keys = [self._key(id_) for id_ in ids]
        values = await client.mget(keys)
        return [ContextItem.model_validate_json(v) for v in values if v is not None]

    async def delete(self, item_id: str) -> bool:
        client = await self._client()
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
        client = await self._client()
        ids = await client.smembers(self._ids_key())
        if ids:
            keys = [self._key(id_) for id_ in ids]
            keys.append(self._ids_key())
            await client.delete(*keys)
        else:
            await client.delete(self._ids_key())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self._conn_manager.prefix!r})"
