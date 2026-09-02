"""Front #3, fase 3: migração de dados pré-vault.

Failing-first: um banco no shape ANTIGO (sem colunas vault/namespace)
precisa ganhar as colunas idempotentemente via ensure_tables, com os
dados existentes preservados sob o vault sentinela __default__; o
vec0 antigo (sem partition key) precisa ser reconstruído por cópia de
blob na inicialização do store.
"""

from __future__ import annotations

import sqlite3

import pytest

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import DEFAULT_VAULT
from tests.conftest import make_embedding

_OLD_CONTEXT_DDL = """CREATE TABLE context_items (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    score       REAL NOT NULL DEFAULT 0.0,
    priority    INTEGER NOT NULL DEFAULT 5,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
)"""

_OLD_EMBEDDINGS_DDL = """CREATE TABLE embeddings (
    item_id       TEXT PRIMARY KEY,
    embedding_blob BLOB NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
)"""


class TestSqliteEnsureTablesMigrates:
    def _make_old_db(self, path):
        conn = sqlite3.connect(str(path))
        conn.execute(_OLD_CONTEXT_DDL)
        conn.execute(_OLD_EMBEDDINGS_DDL)
        conn.execute(
            "INSERT INTO context_items "
            "(id, content, source, created_at) "
            "VALUES ('old1', 'legado', 'retrieval', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

    def test_old_db_gains_scope_columns_and_keeps_data(self, tmp_path):
        from anchor.storage.sqlite import (
            SqliteConnectionManager,
            SqliteContextStore,
            ensure_tables,
        )

        db = tmp_path / "old.db"
        self._make_old_db(db)

        mgr = SqliteConnectionManager(db)
        ensure_tables(mgr.get_connection())  # pre-fix: colunas não aparecem

        store = SqliteContextStore(mgr)
        got = store.get("old1")
        assert got is not None
        assert got.content == "legado"
        assert got.vault == DEFAULT_VAULT
        assert got.namespace == "/"

        # E escrita nova funciona com as colunas.
        store.add(ContextItem(id="new1", content="x", source=SourceType.RETRIEVAL))
        assert store.get("new1") is not None

    def test_ensure_tables_is_idempotent_on_migrated_db(self, tmp_path):
        from anchor.storage.sqlite import SqliteConnectionManager, ensure_tables

        db = tmp_path / "old2.db"
        self._make_old_db(db)
        mgr = SqliteConnectionManager(db)
        ensure_tables(mgr.get_connection())
        ensure_tables(mgr.get_connection())  # segunda passada: sem erro
        cols = {
            r[1]
            for r in mgr.get_connection().execute(
                "PRAGMA table_info(context_items)"
            ).fetchall()
        }
        assert {"vault", "namespace"} <= cols


class TestVecStoreMigratesOldShape:
    def test_old_vec_tables_are_rebuilt_with_data(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        import sqlite_vec

        db = tmp_path / "oldvec.db"
        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Shape antigo: vec_items sem vault/namespace, vec0 sem partition key.
        conn.execute(
            "CREATE TABLE vec_items ("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
            "item_id TEXT UNIQUE NOT NULL, "
            "metadata_json TEXT NOT NULL DEFAULT '{}')"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE vec_index USING vec0("
            "embedding float[8] distance_metric=cosine)"
        )
        emb = make_embedding(1, dim=8)
        cur = conn.execute(
            "INSERT INTO vec_items (item_id, metadata_json) VALUES ('v1', '{}')"
        )
        conn.execute(
            "INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, sqlite_vec.serialize_float32(emb)),
        )
        conn.commit()
        conn.close()

        from anchor.storage.sqlite import SqliteVecVectorStore

        store = SqliteVecVectorStore(db, dimensions=8)  # pre-fix: quebra/perde
        results = store.search(emb, top_k=5)
        assert [i for i, _ in results] == ["v1"]

        # E o dado migrado pertence ao vault default — outro vault não vê.
        other = SqliteVecVectorStore(db, dimensions=8, vault="outro")
        assert other.search(emb, top_k=5) == []


_OLD_VEC_ITEMS_DDL = (
    "CREATE TABLE vec_items ("
    "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
    "item_id TEXT UNIQUE NOT NULL, "
    "metadata_json TEXT NOT NULL DEFAULT '{}')"
)


def _make_old_vec_db(path, dim: int = 8) -> list[float]:
    import sqlite_vec

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_OLD_VEC_ITEMS_DDL)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_index USING vec0("
        f"embedding float[{dim}] distance_metric=cosine)"
    )
    emb = make_embedding(1, dim=dim)
    cur = conn.execute(
        "INSERT INTO vec_items (item_id, metadata_json) VALUES ('v1', '{}')"
    )
    conn.execute(
        "INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
        (cur.lastrowid, sqlite_vec.serialize_float32(emb)),
    )
    conn.commit()
    conn.close()
    return emb


class TestLegacyDbKeepsVaultsApart:
    """Ritual xhigh #1/#9: a migrated DB must key rows on (vault, id).

    Pre-fix the PK stayed (id), so INSERT OR REPLACE from one vault
    deleted the other vault's row (context_items, embeddings) and the
    legacy UNIQUE(item_id) on vec_items rejected the second vault.
    """

    def _old_db(self, path):
        TestSqliteEnsureTablesMigrates()._make_old_db(path)

    def test_context_and_vector_rows_coexist_across_vaults(self, tmp_path):
        from anchor.storage.sqlite import (
            SqliteConnectionManager,
            SqliteContextStore,
            SqliteVectorStore,
            ensure_tables,
        )

        db = tmp_path / "legacy.db"
        self._old_db(db)
        mgr = SqliteConnectionManager(db)
        ensure_tables(mgr.get_connection())

        for vault, text in (("a", "A"), ("b", "B")):
            SqliteContextStore(mgr, vault=vault).add(
                ContextItem(id="d1", content=text, source=SourceType.RETRIEVAL)
            )
            SqliteVectorStore(mgr, vault=vault).add_embedding("d1", make_embedding(1))

        assert SqliteContextStore(mgr, vault="a").get("d1").content == "A"
        assert SqliteContextStore(mgr, vault="b").get("d1").content == "B"
        assert SqliteContextStore(mgr).get("old1").content == "legado"
        hits = SqliteVectorStore(mgr, vault="a").search(make_embedding(1), top_k=5)
        assert [i for i, _ in hits] == ["d1"]

    def test_async_twin_rebuilds_too(self, tmp_path):
        import asyncio

        from anchor.storage.sqlite import (
            AsyncSqliteContextStore,
            SqliteConnectionManager,
            ensure_tables_async,
        )

        db = tmp_path / "legacy-async.db"
        self._old_db(db)
        mgr = SqliteConnectionManager(db)

        async def scenario():
            await ensure_tables_async(await mgr.get_async_connection())
            for vault, text in (("a", "A"), ("b", "B")):
                await AsyncSqliteContextStore(mgr, vault=vault).add(
                    ContextItem(id="d1", content=text, source=SourceType.RETRIEVAL)
                )
            a = await AsyncSqliteContextStore(mgr, vault="a").get("d1")
            old = await AsyncSqliteContextStore(mgr).get("old1")
            await mgr.aclose()
            return a.content, old.content

        assert asyncio.run(scenario()) == ("A", "legado")

    def test_vec_items_accepts_same_id_in_second_vault(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from anchor.storage.sqlite import SqliteVecVectorStore

        db = tmp_path / "legacy-vec.db"
        emb = _make_old_vec_db(db)
        default = SqliteVecVectorStore(db, dimensions=8)
        other = SqliteVecVectorStore(db, dimensions=8, vault="outro")
        other.add_embedding("v1", make_embedding(2, dim=8))

        assert [i for i, _ in default.search(emb, top_k=5)] == ["v1"]
        assert [i for i, _ in other.search(make_embedding(2, dim=8), top_k=5)] == ["v1"]
        assert default.count() == 1
        assert other.count() == 1


class TestVecRebuildIsSafe:
    """Ritual xhigh #3: a wrong-dimension open must never touch the index."""

    def test_wrong_dimension_raises_before_any_ddl(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from anchor.storage.sqlite import SqliteVecVectorStore

        db = tmp_path / "legacy-vec.db"
        emb = _make_old_vec_db(db, dim=8)
        with pytest.raises(ValueError, match="8-dimensional"):
            SqliteVecVectorStore(db, dimensions=16)

        # Still the untouched legacy index: the right dimension migrates it.
        store = SqliteVecVectorStore(db, dimensions=8)
        assert [i for i, _ in store.search(emb, top_k=5)] == ["v1"]

    def test_dimension_is_read_from_the_index(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from anchor.storage.sqlite import SqliteVecVectorStore

        db = tmp_path / "legacy-vec.db"
        emb = _make_old_vec_db(db, dim=8)
        store = SqliteVecVectorStore(db)
        assert [i for i, _ in store.search(emb, top_k=5)] == ["v1"]
        with pytest.raises(ValueError, match="required"):
            SqliteVecVectorStore(tmp_path / "fresh.db")
