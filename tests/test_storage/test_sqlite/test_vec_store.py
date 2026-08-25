"""Tests for the sqlite-vec backed vector store (Phase 3)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from anchor.storage.sqlite import SqliteVecVectorStore  # noqa: E402


@pytest.fixture()
def store() -> SqliteVecVectorStore:
    return SqliteVecVectorStore(":memory:", dimensions=3)


class TestSqliteVecVectorStore:
    def test_knn_ordering(self, store: SqliteVecVectorStore) -> None:
        store.add_embedding("exact", [1.0, 0.0, 0.0])
        store.add_embedding("close", [0.9, 0.1, 0.0])
        store.add_embedding("orthogonal", [0.0, 0.0, 1.0])
        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert [r[0] for r in results] == ["exact", "close", "orthogonal"]
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_where_prefilter(self, store: SqliteVecVectorStore) -> None:
        store.add_embedding("a", [1.0, 0.0, 0.0], {"tenant": "acme"})
        store.add_embedding("b", [0.9, 0.1, 0.0], {"tenant": "globex"})
        results = store.search([1.0, 0.0, 0.0], top_k=5, where={"tenant": "globex"})
        assert [r[0] for r in results] == ["b"]

    def test_where_no_match_returns_empty(self, store: SqliteVecVectorStore) -> None:
        store.add_embedding("a", [1.0, 0.0, 0.0], {"tenant": "acme"})
        assert store.search([1.0, 0.0, 0.0], top_k=5, where={"tenant": "nope"}) == []

    def test_upsert_overwrites(self, store: SqliteVecVectorStore) -> None:
        store.add_embedding("a", [1.0, 0.0, 0.0], {"v": 1})
        store.add_embedding("a", [0.0, 1.0, 0.0], {"v": 2})
        assert store.count() == 1
        results = store.search([0.0, 1.0, 0.0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_delete(self, store: SqliteVecVectorStore) -> None:
        store.add_embedding("a", [1.0, 0.0, 0.0])
        assert store.delete("a") is True
        assert store.delete("a") is False
        assert store.count() == 0

    def test_dimension_validation(self, store: SqliteVecVectorStore) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            store.add_embedding("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="dimensions"):
            store.search([1.0, 0.0], top_k=1)

    def test_declared_dim_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SqliteVecVectorStore(":memory:", dimensions=0)


class TestPortugueseBM25:
    """bm25s + PyStemmer: morphological variants must match in Portuguese."""

    def test_stemming_matches_inflected_forms(self) -> None:
        pytest.importorskip("bm25s")
        pytest.importorskip("Stemmer")
        from anchor.models.context import ContextItem, SourceType
        from anchor.models.query import QueryBundle
        from anchor.retrieval.sparse import SparseRetriever

        items = [
            ContextItem(
                id="run", content="o atleta estava correndo na praia ao amanhecer",
                source=SourceType.RETRIEVAL,
            ),
            ContextItem(
                id="cake", content="a receita de bolo leva farinha e ovos",
                source=SourceType.RETRIEVAL,
            ),
            ContextItem(
                id="sail", content="a regata de vela depende do vento e da maré",
                source=SourceType.RETRIEVAL,
            ),
        ]
        retriever = SparseRetriever(language="portuguese")
        retriever.index(items)
        # "correr" only matches "correndo" through Portuguese stemming
        results = retriever.retrieve(QueryBundle(query_str="correr"), top_k=3)
        assert results
        assert results[0].id == "run"
