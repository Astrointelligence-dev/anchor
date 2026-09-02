"""Tests for the anchor CLI (real index/query since Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("typer")
pytest.importorskip("bm25s")

from typer.testing import CliRunner

from anchor import __version__
from anchor.cli import app

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "retrieval.md").write_text(
        "# Retrieval\n\nHybrid retrieval fuses dense embeddings with BM25 "
        "sparse search using reciprocal rank fusion."
    )
    (docs / "memory.md").write_text(
        "# Memory\n\nThe memory manager supports sliding window and summary buffer strategies."
    )
    (docs / "noise.txt").write_text("Bananas are rich in potassium and sailing depends on wind.")
    return docs


class TestGeneralCLI:
    def test_version_flag_prints_version_and_exits(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_short_version_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.output
        assert "query" in result.output


class TestInfoCommand:
    def test_info_runs_successfully(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestIndexCommand:
    def test_index_directory(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(app, ["index", str(corpus), "--db", str(db)])
        assert result.exit_code == 0
        assert db.exists()
        assert "Chunks" in result.output

    def test_index_single_file(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(app, ["index", str(corpus / "retrieval.md"), "--db", str(db)])
        assert result.exit_code == 0
        assert db.exists()

    def test_index_nonexistent_path_exits_with_error(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["index", str(tmp_path / "missing"), "--db", str(tmp_path / "x.db")]
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_index_unknown_embeddings_spec(self, corpus: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "index",
                str(corpus),
                "--db",
                str(tmp_path / "x.db"),
                "--embeddings",
                "wat",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown embeddings spec" in result.output


class TestQueryCommand:
    def test_query_returns_relevant_result(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        assert runner.invoke(app, ["index", str(corpus), "--db", str(db)]).exit_code == 0

        result = runner.invoke(
            app, ["query", "how does hybrid retrieval work", "--db", str(db), "-k", "2"]
        )
        assert result.exit_code == 0
        assert "retrieval.md" in result.output
        assert "noise.txt" not in result.output

    def test_query_without_index_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["query", "anything", "--db", str(tmp_path / "missing.db")])
        assert result.exit_code == 1
        assert "No index" in result.output

    def test_query_top_k_limits_results(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        runner.invoke(app, ["index", str(corpus), "--db", str(db)])
        result = runner.invoke(
            app, ["query", "memory manager strategies", "--db", str(db), "-k", "1"]
        )
        assert result.exit_code == 0
        assert "Top 1" in result.output


class TestMigrate:
    def _legacy_db(self, path: Path) -> None:
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE context_items (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
            "source TEXT NOT NULL, score REAL NOT NULL DEFAULT 0.0, "
            "priority INTEGER NOT NULL DEFAULT 5, token_count INTEGER NOT NULL "
            "DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}', "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO context_items (id, content, source, created_at) VALUES "
            "('old1', 'legacy', 'retrieval', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

    def test_migrate_is_idempotent_and_keeps_rows(self, tmp_path: Path) -> None:
        db = tmp_path / "legacy.db"
        self._legacy_db(db)
        for _ in range(2):
            result = runner.invoke(app, ["migrate", "--db", str(db)])
            assert result.exit_code == 0, result.output
            assert "migrated" in result.output
        result = runner.invoke(app, ["query", "legacy", "--db", str(db)])
        assert result.exit_code == 0, result.output
        assert "legacy" in result.output

    def test_migrate_missing_db_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["migrate", "--db", str(tmp_path / "nope.db")])
        assert result.exit_code == 1


class TestScopeOptions:
    def test_invalid_vault_and_namespace_are_usage_errors(
        self,
        corpus: Path,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / "scope.db"
        for args, flag in (
            (["--vault", "a/b"], "--vault"),
            (["--vault", "a:b"], "--vault"),
            (["--namespace", " "], "--namespace"),
        ):
            result = runner.invoke(app, ["index", str(corpus), "--db", str(db), *args])
            assert result.exit_code == 1, result.output
            assert flag in result.output
        result = runner.invoke(app, ["query", "x", "--db", str(db), "--include", ""])
        assert result.exit_code == 1
        assert "--include" in result.output

    def test_vault_and_namespace_scope_the_query(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "scope.db"
        assert (
            runner.invoke(
                app,
                ["index", str(corpus), "--db", str(db), "--vault", "docs", "-n", "/kb/a"],
            ).exit_code
            == 0
        )
        hit = runner.invoke(app, ["query", "retrieval", "--db", str(db), "--vault", "docs"])
        assert hit.exit_code == 0
        assert "Retrieval" in hit.output
        other_vault = runner.invoke(app, ["query", "retrieval", "--db", str(db)])
        assert other_vault.exit_code == 1  # __default__ is empty
        excluded = runner.invoke(
            app,
            ["query", "retrieval", "--db", str(db), "--vault", "docs", "--exclude", "/kb"],
        )
        assert "Retrieval" not in excluded.output


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    docs = tmp_path / "wiki"
    (docs / "secret").mkdir(parents=True)
    (docs / "checkout.md").write_text(
        "---\naliases: [Checkout]\n---\n# Checkout\n\nCheckout hands cards to [[payments]] "
        "and scores orders with [[fraud]]."
    )
    (docs / "payments.md").write_text(
        "# Payments\n\nPayments talks to the acquirer; on-call is [[bruno]]."
    )
    (docs / "bruno.md").write_text("# Bruno\n\nBruno maintains [[payments]] and the ledger.")
    (docs / "secret" / "fraud.md").write_text(
        "# Fraud\n\nFraud scoring uses [[vendor]] to score [[checkout]] orders."
    )
    return docs


class TestGraphCLI:
    def _index(self, wiki: Path, db: Path, namespace: str | None = None) -> None:
        args = ["index", str(wiki), "--db", str(db), "--graph"]
        if namespace:
            args += ["--namespace", namespace]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Graph nodes" in result.output

    def test_index_with_graph_and_navigation(self, wiki: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        self._index(wiki, db)

        result = runner.invoke(app, ["graph", "path", "Checkout", "bruno", "--db", str(db)])
        assert result.exit_code == 0, result.output
        assert "checkout -> payments -> bruno" in result.output

        result = runner.invoke(app, ["graph", "explain", "checkout", "bruno", "--db", str(db)])
        assert result.exit_code == 0
        assert "links_to" in result.output
        assert "extracted" in result.output

        result = runner.invoke(app, ["graph", "hubs", "--db", str(db), "-k", "3"])
        assert result.exit_code == 0
        assert "payments" in result.output  # checkout/fraud/payments tie at degree 2

        result = runner.invoke(app, ["graph", "backlinks", "payments", "--db", str(db)])
        assert result.exit_code == 0
        assert "checkout" in result.output
        assert "bruno" in result.output

        result = runner.invoke(
            app, ["graph", "query", "who is on call for Checkout?", "--db", str(db)]
        )
        assert result.exit_code == 0, result.output
        assert "checkout.md" in result.output
        assert "payments.md" in result.output

    def test_scope_hides_a_folder(self, wiki: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        # two indexing passes stamp different namespaces on the two folders
        self._index(wiki / "secret", db, "/secret")
        for name in ("checkout.md", "payments.md", "bruno.md"):
            result = runner.invoke(
                app, ["index", str(wiki / name), "--db", str(db), "--graph", "-n", "/public"]
            )
            assert result.exit_code == 0, result.output
        # fraud is NAMED by the public checkout note, so the node survives the
        # scope; vendor is evidenced only by the secret note and vanishes.
        result = runner.invoke(app, ["graph", "path", "checkout", "vendor", "--db", str(db)])
        assert result.exit_code == 0
        assert "checkout -> fraud -> vendor" in result.output
        result = runner.invoke(
            app, ["graph", "path", "checkout", "vendor", "--db", str(db), "--exclude", "/secret"]
        )
        assert result.exit_code == 1
        assert "No path" in result.output
        result = runner.invoke(
            app, ["graph", "query", "fraud", "--db", str(db), "--exclude", "/secret"]
        )
        assert "fraud.md" not in result.output

    def test_graph_commands_need_an_index(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["graph", "hubs", "--db", str(tmp_path / "none.db")])
        assert result.exit_code == 1
        assert "anchor index --graph" in result.output

    def test_empty_graph_is_reported(self, corpus: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        assert runner.invoke(app, ["index", str(corpus), "--db", str(db)]).exit_code == 0
        result = runner.invoke(app, ["graph", "hubs", "--db", str(db)])
        assert result.exit_code == 1
        assert "empty" in result.output
