"""Tests for DocumentIngester orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anchor.ingestion.chunkers import FixedSizeChunker
from anchor.ingestion.ingester import DocumentIngester
from anchor.ingestion.metadata import MetadataEnricher, generate_doc_id
from anchor.models.context import ContextItem, SourceType
from tests.conftest import FakeTokenizer


@pytest.fixture
def ingester(fake_tokenizer: FakeTokenizer) -> DocumentIngester:
    chunker = FixedSizeChunker(chunk_size=10, overlap=0, tokenizer=fake_tokenizer)
    return DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)


class TestIngestText:
    """Tests for DocumentIngester.ingest_text."""

    def test_basic_ingestion(self, ingester: DocumentIngester) -> None:
        text = "hello world this is a test"
        items = ingester.ingest_text(text)
        assert len(items) >= 1
        assert all(isinstance(item, ContextItem) for item in items)

    def test_empty_text(self, ingester: DocumentIngester) -> None:
        assert ingester.ingest_text("") == []
        assert ingester.ingest_text("   ") == []

    def test_deterministic_ids(self, ingester: DocumentIngester) -> None:
        text = "hello world"
        items1 = ingester.ingest_text(text)
        items2 = ingester.ingest_text(text)
        assert items1[0].id == items2[0].id

    def test_custom_doc_id(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world", doc_id="my-doc")
        assert items[0].id == "my-doc-chunk-0"

    def test_source_type(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world")
        assert all(item.source == SourceType.RETRIEVAL for item in items)

    def test_metadata_propagation(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text(
            "hello world",
            doc_metadata={"title": "Test Doc"},
        )
        assert items[0].metadata["doc_title"] == "Test Doc"
        assert items[0].metadata["parent_doc_id"] is not None

    def test_token_count_populated(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world this is a test")
        for item in items:
            assert item.token_count > 0

    def test_chunk_metadata_fields(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world")
        meta = items[0].metadata
        assert "parent_doc_id" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "word_count" in meta
        assert "char_count" in meta

    def test_metadata_enricher(self, fake_tokenizer: FakeTokenizer) -> None:
        def add_lang(text: str, idx: int, total: int, meta: dict[str, Any]) -> dict[str, Any]:
            return {**meta, "language": "en"}

        enricher = MetadataEnricher(enrichers=[add_lang])
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer, enricher=enricher)
        items = ingester.ingest_text("hello world")
        assert items[0].metadata["language"] == "en"


class TestIngestFile:
    """Tests for DocumentIngester.ingest_file."""

    def test_ingest_txt_file(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Hello world this is content", encoding="utf-8")
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_file(f)
        assert len(items) >= 1
        assert "Hello" in items[0].content

    def test_ingest_md_file(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content here.", encoding="utf-8")
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_file(f)
        assert len(items) >= 1

    def test_ingest_html_file(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Content</p></body></html>", encoding="utf-8")
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_file(f)
        assert len(items) >= 1

    def test_file_not_found(self, ingester: DocumentIngester) -> None:
        with pytest.raises(FileNotFoundError):
            ingester.ingest_file(Path("/nonexistent/file.txt"))

    def test_unsupported_extension(self, tmp_path: Path, ingester: DocumentIngester) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("content", encoding="utf-8")
        from anchor.exceptions import IngestionError

        with pytest.raises(IngestionError, match="No parser registered"):
            ingester.ingest_file(f)

    def test_deterministic_ids_from_file(
        self, tmp_path: Path, fake_tokenizer: FakeTokenizer
    ) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items1 = ingester.ingest_file(f)
        items2 = ingester.ingest_file(f)
        assert items1[0].id == items2[0].id

    def test_string_path(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_file(str(f))
        assert len(items) >= 1


class TestIngestDirectory:
    """Tests for DocumentIngester.ingest_directory."""

    def test_ingest_directory(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        (tmp_path / "a.txt").write_text("First document content", encoding="utf-8")
        (tmp_path / "b.md").write_text("# Second\n\nMore content here", encoding="utf-8")
        (tmp_path / "c.xyz").write_text("Ignored file", encoding="utf-8")

        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_directory(tmp_path)
        # Should ingest .txt and .md but skip .xyz
        assert len(items) >= 2

    def test_filter_by_extension(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        (tmp_path / "a.txt").write_text("Text file", encoding="utf-8")
        (tmp_path / "b.md").write_text("Markdown file", encoding="utf-8")

        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_directory(tmp_path, extensions=[".txt"])
        # Should only ingest .txt
        contents = " ".join(item.content for item in items)
        assert "Text file" in contents

    def test_directory_not_found(self, ingester: DocumentIngester) -> None:
        from anchor.exceptions import IngestionError

        with pytest.raises(IngestionError, match="Directory not found"):
            ingester.ingest_directory(Path("/nonexistent/dir"))

    def test_recursive_glob(self, tmp_path: Path, fake_tokenizer: FakeTokenizer) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content", encoding="utf-8")

        chunker = FixedSizeChunker(chunk_size=50, overlap=0, tokenizer=fake_tokenizer)
        ingester = DocumentIngester(chunker=chunker, tokenizer=fake_tokenizer)
        items = ingester.ingest_directory(tmp_path)
        contents = " ".join(item.content for item in items)
        assert "Nested content" in contents


class TestContextItemCompatibility:
    """Verify ingested items are compatible with retriever.index()."""

    def test_items_have_required_fields(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world test content")
        for item in items:
            assert item.id
            assert item.content
            assert item.source == SourceType.RETRIEVAL
            assert item.token_count >= 0
            assert isinstance(item.metadata, dict)

    def test_items_are_frozen(self, ingester: DocumentIngester) -> None:
        items = ingester.ingest_text("hello world")
        with pytest.raises(Exception):  # noqa: B017
            items[0].content = "modified"  # type: ignore[misc]


class TestDocumentIngesterRepr:
    """Test repr output."""

    def test_repr(self, ingester: DocumentIngester) -> None:
        r = repr(ingester)
        assert "DocumentIngester" in r
        assert "FixedSizeChunker" in r


class TestMetadataModule:
    """Tests for metadata utility functions."""

    def test_generate_doc_id_deterministic(self) -> None:
        id1 = generate_doc_id("hello", "path.txt")
        id2 = generate_doc_id("hello", "path.txt")
        assert id1 == id2
        assert len(id1) == 16

    def test_generate_doc_id_varies_with_content(self) -> None:
        id1 = generate_doc_id("hello")
        id2 = generate_doc_id("world")
        assert id1 != id2

    def test_generate_doc_id_varies_with_path(self) -> None:
        id1 = generate_doc_id("hello", "a.txt")
        id2 = generate_doc_id("hello", "b.txt")
        assert id1 != id2

    def test_metadata_enricher_chain(self) -> None:
        def add_a(text: str, idx: int, total: int, meta: dict[str, Any]) -> dict[str, Any]:
            return {**meta, "a": True}

        def add_b(text: str, idx: int, total: int, meta: dict[str, Any]) -> dict[str, Any]:
            return {**meta, "b": True}

        enricher = MetadataEnricher(enrichers=[add_a, add_b])
        result = enricher.enrich("text", 0, 1, {})
        assert result["a"] is True
        assert result["b"] is True

    def test_metadata_enricher_add(self) -> None:
        enricher = MetadataEnricher()

        def add_c(text: str, idx: int, total: int, meta: dict[str, Any]) -> dict[str, Any]:
            return {**meta, "c": True}

        enricher.add(add_c)
        result = enricher.enrich("text", 0, 1, {})
        assert result["c"] is True

    def test_metadata_enricher_repr(self) -> None:
        enricher = MetadataEnricher()
        assert "MetadataEnricher(enrichers=0)" in repr(enricher)
