"""Phase 3 regression tests: ingestion/parsing quality upgrades.

Pins docs/plans/2026-08-25-sota-upgrade-plan.md Phase 3 (ingestion side):
markdown header propagation, per-page PDF provenance, frontmatter parsing,
new csv/json/docx parsers, contextual enrichment, evidence-based defaults.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from anchor.exceptions import IngestionError
from anchor.ingestion.chunkers import MarkdownHeaderChunker, RecursiveCharacterChunker
from anchor.ingestion.ingester import DocumentIngester
from anchor.ingestion.parsers import CSVParser, DocxParser, JSONParser, MarkdownParser
from tests.conftest import FakeTokenizer


def _ingester(**kwargs: Any) -> DocumentIngester:
    return DocumentIngester(
        chunker=kwargs.pop(
            "chunker", RecursiveCharacterChunker(tokenizer=FakeTokenizer())
        ),
        tokenizer=FakeTokenizer(),
        **kwargs,
    )


class TestChunkingDefaults:
    def test_recursive_defaults_follow_chroma_evidence(self) -> None:
        chunker = RecursiveCharacterChunker(tokenizer=FakeTokenizer())
        assert chunker._chunk_size == 384
        assert chunker._overlap == 0


class TestMarkdownHeaderChunker:
    _DOC = (
        "intro line\n"
        "# Guide\n"
        "guide preamble\n"
        "## Setup\n"
        "install the thing\n"
        "## Usage\n"
        "run the thing\n"
        "### Advanced\n"
        "tweak the thing\n"
    )

    def test_header_paths_in_metadata(self) -> None:
        chunker = MarkdownHeaderChunker(tokenizer=FakeTokenizer())
        pairs = chunker.chunk_with_metadata(self._DOC)
        paths = {meta.get("headers", "") for _, meta in pairs}
        assert "Guide > Setup" in paths
        assert "Guide > Usage > Advanced" in paths
        assert "" in paths  # pre-header intro has no path

    def test_header_path_prepended_to_content(self) -> None:
        chunker = MarkdownHeaderChunker(tokenizer=FakeTokenizer())
        pairs = chunker.chunk_with_metadata(self._DOC)
        setup = next(c for c, m in pairs if m.get("headers") == "Guide > Setup")
        assert setup.startswith("Guide > Setup\n\n")
        assert "install the thing" in setup

    def test_content_prepend_can_be_disabled(self) -> None:
        chunker = MarkdownHeaderChunker(
            include_headers_in_content=False, tokenizer=FakeTokenizer()
        )
        pairs = chunker.chunk_with_metadata(self._DOC)
        setup = next(c for c, m in pairs if m.get("headers") == "Guide > Setup")
        assert not setup.startswith("Guide > Setup")

    def test_sibling_header_replaces_stack_level(self) -> None:
        chunker = MarkdownHeaderChunker(tokenizer=FakeTokenizer())
        pairs = chunker.chunk_with_metadata(self._DOC)
        usage = next(m for _, m in pairs if "run the thing" in _)
        assert usage["headers"] == "Guide > Usage"  # Setup popped, not stacked


class TestPageProvenance:
    class _FakePagedParser:
        """Duck-typed page-aware parser (same shape as PDFParser)."""

        @property
        def supported_extensions(self) -> list[str]:
            return [".fakepdf"]

        def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
            return "page one text\n\npage two text", {"page_count": 2}

        def parse_pages(self, source: Path | bytes) -> list[tuple[int, str]]:
            return [(1, "page one text"), (2, "page two text")]

    def test_chunks_carry_page_numbers(self, tmp_path: Path) -> None:
        doc = tmp_path / "report.fakepdf"
        doc.write_text("binary-ish")
        ingester = _ingester(parsers={".fakepdf": self._FakePagedParser()})
        items = ingester.ingest_file(doc)
        pages = {item.metadata["doc_page"] for item in items}
        assert pages == {1, 2}


class TestNewParsers:
    def test_markdown_frontmatter_parsed_not_discarded(self, tmp_path: Path) -> None:
        doc = tmp_path / "post.md"
        doc.write_text(
            "---\nauthor: Arthur\ntags: [a, b]\n---\n# Title\nbody text"
        )
        text, metadata = MarkdownParser().parse(doc)
        assert metadata["author"] == "Arthur"
        assert metadata["tags"] == ["a", "b"]
        assert metadata["has_frontmatter"] is True
        assert "author:" not in text

    def test_csv_parser(self, tmp_path: Path) -> None:
        doc = tmp_path / "data.csv"
        doc.write_text("name,role\nArthur,founder\nMax,collaborator\n")
        text, metadata = CSVParser().parse(doc)
        assert metadata["columns"] == ["name", "role"]
        assert metadata["row_count"] == 2
        assert "name: Arthur; role: founder" in text

    def test_json_parser(self, tmp_path: Path) -> None:
        doc = tmp_path / "config.json"
        doc.write_text('{"b": 1, "a": {"nested": true}}')
        text, metadata = JSONParser().parse(doc)
        assert metadata["top_level_keys"] == ["a", "b"]
        assert '"nested": true' in text

    @staticmethod
    def _make_docx(paragraphs: list[str], extra_xml: str = "") -> bytes:
        ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        body = "".join(
            f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
        )
        xml = f'<?xml version="1.0"?>{extra_xml}<w:document {ns}><w:body>{body}</w:body></w:document>'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("word/document.xml", xml)
        return buffer.getvalue()

    def test_docx_parser(self) -> None:
        raw = self._make_docx(["First paragraph.", "Second paragraph."])
        text, metadata = DocxParser().parse(raw)
        assert text == "First paragraph.\n\nSecond paragraph."
        assert metadata["paragraph_count"] == 2

    def test_docx_rejects_dtd(self) -> None:
        raw = self._make_docx(["x"], extra_xml="<!DOCTYPE lol [<!ENTITY a 'b'>]>")
        with pytest.raises(IngestionError, match="entity-expansion"):
            DocxParser().parse(raw)

    def test_docx_rejects_non_zip(self) -> None:
        with pytest.raises(IngestionError, match="valid .docx"):
            DocxParser().parse(b"not a zip file")


class TestContextualEnrichment:
    def test_context_prepended_and_original_kept(self) -> None:
        def contextualize(doc_text: str, chunk_text: str) -> str:
            return f"[From a doc of {len(doc_text)} chars]"

        ingester = _ingester(contextualize_fn=contextualize)
        items = ingester.ingest_text("alpha beta gamma delta")
        assert len(items) == 1
        assert items[0].content.startswith("[From a doc of ")
        assert items[0].metadata["original_content"] == "alpha beta gamma delta"

    def test_no_hook_leaves_content_untouched(self) -> None:
        items = _ingester().ingest_text("alpha beta gamma")
        assert items[0].content == "alpha beta gamma"
        assert "original_content" not in items[0].metadata
