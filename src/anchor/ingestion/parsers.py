"""Built-in document parsers for common file formats.

All parsers implement the ``DocumentParser`` protocol and return
``(text, metadata)`` tuples.  Zero external dependencies except
``PDFParser`` which requires the ``pypdf`` optional extra.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser as _StdlibHTMLParser
from pathlib import Path
from typing import Any

from anchor.exceptions import IngestionError

logger = logging.getLogger(__name__)

_ENCODINGS = ("utf-8", "latin-1", "cp1252")


def _read_text(source: Path | bytes) -> str:
    """Read text from a file path or raw bytes with encoding fallback."""
    raw = source if isinstance(source, bytes) else source.read_bytes()

    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue

    return raw.decode("utf-8", errors="replace")


class PlainTextParser:
    """Parse plain text files.

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        """Parse a plain text file.

        Parameters:
            source: File path or raw bytes.

        Returns:
            ``(text, metadata)`` tuple.
        """
        text = _read_text(source)
        metadata: dict[str, Any] = {}
        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix
        lines = text.splitlines()
        metadata["line_count"] = len(lines)
        return text, metadata

    def __repr__(self) -> str:
        return "PlainTextParser()"


class MarkdownParser:
    """Parse Markdown files, extracting headings as metadata.

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        """Parse a Markdown file.

        Extracts the first ``# heading`` as the title and collects all
        headings into metadata.

        Parameters:
            source: File path or raw bytes.

        Returns:
            ``(text, metadata)`` tuple.
        """
        text = _read_text(source)
        metadata: dict[str, Any] = {}

        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

        # Extract frontmatter if present — parsed into metadata, not discarded
        fm_match = self._FRONTMATTER_RE.match(text)
        if fm_match:
            metadata["has_frontmatter"] = True
            try:
                import yaml

                parsed = yaml.safe_load(fm_match.group(1))
                if isinstance(parsed, dict):
                    # Frontmatter values first; computed keys below win.
                    for key, value in parsed.items():
                        metadata.setdefault(str(key), value)
            except yaml.YAMLError:
                logger.warning("Invalid YAML frontmatter; skipping parse")
            # Remove frontmatter from content text
            text = text[fm_match.end() :]

        headings = self._HEADING_RE.findall(text)
        if headings:
            # First h1 as title
            for hashes, title in headings:
                if len(hashes) == 1:
                    metadata["title"] = title.strip()
                    break
            metadata["headings"] = [
                {"level": len(hashes), "text": title.strip()} for hashes, title in headings
            ]

        return text, metadata

    def __repr__(self) -> str:
        return "MarkdownParser()"


class _HTMLTextExtractor(_StdlibHTMLParser):
    """Internal HTML parser that extracts plain text and metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.title: str | None = None
        self._in_title = False
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = True
        elif lower_tag == "script":
            self._in_script = True
        elif lower_tag == "style":
            self._in_style = True
        elif lower_tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag == "title":
            self._in_title = False
        elif lower_tag == "script":
            self._in_script = False
        elif lower_tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return
        if self._in_title:
            self.title = data.strip()
        self.text_parts.append(data)


class HTMLParser:
    """Parse HTML files, stripping tags and extracting text.

    Uses Python's stdlib ``html.parser`` -- zero external dependencies.

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        """Parse an HTML file, stripping tags.

        Parameters:
            source: File path or raw bytes.

        Returns:
            ``(text, metadata)`` tuple with the title extracted if present.
        """
        raw_text = _read_text(source)
        metadata: dict[str, Any] = {}

        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

        extractor = _HTMLTextExtractor()
        extractor.feed(raw_text)

        text = "".join(extractor.text_parts)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if extractor.title:
            metadata["title"] = extractor.title

        return text, metadata

    def __repr__(self) -> str:
        return "HTMLParser()"


class PDFParser:
    """Parse PDF files using pypdf.

    Requires the ``pdf`` optional extra: ``pip install anchor[pdf]``

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        """Parse a PDF file and extract text and metadata.

        Parameters:
            source: File path or raw bytes.

        Returns:
            ``(text, metadata)`` tuple with page count and PDF info.

        Raises:
            IngestionError: If pypdf is not installed.
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            msg = (
                "pypdf is required for PDFParser. "
                "Install it with: pip install anchor[pdf]"
            )
            raise IngestionError(msg) from e

        import io

        if isinstance(source, bytes):
            reader = PdfReader(io.BytesIO(source))
        else:
            reader = PdfReader(str(source))

        pages_text: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        text = "\n\n".join(pages_text)
        metadata: dict[str, Any] = {"page_count": len(reader.pages)}

        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

        pdf_info = reader.metadata
        if pdf_info:
            if pdf_info.title:
                metadata["title"] = pdf_info.title
            if pdf_info.author:
                metadata["author"] = pdf_info.author

        return text, metadata

    def parse_pages(self, source: Path | bytes) -> list[tuple[int, str]]:
        """Parse a PDF into per-page texts for page-level provenance.

        Returns ``(page_number, text)`` tuples (1-based). Pages with no
        extractable text are skipped. ``DocumentIngester`` uses this to
        stamp every chunk with its source page, so citations can point at
        a real location instead of just the document.
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            msg = (
                "pypdf is required for PDFParser. "
                "Install it with: pip install anchor[pdf]"
            )
            raise IngestionError(msg) from e

        import io

        reader = (
            PdfReader(io.BytesIO(source))
            if isinstance(source, bytes)
            else PdfReader(str(source))
        )
        pages: list[tuple[int, str]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append((page_num, page_text))
        return pages

    def __repr__(self) -> str:
        return "PDFParser()"


class CSVParser:
    """Parse CSV/TSV files into a readable line-per-row text form.

    Uses the stdlib ``csv`` module — zero external dependencies.

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        import csv
        import io

        raw = _read_text(source)
        delimiter = "\t" if (
            isinstance(source, Path) and source.suffix.lower() == ".tsv"
        ) else ","
        rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))

        metadata: dict[str, Any] = {}
        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix
        if not rows:
            return "", metadata

        header = rows[0]
        metadata["columns"] = header
        metadata["row_count"] = len(rows) - 1

        # "col: value" lines per row keep header context in every chunk.
        lines: list[str] = []
        for row in rows[1:]:
            pairs = [
                f"{col}: {val}"
                for col, val in zip(header, row, strict=False)
                if val.strip()
            ]
            if pairs:
                lines.append("; ".join(pairs))
        return "\n".join(lines), metadata

    def __repr__(self) -> str:
        return "CSVParser()"


class JSONParser:
    """Parse JSON files into indented text.

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".json"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        import json

        raw = _read_text(source)
        metadata: dict[str, Any] = {}
        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            raise IngestionError(msg) from e

        if isinstance(data, dict):
            metadata["top_level_keys"] = sorted(data.keys())
        elif isinstance(data, list):
            metadata["item_count"] = len(data)

        return json.dumps(data, indent=2, ensure_ascii=False), metadata

    def __repr__(self) -> str:
        return "JSONParser()"


_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxParser:
    """Parse .docx files via stdlib zipfile + ElementTree — no python-docx.

    Extracts paragraph text (``w:t`` runs) from ``word/document.xml``.
    Formatting, tables-as-structure, images, and headers/footers are out
    of scope; for layout-faithful parsing use a dedicated parser (Docling).

    Implements the ``DocumentParser`` protocol.
    """

    __slots__ = ()

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx"]

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        import io
        import zipfile
        from xml.etree import ElementTree

        metadata: dict[str, Any] = {}
        if isinstance(source, Path):
            metadata["filename"] = source.name
            metadata["extension"] = source.suffix

        buffer = io.BytesIO(source) if isinstance(source, bytes) else str(source)
        try:
            with zipfile.ZipFile(buffer) as zf:
                xml_bytes = zf.read("word/document.xml")
        except (zipfile.BadZipFile, KeyError) as e:
            msg = f"Not a valid .docx file: {e}"
            raise IngestionError(msg) from e

        # Entity-expansion attacks (XXE / billion laughs) all require a DTD;
        # legitimate Word documents never put one in document.xml. Rejecting
        # DTDs outright closes both vectors without a defusedxml dependency.
        if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
            msg = "Refusing .docx with a DTD in document.xml (entity-expansion risk)"
            raise IngestionError(msg)

        try:
            root = ElementTree.fromstring(xml_bytes)  # noqa: S314 -- DTDs rejected above
        except ElementTree.ParseError as e:
            msg = f"Invalid document.xml in .docx: {e}"
            raise IngestionError(msg) from e

        paragraphs: list[str] = []
        for para in root.iter(f"{_DOCX_NS}p"):
            runs = [node.text or "" for node in para.iter(f"{_DOCX_NS}t")]
            text = "".join(runs).strip()
            if text:
                paragraphs.append(text)

        metadata["paragraph_count"] = len(paragraphs)
        return "\n\n".join(paragraphs), metadata

    def __repr__(self) -> str:
        return "DocxParser()"
