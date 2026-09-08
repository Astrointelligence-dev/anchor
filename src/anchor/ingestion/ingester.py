"""Document ingestion orchestrator.

Combines a parser, chunker, metadata generator, and tokenizer to
convert raw documents into ``ContextItem`` objects ready for indexing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anchor.exceptions import IngestionError
from anchor.ingestion.chunkers import RecursiveCharacterChunker
from anchor.ingestion.metadata import (
    MetadataEnricher,
    extract_chunk_metadata,
    generate_chunk_id,
    generate_doc_id,
)
from anchor.ingestion.parsers import (
    CSVParser,
    DocxParser,
    HTMLParser,
    JSONParser,
    MarkdownParser,
    PDFParser,
    PlainTextParser,
)
from anchor.models.context import ContextItem, SourceType
from anchor.protocols.ingestion import Chunker, DocumentParser
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

logger = logging.getLogger(__name__)

_EXTENSION_PARSER_MAP: dict[str, DocumentParser] = {}


def _get_default_parser_map() -> dict[str, DocumentParser]:
    """Build the default extension-to-parser mapping lazily."""
    if not _EXTENSION_PARSER_MAP:
        for parser in (
            PlainTextParser(),
            MarkdownParser(),
            HTMLParser(),
            PDFParser(),
            CSVParser(),
            JSONParser(),
            DocxParser(),
        ):
            for ext in parser.supported_extensions:
                _EXTENSION_PARSER_MAP[ext] = parser
    return _EXTENSION_PARSER_MAP


class DocumentIngester:
    """Orchestrates document parsing, chunking, and metadata extraction.

    Converts raw files or text into ``ContextItem`` objects suitable
    for ``retriever.index(items)``.
    """

    __slots__ = (
        "_chunker",
        "_contextualize_fn",
        "_enricher",
        "_parsers",
        "_priority",
        "_source_type",
        "_tokenizer",
    )

    def __init__(
        self,
        chunker: Chunker | None = None,
        tokenizer: Tokenizer | None = None,
        parsers: dict[str, DocumentParser] | None = None,
        enricher: MetadataEnricher | None = None,
        source_type: SourceType = SourceType.RETRIEVAL,
        priority: int = 5,
        contextualize_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        """See class docstring.

        Parameters:
            contextualize_fn: Optional Anthropic-style contextual-retrieval
                hook ``(document_text, chunk_text) -> context``. The
                returned context is prepended to the chunk before indexing
                (the original chunk is kept in ``metadata["original_content"]``).
                Typically an LLM call written by the caller — anchor never
                calls a model itself. Cheap with prompt caching since the
                document prefix repeats across chunks.
        """
        self._chunker: Chunker = chunker or RecursiveCharacterChunker()
        self._tokenizer = tokenizer or get_default_counter()
        self._enricher = enricher
        self._source_type = source_type
        self._priority = priority
        self._contextualize_fn = contextualize_fn

        # Merge default parsers with user overrides
        self._parsers: dict[str, DocumentParser] = dict(_get_default_parser_map())
        if parsers:
            self._parsers.update(parsers)

    def ingest_text(
        self,
        text: str,
        doc_id: str | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> list[ContextItem]:
        """Ingest raw text into context items.

        Parameters:
            text: The document text to ingest.
            doc_id: Optional document ID. Generated from content if not provided.
            doc_metadata: Optional document-level metadata.

        Returns:
            A list of ``ContextItem`` objects, one per chunk.
        """
        if not text or not text.strip():
            return []

        doc_id = doc_id or generate_doc_id(text)

        if hasattr(self._chunker, "chunk_with_metadata"):
            chunks_with_meta = self._chunker.chunk_with_metadata(text, doc_metadata)
            if not chunks_with_meta:
                return []
            return self._build_items_with_metadata(
                chunks_with_meta, doc_id, doc_metadata, doc_text=text
            )

        chunks = self._chunker.chunk(text, doc_metadata)

        if not chunks:
            return []

        return self._build_items(chunks, doc_id, doc_metadata, doc_text=text)

    def ingest_file(
        self,
        path: Path | str,
        doc_id: str | None = None,
    ) -> list[ContextItem]:
        """Ingest a single file into context items.

        The parser is auto-detected from the file extension.  Override
        via the ``parsers`` constructor argument.

        Parameters:
            path: Path to the file to ingest.
            doc_id: Optional document ID. Generated from content + path if not provided.

        Returns:
            A list of ``ContextItem`` objects, one per chunk.

        Raises:
            IngestionError: If no parser is found for the file extension.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        ext = path.suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            msg = f"No parser registered for extension '{ext}'. Supported: {list(self._parsers)}"
            raise IngestionError(msg)

        # Page-aware parsers (duck-typed `parse_pages`, e.g. PDFParser) get
        # per-page chunking so every chunk carries its source page number.
        if hasattr(parser, "parse_pages"):
            return self._ingest_pages(parser, path, doc_id)

        text, doc_metadata = parser.parse(path)
        if not text or not text.strip():
            logger.warning("Parser returned empty text for %s", path)
            return []

        doc_id = doc_id or generate_doc_id(text, str(path))

        if hasattr(self._chunker, "chunk_with_metadata"):
            chunks_with_meta = self._chunker.chunk_with_metadata(text, doc_metadata)
            if not chunks_with_meta:
                return []
            return self._build_items_with_metadata(
                chunks_with_meta, doc_id, doc_metadata, doc_text=text
            )

        chunks = self._chunker.chunk(text, doc_metadata)

        if not chunks:
            return []

        return self._build_items(chunks, doc_id, doc_metadata, doc_text=text)

    def _ingest_pages(
        self,
        parser: DocumentParser,
        path: Path,
        doc_id: str | None,
    ) -> list[ContextItem]:
        """Chunk a paginated document page by page, stamping ``doc_page``."""
        _, doc_metadata = parser.parse(path)
        pages = parser.parse_pages(path)  # type: ignore[attr-defined]
        if not pages:
            logger.warning("Parser returned no pages for %s", path)
            return []

        full_text = "\n\n".join(page_text for _, page_text in pages)
        doc_id = doc_id or generate_doc_id(full_text, str(path))

        chunks_with_meta: list[tuple[str, dict[str, Any]]] = []
        for page_num, page_text in pages:
            for chunk in self._chunker.chunk(page_text, doc_metadata):
                chunks_with_meta.append((chunk, {"doc_page": page_num}))

        if not chunks_with_meta:
            return []
        return self._build_items_with_metadata(
            chunks_with_meta, doc_id, doc_metadata, doc_text=full_text
        )

    def ingest_directory(
        self,
        directory: Path | str,
        glob_pattern: str = "**/*",
        extensions: list[str] | None = None,
    ) -> list[ContextItem]:
        """Ingest all matching files in a directory.

        Parameters:
            directory: Root directory to scan.
            glob_pattern: Glob pattern for file discovery. Defaults to recursive.
            extensions: Optional list of extensions to filter (e.g. ``[".md", ".txt"]``).
                If ``None``, uses all registered parser extensions.

        Returns:
            A list of ``ContextItem`` objects from all ingested files.

        Raises:
            IngestionError: If the directory does not exist.
        """
        directory = Path(directory)
        if not directory.is_dir():
            msg = f"Directory not found: {directory}"
            raise IngestionError(msg)

        allowed_exts = set(extensions) if extensions else set(self._parsers)
        items: list[ContextItem] = []

        for file_path in sorted(directory.glob(glob_pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_exts:
                continue
            try:
                file_items = self.ingest_file(file_path)
                items.extend(file_items)
            except (IngestionError, FileNotFoundError) as exc:
                logger.warning("Skipping %s: %s", file_path, exc)

        logger.info("Ingested %d items from %s", len(items), directory)
        return items

    def _contextualize(
        self, chunk_text: str, doc_text: str | None, metadata: dict[str, Any]
    ) -> str:
        """Apply the contextual-retrieval hook, if configured."""
        if self._contextualize_fn is None or doc_text is None:
            return chunk_text
        context = self._contextualize_fn(doc_text, chunk_text)
        if not context or not context.strip():
            return chunk_text
        metadata["original_content"] = chunk_text
        return f"{context.strip()}\n\n{chunk_text}"

    def _build_items(
        self,
        chunks: list[str],
        doc_id: str,
        doc_metadata: dict[str, Any] | None,
        doc_text: str | None = None,
    ) -> list[ContextItem]:
        """Convert text chunks into ContextItem objects."""
        items: list[ContextItem] = []
        total = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = generate_chunk_id(doc_id, idx)
            metadata = extract_chunk_metadata(
                chunk_text, idx, total, doc_id, doc_metadata,
            )

            if self._enricher:
                metadata = self._enricher.enrich(chunk_text, idx, total, metadata)

            content = self._contextualize(chunk_text, doc_text, metadata)
            token_count = self._tokenizer.count_tokens(content)

            item = ContextItem(
                id=chunk_id,
                content=content,
                source=self._source_type,
                priority=self._priority,
                token_count=token_count,
                metadata=metadata,
            )
            items.append(item)

        return items

    def _build_items_with_metadata(
        self,
        chunks_with_meta: list[tuple[str, dict[str, Any]]],
        doc_id: str,
        doc_metadata: dict[str, Any] | None,
        doc_text: str | None = None,
    ) -> list[ContextItem]:
        """Convert (text, metadata) tuples into ContextItem objects.

        Used when the chunker provides its own metadata (e.g.
        ``ParentChildChunker.chunk_with_metadata``).

        Parameters:
            chunks_with_meta: List of ``(text, metadata)`` tuples.
            doc_id: The parent document ID.
            doc_metadata: Optional document-level metadata.

        Returns:
            A list of ``ContextItem`` objects.
        """
        items: list[ContextItem] = []
        total = len(chunks_with_meta)

        for idx, (chunk_text, chunk_meta) in enumerate(chunks_with_meta):
            chunk_id = generate_chunk_id(doc_id, idx)

            # Start with standard metadata, then layer chunker metadata on top
            metadata = extract_chunk_metadata(
                chunk_text, idx, total, doc_id, doc_metadata,
            )
            metadata.update(chunk_meta)

            if self._enricher:
                metadata = self._enricher.enrich(chunk_text, idx, total, metadata)

            content = self._contextualize(chunk_text, doc_text, metadata)
            token_count = self._tokenizer.count_tokens(content)

            item = ContextItem(
                id=chunk_id,
                content=content,
                source=self._source_type,
                priority=self._priority,
                token_count=token_count,
                metadata=metadata,
            )
            items.append(item)

        return items

    def __repr__(self) -> str:
        return (
            f"DocumentIngester(chunker={self._chunker!r}, "
            f"parsers={list(self._parsers)})"
        )
