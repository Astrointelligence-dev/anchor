"""Document ingestion module for anchor.

Provides chunkers, parsers, metadata utilities, and an orchestrator
for converting raw documents into ``ContextItem`` objects.
"""

from .chunkers import (
    FixedSizeChunker,
    MarkdownHeaderChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceChunker,
)
from .code_chunker import CodeChunker
from .graph_extractors import (
    Extraction,
    GraphExtractor,
    GraphIndexer,
    IndexStats,
    StructureExtractor,
    Wikilink,
    WikilinkExtractor,
    parse_wikilink,
    wikilinks,
)
from .hierarchical import ParentChildChunker, ParentExpander
from .ingester import DocumentIngester
from .metadata import MetadataEnricher, extract_chunk_metadata, generate_chunk_id, generate_doc_id
from .parsers import (
    CSVParser,
    DocxParser,
    HTMLParser,
    JSONParser,
    MarkdownParser,
    PDFParser,
    PlainTextParser,
)
from .table_chunker import TableAwareChunker

__all__ = [
    "CSVParser",
    "CodeChunker",
    "DocumentIngester",
    "DocxParser",
    "Extraction",
    "FixedSizeChunker",
    "GraphExtractor",
    "GraphIndexer",
    "HTMLParser",
    "IndexStats",
    "JSONParser",
    "MarkdownHeaderChunker",
    "MarkdownParser",
    "MetadataEnricher",
    "PDFParser",
    "ParentChildChunker",
    "ParentExpander",
    "PlainTextParser",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "SentenceChunker",
    "StructureExtractor",
    "TableAwareChunker",
    "Wikilink",
    "WikilinkExtractor",
    "extract_chunk_metadata",
    "generate_chunk_id",
    "generate_doc_id",
    "parse_wikilink",
    "wikilinks",
]
