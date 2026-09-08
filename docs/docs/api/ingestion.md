# Ingestion API Reference

API reference for the `anchor.ingestion` module. For usage patterns and
examples, see the [Ingestion Guide](../guides/ingestion.md).

---

## DocumentIngester

Orchestrates document parsing, chunking, and metadata extraction. Converts raw
files or text into `ContextItem` objects suitable for `retriever.index(items)`.

```python
class DocumentIngester(
    chunker: Chunker | None = None,
    tokenizer: Tokenizer | None = None,
    parsers: dict[str, DocumentParser] | None = None,
    enricher: MetadataEnricher | None = None,
    source_type: SourceType = SourceType.RETRIEVAL,
    priority: int = 5,
)
```

| Parameter     | Type                                 | Default                       | Description                              |
|---------------|--------------------------------------|-------------------------------|------------------------------------------|
| `chunker`     | `Chunker \| None`                    | `RecursiveCharacterChunker()` | Chunking strategy                        |
| `tokenizer`   | `Tokenizer \| None`                  | default counter               | Token counter for size-aware splitting   |
| `parsers`     | `dict[str, DocumentParser] \| None`  | built-in parser map           | Extension-to-parser overrides            |
| `enricher`    | `MetadataEnricher \| None`           | `None`                        | Chain of metadata enrichment functions   |
| `source_type` | `SourceType`                         | `SourceType.RETRIEVAL`        | Source type tag for produced items       |
| `priority`    | `int`                                | `5`                           | Priority value for produced items        |

### Methods

#### `ingest_text(text, doc_id=None, doc_metadata=None)`

Ingest raw text into context items.

| Parameter      | Type                       | Default | Description                              |
|----------------|----------------------------|---------|------------------------------------------|
| `text`         | `str`                      | required | Document text to ingest                 |
| `doc_id`       | `str \| None`              | `None`  | Document ID; generated if not provided   |
| `doc_metadata` | `dict[str, Any] \| None`   | `None`  | Document-level metadata                  |

**Returns:** `list[ContextItem]`

#### `ingest_file(path, doc_id=None)`

Parse and chunk a single file. The parser is auto-detected from file extension.

| Parameter | Type             | Default | Description                              |
|-----------|------------------|---------|------------------------------------------|
| `path`    | `Path \| str`    | required | Path to the file                        |
| `doc_id`  | `str \| None`    | `None`  | Document ID; generated if not provided   |

**Returns:** `list[ContextItem]`
**Raises:** `IngestionError` if no parser found; `FileNotFoundError` if file missing.

#### `ingest_directory(directory, glob_pattern="**/*", extensions=None)`

Recursively ingest all matching files in a directory.

| Parameter      | Type               | Default   | Description                                     |
|----------------|--------------------|-----------|-------------------------------------------------|
| `directory`    | `Path \| str`      | required  | Root directory to scan                           |
| `glob_pattern` | `str`              | `"**/*"`  | Glob pattern for file discovery                  |
| `extensions`   | `list[str] \| None`| `None`    | Filter by extensions; `None` = all registered    |

**Returns:** `list[ContextItem]`
**Raises:** `IngestionError` if directory does not exist.

---

## Chunkers

All chunkers implement the `Chunker` protocol:

```python
def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]
```

### FixedSizeChunker

Split text into fixed-size chunks by token count with overlap.

```python
class FixedSizeChunker(
    chunk_size: int = 512,
    overlap: int = 50,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter    | Type               | Default         | Description                    |
|--------------|--------------------|-----------------|--------------------------------|
| `chunk_size` | `int`              | `512`           | Maximum tokens per chunk       |
| `overlap`    | `int`              | `50`            | Overlapping tokens at boundary |
| `tokenizer`  | `Tokenizer \| None`| default counter | Token counter                  |

### RecursiveCharacterChunker

Split text using a hierarchy of separators, falling back to finer splits.

```python
class RecursiveCharacterChunker(
    chunk_size: int = 512,
    overlap: int = 50,
    separators: tuple[str, ...] | None = None,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter    | Type                       | Default                        | Description                   |
|--------------|----------------------------|--------------------------------|-------------------------------|
| `chunk_size` | `int`                      | `512`                          | Maximum tokens per chunk      |
| `overlap`    | `int`                      | `50`                           | Overlapping tokens            |
| `separators` | `tuple[str, ...] \| None`  | `("\n\n", "\n", ". ", " ")`    | Separator hierarchy           |
| `tokenizer`  | `Tokenizer \| None`        | default counter                | Token counter                 |

### SentenceChunker

Split text at sentence boundaries, grouping sentences to fill chunks. Overlap is
measured in sentences.

```python
class SentenceChunker(
    chunk_size: int = 512,
    overlap: int = 1,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter    | Type               | Default         | Description                          |
|--------------|--------------------|-----------------|--------------------------------------|
| `chunk_size` | `int`              | `512`           | Maximum tokens per chunk             |
| `overlap`    | `int`              | `1`             | Overlapping **sentences**            |
| `tokenizer`  | `Tokenizer \| None`| default counter | Token counter                        |

### SemanticChunker

Split text at semantic boundaries using embedding similarity.

```python
class SemanticChunker(
    embed_fn: Callable[[list[str]], list[list[float]]],
    tokenizer: Tokenizer | None = None,
    threshold: float = 0.5,
    chunk_size: int = 512,
    min_chunk_size: int = 50,
)
```

| Parameter        | Type                                       | Default         | Description                          |
|------------------|--------------------------------------------|-----------------|--------------------------------------|
| `embed_fn`       | `Callable[[list[str]], list[list[float]]]` | **required**    | Batch embedding function             |
| `tokenizer`      | `Tokenizer \| None`                        | default counter | Token counter                        |
| `threshold`      | `float`                                    | `0.5`           | Cosine similarity split threshold    |
| `chunk_size`     | `int`                                      | `512`           | Maximum tokens per chunk             |
| `min_chunk_size` | `int`                                      | `50`            | Minimum tokens; smaller merge        |

### CodeChunker

Split source code at function, class, and definition boundaries.

```python
class CodeChunker(
    language: str | None = None,
    chunk_size: int = 512,
    overlap: int = 50,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter    | Type               | Default         | Description                                 |
|--------------|--------------------|-----------------|---------------------------------------------|
| `language`   | `str \| None`      | `None`          | Language name; auto-detected from metadata  |
| `chunk_size` | `int`              | `512`           | Maximum tokens per chunk                    |
| `overlap`    | `int`              | `50`            | Overlap tokens for fallback chunker         |
| `tokenizer`  | `Tokenizer \| None`| default counter | Token counter                               |

Supported languages: `python`, `javascript`, `typescript`, `go`, `rust`.

### TableAwareChunker

Preserve tables as atomic chunks while delegating prose to an inner chunker.

```python
class TableAwareChunker(
    inner_chunker: Any | None = None,
    chunk_size: int = 512,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter       | Type               | Default                        | Description                    |
|-----------------|--------------------|--------------------------------|--------------------------------|
| `inner_chunker` | `Any \| None`      | `RecursiveCharacterChunker()`  | Chunker for non-table text     |
| `chunk_size`    | `int`              | `512`                          | Maximum tokens per chunk       |
| `tokenizer`     | `Tokenizer \| None`| default counter                | Token counter                  |

### ParentChildChunker

Two-level hierarchical chunker producing large parent and small child chunks.

```python
class ParentChildChunker(
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    parent_overlap: int = 100,
    child_overlap: int = 25,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter           | Type               | Default         | Description                         |
|---------------------|--------------------|-----------------|-------------------------------------|
| `parent_chunk_size` | `int`              | `1024`          | Token size for parent chunks        |
| `child_chunk_size`  | `int`              | `256`           | Token size for child chunks         |
| `parent_overlap`    | `int`              | `100`           | Token overlap between parents       |
| `child_overlap`     | `int`              | `25`            | Token overlap between children      |
| `tokenizer`         | `Tokenizer \| None`| default counter | Token counter                       |

#### Methods

- **`chunk(text, metadata=None)`** -- Returns child chunk texts only (`list[str]`).
- **`chunk_with_metadata(text, metadata=None)`** -- Returns `list[tuple[str, dict[str, Any]]]` with `parent_id`, `parent_text`, `parent_index`, `child_index`, and `is_child_chunk` in each metadata dict.

---

## Parsers

All parsers implement the `DocumentParser` protocol:

```python
def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]
```

### PlainTextParser

Parse plain text files. Supported extensions: `.txt`.

Metadata produced: `filename`, `extension`, `line_count`.

### MarkdownParser

Parse Markdown files, extracting headings and detecting frontmatter.
Supported extensions: `.md`, `.markdown`.

Metadata produced: `filename`, `extension`, `title` (first H1), `headings` list,
`has_frontmatter`.

### HTMLParser

Parse HTML files, stripping tags and extracting text. Uses Python's stdlib
`html.parser` with zero external dependencies. Supported extensions: `.html`, `.htm`.

Metadata produced: `filename`, `extension`, `title`.

### PDFParser

Parse PDF files using `pypdf`. Supported extensions: `.pdf`.

!!! note
    Requires the `pdf` optional extra: `pip install astro-anchor[pdf]`.
    Raises `IngestionError` if `pypdf` is not installed.

Metadata produced: `filename`, `extension`, `page_count`, `title`, `author`.

---

## Metadata Functions

### `generate_doc_id(content, source_path=None)`

Generate a deterministic 16-character hex document ID from SHA-256.

| Parameter     | Type           | Default | Description                         |
|---------------|----------------|---------|-------------------------------------|
| `content`     | `str`          | required | Full document text                 |
| `source_path` | `str \| None`  | `None`  | File path used as uniqueness salt   |

**Returns:** `str` -- 16-character hex string.

### `generate_chunk_id(doc_id, chunk_index)`

Generate a chunk ID in the format `"{doc_id}-chunk-{chunk_index}"`.

| Parameter     | Type  | Default  | Description                     |
|---------------|-------|----------|---------------------------------|
| `doc_id`      | `str` | required | Parent document ID              |
| `chunk_index` | `int` | required | Zero-based chunk index          |

**Returns:** `str`

### `extract_chunk_metadata(chunk_text, chunk_index, total_chunks, doc_id, doc_metadata=None)`

Build standard metadata for a single chunk.

| Parameter      | Type                     | Default | Description                       |
|----------------|--------------------------|---------|-----------------------------------|
| `chunk_text`   | `str`                    | required | Chunk text content               |
| `chunk_index`  | `int`                    | required | Zero-based chunk position        |
| `total_chunks` | `int`                    | required | Total chunks in document         |
| `doc_id`       | `str`                    | required | Parent document ID               |
| `doc_metadata` | `dict[str, Any] \| None` | `None`  | Document-level metadata          |

**Returns:** `dict[str, Any]` with keys: `parent_doc_id`, `chunk_index`,
`total_chunks`, `word_count`, `char_count`, plus propagated doc metadata
(prefixed with `doc_`).

---

## MetadataEnricher

Chain of user-provided metadata enrichment functions.

```python
class MetadataEnricher(
    enrichers: list[Callable[[str, int, int, dict[str, Any]], dict[str, Any]]] | None = None,
)
```

| Parameter   | Type                                    | Default | Description                          |
|-------------|-----------------------------------------|---------|--------------------------------------|
| `enrichers` | `list[Callable] \| None`               | `None`  | Initial list of enricher functions   |

Each enricher callable has signature:
`(text: str, chunk_index: int, total_chunks: int, metadata: dict) -> dict`

### Methods

#### `add(fn)`

Register an additional enricher function.

#### `enrich(text, chunk_index, total_chunks, metadata)`

Run all enrichers in order, threading metadata through. Returns the enriched
metadata dict.

---

## ParentExpander

Post-processor that expands child chunks back to parent text, deduplicating by
`parent_id`.

```python
class ParentExpander(
    keep_child: bool = False,
)
```

| Parameter    | Type   | Default | Description                                       |
|--------------|--------|---------|---------------------------------------------------|
| `keep_child` | `bool` | `False` | Keep original child content in `original_child_content` metadata |

### Methods

#### `process(items, query=None)`

Expand child chunks to parent text. Items with `is_child_chunk` in metadata
have their content replaced with `parent_text`. Multiple children from the
same parent are deduplicated (first occurrence wins).

| Parameter | Type                    | Default | Description                |
|-----------|-------------------------|---------|----------------------------|
| `items`   | `list[ContextItem]`     | required | Items to post-process     |
| `query`   | `QueryBundle \| None`   | `None`  | Original query (unused)    |

**Returns:** `list[ContextItem]`

---

## Graph extraction

`GraphExtractor`, `Extraction`, `StructureExtractor`, `WikilinkExtractor`,
`LLMGraphExtractor` and `GraphIndexer` turn `ContextItem`s into knowledge-graph
nodes and edges. See [Knowledge Graph](graph.md).
