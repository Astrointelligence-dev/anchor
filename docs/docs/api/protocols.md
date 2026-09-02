# Protocols API Reference

anchor uses PEP 544 structural protocols throughout its architecture.
Any class with matching method signatures satisfies a protocol -- no
inheritance required. All protocols are `@runtime_checkable`.

All protocols are importable from `anchor`:

```python
from anchor import Retriever, AsyncRetriever, PostProcessor, Reranker
# ... and all others listed below
```

---

## Retrieval

### Retriever

Synchronous retrieval of context items.

```python
@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: QueryBundle, top_k: int = 10,
        *, scope: RetrievalScope | None = None,
    ) -> list[ContextItem]: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `QueryBundle` | required | Query text and metadata |
| `top_k` | `int` | `10` | Maximum items to return |
| `scope` | `RetrievalScope \| None` | `None` | Namespace scope; a retriever that cannot honor it must raise, never ignore it |

**Returns:** List of `ContextItem` ranked by relevance (most relevant first).

### AsyncRetriever

Asynchronous retrieval for non-blocking I/O.

```python
@runtime_checkable
class AsyncRetriever(Protocol):
    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...
```

Same parameters as `Retriever`. Used with `ContextPipeline.abuild()`.

---

## Post-Processing

### PostProcessor

Synchronous transformation of retrieved context items (reranking, filtering,
deduplication, PII removal, etc.).

```python
@runtime_checkable
class PostProcessor(Protocol):
    def process(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `items` | `list[ContextItem]` | required | Items to post-process |
| `query` | `QueryBundle \| None` | `None` | Original query for query-aware transforms |

**Returns:** A new or modified list of `ContextItem` objects.

### AsyncPostProcessor

Asynchronous post-processing (e.g., LLM-based reranking).

```python
@runtime_checkable
class AsyncPostProcessor(Protocol):
    async def aprocess(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]: ...
```

---

## Query Transformation

### QueryTransformer

Synchronous query transformation. Takes a single query and produces one or
more derived queries for improved retrieval (expansion, decomposition, HyDE).

```python
@runtime_checkable
class QueryTransformer(Protocol):
    def transform(self, query: QueryBundle) -> list[QueryBundle]: ...
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `QueryBundle` | The original query to transform |

**Returns:** A list of derived `QueryBundle` objects. Always at least one.

### AsyncQueryTransformer

Asynchronous query transformation (e.g., LLM-based HyDE generation).

```python
@runtime_checkable
class AsyncQueryTransformer(Protocol):
    async def atransform(self, query: QueryBundle) -> list[QueryBundle]: ...
```

---

## Query Classification

### QueryClassifier

Classifies a query and returns a string label for downstream routing.

```python
@runtime_checkable
class QueryClassifier(Protocol):
    def classify(self, query: QueryBundle) -> str: ...
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `QueryBundle` | The query to classify |

**Returns:** A string label representing the query category.

### QueryRouter

Routes queries to named retriever backends.

```python
@runtime_checkable
class QueryRouter(Protocol):
    def route(self, query: QueryBundle) -> str: ...
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `QueryBundle` | The query to route |

**Returns:** The name/key of the target retriever.

---

## Reranking

### Reranker

Synchronous reranking of retrieved results.

```python
@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: QueryBundle, items: list[ContextItem], top_k: int = 10
    ) -> list[ContextItem]: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `QueryBundle` | required | The user query |
| `items` | `list[ContextItem]` | required | Candidate items to rerank |
| `top_k` | `int` | `10` | Maximum items to return |

**Returns:** List of `ContextItem` ranked by relevance (most relevant first).

### AsyncReranker

Asynchronous reranking for cross-encoder inference or API calls.

```python
@runtime_checkable
class AsyncReranker(Protocol):
    async def arerank(
        self, query: QueryBundle, items: list[ContextItem], top_k: int = 10
    ) -> list[ContextItem]: ...
```

---


### ## Memory

### ConversationMemory

Read-side access to conversation memory. Both `SlidingWindowMemory` and
`SummaryBufferMemory` satisfy this protocol.

```python
@runtime_checkable
class ConversationMemory(Protocol):
    @property
    def turns(self) -> list[ConversationTurn]: ...

    @property
    def total_tokens(self) -> int: ...

    def to_context_items(self, priority: int = 7) -> list[ContextItem]: ...

    def clear(self) -> None: ...
```

### MemoryProvider

Provides context items from memory. Used by `ContextPipeline.with_memory()`.
`MemoryManager` is the canonical implementation.

```python
@runtime_checkable
class MemoryProvider(Protocol):
    def get_context_items(self, priority: int = 7) -> list[ContextItem]: ...
```

### CompactionStrategy

Compacts evicted conversation turns into a summary string.

```python
@runtime_checkable
class CompactionStrategy(Protocol):
    def compact(self, turns: list[ConversationTurn]) -> str: ...
```

| Parameter | Type | Description |
|---|---|---|
| `turns` | `list[ConversationTurn]` | Turns to compact (chronological, oldest first) |

**Returns:** A concise summary string.

### AsyncCompactionStrategy

Async variant of `CompactionStrategy`.

```python
@runtime_checkable
class AsyncCompactionStrategy(Protocol):
    async def compact(self, turns: list[ConversationTurn]) -> str: ...
```

### MemoryExtractor

Extracts structured memories from conversation turns.

```python
@runtime_checkable
class MemoryExtractor(Protocol):
    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...
```

**Returns:** Zero or more `MemoryEntry` objects representing facts, preferences,
or other persistent knowledge.

### AsyncMemoryExtractor

Async variant of `MemoryExtractor`.

```python
@runtime_checkable
class AsyncMemoryExtractor(Protocol):
    async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...
```

### MemoryConsolidator

Decides how to merge new memories with existing ones.

```python
@runtime_checkable
class MemoryConsolidator(Protocol):
    def consolidate(
        self, new_entries: list[MemoryEntry], existing: list[MemoryEntry]
    ) -> list[tuple[MemoryOperation, MemoryEntry | None]]: ...
```

**Returns:** A list of `(MemoryOperation, entry | None)` tuples. Operations
are `ADD`, `UPDATE`, `DELETE`, or `NONE`.

### EvictionPolicy

Decides which conversation turns to evict when memory is full.

```python
@runtime_checkable
class EvictionPolicy(Protocol):
    def select_for_eviction(
        self, turns: list[ConversationTurn], tokens_to_free: int
    ) -> list[int]: ...
```

| Parameter | Type | Description |
|---|---|---|
| `turns` | `list[ConversationTurn]` | All turns currently held in memory |
| `tokens_to_free` | `int` | Minimum tokens that must be freed |

**Returns:** List of zero-based indices identifying turns to evict.

### MemoryDecay

Computes a retention score for a memory entry (0.0 = forget, 1.0 = retain).

```python
@runtime_checkable
class MemoryDecay(Protocol):
    def compute_retention(self, entry: MemoryEntry) -> float: ...
```

**Returns:** Float from 0.0 (completely forgotten) to 1.0 (perfectly retained).

### MemoryQueryEnricher

Enriches a query with memory context before retrieval.

```python
@runtime_checkable
class MemoryQueryEnricher(Protocol):
    def enrich(self, query: str, memory_items: list[MemoryEntry]) -> str: ...
```

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | The original user query |
| `memory_items` | `list[MemoryEntry]` | Relevant memory entries |

**Returns:** An enriched query string.

!!! note
    `QueryEnricher` is a deprecated alias for `MemoryQueryEnricher`.

### RecencyScorer

Computes recency scores for memory items.

```python
@runtime_checkable
class RecencyScorer(Protocol):
    def score(self, index: int, total: int) -> float: ...
```

| Parameter | Type | Description |
|---|---|---|
| `index` | `int` | Zero-based position (0 = oldest) |
| `total` | `int` | Total number of items |

**Returns:** Float from 0.0 (oldest) to 1.0 (newest).

### MemoryOperation

Enum of actions a `MemoryConsolidator` can prescribe.

```python
class MemoryOperation(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"
```

---

## Storage

### ContextStore

CRUD operations for `ContextItem` objects.

```python
@runtime_checkable
class ContextStore(Protocol):
    def add(self, item: ContextItem) -> None: ...
    def get(self, item_id: str) -> ContextItem | None: ...
    def get_all(self) -> list[ContextItem]: ...
    def delete(self, item_id: str) -> bool: ...
    def clear(self) -> None: ...
```

### VectorStore

Vector similarity search backend (wraps FAISS, Chroma, Qdrant, etc.).

```python
@runtime_checkable
class VectorStore(Protocol):
    def add_embedding(
        self, item_id: str, embedding: list[float],
        metadata: dict[str, Any] | None = None,
        *, namespace: str = "/",
    ) -> None: ...

    def search(
        self, query_embedding: list[float], top_k: int = 10,
        where: dict[str, Any] | None = None,
        *, scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]: ...

    def delete(self, item_id: str) -> bool: ...
```

**`search` returns:** List of `(item_id, score)` tuples by descending similarity.

### DocumentStore

Storage for raw documents (pre-chunking / pre-indexing).

```python
@runtime_checkable
class DocumentStore(Protocol):
    def add_document(
        self, doc_id: str, content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def get_document(self, doc_id: str) -> str | None: ...
    def list_documents(self) -> list[str]: ...
    def delete_document(self, doc_id: str) -> bool: ...
```

### MemoryEntryStore

Persistent storage for `MemoryEntry` objects with CRUD and search.

```python
@runtime_checkable
class MemoryEntryStore(Protocol):
    def add(self, entry: MemoryEntry) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...
    def list_all(self) -> list[MemoryEntry]: ...
    def delete(self, entry_id: str) -> bool: ...
    def clear(self) -> None: ...
```

### GarbageCollectableStore

Extends `MemoryEntryStore` with access to expired entries for garbage
collection.

```python
@runtime_checkable
class GarbageCollectableStore(Protocol):
    def list_all_unfiltered(self) -> list[MemoryEntry]: ...
    def delete(self, entry_id: str) -> bool: ...
```

Unlike `MemoryEntryStore.list_all` which may filter out expired entries,
`list_all_unfiltered` returns every entry so the garbage collector can
identify and prune them.

---

## Evaluation

### RetrievalEvaluator

Evaluates retrieval quality (precision, recall, MRR, NDCG).

```python
@runtime_checkable
class RetrievalEvaluator(Protocol):
    def evaluate(
        self, retrieved: list[ContextItem], relevant: list[str], k: int = 10,
    ) -> RetrievalMetrics: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `retrieved` | `list[ContextItem]` | required | Items returned by retriever, ranked |
| `relevant` | `list[str]` | required | IDs of truly relevant documents |
| `k` | `int` | `10` | Top results to consider |

### RAGEvaluator

Evaluates RAG output quality (faithfulness, relevancy, precision, recall).

```python
@runtime_checkable
class RAGEvaluator(Protocol):
    def evaluate(
        self, query: str, answer: str, contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGMetrics: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Original user query |
| `answer` | `str` | required | Generated answer |
| `contexts` | `list[str]` | required | Context strings fed to generator |
| `ground_truth` | `str \| None` | `None` | Reference answer for recall |

### HumanEvaluator

Human-in-the-loop evaluation with inter-annotator agreement.

```python
@runtime_checkable
class HumanEvaluator(Protocol):
    def add_judgment(self, judgment: Any) -> None: ...
    def compute_agreement(self) -> float: ...
```

---

## Observability

### SpanExporter

Exports spans to external systems (stdout, file, OTLP, etc.).

```python
@runtime_checkable
class SpanExporter(Protocol):
    def export(self, spans: list[Span]) -> None: ...
```

### MetricsCollector

Collects and exports metrics.

```python
@runtime_checkable
class MetricsCollector(Protocol):
    def record(self, metric: MetricPoint) -> None: ...
    def flush(self) -> None: ...
```

---

## Ingestion

### Chunker

Splits text into smaller pieces for embedding and retrieval.

```python
@runtime_checkable
class Chunker(Protocol):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Full document text |
| `metadata` | `dict[str, Any] \| None` | `None` | Document-level metadata |

**Returns:** List of text chunks.

### DocumentParser

Converts a file into plain text plus metadata.

```python
@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]: ...

    @property
    def supported_extensions(self) -> list[str]: ...
```

**`parse` returns:** `(text, metadata)` tuple.

---

## Multimodal

### ModalityEncoder

Encodes multi-modal content into text for embedding and retrieval.

```python
@runtime_checkable
class ModalityEncoder(Protocol):
    def encode(self, content: MultiModalContent) -> str: ...

    @property
    def supported_modalities(self) -> list[ModalityType]: ...
```

### TableExtractor

Extracts structured tables from documents.

```python
@runtime_checkable
class TableExtractor(Protocol):
    def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]: ...
```

---

## Tokenization

### Tokenizer

Token counting and truncation abstraction.

```python
@runtime_checkable
class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...
```

| Method | Description |
|---|---|
| `count_tokens(text)` | Count tokens in a text string |
| `truncate_to_tokens(text, max_tokens)` | Truncate text to at most `max_tokens` tokens |

---

## Caching

### CacheBackend

Backend for caching pipeline step results.

```python
@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def invalidate(self, key: str) -> None: ...
    def clear(self) -> None: ...
```

See [Cache API Reference](../api/cache.md) for the built-in implementation.

---

## Implementing a Protocol

To implement any protocol, create a class with matching method signatures:

```python
from anchor import Retriever, ContextItem, QueryBundle

class MyRetriever:
    """Custom retriever -- no inheritance needed."""

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        # Your retrieval logic here
        return []

# Verify at runtime
assert isinstance(MyRetriever(), Retriever)
```

!!! tip
    All protocols are `@runtime_checkable`, so you can use `isinstance()`
    checks at runtime for validation and debugging.

---

## See Also

- [Retrieval Guide](../guides/retrieval.md) -- using retrievers and rerankers
- [Memory Guide](../guides/memory.md) -- memory extension points
- [Ingestion Guide](../guides/ingestion.md) -- chunkers and parsers
- [Cache Guide](../guides/cache.md) -- cache backends
