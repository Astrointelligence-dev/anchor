# Retrieval API Reference

API reference for all retrieval classes, protocols, and utilities.
For guides, see [Retrieval](../guides/retrieval.md) and
[Advanced Retrieval](../guides/advanced-retrieval.md).

---

## Protocols

```python
class Retriever(Protocol):
    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...

class AsyncRetriever(Protocol):
    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...

class Reranker(Protocol):
    def rerank(self, query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]: ...

class AsyncReranker(Protocol):
    async def arerank(self, query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]: ...

class TokenLevelEncoder(Protocol):
    def encode_tokens(self, text: str) -> list[list[float]]: ...
```

---

## Core Retrievers

### DenseRetriever

Embedding-based retrieval via a `VectorStore` backend.

```python
DenseRetriever(
    vector_store: VectorStore,
    context_store: ContextStore,
    embed_fn: Callable[[str], list[float]] | None = None,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vector_store` | `VectorStore` | -- | Embedding storage and search backend. |
| `context_store` | `ContextStore` | -- | Resolves item IDs to `ContextItem`. |
| `embed_fn` | `Callable[[str], list[float]] \| None` | `None` | Text embedding function. Required for `index()`. |
| `tokenizer` | `Tokenizer \| None` | `None` | Token counter. Defaults to built-in. |

| Method | Signature | Description |
|---|---|---|
| `index` | `(items: list[ContextItem]) -> int` | Index items. Returns count. Raises `RetrieverError` if `embed_fn` is `None`. |
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Retrieve by similarity. Uses `query.embedding` if set, else calls `embed_fn`. |

### SparseRetriever

BM25-based retrieval. Requires `rank-bm25`.

```python
SparseRetriever(
    tokenize_fn: Callable[[str], list[str]] | None = None,
    tokenizer: Tokenizer | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `index` | `(items: list[ContextItem]) -> int` | Build BM25 index. Raises `RetrieverError` if `rank-bm25` missing. |
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Retrieve by BM25 score (normalized to `[0, 1]`). |

### HybridRetriever

Combines multiple retrievers with Reciprocal Rank Fusion.

```python
HybridRetriever(
    retrievers: list[Retriever],
    rrf_k: int = 60,
    weights: list[float] | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Fuse results with RRF. Skips failed sub-retrievers. Raises `RetrieverError` if all fail. |

---

## Rerankers

### ScoreReranker

Simple reranker with `process()` interface (not the `Reranker` protocol).

```python
ScoreReranker(score_fn: Callable[[str, str], float], top_k: int | None = None)
```

| Method | Signature | Description |
|---|---|---|
| `process` | `(items: list[ContextItem], query: QueryBundle \| None = None) -> list[ContextItem]` | Rescore and sort. Returns input unchanged if `query` is `None`. |

### CrossEncoderReranker

Cross-encoder scoring. Implements `Reranker`.

```python
CrossEncoderReranker(score_fn: Callable[[str, str], float], top_k: int = 10)
```

| Method | Signature | Description |
|---|---|---|
| `rerank` | `(query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Score each item and return top-k. |

### CohereReranker

Batch reranking via API callback. Implements `Reranker`.

```python
CohereReranker(
    rerank_fn: Callable[[str, list[str], int], list[tuple[int, float]]],
    top_k: int = 10,
)
```

The `rerank_fn` takes `(query, documents, top_k)` and returns `(index, score)` tuples.

| Method | Signature | Description |
|---|---|---|
| `rerank` | `(query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Batch-rerank via callback. |

### FlashRankReranker

Local reranking via `flashrank`. Model is lazily loaded.

```python
FlashRankReranker(model_name: str = "ms-marco-MiniLM-L-12-v2", top_k: int = 10)
```

| Method | Signature | Description |
|---|---|---|
| `rerank` | `(query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Rerank using flashrank. Raises `RetrieverError` if not installed. |

### RoundRobinReranker

Round-robin merge and score-based re-sort.

```python
RoundRobinReranker(top_k: int = 10)
```

| Method | Signature | Description |
|---|---|---|
| `rerank` | `(query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Re-sort by existing score descending. |
| `rerank_multiple` | `(query: QueryBundle, result_sets: list[list[ContextItem]], top_k: int \| None = None) -> list[ContextItem]` | Merge result sets round-robin, deduplicating by ID. |

### RerankerPipeline

Chains multiple rerankers sequentially. Final `top_k` applied at the end.

```python
RerankerPipeline(rerankers: list[Reranker], top_k: int = 10)
```

| Method | Signature | Description |
|---|---|---|
| `rerank` | `(query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Pass items through each reranker in sequence. |

---

## Async Retrievers

### AsyncDenseRetriever

Async embedding-based retriever with cosine similarity.

```python
AsyncDenseRetriever(
    embed_fn: Callable[[str], Awaitable[list[float]]],
    similarity_fn: Callable[[list[float], list[float]], float] | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `index` | `(items: list[ContextItem]) -> None` | Store pre-embedded items (need `"embedding"` in metadata). |
| `aindex` | `async (items: list[ContextItem]) -> None` | Embed and store items via `embed_fn`. |
| `aretrieve` | `async (query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Retrieve by cosine similarity. |

### AsyncHybridRetriever

Concurrent fan-out with RRF fusion via `asyncio.gather`.

```python
AsyncHybridRetriever(
    retrievers: list[AsyncDenseRetriever],
    weights: list[float] | None = None,
    k: int = 60,
)
```

| Method | Signature | Description |
|---|---|---|
| `aretrieve` | `async (query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Fan out to all retrievers and fuse with RRF. |

### AsyncCrossEncoderReranker

Async cross-encoder scoring. Scores all items concurrently.

```python
AsyncCrossEncoderReranker(score_fn: Callable[[str, str], Awaitable[float]])
```

| Method | Signature | Description |
|---|---|---|
| `arerank` | `async (query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Score concurrently and return top-k. |

### AsyncCohereReranker

Async batch reranker via callback.

```python
AsyncCohereReranker(rerank_fn: Callable[[str, list[str], int], Awaitable[list[int]]])
```

| Method | Signature | Description |
|---|---|---|
| `arerank` | `async (query: QueryBundle, items: list[ContextItem], top_k: int = 10) -> list[ContextItem]` | Batch-rerank via async callback. |

---

## Query Routing

### KeywordRouter

Routes queries by keyword matching. First match wins.

```python
KeywordRouter(routes: dict[str, list[str]], default: str, *, case_sensitive: bool = False)
```

| Method | Signature | Description |
|---|---|---|
| `route` | `(query: QueryBundle) -> str` | Return route name for first keyword match, or `default`. |

### CallbackRouter

Routes using a user-provided callback.

```python
CallbackRouter(callback: Callable[[QueryBundle], str | None], default: str = "default")
```

| Method | Signature | Description |
|---|---|---|
| `route` | `(query: QueryBundle) -> str` | Call callback. Returns `default` if callback returns `None`. |

### MetadataRouter

Routes based on a metadata field in the query.

```python
MetadataRouter(metadata_key: str = "route", default: str = "default")
```

| Method | Signature | Description |
|---|---|---|
| `route` | `(query: QueryBundle) -> str` | Read `query.metadata[metadata_key]`. Returns `default` if missing. |

### RoutedRetriever

Delegates to named retrievers based on routing.

```python
RoutedRetriever(
    router: KeywordRouter | CallbackRouter | MetadataRouter,
    retrievers: dict[str, Retriever],
    default_retriever: str | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Route and delegate. Raises `RetrieverError` if route unknown and no default. |

---

## Cross-Modal

### CrossModalEncoder

Encodes content from multiple modalities into a shared vector space.

```python
CrossModalEncoder(encoders: dict[str, Callable[[Any], list[float]]])
```

| Member | Signature | Description |
|---|---|---|
| `encode` | `(content: Any, modality: str) -> list[float]` | Encode using named modality. Raises `ValueError` if unknown. |
| `modalities` | `@property -> list[str]` | Sorted list of registered modality names. |

### SharedSpaceRetriever

Cross-modal retriever in a shared embedding space.

```python
SharedSpaceRetriever(
    encoder: CrossModalEncoder,
    query_modality: str = "text",
    similarity_fn: Callable[[list[float], list[float]], float] | None = None,
)
```

| Method | Signature | Description |
|---|---|---|
| `index` | `(items: list[ContextItem], modality: str \| None = None) -> None` | Embed and store items. Uses `metadata["modality"]` if `modality` is `None`. |
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Retrieve by similarity in shared space. |

---

## Late Interaction

### MaxSimScorer

ColBERT-style MaxSim scoring over per-token embeddings.

```python
MaxSimScorer()
```

| Method | Signature | Description |
|---|---|---|
| `score` | `(query_tokens: list[list[float]], doc_tokens: list[list[float]]) -> float` | Sum of per-query-token maximum cosine similarities. |

### ScoredMemoryRetriever

Multi-signal retriever combining recency, relevance, and importance.

```python
ScoredMemoryRetriever(
    store: MemoryEntryStore,
    embed_fn: Callable[[str], list[float]] | None = None,
    vector_store: VectorStore | None = None,
    decay: MemoryDecay | None = None,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.2,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `store` | `MemoryEntryStore` | -- | Backing store for entries. |
| `embed_fn` | `Callable \| None` | `None` | Embedding function for relevance. |
| `vector_store` | `VectorStore \| None` | `None` | Vector index for relevance scoring. |
| `decay` | `MemoryDecay \| None` | `None` | Custom decay. Defaults to 7-day half-life. |
| `alpha` / `beta` / `gamma` | `float` | `0.3` / `0.5` / `0.2` | Recency / relevance / importance weights. |

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `(query: str, top_k: int = 5, *, user_id: str \| None, memory_type: str \| None) -> list[MemoryEntry]` | Retrieve by composite score with optional filters. |
| `add_entry` | `(entry: MemoryEntry) -> None` | Add entry and optionally index embedding. |
| `as_retriever` | `() -> MemoryRetrieverAdapter` | Return `Retriever`-protocol adapter for pipeline use. |

### MemoryRetrieverAdapter

Bridges `ScoredMemoryRetriever` to the `Retriever` protocol.

```python
MemoryRetrieverAdapter(retriever: ScoredMemoryRetriever)
```

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `(query: QueryBundle, top_k: int = 10) -> list[ContextItem]` | Convert `MemoryEntry` to `ContextItem` with `source=MEMORY`, `priority=7`. |

---

## Utility Functions

### rrf_fuse

Standalone Reciprocal Rank Fusion for combining ranked lists.

```python
def rrf_fuse(
    ranked_lists: list[list[ContextItem]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int | None = None,
) -> list[ContextItem]: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ranked_lists` | `list[list[ContextItem]]` | -- | Ranked item lists. |
| `weights` | `list[float] \| None` | `None` | Per-list weights. Defaults to `1.0`. |
| `k` | `int` | `60` | RRF smoothing constant. |
| `top_k` | `int \| None` | `None` | Max items to return. `None` returns all. |

Returns fused items with normalized scores and `retrieval_method: "rrf"` in metadata.
