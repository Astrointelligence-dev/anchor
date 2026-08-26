# Advanced Retrieval

This guide covers advanced retrieval features: reranker pipelines, async
retrievers, query routing, cross-modal search, and
memory-aware retrieval. For core retrieval, see [Retrieval Guide](retrieval.md).

---

## Reranker Pipeline

All rerankers conform to the `Reranker` protocol with `rerank(query, items, top_k)`.

### CrossEncoderReranker

Scores each item with a user-provided `(query_str, doc_content) -> float` function.

```python
from anchor.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(score_fn=my_scorer, top_k=10)
results = reranker.rerank(query, items)
```

### CohereReranker

Batch reranking via a callback that takes `(query_str, documents, top_k)` and
returns `(original_index, score)` tuples.

```python
from anchor.retrieval import CohereReranker

def cohere_rerank(query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
    response = co.rerank(query=query, documents=docs, top_n=top_k)
    return [(r.index, r.relevance_score) for r in response.results]

reranker = CohereReranker(rerank_fn=cohere_rerank, top_k=10)
```

### FlashRankReranker

Local cross-encoder reranking using `flashrank`. The model is lazily loaded.

!!! warning
    Requires: `pip install astro-anchor[flashrank]`

```python
from anchor.retrieval import FlashRankReranker

reranker = FlashRankReranker(model_name="ms-marco-MiniLM-L-12-v2", top_k=10)
results = reranker.rerank(query, items)
```

### RoundRobinReranker

Merges multiple result sets round-robin (deduplicating by ID), or re-sorts by
existing score via `rerank()`.

```python
from anchor.retrieval import RoundRobinReranker

rr = RoundRobinReranker(top_k=10)
merged = rr.rerank_multiple(query, [dense_results, sparse_results], top_k=10)
```

### RerankerPipeline

Chains multiple rerankers sequentially. Intermediate stages pass all items
through; `top_k` is applied only at the end.

```python
from anchor.retrieval import RerankerPipeline

pipeline = RerankerPipeline(rerankers=[cross_encoder, cohere_reranker], top_k=5)
results = pipeline.rerank(query, items)
```

!!! tip
    Place cheaper rerankers first (e.g., `FlashRankReranker`) to reduce the
    candidate set before expensive API-based rerankers (`CohereReranker`).

---

## Async Retrievers

Async variants implement the `AsyncRetriever` protocol (`aretrieve()` method)
for non-blocking I/O during embedding lookups and API calls.

### AsyncDenseRetriever

Async embedding-based retriever. Use `aindex()` to embed items, or `index()`
for pre-embedded items (those with `"embedding"` in metadata).

```python
from anchor.retrieval import AsyncDenseRetriever

async def async_embed(text: str) -> list[float]:
    return await embedding_service.embed(text)

retriever = AsyncDenseRetriever(embed_fn=async_embed)
await retriever.aindex(items)
results = await retriever.aretrieve(query, top_k=5)
```

### AsyncHybridRetriever

Fans out to multiple async retrievers concurrently via `asyncio.gather` and
fuses results with RRF.

```python
from anchor.retrieval import AsyncHybridRetriever

hybrid = AsyncHybridRetriever(retrievers=[ret_a, ret_b], weights=[0.7, 0.3], k=60)
results = await hybrid.aretrieve(query, top_k=10)
```

!!! note
    Failed sub-retrievers are logged and skipped. If all fail, an empty list
    is returned.

### Async Rerankers

- **`AsyncCrossEncoderReranker`** -- scores items concurrently via `asyncio.gather`.
- **`AsyncCohereReranker`** -- batch reranking via an async callback.

```python
from anchor.retrieval import AsyncCrossEncoderReranker

reranker = AsyncCrossEncoderReranker(score_fn=async_scorer)
results = await reranker.arerank(query, items, top_k=5)
```

---

## Query Routing

Query routers direct queries to different retriever backends based on the
query content. Use `RoutedRetriever` to wrap a router and a set of named
retrievers.

### KeywordRouter

Routes based on keyword matching. First keyword match wins; falls back to a
default route.

```python
from anchor.retrieval import KeywordRouter, RoutedRetriever

router = KeywordRouter(
    routes={
        "code": ["function", "class", "import", "def"],
        "docs": ["documentation", "guide", "tutorial"],
    },
    default="general",
    case_sensitive=False,
)

routed = RoutedRetriever(
    router=router,
    retrievers={"code": code_retriever, "docs": docs_retriever, "general": general_retriever},
)
results = routed.retrieve(query, top_k=10)
```

### CallbackRouter

Routes using a user-provided function that takes a `QueryBundle` and returns
a route name (or `None` for the default).

```python
from anchor.retrieval import CallbackRouter, RoutedRetriever

def classify(q):
    if len(q.query_str) > 100:
        return "detailed"
    return None  # falls back to default

router = CallbackRouter(callback=classify, default="quick")
```

### MetadataRouter

Routes based on a metadata field in the `QueryBundle`. Inspects
`query.metadata[key]` and uses its value as the route name.

```python
from anchor.retrieval import MetadataRouter, RoutedRetriever
from anchor.models.query import QueryBundle

router = MetadataRouter(metadata_key="domain", default="general")

query = QueryBundle(query_str="explain auth", metadata={"domain": "security"})
# Routes to the "security" retriever
```

### RoutedRetriever

Wraps any router and a mapping of named retrievers.

| Parameter | Type | Description |
|---|---|---|
| `router` | `KeywordRouter \| CallbackRouter \| MetadataRouter` | The query router. |
| `retrievers` | `dict[str, Retriever]` | Mapping of route names to retrievers. |
| `default_retriever` | `str \| None` | Fallback retriever name if route not found in `retrievers`. |

!!! warning
    If the route maps to an unknown retriever and no `default_retriever` is
    configured, `RetrieverError` is raised.

---

## Cross-Modal Retrieval

Cross-modal retrieval searches across different content types (text, image,
audio) in a shared embedding space.

### CrossModalEncoder

Encodes content from multiple modalities using modality-specific callbacks.

```python
import math
from anchor.retrieval import CrossModalEncoder

def text_encoder(text: str) -> list[float]:
    vals = [ord(c) for c in text[:20]]
    raw = [math.sin(sum(vals)), math.cos(sum(vals))]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]

encoder = CrossModalEncoder(
    encoders={"text": text_encoder, "image": image_encoder}
)

embedding = encoder.encode("hello world", modality="text")
print(encoder.modalities)  # ["image", "text"]
```

### SharedSpaceRetriever

Retriever that searches across modalities in a shared embedding space.

```python
from anchor.retrieval import SharedSpaceRetriever

retriever = SharedSpaceRetriever(
    encoder=encoder,
    query_modality="text",
    similarity_fn=None,  # defaults to cosine similarity
)

retriever.index(items, modality="text")
results = retriever.retrieve(query, top_k=5)
```

Items can specify their modality via `metadata["modality"]`. When no modality
is provided to `index()`, it defaults to `"text"`.

---

## Memory Retrieval

`ScoredMemoryRetriever` combines recency, relevance, and importance signals
to retrieve from a `MemoryEntryStore`.

### ScoredMemoryRetriever

```python
from anchor.retrieval import ScoredMemoryRetriever

retriever = ScoredMemoryRetriever(
    store=memory_store,
    embed_fn=embed_fn,            # optional
    vector_store=vector_store,    # optional
    decay=None,                   # optional MemoryDecay
    alpha=0.3,                    # recency weight
    beta=0.5,                     # relevance weight
    gamma=0.2,                    # importance weight
)

# Add entries
retriever.add_entry(memory_entry)

# Retrieve with filters
top_memories = retriever.retrieve(
    "what did the user say about testing?",
    top_k=5,
    user_id="user-123",
    memory_type="preference",
)
```

The composite score formula:

```
score = alpha * recency + beta * relevance + gamma * importance
```

- **recency**: Exponential decay (7-day half-life) or custom `MemoryDecay`.
- **relevance**: Cosine similarity from `VectorStore` or keyword overlap.
- **importance**: The entry's `relevance_score` field.

### MemoryRetrieverAdapter

Bridges `ScoredMemoryRetriever` to the `Retriever` protocol for pipeline
integration. Converts `MemoryEntry` objects to `ContextItem` objects.

```python
adapter = retriever.as_retriever()
# Now usable in a ContextPipeline via retriever_step()
```

---

## What's Next

- [Retrieval Guide](retrieval.md) -- core dense, sparse, and hybrid retrieval
- [Storage Guide](storage.md) -- store protocols and implementations
- [Retrieval API Reference](../api/retrieval.md) -- full signatures for all classes
