# Retrieval Guide

anchor ships with three retrieval strategies that cover the most common
search paradigms: **dense** (embedding-based), **sparse** (BM25), and
**hybrid** (Reciprocal Rank Fusion). All retrievers implement the `Retriever`
protocol and can be plugged into a `ContextPipeline`.

## Concepts

### Dense Retrieval

Dense retrieval encodes documents and queries into fixed-size embedding vectors
and retrieves the most similar documents by cosine similarity. It excels at
capturing **semantic** meaning -- synonyms, paraphrases, and related concepts
are naturally close in the vector space.

### Sparse Retrieval

Sparse retrieval (BM25) operates over term frequencies. It works best for
**keyword-heavy** queries where exact term matching matters. No embedding
function is required.

### Hybrid Retrieval

Hybrid retrieval combines multiple retrievers and fuses their rankings using
**Reciprocal Rank Fusion (RRF)**. This typically outperforms either strategy
alone because dense and sparse signals are complementary.

---

## DenseRetriever

`DenseRetriever` uses an embedding function to index documents into a
`VectorStore` and retrieves them via similarity search.

```python
from anchor.retrieval import DenseRetriever

retriever = DenseRetriever(
    vector_store=vector_store,
    context_store=context_store,
    embed_fn=embed_fn,       # Callable[[str], list[float]]
    tokenizer=None,          # Optional Tokenizer override
)
```

| Parameter | Type | Description |
|---|---|---|
| `vector_store` | `VectorStore` | Backend for embedding storage and similarity search. |
| `context_store` | `ContextStore` | Backend for resolving item IDs to `ContextItem` objects. |
| `embed_fn` | `Callable[[str], list[float]] \| None` | Function that produces an embedding vector from text. |
| `tokenizer` | `Tokenizer \| None` | Optional tokenizer for counting tokens. Defaults to the built-in counter. |

### Indexing

```python
count = retriever.index(items)  # Returns number of items indexed
```

The `index()` method embeds each item's content via `embed_fn`, stores the
vector in the `VectorStore`, and saves the full `ContextItem` in the
`ContextStore`.

### Retrieving

```python
from anchor.models.query import QueryBundle

query = QueryBundle(query_str="How does authentication work?")
results = retriever.retrieve(query, top_k=5)
```

If the `QueryBundle` already carries a pre-computed `embedding`, the retriever
uses it directly. Otherwise it calls `embed_fn` on the query text.

!!! tip
    Pre-compute the query embedding once and set `query.embedding` if you need
    to share the same embedding across multiple retrievers.

---

## SparseRetriever

`SparseRetriever` uses BM25 for term-frequency-based retrieval.

!!! warning
    The `rank-bm25` package is required. Install it with:
    ```
    pip install astro-anchor[bm25]
    ```

```python
from anchor.retrieval import SparseRetriever

retriever = SparseRetriever(
    tokenize_fn=None,   # Optional custom tokenizer
    tokenizer=None,     # Optional Tokenizer for token counting
)
```

| Parameter | Type | Description |
|---|---|---|
| `tokenize_fn` | `Callable[[str], list[str]] \| None` | Text tokenizer for BM25. Defaults to whitespace + lowercase splitting. |
| `tokenizer` | `Tokenizer \| None` | Optional tokenizer for counting tokens on returned items. |

### Indexing and Retrieving

```python
retriever.index(items)
results = retriever.retrieve(query, top_k=5)
```

Scores are normalized to `[0, 1]` by dividing by the maximum BM25 score in
the corpus. Zero-score items are excluded from results.

!!! note
    The default tokenizer splits on whitespace and lowercases. For better
    results, provide a custom `tokenize_fn` that handles stemming, stop-word
    removal, or subword tokenization.

---

## HybridRetriever

`HybridRetriever` combines multiple retrievers and fuses results with
**Reciprocal Rank Fusion (RRF)**.

```python
from anchor.retrieval import HybridRetriever

hybrid = HybridRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    rrf_k=60,              # RRF smoothing constant
    weights=[0.7, 0.3],    # Per-retriever weights
)
```

| Parameter | Type | Description |
|---|---|---|
| `retrievers` | `list[Retriever]` | Sub-retrievers to combine. At least one is required. |
| `rrf_k` | `int` | RRF smoothing constant (default `60`). Higher values flatten score differences. |
| `weights` | `list[float] \| None` | Per-retriever weights. Defaults to equal weight `1.0` for each. |

### How RRF Works

For each document `d` across `N` ranking lists:

```
RRF(d) = sum(weight_i / (rrf_k + rank_i(d)))
```

The smoothing constant `rrf_k` (default 60, from the original RRF paper)
prevents top-ranked items from dominating excessively. Final scores are
normalized to `[0, 1]`.

!!! tip
    If one retriever fails, `HybridRetriever` skips it and continues with the
    remaining retrievers. It only raises `RetrieverError` when **all**
    sub-retrievers fail.

---

## rrf_fuse Utility

For cases where you want to fuse ranked lists outside of `HybridRetriever`,
use the standalone `rrf_fuse()` function.

```python
from anchor.retrieval import rrf_fuse

fused = rrf_fuse(
    ranked_lists=[dense_results, sparse_results],
    weights=[0.7, 0.3],
    k=60,
    top_k=10,
)
```

| Parameter | Type | Description |
|---|---|---|
| `ranked_lists` | `list[list[ContextItem]]` | Lists of items ranked by relevance. |
| `weights` | `list[float] \| None` | Per-list weights. Defaults to `1.0` each. |
| `k` | `int` | RRF smoothing constant (default `60`). |
| `top_k` | `int \| None` | Maximum items to return. `None` returns all. |

Returns a fused list with normalized scores and `retrieval_method: "rrf"` in
metadata.

---

## ScoreReranker

`ScoreReranker` is a simple postprocessor that rescores items using a
user-provided function. It implements the older `process()` interface (not the
`Reranker` protocol).

```python
from anchor.retrieval import ScoreReranker

reranker = ScoreReranker(
    score_fn=my_scorer,   # Callable[[str, str], float]
    top_k=5,              # Optional truncation
)
reranked = reranker.process(items, query)
```

| Parameter | Type | Description |
|---|---|---|
| `score_fn` | `Callable[[str, str], float]` | Takes `(query_str, doc_content)` and returns a relevance score. |
| `top_k` | `int \| None` | Maximum items to return. `None` keeps all items. |

!!! note
    For the `Reranker` protocol (used by `RerankerPipeline`), see
    [CrossEncoderReranker](advanced-retrieval.md) in the advanced retrieval
    guide.

---

## Complete Example

A runnable end-to-end example that indexes documents, retrieves with both
dense and sparse strategies, and fuses results with hybrid retrieval.

```python
import math

from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.retrieval import (
    DenseRetriever,
    HybridRetriever,
    ScoreReranker,
    SparseRetriever,
    rrf_fuse,
)
from anchor.storage import InMemoryContextStore, InMemoryVectorStore


# --- Deterministic embedding function (for demonstration) ---
def embed_fn(text: str) -> list[float]:
    """Produce a 4-dimensional embedding from character values."""
    vals = [ord(c) for c in text[:50]]
    raw = [
        math.sin(sum(vals)),
        math.cos(sum(vals)),
        math.sin(sum(vals) * 0.5),
        math.cos(sum(vals) * 0.5),
    ]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


# --- Create stores ---
vector_store = InMemoryVectorStore()
context_store = InMemoryContextStore()

# --- Prepare documents ---
docs = [
    ContextItem(content="Authentication uses JWT tokens for session management.", source=SourceType.RETRIEVAL),
    ContextItem(content="The database schema includes users and roles tables.", source=SourceType.RETRIEVAL),
    ContextItem(content="API rate limiting is enforced at 100 requests per minute.", source=SourceType.RETRIEVAL),
    ContextItem(content="User passwords are hashed with bcrypt before storage.", source=SourceType.RETRIEVAL),
]

# --- Dense retriever ---
dense = DenseRetriever(
    vector_store=vector_store,
    context_store=context_store,
    embed_fn=embed_fn,
)
dense.index(docs)

# --- Sparse retriever ---
sparse = SparseRetriever()
sparse.index(docs)

# --- Individual retrieval ---
query = QueryBundle(query_str="How does user authentication work?")
dense_results = dense.retrieve(query, top_k=3)
sparse_results = sparse.retrieve(query, top_k=3)

print("Dense results:")
for item in dense_results:
    print(f"  [{item.score:.3f}] {item.content[:60]}...")

print("\nSparse results:")
for item in sparse_results:
    print(f"  [{item.score:.3f}] {item.content[:60]}...")

# --- Hybrid retrieval ---
hybrid = HybridRetriever(
    retrievers=[dense, sparse],
    weights=[0.6, 0.4],
)
hybrid_results = hybrid.retrieve(query, top_k=3)

print("\nHybrid results:")
for item in hybrid_results:
    print(f"  [{item.score:.3f}] {item.content[:60]}...")

# --- Standalone RRF fusion ---
fused = rrf_fuse(
    ranked_lists=[dense_results, sparse_results],
    weights=[0.6, 0.4],
    top_k=3,
)

print("\nStandalone RRF fusion:")
for item in fused:
    print(f"  [{item.score:.3f}] {item.content[:60]}...")

# --- Reranking ---
def my_scorer(query_str: str, doc: str) -> float:
    """Simple keyword overlap scorer."""
    query_terms = set(query_str.lower().split())
    doc_terms = set(doc.lower().split())
    overlap = len(query_terms & doc_terms)
    return overlap / max(len(query_terms), 1)


reranker = ScoreReranker(score_fn=my_scorer, top_k=2)
reranked = reranker.process(hybrid_results, query)

print("\nReranked results:")
for item in reranked:
    print(f"  [{item.score:.3f}] {item.content[:60]}...")
```

---

## What's Next

- [Advanced Retrieval](advanced-retrieval.md) -- reranker pipelines, async
  retrievers, query routing, and cross-modal search
- [Storage Guide](storage.md) -- store protocols and implementations
- [Retrieval API Reference](../api/retrieval.md) -- full constructor and
  method signatures
