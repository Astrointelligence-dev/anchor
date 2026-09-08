---
icon: material/school
---

# Your First Pipeline

This tutorial picks up where the [Quickstart](quickstart.md) left off. You
will add retrieval, hybrid search, provider formatting, token budgets,
custom steps, async support, query transformation, and diagnostics to your
pipeline. By the end you will have a solid understanding of every major
feature in anchor.

---

## Adding Retrieval

Add semantic search with dense retrieval:

```python
from anchor import (
    ContextPipeline,
    QueryBundle,
    ContextItem,
    SourceType,
    DenseRetriever,
    InMemoryContextStore,
    InMemoryVectorStore,
    retriever_step,
)

# You provide the embedding function (any provider works)
def my_embed_fn(text: str) -> list[float]:
    # Replace with your actual embedding call
    # e.g., openai.embeddings.create(model="text-embedding-3-small", input=text)
    return [0.0] * 384

# Set up retriever
retriever = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

# Index some documents
docs = [
    ContextItem(content="Python lists support .sort() and sorted().", source=SourceType.RETRIEVAL),
    ContextItem(content="Use lambda for custom sort keys.", source=SourceType.RETRIEVAL),
]
retriever.index(docs)

# Build pipeline with retrieval
pipeline = (
    ContextPipeline(max_tokens=4096)
    .add_step(retriever_step("search", retriever, top_k=5))
)

result = pipeline.build(QueryBundle(query_str="How to sort in Python?"))
print(f"Found {len(result.window.items)} context items")
print(f"Used {result.window.used_tokens}/{result.window.max_tokens} tokens")
```

!!! info "Bring your own embeddings"
    anchor never calls an embedding provider directly. You supply the
    `embed_fn` and can use OpenAI, Cohere, local models, or anything else.

---

## Hybrid Retrieval (Dense + BM25)

Combine dense and sparse retrieval with Reciprocal Rank Fusion:

```python
from anchor import (
    ContextPipeline,
    ContextItem,
    SourceType,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    InMemoryVectorStore,
    InMemoryContextStore,
    retriever_step,
)

# Create individual retrievers
vector_store = InMemoryVectorStore()
context_store = InMemoryContextStore()
dense = DenseRetriever(
    vector_store=vector_store,
    context_store=context_store,
    embed_fn=my_embed_fn,
)
sparse = SparseRetriever()

# Index documents in both
items = [
    ContextItem(content="Python lists support .sort() and sorted().", source=SourceType.RETRIEVAL),
    ContextItem(content="Use lambda for custom sort keys.", source=SourceType.RETRIEVAL),
]
dense.index(items)
sparse.index(items)

# Combine with RRF
hybrid = HybridRetriever(
    retrievers=[dense, sparse],
    weights=[0.6, 0.4],  # 60% dense, 40% sparse
)

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(retriever_step("hybrid_search", hybrid, top_k=10))
)
```

!!! note
    BM25 sparse retrieval requires the optional `bm25` extra:
    `pip install astro-anchor[bm25]`

---

## Formatting for Different Providers

anchor can format the assembled context window for any major LLM
provider:

```python
from anchor import AnthropicFormatter, OpenAIFormatter, GenericTextFormatter

# Anthropic format: {"system": "...", "messages": [...]}
pipeline.with_formatter(AnthropicFormatter())

# OpenAI format: {"messages": [{"role": "system", ...}, ...]}
pipeline.with_formatter(OpenAIFormatter())

# Plain text with section headers
pipeline.with_formatter(GenericTextFormatter())
```

!!! tip
    `with_formatter()` returns the pipeline, so you can chain it:
    ```python
    result = pipeline.with_formatter(AnthropicFormatter()).build("Hello")
    ```

---

## Token Budgets

For fine-grained control over how tokens are allocated across sources, use
`TokenBudget` with a preset factory:

```python
from anchor import ContextPipeline, default_chat_budget

budget = default_chat_budget(max_tokens=8192)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

### Default allocations

The `default_chat_budget` allocates tokens as follows:

| Source | Allocation | Tokens (8 192) |
|--------|-----------|---------------|
| System prompts | 10% | 819 |
| Persistent memory | 10% | 819 |
| Conversation turns | 20% | 1 638 |
| Retrieval results | 25% | 2 048 |
| Reserved for LLM response | 15% | 1 228 |
| Shared pool (unallocated) | 20% | 1 638 |

### Preset factories

Three preset factories are available:

- **`default_chat_budget(max_tokens)`** -- Conversational apps. 60% of usable
  tokens go to conversation and memory.
- **`default_rag_budget(max_tokens)`** -- RAG-heavy apps. 40% of usable tokens
  go to retrieval results.
- **`default_agent_budget(max_tokens)`** -- Agentic apps. Includes a 15% tool
  allocation and balances across all sources.

!!! warning
    The `reserve_tokens` field (15% by default) is subtracted from `max_tokens`
    before any items are placed. Make sure your pipeline's `max_tokens`
    is large enough to leave room after the reservation.

### Custom budgets

You can also construct a custom `TokenBudget` directly:

```python
from anchor import TokenBudget, BudgetAllocation, SourceType

budget = TokenBudget(
    total_tokens=8192,
    reserve_tokens=1200,
    allocations=[
        BudgetAllocation(source=SourceType.SYSTEM, max_tokens=800, priority=10),
        BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=4000, priority=5),
    ],
)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

---

## Decorator API

Instead of using `add_step()` with factory functions, you can register pipeline
steps using the `@pipeline.step` decorator. This is especially convenient for
custom post-processing logic:

```python
from anchor import ContextPipeline, ContextItem, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def boost_recent(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Boost scores of recent items."""
    return [
        item.model_copy(update={"score": min(1.0, item.score * 1.5)})
        if item.metadata.get("recent")
        else item
        for item in items
    ]

@pipeline.step(name="quality-filter")
def remove_low_quality(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Filter out low-scoring items."""
    return [item for item in items if item.score > 0.3]

result = pipeline.build("How to sort in Python?")
```

Step functions must accept two arguments -- `items: list[ContextItem]` and
`query: QueryBundle` -- and return a `list[ContextItem]`.

You can also pass `on_error="skip"` to gracefully skip a step if it raises:

```python
@pipeline.step(name="optional-enrichment", on_error="skip")
def enrich(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """This step is skipped if it fails instead of crashing the pipeline."""
    return items
```

!!! note
    Passing an async function to `@pipeline.step` raises a `TypeError`.
    Use `@pipeline.async_step` for async functions (see below).

---

## Async Pipeline

For pipelines that include async steps (e.g., database lookups, API calls),
use `@pipeline.async_step` and call `abuild()` instead of `build()`:

```python
import asyncio
from anchor import ContextPipeline, ContextItem, SourceType, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.async_step
async def fetch_from_db(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Fetch relevant context from an async database."""
    # Replace with your actual async database call
    await asyncio.sleep(0)  # placeholder for async I/O
    new_items = [
        ContextItem(
            content="Retrieved from database",
            source=SourceType.RETRIEVAL,
            score=0.9,
        )
    ]
    return items + new_items

@pipeline.step
def filter_results(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Sync steps and async steps can be mixed in the same pipeline."""
    return [item for item in items if item.score > 0.5]

# Use abuild() instead of build() to run the async pipeline
result = asyncio.run(pipeline.abuild("What is context engineering?"))
```

!!! warning
    If your pipeline contains **any** async steps, you **must** use `abuild()`.
    Calling `build()` on a pipeline with async steps will raise an error.

You can also use `@pipeline.async_step` with keyword arguments:

```python
@pipeline.async_step(name="db-lookup", on_error="skip")
async def db_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    results = await my_async_search(query.query_str)
    return items + results
```

---

## Query Transformation

Query transformers rewrite or expand the user's query before retrieval.
anchor ships with four built-in transformers.

### HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical answer and uses it as the retrieval query.
The intuition: embedding a plausible answer is closer in vector space to the
real answer than the question itself.

```python
from anchor import (
    ContextPipeline,
    HyDETransformer,
    query_transform_step,
    DenseRetriever,
    InMemoryVectorStore,
    InMemoryContextStore,
)

def generate_hypothetical(query: str) -> str:
    """Replace with your actual LLM call."""
    return f"A hypothetical answer to: {query}"

hyde = HyDETransformer(generate_fn=generate_hypothetical)

retriever = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

pipeline = ContextPipeline(max_tokens=8192).add_step(
    query_transform_step("hyde-search", transformer=hyde, retriever=retriever, top_k=10)
)

result = pipeline.build("What causes memory leaks in Python?")
```

!!! note
    anchor never calls an LLM directly. You provide the generation
    function (`generate_fn`) and the transformers handle orchestration.

### Other transformers

- **`MultiQueryTransformer`** -- Generates N alternative phrasings for broader
  retrieval coverage. Provide `generate_fn: (str, int) -> list[str]`.
- **`DecompositionTransformer`** -- Breaks a complex query into simpler
  sub-questions. Provide `generate_fn: (str) -> list[str]`.
- **`StepBackTransformer`** -- Generates a more abstract version of the query
  alongside the original. Provide `generate_fn: (str) -> str`.

### Chaining transformers

Use `QueryTransformPipeline` to chain multiple transformers:

```python
from anchor import QueryTransformPipeline, HyDETransformer, StepBackTransformer, QueryBundle

hyde = HyDETransformer(generate_fn=generate_hypothetical)
step_back = StepBackTransformer(generate_fn=lambda q: f"General context for: {q}")

chain = QueryTransformPipeline(transformers=[step_back, hyde])
queries = chain.transform(QueryBundle(query_str="Why does my Flask app leak memory?"))
# Returns deduplicated list of QueryBundle objects
```

---

## Diagnostics

Every `build()` call returns a `ContextResult` with detailed diagnostics:

```python
result = pipeline.build("What is context engineering?")

print(result.diagnostics)
# {
#     "steps": [
#         {"name": "search", "items_after": 15, "time_ms": 2.1}
#     ],
#     "total_items_considered": 15,
#     "items_included": 10,
#     "items_overflow": 5,
#     "token_utilization": 0.87,
# }

print(f"Build time: {result.build_time_ms:.1f}ms")
print(f"Token utilization: {result.diagnostics['token_utilization']:.0%}")
print(f"Overflow items: {len(result.overflow_items)}")
```

### Interpreting diagnostics

The `diagnostics` dictionary contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `list[StepDiagnostic]` | Per-step name, item count, and timing |
| `memory_items` | `int` | Number of items contributed by memory |
| `total_items_considered` | `int` | Total items before window assembly |
| `items_included` | `int` | Items that fit in the context window |
| `items_overflow` | `int` | Items that did not fit |
| `token_utilization` | `float` | Fraction of token budget used (0.0--1.0) |
| `token_usage_by_source` | `dict[str, int]` | Per-source token counts (when using budgets) |
| `budget_overflow_by_source` | `dict[str, int]` | Per-source overflow counts (when using budgets) |
| `shared_pool_usage` | `int` | Tokens used by non-allocated sources (when using budgets) |
| `skipped_steps` | `list[str]` | Steps that failed with `on_error="skip"` |
| `failed_step` | `str` | Step that caused a pipeline failure |
| `query_enriched` | `bool` | Whether query enrichment was applied |

!!! tip
    A `token_utilization` close to 1.0 means you are making good use of your
    context window. If it is consistently low, consider increasing `top_k` on
    your retriever steps or lowering `max_tokens`.

!!! warning
    If `items_overflow` is high, important context may be getting dropped.
    Consider increasing `max_tokens`, tuning per-source budgets, or filtering
    low-quality items earlier in the pipeline.

---

## Overflow Items

Items that did not fit in the context window are available in
`result.overflow_items`:

```python
for item in result.overflow_items:
    print(f"  [{item.source}] priority={item.priority} tokens={item.token_count}")
```

Use overflow data to decide whether to increase `max_tokens`, adjust budgets,
or apply stricter filtering in earlier pipeline steps.

---

## Priority System

Every `ContextItem` has a `priority` field (1--10) that controls placement
order. Higher priority items are placed first and are never evicted in favor
of lower priority items.

| Priority | Source | Usage |
|----------|--------|-------|
| 10 | System prompts | Instructions, persona, rules |
| 8 | Persistent memory | Long-term facts from `MemoryManager.add_fact()` |
| 7 | Conversation memory | Recent chat turns |
| 5 | Retrieval (default) | RAG results from retrievers |
| 1--4 | Custom | Low-priority supplementary context |

!!! info
    The priority system works together with token budgets. Within a single
    source category, items are ordered by priority first, then by score.

---

## Where to Go Next

Now that you have the basics, dive deeper into specific topics:

**Guides**

- [Pipeline Guide](../guides/pipeline.md) -- Steps, callbacks, decorators, and error handling
- [Retrieval Guide](../guides/retrieval.md) -- Dense, sparse, hybrid retrieval and reranking
- [Memory Guide](../guides/memory.md) -- Sliding window, summary buffer, and long-term facts
- [Knowledge Graph Guide](../guides/knowledge-graph.md) -- Entities, evidenced relations, scope-aware navigation
- [Ingestion Guide](../guides/ingestion.md) -- Parsing, chunking, and indexing documents
- [Query Transformation Guide](../guides/query-transform.md) -- HyDE, multi-query, decomposition
- [Evaluation Guide](../guides/evaluation.md) -- Measuring retrieval and RAG quality
- [Observability Guide](../guides/observability.md) -- Tracing, metrics, and cost tracking

**Concepts**

- [Architecture](../concepts/architecture.md) -- How the pipeline, window, and priority system work

**API Reference**

- [Pipeline API](../api/pipeline.md) -- `ContextPipeline`, `PipelineStep`, factory functions
- [Retrieval API](../api/retrieval.md) -- Retrievers, rerankers, and fusion
- [Memory API](../api/memory.md) -- `MemoryManager`, eviction, consolidation
- [Models API](../api/models.md) -- `ContextItem`, `QueryBundle`, `TokenBudget`
- [Query API](../api/query.md) -- Transformers and classifiers
