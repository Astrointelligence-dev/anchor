# Production Patterns

Battle-tested patterns for deploying anchor in production: error handling,
observability, performance tuning, and testing strategies.

---

## Error Handling and Resilience

Production pipelines must handle failures gracefully. Use `on_error` policies
and callbacks to keep your system running even when individual steps fail.

### Step-Level Error Policies

```python
from anchor import ContextPipeline, retriever_step, filter_step

pipeline = (
    ContextPipeline(max_tokens=8192)
    # If retrieval fails, skip this step and continue with other sources
    .add_step(
        retriever_step("primary-search", primary_retriever, top_k=10),
        on_error="skip",
    )
    # Fallback retriever in case the primary is down
    .add_step(
        retriever_step("fallback-search", fallback_retriever, top_k=5),
        on_error="skip",
    )
    # If filtering fails, raise immediately -- data quality matters
    .add_step(
        filter_step("quality-gate", lambda item: item.score > 0.3),
        on_error="raise",
    )
)
```

!!! tip "Error Policy Options"
    - `on_error="raise"` (default) -- stop the pipeline and propagate the exception
    - `on_error="skip"` -- log the error and continue with items from previous steps
    - `on_error="empty"` -- log the error and continue with an empty item list

### Error Monitoring with Callbacks

```python
import logging

from anchor import ContextPipeline, TracingCallback

logger = logging.getLogger("anchor")

class ErrorAlertCallback(TracingCallback):
    """Send alerts when pipeline steps fail."""

    def on_step_error(self, step_name: str, error: Exception) -> None:
        logger.error(f"Step '{step_name}' failed: {error}")
        # In production: send to PagerDuty, Slack, etc.
        send_alert(
            severity="warning",
            message=f"Pipeline step '{step_name}' failed: {error}",
        )

    def on_build_complete(self, result) -> None:
        if result.diagnostics.get("steps_skipped", 0) > 0:
            logger.warning(
                f"Pipeline completed with "
                f"{result.diagnostics['steps_skipped']} skipped steps"
            )

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_callback(ErrorAlertCallback())
    # ... steps ...
)
```

!!! warning "Callback Safety"
    Callback errors are swallowed and logged at WARNING level.
    A failing callback never breaks the pipeline.

---

## Token Budget Tuning

Choosing the right token budget is critical for balancing context quality
and cost. Start with a preset and tune from there.

### Built-in Presets

```python
from anchor import ContextPipeline
from anchor.models.budget_defaults import (
    default_chat_budget,
    default_rag_budget,
    default_agent_budget,
)

# Preset budgets for common application archetypes
chat_pipeline  = ContextPipeline(budget=default_chat_budget(max_tokens=16384))
rag_pipeline   = ContextPipeline(budget=default_rag_budget(max_tokens=32768))
agent_pipeline = ContextPipeline(budget=default_agent_budget(max_tokens=65536))
```

### Custom Budget Allocation

For fine-grained control, allocate tokens across sources:

```python
from anchor import ContextPipeline
from anchor.models.budget import TokenBudget, BudgetAllocation
from anchor.models.context import SourceType

budget = TokenBudget(
    total_tokens=16384,
    reserve_tokens=1024,  # reserve for LLM response overhead
    allocations=[
        BudgetAllocation(source=SourceType.SYSTEM, max_tokens=1024, priority=10),
        BudgetAllocation(source=SourceType.CONVERSATION, max_tokens=4096, priority=7),
        BudgetAllocation(source=SourceType.MEMORY, max_tokens=512, priority=8),
        BudgetAllocation(
            source=SourceType.RETRIEVAL,
            max_tokens=8192,
            priority=5,
            overflow_strategy="truncate",
        ),
    ],
)

pipeline = ContextPipeline(budget=budget)
```

!!! warning "Overflow Strategies"
    Each ``BudgetAllocation`` has an ``overflow_strategy``:

    - `"truncate"` (default) -- truncate content to fit within the allocation
    - `"drop"` -- drop entire items that exceed the allocation

### Monitoring Token Usage

```python
result = pipeline.build(query)
diag = result.diagnostics

print(f"Tokens used:     {diag['tokens_used']}")
print(f"Tokens budget:   {diag['tokens_budget']}")
print(f"Utilization:     {diag['token_utilization']:.1%}")
print(f"Items included:  {diag['items_included']}")
print(f"Items overflow:  {diag['items_overflow']}")

# Alert if utilization is consistently low (wasting budget)
if diag["token_utilization"] < 0.3:
    logger.info("Token utilization below 30% -- consider reducing budget")
```

---

## Memory Management at Scale

### Eviction Strategies

```python
from anchor import (
    MemoryManager,
    SlidingWindowMemory,
    ImportanceEviction,
    PairedEviction,
    InMemoryEntryStore,
)
from anchor.memory import FIFOEviction

# FIFO eviction (default) -- simplest, good for most cases
fifo_memory = SlidingWindowMemory(max_tokens=4096)

# Importance-based eviction -- keeps high-value turns longer
importance_memory = SlidingWindowMemory(
    max_tokens=4096,
    eviction_policy=ImportanceEviction(
        importance_fn=lambda turn: 1.0 if "action" in turn.content.lower() else 0.5,
    ),
)

# Paired eviction -- evicts user+assistant turn pairs together
# to prevent orphaned context
paired_memory = SlidingWindowMemory(
    max_tokens=4096,
    eviction_policy=PairedEviction(),
)
```

### Consolidation and Garbage Collection

```python
memory = MemoryManager(
    conversation_tokens=4096,
    persistent_store=InMemoryEntryStore(),
)

# Periodically consolidate duplicate/overlapping facts
consolidated = memory.consolidate_facts(
    similarity_threshold=0.85,
    merge_strategy="keep_newest",
)
print(f"Consolidated {consolidated} duplicate facts")

# Remove stale facts older than 30 days
removed = memory.gc_facts(max_age_days=30)
print(f"Garbage collected {removed} stale facts")
```

!!! tip "Consolidation in Production"
    Run `consolidate_facts()` on a schedule (e.g., daily cron job) rather
    than on every request. Consolidation requires comparing all fact pairs,
    which is O(n^2) in the number of facts.

---

## Hybrid Retrieval Optimization

### Weight Tuning

```python
from anchor import HybridRetriever, DenseRetriever, SparseRetriever

# Start with 70/30 dense/sparse and tune based on benchmarks
hybrid = HybridRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3],
    rrf_k=60,
)

# Grid search over weights using retrieval metrics
best_score = 0
best_weights = [0.5, 0.5]

for dense_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
    sparse_weight = 1.0 - dense_weight
    hybrid = HybridRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight],
        rrf_k=60,
    )

    # Benchmark using your dataset (see the Evaluation Workflow cookbook)
    score = run_benchmark(hybrid, benchmark_dataset)
    if score > best_score:
        best_score = score
        best_weights = [dense_weight, sparse_weight]

print(f"Best weights: dense={best_weights[0]}, sparse={best_weights[1]}")
```

### Reranker Selection

```python
from anchor import CrossEncoderReranker, reranker_step

# Word-overlap scorer (fast, no API needed)
def overlap_scorer(query: str, doc: str) -> float:
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    return len(q_words & d_words) / max(len(q_words), 1)

fast_reranker = CrossEncoderReranker(score_fn=overlap_scorer, top_k=10)

# Cross-encoder scorer (slower, more accurate, requires a model)
# from sentence_transformers import CrossEncoder
# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# def cross_encoder_scorer(query: str, doc: str) -> float:
#     return model.predict([(query, doc)])[0]
# accurate_reranker = CrossEncoderReranker(
#     score_fn=cross_encoder_scorer, top_k=10
# )
```

!!! note "Reranker Trade-offs"
    - **Word-overlap**: fast (< 1ms per doc), no dependencies, moderate quality
    - **Cross-encoder**: slow (50-100ms per doc), high quality, needs GPU for scale
    - **Compromise**: retrieve top 50 with dense, rerank top 10 with cross-encoder

---

## Observability Setup

### TracingCallback

```python
import time
from anchor import ContextPipeline, TracingCallback

class ProductionTracer(TracingCallback):
    """Full lifecycle tracing for production monitoring."""

    def on_build_start(self, query) -> None:
        self.start_time = time.time()
        logger.info(f"Pipeline build started: {query.query_str[:50]}")

    def on_step_start(self, step_name: str) -> None:
        logger.debug(f"Step '{step_name}' started")

    def on_step_complete(self, step_name: str, items_count: int) -> None:
        logger.debug(f"Step '{step_name}' completed: {items_count} items")

    def on_step_error(self, step_name: str, error: Exception) -> None:
        logger.error(f"Step '{step_name}' failed: {error}")

    def on_build_complete(self, result) -> None:
        elapsed = time.time() - self.start_time
        logger.info(
            f"Pipeline build complete: "
            f"{result.diagnostics['items_included']} items, "
            f"{elapsed:.2f}s"
        )
        # Emit metrics to your monitoring system
        metrics.histogram("pipeline.build_time_ms", elapsed * 1000)
        metrics.gauge(
            "pipeline.token_utilization",
            result.diagnostics["token_utilization"],
        )
```

### CostTracker

```python
from anchor import CostTracker

tracker = CostTracker()

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_callback(tracker)
    # ... steps ...
)

# After building context
result = pipeline.build(query)

# Inspect costs
print(f"Total tokens: {tracker.total_tokens}")
print(f"Estimated cost: ${tracker.estimated_cost:.4f}")
print(f"Per-step breakdown: {tracker.step_costs}")
```

### OpenTelemetry (OTLP) Export

```python
from anchor import ContextPipeline
from anchor.observability.otlp import OTLPSpanExporter

# Export traces to your OTLP-compatible backend (OTLP/HTTP)
exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318",
    service_name="my-rag-service",
    headers={"Authorization": "Bearer <token>"},
)

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_callback(exporter)
    # ... steps ...
)
```

!!! tip "Structured Logging"
    Combine `TracingCallback` with structured logging (e.g., `structlog`)
    to get machine-parseable logs with correlation IDs that link pipeline
    builds to upstream HTTP requests.

!!! note "OTLP Dependencies"
    OTLP export requires `pip install astro-anchor[otlp]`.

---

## Testing Context Pipelines

### Unit Testing Individual Steps

```python
import pytest
from anchor import ContextItem, QueryBundle, SourceType

def test_quality_filter():
    """Test that the quality filter removes low-score items."""
    items = [
        ContextItem(content="Good result", source=SourceType.RETRIEVAL, score=0.9),
        ContextItem(content="Bad result", source=SourceType.RETRIEVAL, score=0.05),
        ContextItem(content="OK result", source=SourceType.RETRIEVAL, score=0.4),
    ]

    quality_filter = lambda item: item.score > 0.3
    filtered = [item for item in items if quality_filter(item)]

    assert len(filtered) == 2
    assert all(item.score > 0.3 for item in filtered)


def test_custom_retriever_returns_context_items():
    """Test that a custom retriever returns properly typed results."""
    retriever = MyCustomRetriever()
    retriever.index(sample_items)

    query = QueryBundle(query_str="test query")
    results = retriever.retrieve(query, top_k=5)

    assert isinstance(results, list)
    assert all(isinstance(item, ContextItem) for item in results)
    assert len(results) <= 5
```

### Integration Testing the Full Pipeline

```python
def test_full_pipeline_build():
    """Test end-to-end pipeline build with all components."""
    # Setup
    retriever = create_test_retriever(sample_docs)
    memory = create_test_memory()
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi there!")

    pipeline = (
        ContextPipeline(max_tokens=4096)
        .add_step(retriever_step("search", retriever, top_k=5))
        .with_memory(memory)
        .with_formatter(GenericTextFormatter())
        .add_system_prompt("You are a test assistant.")
    )

    # Act
    result = pipeline.build("What is context engineering?")

    # Assert
    assert result.formatted_output is not None
    assert result.diagnostics["items_included"] > 0
    assert result.diagnostics["token_utilization"] > 0
    assert result.diagnostics["token_utilization"] <= 1.0

    # Verify sources are present
    sources = {item.source for item in result.window.items}
    assert SourceType.SYSTEM in sources
    assert SourceType.RETRIEVAL in sources
    assert SourceType.MEMORY in sources


def test_pipeline_handles_empty_retrieval():
    """Test that the pipeline handles no results gracefully."""
    empty_retriever = create_test_retriever([])  # no docs

    pipeline = (
        ContextPipeline(max_tokens=4096)
        .add_step(retriever_step("search", empty_retriever, top_k=5))
        .add_system_prompt("You are a test assistant.")
    )

    result = pipeline.build("query with no results")

    # Should still build successfully with just the system prompt
    assert result.formatted_output is not None
    assert result.diagnostics["items_included"] >= 1  # at least system prompt
```

!!! warning "Deterministic Tests"
    Use deterministic embedding functions (like the `embed_fn` in the examples)
    for tests. Real embedding models produce slightly different vectors across
    runs, which makes assertions flaky.

---

## Performance Tips

### Async Pipelines

```python
import asyncio
from anchor import ContextPipeline, async_retriever_step

# Use async steps for I/O-bound operations (API calls, database queries)
pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(async_retriever_step("vector-db", async_retriever, top_k=10))
    .add_step(async_retriever_step("graph-db", async_graph_retriever, top_k=5))
)

# abuild() runs async steps concurrently when possible
result = await pipeline.abuild(query)
```

### Caching Retrieved Results

```python
from functools import lru_cache
from anchor import ContextItem, QueryBundle

class CachedRetriever:
    """Wraps a retriever with an LRU cache for repeated queries."""

    def __init__(self, inner_retriever, cache_size: int = 256):
        self._inner = inner_retriever
        self._cache_size = cache_size

        @lru_cache(maxsize=cache_size)
        def _cached_retrieve(query_str: str, top_k: int):
            q = QueryBundle(query_str=query_str)
            return tuple(self._inner.retrieve(q, top_k=top_k))

        self._cached_retrieve = _cached_retrieve

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return list(self._cached_retrieve(query.query_str, top_k))
```

### Lazy Loading Heavy Dependencies

```python
from anchor import ContextPipeline

class LazyEmbeddingRetriever:
    """Load the embedding model only on first use."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._retriever = None

    def _ensure_loaded(self):
        if self._model is None:
            # Heavy import and model load happens only once
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._retriever = DenseRetriever(
                vector_store=InMemoryVectorStore(),
                context_store=InMemoryContextStore(),
                embed_fn=lambda text: self._model.encode(text).tolist(),
            )

    def retrieve(self, query, top_k=10):
        self._ensure_loaded()
        return self._retriever.retrieve(query, top_k=top_k)
```

!!! tip "Profile Before Optimizing"
    Use `result.diagnostics["steps"]` to identify which pipeline steps are
    slowest before adding complexity like caching or async. The step timing
    breakdown is included in every `BuildResult`.

!!! tip "Connection Pooling"
    When using external vector databases (Pinecone, Weaviate, Qdrant), reuse
    client connections across requests. Create the client once at startup and
    pass it to your retriever, rather than creating a new connection per query.
