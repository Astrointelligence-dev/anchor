# Pipeline API Reference

API reference for the pipeline module -- the orchestration layer of anchor.

## `ContextPipeline`

The main orchestrator that assembles context from multiple sources into a
token-aware, priority-ranked context window.

```python
from anchor import ContextPipeline

class ContextPipeline:
    def __init__(
        self,
        max_tokens: int = 8192,
        tokenizer: Tokenizer | None = None,
        budget: TokenBudget | None = None,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `8192` | Maximum token budget for the context window. Must be positive. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. Falls back to the built-in `TiktokenCounter`. |
| `budget` | `TokenBudget \| None` | `None` | Optional token budget for fine-grained per-source allocation. |

**Raises:** `ValueError` if `max_tokens <= 0`.

### Properties

| Property | Type | Description |
|---|---|---|
| `max_tokens` | `int` | The maximum token budget for the context window. |
| `formatter` | `Formatter` | The current output formatter. |
| `steps` | `list[PipelineStep]` | A copy of the registered pipeline steps. |
| `system_items` | `list[ContextItem]` | A copy of the registered system items. |
| `budget` | `TokenBudget \| None` | The optional token budget. |

### Methods

#### `add_step(step) -> ContextPipeline`

Add a pipeline step. Returns `self` for chaining.

```python
def add_step(self, step: PipelineStep) -> ContextPipeline: ...
```

#### `with_memory(memory) -> ContextPipeline`

Attach a memory provider. Any object satisfying the `MemoryProvider` protocol
(i.e. having a `get_context_items() -> list[ContextItem]` method) is accepted.

```python
def with_memory(self, memory: MemoryProvider) -> ContextPipeline: ...
```

#### `with_budget(budget) -> ContextPipeline`

Attach a `TokenBudget` for fine-grained allocation.

```python
def with_budget(self, budget: TokenBudget) -> ContextPipeline: ...
```

#### `with_formatter(formatter) -> ContextPipeline`

Set the output formatter.

```python
def with_formatter(self, formatter: Formatter) -> ContextPipeline: ...
```

#### `add_system_prompt(content, priority=10) -> ContextPipeline`

Add a system prompt as a high-priority context item with `source=SourceType.SYSTEM`.

```python
def add_system_prompt(self, content: str, priority: int = 10) -> ContextPipeline: ...
```

#### `add_callback(callback) -> ContextPipeline`

Register an event callback for pipeline observability.

```python
def add_callback(self, callback: PipelineCallback) -> ContextPipeline: ...
```

#### `with_query_enricher(enricher) -> ContextPipeline`

Attach a query enricher for memory-aware query expansion. The enricher is called
after memory items are collected but before pipeline steps execute.

```python
def with_query_enricher(self, enricher: ContextQueryEnricher) -> ContextPipeline: ...
```

#### `step(fn=None, *, name=None, on_error="raise")`

Decorator to register a synchronous function as a pipeline step. Usable with or
without arguments: `@pipeline.step` or `@pipeline.step(name="x", on_error="skip")`.

**Raises:** `TypeError` if the function is async (use `async_step` instead).

#### `async_step(fn=None, *, name=None, on_error="raise")`

Decorator to register an async function as a pipeline step. Same usage pattern
as `step`.

**Raises:** `TypeError` if the function is not async.

#### `build(query) -> ContextResult`

Execute the full pipeline synchronously and return assembled context.

```python
def build(self, query: str | QueryBundle) -> ContextResult: ...
```

Accepts either a plain string (auto-wrapped in `QueryBundle`) or a `QueryBundle`.

#### `abuild(query) -> ContextResult`

Execute the full pipeline asynchronously. Supports both sync and async steps:
sync steps are called directly, async steps are awaited.

```python
async def abuild(self, query: str | QueryBundle) -> ContextResult: ...
```

---

## `PipelineStep`

A single composable step in the context pipeline. This is a dataclass.

```python
from anchor import PipelineStep

@dataclass(slots=True)
class PipelineStep:
    name: str
    fn: SyncStepFn | AsyncStepFn
    is_async: bool = False
    on_error: Literal["raise", "skip"] = "raise"
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Human-readable name for diagnostics. |
| `fn` | `StepFn` | (required) | The callable implementing the step logic. |
| `is_async` | `bool` | `False` | Whether `fn` is an async function. |
| `on_error` | `"raise" \| "skip"` | `"raise"` | Error handling policy. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata. |

### Methods

#### `execute(items, query) -> list[ContextItem]`

Execute the step synchronously. Raises `TypeError` if the step is async.

#### `aexecute(items, query) -> list[ContextItem]`

Execute the step asynchronously. Works for both sync and async step functions.

---

## Factory Functions

### `retriever_step(name, retriever, top_k=10, *, scope=None) -> PipelineStep`

Create a step from a `Retriever` protocol implementation. Appends retrieved
items to the current list. `scope` is intersected with the scope published by
the running agent turn (`Agent.with_scope`, a parent's scope), so pipeline
retrieval can only narrow.

```python
from anchor import retriever_step
step = retriever_step("search", my_retriever, top_k=5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `retriever` | `Retriever` | (required) | Object with `retrieve(query, top_k, *, scope=None)` method. |
| `top_k` | `int` | `10` | Maximum items to retrieve. |
| `scope` | `RetrievalScope \| None` | `None` | Static namespace scope for this step (include/exclude prefixes, exclude wins). |

### `async_retriever_step(name, retriever, top_k=10, *, scope=None) -> PipelineStep`

Async variant. Wraps an `AsyncRetriever` (must have `aretrieve(query, top_k, *, scope=None)`).

```python
from anchor import async_retriever_step
step = async_retriever_step("async-search", my_async_retriever, top_k=5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `retriever` | `AsyncRetriever` | (required) | Object with `aretrieve(query, top_k)` method. |
| `top_k` | `int` | `10` | Maximum items to retrieve. |

### `filter_step(name, predicate) -> PipelineStep`

Create a step that filters items by a predicate function. Items where the
predicate returns `False` are removed.

```python
from anchor import filter_step
step = filter_step("score-gate", lambda item: item.score > 0.5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `predicate` | `Callable[[ContextItem], bool]` | (required) | Returns `True` to keep an item. |

### `postprocessor_step(name, processor) -> PipelineStep`

Create a step from a `PostProcessor` protocol implementation.

```python
from anchor import postprocessor_step
step = postprocessor_step("dedup", my_deduplicator)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `processor` | `PostProcessor` | (required) | Object with `process(items, query)` method. |

### `async_postprocessor_step(name, processor) -> PipelineStep`

Async variant. Wraps an `AsyncPostProcessor` (must have `aprocess(items, query)`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `processor` | `AsyncPostProcessor` | (required) | Object with `aprocess(items, query)` method. |

### `reranker_step(name, reranker, top_k=10) -> PipelineStep`

Create a step from a `Reranker` protocol implementation. The reranker scores
and returns the top-k most relevant items.

```python
from anchor import reranker_step
step = reranker_step("rerank", my_reranker, top_k=3)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `reranker` | `Reranker` | (required) | Object with `rerank(query, items, top_k)` method. |
| `top_k` | `int` | `10` | Maximum items the reranker should return. |

### `async_reranker_step(name, reranker, top_k=10) -> PipelineStep`

Async variant. Wraps an `AsyncReranker` (must have `arerank(query, items, top_k)`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `reranker` | `AsyncReranker` | (required) | Object with `arerank(query, items, top_k)` method. |
| `top_k` | `int` | `10` | Maximum items the reranker should return. |

### `query_transform_step(name, transformer, retriever, top_k=10) -> PipelineStep`

Create a step that expands the query into multiple variants, retrieves for each,
and merges results using Reciprocal Rank Fusion (RRF). New items are
deduplicated by ID.

```python
from anchor import query_transform_step
step = query_transform_step("multi-query", my_transformer, my_retriever, top_k=5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `transformer` | `QueryTransformer` | (required) | Expands a single query into multiple queries. |
| `retriever` | `Retriever` | (required) | Retriever to run against each expanded query. |
| `top_k` | `int` | `10` | Maximum items to retrieve per query variant. |

### `classified_retriever_step(name, classifier, retrievers, default=None, top_k=10) -> PipelineStep`

Create a step that classifies the query and routes to the appropriate retriever.

```python
from anchor import classified_retriever_step
step = classified_retriever_step(
    "router",
    classifier=my_classifier,
    retrievers={"technical": tech_retriever, "general": general_retriever},
    default="general",
    top_k=5,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Step name for diagnostics. |
| `classifier` | `QueryClassifier` | (required) | Object with `classify(query) -> str` method. |
| `retrievers` | `dict[str, Retriever]` | (required) | Mapping from class label to retriever. |
| `default` | `str \| None` | `None` | Fallback key when the label is not found. |
| `top_k` | `int` | `10` | Maximum items to retrieve. |

**Raises:** `RetrieverError` if the label has no matching retriever and no default.

### `auto_promotion_step(extractor, store, consolidator=None, name="auto_promotion", on_error="skip") -> PipelineStep`

Create a step that extracts and stores memories from context. This is a
side-effect-only step that returns items unchanged.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `extractor` | `MemoryExtractor` | (required) | Extracts `MemoryEntry` objects from conversation turns. |
| `store` | `MemoryEntryStore` | (required) | Persistence backend for memory entries. |
| `consolidator` | `MemoryConsolidator \| None` | `None` | Optional deduplication against existing entries. |
| `name` | `str` | `"auto_promotion"` | Step name for diagnostics. |
| `on_error` | `"raise" \| "skip"` | `"skip"` | Error handling policy. |

### `graph_retrieval_step(graph, store, entity_extractor, ...) -> PipelineStep`

Create a step that retrieves memory entries linked to graph entities via BFS
traversal.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph` | `SimpleGraphMemory` | (required) | Graph memory instance to traverse. |
| `store` | `MemoryEntryStore` | (required) | Store holding `MemoryEntry` objects. |
| `entity_extractor` | `Callable[[str], list[str]]` | (required) | Maps query string to entity IDs. |
| `max_depth` | `int` | `2` | Maximum BFS traversal depth. |
| `max_items` | `int` | `5` | Maximum `ContextItem` objects to return. |
| `name` | `str` | `"graph_retrieval"` | Step name for diagnostics. |
| `on_error` | `"raise" \| "skip"` | `"skip"` | Error handling policy. |

### `create_eviction_promoter(extractor, store, consolidator=None) -> Callable`

Create an `on_evict` callback that promotes evicted conversation turns to
long-term memory. Designed for `SlidingWindowMemory(on_evict=...)`.

```python
from anchor import create_eviction_promoter, SlidingWindowMemory

promoter = create_eviction_promoter(extractor, store, consolidator)
memory = SlidingWindowMemory(max_tokens=4096, on_evict=promoter)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `extractor` | `MemoryExtractor` | (required) | Extracts `MemoryEntry` objects from turns. |
| `store` | `MemoryEntryStore` | (required) | Persistence backend. |
| `consolidator` | `MemoryConsolidator \| None` | `None` | Optional deduplication. |

**Returns:** A callable with signature `(list[ConversationTurn]) -> None`.

!!! note
    Errors inside the eviction promoter are logged but never propagated to
    prevent crashing the memory pipeline.

---

## `MemoryContextEnricher`

Enriches queries by appending recent conversation context. Helps retrieval steps
find documents relevant to the ongoing conversation, not just the literal query.

```python
from anchor import MemoryContextEnricher

MemoryContextEnricher(
    max_items: int = 5,
    template: str = "{query}\n\nConversation context: {context}",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_items` | `int` | `5` | Maximum number of recent memory items to include. Must be positive. |
| `template` | `str` | `"{query}\n\nConversation context: {context}"` | Format string with `{query}` and `{context}` placeholders. |

**Usage:**

```python
enricher = MemoryContextEnricher(max_items=3)
pipeline = ContextPipeline(max_tokens=8192).with_query_enricher(enricher)
```

Satisfies the `MemoryQueryEnricher` protocol. When attached via `with_query_enricher()`,
it receives memory items and appends a summary to the query before retrieval steps execute.

---

## `PipelineCallback`

A runtime-checkable protocol for pipeline event callbacks. All methods are
optional -- implement only the ones you need.

```python
from anchor import PipelineCallback

class PipelineCallback(Protocol):
    def on_pipeline_start(self, query: QueryBundle) -> None: ...
    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None: ...
    def on_step_end(self, step_name: str, items: list[ContextItem], time_ms: float) -> None: ...
    def on_step_error(self, step_name: str, error: Exception) -> None: ...
    def on_pipeline_end(self, result: ContextResult) -> None: ...
```

| Method | Called When |
|---|---|
| `on_pipeline_start` | Pipeline execution begins. |
| `on_step_start` | A step is about to execute. |
| `on_step_end` | A step completed successfully. |
| `on_step_error` | A step raised an exception. |
| `on_pipeline_end` | Pipeline execution completed. |

---

## See Also

- [Pipeline Guide](../guides/pipeline.md) -- walkthrough with complete examples
- [Models API Reference](models.md) -- `ContextItem`, `ContextResult`, `TokenBudget`
- [Exceptions Reference](exceptions.md) -- error classes
