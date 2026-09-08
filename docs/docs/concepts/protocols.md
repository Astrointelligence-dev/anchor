# Protocol-Based Architecture

anchor uses Python Protocols (PEP 544) to define all extension
points. This page explains what protocols are, why anchor chose them
over class inheritance, and how to implement your own.

## What Are Protocols?

A Protocol is a way to declare an interface in Python using structural
subtyping -- also known as "static duck typing." A class satisfies a
protocol if it has the right methods with the right signatures. No base
class or registration is needed.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query, top_k=10):
        ...
```

Any object with a `retrieve(query, top_k)` method satisfies the `Retriever`
protocol -- even if it has never seen the protocol definition.

!!! note "PEP 544"
    Protocols were introduced in Python 3.8 via PEP 544. They are part of
    the `typing` module and are fully supported by mypy, pyright, and other
    type checkers.

## Why Protocols Over Inheritance?

Traditional frameworks use abstract base classes (ABCs) to define
interfaces. This creates problems:

| Concern | Inheritance (ABC) | Protocol |
|---------|:-----------------:|:--------:|
| Must import base class | yes | **no** |
| Must call `super().__init__()` | often | **never** |
| Works with third-party classes | no | **yes** |
| Runtime `isinstance()` checks | yes | **yes** (`@runtime_checkable`) |
| IDE autocompletion | yes | **yes** |
| Type checker validation | yes | **yes** |

With protocols, you can wrap any existing object -- a Pinecone client, a
custom database class, a test stub -- without modifying its inheritance
chain. If it has the right methods, it works.

## Protocol Families

anchor defines protocols across seven families. Every protocol is
`@runtime_checkable`, so you can use `isinstance()` checks at runtime.

### Retrieval Protocols

For fetching and ranking context items.

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `Retriever` | `retrieve(query, top_k)` | Synchronous retrieval |
| `AsyncRetriever` | `aretrieve(query, top_k)` | Async retrieval |
| `Reranker` | `rerank(query, items, top_k)` | Synchronous reranking |
| `AsyncReranker` | `arerank(query, items, top_k)` | Async reranking |
| `PostProcessor` | `process(items, query)` | Post-retrieval transformation |
| `AsyncPostProcessor` | `aprocess(items, query)` | Async post-processing |

### Memory Protocols

For conversation history, persistent facts, and memory lifecycle.

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `MemoryProvider` | `get_context_items(priority)` | Provides items to the pipeline |
| `ConversationMemory` | `turns`, `to_context_items()` | Conversation turn management |
| `CompactionStrategy` | `compact(turns)` | Summarize evicted turns |
| `AsyncCompactionStrategy` | `compact(turns)` | Async summarization |
| `MemoryExtractor` | `extract(turns)` | Extract structured facts from turns |
| `AsyncMemoryExtractor` | `extract(turns)` | Async fact extraction |
| `MemoryConsolidator` | `consolidate(new, existing)` | Merge/deduplicate memories |
| `EvictionPolicy` | `select_for_eviction(turns, tokens_to_free)` | Choose turns to evict |
| `MemoryDecay` | `compute_retention(entry)` | Score memory retention (0.0--1.0) |
| `MemoryQueryEnricher` | `enrich(query, memory_items)` | Augment query with memory context |
| `RecencyScorer` | `score(index, total)` | Compute recency weight |

### Storage Protocols

For persisting context items, vectors, documents, and memory entries.

| Protocol | Key Methods | Description |
|----------|------------|-------------|
| `ContextStore` | `add`, `get`, `get_all`, `delete`, `clear` | Context item persistence |
| `VectorStore` | `add_embedding`, `search`, `delete` | Vector similarity search |
| `DocumentStore` | `add_document`, `get_document`, `list_documents`, `delete_document` | Raw document storage |
| `MemoryEntryStore` | `add`, `search`, `list_all`, `delete`, `clear` | Memory entry persistence |
| `GarbageCollectableStore` | `list_all_unfiltered`, `delete` | Extends MemoryEntryStore for GC |

### Observability Protocols

For tracing, metrics, and monitoring.

| Protocol | Key Methods | Description |
|----------|------------|-------------|
| `SpanExporter` | `export(spans)` | Export trace spans to backends |
| `MetricsCollector` | `record(metric)`, `flush()` | Collect and flush metric points |

### Query Protocols

For query transformation, classification, and routing.

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `QueryTransformer` | `transform(query)` | Expand or rewrite queries |
| `AsyncQueryTransformer` | `atransform(query)` | Async query transformation |
| `QueryClassifier` | `classify(query)` | Assign a label to a query |
| `QueryRouter` | `route(query)` | Route to a named retriever |

### Ingestion Protocols

For document parsing and chunking.

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `Chunker` | `chunk(text, metadata)` | Split text into chunks |
| `DocumentParser` | `parse(source)` | Extract text + metadata from files |

### Evaluation Protocols

For assessing retrieval and generation quality.

| Protocol | Key Method | Description |
|----------|-----------|-------------|
| `RetrievalEvaluator` | `evaluate(retrieved, relevant, k)` | Precision, recall, MRR, NDCG |
| `RAGEvaluator` | `evaluate(query, answer, contexts, ground_truth)` | Faithfulness, relevancy |
| `HumanEvaluator` | `add_judgment`, `compute_agreement` | Human-in-the-loop evaluation |

### Infrastructure Protocols

For caching and tokenization.

| Protocol | Key Methods | Description |
|----------|------------|-------------|
| `Tokenizer` | `count_tokens(text)`, `truncate_to_tokens(text, max_tokens)` | Token counting |
| `CacheBackend` | `get`, `set`, `invalidate`, `clear` | Pipeline step caching |

## Implementing a Protocol

To implement a protocol, just write a class with the matching methods.
Here is a concrete example implementing the `Retriever` protocol:

```python
from anchor import ContextItem, SourceType
from anchor.models.query import QueryBundle


class KeywordRetriever:
    """A simple keyword-based retriever -- satisfies the Retriever protocol."""

    def __init__(self, documents: list[str]):
        self._docs = documents

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        query_words = set(query.query_str.lower().split())
        scored = []
        for doc in self._docs:
            overlap = len(query_words & set(doc.lower().split()))
            if overlap > 0:
                scored.append((doc, overlap))
        scored.sort(key=lambda x: -x[1])
        return [
            ContextItem(
                content=doc,
                source=SourceType.RETRIEVAL,
                score=min(1.0, count / len(query_words)) if query_words else 0.0,
            )
            for doc, count in scored[:top_k]
        ]
```

This class works with `retriever_step()` and `ContextPipeline` without
importing any base class:

```python
from anchor import ContextPipeline
from anchor.pipeline.step import retriever_step

retriever = KeywordRetriever(["Python is great", "RAG combines retrieval with generation"])
pipeline = ContextPipeline(max_tokens=8192).add_step(retriever_step("keyword", retriever))
result = pipeline.build("What is RAG?")
```

!!! tip "Runtime checking"
    All protocols are `@runtime_checkable`, so you can verify at runtime:
    ```python
    from anchor.protocols import Retriever
    assert isinstance(retriever, Retriever)  # True
    ```

## Async Protocol Pairs

Many protocols come in sync/async pairs. The async variant uses a different
method name (prefixed with `a`) to avoid ambiguity:

| Sync | Async | Sync Method | Async Method |
|------|-------|-------------|--------------|
| `Retriever` | `AsyncRetriever` | `retrieve()` | `aretrieve()` |
| `Reranker` | `AsyncReranker` | `rerank()` | `arerank()` |
| `PostProcessor` | `AsyncPostProcessor` | `process()` | `aprocess()` |
| `QueryTransformer` | `AsyncQueryTransformer` | `transform()` | `atransform()` |

Use the sync variant with `pipeline.build()` and the async variant with
`pipeline.abuild()`.

!!! warning "Sync steps in async pipelines"
    `abuild()` can run both sync and async steps -- sync functions are
    called directly. But `build()` cannot run async steps and will raise
    `TypeError` if it encounters one.

## See Also

- [Architecture](architecture.md) -- how the pipeline uses protocols
- [Token Budgets](token-budgets.md) -- per-source allocation
- [Retrieval Guide](../guides/retrieval.md) -- practical retriever examples
