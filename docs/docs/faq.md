---
title: FAQ
description: Frequently asked questions about anchor
---

# Frequently Asked Questions

---

## General

??? question "What is anchor?"
    anchor is a Python library for context engineering -- the discipline
    of building systems that provide the right information to language models at
    the right time. It provides pipelines for assembling context from multiple
    sources (retrieval, memory, system prompts) and formatting it for any LLM
    provider.

??? question "How is anchor different from LangChain or LlamaIndex?"
    anchor focuses specifically on the context assembly layer rather than
    being a full application framework. It is lightweight, protocol-based (no
    heavy inheritance hierarchies), and gives you fine-grained control over token
    budgets, priority-based overflow, and provider-specific formatting. You can
    use it alongside LangChain or LlamaIndex if you want.

??? question "Is anchor production-ready?"
    Yes. It is designed for production use with built-in error handling policies
    (`on_error="skip"`), observability callbacks (`TracingCallback`, `CostTracker`),
    comprehensive diagnostics, and thorough test coverage. See the
    [Production Patterns](cookbook/production-patterns.md) cookbook for deployment
    guidance.

??? question "Does anchor require an API key?"
    No. The core library (pipelines, memory, retrieval, formatting) works entirely
    locally with no API calls. An API key is only needed if you use the `Agent`
    class (which calls the Anthropic API) or if you use an embedding function that
    calls an external service.

---

## Installation

??? question "What Python version is required?"
    anchor requires Python 3.10 or later.

??? question "How do I install optional dependencies?"
    Use extras to install optional dependencies:

    ```bash
    # BM25 sparse retrieval
    pip install astro-anchor[bm25]

    # Anthropic Agent support
    pip install astro-anchor[anthropic]

    # PDF parsing
    pip install astro-anchor[pdf]

    # OpenTelemetry export
    pip install astro-anchor[otlp]

    # Everything
    pip install astro-anchor[all]
    ```

??? question "Can I install anchor in a Jupyter notebook?"
    Yes. Run `!pip install astro-anchor` in a cell. All examples in the
    documentation are designed to work in notebooks.

---

## Pipelines

??? question "What is the difference between `build()` and `abuild()`?"
    `build()` is synchronous and runs all pipeline steps sequentially.
    `abuild()` is asynchronous and can run async steps concurrently. Use
    `abuild()` when your pipeline includes async retrievers or when you want
    to parallelize I/O-bound operations.

    ```python
    # Synchronous
    result = pipeline.build("my query")

    # Asynchronous
    result = await pipeline.abuild("my query")
    ```

??? question "How do I handle errors in pipeline steps?"
    Use the `on_error` parameter when adding a step:

    ```python
    pipeline.add_step(
        retriever_step("search", my_retriever, top_k=10),
        on_error="skip",  # "raise" (default), "skip", or "empty"
    )
    ```

    - `"raise"`: stop the pipeline and propagate the exception
    - `"skip"`: log the error, continue with items from previous steps
    - `"empty"`: log the error, continue with an empty item list

??? question "Can I add custom steps to the pipeline?"
    Yes. Use the decorator API or pass a callable to `add_step()`:

    ```python
    @pipeline.step(name="my-custom-step")
    def boost_recent(items, query):
        """Boost items from the last 24 hours."""
        for item in items:
            if is_recent(item):
                item = item.model_copy(update={"score": item.score * 1.5})
        return items
    ```

??? question "What order do pipeline steps run in?"
    Steps run in the order they are added. Items flow from one step to the
    next. System prompts and memory items are injected after all steps run,
    during the final context window assembly.

??? question "Can I mix sync and async steps?"
    Yes. When using `abuild()`, sync steps are called directly and async
    steps are awaited. Both work seamlessly in the same pipeline.

---

## Memory

??? question "What eviction strategies are available?"
    anchor provides several eviction strategies for `SlidingWindowMemory`:

    - **FIFO** (default): removes the oldest turns first
    - **ImportanceEviction**: keeps high-importance turns longer based on a scoring function
    - **SummaryEviction**: summarizes evicted turns into a condensed fact before removing them

??? question "How do I persist memory across sessions?"
    Use a persistent `MemoryEntryStore` instead of `InMemoryEntryStore`:

    ```python
    from anchor import MemoryManager, JsonFileMemoryStore

    memory = MemoryManager(
        conversation_tokens=4096,
        persistent_store=JsonFileMemoryStore("memory.json"),
    )
    ```

    Conversation turns are in-memory only (by design, as they are ephemeral).
    Persistent facts stored via `add_fact()` are saved to the store.

??? question "How does fact deduplication work?"
    When you call `memory.add_fact()`, the library checks for semantic duplicates
    using text similarity. If a substantially similar fact already exists, the new
    fact is silently dropped. This prevents the fact store from filling up with
    redundant information.

??? question "What is graph memory?"
    The knowledge graph (`KnowledgeGraph`) stores entities and notes as nodes
    and typed, evidenced relations as edges, over memory and documents at the
    same time. It answers "what is connected to Project X, and why?" by
    walking relationships (`neighbors`, `path`, `explain`) under the same
    scope rules as retrieval. See the [Knowledge Graph guide](guides/knowledge-graph.md).

---

## Retrieval

??? question "What embedding functions are supported?"
    Any function with the signature `(str) -> list[float]` works as an embedding
    function. You can use:

    - OpenAI embeddings (`openai.embeddings.create`)
    - Sentence Transformers (`model.encode`)
    - Cohere embeddings
    - Any custom function that returns a float vector

    ```python
    from openai import OpenAI
    client = OpenAI()

    def embed_fn(text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    ```

??? question "How does BM25 sparse retrieval work?"
    BM25 (Best Matching 25) is a keyword-based ranking algorithm that scores
    documents by term frequency and inverse document frequency. It excels at
    exact keyword matching where dense embeddings might miss specific terms.

    ```python
    from anchor import SparseRetriever

    sparse = SparseRetriever()  # requires pip install astro-anchor[bm25]
    sparse.index(items)
    results = sparse.retrieve(query, top_k=10)
    ```

??? question "How do I choose hybrid retrieval weights?"
    Start with 70% dense / 30% sparse and tune using evaluation metrics. Dense
    retrieval is better for semantic similarity, while sparse retrieval is better
    for exact keyword matching. Use the [Evaluation Workflow](cookbook/evaluation-workflow.md)
    to find optimal weights for your dataset.

??? question "Can I use an external vector database?"
    Yes. Implement the `VectorStore` protocol with your database client:

    ```python
    class PineconeVectorStore:
        def __init__(self, index):
            self.index = index

        def add(self, ids, embeddings, metadata=None):
            self.index.upsert(vectors=zip(ids, embeddings, metadata or []))

        def query(self, embedding, top_k=10):
            return self.index.query(vector=embedding, top_k=top_k)
    ```

---

## Token Budgets

??? question "How do I choose the right token budget preset?"
    Match the preset to your model's context window, leaving headroom for the
    model's response:

    | Model | Context Window | Recommended Budget |
    |-------|---------------|--------------------|
    | Claude Haiku | 200K | `MEDIUM` (16K) to `LARGE` (32K) |
    | Claude Sonnet | 200K | `LARGE` (32K) to `XL` (64K) |
    | GPT-4o | 128K | `MEDIUM` (16K) to `LARGE` (32K) |
    | GPT-4o-mini | 128K | `SMALL` (4K) to `MEDIUM` (16K) |

    Start smaller and increase only if retrieval quality suffers.

??? question "What happens when the token budget overflows?"
    By default, the lowest-priority items are dropped until the context fits
    within the budget. Priority order (highest to lowest):

    1. System prompts (priority 10)
    2. Persistent facts (priority 8)
    3. Conversation turns (priority 7)
    4. Retrieved documents (priority 5)

    You can change the overflow policy to `"truncate_oldest"` or `"error"`.

??? question "Can I set a custom token budget for each source?"
    Yes. Use `TokenBudgetConfig` for fine-grained allocation:

    ```python
    from anchor import TokenBudgetConfig

    budget = TokenBudgetConfig(
        total=16384,
        system_prompt=1024,
        memory_conversation=4096,
        memory_facts=512,
        retrieval=8192,
    )
    ```

---

## Formatting

??? question "How do I switch between LLM providers?"
    Swap the formatter on the pipeline:

    ```python
    from anchor import (
        AnthropicFormatter,
        OpenAIFormatter,
        GenericTextFormatter,
    )

    # For Anthropic Claude
    pipeline.with_formatter(AnthropicFormatter())

    # For OpenAI GPT
    pipeline.with_formatter(OpenAIFormatter())

    # For any provider (plain text)
    pipeline.with_formatter(GenericTextFormatter())
    ```

??? question "Can I create a custom formatter?"
    Yes. Implement the `Formatter` protocol:

    ```python
    from anchor import Formatter, ContextWindow

    class MyFormatter:
        def format(self, window: ContextWindow) -> dict:
            # Transform the context window into your provider's format
            return {
                "system": window.system_text(),
                "context": window.retrieval_text(),
                "history": window.conversation_items(),
            }
    ```

??? question "What does the formatted output look like for Anthropic?"
    `AnthropicFormatter` produces a dict with `system` and `messages` keys that
    can be passed directly to `anthropic.messages.create()`:

    ```python
    result = pipeline.build(query)
    formatted = result.formatted_output
    # formatted = {
    #     "system": [{"type": "text", "text": "You are a helpful assistant."}],
    #     "messages": [
    #         {"role": "user", "content": "Hello"},
    #         {"role": "assistant", "content": "Hi there!"},
    #     ],
    # }
    ```

---

## Troubleshooting

??? question "I'm getting `ModuleNotFoundError: No module named 'rank_bm25'`"
    Install the BM25 extra:

    ```bash
    pip install astro-anchor[bm25]
    ```

??? question "I'm getting `TokenBudgetExceeded` errors"
    This happens when your overflow policy is set to `"error"` and the assembled
    context exceeds the token budget. Solutions:

    1. Increase `max_tokens`
    2. Reduce `top_k` on your retriever steps
    3. Switch to `overflow_policy="truncate_lowest_priority"` (the default)

??? question "My pipeline is slow -- how do I debug it?"
    Check the step-by-step timing in diagnostics:

    ```python
    result = pipeline.build(query)
    for step in result.diagnostics.get("steps", []):
        print(f"{step['name']}: {step['time_ms']:.1f} ms")
    ```

    Common bottlenecks:

    - **Embedding computation**: use a faster model or cache embeddings
    - **Reranking**: reduce the number of candidates passed to the reranker
    - **External API calls**: use async steps with `abuild()`

??? question "Retrieved results seem irrelevant -- how do I improve quality?"
    Try these steps in order:

    1. **Check your embedding function**: ensure it is producing meaningful vectors
    2. **Adjust chunk sizes**: smaller chunks (128-256 tokens) often improve precision
    3. **Add a reranker**: `CrossEncoderReranker` significantly improves relevance
    4. **Try hybrid retrieval**: combine dense and sparse search for better coverage
    5. **Use evaluation metrics**: run the [Evaluation Workflow](cookbook/evaluation-workflow.md)
       to measure and compare configurations

??? question "How do I enable debug logging?"
    Set the log level for the `anchor` logger:

    ```python
    import logging
    logging.getLogger("anchor").setLevel(logging.DEBUG)
    ```

    This logs step-by-step pipeline execution, token counting, and eviction events.
