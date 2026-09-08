---
title: anchor
description: Context engineering toolkit for AI applications
hide:
  - toc
---

<style>
.md-content .md-typeset h1 { display: none; }
</style>

<div class="hero" markdown>

<img src="assets/logo-light.svg" alt="anchor" width="220" class="only-light">
<img src="assets/logo-dark.svg" alt="anchor" width="220" class="only-dark">

# anchor

### Context is the product. The LLM is just the consumer.

The Python toolkit for context engineering -- assemble RAG, memory, tools,
and system prompts into a single, token-aware pipeline.

[Get Started :material-arrow-right:](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/Astrointelligence-dev/anchor){ .md-button }

[![PyPI](https://img.shields.io/pypi/v/astro-anchor?color=3b82f6)](https://pypi.org/project/astro-anchor/)
[![Downloads](https://img.shields.io/pypi/dm/astro-anchor?color=64748b)](https://pypi.org/project/astro-anchor/)
[![Python](https://img.shields.io/pypi/pyversions/astro-anchor?color=6B8E6B)](https://pypi.org/project/astro-anchor/)
[![License: MIT](https://img.shields.io/badge/license-MIT-64748b)](https://github.com/Astrointelligence-dev/anchor/blob/main/LICENSE)

</div>

---

## Why anchor?

Most AI frameworks focus on the LLM call. But the real challenge is assembling the
right **context** -- the system prompt, conversation memory, retrieved documents,
and tool outputs that the model actually sees.

anchor gives you a single, composable pipeline that manages all of it
within a strict token budget. No duct-taping RAG, memory, and tools together.
Build intelligent context pipelines in minutes.

---

## Features

<div class="grid cards" markdown>

-   :material-magnify-plus-outline:{ .lg .middle } **Hybrid RAG**

    ---

    Dense embeddings + BM25 sparse retrieval with Reciprocal Rank Fusion.
    Combine multiple retrieval strategies in a single pipeline for higher
    recall and precision.

    [:octicons-arrow-right-24: Retrieval guide](guides/retrieval.md)

-   :material-brain:{ .lg .middle } **Smart Memory**

    ---

    Token-aware sliding window with automatic eviction. Oldest turns are
    evicted when the conversation exceeds its budget -- recent context is
    never lost.

    [:octicons-arrow-right-24: Memory guide](guides/memory.md)

-   :material-scale-balance:{ .lg .middle } **Token Budgets**

    ---

    Priority-ranked assembly fills from highest-priority items down.
    Per-source allocations let you reserve tokens for system prompts,
    memory, retrieval, and responses independently.

    [:octicons-arrow-right-24: Token budgets](concepts/token-budgets.md)

-   :material-swap-horizontal:{ .lg .middle } **Provider Agnostic**

    ---

    Anthropic, OpenAI, or plain text. Format the assembled context for any
    LLM provider with a single method call. Swap providers without changing
    your pipeline.

    [:octicons-arrow-right-24: Formatters guide](guides/formatters.md)

-   :material-puzzle-outline:{ .lg .middle } **Protocol-Based**

    ---

    Every extension point is defined as a PEP 544 structural protocol.
    Bring your own retriever, tokenizer, reranker, or memory store -- no
    base classes required.

    [:octicons-arrow-right-24: Protocols](concepts/protocols.md)

-   :material-shield-check-outline:{ .lg .middle } **Type-Safe**

    ---

    All models are frozen Pydantic v2 dataclasses with full `py.typed`
    support. Catch integration errors at type-check time, not at runtime.

    [:octicons-arrow-right-24: API reference](api/models.md)

-   :material-robot-outline:{ .lg .middle } **Agent Framework**

    ---

    Built-in tool registration, skills, and memory+RAG skills that give
    your agent long-term recall. Compose agents from the same pipeline
    primitives.

    [:octicons-arrow-right-24: Agent guide](guides/agent.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Full Observability**

    ---

    Tracing, metrics, cost tracking, and native OTLP export. Know exactly
    what your pipeline is doing, how long it takes, and what it costs.

    [:octicons-arrow-right-24: Observability guide](guides/observability.md)

</div>

---

## Installation

=== "pip"

    ```bash
    pip install astro-anchor
    ```

=== "uv"

    ```bash
    uv add astro-anchor
    ```

=== "Extras"

    ```bash
    pip install astro-anchor[bm25]   # BM25 sparse retrieval (rank-bm25)
    pip install astro-anchor[cli]    # CLI tools (typer + rich)
    pip install astro-anchor[all]    # Everything
    ```

---

## 30-Second Quickstart

Build your first context pipeline:

```python
from anchor import ContextPipeline, MemoryManager, AnthropicFormatter

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(MemoryManager(conversation_tokens=4096))
    .with_formatter(AnthropicFormatter())
    .add_system_prompt("You are a helpful assistant.")
)

result = pipeline.build("What is context engineering?")
print(result.formatted_output)   # Ready for the Anthropic API
print(result.diagnostics)        # Token usage, timing, overflow info
```

!!! tip "Plain strings just work"
    `build()` accepts either a plain `str` or a `QueryBundle` object.
    Plain strings are automatically wrapped in a `QueryBundle` for you.

---

## How It Works

```mermaid
graph LR
    A[User Query] --> B(ContextPipeline)
    B --> C{Pipeline Steps}
    C --> D[Retriever Steps]
    C --> E[PostProcessor Steps]
    C --> F[Filter Steps]

    G[System Prompts<br/>priority=10] --> H(ContextWindow)
    I[Memory Manager<br/>priority=7] --> H
    D --> H
    E --> H
    F --> H

    H -->|Token-aware<br/>priority-ranked| J(Formatter)
    J -->|Anthropic / OpenAI<br/>/ Generic| K[ContextResult]

    K --> L[formatted_output]
    K --> M[diagnostics]
    K --> N[overflow_items]

    style B fill:#3b82f6,color:#fff,stroke:none
    style H fill:#3b82f6,color:#fff,stroke:none
    style J fill:#6B8E6B,color:#fff,stroke:none
    style K fill:#3b82f6,color:#fff,stroke:none
```

Every `ContextItem` carries a **priority** (1--10). When the total exceeds
`max_tokens`, the pipeline fills from highest priority down. Items that do not
fit are tracked in `result.overflow_items`.

---

## Comparison

| Feature | LangChain | LlamaIndex | mem0 | **anchor** |
|:--------|:---------:|:----------:|:----:|:-----------------:|
| Hybrid RAG (Dense + BM25 + RRF) | :material-circle-half-full: partial | :material-check: yes | :material-close: no | :material-check-bold: **yes** |
| Token-aware Memory | :material-circle-half-full: partial | :material-close: no | :material-check: yes | :material-check-bold: **yes** |
| Token Budget Management | :material-close: no | :material-close: no | :material-close: no | :material-check-bold: **yes** |
| Provider-agnostic Formatting | :material-close: no | :material-close: no | :material-close: no | :material-check-bold: **yes** |
| Protocol-based Plugins (PEP 544) | :material-close: no | :material-circle-half-full: partial | :material-close: no | :material-check-bold: **yes** |
| Zero-config Defaults | :material-close: no | :material-close: no | :material-check: yes | :material-check-bold: **yes** |
| Built-in Agent Framework | :material-check: yes | :material-check: yes | :material-close: no | :material-check-bold: **yes** |
| Native Observability (OTLP) | :material-circle-half-full: partial | :material-circle-half-full: partial | :material-close: no | :material-check-bold: **yes** |

---

## Token Budgets

For fine-grained control over how tokens are allocated across sources,
use the preset budget factories:

```python
from anchor import ContextPipeline, default_chat_budget

budget = default_chat_budget(max_tokens=8192)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

Three presets are available:

| Preset | Best for | Conversation | Retrieval | Response |
|:-------|:---------|:------------:|:---------:|:--------:|
| `default_chat_budget` | Conversational apps | 60% | 15% | 15% |
| `default_rag_budget` | RAG-heavy apps | 25% | 40% | 15% |
| `default_agent_budget` | Agentic apps | 30% | 25% | 15% |

!!! note
    Each budget automatically reserves 15% of tokens for the LLM response.
    Per-source overflow strategies (`"truncate"` or `"drop"`) control what
    happens when a source exceeds its cap.

---

## Decorator API

Register pipeline steps with decorators instead of factory functions:

```python
from anchor import ContextPipeline, ContextItem, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def boost_recent(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Boost the score of recent items."""
    return [
        item.model_copy(update={"score": min(1.0, item.score * 1.5)})
        if item.metadata.get("recent")
        else item
        for item in items
    ]

result = pipeline.build("What is context engineering?")
```

!!! tip
    Use `@pipeline.async_step` for async functions and call `abuild()`
    instead of `build()`.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Getting Started**

    ---

    Installation, first pipeline, and all the basics.

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

-   :material-lightbulb-outline:{ .lg .middle } **Core Concepts**

    ---

    Context engineering, architecture, protocols, and token budgets.

    [:octicons-arrow-right-24: Concepts](concepts/index.md)

-   :material-book-open-variant:{ .lg .middle } **Guides**

    ---

    Pipeline, retrieval, memory, agents, observability, and more.

    [:octicons-arrow-right-24: Guides](guides/index.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Full API documentation for every module.

    [:octicons-arrow-right-24: API docs](api/index.md)

</div>
