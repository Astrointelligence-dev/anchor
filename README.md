<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/artcgranja/anchor/main/docs/docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/artcgranja/anchor/main/docs/docs/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/artcgranja/anchor/main/docs/docs/assets/logo-light.svg" alt="anchor" width="280">
  </picture>
</p>

<p align="center">
  <strong>Context engineering toolkit for AI applications.</strong>
</p>

<p align="center">
  <a href="https://artcgranja.github.io/anchor/">Docs</a> ·
  <a href="https://artcgranja.github.io/anchor/getting-started/quickstart/">Quickstart</a> ·
  <a href="https://artcgranja.github.io/anchor/cookbook/">Cookbook</a> ·
  <a href="https://artcgranja.github.io/anchor/api/">API Reference</a> ·
  <a href="https://github.com/artcgranja/anchor/issues">Issues</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/astro-anchor/"><img src="https://img.shields.io/pypi/v/astro-anchor?color=3b82f6&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/astro-anchor/"><img src="https://img.shields.io/pypi/dm/astro-anchor?color=64748b&style=flat-square" alt="Downloads"></a>
  <a href="https://pypi.org/project/astro-anchor/"><img src="https://img.shields.io/pypi/pyversions/astro-anchor?color=6B8E6B&style=flat-square" alt="Python"></a>
  <a href="https://github.com/artcgranja/anchor/blob/main/LICENSE"><img src="https://img.shields.io/github/license/artcgranja/anchor?color=64748b&style=flat-square" alt="License"></a>
  <a href="https://github.com/artcgranja/anchor"><img src="https://img.shields.io/github/stars/artcgranja/anchor?style=flat-square&color=3b82f6" alt="Stars"></a>
</p>

---

Stop duct-taping RAG, memory, and tools together. **anchor** gives you a single, token-aware pipeline that assembles context for any LLM — with smart budgets, hybrid retrieval, and provider-agnostic formatting out of the box.

## Why anchor?

| Feature | LangChain | LlamaIndex | mem0 | **anchor** |
|---------|:---------:|:----------:|:----:|:----------:|
| Hybrid RAG (Dense + BM25 + RRF) | partial | yes | no | **yes** |
| Token-aware Memory | partial | no | yes | **yes** |
| Token Budget Management | no | no | no | **yes** |
| Provider-agnostic Formatting | no | no | no | **yes** |
| Protocol-based Plugins (PEP 544) | no | partial | no | **yes** |
| Zero-config Defaults | no | no | yes | **yes** |
| 2000+ tests, 94% coverage | — | — | — | **yes** |

## Quick Install

```bash
pip install astro-anchor
```

<details>
<summary>Optional extras</summary>

```bash
pip install astro-anchor[bm25]        # BM25 sparse retrieval
pip install astro-anchor[anthropic]   # Anthropic Claude support
pip install astro-anchor[cli]         # CLI tools (typer + rich)
pip install astro-anchor[flashrank]   # FlashRank reranking
pip install astro-anchor[otlp]        # OpenTelemetry tracing
pip install astro-anchor[all]         # Everything
```

</details>

## 30 Seconds to Your First Pipeline

```python
from anchor import ContextPipeline, MemoryManager, AnthropicFormatter

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(MemoryManager(conversation_tokens=4096))
    .with_formatter(AnthropicFormatter())
    .add_system_prompt("You are a helpful assistant.")
)

result = pipeline.build("What is context engineering?")
print(result.formatted_output)   # Ready for Claude API
print(result.diagnostics)        # Token usage, timing, overflow info
```

## Features

- **Hybrid RAG** — Dense embeddings + BM25 sparse retrieval with Reciprocal Rank Fusion
- **Smart Memory** — Token-aware sliding window with automatic eviction
- **Token Budgets** — Priority-ranked context assembly that never exceeds your window
- **LLM Providers** — Unified interface for Anthropic, OpenAI, Gemini, Grok, Ollama, OpenRouter, LiteLLM with fallback
- **MCP Bridge** — Bidirectional Model Context Protocol integration for consuming and exposing MCP tools
- **Provider Agnostic** — Format output for Anthropic, OpenAI, or plain text
- **Protocol-Based** — Plug in any vector store, tokenizer, or retriever via PEP 544 Protocols
- **Type-Safe** — Pydantic v2 models throughout, full `py.typed` support
- **Agent Framework** — Build tool-calling agents with `@tool` decorator and skills
- **Full Observability** — OpenTelemetry tracing, cost tracking, per-step diagnostics

<details>
<summary><strong>Hybrid Retrieval Example</strong></summary>

```python
import math
from anchor import (
    ContextPipeline, ContextItem, QueryBundle, SourceType,
    DenseRetriever, InMemoryContextStore, InMemoryVectorStore,
    retriever_step,
)

def my_embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw

dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

items = [
    ContextItem(content="Python is great for data science.", source=SourceType.RETRIEVAL),
    ContextItem(content="RAG combines retrieval with generation.", source=SourceType.RETRIEVAL),
]
dense.index(items)

pipeline = ContextPipeline(max_tokens=8192).add_step(retriever_step("search", dense, top_k=5))
query = QueryBundle(query_str="How does RAG work?", embedding=my_embed_fn("How does RAG work?"))
result = pipeline.build(query)
```

</details>

<details>
<summary><strong>Memory Management Example</strong></summary>

```python
from anchor import ContextPipeline, QueryBundle, MemoryManager

memory = MemoryManager(conversation_tokens=4096)
memory.add_user_message("Help me migrate from MySQL to Postgres")
memory.add_assistant_message("Sure! What MySQL version are you using?")
memory.add_user_message("MySQL 8.0, about 50GB")

pipeline = ContextPipeline(max_tokens=8192).with_memory(memory)
result = pipeline.build(QueryBundle(query_str="What are the risks?"))
# Memory is automatically included, oldest turns evicted if over budget
```

</details>

<details>
<summary><strong>Decorator API Example</strong></summary>

```python
from anchor import ContextPipeline, ContextItem, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def boost_recent(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    return [
        item.model_copy(update={"score": min(1.0, item.score * 1.5)})
        if item.metadata.get("recent") else item
        for item in items
    ]

result = pipeline.build("What is context engineering?")
```

</details>

<details>
<summary><strong>Provider Formatting</strong></summary>

```python
from anchor import AnthropicFormatter, OpenAIFormatter

# Anthropic: {"system": "...", "messages": [...]}
result = pipeline.with_formatter(AnthropicFormatter()).build(query)

# OpenAI: {"messages": [{"role": "system", ...}, ...]}
result = pipeline.with_formatter(OpenAIFormatter()).build(query)
```

</details>

## Architecture

```
ContextPipeline
  │
  ├── System Prompts (priority=10)
  ├── Memory Manager (priority=7)
  ├── Pipeline Steps
  │     ├── Retriever Steps (append items)
  │     ├── PostProcessor Steps (transform items)
  │     └── Filter Steps (filter items)
  │
  ▼
ContextWindow (token-aware, priority-ranked)
  │
  ▼
Formatter (Anthropic / OpenAI / Generic)
  │
  ▼
ContextResult (formatted output + diagnostics)
```

## Token Budgets

```python
from anchor import ContextPipeline, default_chat_budget

budget = default_chat_budget(max_tokens=8192)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

Three presets available: `default_chat_budget` (conversational), `default_rag_budget` (retrieval-heavy), `default_agent_budget` (balanced).

## Development

```bash
git clone https://github.com/artcgranja/anchor.git
cd anchor
uv sync
uv run pytest     # 2000+ tests
uv run ruff check src/ tests/
```

## Roadmap

- **v0.1.0** — Hybrid RAG, Memory, Pipeline, Formatters, Async, Decorator API, Agent Framework, [Full docs](https://artcgranja.github.io/anchor/)
- **v0.2.0** (next) — Agent loop 2026: event stream, usage limits, subagents, hooks + approval seam, structured output, memory tool, MCP bridge; embeddings layer, sqlite-vec + pgvector (HNSW), skills to the agentskills.io spec
- **v0.3.0** — Production deployment guides, additional vector backends by demand
- **v0.4.0** — GraphRAG, LangChain/LlamaIndex adapters, webhook-based event system
- **v1.0.0** — Stable public API, plugin ecosystem, managed cloud offering

## License

MIT — see [LICENSE](LICENSE) for details.
