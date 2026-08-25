# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Embeddings layer: `EmbeddingProvider` protocol (query/document asymmetry, native batching, async, `dimensions` for Matryoshka truncation) + providers behind extras — `OpenAIEmbeddingProvider` (`[openai]`), `VoyageEmbeddingProvider` (`[voyage]`), `SentenceTransformerEmbeddingProvider` (`[local-embeddings]`, BGE-M3 default for multilingual/PT) — and `CallableEmbeddingProvider` unifying the legacy `embed_fn` shapes
- Metadata filtering: `where: dict` parameter on `VectorStore.search` / `AsyncVectorStore.search`, pushed down in every backend (InMemory dict match, SQLite `json_extract` in SQL, Postgres JSONB containment `@>` with a GIN index) and exposed on `DenseRetriever.retrieve` / `AsyncDenseRetriever.aretrieve` — unblocks user/tenant/type scoping
- `DenseRetriever.index` embeds documents in one batch call (was one embedding call per item)
- `AsyncDenseRetriever` store-backed mode (`vector_store=` + `context_store=`) over the `AsyncVectorStore` protocol — pgvector (`PostgresVectorStore`) and `AsyncSqliteVectorStore` are now reachable from a built-in retriever
- Postgres: `ensure_tables` now creates the pgvector **HNSW** index (works on empty tables, unlike the previously suggested IVFFlat) and a JSONB GIN index; `embedding_dim` is a required parameter
- pgvector integration test suite gated by `ANCHOR_TEST_POSTGRES_DSN` (previously zero Postgres tests)
- Skills: agentskills.io spec compliance — real YAML frontmatter (multi-line descriptions, quoting), spec fields `license`, `compatibility`, `metadata`, `allowed-tools`, and name==directory validation; `pyyaml` added as a core dependency
- Skills: level-3 progressive disclosure — `read_skill_file` meta-tool loads `references/` lazily (path-traversal guarded, 50KB cap); `run_skill_script` executes `scripts/` on demand (opt-in via `Agent(allow_skill_scripts=True)`, 60s timeout); activation responses list bundled files
- Skills: `always`-skill `instructions` are now injected into the system prompt (previously dead text that never reached the model)
- Skills: minimal eval harness (`run_skill_eval`, `SkillEvalCase/Result/Report`) for with/without-skill A/B baselines with a callback judge
- `SkillRegistry.all_skills()`, `always_skills()`, `always_instructions()`, and a `max_chars` cap on `skill_discovery_prompt()` (~1% of context budget in the agent)
- `metadata["raw_score"]` on all retriever and reranker results, preserving the unclamped/unnormalized score (raw cosine, raw BM25, raw reranker logits) so quality thresholds are possible
- `min_score` parameter on `DenseRetriever` and `SparseRetriever` to filter results below a raw-score threshold
- SOTA 2026 research docs (`docs/research/2026-08-25-*`) and phased upgrade plan (`docs/plans/2026-08-25-sota-upgrade-plan.md`)
- Multi-provider LLM interface (`anchor.llm`) with support for Anthropic, OpenAI, Gemini, Grok, Ollama, OpenRouter, and LiteLLM
- `LLMProvider` protocol and `BaseLLMProvider` ABC with built-in retry and timeout logic
- `create_provider()` factory with `"provider/model"` string format and automatic lazy loading
- `FallbackProvider` for automatic provider failover (fallback only before first stream chunk)
- Provider error hierarchy: `ProviderError`, `RateLimitError`, `ServerError`, `TimeoutError`, `AuthenticationError`, `ModelNotFoundError`, `ContentFilterError`
- Thread-safe provider registry with `threading.Lock`
- Shared `_openai_compat` module for OpenAI/LiteLLM code deduplication
- Anthropic streaming usage tracking (`input_tokens` + `output_tokens`)
- LLM Providers API reference and guide documentation
- Unit tests for `_math.py` (cosine_similarity and clamp functions)
- `MemoryRetrieverAdapter` tests verifying Retriever protocol compliance
- `PipelineExecutionError` wrapping test with diagnostics verification
- Golden path integration test mirroring README usage pattern
- Example: `examples/hybrid_rag.py` -- hybrid RAG pipeline with dense retrieval
- Example: `examples/custom_retriever.py` -- custom Retriever protocol implementation
- Example: `examples/budget_management.py` -- token budget management and overflow handling
- README sections for Priority System (1--10 scale) and Token Budgets

### Changed
- Skills: `Skill.activation` default flipped from `"always"` to `"on_demand"` (matches the loader default and progressive-disclosure semantics); canonical frontmatter location for activation is now `metadata: {activation: ...}` (top-level key still accepted)
- Skills: the discovery listing is static (no per-round `[active]` mutation) and joined to the system prompt with newlines, computed once per turn — prompt-cache friendly; the agent's `AnthropicFormatter` now has caching enabled
- Skills: tool-name collisions raise at registration time (`SkillRegistry.register` and `Agent.with_skill`), never mid-conversation
- Reranker `rerank()`/`arerank()` signatures: `top_k: int = 10` → `top_k: int | None = None`; an explicit `top_k` now always overrides the constructor value, `None` falls back to it (applies to all rerankers, the `Reranker`/`AsyncReranker` protocols, and `reranker_step`/`async_reranker_step`)
- `AsyncCohereReranker` callback now returns `(index, score)` tuples, matching the sync `CohereReranker` shape, and applies scores to results
- `HybridRetriever` and `AsyncHybridRetriever` now delegate fusion to the canonical `rrf_fuse` (single RRF implementation); `rrf_fuse` gained a `retrieval_method` label parameter
- `AsyncHybridRetriever` raises `RetrieverError` when all sub-retrievers fail, matching the sync behavior (previously returned `[]`)
- All cosine-similarity call sites (`SemanticChunker`, late interaction, cross-modal, `EmbeddingClassifier`) now use the strict `anchor._math.cosine_similarity` — dimension mismatches raise instead of silently truncating
- Repo workflow: `docs/superpowers/` retired; specs/plans live in `docs/plans/`, research in `docs/research/`
- `Agent` constructor: `client` parameter replaced with `llm: LLMProvider` and `fallbacks: list[str]`
- `Role` and `StopReason` enums changed from `(str, Enum)` to `StrEnum` for correct string formatting
- `AgentTool`: removed `to_anthropic_schema()`, `to_openai_schema()`, `to_generic_schema()`; replaced with unified `to_tool_schema() -> ToolSchema`

### Removed
- `ScoreReranker` (dead duplicate of `CrossEncoderReranker` implementing the wrong protocol)

### Fixed
- **Reranker `top_k` sentinel bug**: passing `top_k=10` explicitly was indistinguishable from the default and silently discarded in all five rerankers; the pipeline `reranker_step` default hit this on every run
- `RecursiveCharacterChunker` applied overlap at every recursion level, duplicating overlap text in nested sub-chunks; overlap is now applied exactly once over the final chunk list
- `SqliteVectorStore` unpacked stored embeddings using the *query* vector's dimension — a query/stored dimension mismatch silently mis-unpacked or crashed with `struct.error`; the dimension now derives from the stored blob and mismatches raise a clear `ValueError`
- 34 pre-existing test failures caused by missing optional dependencies (tiktoken, rank-bm25)
- `FallbackProvider.astream` mid-stream fallback semantics (yields now outside try/except)
- `test_consolidator.py`: eliminated shared mutable state (`_orthogonal_index` dict) by converting to factory function pattern (`make_orthogonal_embed()`)
- `test_graph_memory.py`: updated `link_memory` unknown entity test to expect `KeyError` instead of `ValueError`
- README: fixed retrieval example with runnable `embed_fn` and `ContextItem` creation
- README: updated test count from 961 to 1088

## [0.1.0] - 2026-02-20

### Added
- Core context pipeline with sync/async support (`ContextPipeline`)
- Token-aware sliding window memory (`SlidingWindowMemory`)
- Summary buffer memory with progressive compaction (`SummaryBufferMemory`)
- Memory manager facade unifying conversation and persistent memory (`MemoryManager`)
- Hybrid RAG retrieval: dense, sparse (BM25), and hybrid (RRF) retrievers
- Multi-signal memory retrieval with recency/relevance/importance scoring (`ScoredMemoryRetriever`)
- Provider-agnostic formatting: Anthropic, OpenAI, and generic text formatters
- Anthropic multi-block system formatting with prompt caching support
- Protocol-based extensibility (PEP 544) for all extension points
- Token budget management with per-source allocations and overflow tracking
- Pluggable eviction policies: FIFO, importance-based, and paired (user+assistant)
- Memory decay: Ebbinghaus forgetting curve and linear decay
- Recency scoring: exponential and linear strategies
- Memory consolidation with content-hash dedup and cosine-similarity merging
- Simple graph memory with BFS traversal for entity-relationship tracking
- Memory garbage collection with two-phase expired+decayed pruning
- Memory callback protocol for lifecycle observability
- Pipeline query enrichment with memory context
- Auto-promotion of evicted turns to long-term memory
- In-memory reference implementations for all storage protocols
- JSON file-backed persistent memory store
- CLI with index and query commands (via typer+rich)
- 961 tests with 94% coverage
