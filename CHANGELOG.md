# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- **`Retriever`/`AsyncRetriever` protocols take `scope`**: every built-in retriever honors `retrieve(..., scope=RetrievalScope)` — composites (`HybridRetriever`, `AsyncHybridRetriever`, `RoutedRetriever`) forward it, `SparseRetriever` masks out-of-scope documents *before* the top-k cut, `SharedSpaceRetriever` filters, `MemoryRetrieverAdapter` applies it to the root namespace. A custom retriever without the keyword keeps working while no scope is active and fails loudly under one — never a silent widening. `retriever_step`/`async_retriever_step` intersect their static `scope` with the scope published by the running agent turn
- **Vault names are validated at every mount** (`validate_vault`: non-empty, no `/`, no `:`) — store constructors and `anchor index/query --vault` now raise/exit on a bad name instead of writing rows that cannot be read back
- **`SqliteVecVectorStore(db, dimensions=None)`**: the dimension is read from an existing index and a declared value that disagrees with the file raises `ValueError` before any DDL (it used to rebuild the index at the wrong dimension and strand every embedding)
- **Redis context keys moved to `ctxv:{vault}:{id}` / `ctxv-ids:{vault}`**: the `__default__` store migrates pre-vault `ctx:` keys in place on first use (`RENAMENX`, idempotent); the read-only legacy fallback is gone
- **`PostgresConnectionManager` sets `hnsw.iterative_scan=relaxed_order` as a pool session default** (pgvector ≥ 0.8; a placeholder on older versions) — every vault-bound search is a filtered HNSW search, so recall no longer stops at `ef_search` candidates
- **Tool results are capped by default**: results (and error texts) longer than ~10k tokens are truncated head+tail with a marker before entering the messages — the loop finally calls the truncation the token-budget story always implied. Configure agent-wide with `Agent(tool_result_max_tokens=...)` (`None` disables), per tool with `AgentTool(max_result_tokens=...)`; subagent results are exempt (bounded by the child's own loop; validated JSON arrives whole)
- **Async tool execution schedules by side effect**: consecutive `AgentTool(read_only=True)` calls run concurrently (cap 10); every undeclared tool is treated as a write and runs alone — two writes can never overlap. This serializes tools that ran in parallel before, **including all MCP tools** (`readOnlyHint` mapping is a tracked follow-up) — declare your side-effect-free tools `read_only=True` to regain fan-out. Subagent dispatch (`task`/`as_tool`) stays parallel, now capped at 10 concurrent
- **New `stopped_by="stuck"`**: repeating an identical (tool, args, result) call across consecutive rounds first gets the model a corrective nudge (3 identical errors / 4 identical results); repeating again ends the turn with a graceful wrap-up round, same contract as `usage_limit` — no exception. A stuck child returns a marked partial result to its parent
- **`UsageLimits` is now run-wide, not per-turn**: the agent that starts a turn with limits creates a shared pool and every subagent spawned during it (`task`/`as_tool`) debits the same budget — an orchestrator's `total_tokens_limit` now sees its children's spend. A child cut mid-run wraps up gracefully and returns a marked partial result; a child may carry its own narrower per-turn limits (`SubagentDefinition(usage_limits=...)`). `UsageLimitReached.used`/`limit` widen to `int | float` and the event gains `kind="cost"` and `scope` (`"run"` pool vs `"turn"` own-limit)
- **New `stopped_by="output_missing"`**: a structured-output turn whose final round still produced no valid `final_result` ends gracefully with this verdict instead of raising from inside the loop — the loop always yields a terminal `TurnFinished`, and `run()`/`arun()` are the single place that raises (`chat`/`stream` consumers see the verdict, not an exception)
- `Agent` constructor: `client` parameter replaced with `llm: LLMProvider` and `fallbacks: list[str]`
- `AgentTool`: removed `to_anthropic_schema()`, `to_openai_schema()`, `to_generic_schema()`; replaced with unified `to_tool_schema() -> ToolSchema`

### Added
- **Vaults, namespaces and `RetrievalScope` (roadmap #3)**: two scoping levels over everything anchor retrieves. A **vault** is a hard isolation mount bound at store construction (`SqliteVecVectorStore(db, vault="dnd")` — vec0 `PARTITION KEY`; indexed column + composite `(vault, id)` keys on SQLite/Postgres; key path on Redis): callers never write the boundary into a query, and the same id coexists across vaults. A **namespace** is a hierarchical materialized path on every `ContextItem` (`/campanha-1/sessoes`), pre-filtered in the index via boundary-aware prefix ranges (never `LIKE`; on vec0 through a pushed-down rowid subquery — a bare `OR` in the KNN silently post-filters). `RetrievalScope(include=..., exclude=...)` is navigation only — exclude always wins, `intersect()` only narrows; `Agent.with_scope()` publishes it per tool call (the budget-pool window pattern) so subagents inherit parent∩own and **can never widen**; `SubagentDefinition.scope` narrows declaratively. `rag_tools({"juridico": r1, "notas": r2})` mounts named vaults — the model picks among mounted names only, an unmounted vault is an error result. The `where` filter gains the operator core `$eq $ne $in $nin $gt $gte $lt $lte` (unknown operators raise). CLI: `anchor index --vault --namespace`, `anchor query --vault --include --exclude`, `anchor migrate`; pre-vault databases migrate in place under the `__default__` sentinel (vec0 rebuilt by blob copy, no re-embedding). pgvector uses 0.8+ iterative scans when available (older servers degrade to today's behavior); sqlite-vec pin raised to `>=0.1.9` (TEXT-metadata delete bug in 0.1.6–0.1.8)
- **`claude_cli` LLM provider (`[claude-cli]` extra)**: runs any anchor `Agent` on a Claude Code subscription (Pro/Max OAuth or `claude setup-token`) via `claude-agent-sdk` — no API key. Tool calling round-trips through an in-process MCP server plus a `PreToolUse` hook returning `permissionDecision: "defer"`, which stops the CLI turn *without executing* and surfaces a normal `ToolCall`/`StopReason.TOOL_USE` to the loop; mandatory overhead guards (`setting_sources=[]`, `strict_mcp_config=True`) cut per-call prompt cost from ~178k to ~280 tokens; the CLI's own agentic tools (`builtin_tools`) are opt-in with a prompt-injection warning; billed cost is reported into usage accounting
- **`UsageLimits.cost_limit` (USD)**: hard cost cap on the shared pool. Rounds are priced from, in order: provider-reported billed cost (`claude_cli` sends it), the runtime-overridable `anchor.llm.pricing.MODEL_PRICING` table, and genai-prices (`pip install astro-anchor[pricing]`); an unpriced model warns once and debits $0 (tokens still count). `RoundUsage.cost_usd` records the per-round price
- **Aggregated multi-agent diagnostics**: `TurnDiagnostics.children` (`ChildTurn`: tool call id, subagent name, the child's own diagnostics — including a machine-visible `stopped_by` for partial results) plus `run_total_tokens`/`run_total_tool_calls`/`run_total_cost_usd`. Children that die mid-turn (provider error, timeout) still appear with the accounting they accrued
- **Call-time nesting guard**: subagent runners re-check the no-nesting rule on every dispatch, closing the registration-order bypass (`as_tool()` first, `with_subagents` on the wrapped agent later) that could silently build a 3-level tree
- **Approval / human-in-the-loop (inline)**: `Agent.with_approval(callback)` pauses a tool call until the callback returns an `ApprovalDecision` (deny becomes an `is_error` tool result carrying the reason; approve may rewrite the input). Triggered by `AgentTool(requires_approval=True)` (also on `@tool`) or a pre-hook answering `HookResult(decision="ask")`; fails closed without a callback. Async callbacks pause the `astream`/`achat` turn indefinitely — approval timeouts are the application's choice
- **Structured output on the main run**: `Agent.with_output_model(Model)` + `run()`/`arun()` return a validated instance. Default tool mode adds a synthetic `final_result` tool and forces `tool_choice="any"` so the model cannot stop in plain text; invalid arguments retry through the loop's own error-result mechanic (`max_output_retries`). `mode="prompted"` reuses the subagent validate-and-retry path. Output JSON also lands in `TurnFinished.output` / `agent.last_output`
- **Memory tool (`memory_20250818`-compatible, any provider)**: `Agent.with_memory_tool(path)` registers a `memory` tool (view/create/str_replace/insert/delete/rename) over `FileMemoryBackend` — reference-compatible result strings, strict `/memories` containment (URL-encoded traversal rejected), protected root, and the memory protocol appended to the system prompt
- **MCP bridge gaps**: `MCPServerConfig.defer_tools=True` keeps a server's tools out of the prompt until `search_tools` loads them; MCP tool names that shadow existing tools now fail loudly at connect; prompts/resources are finally reachable — `MCPClientPool.all_prompts/get_prompt/all_resources/read_resource` and `Agent.mcp_prompts/mcp_get_prompt/mcp_resources/mcp_read_resource` (app-facing; never injected automatically)
- **Live smoke tests** (`tests/live/`): a real fastmcp server over stdio (credential-free, runs in CI) plus env-gated Anthropic/OpenAI/Gemini turns — the first tests to touch real APIs
- **Agent event stream**: `Agent.stream()`/`Agent.astream()` expose the full tool-use loop as one ordered stream of typed events (`AgentEvent`: `TurnStarted`, `RoundStarted`, `TextDelta`, `ToolStarted`/`ToolFinished` correlated by `tool_call_id`, `CompactionStarted`/`CompactionFinished`, `RoundFinished` with per-round `RoundUsage`, terminal `TurnFinished` with `TurnDiagnostics`); `chat`/`achat` are now text-only projections of the same loop. Async tool calls emit `ToolFinished` live in completion order (results return to the model in call order); subagent events are forwarded flat into the parent stream with `parent_tool_call_id` set; diagnostics persist even when the consumer abandons the generator
- **Usage limits in the agent loop**: `Agent.with_usage_limits(UsageLimits(total_tokens_limit=..., tool_calls_limit=...))` — crossing a limit emits `UsageLimitReached`, grants the model one wrap-up round (final-round notice + `tool_choice="none"`) and ends the turn with `stopped_by="usage_limit"`; no exception, state survives. Complements the pipeline `TokenBudget` (which governs what enters the context window)
- Cross-provider usage estimation: when a provider reports no usage on the stream (all but Anthropic today), `RoundUsage.prompt_tokens`/`completion_tokens` are estimated with the agent's tokenizer, so limits and accounting hold on every provider; provider-reported usage is never overridden. `RoundUsage` gains `tool_calls` and `total_tokens`; `TurnDiagnostics` gains `total_tokens`/`total_tool_calls`
- **Generic `tool_choice` on every provider**: pass `tool_choice="auto"|"any"|"none"` or `{"type": "tool", "name": ...}` (Anthropic shape) to `stream`/`invoke` — mapped natively per provider (Anthropic passthrough, OpenAI/Grok/OpenRouter/Ollama/LiteLLM `required`/function form, Gemini `function_calling_config`); the agent loop forces `tool_choice="none"` on the final round so the model must answer
- **Prompt caching (Anthropic)**: `AnthropicProvider(prompt_caching=True)` sends GA top-level `cache_control={"type": "ephemeral"}` (server picks the breakpoint); cache hits are visible via new `Usage.cache_read_tokens`/`cache_creation_tokens`, threaded into `RoundUsage`/`Agent.last_turn`. The `Agent` enables it by default when it builds an Anthropic provider — the tool loop re-sends a stable prefix every round, exactly the case caching pays for (opt out with `llm=AnthropicProvider(prompt_caching=False)`)
- **Server-side context management (Anthropic, beta)**: `Agent.with_context_management({"edits": [...]})` passes `context_management` through each round; the provider auto-routes to `client.beta.messages.*` with the right flags (`context-management-2025-06-27` for `clear_*` edits, `compact-2026-01-12` for `compact_20260112`); `compaction` blocks are parsed (invoke and streaming) and round-trip verbatim via `Message.raw_content`/`StreamChunk.raw_block`/`LLMResponse.raw_content`
- **Emulated compaction (any provider)**: `Agent.with_compaction(trigger_tokens, keep_last=4, compact_fn=None)` summarizes older loop messages into one `[Conversation summary]` message when the turn exceeds the trigger — default summarizer reuses `TierCompactor` with the agent's own LLM; never severs a tool_use/tool_result pair and skips when the head wouldn't shrink
- **Subagents (MULTI_AGENT.md implemented)**: `Agent.as_tool(name, description, output_model=..., max_output_retries=...)` wraps any agent as an orchestrator tool with a clean context (the `task` string is the only channel in), condensed structured returns (schema instruction + Pydantic validation + one self-contained retry on invalid JSON), and a registration-time no-nesting guard; declarative layer via `Agent.with_subagents([SubagentDefinition(...)])` + a single `task(agent_name, task)` meta-tool with a cache-stable discovery listing in the system prompt; parallel subagent spawns compose with concurrent tool execution in `achat`
- Pre/post tool hooks: `Agent.with_hooks(pre_tool_use=[...], post_tool_use=[...])` — pre-hooks can deny (reason is fed back to the model as an `is_error` tool result) or rewrite input, post-hooks can replace output; a raising pre-hook fails closed (call denied)
- Observer callbacks: `Agent.with_callbacks([...])` with `AgentCallback` protocol (`on_round_start/end`, `on_tool_start/end/error`), fire-and-forget via the shared `fire_callbacks`; `TracingAgentCallback` + `SpanKind.TOOL` wire tool execution into observability
- Per-round token accounting: `Agent.last_turn` (`TurnDiagnostics`) records per-round provider usage, tool-schema tokens, tool-result tokens, and `stopped_by` (`stop`/`max_rounds`/`max_tokens`); `Agent.with_budget(TokenBudget)` attaches a budget to the context pipeline
- Deferred tool loading: `AgentTool(defer_loading=True)` keeps a tool's schema out of the prompt until the auto-registered `search_tools` meta-tool (keyword/regex over names+descriptions) loads it — loaded tools stay loaded for the session
- `input_examples` on `AgentTool`/`ToolSchema`, forwarded by the Anthropic provider (GA field)
- Per-tool timeouts: `AgentTool(timeout=...)` / `Agent(tool_timeout=...)`, enforced with `asyncio.wait_for` on async tool callers (MCP, subagents); a timeout becomes an `is_error` tool result the model sees
- **MCP bridge (`[mcp]` extra, fastmcp)**: consume external MCP servers as agent tools — `MCPServerConfig` (with config-string parser), `FastMCPClientBridge`, and `MCPClientPool` for multi-server management with lazy connection from `Agent(mcp_servers=[...])`; expose an anchor agent as an MCP server via `FastMCPServerBridge`; PEP 544 `MCPClient`/`MCPServer` protocols, tool adapters, and an error hierarchy under `AstroContextError`
- **Progressive summarization memory**: `ProgressiveSummarizationMemory` cascades older turns through summary tiers instead of evicting them, with `TierCompactor` doing LLM summarization plus durable fact extraction; includes the data models, a `ProgressiveSummarizationCallback` lifecycle protocol, and `MemoryManager` integration
- **Persistent storage backends**: SQLite, PostgreSQL, and Redis implementations of the storage protocols (`[sqlite]`/`[postgres]`/`[redis]` extras) for conversation and memory persistence beyond the in-memory and JSON-file references
- Golden-set eval harness (`anchor.evaluation.golden`): `GoldenCase` JSONL loading, `evaluate_retriever`, `assert_metric_floor` — the CI-gate primitive for retrieval changes; `RetrievalMetricsCalculator` now supports graded NDCG via `{id: grade}` relevance maps
- `SqliteVecVectorStore` (`[sqlite-vec]` extra): real KNN inside SQLite via the vec0 virtual table (C cosine distance, declared dimensions, `where` pre-filtering through rowid IN) — replaces the full-scan Python cosine path for local persistence
- `MarkdownHeaderChunker`: structure-aware chunking — splits at headings, stamps each chunk with its `H1 > H2` path in metadata and (by default) in content
- Parsers: `CSVParser` (header-context rows), `JSONParser`, `DocxParser` (stdlib zipfile+ElementTree, DTD-rejecting); `MarkdownParser` now parses YAML frontmatter into metadata instead of discarding it
- Per-page PDF provenance: `PDFParser.parse_pages` + page-aware ingestion stamps every chunk with `doc_page` for citations
- Contextual retrieval hook: `DocumentIngester(contextualize_fn=...)` prepends caller-generated chunk context (Anthropic contextual-retrieval pattern), preserving the original in `metadata["original_content"]`
- Real CLI: `anchor index` (ingest → chunks + optional dense vectors in one SQLite file) and `anchor query` (BM25 + optional dense, RRF-fused) — works fully offline; previously both were placeholders
- Embeddings layer: `EmbeddingProvider` protocol (query/document asymmetry, native batching, async, `dimensions` for Matryoshka truncation) + providers behind extras — `OpenAIEmbeddingProvider` (`[openai]`), `VoyageEmbeddingProvider` (`[voyage]`), `SentenceTransformerEmbeddingProvider` (`[local-embeddings]`, BGE-M3 default for multilingual/PT) — and `CallableEmbeddingProvider` unifying the legacy `embed_fn` shapes
- Metadata filtering: `where: dict` parameter on `VectorStore.search` / `AsyncVectorStore.search`, pushed down in every backend (InMemory dict match, SQLite `json_extract` in SQL, Postgres JSONB containment `@>` with a GIN index) and exposed on `DenseRetriever.retrieve` / `AsyncDenseRetriever.aretrieve` — unblocks user/tenant/type scoping
- `AsyncDenseRetriever` store-backed mode (`vector_store=` + `context_store=`) over the `AsyncVectorStore` protocol — pgvector (`PostgresVectorStore`) and `AsyncSqliteVectorStore` are now reachable from a built-in retriever
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
- Sync tools deliberately run without a timeout — a worker-thread "timeout" cannot cancel the call, only abandon a thread whose side effects keep running; the async path (`tool.timeout` / `Agent(tool_timeout=...)`) is where timeouts are enforceable. `run_skill_script` drops its local head-only 10k-char cut in favor of the loop's head+tail cap (per-tool `max_result_tokens=2500`)
- fastmcp pinned to `>=3.4,<4` (locked 3.4.7); the 4.x stateless-protocol (MCP 2026-07-28) migration is tracked for when it leaves beta
- `anchor._text.strip_markdown_fences`: single shared fence stripper (replaces three inline copies; fixes the compactor variant that dropped the last line even without a closing fence)
- Provider `call_kwargs` construction deduplicated (was copy-pasted 12×): one `_build_call_kwargs` in the Anthropic provider, one shared `build_call_kwargs` for the OpenAI-compatible family (covers Grok/OpenRouter/Ollama/LiteLLM)
- `AnthropicFormatter` caching now emits valid wire format: `cache_control` sits on a content block of the context message (applied after role merging), not as a top-level message key
- `_make_subagent_tool` uses the `@tool` decorator (full Pydantic input validation) instead of a hand-built schema; meta-tool names `task`/`search_tools` are collision-checked at registration/creation time
- `Agent.last_turn` resets to `None` at turn start — an abandoned generator no longer shows the previous turn's diagnostics
- `Agent.chat`/`achat` share one loop body (turn preparation, round assembly, chunk ingestion, accounting) instead of ~70 duplicated lines; `achat` now executes independent tool calls concurrently (`asyncio.gather`, order-preserving, all results in one message)
- Tool errors are forwarded to the model with diagnostics (`Error: tool 'x' failed: RuntimeError: boom`) and `is_error=True` — previously a bare `"Error: tool 'name' failed."`; the Anthropic provider now sets `is_error` on `tool_result` blocks (previously dropped)
- Round-limit signalling: one round before `max_rounds` the agent injects a final-round notice so the model wraps up instead of being cut off; tool calls made on the final round are **not executed** (their results could never reach the model); the outcome is visible as `last_turn.stopped_by == "max_rounds"` instead of a silent stop
- `Agent` default tokenizer: tiktoken-backed when installed (whitespace-word fallback otherwise) — token accounting and the skill-listing cap now use real counts; `Agent(tokenizer=...)` accepts any `Tokenizer`
- BM25 backend: **bm25s** (numpy sparse scoring, up to 500x faster, Lucene-variant correctness) with Snowball stemming + stopwords via `SparseRetriever(language=...)` — Portuguese finally stems (`correndo` matches `correr`); rank_bm25 remains the fallback and the path for custom `tokenize_fn`; the `bm25` extra now ships `bm25s + PyStemmer`
- Chunking defaults: `RecursiveCharacterChunker` 512/50 → **384/0** per Chroma's chunking evaluation (recursive at 200-400 tokens, zero overlap, within ~2 recall points of semantic chunking)
- Chunker sizing loops are O(n) (per-word token counts) instead of re-encoding the growing chunk per word (O(n²)); exact for whitespace tokenizers, approximate for BPE
- `ParentChildChunker` stores parent text once on the chunker (content-hash ids, globally unique across documents — the old `parent-{idx}` collided between docs) instead of duplicating the full parent into every child's metadata; `ParentExpander` takes a `parent_lookup`
- `LLMRAGEvaluator`: unconfigured metric dimensions are now `None` (not evaluated) instead of a silent 0.0; constructing with zero callbacks raises
- Skills: `Skill.activation` default flipped from `"always"` to `"on_demand"` (matches the loader default and progressive-disclosure semantics); canonical frontmatter location for activation is now `metadata: {activation: ...}` (top-level key still accepted)
- Skills: the discovery listing is static (no per-round `[active]` mutation) and joined to the system prompt with newlines, computed once per turn — prompt-cache friendly; the agent's `AnthropicFormatter` now has caching enabled
- Skills: tool-name collisions raise at registration time (`SkillRegistry.register` and `Agent.with_skill`), never mid-conversation
- Reranker `rerank()`/`arerank()` signatures: `top_k: int = 10` → `top_k: int | None = None`; an explicit `top_k` now always overrides the constructor value, `None` falls back to it (applies to all rerankers, the `Reranker`/`AsyncReranker` protocols, and `reranker_step`/`async_reranker_step`)
- `AsyncCohereReranker` callback now returns `(index, score)` tuples, matching the sync `CohereReranker` shape, and applies scores to results
- `HybridRetriever` and `AsyncHybridRetriever` now delegate fusion to the canonical `rrf_fuse` (single RRF implementation); `rrf_fuse` gained a `retrieval_method` label parameter
- `AsyncHybridRetriever` raises `RetrieverError` when all sub-retrievers fail, matching the sync behavior (previously returned `[]`)
- All cosine-similarity call sites (`SemanticChunker`, late interaction, cross-modal, `EmbeddingClassifier`) now use the strict `anchor._math.cosine_similarity` — dimension mismatches raise instead of silently truncating
- `DenseRetriever.index` embeds documents in one batch call (was one embedding call per item)
- Postgres: `ensure_tables` now creates the pgvector **HNSW** index (works on empty tables, unlike the previously suggested IVFFlat) and a JSONB GIN index; `embedding_dim` is a required parameter
- `Role` and `StopReason` enums changed from `(str, Enum)` to `StrEnum` for correct string formatting
- Repo workflow: `docs/superpowers/` retired; specs/plans live in `docs/plans/`, research in `docs/research/`

### Fixed
- **Pre-vault migrations rebuilt from the ritual xhigh of roadmap #3** (sessão 10): SQLite/sqlite-vec legacy tables kept a bare-id key, so `INSERT OR REPLACE` from one vault silently deleted another vault's row (and `vec_items` rejected a second vault) — `ensure_tables` now rebuilds the `(vault, id)` key in one transaction, rowids preserved; Postgres legacy tables lacked the unique index `ON CONFLICT (vault, id)` needs (42P10 on every write) — the migration probes the catalog, recreates the composite PK and puts `COLLATE "C"` on `namespace` (prefix ranges under libc `en_US` collations dropped descendants on include and leaked them on exclude); the vec0 rebuild is transactional with a TEMP staging table
- **Same `where`, same answer on every backend**: SQL `$ne`/`$nin` no longer drop rows missing the key (memory and Postgres kept them); range operators compare like with like — the Postgres `::numeric` cast crashed every search on a vault holding one non-numeric value, SQLite matched strings on `$gt` (TEXT sorts above INTEGER)
- **Scope reaches the pipeline**: the turn's `ContextPipeline.build()` ran before the published scope existed, so `with_scope()` and a parent's scope never applied to `retriever_step` retrieval (sync, async, subagent) — the build now runs under the same published scope as tool calls
- `DenseRetriever`/`AsyncDenseRetriever` refuse a vector store and a context store mounted on different vaults (they indexed vectors into one and content into the other, silently)
- `anchor query` is back on `HybridRetriever` (a dense failure aborted the command instead of degrading to BM25 with a warning); `anchor migrate` reads the vec0 dimension from the index instead of a regex, and reports a missing `sqlite-vec` instead of a traceback
- `SqliteConnectionManager.get_async_connection` closes a losing first-call connection when its setup fails (`database is locked` while the winner switches to WAL) and retries — the leaked non-daemon worker thread used to hang interpreter shutdown, i.e. the whole test suite
- **`create_provider()` deadlocked on first use of any provider**: the registry `_LOCK` was a non-reentrant `threading.Lock` held during the lazy provider import, and every provider module self-registers on import by re-acquiring the same lock on the same thread — in a fresh process, `Agent(model="anthropic/claude-...")` (or any `create_provider()` call) hung forever. Shipped in 0.1.1; the mocked suite never saw it because provider tests import the classes directly first. Fixed with `RLock` plus a regression test exercising the real lazy-import path under a joined-with-timeout thread
- MCP stdio connections with arguments actually work: the bridge passed a concatenated `"command arg arg"` string that fastmcp's transport inference rejects — found by the first live smoke test (the mocked suite patched `Client` and never saw it); the bridge now builds an explicit `StdioTransport`
- Gemini streaming tool calls now work: the stream parser emits generated call ids, sequential indices for parallel calls, JSON argument fragments (was `str(dict)`), and a `TOOL_USE` stop reason override mirroring the non-stream path — previously tools requested via Gemini streaming silently never executed
- **Agent without memory never sent the user's message**: the formatter emits no conversation when no memory is attached, so `chat("...")` produced a system-only request (a real API call would 400); the turn now guarantees the user message is present
- **Reranker `top_k` sentinel bug**: passing `top_k=10` explicitly was indistinguishable from the default and silently discarded in all five rerankers; the pipeline `reranker_step` default hit this on every run
- `RecursiveCharacterChunker` applied overlap at every recursion level, duplicating overlap text in nested sub-chunks; overlap is now applied exactly once over the final chunk list
- `SqliteVectorStore` unpacked stored embeddings using the *query* vector's dimension — a query/stored dimension mismatch silently mis-unpacked or crashed with `struct.error`; the dimension now derives from the stored blob and mismatches raise a clear `ValueError`
- 34 pre-existing test failures caused by missing optional dependencies (tiktoken, rank-bm25)
- `FallbackProvider.astream` mid-stream fallback semantics (yields now outside try/except)
- `test_consolidator.py`: eliminated shared mutable state (`_orthogonal_index` dict) by converting to factory function pattern (`make_orthogonal_embed()`)
- `test_graph_memory.py`: updated `link_memory` unknown entity test to expect `KeyError` instead of `ValueError`
- README: fixed retrieval example with runnable `embed_fn` and `ContextItem` creation
- README: updated test count from 961 to 1088
- **NDCG graded a short result set against a shrunken ideal**: the ideal ranking was truncated at the number of items *returned* instead of at k — a broken retriever returning only its 2 relevant hits scored a perfect NDCG@10, so the golden-set CI gate passed on real regressions that shrank the result set (both the binary and graded variants; shipped in 0.1.x)
- **`SlidingWindowMemory` invoked `on_evict` while holding its non-reentrant lock**: a callback that read the memory back (e.g. `total_tokens`) self-deadlocked, and slow eviction work (LLM summarization in progressive memory) serialized every thread on the window; the callback now fires after the lock is released (shipped in 0.1.x)

### Removed
- `anchor index --language/-l`: it was displayed but never persisted (BM25 is built at query time — `anchor query -l` is where the language matters)
- Late-interaction retriever (`LateInteractionRetriever`/`LateInteractionScorer`/`MaxSimScorer`, `TokenLevelEncoder` protocol): required per-token document embeddings recomputed per query — never viable as shipped, no known users; a ColBERT-style retriever can return behind an extra when demanded
- `TokenBudgetExceededError` (never raised anywhere; usage limits stop gracefully instead of raising) and the orphan `anchor.models.streaming` module (`StreamDelta`/`StreamUsage`/`StreamResult` — superseded by `AgentEvent`)
- `ScoreReranker` (dead duplicate of `CrossEncoderReranker` implementing the wrong protocol)

## [0.1.1] - 2026-03-13

First PyPI release. Same feature set as 0.1.0 — the delta is packaging,
branding, and documentation (verified against the published sdist).

### Changed
- Project renamed from `astro-context` to **anchor**; the import is now `anchor` and the PyPI package name is `astro-anchor`
- Comprehensive documentation overhaul (MkDocs Material, GitHub Pages deploy workflow), modernized README, Astro brand identity and logos

### Fixed
- mypy arg-type error for the `Skill` activation literal; deterministic `_fake_embed` and otel import-blocking in tests

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
