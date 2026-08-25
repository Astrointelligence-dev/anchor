# SOTA 2026 Upgrade Plan

**Date:** 2026-08-25 · **Baseline:** `dev` @ `4138306` (v0.1.1)
**Research:** `docs/research/2026-08-25-sota-gap-analysis.md` (skills, agent loop, context, MCP, harness) · `docs/research/2026-08-25-retrieval-stack-analysis.md` (embeddings, retrieval, storage, chunking, parsing, eval). All claims there carry file:line references and sources; this plan only sequences the work.

**Rules of engagement:** one phase per branch/worktree; every phase lands with tests that would have caught what it fixes; phases 1–3 (retrieval track) and 4 (agent track) are independent and can run in parallel worktrees. Storage-layer-gaps (`docs/plans/2026-03-14-storage-layer-gaps-design.md`) continues on its existing worktree, unaffected.

---

## Phase 0 — Correctness fixes *(no design, small diffs, do first)*

Bugs are live today regardless of any SOTA ambition. Full details: retrieval analysis §0.

- [x] Fix the `top_k != 10` sentinel in all 5 rerankers → `top_k: int | None = None` (`retrieval/rerankers.py:84,152,247,315,422`); regression test: explicit `top_k=10` must be honored
- [x] Stop destroying score semantics: preserve raw scores in `metadata["raw_score"]`; remove negative-cosine clamp (`dense.py:73`), FlashRank clamp (`rerankers.py:267`); keep normalization only as opt-in; add `min_score` support to retrievers
- [x] Deduplicate RRF: `HybridRetriever` and `AsyncHybridRetriever` call `rrf_fuse` (`_rrf.py:14`) instead of inline copies
- [x] Deduplicate cosine similarity: all 5 copies → `_math.cosine_similarity` (strict); test: dim mismatch raises, never truncates
- [x] Fix recursive-chunker overlap double-application (`chunkers.py:198`)
- [x] Dimension validation in vector stores: declared at construction, mismatch = clear error (SQLite `_vector_store.py:71`, Postgres `_schema.py:14`)
- [x] Delete dead `ScoreReranker` (`retrieval/reranker.py`); align sync/async `CohereReranker` callback shapes
- [x] Align sync/async failure behavior (`HybridRetriever` raises vs `AsyncHybridRetriever` returns `[]`)

**Done when:** all existing tests green + new regression tests for each item.

---

## Phase 1 — Skills to spec *(the interop play)*

Adopt the agentskills.io open standard (40+ products read it; full compliance means anchor loads public skills and anchor skills run in Claude Code/Codex). Details: gap analysis §1.

- [ ] Real YAML frontmatter parsing (replace hand-rolled splitter, `loader.py:28-57`); accept spec fields `license`, `compatibility`, `metadata`, `allowed-tools`; move `activation` under `metadata` as anchor extension; validate name == directory name
- [ ] Level-3 progressive disclosure: skill-scoped `read_reference` tool for lazy `references/` loading; `scripts/` executed via opt-in exec tool gated by `allowed-tools`
- [ ] Decide fate of `tools.py` import path: keep as explicit trusted-Python escape hatch or deprecate (it is arbitrary code exec at load, `loader.py:94-136`)
- [ ] Inject `always`-skill `instructions` into the system prompt at build (today dead text — `memory/skill.py:30` never reaches the model)
- [ ] Cache-stable discovery listing: static block, newline-separated, no per-round `[active]` mutation (`registry.py:132`, `agent.py:395-402`); enable `AnthropicFormatter` caching (today hardcoded off, `agent.py:120`); listing token cap (Claude Code uses 1% of context)
- [ ] Namespace skill tools (`skillname__tool`), collision check at registration, not mid-conversation (`registry.py:121`)
- [ ] Fix `activation` default mismatch (`models.py:44` says `always`, `loader.py:174` says `on_demand`)
- [ ] Skill eval harness: with/without-skill baseline runs, trigger tuning on descriptions (≥3 scenarios per skill)

**Done when:** an unmodified skill from anthropics/skills loads and runs in anchor; anchor's example skill validates with `skills-ref validate`.

---

## Phase 2 — Embeddings layer + filtered vector search *(the missing layer)*

Details: retrieval analysis §1–§3.

- [ ] `EmbeddingProvider` protocol: `embed_query` / `embed_documents` (asymmetry), batch, async, `dimensions` (Matryoshka), `dtype` (float/int8); replaces the 3 incompatible `embed_fn` shapes
- [ ] 2–3 providers as optional extras: OpenAI, Voyage or Gemini, local sentence-transformers (PT default: BGE-M3)
- [ ] Batch indexing in `DenseRetriever.index` (today 1 call per item)
- [ ] `filter: dict | None` on `VectorStore.search` protocol + all backends (unblocks user_id/type/date scoping, multi-tenant, self-query)
- [ ] Make pgvector reachable: sync store or async `DenseRetriever` over `AsyncVectorStore`; **HNSW index created in `ensure_tables`** (not an IVFFlat comment); `embedding_dim` required param
- [ ] Converge sync/async retrievers on one code path over the store protocols (`AsyncDenseRetriever` currently bypasses `AsyncVectorStore` entirely)
- [ ] Postgres integration tests (today zero)

**Done when:** end-to-end test ingest → embed (batched) → store → filtered retrieve → rerank passes on InMemory, SQLite, and Postgres.

---

## Phase 3 — Retrieval quality per stage

Details: retrieval analysis §4–§10. Every change here gates on the golden-set metric, so build that first.

- [ ] Golden-set eval harness: 50–200 real queries, recall@k / graded NDCG via correct implementations; CI-gate for retrieval changes; fix `LLMRAGEvaluator` silent-0.0 (missing callback → error, not score)
- [ ] BM25: swap rank_bm25 → **bm25s** + PyStemmer Portuguese + stopwords; pluggable tokenizer; keep raw scores
- [ ] **sqlite-vec** backend (+ FTS5 for single-file hybrid) as the local vector store; retire the full-scan Python cosine path past small N
- [ ] Chunking: defaults to 256–400 tokens / 0–50 overlap (Chroma evidence); propagate markdown headers into chunk metadata; `ParentChildChunker` stores parent **id** not full text (`hierarchical.py:137`); fix quadratic sizing loops; tokenizer encoding parameterized (o200k_base option)
- [ ] Parsing: per-page metadata on PDF chunks (citation provenance); parse frontmatter into metadata instead of discarding (`parsers.py:105`); add csv/json/docx parsers
- [ ] Contextual enrichment step (Anthropic-style chunk-context prepend, LLM-callback-delegated) as an opt-in ingestion step
- [ ] Real CLI: `anchor index` = ingest → embed → sqlite-vec; `anchor query` = hybrid + rerank (today both are stubs, `cli.py:66-120`)

**Done when:** golden-set recall@k improves or holds on every merged item; CLI demo works offline with a local embedder.

---

## Phase 4 — Agent loop to 2026

Details: gap analysis §2–§5, §8. Independent of phases 2–3.

- [ ] Deduplicate `chat`/`achat` (~70 duplicated lines, `agent.py:388-454` vs `:496-562`) — precondition for everything below
- [ ] Tool errors forwarded to the model with diagnostic text (today `"Error: tool 'name' failed."`, `agent.py:236`); per-tool timeout; parallel execution of independent calls in `_arun_tools` (`asyncio.gather`)
- [ ] Pre/post tool-call hooks (approval seam, observability wiring)
- [ ] Budget integration: real tokenizer (replace whitespace counter, `agent.py:38`), `SourceType.TOOL` items for tool schemas, `TokenBudget` attached, rounds 2..N accounted
- [ ] Subagents: `agent-as-tool` primitive — clean pipeline, restricted tools, JSON-schema condensed return (implements MULTI_AGENT.md: isolation + asymmetric returns)
- [ ] Deferred tool loading + tool-search meta-tool for large tool counts; `input_examples` on `AgentTool`/`ToolSchema`
- [ ] Round-limit signalling (loop currently falls off `max_rounds` silently)

**Done when:** a 2-level agent (orchestrator + subagent) runs a task with per-round token accounting visible in diagnostics.

---

## Phase 5 — Platform reach *(after the above, pick by demand)*

- [ ] Anthropic provider: context editing (`clear_tool_uses`), server-side compaction (`compact_20260112`), prompt-caching flags; emulation path for other providers
- [ ] Memory-tool command-set compatibility (`view/create/str_replace/insert/delete/rename`, path-traversal guards); file-based memory backend
- [ ] Docling parser extra (tier-2 structured parsing); VLM parsing as cookbook recipe
- [ ] LanceDB backend; tree-sitter AST code chunking; BGE-reranker-v2-m3 extra
- [ ] Late-interaction retriever: rewrite with precomputed doc embeddings or delete
- [ ] MCP 2026-07-28 readiness: deterministic tool ordering, `defer_loading` for MCP tools, stateless transport tracking
- [ ] Redis vector store (Vector Sets) — or drop the implied promise

---

## Review log

### Phase 0 — shipped 2026-08-25
All 8 items landed in one pass. Evidence: full suite 2518 passed / 4 skipped; new regression file `tests/test_retrieval/test_phase0_regressions.py` (18 tests) pins every fix; ruff clean on all touched files. Deviations from plan: (a) score fields stay [0,1] (ContextItem pydantic constraint) with the raw value in `metadata["raw_score"]` — RRF already preserved `rrf_raw_score`, kept as-is; (b) explicit reranker `top_k` now *overrides* the constructor (per the original docstrings) instead of `min()`; (c) SQLite dimension validation done by deriving dim from the stored blob (strict cosine raises on mismatch) rather than a constructor param — declared-dimension API deferred to Phase 2's protocol rework. Breaking changes (pre-1.0): `AsyncCohereReranker` callback shape, `ScoreReranker` removed, async all-fail now raises.
