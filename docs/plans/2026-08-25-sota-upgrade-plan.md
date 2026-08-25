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

- [x] Real YAML frontmatter parsing (replace hand-rolled splitter, `loader.py:28-57`); accept spec fields `license`, `compatibility`, `metadata`, `allowed-tools`; move `activation` under `metadata` as anchor extension; validate name == directory name
- [x] Level-3 progressive disclosure: skill-scoped `read_reference` tool for lazy `references/` loading; `scripts/` executed via opt-in exec tool gated by `allowed-tools`
- [x] Decide fate of `tools.py` import path: keep as explicit trusted-Python escape hatch or deprecate (it is arbitrary code exec at load, `loader.py:94-136`)
- [x] Inject `always`-skill `instructions` into the system prompt at build (today dead text — `memory/skill.py:30` never reaches the model)
- [x] Cache-stable discovery listing: static block, newline-separated, no per-round `[active]` mutation (`registry.py:132`, `agent.py:395-402`); enable `AnthropicFormatter` caching (today hardcoded off, `agent.py:120`); listing token cap (Claude Code uses 1% of context)
- [x] Namespace skill tools (`skillname__tool`), collision check at registration, not mid-conversation (`registry.py:121`)
- [x] Fix `activation` default mismatch (`models.py:44` says `always`, `loader.py:174` says `on_demand`)
- [x] Skill eval harness: with/without-skill baseline runs, trigger tuning on descriptions (≥3 scenarios per skill)

**Done when:** an unmodified skill from anthropics/skills loads and runs in anchor; anchor's example skill validates with `skills-ref validate`.

---

## Phase 2 — Embeddings layer + filtered vector search *(the missing layer)*

Details: retrieval analysis §1–§3.

- [x] `EmbeddingProvider` protocol: `embed_query` / `embed_documents` (asymmetry), batch, async, `dimensions` (Matryoshka), `dtype` (float/int8); replaces the 3 incompatible `embed_fn` shapes
- [x] 2–3 providers as optional extras: OpenAI, Voyage or Gemini, local sentence-transformers (PT default: BGE-M3)
- [x] Batch indexing in `DenseRetriever.index` (today 1 call per item)
- [x] `filter: dict | None` on `VectorStore.search` protocol + all backends (unblocks user_id/type/date scoping, multi-tenant, self-query)
- [x] Make pgvector reachable: sync store or async `DenseRetriever` over `AsyncVectorStore`; **HNSW index created in `ensure_tables`** (not an IVFFlat comment); `embedding_dim` required param
- [x] Converge sync/async retrievers on one code path over the store protocols (`AsyncDenseRetriever` currently bypasses `AsyncVectorStore` entirely)
- [x] Postgres integration tests (today zero)

**Done when:** end-to-end test ingest → embed (batched) → store → filtered retrieve → rerank passes on InMemory, SQLite, and Postgres.

---

## Phase 3 — Retrieval quality per stage

Details: retrieval analysis §4–§10. Every change here gates on the golden-set metric, so build that first.

- [x] Golden-set eval harness: 50–200 real queries, recall@k / graded NDCG via correct implementations; CI-gate for retrieval changes; fix `LLMRAGEvaluator` silent-0.0 (missing callback → error, not score)
- [x] BM25: swap rank_bm25 → **bm25s** + PyStemmer Portuguese + stopwords; pluggable tokenizer; keep raw scores
- [x] **sqlite-vec** backend (+ FTS5 for single-file hybrid) as the local vector store; retire the full-scan Python cosine path past small N
- [x] Chunking: defaults to 256–400 tokens / 0–50 overlap (Chroma evidence); propagate markdown headers into chunk metadata; `ParentChildChunker` stores parent **id** not full text (`hierarchical.py:137`); fix quadratic sizing loops; tokenizer encoding parameterized (o200k_base option)
- [x] Parsing: per-page metadata on PDF chunks (citation provenance); parse frontmatter into metadata instead of discarding (`parsers.py:105`); add csv/json/docx parsers
- [x] Contextual enrichment step (Anthropic-style chunk-context prepend, LLM-callback-delegated) as an opt-in ingestion step
- [x] Real CLI: `anchor index` = ingest → embed → sqlite-vec; `anchor query` = hybrid + rerank (today both are stubs, `cli.py:66-120`)

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

### Phase 3 — shipped 2026-08-25
All 7 items landed. Evidence: 2613 passed / 3 skipped; new suites `tests/test_ingestion/test_phase3_quality.py` (14), `tests/test_evaluation/test_golden.py` (5), `tests/test_storage/test_sqlite/test_vec_store.py` (8, incl. a Portuguese-stemming BM25 test), CLI suite rewritten against the real commands (11); CLI verified E2E by hand (index a mixed-corpus dir, query returns the right doc offline). Deviations: (a) FTS5 single-file hybrid deferred — bm25s (in-memory, memory-mappable) covers sparse and the CLI rebuilds it from the context store at query time; revisit if corpora outgrow that; (b) tokenizer-encoding parameterization was already present (`TiktokenCounter(encoding_name=...)`) — no change needed; (c) golden-set gating is infrastructure + tests here; the actual 50-200-query golden set is corpus-specific and must be authored per project; (d) `FixedSizeChunker` keeps 512/50 defaults (parent/child sizing depends on them), only the ingester default `RecursiveCharacterChunker` moved to 384/0.

### Phase 2 — shipped 2026-08-25
All 7 items landed. Evidence: 2603 passed / 3 skipped; new suite `tests/test_retrieval/test_phase2_embeddings_filtering.py` (18 tests: protocol conformance, batch-call counting, `where` on InMemory/SQLite sync+async, store-backed async E2E, HNSW schema assertions) + env-gated pgvector integration suite. Deviations: (a) filter param named `where` (Chroma-style) not `filter` (builtin shadowing); equality-only semantics — range/negation operators deferred until needed; (b) protocol carries `dimensions` but not `dtype` — no backend stores int8 today, quantization lands with the sqlite-vec backend (Phase 3); (c) sync/async convergence achieved by teaching `AsyncDenseRetriever` the store-backed path (the in-process legacy mode kept for its 28 existing tests) rather than a shared-code rewrite; (d) `where` is passed to stores only when set, so pre-existing custom VectorStore implementations keep working.

### Phase 1 — shipped 2026-08-25
All 8 items landed. Evidence: 2585 passed / 2 skipped; new suite `tests/test_agent/test_skills/test_spec_compliance.py` (16 tests) + spec-style fixture `tests/fixtures/skills/spec-demo/` (multi-line YAML description, license, metadata.activation, references/, scripts/). Deviations: (a) tool namespacing replaced by registration-time collision checks — renaming tools would change what the model calls for zero gain while collisions are rare; full namespacing deferred until a real collision case appears; (b) `tools.py` import path KEPT as a documented trusted-Python escape hatch (spec-portable skills should use scripts/); (c) `skills-ref validate` not runnable in this environment — parity asserted via loader tests against the spec rules (name format/length, description length, compatibility cap, name==dir). Breaking: `Skill.activation` default is now `on_demand`; `minimal` fixture renamed `minimal-helper` to satisfy name==dir.

### Phase 0 — shipped 2026-08-25
All 8 items landed in one pass. Evidence: full suite 2518 passed / 4 skipped; new regression file `tests/test_retrieval/test_phase0_regressions.py` (18 tests) pins every fix; ruff clean on all touched files. Deviations from plan: (a) score fields stay [0,1] (ContextItem pydantic constraint) with the raw value in `metadata["raw_score"]` — RRF already preserved `rrf_raw_score`, kept as-is; (b) explicit reranker `top_k` now *overrides* the constructor (per the original docstrings) instead of `min()`; (c) SQLite dimension validation done by deriving dim from the stored blob (strict cosine raises on mismatch) rather than a constructor param — declared-dimension API deferred to Phase 2's protocol rework. Breaking changes (pre-1.0): `AsyncCohereReranker` callback shape, `ScoreReranker` removed, async all-fail now raises.
