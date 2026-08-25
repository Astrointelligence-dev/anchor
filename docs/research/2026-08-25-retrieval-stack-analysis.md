# Retrieval Stack Analysis — anchor vs SOTA 2026, stage by stage

**Date:** 2026-08-25 · **Baseline:** `dev` @ `4138306` (v0.1.1)
**Companion to:** `2026-08-25-sota-gap-analysis.md` (skills, agent loop, context, MCP, harness). This doc covers the layers under the RAG pipeline: embeddings, vector search, storage, sparse, reranking, chunking, parsing, query, evaluation. Each stage: what's implemented, what's good, what to improve.

Verdict up front: **the architecture (protocol-first, PEP 544) is the right shape** — everything below slots in without redesign. But the stack has (a) five real correctness bugs, (b) one missing layer entirely (embeddings), and (c) one protocol gap (metadata filtering) that blocks the whole category of filtered retrieval.

---

## 0. Correctness bugs (fix regardless of SOTA)

| # | Bug | Where | Effect |
|---|---|---|---|
| 1 | **`top_k != 10` sentinel** — `min(top_k, self._top_k) if top_k != 10 else self._top_k` | all 5 rerankers (`rerankers.py:84,152,247,315,422`) | explicit `top_k=10` silently discarded; `reranker_step` default passes 10, so pipeline top_k never reaches rerankers. Fix: `top_k: int \| None = None`. |
| 2 | **Score semantics destroyed 3×** — BM25 max-norm (`sparse.py:79`), RRF min-max renorm (`_rrf.py:60`), negative-cosine clamp (`dense.py:73`), FlashRank clamp (`rerankers.py:267`) | retrieval | top hit always 1.0; cross-query comparison and thresholding impossible. Fix: preserve raw score in `metadata["raw_score"]`, stop clamping, add optional `min_score`. |
| 3 | **RRF triplicated** — canonical `_rrf.py:14` + inline copies in `hybrid.py:75-107` and `async_retriever.py:182-216` | fusion | a fix to one won't reach the others. Same story: cosine similarity ×5 (two with `strict=False` that silently truncate on dim mismatch — `chunkers.py:320`, `late_interaction.py:22`). |
| 4 | **Recursive overlap double-application** — `_apply_overlap` called inside recursive `_split` (`chunkers.py:198`) | chunking | nested sub-chunks get overlap applied twice. |
| 5 | **No dimension validation** — SQLite infers dim from the *query* vector (`sqlite/_vector_store.py:71`); Postgres hardcodes 1536 (`_schema.py:14`) | storage | mismatch = silent `struct.error`/mis-unpack instead of a clear error. |

Plus: `chat`/`achat` sync-async divergences leak here too — `HybridRetriever` raises when all sub-retrievers fail, `AsyncHybridRetriever` returns `[]`; sync/async `CohereReranker` callbacks have different shapes.

---

## 1. Embeddings — **the missing layer**

**Live:** nothing. No provider abstraction, no built-in model, zero `embed` hits in `llm/`. The contract is a bare user callable in **three incompatible shapes** (single-text sync, batch sync, single-text async). No batching (indexing 100k chunks = 100k sequential calls), no normalization, one ad-hoc cache (`consolidator.py:53`, clear-on-full).

**Good:** honesty — the SDK never fakes a vector; deps stay minimal.

**SOTA 2026:** every serious provider exposes the same four knobs the abstraction must carry:
- **query/document asymmetry** (`embed_query` vs `embed_documents`) — E5 prefixes, Qwen3 instruction-on-query-only (+1–5% retrieval);
- **`dimensions` param** (Matryoshka truncation: 512d retains 94–98%, 4–8× storage cut);
- **`dtype`** (int8 near-free at 4× smaller; binary 32× + rescoring);
- **batch + async**.

Model landscape: Voyage-4 series ($0.02–0.12/M, 32K ctx, MRL) for quality/cost; `gemini-embedding-001` for multilingual (but **2,048-token input cap** — constrains chunking); OpenAI-3-small as commodity; open-weight Qwen3-Embedding (#1 MTEB multilingual) / **BGE-M3** for PT self-hosted. There's a dedicated **MTEB-PT** benchmark now — evaluate Portuguese specifically, multilingual averages hide variance.

**Do:** `EmbeddingProvider` protocol (LlamaIndex-shaped: `embed_query`, `embed_documents_batch`, async variants, `dimensions`, `dtype`) + 2–3 providers (OpenAI, Voyage or Gemini, local sentence-transformers) as optional extras; unify the three callable shapes behind it; add batch indexing to `DenseRetriever.index`.

---

## 2. Vector search (retrievers)

**Live:** `DenseRetriever` delegating to store; `SparseRetriever` (rank_bm25); `HybridRetriever` RRF k=60 with weights; 3 trivial routers + a separate `EmbeddingClassifier` routing path; ColBERT-style `LateInteractionRetriever` (O(|Q|·|D|) pure-Python, re-encodes every candidate per query — unusable at scale, unwired); `ScoredMemoryRetriever` (`0.3*recency + 0.5*relevance + 0.2*importance`, scores every entry per query); parallel-universe async classes sharing zero code (`AsyncDenseRetriever` bypasses `AsyncVectorStore` — in-process list).

**Good:** hybrid dense+BM25+RRF **is still the 2026 baseline** — the shape is right. RRF k=60 is the literature value. Router/classifier callbacks are honest. 243 retrieval tests.

**Improve:**
- **No metadata filtering anywhere** — `VectorStore.search(query_embedding, top_k)` has no filter param; metadata is write-only in every backend. This is the single biggest capability gap: user_id/doc-type/date scoping, multi-tenant, self-query all blocked. Add `filter: dict | None` to the protocol and implement per backend (pgvector 0.8's iterative index scans exist precisely for filtered ANN).
- No `min_score`, no MMR/diversity, silent drop of items missing from context store.
- Consolidate sync/async on one code path consuming the store protocols; delete or rewrite `LateInteractionRetriever` (precompute doc token embeddings or drop it — ColPali-era late interaction needs storage support, not per-query re-encoding).
- Two routing systems (`router.py` vs `classifiers.py`) → one interface.

---

## 3. Vector storage

**Live:** 3-method protocol (`add_embedding/search/delete`); InMemory (locked full scan, warn >5k); SQLite (float32 blobs, `SELECT` **all rows no LIMIT**, cosine in Python — ~300MB materialized per query at its self-declared 50k ceiling); Postgres pgvector (real `<=>` query, but **async-only → unreachable from sync `DenseRetriever`**, no index created — a comment tells the user to make IVFFlat by hand), **no Redis vector store**, zero Postgres/Redis tests.

**Good:** correct upsert everywhere; clean protocol seam; pgvector SQL is the right syntax.

**SOTA 2026:**
- Brute force is genuinely fine under ~100K vectors — but in numpy/C, not a Python `sum()` loop. anchor has no numpy; the lazy fix is **sqlite-vec** (v0.1.9, pure C, zero deps, int8/binary vectors, metadata + partition-key filtering built in, pairs with FTS5 for hybrid in one file) as the local backend.
- pgvector 0.8.6: create **HNSW** (`m=16, ef_construction=64`) in `ensure_tables`, not an IVFFlat comment; `halfvec` for 50% storage; iterative scans for filtered queries; make dim a required parameter.
- **LanceDB** as the embedded option past SQLite scale (IVF-PQ, native BM25, hybrid + rerankers built in).
- Protocol needs: `filter`, batch add, `count`, dimension declared at construction.

---

## 4. Sparse / BM25

**Live:** `rank_bm25.BM25Okapi`, tokenizer = `text.lower().split()` (no stemming/stopwords/punctuation), max-normalized scores, full corpus rebuild on every `index()`, all in RAM.

**Good:** lazy optional import; heapq top-k; the extra exists.

**SOTA 2026:** **rank_bm25 is legacy — bm25s is up to 500× faster** (numpy sparse scoring, correct Lucene/ATIRE variants, memory-mappable). For Portuguese: bm25s + **PyStemmer** (`Stemmer("portuguese")`) + `stopwords="pt"` — stemming measurably improves BM25; the current whitespace tokenizer is the worst case for PT morphology. On SQLite, **FTS5** gives BM25 for free in the same file (but unicode61 doesn't stem — pre-stem tokens).

**Do:** swap extra `bm25 = ["bm25s", "PyStemmer"]`, pluggable tokenizer with a PT default, keep raw scores.

---

## 5. Reranking

**Live:** FlashRank (`ms-marco-MiniLM-L-12-v2`, lazy, clamped scores) + callback-based CrossEncoder/Cohere + pipeline/round-robin composition + async variants with divergent callback shapes + dead `ScoreReranker` duplicate. The sentinel bug (§0.1) hits all of them.

**Good:** the *composition* design (RerankerPipeline, first-stage → rerank flow) is right; reranking is the single biggest retrieval lever per 2026 evidence, and anchor treats it as first-class. 46 tests.

**SOTA 2026:** FlashRank is still the valid zero-torch CPU tier but its bundled models are old-gen. Current ladder: **BGE-reranker-v2-m3** (multilingual self-host default) / Qwen3-Reranker → API tier Voyage rerank-2.5-lite ($0.001/100-doc request) / Cohere Rerank 4. The AnswerAI `rerankers` lib is the de-facto unified abstraction — mirror its interface shape. Skip-rerank guidance: skip when hybrid top-5 already saturates recall or <100ms budgets.

**Do:** fix sentinel; delete `ScoreReranker`; align sync/async callback shapes; add a BGE-v2-m3 option (sentence-transformers extra); document the skip heuristic.

---

## 6. Chunking

**Live:** 7 chunkers (fixed, recursive, sentence, semantic, code-regex, table-aware, parent-child), token-sized via tiktoken `cl100k_base` singleton, defaults 512/50.

**Good:** genuinely broad coverage — table-aware with header-repetition on split and parent-child are ahead of most SDKs; deterministic IDs; metadata propagation (`parent_doc_id`, `chunk_index`, `doc_*` prefixing) is clean.

**Improve (evidence-based):**
- **Chroma's chunking eval:** well-parameterized recursive at 200–400 tokens **with zero overlap** hits ~89.5% recall; best semantic chunker beats it by <2 points at much higher cost. So: change defaults toward 256–400/0–50, and *de-prioritize* `SemanticChunker` (which also uses a fixed absolute threshold instead of the percentile method, plus the `strict=False` cosine).
- **Structure-aware is the real win**: markdown headers propagated into chunk metadata/prefix (the `MarkdownParser` already finds headings — they just don't reach chunks); code chunking via **tree-sitter AST** (cAST evidence) instead of regex boundaries.
- Performance: sizing loops are quadratic (re-join + recount per word — `chunkers.py:65-92`); `ParentChildChunker` stores the **entire parent text in every child's metadata** (`hierarchical.py:137`) — store parent id + fetch on expand instead.
- Tokenizer: cl100k is GPT-3.5/4-era; offer o200k_base and make encoding a parameter.

---

## 7. Parsing

**Live:** txt (encoding fallback chain — nice), markdown (frontmatter *discarded*), HTML (stdlib), PDF (`pypdf` text-only, no per-page metadata). No docx/xlsx/csv/json, no OCR, no tables/layout from PDF.

**Good:** stdlib-only, honest scope, clean `DocumentParser` protocol to extend.

**SOTA 2026 is a tiered story:**
1. Born-digital simple → pypdf is acceptable; **PyMuPDF4LLM** (`to_markdown()`: headers/tables/multi-column) is the upgrade — mind the AGPL.
2. Structured/self-hosted → **Docling** (IBM, MIT; ~97.9% table-cell accuracy) or Marker.
3. Hard docs/scans → **VLM parsing at ~$1/1,000 pages** (Gemini Flash, 258 tok/page) or Mistral OCR.

**Do (minimum bar):** per-page metadata on PDF chunks (citation provenance — today only doc-level `page_count`); parse frontmatter into metadata instead of discarding; add csv/json/docx parsers; then a `DoclingParser` optional extra as the tier-2 default. VLM tier stays a cookbook recipe, not a dep.

---

## 8. Query layer

**Live:** HyDE, multi-query, decomposition, step-back, conversation rewriter, transform pipeline with dedup, 3 classifiers — all callback-delegated (the library never calls a model itself).

**Good:** this is a complete 2026 toolbox; callback delegation matches the SDK's no-hidden-LLM-calls philosophy; `query_transform_step` fusing variants via RRF is the right wiring.

**Improve:** `atransform` is unreachable (no async step factory); `DecompositionTransformer` drops the original query; `ContextualQueryTransformer` joins the **entire** history unbounded (needs a token cap); self-query/metadata-extraction transformer only makes sense after §2's metadata filtering lands.

---

## 9. Evaluation

**Live:** precision/recall/F1@k, MRR, NDCG (binary-relevance only), hit rate; `LLMRAGEvaluator` fully callback-delegated where **missing callbacks silently score 0.0**; A/B and batch tooling.

**Good:** pure-computation metrics with tests; A/B runner exists — rare in SDKs.

**SOTA 2026:** golden set of 50–200 real queries + **recall@k gating changes in CI** is the practice that matters (that's how you validate every change from this doc: chunk defaults, embedder, bm25s swap, reranker). Use `ir_measures` for metric correctness (graded NDCG); RAGAS-style judge metrics need actual judge prompts or explicit "not configured" errors instead of silent 0.0.

**Do:** fix silent 0.0 → raise/`None`; graded NDCG; add a small golden-set harness + CLI (`anchor eval run golden.jsonl`) — this also pairs with the skill-eval harness from the companion doc.

---

## 10. CLI

`index` and `query` are placeholders that count tokens and echo the query (`cli.py:66-120`). Once §1 lands, the CLI can be real in ~50 lines: ingest → embed → sqlite-vec store → hybrid query. Best demo surface the project has.

---

## Prioritized plan

**Wave 0 — correctness (small diffs, no design):** the 5 bugs in §0 + dedupe RRF/cosine + delete `ScoreReranker`. Guard with tests that would have caught them (explicit `top_k=10`, negative cosine, dim mismatch).

**Wave 1 — the missing layer:** `EmbeddingProvider` protocol + providers + batching; `filter` param through `VectorStore` protocol and backends; sync/async convergence (one retriever code path over the store protocols); Postgres reachable + HNSW by default; dimension declared and validated.

**Wave 2 — quality per stage:** bm25s + PT stemming; sqlite-vec backend (+FTS5 hybrid single-file); chunk defaults 256–400/0–50 + markdown-header propagation; per-page PDF provenance + frontmatter to metadata; contextual enrichment step (Anthropic-style prepend, LLM-callback-delegated, cheap with caching); golden-set eval harness; real CLI.

**Wave 3 — reach:** Docling parser extra; LanceDB backend; tree-sitter code chunking; BGE-reranker extra; late-interaction rewrite-or-delete decision.

Sources for every SOTA claim: see agent research reports (Chroma chunking eval, MTEB/MTEB-PT, Voyage/MongoDB docs, pgvector 0.8.6, sqlite-vec, bm25s paper, OmniDocBench/Docling benchmarks, reranker head-to-heads) — URLs embedded in `2026-08-25-sota-gap-analysis.md` companion and this doc's sections.
