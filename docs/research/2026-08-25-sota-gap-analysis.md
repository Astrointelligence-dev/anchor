# SOTA 2026 Gap Analysis — anchor vs state of the art

**Date:** 2026-08-25 · **Baseline:** `dev` @ `4138306` (v0.1.1)
**Method:** full-repo exploration + two web-research passes over primary sources (Anthropic engineering/platform docs, agentskills.io spec, MCP spec changelog, framework docs). Supersedes `gap-analysis-results.md` (2026-02-20) for everything agent/skills-related.

---

## 1. Skills — the core comparison

### What anchor does today

- Skill = frozen Pydantic model (`agent/skills/models.py:12`): `name`, `description`, `instructions`, `tools`, `activation: always|on_demand`, `tags`. Two authoring paths: Python factory or `SKILL.md` directory.
- Exposure is two-tier: (1) discovery listing of on-demand skills appended to the system prompt each round (`registry.py:132`), (2) an `activate_skill` meta-tool that returns `instructions` as tool-result text and adds the skill's tools to the schema next round (`activate.py:14`, `agent.py:204`).
- This shape is **directionally correct** — it is the same pattern as Claude Code's Skill tool (listing in system prompt + tool-call invocation). The gaps are in the details below.

### SOTA 2026

Agent Skills is now an **open cross-vendor standard** ([agentskills.io](https://agentskills.io/specification), published Dec 2025; adopted by OpenAI/Codex, VS Code/Copilot, Cursor, Gemini CLI, Goose, Letta, JetBrains — 40+ products). Canonical model:

| Level | Loaded | Cost |
|---|---|---|
| 1 — metadata (`name`+`description`) | always, system prompt | ~100 tok/skill |
| 2 — SKILL.md body | on trigger | <5k tok (≤500 lines) |
| 3 — `references/` read on demand, `scripts/` **executed, never loaded** | as needed | ~0 until used |

Spec frontmatter: `name` (≤64, kebab, must match dir name), `description` (≤1024, third person, what + when + trigger keywords), `license`, `compatibility`, `metadata`, `allowed-tools`. Real YAML. Validation via `skills-ref validate`. Sources: [equipping agents](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), [best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices), [Claude Code skills](https://code.claude.com/docs/en/skills).

### Gaps (ranked)

1. **Progressive disclosure stops at Level 2, collapsed into one hop.** `loader.py:190` reads the SKILL.md body whole at load time; nothing else in the skill dir is ever read. No `references/`, no `scripts/`, no `assets/`, and no file-read tool that would make Level 3 possible. SOTA: bundled content is "effectively unbounded" because references load lazily and scripts run instead of loading.
2. **`instructions` of `always` skills are dead text.** Set by `memory_skill` (`memory/skill.py:30`), read only by `activate.py:46`, never surfaced to the model. Always-on guidance should be injected into the system prompt at build time (the CLAUDE.md analog).
3. **Frontmatter parser is not YAML** (`loader.py:28-57`) — hand-rolled `str.partition` per line. Quoted values, multi-line descriptions, and nested keys break silently. A description >1 line truncates at the first newline — and description quality is the single biggest trigger lever per Anthropic's best practices.
4. **Prompt-cache-hostile discovery.** The listing mutates per round (`" [active]"` marker), is space-concatenated onto the system text (`agent.py:395-402`), and `AnthropicFormatter` caching is hardcoded off (`agent.py:120`). SOTA discipline: stable prefix, append-only, explicit `cache_control` breakpoints ([Manus lessons](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Manus); cached input ~10× cheaper).
5. **Format incompatibility with the ecosystem.** anchor is *close* to the spec (name/description limits match) but diverges: `activation` is a nonstandard key, `tools.py` import is a nonstandard mechanism, and spec keys (`license`, `compatibility`, `metadata`, `allowed-tools`) are unknown. Full spec compliance would let anchor load the ~490k public skills (and superpowers) unchanged — and let anchor skills run in Claude Code/Codex.
6. **No namespacing for skill tools** — duplicate names across active skills raise `ValueError` mid-conversation (`registry.py:121`). MCP tools already get a `server_` prefix; skill tools should too, with collision checks at registration, not per round.
7. **`tools.py` import = arbitrary code execution at load time** (`loader.py:94-136`), no allowlist, no sandbox. The spec's answer is `scripts/` executed via a bash tool under `allowed-tools` gating, not import-time execution.
8. **No deactivation, no listing budget.** Registry supports `deactivate`/`reset` but the loop never calls them. Claude Code caps the skill listing at 1% of the context window and drops least-used descriptions on overflow; on compaction it re-attaches only the first 5k tokens per invoked skill.
9. **No skill evals.** SOTA is eval-driven skill development (≥3 scenarios, baseline with/without skill, description trigger tuning) — [evaluating skills](https://agentskills.io/skill-creation/evaluating-skills). This is also already on astro-platform's roadmap ("skill testing framework").
10. **Model default mismatch:** `Skill.activation` defaults to `"always"` (`models.py:44`) but file-loaded skills default to `"on_demand"` (`loader.py:174`).

---

## 2. Tool loading & management

**Live:** all tools (base + active skills + `activate_skill` + MCP) fully loaded into every request; recomputed per round; O(n) linear scan per tool call; sequential execution even in `achat` (`agent.py:268`); exceptions flattened to `"Error: tool 'name' failed."` with zero diagnostic (`agent.py:236`); no timeout, retry, approval hook, or `input_examples`.

**SOTA** ([advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use), beta `advanced-tool-use-2025-11-20`):
- **Tool Search Tool / `defer_loading: true`** — only a ~500-token search tool upfront; 72k→8.7k tokens (−85%), MCP accuracy Opus 4.5 79.5%→88.1%. Recommended above ~10 tools or 10k tokens of defs. anchor's skills activation is a coarse version of this; the per-tool deferred variant is the standard.
- **Programmatic Tool Calling** (`allowed_callers`) — model writes code orchestrating tools in a sandbox; intermediate results never enter context (−37% tokens). [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp): 150k→2k (−98.7%).
- **Tool Use Examples** (`input_examples`): complex-param accuracy 72%→90%.
- Frameworks: OpenAI Agents SDK static/dynamic MCP tool filters; Pydantic AI toolsets + deferred approval tools; LangGraph middleware; Google ADK tool confirmation.

**Gap:** `ToolSchema` (`llm/models.py:106`) has no `cache_control`, no `strict`, no examples, no defer flag; no tool-choice control anywhere.

---

## 3. Context pipeline, budgets, caching

**Live:** the pipeline is real and mature (priority assembly, `TokenBudget`, per-source diagnostics) but **the Agent opts out of all of it**: `pipeline.build()` runs once per user turn; rounds 2..N append to a plain list bypassing budget/priority/formatter; `TokenBudget` never attached; tool schemas never token-counted (`SourceType.TOOL` unused); tokenizer replaced by a whitespace word counter (`agent.py:38`); caching off.

**SOTA:**
- **Server-side context editing** — `clear_tool_uses_20250919`, `clear_thinking_20251015` ([context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)); `clear_at_least` exists specifically to make edits worth the cache invalidation.
- **Server-side compaction** — `compact_20260112` (trigger default 150k, `pause_after_compaction`); Anthropic now recommends it over client-side compaction ([compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)). Memory tool + context editing measured at **84% token savings, +39% performance** on a 100-turn benchmark.
- **KV-cache-aware layout** as first-class discipline: stable prefix, deterministic serialization, breakpoints. MCP 2026-07-28 requires deterministic `tools/list` ordering *for cache hits*.

**Gap:** anchor's differentiator claim is "token budget management" — it holds at the pipeline layer but not in the agent loop, and the frontier moved to server-side editing the SDK should expose (native for Anthropic, emulated for others).

---

## 4. Memory

**Live:** the strongest module — multi-tier managers, scored retrieval, decay, consolidation, graph memory, GC. Progressive summarization (4-tier cascade) was specced (superpowers-era doc, removed 2026-08-25 — recoverable at git `b469b31`) but never built. Storage-layer gaps (ConversationStore etc.) in flight on the worktree.

**SOTA:** tiered consensus (in-context core + retrieval-backed store + explicit forgetting) — anchor matches. The delta: **Anthropic memory tool compatibility** (`memory_20250818`: `view/create/str_replace/insert/delete/rename` under `/memories`, path-traversal guards) so anchor memory can serve as the client-side store for API-native memory ([memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)); file-based memory (CLAUDE.md/NOTES.md pattern) as a first-class backend; Letta-style sleep-time consolidation as the ceiling.

---

## 5. Multi-agent

**Live: absent.** No subagent spawn, no context isolation, no structured returns, no handoffs. `_arun_tools` doesn't even parallelize independent calls.

**SOTA:** orchestrator owns full context + ephemeral isolated subagents returning condensed structured summaries (1-2k tokens) — Anthropic's research system measured **+90.2%** over single-agent on breadth-first tasks ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)); Cognition's counterpoint (shared-state coding tasks) refined *when*, not *whether*. All major frameworks converged on this shape.

**This is the sharpest irony in the repo:** MULTI_AGENT.md — subagent isolation + asymmetric returns — is Astro's stated core IP, and the SDK doesn't implement it. An `agent-as-tool` primitive (spawn with clean pipeline, restricted tools, JSON-schema'd return) is the highest-leverage new feature after skills.

---

## 6. RAG

**Live:** hybrid dense+BM25+RRF + rerankers (FlashRank extra) — the 2026 *baseline*, genuinely competitive. `search_docs` skill tool hardcodes `top_k=5` and truncates results to 500 chars.

**SOTA deltas:** cross-encoder reranking is the single biggest lever (Recall@5 0.816 vs 0.695 RRF-alone); **contextual retrieval** (LLM-generated chunk context before embedding; −49% failure with contextual BM25, −67% with reranking, cheap via prompt caching — [contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)) is missing from ingestion; the 2025-26 shift is **agentic retrieval** — grep/filesystem navigation + just-in-time loading as an alternate path for code/doc corpora; vector RAG is one tool the agent calls, not the pipeline.

---

## 7. MCP

**Live:** bidirectional FastMCP 3.0 bridge, protocols, pooling, prefixing, caching — solid March 2026 work. But: prompts/resources are listed yet never wired into context; `from_agent` re-exposes only base tools (not skill/MCP tools); `expose_tool` discards the computed schema; no deferred loading.

**SOTA:** MCP **2026-07-28 RC is a major rewrite** — stateless (no initialize/session), `server/discover`, Tasks as extension, Roots/Sampling/Logging deprecated, `ttlMs`/`cacheScope` hints, deterministic tool ordering ([changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)). fastmcp pin `>=3.0,<4` needs watching; the bridge should adopt deterministic ordering + defer_loading for MCP tools now.

---

## 8. Harness (loop, errors, hooks, permissions)

**Live:** no hooks, no permission/approval seam, no per-tool timeout, no retry, silent `max_rounds` fall-off, tool errors without diagnostics, `chat`/`achat` ~70 duplicated lines, observability module exists but is not wired into the tool loop.

**SOTA:** hook points around model/tool calls (Claude Agent SDK `PreToolUse`; LangGraph middleware; ADK plugins), layered permissions, execution-based verification ("the harness is the agent"), durable execution options. Minimum bar for anchor: pre/post tool-call hooks, error text forwarded to the model, timeout, `asyncio.gather` for independent calls, one shared loop body.

---

## Prioritized roadmap

**P0 — Skills to spec (the interop play):**
1. Real YAML frontmatter + agentskills.io field set; keep `activation` as an anchor extension under `metadata`. Directory name = skill name validation.
2. Level 3 disclosure: skill-scoped `read_reference` tool (lazy `references/`), `scripts/` executed via an opt-in exec tool gated by `allowed-tools` — kill the `tools.py` import-at-load path (or keep it as explicit trusted-Python escape hatch).
3. Inject `always`-skill instructions into the system prompt at build; on-demand listing as a stable, newline-formatted block (no per-round mutation) with caching enabled and a listing token cap.
4. Namespace skill tools (`skill_name__tool`), collision check at registration.
5. Skill eval harness (with/without baseline, trigger tuning) — already on the platform roadmap.

**P1 — Agent loop to 2026:**
6. Subagents: `agent-as-tool` with clean context + JSON-schema condensed returns (MULTI_AGENT.md, implemented).
7. Deferred tool loading + tool-search meta-tool once tool count grows; `input_examples` on `AgentTool`/`ToolSchema`.
8. Budget integration: real tokenizer in the agent, `SourceType.TOOL` items, `TokenBudget` attached, rounds 2..N through the pipeline (or an explicit append-only cache-stable path — but measured).
9. Loop hygiene: shared chat/achat body, parallel tool exec, error forwarding, timeouts, pre/post hooks.

**P2 — Platform:**
10. Anthropic provider: context editing + server-side compaction + prompt caching flags surfaced; emulation path for other providers.
11. Memory-tool command-set compatibility; file-based memory backend.
12. Contextual retrieval in ingestion; agentic (grep-style) retrieval path.
13. MCP 2026-07-28 readiness: deterministic ordering, defer_loading, stateless transport tracking.
