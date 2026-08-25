# Usage limits no agent loop — plano

**Date:** 2026-08-25 · **Baseline:** `dev` @ `995817f` · **Research:** `docs/research/2026-08-25-usage-limits.md`
**Decisões do Arthur (2026-08-25):** (1) breach → **graceful wrap-up** (round final com notice + `tool_choice="none"`, maquinário existente) e `stopped_by="usage_limit"` — sem exceção; (2) **fallback por tokenizer** quando o provider não reporta usage no stream (cap honesto nos 7 providers); (3) `TokenBudgetExceededError` **deletado** (dead code, convenção errada); (4) escopo **por turno** do agent (pool de árvore = extensão P2).

---

## Contrato

**`UsageLimits`** (`agent/models.py`, frozen): `total_tokens_limit: int | None` (gt 0), `tool_calls_limit: int | None` (ge 0). Rounds já são governados por `max_rounds` (1 request LLM por round — request_limit seria redundante); custo USD por demanda (exigiria ligar `llm/pricing.py` ao loop).

**API**: `Agent.with_usage_limits(limits) -> Agent` (fluente, aditivo ao `__slots__`).

**Semântica do breach** (pós-hoc, padrão SOTA "conte o que sabe, tokens só depois que chegam"):
- Check único por round, após `RoundFinished`, **só quando o loop continuaria** (turno que termina sozinho não é "cortado").
- Breach → evento `UsageLimitReached(kind, used, limit)` + próximo round vira round final (`_FINAL_ROUND_NOTICE` — o texto já diz "the tool budget is exhausted" — + `tool_choice="none"`), tools desse round não executam, turno fecha com `stopped_by="usage_limit"`.
- Overshoot documentado: o round corrente completa (tools incluídas — precisam de results na conversa para o wrap-up ser request válida) + 1 chamada de wrap-up bounded por `max_response_tokens`. Consistente com Claude SDK ("up to one API call's worth").

**Métrica**: `total = Σ rounds (prompt + completion + cache_creation + cache_read)`. Com o fallback: `prompt == 0` → `_messages_tokens(llm_messages) + schema_tokens`; `completion == 0` → tokenizer sobre `state.text` + fragments de args. `tool_schema/tool_result_tokens` seguem como campos de visibilidade (subconjuntos do prompt — somá-los duplicaria). `RoundUsage` ganha `tool_calls: int` (conta o `tool_calls_limit` e melhora o accounting). Docstring do `RoundUsage` atualizada ("estimado quando o provider não reporta").

**`stopped_by`**: Literal ganha `"usage_limit"` (`agent/models.py:130`).

---

## Fase 1 — Enforcement + estimation

- [ ] `UsageLimits` + `with_usage_limits` + export (agent/__init__, anchor/__init__, test_exports)
- [ ] Evento `UsageLimitReached(kind: Literal["total_tokens","tool_calls"], used, limit)` em `events.py` + exports
- [ ] Fallback de estimation em `_round_usage` (prompt/completion via tokenizer quando 0) + `tool_calls` no `RoundUsage`
- [ ] Loop: `final_round = wrap_up or _is_final_round(...)`; `_maybe_final_round_notice(final_round, ...)` e `_should_run_tools(state, final_round)` passam a receber o bool; check pós-`RoundFinished`; `stopped_by="usage_limit"` quando wrap-up rodou
- [ ] Testes: breach de total_tokens → wrap-up round com notice + tool_choice none + `UsageLimitReached` no stream + `stopped_by="usage_limit"`; breach de tool_calls; turno que termina sozinho abaixo do limite não é afetado; breach no round `max_rounds-1` não ganha round extra; fallback de estimation (provider sem usage → prompt/completion > 0); provider com usage não é sobrescrito; `test_per_round_accounting_visible` atualizado à semântica nova

**Done when:** um agent com `UsageLimits(total_tokens_limit=N)` em qualquer provider (fake sem usage) para graceful com resposta final e `stopped_by="usage_limit"`; suite verde.

## Fase 2 — Limpeza + docs

- [ ] Deletar `TokenBudgetExceededError` (exceptions.py, exports, test_exports, `docs/docs/api/exceptions.md`, llms.txt se citado); breaking pre-1.0
- [ ] Guia do agent (mkdocs): seção "Usage limits" (with_usage_limits + wrap-up + evento); nota em `concepts/token-budgets.md` distinguindo budget do pipeline (alocação por source no build) de usage limits do loop (cap de turno); CHANGELOG
- [ ] Backlog do sota-upgrade-plan: marcar P0 #2

**Done when:** grep por TokenBudgetExceededError limpo; docs buildam; claim do README passa a ser verdade nas duas camadas.

## Fora de escopo (registrado)

- Pool de árvore com subagents (contador compartilhado via contextvar, padrão `_EVENT_SINK`) — P2, quando um caso multi-agent pedir
- `cost_limit` USD (ligar `llm/pricing.py` ao loop) — por demanda
- Pré-flight de tool batch estilo Pydantic (results de erro sem executar) — só se o overshoot pós-hoc doer na prática

## Review

(preencher pós-implementação — ritual: ponytail reviewer + adversarial judge)
