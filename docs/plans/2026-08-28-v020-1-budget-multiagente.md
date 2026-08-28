# v0.2 · #1 — Orçamento e limites compartilhados numa run multi-agente

**Status:** implementado (sessão 6, 2026-08-28) · **Tamanho:** pequeno · **Depende de:** nada
**Sessão:** abrir com pesquisa, depois implementar.

---

## Por que este é o primeiro

O orçamento de tokens é o diferencial do anchor. E ele **vaza exatamente onde
o custo explode**: na fronteira do subagente.

`UsageLimits` é documentado como *per-turn*, aplicado pelo loop de um `Agent`
([models.py:141](../../src/anchor/agent/models.py)). Cada subagente é um
`Agent` próprio, com os próprios limites
([subagent.py:188](../../src/anchor/agent/subagent.py)). O orquestrador só vê
o **texto que voltou** — nunca os tokens que o filho queimou.

```
orquestrador  total_tokens_limit=50_000
  └─ task → 5 subagentes × 60k tokens = 300k gastos
     o limite do orquestrador nunca dispara
```

Não é detalhe de orquestração. É a promessa central do produto furada.

---

## Contexto (o que já existe)

| Peça | Onde | Estado |
|---|---|---|
| `UsageLimits(total_tokens_limit, tool_calls_limit)` | `agent/models.py:141` | por turno, por agente |
| Wrap-up gracioso + `stopped_by="usage_limit"` | `agent/agent.py` | ✅ funciona bem |
| `RoundUsage` por round | `agent/models.py` | prompt/completion/cache/schema/result |
| `TurnDiagnostics.rounds` | `agent/models.py:157` | acumulado do turno |
| Subagente como tool | `agent/subagent.py:188` | `_make_subagent_tool` |
| `_guard_no_nesting` | `agent/subagent.py:55` | subagente não gera subagente |
| `CostTracker` | `observability/cost.py` | custo em USD, separado dos limites |

---

## Pesquisa a fazer (abrir a sessão com isto)

1. Como OpenAI Agents SDK, Claude Agent SDK e PydanticAI v2 (redesign
   "harness-first", jun/2026) tratam orçamento atravessando handoff/subagente?
   Existe pool compartilhado ou cada agente é uma ilha?
2. O Claude Agent SDK tem `max_budget_usd` e `task_budget` em
   `ClaudeAgentOptions` — como o budget desce para subagentes lá? É o
   precedente mais próximo.
3. Semântica quando o pool estoura **dentro** de um filho: o filho faz wrap-up
   e devolve parcial, ou o pai aborta? O que a literatura de doom-loop /
   iteration cap recomenda?
4. Contabilidade concorrente: no caminho async as tools rodam em paralelo
   (`_astream_tools`), então dois subagentes debitam do mesmo pool ao mesmo
   tempo. Padrão para isso sem lock global?

---

## Decisões já tomadas

- O pool é **compartilhado, não replicado**: um orçamento para a run inteira,
  debitado por qualquer agente que gaste.
- Limite de filho **só estreita**, nunca alarga. O efetivo é a interseção com
  o do pai — senão o filho escapa do próprio confinamento.
- Estourar não levanta exceção: mantém o contrato atual de
  `stopped_by="usage_limit"` com wrap-up.

## Decisões em aberto

Fechadas com o Arthur em 2026-08-28, após a pesquisa:

- [x] **Pool é objeto mutável compartilhado** (não valor remanescente) —
      padrão unânime no SOTA (OpenAI `context.usage`, Pydantic `ctx.usage`,
      Claude SDK pool da árvore). Interno (`_UsagePool` + contextvar espelho
      do `_EVENT_SINK`); a superfície pública continua `UsageLimits`.
- [x] **`tool_calls_limit` também compartilhado** — Pydantic AI compartilha
      tudo do `RunUsage`; `RoundUsage` já carrega `tool_calls`.
- [x] **USD entra no pool agora** (decisão do Arthur contra a recomendação
      de adiar): `UsageLimits.cost_limit` estilo Pydantic v2, preços via
      `genai-prices` como extra opcional `[pricing]`. Modelo sem preço
      conhecido: warn uma vez + custo 0 naquele round (não silencioso, não
      levanta no meio do turno). O "Fora: custo em USD" abaixo cai.
- [x] **Diagnóstico via campo `children`** no `TurnDiagnostics` — rounds do
      pai continuam só dele; `children` traz tool_call_id, nome e o
      `TurnDiagnostics` de cada turno de filho, com `stopped_by`
      machine-visible (anti doom-loop de respawn).

### Achados da pesquisa que amarram o design

- Zero locks no débito em todos os SDKs (atomicidade do event loop único);
  invariante: nunca `await` entre check e débito. `threading.Lock` só quando
  tools sync migrarem pra thread pool (precedente em `cost.py:88`).
- Overshoot limitado (check antes, débito depois; pior caso N filhos × 1
  round) é prática estabelecida; reserva antecipada (LiteLLM) gerou bugs
  reais de reservas fantasma — evitar.
- Wrap-up gracioso no filho com parcial ao pai = lado novo do campo
  (smolagents, Claude SDK, Anthropic eng); Pydantic/OpenAI levantam e matam
  a árvore, com usuários pedindo o parcial. O veredito do filho precisa ser
  machine-visible no tool result, senão o pai respawna em pool vazio.
- Claude SDK: filho não estreita budget (só maxTurns/model/effort) — a
  interseção pai∩filho do anchor vai além do precedente. Cap duro em tokens
  é o pedido #1 da comunidade lá (issue #1024).

---

## Escopo

- [x] Pool compartilhado atravessando `task`/`as_tool` — `_UsagePool` +
      contextvar `_USAGE_POOL` (espelho do `_EVENT_SINK`); débito em
      `_close_round`, check em `_check_usage_limits` (pool primeiro)
- [x] Interseção pai∩filho — `SubagentDefinition.usage_limits` (limites
      próprios do filho valem por cima do pool; `scope="turn"` vs `"run"`)
- [x] Wrap-up do filho quando o pool estoura no meio dele — e filho que
      **começa** com pool estourado vai direto pro wrap-up (1 round);
      marcador `[partial result]` no tool result + retry de output_model
      suprimido em pool vazio (anti doom-loop)
- [x] Contabilidade correta com subagentes concorrentes — lock-free
      (débito síncrono, sem await entre check e débito; padrão SOTA)
- [x] Diagnóstico agregado — `TurnDiagnostics.children` (`ChildTurn` com
      tool_call_id, nome e diagnostics) + `run_total_tokens`/
      `run_total_tool_calls`/`run_total_cost_usd`
- [x] Testes: 13 novos em `tests/test_agent/test_shared_budget.py`
- [x] **Adicionado por decisão do Arthur:** `cost_limit` USD no pool via
      `genai-prices` (extra `[pricing]`); modelo sem preço → warn 1× + $0

**Fora:** rate limiting.

## Verificação

- [x] Teste que reproduz o vazamento e falha contra o `HEAD` — executado
      antes do fix: `AssertionError: child spend never tripped the
      parent's limit`.
- [x] Suíte completa verde: 2802 passed (13 novos), ruff 157 = baseline,
      mypy 145 = baseline.
- [x] Live (claude_cli, sem API key): orquestrador + 2 subagentes,
      limite 2500 → breach `scope=run used=3922`, `stopped_by=
      "usage_limit"`, filho visível em `children` (2200 tokens),
      `run_total_tokens=5038` vs 2838 próprios; segundo subagente nunca
      spawnou porque o pool acabou.

## Review

_(preencher ao final)_
