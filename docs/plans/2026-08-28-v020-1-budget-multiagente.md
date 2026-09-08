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

**Sessão 6 (2026-08-28), 2 commits: `cd8d4b7` (feature) + fix round do ritual.**

### Ritual (10 ângulos /code-review xhigh + juiz adversarial com 9 probes)

Veredito do juiz: **APPROVE WITH CONDITIONS**. 15 findings confirmados
reportados; os que mudaram código:

1. **Pool ambiente vazava (HIGH, 4 sintomas provados por execução)** — o
   contextvar era setado dentro do generator body, mutando o contexto do
   *consumer* pelo turno inteiro. Stream abandonado → agente sem limites
   virava "1 round + usage_limit" para sempre; streams intercalados
   cross-debitavam; o finalizer async levantava `ValueError` no reset
   cross-context **pulando o `_finish_turn`**. Fix (shape do juiz, menor
   que o meu): pool em slot (`_active_pool`) + contextvar publicado só na
   janela de cada tool call, colado no `_EVENT_SINK` — deleta Token,
   `_pool_exit` e a linha frágil do finally.
2. **`run()` prompted re-armava o budget por retry** (2 pools, 328+379
   contra limite 100 no probe) — `_prompted_run_pool()` segura um pool
   pelo loop de retries (frame normal, sem generator).
3. **`calc_price` com cache tokens levantava `ValueError`** (genai-prices
   trata cache como subconjunto do input; Anthropic exclui) e subprecificava
   ~54% — mapeamento cache-inclusivo + `except Exception`.
4. **Reuse perdido (ponytail rung 2)**: o repo já tinha
   `llm/pricing.py` (`MODEL_PRICING` + `calculate_cost`, API pública com
   override) — o genai-prices virou *fallback* atrás dela num entrypoint
   único (`estimate_round_cost`), com `provider_id` (ids multi-segmento
   litellm/openrouter precificam) e normalização de sufixo de data.
5. **Custo reportado pelo provider era jogado fora** — `Usage.total_cost`
   (claude_cli manda o billed real) agora vence a tabela.
6. **Nesting guard tinha bypass por ordem de registro** — re-check no
   call time dos 4 dispatchers; árvore de 3 níveis morre com tool error.
7. **`scope` fluía da config do pai** — filho nunca cria pool (papel via
   `_EVENT_SINK`); limites próprios de filho são sempre `scope="turn"`;
   `pool.owner is self` substitui a identidade frágil de limits.
8. **Filho que morre mid-turn sumia do `children`** (pool debitado,
   diagnóstico não) — forward sintético do `TurnFinished` a partir de
   `sub.last_turn`; cobre provider error e timeout.
9. Gate de `ImportError` do cost_limit virou warning (MODEL_PRICING é
   baseline); mensagem de retry-suppression neutra (vale pro `run()` do
   usuário); ordem own-limits→pool no check (o corte mais estreito é o
   reportado); docs reescritos (guides/agent.md afirmava o inverso do
   contrato novo), CHANGELOG com entrada breaking; ~30 linhas de
   duplicação sync/async removidas (helpers `_partial`/
   `_guard_retry_budget`/`_child_sink`).

**Refutados pelo juiz (probes)**: mesmo objeto `UsageLimits` em pai+filho
é inofensivo; contabilidade exata no happy path (303==303); smart union
int|float preserva tipos; `model_copy` em frozen ok; pool não vaza entre
turnos completos.

**Aceito sem fix (documentado em docstring)**: filho que *começa* em pool
exausto não emite evento próprio (o breach já foi emitido por quem
exauriu); filho estruturado cortado que emite JSON válido volta limpo —
o veredito vive em `ChildTurn.diagnostics.stopped_by` (marcador
corromperia o contrato JSON); wrap-up sem notice para filho sem tools.

### Lições

- **Ponytail rung 2 falhou uma vez**: implementei pricing novo sem
  grepar por tabela existente (`llm/pricing.py` era pública e
  documentada). O reuse-angle pegou; custo: retrabalho da frente de custo
  inteira no fix round.
- **Contextvar + generator é armadilha conhecida agora**: setar var em
  generator body muta o contexto do caller. O padrão seguro do repo é o
  do `_EVENT_SINK` — set/reset no mesmo frame, sem yield entre eles.
  Estado de turno vive em slot, não em var ambiente.
- O fix do juiz era menor que o meu plano (role-based): slot + janela de
  tool call. O seam certo deletou código.

### Follow-ups registrados

- `ToolResult.metadata` como canal estruturado para veredito de filho
  (mataria o marcador textual e o gap do JSON) — decisão do Arthur.
- `FallbackProvider.model_id` reporta o primário — rounds servidos pelo
  fallback precificam errado na tabela (mitigado quando o provider
  reporta custo).
- Reentrância de subagente compartilhado (2 `task` concorrentes no mesmo
  filho compartilham `_last_turn`/`_child_turns`) — pré-existente,
  agora user-visible via `children`.
- Custo do agent loop não alimenta o `CostTracker` (duas superfícies de
  custo desconectadas).
- Compactação não debita o pool (a chamada de summarize é invisível;
  o resumo conta no prompt seguinte) — herdado do per-turn.
- `uv.lock` deste commit varreu entradas atrasadas do claude-cli
  (re-lock mecânico; anotado pelo conventions-angle).
