# v0.2 · #1 — Orçamento e limites compartilhados numa run multi-agente

**Status:** não iniciado · **Tamanho:** pequeno · **Depende de:** nada
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

- [ ] O pool é objeto explícito (`UsageBudget` mutável, injetado) ou o pai
      passa o remanescente por valor a cada `task`?
- [ ] `tool_calls_limit` também é compartilhado, ou só tokens?
- [ ] O custo em USD (`CostTracker`) entra no mesmo pool ou fica separado?
- [ ] Diagnóstico: `TurnDiagnostics` do pai passa a incluir os rounds do
      filho, ou ganha um campo `children`?

---

## Escopo

- [ ] Pool compartilhado atravessando `task`/`as_tool`
- [ ] Interseção pai∩filho no `SubagentDefinition`
- [ ] Wrap-up do filho quando o pool estoura no meio dele
- [ ] Contabilidade correta com subagentes concorrentes
- [ ] Diagnóstico agregado visível no `TurnFinished` do pai
- [ ] Testes: pai 50k + 5 filhos → para no limite, não em 300k

**Fora:** custo em USD como limite duro (é outro eixo), rate limiting.

## Verificação

- Teste que reproduz o vazamento de hoje e falha contra o `HEAD` atual
  (mutação obrigatória — o bug tem que ser demonstrável antes do fix).
- Suíte completa verde.
- Live: um agente real com 2 subagentes e limite apertado, provando que o
  `stopped_by` sai correto e o total gasto respeita o pool.

## Review

_(preencher ao final)_
