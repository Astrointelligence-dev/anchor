# Budget enforcement no loop (usage limits) — pesquisa

**Date:** 2026-08-25 · **Baseline:** `dev` @ `995817f` · **Item:** backlog pós-Phase 5, P0 #2 (`docs/plans/2026-08-25-sota-upgrade-plan.md:108`)
**Método:** 2 frentes — (A) maquinário de budget existente no anchor, (B) SOTA 2026 de usage limits em 6 SDKs (fontes primárias).

---

## A. Estado atual do anchor

### O que existe e o que ele NÃO governa

- **`TokenBudget`** (`models/budget.py:23-60`) é alocação por source (SYSTEM/MEMORY/RETRIEVAL/TOOL…) com priority e overflow strategy — enforcement vive **inteiro no pipeline** (`_assemble_result`, `pipeline.py:389-455`) e roda **uma vez por turno**, na montagem do contexto do round 0. Rounds 2..N fazem `messages.append(...)` sem passar por budget nenhum (decisão deliberada da Phase 4, registrada em `sota-upgrade-plan.md:141` deviation c). O cap de turno é uma peça **nova**, não uma extensão do `TokenBudget`.
- **`TokenBudgetExceededError`** (`exceptions.py:30-31`): dead code confirmado — zero `raise` em src/, mas exportado e documentado (`docs/docs/api/exceptions.md:64-66`) como se fosse levantável. Já apontado na gap analysis (`gap-analysis-results.md:30`).
- **Claim do produto**: README vende "Token Budget Management: yes" como diferencial vs concorrentes (`README.md:39,86`); a própria gap analysis registra que "it holds at the pipeline layer but not in the agent loop" (`2026-08-25-sota-gap-analysis.md:66`). Este item fecha essa promessa.

### Accounting pronto no loop (pós event stream)

- `RoundUsage` por round: `prompt/completion` (provider), `tool_schema_tokens`/`tool_result_tokens` (tokenizer próprio), cache tokens. Fabricado em `_close_round` (`agent.py:721-740`); o acumulado `rounds` está completo exatamente em `agent.py:1286-1292`/`:1360-1366`, onde `RoundFinished` é emitido — **o ponto de check natural**.
- **Cegueira de provider**: só o Anthropic reporta usage no stream (`anthropic.py:528-546`); OpenAI/LiteLLM/Grok/OpenRouter/Ollama/Gemini emitem `StreamChunk` sem usage no caminho de stream — o único que o loop usa. `prompt/completion = 0` nesses providers (docstring de `RoundUsage`, `models.py:111-114`). Contadores por tokenizer (`tool_schema_tokens`, `tool_result_tokens`, `_messages_tokens`) funcionam em qualquer provider.
- `total_cost` nunca é populado no loop; pricing existe isolado em `llm/pricing.py` + `observability/cost.py`, nada liga os dois.

### Governança existente para reusar

- `max_rounds` (1 request LLM por round — request_limit e rounds são a mesma coisa no anchor) + `_maybe_final_round_notice` (o texto literal já diz "the tool budget is exhausted", `agent.py:67-70`) + `tool_choice="none"` no round final + `_should_run_tools` (round final não executa tools). **O padrão graceful de "avisar e concluir" já está construído** — um cap de tokens pode reaproveitá-lo por inteiro.
- `stopped_by: Literal["stop","max_rounds","max_tokens"]` (`models.py:130`) chega ao consumidor por `TurnFinished.diagnostics` e `last_turn`; precisa de um 4º valor.
- Callbacks não podem parar o turno (exceções engolidas, `agent.py:555-558`). `SubagentDefinition.max_rounds` é o único cap propagado a subagents.

---

## B. SOTA — como os SDKs param a execução

| SDK | Config | O que limita | Breach | Escopo subagents |
|---|---|---|---|---|
| **Pydantic AI** | `UsageLimits` por run (`request_limit=50` default, tool_calls/input/output/total_tokens, `cost_limit` USD, `per_request_input_tokens_limit`) | requests pré-flight; tokens pós-hoc (pré-flight opt-in via count-tokens); tool batch paralelo **projetado** — se excederia, nenhuma executa | exceção `UsageLimitExceeded` — run perdido (dor documentada: issue #1083, `capture_run_messages` como band-aid) | pool explícito: `usage=ctx.usage` soma no RunUsage do pai; limits do pai governam a árvore |
| **OpenAI Agents** | `max_turns=25` | turnos, pré-flight | exceção `MaxTurnsExceeded`; novidade: `error_handlers["max_turns"]` pode devolver `final_output` em vez de levantar | mesmo run/contador nos handoffs |
| **Claude Agent SDK** | `max_turns`, **`max_budget_usd`** (estimativa client-side) | turnos + USD, pós-hoc ("may exceed by up to one API call's worth") | **sem exceção: `ResultMessage.subtype="error_max_budget_usd"`** com estado/custo/sessão preservados; recovery = resume com limite maior. Caps de token não existem (issue #1024 aberta pedindo) | budget da árvore inteira; mata subagents em background no breach |
| **LangGraph** | `recursion_limit=1000` | supersteps | exceção `GraphRecursionError` por default; **`RemainingSteps`** managed value permite o graph terminar graceful com estado válido (o prebuilt react agent troca a resposta e encerra sem exceção) | graph inteiro |
| **Google ADK** | `max_llm_calls=500` | LLM calls, pré-flight | exceção `LlmCallsLimitExceededError` | pool implícito (InvocationContext compartilhado) |
| **smolagents** | `max_steps` | steps | **nunca exceção**: `provide_final_answer()` — uma chamada extra pós-limite pedindo a resposta final da memória | — |

**Sínteses transversais:**
1. **Exceção pura tem dor documentada** (Pydantic #1083: o run é jogado fora). O modelo Claude-style — parada graceful sinalizada em dado tipado com estado preservado — é o que encaixa num loop que já emite eventos e já tem `stopped_by`.
2. **Regra pré-flight/pós-hoc**: "conte o que já sabe antes de gastar; tokens só depois que chegam". Overshoot de até 1 round é assumido e documentado por todos que fazem pós-hoc.
3. **Ninguém shipped o wrap-up avisando o modelo** — smolagents (chamada extra pós-limite) e LangGraph (`RemainingSteps`) são os precedentes parciais. O final-round notice do anchor já é mais avançado que ambos; estendê-lo ao budget seria inédito entre os 6.
4. **Naming converge**: `*Limits`/`*_limit` para config, "budget" reservado a USD. `TokenBudgetExceededError` destoa da convenção (`UsageLimitExceeded` seria o canônico — ou nenhum erro, no caminho graceful).
5. **Defaults de segurança altos** contra loop infinito (50/25/500/1000) mesmo com todo o resto opt-in — o anchor já tem isso via `max_rounds=10`.

---

## C. Decisões de design em aberto (para o Arthur)

1. **Comportamento no breach**: (a) **graceful wrap-up** — breach detectado → próximo round vira round final (notice + `tool_choice="none"`, maquinário existente) → turno termina com texto usável e `stopped_by="usage_limit"` (custo: 1 chamada a mais, bounded por `max_response_tokens`); (b) hard stop imediato pós-round (estilo Claude SDK, sem gasto extra, turno pode terminar sem resposta); (c) exceção (estilo Pydantic, contra o precedente do próprio `max_rounds` que não levanta).
2. **Métrica do `total_tokens_limit`**: (a) somas de `RoundUsage` como estão — sub-conta feio em 6/7 providers (prompt/completion=0 no stream); (b) **fallback por tokenizer** — quando o provider não reporta, estimar prompt via `_messages_tokens(llm_messages)` e completion via `count_tokens(state.text)` — cap honesto cross-provider, muda a semântica documentada do RoundUsage (de "0 quando não reporta" para "estimado quando não reporta").
3. **Destino do `TokenBudgetExceededError`**: deletar (dead code, convenção errada) vs reaproveitar como erro de um modo `raise` opt-in.
4. **Escopo**: por turno do agent que roda (subagents têm seus próprios limits herdáveis?) vs pool da árvore (Pydantic/Claude style) — pool exige contador compartilhado via contextvar (padrão do `_EVENT_SINK` já existente).
5. **Campos**: `total_tokens_limit` + `tool_calls_limit` cobrem o item (rounds = `max_rounds` que já existe; `cost_limit` USD exigiria ligar `llm/pricing.py` ao loop — por demanda).
