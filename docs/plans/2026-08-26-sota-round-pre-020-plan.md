# Rodada SOTA pré-0.2.0 — plano

**Date:** 2026-08-26 · **Baseline:** `dev` @ `fbf9509` · **Research:** `docs/research/2026-08-26-sota-round-pre-020.md`
**Decisões do Arthur (2026-08-26):** (1) HITL **inline agora** (callback segura o turno; durável registrado pós-ConversationStore); (2) structured output = **output tool forçada** default + prompted fallback (native Anthropic por demanda); (3) MCP = **fastmcp 3.x estável + gaps do bridge** (migração 4.x como follow-up); (4) memory tool = **AgentTool próprio multi-provider** + backend file-based; (5) retrieval: late-interaction **deletado**, Redis **dropado** (sem veto à recomendação).

**Rules of engagement:** uma fase por commit; testes que pegariam o que cada fase conserta; suite verde em todos os commits; ritual reviewer+juiz no diff acumulado ao final da rodada.

---

## Fase 1 — Approval/HITL inline

Contrato (estilo `can_use_tool` do Claude SDK, adaptado às convenções do repo):
- `ApprovalRequest(frozen)`: `tool_call_id`, `name`, `tool_input`. `ApprovalDecision(frozen)`: `approved: bool`, `reason: str | None`, `updated_input: dict | None`.
- `Agent.with_approval(callback)` — callback sync ou async (`ApprovalRequest -> ApprovalDecision | Awaitable`); async só no caminho `astream`/`achat` (sync com callback async → TypeError com orientação, padrão do guard MCP).
- Gatilhos: `AgentTool.requires_approval: bool = False` **ou** pre-hook retornando `HookResult(decision="ask")` (Literal ganha `"ask"`). Ordem: hooks primeiro; "ask"/flag roteiam ao callback.
- **Fail closed**: gatilho sem callback configurado = deny com reason explicativa.
- Deny → mecânica existente (`_error_result` → tool result `is_error` com a reason — o modelo se ajusta). Allow com `updated_input` → executa com o input novo.
- Ponto de execução: dentro de `_execute_call`/`_aexecute_call` após `_resolve_call` (o callback roda no frame da task; consumidor async fica suspenso no `__anext__` enquanto o approver decide — TUI mostra o diálogo no próprio callback).
- Deliberado (YAGNI): sem evento `ApprovalRequested` no stream (o approver É o dono do callback); sem timeout embutido (nenhum SDK tem — aplicação decide); pausa durável fica para pós-ConversationStore.

- [x] Models + `with_approval` + `"ask"` no HookResult + `requires_approval` no AgentTool + exports
- [x] Roteamento em `_execute_call`/`_aexecute_call`; fail-closed; updated_input
- [x] Testes: allow/deny/rewrite, ask via hook, fail-closed, callback async, tools paralelas com decisões distintas, reason chega ao modelo, sync+async

**Done when:** um agent com tool `requires_approval=True` pausa no callback, deny devolve reason ao modelo, allow executa — nos dois caminhos.

## Fase 2 — Structured output no run principal

Contrato (modo tool, default Pydantic AI):
- `Agent.with_output_model(model, *, mode="tool" | "prompted", max_output_retries=1)`.
- **Modo tool**: tool sintética `final_result` (schema = `model_json_schema()`) entra nos sendables; quando o modelo a chama, **não executa como tool** — valida os args: válido → loop para, output capturado; inválido → tool result de erro com o ValidationError (retry pela mecânica que o loop já tem), bounded por `max_output_retries`. Modelo que para em texto puro → um nudge de retry pedindo a tool; esgotado → `ValueError` (padrão dos subagents).
- **Modo prompted**: reusa verbatim `_schema_instruction`/`_validate_output`/`_retry_prompt` dos subagents.
- Consumo: `Agent.run(message) -> BaseModel` / `Agent.arun(message)` — consomem o stream internamente e retornam a instância validada; `TurnFinished` ganha `output: str | None` (JSON normalizado) e `agent.last_output`. `chat`/`achat` seguem intocados (agents de texto).
- Collision check: `final_result` vs tools existentes (padrão `task`/`search_tools`).

- [x] `with_output_model` + tool sintética + special-case no fluxo de tools + captura/validação/retry
- [x] `run`/`arun` + `TurnFinished.output` + exports
- [x] Testes: happy path, retry por validação, nudge de texto→tool, convivência com tools reais no mesmo turno, prompted mode, sync+async

**Done when:** `agent.with_output_model(Report).arun("...")` devolve `Report` validado com tools reais no meio do caminho.

## Fase 3 — MCP bridge gaps (fastmcp 3.x)

- [x] `MCPServerConfig.defer_tools: bool = False` → `defer_loading` no `mcp_tool_to_agent_tool` (wire de uma linha + teste)
- [x] Collision check de nomes MCP vs tools ativos no `_ensure_mcp` (hoje first-match-wins silencioso)
- [x] Prompts/resources agregados: `MCPClientPool.all_prompts()/get_prompt()/all_resources()/read_resource()` + acessores no Agent (`mcp_prompts()`, `mcp_get_prompt()`, `mcp_resources()`, `mcp_read_resource()`) — **app-facing, nunca automático no contexto** (padrão dos clients de referência); exposição model-facing fica por demanda
- [x] Atualizar pin para `fastmcp>=3.4,<4` (estável atual); follow-up 4.x registrado no backlog

**Done when:** um server com `defer_tools=True` não polui o prompt até o `search_tools` carregar; prompts/resources acessíveis pela API do pool.

## Fase 4 — Memory tool compat `memory_20250818`

- [x] `FileMemoryBackend(base_path)`: os 6 comandos com as strings de resultado de referência (view com line numbers 6-char, erros literais do spec); guards: prefixo `/memories` + canonicalização + contenção (padrão `_resolve_in_skill`), rejeição de traversal urlencoded, raiz protegida contra delete/rename, rename sem overwrite, truncação de view (16k chars + view_range)
- [x] `memory_tool(backend) -> AgentTool` (tool única `memory`, schema com discriminador `command`) + `Agent.with_memory_tool(backend)` que registra o tool E appenda a instrução de uso ao system suffix (multi-provider não ganha a injeção automática da API)
- [x] Testes: cada comando, traversal attempts, raiz, rename collision, round-trip com o loop (modelo chama via FakeLLM)

**Done when:** um turno com o memory tool cria/lê/edita arquivos sob o backend em qualquer provider; nenhum path escapa do base_path.

## Fase 5 — Smoke tests live

- [x] `tests/live/` no padrão pgvector (importorskip + `pytestmark` por env): Anthropic (`ANTHROPIC_API_KEY`): 1 turno E2E com tool + event stream + caching + tool_choice; structured output; memory tool. OpenAI/Gemini (gated pelas keys): 1 turno com tool cada
- [x] MCP live **sem credencial**: server fastmcp real via stdio subprocess — list_tools/call_tool/prompts/resources reais (roda no CI)
- [x] CI: job opcional documentado (env vars não injetadas por default — suites gated ficam para execução local/manual)

**Done when:** `ANTHROPIC_API_KEY=... pytest tests/live` passa localmente; o MCP live roda verde no CI sem credenciais.

## Fase 6 — Deleções de retrieval + docs da rodada

- [x] Deletar late-interaction retriever + testes (decisão: sem uso, reescrita não justificada); changelog
- [x] Dropar Redis do roadmap/docs (grep por menções); changelog
- [x] Guia do agent: seções Approval e Structured Output; guia MCP: prompts/resources + defer_tools; guia memory: memory tool; CHANGELOG consolidado da rodada

## Fora de escopo (registrado)

- HITL durável (turno pendente serializável) — pós-ConversationStore/storage-layer-gaps
- Native structured output Anthropic (`output_config` passthrough) — por demanda
- fastmcp 4.x / protocolo stateless 2026-07-28 — quando sair do beta
- Exposição model-facing de MCP resources (tools de list/read) — por demanda

## Review

### Shipped 2026-08-26 — rodada completa + condições (`78be97a..HEAD`)

**6 fases entregues** + condições do ritual. Suite: 2709 verdes (+16 testes de condição; 8 skips = 3 base + 5 live gated). Zero lint novo.

**Ritual** (ponytail reviewer + juiz adversarial): 13 findings do reviewer, **13/13 confirmados pelo juiz** (nenhum refutado) + 2 achados do sweep próprio do juiz. Destaques:

- **P1 (o grande)**: structured output × round final/wrap-up era falha garantida por construção — o `tool_choice="none"` do round final vencia o `"any"` e `_should_run_tools` recusava o round final, então `run()` levantava ValueError mesmo com modelo obediente (reproduzido nas 3 variantes). Fix validado por protótipo do juiz: round final com output pendente força `{"type":"tool","name":"final_result"}` (os 3 mapeadores suportam), `_should_run_tools` executa o round final quando todas as calls são `final_result` (terminal — o modelo nunca precisa ver resultado), e notice variante pede a call em vez de "do not call tools". Decisão pinada: output capturado durante wrap-up de usage limit mantém `stopped_by="usage_limit"` (o corte de budget fica visível).
- **P2s**: collision intra-MCP (`taken` congelado — dois servers com a mesma tool passavam); `run()` prompted + memory agora é recusado (precedente `_guard_clean_context` — schema/retries poluíam a conversa persistida); 3 seções de docs com API deletada removidas.
- **P3s aplicados em lote**: isinstance no retorno do approval callback (lixo virava AttributeError escapando o turno; agora deny fail-closed) + tipo `ApprovalCallback` apertado; `iscoroutine` no close do guard sync; reconfiguração de output model limpa estado; memory tool com try/except retornando strings de erro **sem vazar o base_path do host** + fix da lista de linhas multiline no str_replace + validação de view_range; `all_prompts`/`all_resources` best-effort (um server sem a capability não derruba o agregado); `state` morto removido do `_round_verdict`; `with_tools` protege o nome `memory` (achado N1 do juiz); concorrência de approvals documentada (Lock julgado seguro mas desnecessário — o app serializa no callback se quiser); testes de borda `used == limit` e awaitable não-corrotina.

**Corretos-mas-suspeitos registrados** (sem ação): `_output_failures` mistura nudges e validação num budget só; `ToolStarted` emitido antes da resolução do approval (formato pré-existente do stream); `as_tool`/`SubagentDefinition` sem `requires_approval` (YAGNI); nomes de server MCP duplicados no pool (N2 — nota).

### Shipped 2026-08-27 — review do próprio commit da rodada (`3464dea..79ddb13`)

`/code-review` de `3464dea` (o commit de condições acima): **14 findings, 14 confirmados por probe executado**. Três deles eram a mesma cadeia, criada pelo fix P1 da rodada anterior:

- **Round final deixou de ser terminal.** `_round_verdict` só parava o loop com `run_tools=False`, e a exceção nova ("round final executa `final_result`") tornou `run_tools=True` alcançável num round final. Um wrap-up de usage limit cujo `final_result` **não** capturava output rodava até `max_rounds`: 10 chamadas de LLM, 10 `UsageLimitReached` duplicados, e `_maybe_final_round_notice` re-anexando o mesmo aviso a cada round.
- **`max_output_retries` era letra morta.** `_make_output_tool` cria a tool sem `input_model`, então `validate_input` caía no `_basic_validate` e rejeitava os args **antes** de `record()` — o único lugar que incrementava `_output_failures`. O amplificador que tornava o item acima ilimitado.
- **`with_output_model` que levanta deixava o agent inconfigurável**: `_output_tool = None` rodava antes do collision check, sobrando `_output_model` setado, `_output_mode="tool"` e nenhuma tool pra satisfazer.

**Ritual (reviewer + juiz + discussão).** O juiz voltou APPROVE WITH CONDITIONS e pegou uma **regressão introduzida pelo próprio fix**: o parâmetro `final_round` que eu adicionei a `_round_verdict` colapsava duas condições com **contratos opostos** — usage limit (nunca levanta, decidido em 2026-08-25) e `max_rounds` (levanta alto) — e a colocação abaixo do raise fazia o wrap-up levantar `ValueError`, contradizendo a docstring do mesmo diff. Discussão de 4 pontos: o juiz moveu em 3 (retirou a recomendação de restaurar o raise ao descobrir que **raise dentro do `try` faz o `yield TurnFinished` nunca acontecer**; separou `input_model` do fix de contagem; retirou o argumento de "recuperar" na ambiguidade de duas `final_result`, mantendo só a crítica de colocação — que estava certa: o guard vale em qualquer round).

**Aplicado:**
- Terminação keyed no **wrap-up** (`stopped_by == "usage_limit"`), não em "round final" — parâmetro deletado, `max_rounds` mantém o erro alto. A duplicação do notice cai fora sozinha.
- Contagem de falha de output em **`_error_result`**, o funil por onde toda falha de `final_result` passa (schema, deny de hook, approval fail-closed) — e **por último**, para que um `_record_tool_call` que levante não conte duas vezes pelo retry do `_astream_tools`. `input_model=output_model` na output tool: **um validador só**, em vez de `_basic_validate` e `model_validate` com dois dialetos de erro pro mesmo schema.
- Duas `final_result` num round = ambiguidade recusada **em qualquer round** (antes só no último; ambas executavam e a segunda sobrescrevia `_last_output` em silêncio).
- MCP: `all_prompts`/`all_resources` viram um helper `_per_server` que re-levanta o que não é `Exception` — `CancelledError` deixava de ser cancelamento e virava "capability indisponível".
- Memory tool: scan não-sobreposto, `old_str` vazio recusado (o scan não avançava — loop infinito), `view_range` com a especificidade da referência, `except Exception` com log antes de sanitizar. **Docstring do módulo passa a nomear as 3 divergências** — a referência Anthropic não erra em `view_range` inválido e o scan dela se sobrepõe (bug real), então "paridade string-por-string" era falso.
- **`stopped_by="output_missing"`** (commit separado, breaking): `_round_verdict` não levanta mais, devolve veredito; `run()`/`arun()` viram o único ponto que levanta. Aplica ao caso "output nunca chegou" a convenção já decidida na rodada de usage limits.
- CLAUDE.md apontava pra `tasks/todo.md` e `tasks/lessons.md`, inexistentes → `docs/plans/`.

**SOTA lido via context7** (Pydantic AI, OpenAI Agents SDK, LangGraph, referência Anthropic, fastmcp): o wrap-up round do anchor segue **inédito** — os outros levantam, e o LangGraph para graceful mas descarta a resposta do modelo. Retry budget: Pydantic AI **separa** `retries={'tools': N, 'output': M}`; o anchor usa contador único. Best-effort do MCP bate com o `AggregateProvider` do fastmcp (e o anchor é mais estrito).

**Follow-ups registrados:** separar retry budget de tools vs output (estilo Pydantic AI) — `input_model` abriu a porta; `ProxyProvider` do fastmcp distingue `METHOD_NOT_FOUND` de "server quebrou", o anchor conflata (pré-existente).

Suite: **2723 verdes** (+14 testes de condição, cada um verificado falhando contra o estado anterior). ruff 157 / mypy 144 = baseline do HEAD, sem regressão.
