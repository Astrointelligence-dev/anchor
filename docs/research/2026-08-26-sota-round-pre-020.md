# Rodada SOTA pré-0.2.0 — pesquisa (HITL, structured output, MCP, memory tool, smoke tests)

**Date:** 2026-08-26 · **Baseline:** `dev` @ `fbf9509` · **Escopo (Arthur):** os 4 itens antes da release 0.2.0.
**Método:** 3 frentes — (A) SOTA HITL + structured output (5 SDKs, fontes primárias), (B) specs MCP 2026-07-28 + memory tool Anthropic, (C) mapa interno das 4 frentes com file:line.

---

## 1. Approval/HITL

### SOTA
- **Dois modos, todos convergiram em oferecer ambos**: *inline* (callback async bloqueia o turno em voo — Claude SDK `can_use_tool`, "can stay pending indefinitely"; OpenAI `on_approval_request` p/ hosted MCP) e *durável* (turno termina com pendências tipadas; retomada é chamada nova — Pydantic AI `DeferredToolRequests` na union do output + `message_history`; OpenAI `RunState.to_string()/from_json`; LangGraph `interrupt()`+checkpointer; Claude hooks `defer` + resume de sessão).
- **Deny é unânime**: vira tool result com mensagem custom ("tells the model why, so it avoids retrying"); padrão de produto "suggest alternative".
- Marcação da tool: `requires_approval=True` (Pydantic) / `needs_approval` bool-ou-callable com **fail closed** quando args não inspecionáveis (OpenAI).
- Claude SDK: `PermissionResultAllow(updated_input=...)` reescreve args; ordem de avaliação hooks → deny rules → ask rules → mode → allow rules → callback; hook `PreToolUse` com `permissionDecision: allow|deny|ask|defer`.
- Timeout de aprovação: nenhum SDK embute — responsabilidade da aplicação; o SDK garante cancelamento limpo.
- Ninguém emite "evento de aprovação e continua streamando" — ou o stream termina com as pendências, ou o callback roda fora de banda. (Responses API modela como par de items tipados na conversa — o mais próximo de evento de 1ª classe.)

### Anchor (mapa)
- `HookResult.decision` é `Literal["allow","deny"]` (hooks.py:32) — sem "ask"; `PreToolHook` não recebe `tool_call_id` nem o AgentTool.
- Deny → `_error_result` → `ToolResult(is_error=True)` — a mecânica de "deny vira tool result" **já existe**.
- Suspensão: sync tem ponto natural (`yield ToolStarted` antes de `_execute_call`, agent.py:823/829); async emite todos os `ToolStarted` antes de criar as tasks (agent.py:862-868) — um ask por-call exige interleaving.
- **Pausa durável é inviável hoje**: `messages` é local do generator; memory guarda resumo truncado (200 chars) sem ids/args — retomada real = ConversationStore (storage-layer-gaps).

## 2. Structured output no run principal

### SOTA
- **3 estratégias**: (1) prompted+validate+retry — universal, menos confiável, fallback explícito em todo SDK; (2) **output tool forçada via tool_choice** (default Pydantic AI) — portável a qualquer modelo com tool calling, "chamou a output tool e validou" é condição de parada natural, retry vira tool result de erro (mecânica que o loop já tem); precisa de política p/ "output tool + function tools na mesma resposta" (`end_strategy`); (3) **native json_schema** (default OpenAI Agents; **Anthropic GA**: `output_config.format={"type":"json_schema",...}`, combinável com tools na mesma request, sem beta header) — garantia mais forte, mas subset de schema por provider + latência de compilação; SDKs validam no cliente contra o schema original mesmo assim.
- Regra consolidada 2026: native onde o provider suporta com tools; tool forçada como caminho portável; prompted como último fallback. Retry de validação continua valioso mesmo com native (validação semântica).

### Anchor (mapa)
- Maquinário dos subagents 100% reutilizável (funções livres: `_schema_instruction`/`_validate_output`/`_retry_prompt`, subagent.py:80-115).
- `tool_choice={"type":"tool","name":...}` já suportado ponta-a-ponta nas 3 famílias (anthropic passthrough, openai function-form, gemini ANY+allowed) — **nenhum caller usa ainda**.
- Encaixe do retry: o ponto `if not run_tools: break` (modelo terminou sem tools); `_should_run_tools` executaria uma output tool como tool comum — precisa de special-case.

## 3. MCP 2026-07-28

### Spec (mudou muito)
- Protocolo **stateless**: `initialize` removido; `_meta` obrigatório com protocolVersion+clientCapabilities por request; `server/discover`; `subscriptions/listen` no lugar do SSE/subscribe; MRTR substitui requests server-initiated (elicitation/sampling/roots — os dois últimos **deprecados**); `resultType` obrigatório; SSE resumability removida; `ttlMs`/`cacheScope` obrigatórios nos list results; **ordering determinístico virou SHOULD (server-side)** — explicitamente por prompt cache. **Não existe deferred loading no protocolo** (é feature Anthropic; o `search_tools` client-side do anchor cobre).
- Clients de referência: prompts = slash commands (user-controlled), resources = @ mentions/attachments (application-controlled) — **nunca automático no contexto**.
- **fastmcp**: anchor pinado em 3.1.0 (era legacy 2025-11-25, `mcp` 1.26.0); estável atual 3.4.7; suporte 2026-07-28 só na 4.x **beta** ("expect sharp edges"); interop funciona via servidores dual-era.

### Anchor (mapa)
- `defer_loading` NÃO é setado no bridge (tools.py:41-46) — wire de uma linha + config.
- Ordering já determinístico de fato (lists preservadas, gather preserva ordem) — falta collision check p/ nomes MCP (first-match-wins silencioso, `_find_tool`).
- Prompts/resources: **implementados no bridge (client.py:158-216), zero chamadores** — pool não agrega, nada chega ao app.
- Timeout MCP: `MCPServerConfig.timeout` só age no client fastmcp; `AgentTool.timeout=None` → cap efetivo é o global do Agent.

## 4. Memory tool (`memory_20250818`)

- **GA sem beta header**; única versão; modelos Claude 4+. Declaração: `{"type":"memory_20250818","name":"memory"}`; execução é 100% client-side.
- 6 comandos com shapes exatos (view c/ view_range, create, str_replace, insert, delete, rename) + **strings de resultado de referência** documentadas (números de linha 6-char right-aligned, mensagens de erro literais) — spec completo no relatório do subagent.
- Guards obrigatórios: path traversal (`/memories` prefix + canonicalização + contenção, incl. URL-encoded), caps de tamanho, expiração. `delete`/`rename` da raiz rejeitados; rename não sobrescreve.
- SDK anthropic 0.86.0 (no venv) já traz `BetaAbstractMemoryTool` + `BetaLocalFilesystemMemoryTool` — mas são Anthropic-only (namespace beta do SDK).
- Pareamento desenhado com context editing: `clear_tool_uses` + `exclude_tools:["memory"]`; API injeta system prompt automático quando o tool nativo está presente (multi-provider precisa da instrução própria).
- Anchor: nenhum precedente de tool com subcomandos (padrão é um tool por operação — `memory_tools` tem 4); guard de path pronto em `_resolve_in_skill` (resources.py:26-52: `.resolve()` + contenção); `JsonFileMemoryStore` é entry-based, não file-based.

## 5. Smoke tests live

- Único env-gated do repo: pgvector (`importorskip` + `pytestmark skipif` por `ANCHOR_TEST_POSTGRES_DSN`, teste único E2E, sem fixtures). **Nenhum teste toca API de provider**; caching/tool_choice/context_management verificados só em call_kwargs. CI não injeta credenciais (suites gated nunca rodam lá).
- Oportunidade sem credencial: MCP live contra um server fastmcp real via stdio (subprocess local) — roda até no CI.

## 6. Decisões em aberto (para o Arthur)

1. **HITL**: inline agora (callback async que segura o turno; `requires_approval` no AgentTool + decision "ask" nos hooks roteando ao callback; sem callback = fail closed) com durável registrado pós-ConversationStore — vs durável agora (exigiria adiantar storage-layer-gaps).
2. **Structured output**: output tool forçada como default portável + prompted como fallback opt-in (native Anthropic `output_config` como passthrough por demanda) — vs native-first.
3. **MCP**: ficar na fastmcp 3.x estável e entregar defer_loading wire + collision check + prompts/resources agregados no pool com API app-facing (padrão dos clients de referência: nunca automático) — vs migrar pra 4.x beta agora.
4. **Memory tool**: AgentTool próprio do anchor (schema explícito, funciona nos 7 providers) + `FileMemoryBackend` com os guards do spec — vs reusar o helper Anthropic-only do SDK.
5. **Pendências de retrieval** (recomendação: resolver por deleção): late-interaction retriever (reescrever ou **deletar**) e Redis vector store (implementar ou **dropar** a promessa) — nada de usuário depende de nenhum dos dois.
