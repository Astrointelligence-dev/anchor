# Event stream do agent loop — pesquisa

**Date:** 2026-08-25 · **Baseline:** `dev` @ `f7d1225` · **Item:** backlog pós-Phase 5, P0 #1 (`docs/plans/2026-08-25-sota-upgrade-plan.md:107`)
**Método:** 3 frentes paralelas — (A) mapa do loop atual do anchor, (B) SOTA 2026 de event streaming em agent SDKs (fontes primárias), (C) requisitos do consumidor real (astro-tui).

---

## A. Estado atual do anchor

**Correção de premissa do backlog:** `chat`/`achat` **já são geradores de texto incremental** (`Iterator[str]`, `agent.py:1009`; `AsyncGenerator[str, None]`, `agent.py:1088`). O gap real: **só texto sai pelo iterator** — tools, rounds, usage, subagents saem por callbacks síncronos laterais (`AgentCallback`, `hooks.py:45-71`), sem ordem garantida entre os dois canais do ponto de vista do consumidor.

### O loop (pós-dedup da Phase 4)

- Corpo do round: `for round_index in range(self._max_rounds)` (`agent.py:1035`/`:1120`) → `on_round_start` → compaction (`_maybe_compact`) → final-round notice → `_round_request` → **stream do provider** (`self._llm.stream/astream`, únicos pontos de chamada no loop, `agent.py:1045`/`:1130`) → `_ingest_chunk` por chunk.
- **`_ingest_chunk` (`agent.py:837-870`) é o funil único**: todo chunk do provider passa por ele — `content` (→ yield), `tool_call_delta` (acumulado por index), `usage` (max monotônico), `raw_block` (compaction), `stop_reason`. É o ponto natural de conversão delta→evento.
- 3 saídas de round: (a) stop sem tools; (b) round final pediu tools → não executa (`agent.py:1063-1064`); (c) tools executam (`_run_tools` sequencial / `_arun_tools` com `asyncio.gather`, sem cap) e o loop continua.
- `RoundUsage` é fabricado exatamente onde `on_round_end` dispara (`agent.py:1081-1084`) — mas o callback só recebe `round_index` (`hooks.py:56`).
- `_finish_turn` (`agent.py:990-1001`) monta `TurnDiagnostics` e persiste na memory — **fora do for, sem try/finally**: não roda se o consumidor abandonar o gerador ou se uma exceção subir.

### Callbacks e hooks

- `AgentCallback` (Protocol, tudo sync): `on_round_start/end(round_index)`, `on_tool_start/end/error(name, tool_input, ...)`. Dispatch via `fire_callbacks` (`_callbacks.py:10-36`): duck-typed, exceção engolida + WARNING. Callback `async def` retornaria coroutine nunca awaited. Callback lento **bloqueia o event loop** (roda inline em `_aexecute_call`).
- Único implementador: `TracingAgentCallback` (`observability/callback.py:225`) — spans chaveados por **nome** de tool (calls paralelas colidem, `ponytail:` em `:272-273`).
- Veto hooks (pre deny/rewrite, post replace) são canal de **controle**, separado — ficam como estão.

### Streaming nos providers

- Todos os 7 suportam `stream`/`astream`; `StreamChunk` (`llm/models.py:111-123`) = `content | tool_call_delta | usage | stop_reason | raw_block`.
- **Anthropic é o único que emite usage no stream** e o único com `raw_block` (compaction). OpenAI/LiteLLM emitem `stop_reason` sem usage → `RoundUsage` fica 0 nesses providers.
- **Bug latente (Gemini)**: tool args chegam como `str(dict)` Python, não JSON (`gemini.py:473-478`) — colide com `json.loads` em `agent.py:725`.
- `anchor.models.streaming` (`StreamDelta`/`StreamUsage`/`StreamResult`) é **órfão**: exportado no top-level, zero uso em `src/` fora do `__init__`, só testes.

### Subagents

- `_run_sync`/`_run_async` drenam o gerador do filho num `"".join(...)` (`subagent.py:123`/`:144`) — o texto some. Agent filho não herda callbacks (`agent.py:371-382`). **Zero eventos internos do filho chegam ao pai** — só `on_tool_start/end` do tool `task`/`<nome>` no nível do pai.

### Buracos de evento encontrados (fatos, não design)

1. `_arun_tools`: o ramo `isinstance(result, BaseException)` (`agent.py:693-700`) monta o erro **sem passar por `_error_result`** → `on_tool_error` não dispara para exceções vindas do gather.
2. `on_round_end` e `_finish_turn` não executam em abandono do gerador nem em exceção (sem try/finally).
3. `on_tool_start/end` não carregam `tool_call_id` — com `asyncio.gather`, duas calls concorrentes da mesma tool são indistinguíveis (`ToolCall.id` existe em `agent.py:677` e não é propagado).
4. Compaction roda chamada LLM bloqueante dentro do round sem nenhum evento próprio (`agent.py:1037`).
5. `_is_final_round` (`agent.py:892-893`): `bool(round_index)` faz `max_rounds=1` nunca marcar round final.

### Convenções do repo (para o design seguir)

Pydantic `BaseModel frozen=True` para tipos de valor (não dataclass); `StrEnum`; espelho sync/async por prefixo `a`; config fluente `with_*` retornando self; `__slots__` exaustivo no `Agent` (`agent.py:131-162` — atributo novo precisa entrar lá); `__all__` em dois níveis (`agent/__init__.py` + `anchor/__init__.py`, com `tests/test_exports.py`); testes message-level com `FakeLLMProvider` (lista de chunks por chamada) + builders `_text_response`/`_tool_use_response`/`_multi_tool_use_response`.

---

## B. SOTA 2026 — como os SDKs expõem o stream

Detalhe completo com URLs no relatório do subagent; aqui o resumo decisório.

| | Interface primária | Taxonomia | Subagents | Resultado final |
|---|---|---|---|---|
| **Pydantic AI** | `run_stream_events()` → union `AgentStreamEvent` (`event_kind` literal); `iter()` para nó-a-nó; `event_stream_handler` concorrente | Part start/delta/end + `FunctionToolCallEvent`/`ResultEvent` + `FinalResultEvent` ("o output final começou") | — | último evento (`AgentRunResultEvent`) **e** `run.result`/`run.usage` (propriedade viva) |
| **OpenAI Agents** | `run_streamed()` → `stream_events()`; union por `type` | raw (deltas brutos) vs semântico (`RunItemStreamEvent`: tool_called/tool_output/handoff…) em camadas | flat, sem nesting — handoff troca o "agente corrente" | envelope `RunResultStreaming` (`final_output`, `cancel("immediate"/"after_turn")`, `to_state()`) |
| **Claude Agent SDK** | `query()` async iterator de **mensagens completas**; deltas brutos opt-in (`include_partial_messages`) | blocos (Text/Thinking/ToolUse/ToolResult) | **flat + `parent_tool_use_id`** em cada mensagem; deltas de subagent não propagam (limitação assumida); `TaskProgressMessage` com task_id+usage | `ResultMessage` como último item do stream |
| **LangGraph** | `astream(stream_mode=...)`; v3 `GraphRunStream` pull-based com projections tipadas | `astream_events` v2 = firehose, considerado erro de design ("parsing logs") | namespace hierárquico (`ns` tuple, `depth`) | estado final |
| **Vercel AI SDK** | `fullStream` (parts) + `textStream` derivado | `start`/`start-step`/`text-delta`/`tool-call(3 fases)`/`finish-step(usage)`/`finish(totalUsage)`/`error`/`abort` | — | promises no result + part `finish` |

**Blueprint convergente** (evidência acima):

1. **Union discriminado** (`kind`/`type` literal) num **único async iterator** — todos.
2. **Projeção "só texto" derivada** para o caso comum (Vercel `textStream`, Pydantic `stream_text()`, LangGraph `run.messages`).
3. **Default semântico; delta bruto do provider é opt-in** (OpenAI raw layer, Claude `include_partial_messages`).
4. **Step/round delimitado com usage por step** + agregado no fim (Vercel `finish-step`, Pydantic `run.usage`).
5. **Subagent flat com parent pointer** é o modelo mais simples que serve TUI (Claude); namespace hierárquico (LangGraph) só se árvores profundas importarem.
6. **Resultado final: último evento no stream E propriedade no objeto** — as duas convenções juntas.
7. **Pull-based** (caller dirige iterando) = backpressure de graça (LangGraph v3); OpenAI paga o preço do push (obriga drenar).
8. **Ninguém oferece stream de eventos sync como interface principal** — sync existe como wrapper documentado com limitação de event loop.
9. Callbacks nunca são a interface primária de streaming: ou observabilidade concorrente (Pydantic `event_stream_handler`) ou controle (Claude hooks).

---

## C. Requisitos do consumidor (astro-tui)

Estado: Textual, 3 commits, showcase com 10 screens; **integração quebrada** — importa `astro_context`, que não existe pós-rename (`chat.py:10-18`, `pyproject.toml:21` → `../astro-context`). Consumo atual: `chat()` sync em thread worker, `call_from_thread` por chunk de texto; zero uso de callbacks/hooks; sidebar e diagnostics atualizados só pós-turno via `last_result`; screen Observability alimentada com `random.uniform`. Widgets prontos e ociosos: `StatusIndicator`, `DiagnosticsPanel`, `MetricBars`.

Requisitos derivados do código (R = requisito; [INF] = inferência):

- **R1** — stream `AsyncGenerator` consumível por `async for` num `@work` async do Textual (achat já é; chat exclui MCP).
- **R2 (bloqueante)** — `tool_call_id` nos eventos de tool: com gather, start/end sem id não casam.
- **R3** — ordem total texto ⊕ tools ⊕ rounds num único canal (RichLog append-only precisa de sequência única).
- **R4** — round index **e total** (`max_rounds`) + `stopped_by` como evento terminal.
- **R5** — `RoundUsage` ao vivo por round (widgets prontos esperando).
- **R6** — custo: usage + model no stream, TUI computa [INF].
- **R7** — progresso de subagent (hoje caixa-preta total).
- **R8** — erro de tool ≠ erro de turno; erro como evento. Inclui consertar o buraco do gather (§A.4 item 1).
- **R9** — eventos append-only-friendly (cada evento renderizável como linha nova) [INF].
- **R10** — `AgentCallback` continua existindo (TracingAgentCallback depende).
- **R0 (fora deste repo)** — consertar o rename `astro_context`→`anchor` na TUI antes de qualquer E2E.

---

## D. Decisões de design em aberto (para o Arthur)

1. **Arquitetura da API**: (a) `astream()` novo ao lado de chat/achat (3ª cópia do loop — ruim pós-dedup da Phase 4); **(b) o loop passa a emitir eventos e `achat`/`chat` viram projeções de texto do mesmo stream** (um código só, blueprint #2); (c) só callbacks async. Recomendação: (b).
2. **Subagents no stream**: flat com `parent_tool_call_id` (Claude, recomendado) vs progress-only vs manter caixa-preta.
3. **Escopo**: incluir os 5 buracos/bugs do §A (estão no código tocado) ou fase separada.
4. **`anchor.models.streaming` órfão**: deletar e substituir pelos novos tipos de evento, ou reaproveitar nomes.
5. **Sync**: `chat()` continua texto-only; stream de eventos é async-only (consenso #8) — ou wrapper sync?
