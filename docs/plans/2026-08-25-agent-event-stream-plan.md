# Event stream do agent loop — plano

**Date:** 2026-08-25 · **Baseline:** `dev` @ `f7d1225` · **Research:** `docs/research/2026-08-25-agent-event-stream.md`
**Decisões do Arthur (2026-08-25):** (1) loop único emitindo eventos, `chat`/`achat` viram projeções de texto — sem 3ª cópia do loop; (2) subagents flat no stream do pai com `parent_tool_call_id` (modelo Claude SDK); (3) os 5 buracos/bugs da pesquisa entram nesta fase.

**Rules of engagement:** um commit por fase; cada fase com testes message-level (`FakeLLMProvider`) que pegariam o que ela conserta; os 2670 testes existentes ficam verdes em todos os commits (chat/achat mantêm assinatura e comportamento).

---

## Contrato de eventos

`src/anchor/agent/events.py` — pydantic `frozen=True`, discriminados por `type: Literal[...]`, todos com `parent_tool_call_id: str | None = None` (preenchido só em eventos encaminhados de subagent):

| Evento | Campos | Quando |
|---|---|---|
| `TurnStarted` | — | após `_prepare_turn` |
| `RoundStarted` | `round`, `max_rounds` | topo do round (R4: "round 2/8") |
| `CompactionStarted` / `CompactionFinished` | — / `tokens_before`, `tokens_after` | em volta do `_maybe_compact` (chamada LLM hoje invisível) |
| `TextDelta` | `text` | por chunk de texto (alvo da projeção chat/achat) |
| `ToolStarted` | `tool_call_id`, `name`, `tool_input` | antes de executar (R2) |
| `ToolFinished` | `tool_call_id`, `name`, `result`, `is_error` | ao completar — erro de tool é `is_error=True`, não exceção (R8) |
| `RoundFinished` | `round`, `usage: RoundUsage` | fecho do round, com o RoundUsage que já é fabricado ali (R5) |
| `TurnFinished` | `text`, `diagnostics: TurnDiagnostics` | evento terminal (resultado como último evento **e** via `last_turn`) |

Exceções de turno (provider, MCP) continuam levantando — Python idiomático; o `try/finally` garante a contabilidade. Sem camada de deltas brutos do provider (opt-in de raw é YAGNI até alguém pedir).

**API pública:**
- `Agent.astream(message) -> AsyncIterator[AgentEvent]` — interface primária (TUI: `async for` num `@work`).
- `Agent.stream(message) -> Iterator[AgentEvent]` — espelho sync (convenção do repo; o corpo sync existe de qualquer forma para `chat`).
- `chat`/`achat` — assinaturas intactas, viram `for ev in stream: if TextDelta: yield ev.text`.
- `AgentCallback` continua (R10): disparado dos mesmos pontos semânticos, assinaturas intactas.

---

## Fase 1 — Loop vira gerador de eventos *(o core)*

- [ ] `events.py` com o contrato acima; export em `agent/__init__.py` + `anchor/__init__.py` (+ `tests/test_exports.py`)
- [ ] Corpos de `chat` (`agent.py:1009-1086`) e `achat` (`:1088-1171`) viram `_stream_turn`/`_astream_turn` gerando `AgentEvent`; `chat`/`achat`/`stream`/`astream` são cascas finas
- [ ] `try/finally` no gerador: `_finish_turn` + `on_round_end` rodam em abandono do consumidor e em exceção (buraco §A.2)
- [ ] `tool_call_id` propagado do `ToolCall.id` para eventos (buraco §A.3)
- [ ] Fase de tools async: `asyncio.as_completed` no lugar do `gather` — `ToolStarted` de todos no início (fiel: começam juntos), `ToolFinished` ao vivo conforme cada uma completa; paralelismo preservado; wrapper por task converte exceção em `ToolResult(is_error=True)` **passando por `_error_result`** → conserta o `on_tool_error` perdido (buraco §A.1); ordem dos results de volta ao modelo preservada
- [ ] Sync: `_run_tools` vira gerador de eventos com results coletados (sequencial, como hoje)
- [ ] `RoundStarted`/`RoundFinished`/`TurnStarted`/`TurnFinished` emitidos com `max_rounds`, `RoundUsage`, `TurnDiagnostics`
- [ ] Testes: sequência ordenada de eventos num turno com 2 tools paralelas (ids casando start→finish); tool que levanta exceção no gather vira `ToolFinished(is_error=True)` + `on_tool_error` disparado; abandono do gerador ainda persiste memory/diagnostics; projeção: `achat` produz exatamente os `TextDelta` do `astream`

**Done when:** suite existente verde sem alteração de asserts de comportamento; novos testes de ordem/ids/finally verdes.

## Fase 2 — Subagents flat no stream

- [ ] `_run_sync`/`_run_async` (`subagent.py:117-157`) drenam `sub.stream()`/`sub.astream()` em vez de `chat`/`achat`; texto final continua vindo do `TurnFinished.text` do filho
- [ ] Eventos do filho encaminhados ao stream do pai com `parent_tool_call_id` = id da call do tool `task`/`<nome>` (re-emissão via `model_copy`)
- [ ] Async: merge na fase de tools — `asyncio.Queue` interna drenada enquanto as tasks rodam (eventos do filho chegam DURANTE a execução, não no fim); sync: yield inline
- [ ] `TurnFinished` do filho não vaza como terminal do pai (encaminhado com parent_id, consumidor distingue)
- [ ] Testes: orquestrador + subagent — eventos do filho aparecem com parent_id entre `ToolStarted` e `ToolFinished` do task; usage do filho visível via `RoundFinished` encaminhado

**Done when:** o done-when do MULTI_AGENT.md ganha visibilidade: um `async for` mostra progresso do subagent ao vivo.

## Fase 3 — Fixes periféricos + limpeza

- [ ] Gemini: `arguments_fragment` com `json.dumps(args)` em vez de `str(dict)` (`gemini.py:473-478`) + teste de round-trip com `json.loads` do loop
- [ ] `_is_final_round`: remover o `bool(round_index)` — `max_rounds=1` marca round final (`agent.py:892-893`) + teste
- [ ] Deletar `anchor.models.streaming` órfão (`StreamDelta`/`StreamUsage`/`StreamResult` — zero uso em src/) e seus exports/testes; breaking pre-1.0, changelog
- [ ] Compaction: `CompactionStarted`/`Finished` emitidos em `_maybe_compact`/`_amaybe_compact`
- [ ] Docstrings do módulo agent + entrada no mkdocs (`guides/`) com o exemplo TUI-shaped (`async for ev: match ev`)

**Done when:** suite completa verde; grep por `StreamDelta` limpo; docs buildam.

---

## Fora de escopo (registrado)

- Migração da astro-tui (repo irmão; rename `astro_context`→`anchor` já destacado como task separada). A TUI consome `astream()` quando migrar.
- Cancel graceful estilo OpenAI (`cancel("after_turn")`) — o pull-based já dá interrupção por abandono + finally; cancel rico só se a TUI pedir.
- Camada opt-in de deltas brutos do provider; namespacing hierárquico de subagents (flat+parent cobre TUI).
- Budget enforcement (P0 #2) — próxima fase do backlog, vai se apoiar nos eventos (`RoundFinished.usage`).

## Review

(preencher pós-implementação — ritual: ponytail reviewer + adversarial judge)
