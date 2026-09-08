# Provider `claude_cli` — usar a assinatura do Claude Code como backend de LLM

**Data:** 2026-08-28 · **Branch:** dev

**Objetivo:** anchor deixar de exigir `ANTHROPIC_API_KEY` para rodar modelos
Claude. `Agent(model="claude_cli/sonnet")` passa a usar o binário `claude` já
autenticado (assinatura Pro/Max via OAuth, ou `claude setup-token`), **com
tool-calling completo**.

---

## Pesquisa — como o ecossistema resolve isso (SOTA, ago/2026)

Três wrappers públicos do Claude CLI como model provider:

| Projeto | Framework | Tool calling |
|---|---|---|
| [`langchain-claude-cli`](https://github.com/clriesco/langchain-claude-cli) | LangChain | **sim** — MCP in-process + hook `PreToolUse` → `defer` |
| [`langchain-claude-code`](https://pypi.org/project/langchain-claude-code/) | LangChain | via Agent SDK |
| [`pydantic_ai_claude_code`](https://github.com/wehnsdaefflae/pydantic_ai_claude_code) | Pydantic AI | sim, subprocesso direto |

O padrão que venceu é o do `langchain-claude-cli`, e ele resolve exatamente o
problema que eu tinha dado como insolúvel ("o CLI é um agente, executa as
próprias ferramentas"):

1. As ferramentas do chamador são registradas como **servidor MCP in-process**
   (`create_sdk_mcp_server`), namespace `mcp__anchor__*`.
2. Um hook `PreToolUse` casando `mcp__anchor__.*` devolve
   `permissionDecision: "defer"`.
3. O CLI **para o turno sem executar** e devolve `stop_reason: "tool_deferred"`
   + `deferred_tool_use`. O loop do chamador recebe o tool call.
4. Para fechar o ciclo, o *handler* MCP devolve o resultado já guardado quando
   a mesma chamada é refeita — não é preciso `resume` para correção, só para
   economia de tokens.

### Verificado nesta máquina (não é leitura de README)

`claude` 2.1.247 + `claude-agent-sdk` 0.2.147. Probe em
`scratchpad/probe_defer.py`:

```
BLOCK: ToolUseBlock mcp__anchor__get_weather {'city': 'Tokyo'}
HOOK FIRED: mcp__anchor__get_weather {'city': 'Tokyo'}
stop_reason: "tool_deferred"
deferred_tool_use: DeferredToolUse(id='toolu_01SV…', name='mcp__anchor__get_weather', input={'city': 'Tokyo'})
TOOL EXECUTED? []          # ← a função Python NÃO rodou
```

`permissionDecision: "defer"` é oficial (`PreToolUseHookSpecificOutput` do SDK,
`Literal["allow","deny","ask","defer"]`) e o binário do CLI carrega
`deferred_tool_use` nativamente.

### Outras medições

| Fato | Medição |
|---|---|
| System prompt default do Claude Code | **177.821** tokens de overhead |
| `system_prompt=…` + `tools=[]` + `setting_sources=[]` + `strict_mcp_config=True` | **~284** tokens (**725** com 1 tool MCP) |
| Sem `strict_mcp_config` | MCPs do usuário vazam: **+65k** tokens |
| Streaming token-a-token | ✅ `include_partial_messages` |
| Usage + `total_cost_usd` reais | ✅ no `ResultMessage` |
| Erros | ✅ `api_error_status` (401/404/429/5xx) + `is_error` |
| Chamada auxiliar em Haiku | ~900 tokens por run, embutida no CLI — inevitável |
| Replay de turnos `assistant` via stream-json | ❌ cada `user` vira turno novo |

---

## Decisões de design

1. **Depender de `claude-agent-sdk`** (extra opcional `claude-cli`), não
   subprocesso na mão. MCP in-process e hooks-como-callback existem só via o
   protocolo de controle do SDK; refazer isso à mão seria reimplementar o SDK
   oficial da Anthropic. O SDK ainda empacota o próprio CLI.
2. **Ponte sync↔async.** O SDK é async-only; `LLMProvider` exige
   `invoke`/`stream` síncronos. Uma thread worker com event loop próprio
   (~25 linhas), segura mesmo chamada de dentro de um loop.
3. **Histórico achatado** num prompt único com rótulos de papel (o CLI não
   aceita replay de `assistant`). Correção do ciclo de tools vem do mapa de
   entrega, não do histórico. `ponytail:` marcando `resume` + prefix-cache
   como upgrade de custo, não de correção.
4. **`max_tokens`/`temperature`/`stop` ignorados** — sem equivalente no CLI.
   Documentado; não levanta erro porque o `Agent` passa `max_tokens` sempre.
5. **Custo vem do CLI** (`total_cost_usd`), não de `pricing.py`.

---

## Itens

- [ ] `pyproject.toml` — extra `claude-cli = ["claude-agent-sdk>=0.2,<1"]`, entrar em `all`
- [ ] `src/anchor/llm/providers/claude_cli.py` — `ClaudeCLIProvider(BaseLLMProvider)`
  - [ ] `provider_name = "claude_cli"`; `_resolve_api_key()` → `None`
  - [ ] `_build_options()`: `system_prompt`, `tools=[]`, `setting_sources=[]`,
        `strict_mcp_config=True`, `model`, `cli_path` opcional
  - [ ] `ToolSchema[]` → `create_sdk_mcp_server("anchor", …)` + `allowed_tools`
  - [ ] hook `PreToolUse` `mcp__anchor__.*` → `defer`, com mapa de entrega
        (`(name, args_canônicos)` → resultado) para fechar o ciclo
  - [ ] `deferred_tool_use` → `LLMResponse.tool_calls` + `StopReason.TOOL_USE`,
        des-namespaceando `mcp__anchor__`
  - [ ] streaming: `include_partial_messages` → `StreamChunk`
  - [ ] `ResultMessage` → `Usage` (incl. cache tokens e `total_cost_usd`)
  - [ ] `api_error_status` → `AuthenticationError`/`ModelNotFoundError`/
        `RateLimitError`/`ServerError`
  - [ ] ponte sync (thread + loop próprio) para `_do_invoke`/`_do_stream`
- [ ] `src/anchor/llm/registry.py` — `claude_cli` em `_PROVIDER_MODULES`,
      `_PROVIDER_PACKAGES`, `_PROVIDER_EXTRAS`
- [ ] `tests/llm/providers/test_claude_cli.py` — SDK mockado: options, ponte de
      tools (defer + entrega), parsing de stream, usage/custo, mapa de erros,
      registro
- [ ] `tests/live/test_providers_live.py` — caso live opt-in (skip sem `claude`)
- [ ] docs: `guides/llm-providers.md`, `api/llm.md`, README (tabela + limitações)
- [ ] suíte completa verde

## Fora de escopo

- `resume` + prefix-cache de sessão (economia de tokens, não correção)
- Modo agêntico (deixar as ferramentas nativas do CLI — Read/Bash/… — rodarem
  dentro do run). É outra feature.
- Structured output via `output_format` nativo do CLI.

## Review

**Entregue.** Escopo escolhido pelo Arthur: SDK + modo agêntico.

### Arquivos

| Arquivo | |
|---|---|
| `src/anchor/llm/providers/claude_cli.py` | novo — 596 linhas |
| `src/anchor/llm/registry.py` | `claude_cli` registrado + **fix de deadlock** |
| `pyproject.toml` | extra `claude-cli`, entra em `all` e `all-providers` |
| `tests/llm/providers/test_claude_cli.py` | novo — 63 testes, SDK falso em `sys.modules` |
| `tests/llm/test_registry.py` | regressão do deadlock |
| `tests/live/test_providers_live.py` | 2 casos live (skip sem `claude`/SDK) |
| docs | `guides/llm-providers.md`, `api/llm.md`, README |

### Bug pré-existente encontrado no caminho (release-blocker)

`create_provider()` fazia **deadlock na primeira chamada de qualquer provider**
num processo novo. `_LOCK` era um `threading.Lock` não-reentrante, e
`create_provider` o segurava durante `_try_import_provider()` → import do
módulo → `register_provider()` → `with _LOCK` de novo.

Reproduzido: `anthropic/…`, `openai/…`, `gemini/…` e `claude_cli/…` todos
travam em processo limpo. Ou seja, `Agent(model="anthropic/claude-…")` pendura
para sempre em `astro-anchor` 0.1.1. A suíte não pegava porque os testes
importam as classes de provider direto antes, populando `_PROVIDERS`.

Fix: `threading.RLock()` (uma linha). Mutação confirmada — com `Lock` o teste
de regressão falha em 5s, com `RLock` passa.

### Desvios do plano

1. **`ResultMessage.result` tem precedência sobre o texto acumulado** (o plano
   não decidia). Em modo agêntico o texto acumulado carrega a narração
   intermediária; `result` é a resposta. Texto acumulado vira fallback (é o que
   sobra num run deferido, onde `result` vem vazio).
2. **`allowed_tools` também recebe a lista de builtins.** Sem isso o CLI pede
   permissão que não consegue obter em modo print e a ferramenta morre.
3. **`_iter_sync` não dá `join` na thread.** Com `join` num `finally`, parar de
   consumir um stream no meio bloqueava até o CLI terminar. Fila ilimitada +
   thread daemon: a bomba termina sozinha.
4. **`TimeoutError` importado com alias** (`ProviderTimeoutError`), diferente de
   `anthropic.py`: aqui é preciso capturar o `TimeoutError` embutido que
   `asyncio.timeout` levanta.

### Verificação

- Suíte completa: **2784 passed, 3 skipped** (baseline 2723).
- Ruff limpo nos arquivos novos e editados (o repo tem 50 achados de baseline
  em `src/`, não tocados).
- Live, contra o CLI real (`claude` 2.1.247 + SDK 0.2.147):
  - texto: `PONG`, usage 280/5, custo US$ 0.00156
  - streaming: deltas token-a-token + usage no chunk final
  - tool call deferido → `ToolCall(name='get_weather', …)`, `stop_reason=tool_use`
  - ciclo fechado: resultado da ferramenta chega ao modelo, resposta final com "25°C"
  - `Agent` completo: ferramenta executada **no processo do anchor**
    (`CALLS == ['Tokyo']`), 2 rounds, resposta correta
  - agêntico: `Read` nativo do CLI leu o arquivo e devolveu "4271"
- `tests/live` passam 2/2.

### Notas

- `claude-agent-sdk` **não** entrou no grupo `dev`: os testes mockados não
  precisam dele e ele empacota o CLI (~200 MB). Consequência: depois de um
  `uv sync` os testes live pulam até reinstalar com
  `uv pip install claude-agent-sdk`.
- Ainda fora de escopo (economia de tokens, não correção): `resume` +
  prefix-cache de sessão. Hoje cada round re-envia o histórico achatado.
