# Sessão de release engineering — caminho limpo para o deploy da 0.2.0

**Data:** 2026-08-31 · **Branch:** dev · **Plano aprovado pelo Arthur no início da sessão.**

## Contexto

A 0.2.0 só será cortada depois das frentes #3–#5 do roadmap (decisão desta
sessão). Esta sessão remove todos os outros bloqueios do deploy, para que
cortar a release vire só: bump + tag + push.

**Decisões do Arthur nesta sessão:**

1. Escopo: a 0.2.0 espera #3–#5 (vault, grafo, mem0) — sem corte hoje.
2. Review gate: o ultra não coube (diff 44,5k linhas > limite 8k) → review
   local xhigh pelo ritual do /code-review, alvo `d50dd3c..dev`.
3. Publish: GitHub Actions + PyPI trusted publishing (workflow criado).
4. Cookbook + migração downstream: depois do publish da 0.2.0.
5. 0.1.2 de emergência: **aprovado no plano, cancelado durante a execução**
   — ver achado abaixo.

## Achado que mudou o plano

O passo de verificação da base (E.2) comparou o sdist publicado do 0.1.1
com o git: **o pacote no PyPI é a árvore de `110711d` + bump local de
versão** (fontes idênticas, só o pyproject difere). Não contém `anchor.llm`
— logo **o deadlock do `create_provider` nunca foi publicado**; afetava só
instalações da dev via git. A emergência 0.1.2 evaporou e o Arthur cancelou.
O vault dizia "pendura para sempre no 0.1.1 publicado" — corrigido.

Consequências aplicadas:
- CHANGELOG `[0.1.1]` reescrito com o conteúdo real (rename
  astro-context→anchor, primeira publicação, docs/branding); camada
  multi-provider voltou para `[Unreleased]`, com `client→llm` e a remoção
  dos `to_*_schema` promovidas a Breaking (são breaking contra o 0.1.1
  publicado de verdade).
- Retro-tag local `v0.1.1` em `110711d` (não empurrar: o release.yml
  falharia no check tag==versão — pyproject lá diz 0.1.0).
- Fix RLock sai naturalmente na 0.2.0.

## Checklist da sessão

- [x] Push da dev (6 commits acumulados → origin/dev)
- [x] `release.yml` — tag `v*` → suíte 3.11–3.13 → check tag==versão →
      `uv build` → publish OIDC (environment `pypi`, sem token) (`4184183`)
- [x] CI passa a rodar na dev (`4184183`) — antes só main: nada da v0.2
      tinha passado pelo CI do GitHub
- [x] Reparo do CHANGELOG: headers únicos, 6 entradas faltantes escritas,
      0 entradas perdidas (diff de bullets verificado) (`d6301bd`)
- [x] Correção do `[0.1.1]` contra o sdist publicado (`48ec31e`)
- [x] API reference do agent: superfície v0.2 completa, +436 linhas,
      mkdocs build limpo (`22f09ee`)
- [x] 0.1.2: investigado, premissa refutada, cancelado pelo Arthur
- [x] Review xhigh `d50dd3c..dev`: 10 finders → 10 verificadores →
      sweep — 50 CONFIRMED + 1 PLAUSIBLE, 8 REFUTED; 19 corrigidos
      nesta sessão (5 commits), resto em follow-ups abaixo
- [x] Suíte final 2871 verdes; ruff/mypy = baseline; push da sessão
- [x] Obsidian (repo note + daily) + Review section aqui

## Pendências que o Arthur precisa fazer (1 min cada)

- **PyPI trusted publisher** (antes da primeira tag): projeto
  `astro-anchor` → Publishing → owner `artcgranja`, repo `anchor`,
  workflow `release.yml`, environment `pypi`.

## O que resta para o deploy da 0.2.0 (depois desta sessão)

1. Frentes #3, #4, #5 do roadmap (uma sessão cada; #4 maior).
2. Corte: bump 0.2.0 + seção no CHANGELOG + PR dev→main + tag `v0.2.0`
   (o workflow publica sozinho).
3. Pós-publish: cookbook + migração astro-skills/tui/context.
4. Bloqueios do #4 seguem com o Arthur: tipo de aresta; incremental vs
   full-rebuild.

## Review — xhigh `d50dd3c..dev` (10 finders → 10 verificadores → sweep)

**Placar: 50 CONFIRMED + 1 PLAUSIBLE, 8 REFUTED.** Cada verdict veio de um
verificador independente com quote de linha; os marcados ⚙ tiveram repro
executado. Ledger completo abaixo; o ReportFindings levou os 15 mais graves.

### Correctness — loop/subagentes/providers

1. ⚙ **Turnos concorrentes no mesmo subagente corrompem estado** — `task`/
   `as_tool` são `read_only=True` (subagent.py:311/277); duas chamadas ao
   mesmo agent_name num round batcham juntas sobre a MESMA instância child.
   Repro: pool debitou 140 de 240 reais (o `finally` do primeiro turno faz
   `_active_pool = None` no irmão vivo, agent.py:2322) e o turno B devolveu
   o `output` do turno A (agent.py:2132 lê `_last_output` alheio).
2. ⚙ **ChildTurn fantasma** — exceção antes do `_reset_turn_state` (guard
   MCP sync agent.py:2169; `_ensure_mcp` 2258) faz subagent.py:139/156
   encaminhar o `last_turn` do turno ANTERIOR: contabilidade duplicada em
   `run_total_*`.
3. ⚙ **`_pool_entry` usa `_EVENT_SINK` como sinal de "sou filho"**
   (agent.py:2066) — mas o sink é setado em VOLTA de toda tool call
   (1403/1471): um Agent independente com usage_limits chamado dentro do
   corpo de qualquer tool não enraíza pool; filhos dele nunca debitam.
4. ⚙ **claude_cli serve resultado stale para chamada nova** — hook defer
   permite por NOME (claude_cli.py:294) e o handler cai no `by_name`
   (271): `search(b)` recebe o resultado de `search(a)`; o loop nunca vê a
   chamada nova.
5. ⚙ **claude_cli replay descarta `is_error`** — `_delivery_maps` guarda só
   `content` (156) e o handler devolve sempre success-shape (272); um
   deny de hook replay como sucesso.
6. ⚙ **Retry re-emite stream do zero** — em `BaseLLMProvider.stream/astream`
   os yields ficam DENTRO do try do retry (base.py:147-151/187-191); erro
   transiente no meio duplica todo o texto já entregue (repro:
   'Hello worldHello world'). O FallbackProvider tem o guard; o base não.
7. **`timeout` do construtor ignorado** por todos os providers de API
   (openai.py:90/97, anthropic.py:160/167, gemini.py:129, família compat)
   — só claude_cli usa (`asyncio.timeout`, 377). `create_provider(...,
   timeout=5.0)` aceita e ignora em silêncio.
8. ⚙ **Structured output valida o texto acumulado de TODOS os rounds**
   (agent.py:2206 → subagent.py:96/206/230): narração do round 1 gruda no
   JSON final, queima retries e pode levantar com resposta válida.
   Atinge `as_tool(output_model=)`, `task` e `run(mode="prompted")`.
9. ⚙ **Round sem tools não appenda a resposta do assistant** (append só no
   tool phase, agent.py:1315 sob `run_tools`): retry de output vira
   `['user','user']` — o modelo re-tenta sem ver a própria resposta.
10. ⚙ **Prompt caching nunca chega ao wire** — `_formatted_to_messages`
    achata os blocos com `cache_control` (agent.py:1580-1602) E
    `create_provider` nunca liga `prompt_caching` (default False,
    anthropic.py:145). Dois comentários no código prometem o contrário.
11. ⚙ **`with_compaction(keep_last=0)` = IndexError no meio do turno**
    (agent.py:1975-1978, `messages[len(messages)]`); sem validação.
12. ⚙ **`astream` persiste a mensagem do usuário antes do `_ensure_mcp`
    poder levantar** (agent.py:2256-2258): retry após falha de conexão
    duplica a mensagem na memória. O `stream` sync é seguro.
13. ⚙ **LLM síncrono bloqueia o event loop** — todo o caminho async usa
    `add_*_message` sync (agent.py:2257/1551/2102); com
    ProgressiveSummarizationMemory, eviction roda `summarize` (HTTP sync)
    NO loop: heartbeat concorrente travou 4s no repro. Os `aadd_*`
    existem e ninguém chama.

### Correctness — memória progressiva/storage/MCP/ingestão

14. ⚙ **Race do swap de `_on_evict`** — swap/restore fora do lock
    (progressive.py:224-230; o comentário mente): thread B restaura o
    lambda morto de A permanentemente; evictions futuras somem.
15. ⚙ **Deadlock por lock não-reentrante** — o callback de eviction roda
    segurando `SlidingWindowMemory._lock` (Lock, não RLock;
    sliding_window.py:60/134-136); callback do usuário que lê
    `memory.total_tokens` trava para sempre; e o summarize (LLM,
    segundos) roda sob o lock.
16. ⚙ **Read-modify-write através de await perde tiers** —
    progressive.py:417-441 lê tier-1 sob lock, awaita com lock solto,
    escreve incondicional: gather de 2 evictions perdeu os turnos da
    primeira (repro).
17. **Exceções sintéticas** — 4 sites `except Exception:` →
    `Exception("compaction failed")` (progressive.py:298-301/352/428/473):
    consumidor do callback nunca vê a causa real.
18. ⚙ **sqlite `search_filtered` filtra tags depois do LIMIT**
    (sqlite/_entry_store.py:144-153): devolveu `[]` com 5 matches válidos;
    diverge da referência e do Postgres.
19. ⚙ **Race na `get_async_connection`** (sqlite/_connection.py:76-96, sem
    lock): 2 conexões criadas, a perdedora vaza thread não-daemon —
    repro pendurou o processo >120s no shutdown.
20. **Redis: índice stale destrói dado de outro usuário** — `add` upsert
    não limpa membership antiga (redis/_entry_store.py:39-49);
    `delete_by_user` deleta sem rechecar `entry.user_id` (206-221).
21. ⚙ **`expose_tool` descarta `input_schema`** (mcp/server.py:49) — fastmcp
    re-deriva da assinatura: enum/required/description perdidos; fn com
    `**kwargs` (memory tool, tools MCP) **crasha no registro**; e
    `from_agent` silenciosamente omite memory/MCP tools.
22. ⚙ **Meta-tools de skill sem guard de colisão** — repro:
    `['activate_skill', 'activate_skill']` enviados; o user tool ganha por
    first-match e o meta-tool fica inalcançável (agent.py:788, contraste
    com o guard MCP em 1196-1205).
23. **`ParentExpander` sem `parent_lookup` dropa filhos irmãos** (dedup
    antes do lookup, hierarchical.py:222-231) e devolve o filho rotulado
    como pai; nada avisa; docs ainda ensinam `parent_text` removido;
    nenhum wiring de `parent_lookup` no repo.
24. **CLI perdeu o cap de 10MB e o tratamento de OSError** — arquivo
    chmod-000 aborta o index inteiro com traceback (ingester.py:266 não
    pega PermissionError; parsers.py:25 lê tudo em memória).
25. ⚙ **Frontmatter YAML estrito dropa skills antes válidas em silêncio**
    — `description: Use this: when...` agora é ScannerError → skip com
    logger.warning no scan de diretório (loader.py:78-80/321-322).

### Eficiência (confirmadas)

26. ⚙ Re-tokenização quadrática do transcript por round em provider sem
    usage (agent.py:1771-1773/1952-1965) — e resultado ≥10k chars fura o
    cache do TiktokenCounter (threshold é CHARS, counter.py:46; o cap
    default é 10k TOKENS ≈ 40k chars). Compaction re-anda o transcript
    por round também (1973).
27. Rebuild de schemas + recount por round (agent.py:1650-1663) — memo
    por turno invalidado em deferred-load/ativação resolve.
28. Awaits sequenciais: `_aretrieve_from_store` (async_retriever.py:217),
    `aindex` (120-124), `_embed_documents` fallback (96).
29. Lookup linear de tool por call + dict reconstruído por round
    (agent.py:831/1426) — custo real baixo.
30. `_check_usage_limits` re-debita todos os rounds por round
    (agent.py:1931) — microssegundos; forma O(1) já existe no pool run.

### Cleanup/reuse (confirmadas, seleção)

31. `_math.clamp`: 0 callers de produção, 12 cópias inline.
32. `final_result` literal em 8 sites; sem constante.
33. `clean_schema` não aplicada ao output tool (e `clean_schema` dropa
    `$defs` — consertar ela primeiro, depois rotear).
34. Cap de resultado: isenção por marker attr; sem opt-out tipado por
    tool (gt=0 proíbe sentinela).
35. Attrs mágicos `_mcp_async_caller`/`_anchor_async_caller` +
    `_aexecute_tool` sem caller de produção + 2 comentários stale
    (mcp/tools.py:30/50) + fn `async def` direto cai no caminho sync sem
    guard de coroutine.
36. progressive/compactor: 4 duplicações sync/async (~90-110 linhas),
    flag `sync` morta, `_DEFAULT_TIER_CONFIG` morto, `aadd_turn`
    re-implementa `aadd_message`.
37. claude_cli: fold sync/async duplicado 4×; `_map_error` por nome sem
    a justificativa dos siblings.
38. Redis `_matches` re-implementa `_filters.matches_filters`; key
    builders 2×2; vec_store duplica clause builder; postgres re-lista
    campos (docstring justifica não-reuso, mas a lista canônica
    compartilhada falta); MarkdownParser duplica frontmatter parser do
    skills loader; manager.py branch isinstance duplicado.

### PLAUSIBLE

39. FixedSizeChunker sob BPE pode exceder `chunk_size` (1056/2000 trials
    no cl100k) — documentado como aproximação; quebra só consumidor com
    limite duro sem folga.

### REFUTED (com a prova)

- Cancel scope do fastmcp cross-task: design explícito do Client
  (session_task asyncio dedicado); repro limpo contra servidor stdio real.
- `_usage_int`: campos de cache do SDK real são Optional/None — guard de
  produção legítimo (docstring é que está errada).
- Cosine estrito no cross-modal: intent documentado; dims diferentes já
  eram garbage silencioso antes.
- Skill name==dir: warn-and-skip é mecanismo pré-existente; caminho loud
  existe.
- AsyncHybrid raise / activation default: breaking documentado e razoável.
- Guard de path do resources.py: equivalente em força ao do memory_tool
  (`%2e` é inerte sem URL-decode na cadeia; resolve() pega tudo).
- `_subagent_name` hard-coded: "task" é duplamente reservado; nenhum
  segundo dispatcher pode surgir pela API pública.

### Sweep (8 novos; 40-41 verificados por mim na fonte)

40. ⚙ **`json.loads` sem guard nos args streamados** (agent.py:1561) — o
    caminho non-streaming tem try/except (parse_response); JSON malformado
    do modelo mata o turno inteiro sem TurnFinished.
41. ⚙ **`parse_stream_chunk` só converte `delta.tool_calls[0]`**
    (_openai_compat.py:257) — deltas paralelos no mesmo chunk descartados;
    segunda call nunca forma ou alimenta o 40.
42. `_PRICE_WARNED` consultado ANTES da tabela (agent.py:130) — modelo
    preça $0 para sempre mesmo após override documentado de MODEL_PRICING.
43. `_graded_ndcg` trunca o IDCG em len(retornados), não em k
    (evaluation/retrieval.py:153) — retriever que devolve menos itens
    infla o NDCG; o gate de CI passa numa regressão real.
44. `anchor index` re-indexação estranda chunks antigos (cli.py:177 —
    doc_id = hash de conteúdo+path; nada deleta as linhas velhas).
45. `anchor index --language` é flag morta exibida como persistida
    (cli.py:153/194).
46. Fixtures sqlite de tests/test_storage/conftest.py nunca fecham o
    ConnectionManager (sem teardown) — conexões/WAL vazam a suíte inteira.
47. Escape-hatch de `_astream_tools._one` chama `_error_result` sem o
    tool resolvido (agent.py:1477) — cap por-tool e input reescrito
    ignorados só nesse caminho.

### Fixes aplicados nesta sessão

**19 findings corrigidos em 5 commits temáticos**, cada um com teste de
regressão provado por mutação (falha contra o pré-fix; exceções anotadas
nos commits). Suíte final: **2871 verdes** (+33 sobre o baseline 2838);
ruff 155 / mypy 145 = baseline exato.

- `1e7de68` fix(llm,agent): retry re-emitia stream (6), deltas paralelos
  descartados (41), json.loads sem guard (40), timeout ignorado (7).
- `7912dd5` fix(storage): tags×LIMIT (18), race da conexão async (19),
  índice stale do Redis destruindo dado alheio (20).
- `44907a9` fix(memory): deadlock de eviction (15), race do swap (14),
  RMW async perdendo tiers (16), causa raiz nas exceções (17).
- `6ffa2d8` fix(agent,llm,evaluation): keep_last=0 (11-borda), ordem do
  astream (12), colisão de meta-tools de skill (22), memo de preço (42),
  NDCG ideal×k (43), claude_cli stale by_name (4) + isError (5).
- `018bc65` fix(agent)!: serialização por child (1), ChildTurn fantasma
  (2), validação no último round (8), retry vê a própria resposta (9),
  prompt caching default no provider Anthropic do Agent (10-mínimo).

### Follow-ups registrados (não corrigidos nesta sessão)

**Prioritários (candidatos à próxima sessão ou à do vault #3):**
- **(13/5)** Memória síncrona bloqueia o event loop no caminho async —
  precisa de adds async no MemoryManager + separar o `_finish_turn`;
  os `aadd_*` da progressive existem e ninguém chama.
- **(3)** `_pool_entry` usa `_EVENT_SINK` como sinal de "sou filho" —
  agente independente dentro de tool body não enraíza pool.
- **(10-resto)** Plumbing dos blocos `cache_control` do formatter até o
  wire (`raw_content`); hoje o caching efetivo é o auto-caching
  top-level do provider.
- **(21)** `expose_tool` descarta `input_schema` (crasha com `**kwargs`;
  `from_agent` omite memory/MCP tools em silêncio) — precisa da API de
  schema explícito do fastmcp.
- **(23)** `ParentExpander` sem `parent_lookup` dropa filhos irmãos e
  degrada em silêncio; docs de ingestion ainda ensinam `parent_text`.
- **(24/44/45)** CLI: cap de tamanho + OSError perdidos; re-indexação
  estranda chunks antigos; `--language` do index é flag morta.
- **(25)** Frontmatter YAML estrito dropa skills antes válidas com só
  um warning no scan de diretório.

**Eficiência (26-30):** re-tokenização quadrática por round (+ cache do
TiktokenCounter furado por resultados grandes — threshold em chars vs
cap em tokens), rebuild de schemas por round, awaits sequenciais no
retrieval async, lookups lineares. **Cleanup (31-38, 46-47):** clamp,
constante final_result, clean_schema + output tool, opt-out tipado do
cap, attrs mágicos + `_aexecute_tool` morto, duplicações sync/async
(progressive/compactor/claude_cli), duplicações redis/postgres/parsers,
fixtures sqlite sem teardown, escape-hatch sem tool resolvido.
**PLAUSIBLE (39):** overshoot BPE do FixedSizeChunker (documentado).
