# v0.2 · #5 — Consolidador de memória guiado por LLM (paridade mem0)

**Status:** em execução (sessão 12, 2026-09-08: pesquisa feita, 7 decisões fechadas, fases A–E) · **Tamanho:** pequeno · **Depende de:** nada
**Combina com:** #4 (o `GraphIndexingEntryStore` já mantém o grafo para qualquer escritor do store)
**Pesquisa:** `docs/research/2026-09-08-consolidador-mem0.md` (sumário executivo + 4 relatórios na íntegra)

---

## O que é

O modelo do paper do mem0 (2025) em duas fases:

1. **Extração** — um LLM tira fatos das últimas M mensagens.
2. **Update** — busca as S memórias mais similares e um LLM decide
   `ADD` / `UPDATE` / `DELETE` / `NOOP` para cada fato novo contra as antigas.

É a segunda fase que faz a memória parar de crescer para sempre e passar a
**evoluir**: corrigir o que mudou, invalidar o que ficou falso, ignorar o
redundante.

> **A pesquisa corrigiu a premissa.** O mem0 OSS 2.0.0 (2026-04-14)
> abandonou UPDATE/DELETE por uma chamada ADD-only + dedup MD5 ("nothing is
> overwritten"), com issues documentando contradições coexistindo no store.
> A paridade que vale é a do paper e do Graphiti (invalidação bi-temporal),
> não a do mem0 atual. Era JSON mode, não tool call.

---

## Contexto — o esqueleto já está no anchor

| Peça | Onde | Estado |
|---|---|---|
| `MemoryOperation` = ADD / UPDATE / DELETE / NONE | `protocols/memory.py:18` | ✅ o enum do mem0 |
| `MemoryConsolidator.consolidate(new, existing)` | `protocols/memory.py:113` | ✅ protocolo pronto; **`(DELETE, None)` é inaplicável** (sem id do alvo) |
| `_store_with_consolidation` | `pipeline/memory_steps.py:28` | único aplicador; **ignora DELETE**; passa `list_all()` inteiro |
| `SimilarityConsolidator` | `memory/consolidator.py:21` | hash + cosseno. Erra contradição por design (≥0,85 → merge) |
| `CallbackExtractor` | `memory/extractor.py:19` | extração é callable do usuário |
| `MemoryManager` | `memory/manager.py` | não conhece extractor/consolidator; sem hook de fim de turno |
| `LLMGraphExtractor` / `TierCompactor` | `ingestion/graph_extractors.py:241`, `memory/compactor.py` | o padrão LLM injetado → JSON → fail-soft a reusar |
| `MemoryEntry.expires_at` + filtro nos 5 backends | `models/memory.py:60`; rel. 3 §A.4 | a invalidação já existe como mecanismo |

Falta: `LLMExtractor`, `LLMConsolidator`, o ramo DELETE, o encaixe no turno e
o golden set. O resto é encaixe.

---

## Decisões (fechadas 2026-09-08, todas nas recomendadas)

1. **Gate de novidade:** hash igual → NONE sem LLM; `existing` vazio ou max
   cosseno < `new_threshold` (com `embed_fn`) → ADD sem LLM; o resto vai ao
   LLM com os S candidatos (top-5 por fato via cosseno, senão os
   `max_candidates=20` mais recentes). **Cosseno alto nunca decide NOOP**
   (MemStrata: AUROC 0,59 contradição vs paráfrase).
2. **DELETE = invalidação:** `expires_at=now` + `metadata["invalidated_by"]`.
   Zero schema novo. Complementos: ramo DELETE no aplicador; decorador do
   grafo faz `unlink_item` em entrada expirada; `retention` no GC.
3. **Dois prompts** (extractor e consolidator), com o gate entre eles.
4. **Encaixe no turno:** `MemoryManager.remember()` sync; `Agent` chama após
   o `try/finally` de `stream`/`astream`, antes de `TurnFinished`;
   `asyncio.to_thread` no async; `remember_every=1` (contador) e `remember()`
   público para flush manual. Sem debounce por timer.
5. **Formato da decisão:** JSON no `content` validado por Pydantic
   (`Decision(op, target, content)`), alvos por índice inteiro, NONE
   implícito, órfão de UPDATE → ADD, órfão de DELETE → ignorado com warning.
6. **UPDATE in-place** com id preservado, hash recomputado e
   `metadata["previous_content"]`. DELETE só para "deixou de ser verdade e
   nada substitui".
7. **Avaliação:** golden set de ~48 casos com gabarito por substring +
   `evaluation/consolidation.py`; baseline `SimilarityConsolidator` em CI
   (deve falhar contradição), `LLMConsolidator` live via `claude_cli`.

Herdadas do plano original: entra como implementação do protocolo existente;
o LLM é o do agente, injetado; `SimilarityConsolidator` continua default.

---

## Execução (um commit por fase, suíte verde antes de cada)

### Fase A — protocolo, DELETE aplicado, grafo, GC
- [x] `protocols/memory.py`: docstring de `consolidate` — `existing` =
      candidatos; DELETE carrega a existente a invalidar; NONE → `None`.
- [x] `pipeline/memory_steps.py`: ramo DELETE grava a entry marcada
      (`store.add`), `(DELETE, None)` = no-op documentado.
- [x] `ingestion/graph_extractors.py`: `GraphIndexingEntryStore.add` com
      `entry.is_expired` → grava + `unlink_item`.
- [x] `memory/gc.py`: `retention: timedelta | None` — expirado com
      `invalidated_by` só cai após `retention`.
- [x] Testes: ramo DELETE; decorador; GC retention; `expires_at` no contrato
      compartilhado para SQLite/Postgres/Redis.

### Fase B — `LLMExtractor` + `LLMConsolidator`
- [x] `memory/extractor.py`: `LLMExtractor(llm, *, roles=("user","assistant"))`.
- [x] `memory/consolidator.py`: `LLMConsolidator(llm, *, embed_fn=None,
      new_threshold=0.3, top_k=5, max_candidates=20)`; `Decision` Pydantic;
      hash-gate; seleção de S; faixa "novo"; um `invoke`; materialização;
      fail-soft global = ADD.
- [x] Exports (`memory/__init__.py`, `anchor/__init__.py`) + smoke.
- [x] Testes com `FakeLLM` (JSON fixo): SP→Rio UPDATE (id e hash), café,
      paráfrase NONE, DELETE invalida, órfãos, JSON ruim/provider erro,
      `existing==[]` e hash-gate não chamam o LLM, extractor filtra `tool`.

### Fase C — `MemoryManager.remember()` + `Agent`
- [x] Manager: `extractor`, `consolidator`, `extract_window=10`,
      `remember_every=1`; `remember()` devolve as ops e dispara
      `on_extraction`/`on_consolidation` se houver callbacks.
- [x] `agent.py`: 2 linhas em `stream` e `astream` após o `finally`.
- [x] Testes: `remember()` com stubs; grafo mantido em UPDATE/DELETE via
      `remember()`; `Agent` sync/async chama 1× por turno completo e 0× em
      abandono; `remember_every`.

### Fase D — golden set + `evaluation/consolidation.py`
- [x] `tests/fixtures/consolidation_golden.jsonl` (~48 casos, 10 cenários).
- [x] `evaluation/consolidation.py`: `ConsolidationCase`,
      `load_consolidation_set`, `evaluate_consolidator`, relatório com
      `state_ok`, `size_ok`, `probes_ok`, `passed` (estado final por substring —
      UPDATE in-place ≡ DELETE+ADD; `op_accuracy`/`llm_calls` ficaram de fora:
      ops equivalentes e custo medido no teste live); `assert_metric_floor` reusado.
- [x] Teste em CI: baseline `SimilarityConsolidator` com pisos honestos;
      `tests/live/test_memory_llm_live.py` com `claude_cli`.

### Fase E — docs + CHANGELOG
- [x] `guides/memory.md`, `api/memory.md`, `api/protocols.md`,
      `api/pipeline.md`, `llms.txt`; CHANGELOG Added + Fixed.

Depois: ritual xhigh (reviewer ponytail → juiz adversarial → discussão →
correções), Review abaixo.

**Fora:** `AsyncMemoryConsolidator`, debounce por timer, `valid_from/valid_to`
em `MemoryEntry`, store de memória com embedding, tool-call structured
output, ADD-only do mem0 v2. Motivos no doc de pesquisa.

## Verificação

- "moro em SP" → "me mudei pro Rio" vira `UPDATE` (mesmo id, hash novo,
  `previous_content`), não duas memórias contraditórias.
- "não gosto de café" → "passei a gostar de café" idem.
- Fato repetido com outras palavras → `NONE`, não duplicata.
- `DELETE` some de `search`/`list_all` em todos os backends, continua em
  `list_all_unfiltered` até `retention`, e a evidência sai do grafo.
- Custo medido: chamadas de LLM por turno com e sem o gate (golden set).
- Live com `claude_cli`: o cenário SP→Rio real.

## Review

_(preencher ao final)_
