# v0.2 · #5 — Consolidador de memória guiado por LLM (paridade mem0)

**Status:** fechada (sessão 12, 2026-09-08: pesquisa, 7 decisões, fases A–E, ritual xhigh + juiz) · **Tamanho:** pequeno · **Depende de:** nada
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

### Números da frente

| O quê | Valor |
|---|---|
| Commits | `4e1661f..HEAD` na dev (pesquisa+plano, fases A–E, calibração, ritual, sweep, condições do juiz) |
| Suíte | 3140 verdes / 10 skipped (baseline 3088), Postgres real +1 (docker pgvector), ruff 150 = baseline, mypy 140 = baseline |
| Golden set (48 casos, 10 cenários) — baselines determinísticos | adiciona-tudo **0,271**; `SimilarityConsolidator` (bag-of-words, 0,5) **0,354** — por desenho: não enxergam contradição |
| Golden set — par LLM (sonnet via `claude_cli`, sem `embed_fn`) | run 1 **0,875** (99 chamadas), run 2 (prompt "só o fato atual") **0,896** (89), run 3 (pós-ritual) **0,917** (98); ~5,5 min por rodada; piso do teste live: 0,75 |
| Cenário do plano (SP→Rio) live | `UPDATE` no mesmo id `m1`, `previous_content` = "User lives in São Paulo", data relativa resolvida, **2 chamadas** (1 extração + 1 consolidação), 6,5 s |
| Custo por turno | 1 chamada de extração por turno com turnos novos + ≤1 de consolidação; hash igual, store vazio e (com `embed_fn`) fato sem vizinho acima de `new_threshold` não chamam o modelo; cada fato cujos candidatos não couberam no cap vira ADD sem chamada |

### O que os baselines mostram

Os 13 casos que o caminho determinístico passa são exatamente os cenários de
ADD (fato novo, fato temporário, contradição falsa). Mudança de fato, flip
de preferência, paráfrase, enriquecimento, contagem parcial e re-afirmação —
35 casos — exigem o modelo. É o piso de regressão em CI e a régua do par LLM.

### Ritual xhigh (2026-09-08)

8 finders (5 correção + limpeza + altitude + convenções; uma primeira rodada de
10 caiu no limite de sessão e foi refeita) → reprodução por script dos
candidatos principais → 1 sweep → juiz adversarial com duas lentes (ponytail
+ SOTA lido no código do mem0 v1, Graphiti e LangMem).

**Confirmados e corrigidos (commit `83a60d1`):**
1. Decisões em lote aplicadas sobre o snapshot: DELETE seguido de UPDATE no
   mesmo alvo ressuscitava a entrada; UPDATE seguido de DELETE gravava o texto
   antigo. → `_Batch`: cópia viva dos candidatos, encadeamento, alvo deletado
   não é tocado de novo (warning).
2. `invalidated_by` apontava para `fact.id` quando o sobrevivente era um
   UPDATE (que preserva o id do alvo) → id realmente gravado.
3. Cap de `max_candidates` cortava os candidatos de fatos ainda perguntados
   (o modelo respondia às cegas) → fato sem candidato no cap vira ADD sem
   chamada.
4. **Janela de extração era uma cauda fixa**: com `remember_every=1` cada
   turno re-extraía os 10 últimos (paráfrase furava o hash-gate, 1 chamada
   extra por fato ainda na janela; `updated_at`/`access_count` inflados) e
   com `2·remember_every > extract_window` turnos eram pulados. → cursor por
   identidade/igualdade do último turno extraído, avançado só após extração
   bem sucedida; `extract_window` = teto de turnos *novos*.
5. DELETE de alvo fora do store gravava um tombstone fantasma (a leitura
   pré-#5 de `(DELETE, new_entry)` = "descarta") → ignorado com warning.
6. `memory_type` não-hasheável escapava do fail-soft do extractor; dedupe
   comparava sem `strip`.
7. Cache de embedding limpava tudo ao encher (O(n) embeds por turno acima
   de 1000 entradas) → `functools.lru_cache` por texto (Similarity e LLM).
8. Entrada restaurada (des-expirada, mesmo texto) não voltava ao grafo — o
   early-return do decorador ignorava que `current` estava expirado.
9. `__getattr__` do decorador recursava em `copy`/`pickle`.
10. `on_consolidation` de UPDATE não carregava a entrada anterior → o
    aplicador (`apply_consolidation`) dispara o callback com `previous`.
11. Evaluator: lista plana de `ConversationTurn`, passo vazio, manager por
    passo com `_add_message` privado → um manager por caso, adders públicos,
    validadores.

**Sweep (`a08f1fc`):** erro de provider no `LLMExtractor` era engolido como
`[]` e o cursor avançava (turnos perdidos num 429) → `ask_json(fail_soft=
False)`: propaga, `after_turn` loga e o próximo turno re-tenta; fato cuja
única decisão era inutilizável sumia → ADD; overlap por `\w+` (pontuação);
cursor por igualdade para backends que reconstroem turnos; `turns` não
vazio; `ConsolidationReport.k` fixo; docs que ficaram falsas.

**Juiz — APPROVE WITH CONDITIONS (`cd1f97f`):** C1 sem `embed_fn` cada
fato via só `top_k=5` por overlap (não calibrado) e a contradição antiga sem
palavra em comum ficava fora → sem `embed_fn` o corte por fato é o cap
inteiro (o plano dizia "os 20 mais recentes"; o overlap ordena dentro do
cap); C2 turnos `tool` contavam na janela antes do filtro de roles (10 tool
calls expulsavam a mensagem do usuário) → filtrados no manager; C3 JSON com
prosa em volta era descartado → fatia do valor mais externo (o
`extract_json` do mem0); C4 doc: as 2 chamadas por turno ficam fora do
`UsageLimits`. Menores aplicados: `op` em maiúsculas aceito; callback de
UPDATE encadeado vê a versão anterior.

**Discussão com o juiz (sem SendMessage nesta sessão, registrada aqui):**
- C3(b), "resposta malformada avança o cursor = perda de fato": mantido.
  Erro de provider é transitório (re-tentar paga); resposta malformada é
  comportamento do modelo sobre a mesma entrada — re-tentar sempre custa uma
  chamada por turno até a janela expulsar os turnos. Com a fatia do C3(a) a
  taxa cai; o resto é custo aceito e documentado.
- Rota ponytail "deletar `_overlap` e ordenar por recência" (C1): recusada.
  O overlap custa 6 linhas e ordena os candidatos que compartilham palavras
  (o caso do sweep, "Rio"); sem ele a recência sozinha erra o alvo com
  palavra em comum quando o store passa do cap.
- Rota ponytail "deletar `extract_window`" (C2): recusada. O primeiro
  `remember()` numa conversa longa preexistente e `remember_every=5` precisam
  do teto; o filtro de `tool` resolve a causa.

### Limpeza que o ritual trouxe

`apply_consolidation` mora em `memory/consolidator.py` (o manager deixa de
importar o pipeline; `memory_steps` importa de lá); `ask_json` em `_text.py`
une os três `invoke → strip_markdown_fences → json.loads → fail-soft`
(graph extractor, extractor, consolidator); `load_jsonl` compartilhado;
`FakeLLM` no `conftest`; `assert_metric_floor` tipado por união; parâmetros
sem chamador (`store_factory`, `k`) removidos; métricas booleanas; sem thread
hop no `astream` quando não há memória.

### Desvios do plano

- **Sem `embed_fn` a seleção não é "os 20 mais recentes"**: é overlap de
  palavras com recência no desempate, até o cap por fato. Motivo acima.
- **`retention` do GC vale para toda entrada expirada**, não só invalidadas
  — mais simples e o juiz concordou; documentado no `MemoryGarbageCollector`.
- **`op_accuracy`/`llm_calls` fora do relatório**: UPDATE in-place ≡
  DELETE+ADD no estado final; custo medido no teste live.
- **Erros de provider no `LLMExtractor` propagam** (o plano dizia fail-soft
  para o par): o manager é a camada fail-soft e guarda os turnos para
  re-tentar; o consolidador continua fail-soft (ADD) porque não perde
  informação.

### Follow-ups

- As 2 chamadas por turno não entram no `UsageLimits`/`TurnDiagnostics` —
  precisa de um seam do pool fora do loop do agente.
- Entrada com TTL que expira sem escritor mantém evidência no grafo até o GC
  (pré-existente; `retention` alarga a janela). Regra de leitura no
  `GraphIndexer` seria o fix geral.
- `SimilarityConsolidator` continua errando contradição por desenho (0,354 no
  golden set) — vale um aviso no guia já feito; um dia, gate por overlap +
  LLM só na faixa cinza (decisão 1 com embeddings de verdade).
- Golden set: 4 casos penalizam comportamento defensável do modelo
  (`renamed_dog` e `back_to_sp` citam o valor antigo entre parênteses;
  `new_pb_5k`/`borrowed_car` adicionam um fato episódico a mais). Afrouxar
  ou aceitar como variância; hoje ficam como estão (piso 0,75).
- Teste live `test_golden_set_with_the_llm_pair` é `slow` (~5,5 min, ~95
  chamadas): rodar por decisão, não em CI.

### Lições

- **Interseções entre gate e cap são superfície de bug**: o gate decidia
  "perguntar" por fato e o cap cortava a união depois — cada peça certa
  isolada e errada em composição (mesma lição da frente #2).
- **Cursor, não janela**: qualquer "processe os últimos N" sobre um fluxo
  que cresce precisa de marca d'água; a janela sozinha re-processa ou pula.
- **Fail-soft tem dono**: quem engole erro decide se a entrada se perde.
  Extractor engolindo 429 = fato perdido em silêncio; o lugar certo era o
  `after_turn`, que já era fail-soft e sabe re-tentar.
- **Um `assert` de contagem de warnings prende o texto da mensagem** — 3
  testes quebraram ao unificar o helper; melhor prender a chave estável.
