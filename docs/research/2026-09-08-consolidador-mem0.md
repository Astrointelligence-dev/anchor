# Consolidador de memória guiado por LLM (frente #5) — pesquisa dirigida

**Data:** 2026-09-08 · **Baseline:** `dev` @ `3c88e66` (3088 verdes, ruff 150, mypy 140) · **Item:** `docs/plans/2026-08-28-v020-5-consolidador-mem0.md`
**Método:** 4 agentes em paralelo — (1) código-fonte clonado e lido de mem0
(2.0.20, v2.0.0, v1.0.11), LangMem + trustcall, Letta (branch `archive`),
Graphiti, e os prompts do Dream do Claude Code extraídos do binário; (2) custo,
gatilhos e gate de novidade na literatura 2025–2026 (fontes primárias com URL);
(3) invalidação temporal e avaliação (Graphiti/Zep, mem0^g, MemOS, benchmarks
de 2026, harness local); (4) mapa do código da `dev` com `arquivo:linha`. Os
quatro relatórios completos estão abaixo do sumário, na íntegra.

---

## Sumário executivo

### A premissa do doc de entrada envelheceu

| Afirmação no doc | O que a fonte diz | Onde |
|---|---|---|
| "O mem0 tem duas fases: extração + update com **tool call** ADD/UPDATE/DELETE/NOOP" | Era assim no paper (abr/2025) e até a v1.0.11 — e era JSON mode, não tool call. O mem0 OSS **2.0.0 (2026-04-14, PR #4805) abandonou UPDATE/DELETE**: 1 chamada ADD-only + dedup MD5, "memories accumulate; nothing is overwritten". O `DEFAULT_UPDATE_MEMORY_PROMPT` continua no arquivo **sem chamador**, o `uuid_mapping` é criado e nunca lido, e o `linked_memory_ids` que o prompt pede é ignorado. Issues #4896/#4956/#5867 documentam o custo: contradições coexistindo no store. | rel. 1 §2.1 (`mem0/memory/main.py:919-1205`); release v2.0.0 |
| "Busca as S memórias mais similares" | v1.0.11: busca `limit=5` **por fato** extraído, união dedup. 2.0.20: uma busca top-10 pela conversa inteira. Graphiti: até 10 duplicatas + 10 candidatas a invalidação **por aresta**. LangMem: `query_limit=5`. | rel. 1 §1 |
| "O `mem0^g` marca relação obsoleta como inválida em vez de deletar" | O paper diz isso; o código fazia `DELETE r` até v0.1.115 e só virou `SET r.valid=false, r.invalidated_at` depois da issue #4187 (mar/2026). Em 2.0.0 o grafo saiu do OSS (~4000 linhas) — virou feature do Platform. | rel. 3 §A.2 |
| "Consolidação por LLM a cada turno é cara" | Confirmado e quantificado: chamada de utilidade 2.580 ms vs novidade por cosseno 32 ms (A-MAC, ICLR'26); LightMem 18,4 chamadas/conversa vs 812 (Mem0). Ninguém maduro roda LLM síncrono por turno: mem0 Platform `PENDING` async, LangMem debounce, Letta a cada 5 turnos em background. | rel. 2 §1–2 |
| "A literatura de 2026 tem gate de novidade barato" | Tem, com uma ressalva que muda o desenho: **cosseno não separa contradição de paráfrase** (AUROC 0,59, MemStrata). O gate barato só decide "novo demais → ADD" e "hash igual → NONE"; a faixa de similaridade alta é exatamente onde o LLM precisa decidir. O `SimilarityConsolidator` erra por design nesse caso (≥0,85 → merge). | rel. 3 §A.3, §B.3 #3; rel. 2 §2 |

### Decisões em aberto do plano → recomendação

**1. Gate de novidade → sim: hash + faixa "novo"; nunca "cosseno alto = NOOP"** (confiança alta)

- hash igual → NONE sem LLM (`SimilarityConsolidator.consolidate:115,121-124` já faz; mem0 v2 MD5; Graphiti fast path verbatim).
- `existing` vazio, ou max cosseno abaixo de `new_threshold` → ADD sem LLM (mem0 v2 faz isso para tudo; custo em qualidade zero para fato realmente novo).
- resto → **uma** chamada ao LLM com os S candidatos: top-5 por fato novo via cosseno quando há `embed_fn` (reusa `_cosine_similarity`, `consolidator.py:14`), senão os `max_candidates=20` mais recentes. Sem `embed_fn` não existe faixa "novo": tudo que passa o hash vai ao LLM com os recentes.
- Evidência: A-MAC 32 ms vs 2.580 ms; LightMem 30× menos chamadas e RecMem −87% tokens sem perder acurácia; MemStrata AUROC 0,59 contra usar cosseno alto como NOOP; ConsistencyGate −1,5% F1 como teto de perda de um gate barato.

**2. DELETE → invalidação, não remoção** (confiança alta)

- `expires_at = now` + `metadata["invalidated_by"] = <id da entrada nova>`. Zero mudança de schema: os 5 backends já filtram `expires_at` em `search`/`list_all`/`search_filtered` (tabela A.4 do rel. 3); `is_expired` usa `>=` e o SQL usa `>`, então expira no mesmo instante. Fato re-afirmado depois volta como ADD (Graphiti re-cria; mem0^g re-ativa).
- Convergência: Graphiti (`invalid_at`/`expired_at`, `edge_operations.py:563-571`, nunca deleta), mem0^g pós-#4187, MemOS (`archived` + `history`), Engram/MemStrata/TOKI (2026), LangMem `enable_deletes=False` por default; as métricas de 2026 (FAMA, stale-fact-error rate) só existem com invalidação.
- Três complementos obrigatórios: (i) ramo DELETE em `_store_with_consolidation` (`pipeline/memory_steps.py:44-46`), que hoje ignora; (ii) `GraphIndexingEntryStore.add` com entrada expirada deve `unlink_item` — hoje cai no early-return `graph_extractors.py:402-403` e o grafo guarda evidência da entrada invalidada; fica no **decorador**, não no chamador, mantendo a regra da ADR-015 (todo escritor mantém o grafo); (iii) `MemoryGarbageCollector(retention=timedelta)` para o histórico não sumir no primeiro `collect()` (`gc.py:161-166`).
- Alternativa descartada: hard delete (rel. 4 §10). O argumento "o grafo já guarda histórico nas arestas" não cobre memória sem grafo, e perde a métrica.

**3. Extração e consolidação num prompt só ou dois → dois, com o gate entre eles** (confiança alta)

- O protocolo já separa `MemoryExtractor` e `MemoryConsolidator`. Um prompt só impediria o gate (chamar o consolidador depende dos fatos extraídos) e a composição livre (`LLMExtractor` + `SimilarityConsolidator` barato; `CallbackExtractor` + `LLMConsolidator`).
- Mercado: a única implementação de 1 chamada com UPDATE/DELETE é o LangMem (tools tipadas via trustcall, `existing` ≤ 5); mem0 v2 só chegou a 1 chamada removendo as duas ops; Graphiti e mem0 v1 separam. Estudo controlado "1 × 2 prompts" com a mesma retrieval: não existe.

**4. Consolidação síncrona no turno ou em background → fora do `finally`, `to_thread` no async, contador de turnos** (confiança média — a cadência default é escolha)

- Encaixe: `MemoryManager.remember()` sync — extrai de `conversation.turns[-window:]` filtrando `role in {user, assistant}` (turnos `tool` são ruído, `agent.py:1606-1616`), gate, consolida, aplica via `_store_with_consolidation`. O `Agent` chama depois do `try/finally` de `stream`/`astream` e antes de `yield TurnFinished` (`agent.py:2347` / `:2439`); no async, `await asyncio.to_thread(...)`. **Não** dentro de `_finish_turn`: repete o bug 13/5 (LLM síncrono no loop) e rodaria em abandono do generator.
- Cadência: `remember_every: int` (contador, estilo Letta `turns_counter % 5`) e `remember()` público para flush manual / fim de sessão. Default **1** (sem o footgun de conversa curta nunca consolidada; o gate segura o custo) ou **5** (Letta; SeCom mostra turno único fragmentando: 65,6 vs segmento 71,6). Recomendo 1 e documentar 5 como ajuste de custo. Debounce por timer fica fora: thread/timer numa lib sync é complexidade; quem quer background de verdade chama `remember()` de uma task própria (`ponytail:`).

### Decisões extras que a pesquisa levantou

**5. Formato da decisão → JSON no `content` validado por Pydantic; alvos por índice inteiro** (alta)

- Padrão da casa (`LLMGraphExtractor` `graph_extractors.py:257-275`, `TierCompactor`): `invoke` → `strip_markdown_fences` → `json.loads` → fail-soft com warning. Um modelo `Decision(op: Literal["add","update","delete"], target: int | None, content: str | None)` dá o schema fechado que Graphiti/trustcall obtêm por structured output, sem o plumbing de tool call — que fica como upgrade de uma linha (`tools=` + `tool_choice=`); o `claude_cli` não garante `tool_choice` (`claude_cli.py:554-559`).
- Índices contínuos `0..n` para `existing`: todos os 4 sistemas fazem (mem0 `temp_uuid_mapping`, Graphiti `idx` com validação de faixa, trustcall `field_validator`). Índice fora da faixa → warning; UPDATE órfão degrada para ADD (o fato não se perde), DELETE órfão é ignorado. NONE implícito: o que não aparece na saída não muda (LangMem; mem0 v1 pedia a lista inteira de volta com NONE — custo de saída e chance de "editar" o que devia ficar quieto). Salvaguarda anti-delete no prompt: assimetria de custo ("when unsure, leave it", Claude Code; "Necessity Principle", mem0^g).

**6. Semântica de UPDATE → in-place com id preservado e `metadata["previous_content"]`** (média)

- `store.add` sobrescreve por id em todo backend (`protocols/storage.py:231-243`): UPDATE só é UPDATE se a entry devolvida carrega o id da existente — `existing.model_copy(update={content, content_hash, updated_at})` como `manager.update_fact` (`manager.py:245-252`), recomputando o hash (o bug do `3c88e66`). Um `LLMConsolidator` que devolva `MemoryEntry(content=novo)` vira ADD silencioso.
- Prompt: UPDATE para "mesma coisa, informação diferente ou enriquecida" (SP→Rio; "gosto de café" → "sem açúcar"); DELETE só para "deixou de ser verdade e nada substitui". O exemplo 3 do prompt do mem0 v1 ("Dislikes cheese pizza" → DELETE de "Loves", sem ADD) perde informação — não copiar. Mantém a verificação do plano ("SP→Rio vira UPDATE").
- Alternativa (rel. 3 §A.4 item 5): contradição = ADD novo + DELETE(invalida) antigo com `metadata["supersedes"]` — cadeia completa de histórico. Um nível (`previous_content`) cobre o golden set; a cadeia é mudança só no consolidador se um dia precisar.

**7. Avaliação → golden set de consolidação com gabarito determinístico + baseline honesto** (alta)

- Não existe dataset público rotulado com ADD/UPDATE/DELETE/NOOP. O LoCoMo foi desacreditado (o juiz aceita 62,81% de respostas propositalmente erradas; a disputa mem0 × Zep de 2025 deixou três números para a Zep: 58,44 / 65,99 / 75,14). O que mede consolidação em 2026 mede **estado do store e stale-fact no top-k** (ForgetEval por substring, MemStrata stale-fact-error rate, MemConflict SEH@K); MemoryAgentBench FactConsolidation: Mem0 18%, Zep 7%, BM25 48%.
- Proposta: `tests/fixtures/consolidation_golden.jsonl` com ~48 casos em 10 cenários (rel. 3 §B.3; 4–10 reais do `knowledge-update` do LongMemEval, que tem 78 pares fato-antigo/fato-novo marcados) + `evaluation/consolidation.py` (~100 linhas espelhando `golden.py`) com `stale_alive_rate`, `probe_stale_hit_rate`, `dup_rate` como gate e `op_accuracy` como diagnóstico (UPDATE in-place ≡ DELETE+ADD no estado final). Roda em CI contra o `SimilarityConsolidator` — baseline que **deve falhar** contradição por design, piso de regressão como o A/B da #4 — e live com `claude_cli` para o `LLMConsolidator`. Custo por turno medido: nº de `invoke` com e sem gate (o plano pede).

### Furos no código que a frente fecha (todos pré-existentes; `arquivo:linha` no rel. 4)

1. `(DELETE, None)` é inaplicável — `protocols/memory.py:137` diz que o entry "may be None"; sem id do alvo. Correção compatível: docstring fixa "DELETE → entry = a existente a invalidar; `existing` = candidatos, não o store inteiro" + ramo no único aplicador. Zero testes cobrem DELETE hoje.
2. `_store_with_consolidation` passa `store.list_all()` inteiro (`memory_steps.py:42`) — O(n) tokens num LLM; a seleção de S vive no consolidador.
3. A busca é substring/LIKE em todos os backends (`_filters.py:41-58`; sqlite `:51-59`; postgres `:79-92`; redis `:83-95`); nenhum store de memória com embedding; `ScoredMemoryRetriever` exige um `vector_store` que ninguém alimenta.
4. `on_extraction`/`on_consolidation` nunca disparam (`callbacks.py:60-84`) — `remember()` é o primeiro lugar natural para dispará-los (2 linhas).
5. Memória síncrona no `finally` do turno async (follow-up 13/5 da sessão 8, `agent.py:2178-2191,2346,2438`) — o encaixe da #5 não pode agravar.
6. Testes de expiração só cobrem InMemory/JsonFile (`test_entry_store_shared.py:1-3`); SQLite/Postgres/Redis sem teste de `expires_at`. A decisão 2 depende disso, então o contrato compartilhado ganha os três.
7. Docs drift: `docs/docs/llms.txt:99-100` assinaturas erradas; `guides/memory.md:261-263` "never calls an LLM"; `api/memory.md:276` "ADD / UPDATE / NONE".

### Menor encaixe (rel. 4 §10, ajustado pelas decisões 2 e 7)

| # | Arquivo | Mudança |
|---|---|---|
| 1 | `protocols/memory.py:132-137` | docstring: `existing` = candidatos; DELETE carrega a existente a invalidar; NONE → `None`. Assinatura intacta; sem `AsyncMemoryConsolidator`. |
| 2 | `pipeline/memory_steps.py:44-46` | ramo DELETE: `store.add(entry.model_copy(update={"expires_at": now, "metadata": {..., "invalidated_by": ...}}))` — a entry já vem marcada do consolidador; o aplicador só grava. |
| 3 | `ingestion/graph_extractors.py:399-405` | `if entry.is_expired: inner.add(entry); graph.unlink_item(entry.id); return` (2 linhas no decorador). |
| 4 | `memory/gc.py` | `retention: timedelta \| None` — expirado com `invalidated_by` só é apagado após `retention`. |
| 5 | `memory/extractor.py` | `LLMExtractor(llm, *, roles=("user","assistant"))`: prompt → JSON `[{"content","tags"?,"memory_type"?}]` → `MemoryEntry` com `source_turns` (convenção de `CallbackExtractor.extract:77-87`); fail-soft. |
| 6 | `memory/consolidator.py` | `LLMConsolidator(llm, *, embed_fn=None, new_threshold=..., top_k=5, max_candidates=20)`: hash-gate → seleção de S → faixa "novo" → um `invoke` com `existing` indexado → `Decision` validado → materialização (UPDATE in-place com hash e `previous_content`; DELETE = marcada; órfãos com warning). Fail-soft global: warning + ADD para todos (comportamento sem consolidador). |
| 7 | `memory/manager.py` | `__slots__` + ctor: `extractor`, `consolidator`, `extract_window=10`, `remember_every=1`; `remember()` sync devolvendo as ops (dispara `on_extraction`/`on_consolidation` se houver callbacks). |
| 8 | `agent/agent.py:2347, :2439` | 2 linhas cada: `remember()` / `await asyncio.to_thread(remember)` após o `finally`, antes de `TurnFinished`. |
| 9 | `evaluation/consolidation.py` + `tests/fixtures/consolidation_golden.jsonl` | `ConsolidationCase`, `evaluate_consolidator`, relatório com as métricas do item 7; `assert_metric_floor` reusado. |
| 10 | testes | `test_llm_consolidator.py` (SP→Rio UPDATE com id e hash; café; paráfrase NONE; DELETE invalida; órfãos; fail-soft; `existing == []` e hash-gate não chamam o LLM; extractor filtra `tool`); ramo DELETE em `test_memory_steps.py`; `remember()` + grafo em `test_manager_graph.py`; `Agent` sync/async chama `remember` 1×; `expires_at` no contrato compartilhado para SQLite/Postgres/Redis; golden set com baseline; `tests/live/test_memory_llm_live.py` gateado por `claude_cli`. |
| 11 | docs + CHANGELOG | `guides/memory.md`, `api/memory.md`, `api/protocols.md`, `api/pipeline.md`, `llms.txt`; Added (feature) + Fixed (DELETE ignorado). |

### Fora de escopo, com motivo

- **`AsyncMemoryConsolidator`** — `asyncio.to_thread` cobre; o protocolo fica só sync.
- **Debounce por timer** — thread/timer numa lib sync; `remember()` público serve a quem orquestra fora.
- **`valid_from`/`valid_to` em `MemoryEntry`** — world-time vive no grafo, que já é bi-temporal; `expires_at` é o transaction-time suficiente para o store. Nenhum caso do golden set precisa de as-of.
- **Store de memória com embedding** — `embed_fn` opcional no consolidador cobre a seleção de S; indexar `MemoryEntry` num `VectorStore` é outra frente.
- **Structured output por tool call** — upgrade de uma linha quando um provider errar o JSON.
- **ADD-only do mem0 v2** — regressão de contrato para um SDK que já expõe UPDATE/DELETE, e as issues mostram o custo.

---

## Relatório 1 — Fase de update nos sistemas de referência (código-fonte lido)

# Pesquisa: fase de update/consolidação de memória em sistemas para agentes LLM

Data: 2026-09-08. Tudo abaixo foi lido do código-fonte clonado em
`/private/tmp/claude-501/-Users-arthurgranja-github-anchor/51691894-7963-40bb-9db8-2c9902ed6c81/scratchpad/repos/`
(caminhos relativos a essa pasta). Onde não consegui ler, está marcado **não verificado**.

Snapshots usados:

| Repo | Commit / tag | Data | Pasta |
|---|---|---|---|
| mem0 (atual) | `dae67f74`, `version = "2.0.20"` | 2026-09-04 | `mem0/` |
| mem0 (legado, último 1.x) | tag `v1.0.11` | 2026-04-06 | `mem0-v1.0.11/` |
| mem0 (primeiro 2.x) | tag `v2.0.0` | 2026-04-16 | `mem0-v2.0.0/` |
| LangMem | `f8c7ebd6` | 2026-09-02 | `langmem/` |
| trustcall (dependência do LangMem) | `8c7312b5` | 2025-07-17 | `trustcall/` |
| Letta (branch `archive`, v0.16.8) | `56ba9c25` | 2026-08-13 | `letta-archive/` |
| Graphiti | `7db96847` | 2026-09-08 | `graphiti/` |

Aviso importante sobre dois dos alvos:
- **mem0 abandonou o ADD/UPDATE/DELETE no OSS em 2.0.0 (2026-04-14).** O pipeline atual é *ADD-only*. O prompt de update ainda existe no arquivo mas não tem chamador. A pesquisa cobre as duas versões.
- **Letta moveu o código para `letta-ai/letta-code` (TypeScript); o repo `letta-ai/letta` é só landing page** (`letta/README.md`: "The retired Letta V1 server source is preserved on the `archive` branch"). Li o branch `archive` (Python, v0.16.8). O código TS atual **não foi verificado**.

---

## 1. Tabela comparativa

| Sistema | Janela de extração M | Similares S no contexto | Formato da decisão | Ops suportadas | Salvaguarda anti-delete | DELETE = hard ou invalidação | Sync/async | Chamadas LLM por `add` |
|---|---|---|---|---|---|---|---|---|
| **mem0 2.0.20 (OSS, atual)** | Todas as `messages` do `add()` concatenadas (`parse_messages`) + últimas 10 msgs da sessão (SQLite) | top_k=10 por busca vetorial **da conversa inteira** (1 embedding) | JSON mode (`response_format={"type":"json_object"}`), `{"memory":[{id,text,attributed_to,linked_memory_ids}]}` | **Só ADD** (+ dedup por MD5 exato) | Não há delete pelo LLM. `linked_memory_ids` é pedido no prompt mas **ignorado pelo código** | `Memory.delete()` manual = hard delete no vector store + linha `DELETE` no `history` (SQLite) | Sync (`Memory`) e async (`AsyncMemory`) com pipeline idêntico | **1** (+ embeddings em batch; +0 para entidades, spaCy) |
| **mem0 v1.0.11 (legado)** | Todas as `messages` concatenadas | Por **cada fato extraído**: busca vetorial `limit=5`; união dedup por id (S ≤ 5·F) | JSON mode; `{"memory":[{id,text,event,old_memory}]}`, `event ∈ {ADD,UPDATE,DELETE,NONE}` | ADD/UPDATE/DELETE/NONE | ids temporários `"0".."n"`; id desconhecido → `KeyError` capturado e ação ignorada; prompt: "return the IDs ... from the input IDs only" | Vector store: **hard delete** + `history` com `is_deleted=1`. Grafo (mem0^g): **soft delete** `SET r.valid=false, r.invalidated_at=datetime()` | Sync; vector e grafo em paralelo via `ThreadPoolExecutor`; plataforma tem `async_mode=True` default | **2** (extração + update) + 2–3 no grafo (entidades, relações, delete) |
| **LangMem** | `messages` passados ao manager (uma sessão `<session_id>`) | `MemoryStoreManager`: `query_limit=5` (buscas em janelas dilatadas 1,2,4… msgs, ou queries geradas por LLM), top-5 por score | **Tool calls** (trustcall): schema `Memory` para insert, `PatchDoc` (JSON Patch RFC 6902) para update, `RemoveDoc` para delete | insert/update/delete via flags `enable_inserts/updates/deletes` | `enable_deletes=False` por padrão em `create_memory_manager` e `create_memory_store_manager`; `RemoveDoc.json_doc_id` validado por `field_validator` contra os ids existentes; `PatchDoc` com id inexistente é descartado com log | **Hard delete** (`store.adelete`) | Sync ou async; `ReflectionExecutor` faz debounce (`after_seconds`, cancela task pendente por `thread_id`) | 1 por passo (`max_steps`, default 1) + 1 opcional de geração de queries; +1 por `phase` |
| **Letta (archive, v0.16.8)** | Sleeptime: mensagens desde `last_processed_message_id` + resposta atual | Não há busca: o sleeptime agent vê **os blocos inteiros de core memory** (compartilhados por `block_ids`) | **Tool calls** do agente: `memory_replace(old,new)`, `memory_insert(line)`, `memory_rethink(bloco inteiro)`, `memory_finish_edits` | replace/insert/rewrite; delete = `new_string=""` | `memory_replace` exige `old_string` único e verbatim (erro se 0 ou >1 ocorrências); rejeita prefixos de linha; prompt manda "be selective" | **Hard** (o bloco é sobrescrito; sem histórico no código lido) | Assíncrono: `safe_create_task` em background a cada `turns_counter % 5 == 0` | 0 no hot path; N passos do agente sleeptime (loop até `memory_finish_edits`) |
| **Graphiti / Zep** | 1 episódio (`add_episode`) + `previous_episodes` p/ contexto | Por **aresta extraída**: (a) arestas entre os mesmos nós + hybrid search RRF limit=10 restrita a elas; (b) hybrid search global limit=10 como candidatas a invalidação | **Structured output** Pydantic `EdgeDuplicate{duplicate_facts:[int], contradicted_facts:[int]}` | dedup (reuse) / contradiz → invalida; nunca delete | idx fora de faixa → `logger.warning` e ignorado; dedup só do range EXISTING FACTS; contradição só muda `invalid_at` se `valid_at` antigo < `valid_at` novo | **Invalidação temporal**: `invalid_at` + `expired_at=now`; a aresta fica no grafo (`entity_edges = resolved_edges + invalidated_edges`) | Async (`semaphore_gather`) | 1 `resolve_edge` por aresta com candidatos (small model) + 1 `extract_timestamps` por aresta nova (+ extração de nós/arestas) |
| **Claude Code auto memory + Dream** | Dream: logs/transcripts dos últimos 1–3 dias | Todo o diretório de memória (`ls`, lê index + topic files) | Edição de arquivos markdown por um sub-agente | merge / rewrite / delete de arquivos; prune do índice | Prompt: "When unsure, leave it" (team), não editar CLAUDE.md, não promover memórias pessoais; índice ≤200 linhas / 25KB | **Hard** (arquivo apagado/reescrito) | Background, gatilho 24h + ≥5 sessões (**fonte secundária**) | 1 sessão de agente por dream |

---

## 2. mem0

### 2.1 Pipeline atual (2.0.20): "V3 phased batch pipeline", ADD-only

`mem0/mem0/memory/main.py:879` `_add_to_vector_store(self, messages, metadata, filters, infer, prompt=None)`:

- `infer=False` (`main.py:880-917`): cada mensagem não-system vira uma memória bruta, evento `ADD`, sem LLM.
- `infer=True` — comentário literal `# === V3 PHASED BATCH PIPELINE ===` (`main.py:919`):
  - **Phase 0** (`main.py:922-924`): `last_messages = self.db.get_last_messages(session_scope, limit=10)`; `parsed_messages = parse_messages(messages)` (concatenação `"user: ...\nassistant: ...\n"`, `mem0/memory/utils.py:61-75`).
  - **Phase 1** (`main.py:927-934`): **um** embedding da conversa inteira e `self.vector_store.search(query=parsed_messages, vectors=query_embedding, top_k=10, filters=search_filters)`. Ou seja **S = 10, busca por mensagem (conversa), não por fato**.
  - Mapeamento anti-alucinação (`main.py:936-941`):
    ```python
    # Map UUIDs to integers (anti-hallucination)
    existing_memories = []
    uuid_mapping = {}
    for idx, mem in enumerate(existing_results):
        uuid_mapping[str(idx)] = mem.id
        existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})
    ```
    **`uuid_mapping` nunca é lido depois disso** (grep: só `main.py:935,937` e o gêmeo async `2597,2599`). É resquício do pipeline antigo.
  - **Phase 2** (`main.py:943-972`): `system_prompt = ADDITIVE_EXTRACTION_PROMPT` (+ `AGENT_CONTEXT_SUFFIX` se só `agent_id`); `user_prompt = generate_additive_extraction_prompt(existing_memories=..., new_messages=parsed_messages, last_k_messages=last_messages, custom_instructions=prompt or self.custom_instructions)`; **uma** chamada `self.llm.generate_response(..., response_format={"type": "json_object"})`. Falha do LLM agora **propaga `LLMError`** (`main.py:965-972`) em vez de retornar `[]`.
  - Parse (`main.py:975-988`): `json.loads(...).get("memory", [])`, fallback `extract_json`.
  - **Phase 3** (`main.py:996-1006`): `embed_batch` dos textos.
  - **Phase 4/5** (`main.py:1010-1041`): dedup por `hashlib.md5(text)` contra os hashes das 10 memórias recuperadas e dentro do batch; monta payload com `data`, `text_lemmatized` (BM25), `hash`, `created_at`, `updated_at`, `attributed_to`.
  - **Phase 6** (`main.py:1050-1085`): `vector_store.insert` em batch; `db.batch_add_history` com `event="ADD"`.
  - **Phase 7** (`main.py:1088-1191`): extração de entidades via spaCy (`extract_entities_batch`) para uma coleção paralela `{collection}_entities` com `linked_memory_ids` — **isto é o único uso de `linked_memory_ids` no código**; o campo `linked_memory_ids` que o prompt pede no JSON de saída do LLM **não é lido** (grep `linked_memory_ids` em `main.py`: linhas 624-693 e 1157-1175, todas do entity store).
  - **Phase 8** (`main.py:1193-1205`): `db.save_messages(messages, session_scope)`; retorna `[{"id","memory","event":"ADD"}]`.

**Custo por `add`: 1 chamada LLM + 1 embedding da query + 1 `embed_batch` das memórias + 1 `embed_batch` das entidades.**

Documentação oficial confirma o desenho (`mem0/docs/migration/oss-v2-to-v3.mdx`, seção "How the New Algorithm Works"):

> Input conversation → Retrieve top-10 related existing memories (for deduplication context) → Single LLM call: extract all distinct new facts → Batch embed extracted memories → Hash-based deduplication (MD5, prevents exact duplicates) → Batch insert into vector store → Entity extraction (for entity matching)
>
> The previous algorithm used two LLM calls: one to extract candidate facts, one to decide ADD/UPDATE/DELETE actions against existing memories. The new algorithm collapses this into a single call that only adds. The model spends its capacity on understanding the input rather than diffing against existing state.

E na seção "Update Add Calls":

> The ADD-only model means memories accumulate over time. When information changes, the new fact is stored alongside the old one. Retrieval handles ranking: the most relevant, current information surfaces first.

`docs/changelog/sdk.mdx`, entrada `v2.0.0` (2026-04-14):

> **Single-Pass Extraction:** Replaced 2-LLM-call pipeline with additive extraction using `ADDITIVE_EXTRACTION_PROMPT`. Memories accumulate via `linked_memory_ids`: no more UPDATE/DELETE events (#4805)
> **`add()` returns ADD-only events**: No more `"UPDATE"` or `"DELETE"` events. Memories accumulate; nothing is overwritten (#4805)
> **External Graph Store Removed (OSS):** `mem0/memory/graph_memory.py`, `memgraph_memory.py`, `kuzu_memory.py`, `apache_age_memory.py`, and `mem0/graphs/` ... deleted, about 4,000 lines.
> **Core:** Guard `temp_uuid_mapping` lookups against LLM-hallucinated IDs with safe `.get()` and warnings (#4674)

A doc afirma "+20 pontos no LoCoMo (71.4 → 91.6) e +26 no LongMemEval (67.8 → 93.4)" — benchmark do próprio vendor, não verificado.

#### Trechos relevantes do `ADDITIVE_EXTRACTION_PROMPT` (`mem0/configs/prompts.py:468-945`)

Papel:

> You are a Memory Extractor — a precise, evidence-bound processor responsible for extracting rich, contextual memories from conversations. Your sole operation is ADD: identify every piece of memorable information and produce self-contained, contextually rich factual statements.

Seção "Existing Memories" (como os S=10 similares são usados):

> Memories currently in the system relevant to this conversation. Formatted as: [{"id": "uuid-string", "text": "..."}, ...]
> Use these ONLY for deduplication and linking — do NOT extract new memories from Existing Memories. Your extractions must come exclusively from New Messages. If new information in New Messages is semantically equivalent to an Existing Memory with no meaningful new context, skip it.
> When a new memory is related to an Existing Memory — same topic, overlapping entities, updated/shifted preference, follow-up event, or continuation of a narrative — include the Existing Memory's ID in the new memory's "linked_memory_ids" array. Your ADD output IDs remain sequential ("0", "1", ...) but linked_memory_ids uses the UUIDs from this list.
> IMPORTANT: An existing memory about an entity (e.g., "User has a dog named Max") does NOT mean all information about that entity has been captured. [...] Only skip extraction when the specific fact or event itself is already captured

(Observação: o código passa `"id": "0","1",...` — inteiros — e não UUIDs, então a instrução "linked_memory_ids uses the UUIDs from this list" é inconsistente com o input real; como o campo é ignorado, não tem efeito.)

Regra "quando em dúvida, extraia":

> **When in doubt, extract.** A slightly redundant memory is far less costly than a missing one. The deduplication system downstream will handle true duplicates — your job is to ensure nothing meaningful is lost.

Contradição vira **link**, não delete ("Memory Linking"):

> - **Contradiction**: New information that conflicts with an existing memory

Anti-contaminação de contexto ("Integrity Rules"):

> **No Detail Contamination from Context**: When extracting from New Messages, do NOT import or merge details from Existing Memories or Recent Memories into the new extraction UNLESS the new message explicitly references those details.

Formato de saída:

```
{
  "memory": [
    {"id": "0", "text": "First extracted memory", "attributed_to": "user", "linked_memory_ids": ["uuid-of-related-existing-memory"]},
    {"id": "1", "text": "Second extracted memory", "attributed_to": "assistant"}
  ]
}
- id (string, required): Sequential integers as strings starting at "0".
- text (string, required): A contextually rich, self-contained factual statement (15-80 words).
- attributed_to (string, required): "user" | "assistant"
- linked_memory_ids (array of strings, optional)
- If nothing is worth extracting, return: {"memory": []}
```

Builder do user prompt (`prompts.py:1016-1062`): seções `## Summary`, `## Last k Messages`, `## Recently Extracted Memories`, `## Existing Memories`, `## New Messages`, `## Observation Date`, `## Current Date`, `## Custom Instructions` (opcional), `# Output:`. No OSS, `summary` e `recently_extracted_memories` chegam sempre `None` (`main.py:950-955` só passa `existing_memories`, `new_messages`, `last_k_messages`, `custom_instructions`) — são portas para o Platform ("Ported from platform/backend/shared/core/config/prompts.py", `prompts.py:465-467`).

### 2.2 Pipeline legado (v1.0.11): 2 chamadas, ADD/UPDATE/DELETE/NONE

`mem0-v1.0.11/mem0/memory/main.py:475` `_add_to_vector_store(self, messages, metadata, filters, infer)`:

- **(e) janela M**: `parsed_messages = parse_messages(messages)` (`main.py:513`) — todas as mensagens do `add()` numa string; extração **uma vez por chamada** de `add`, não por mensagem.
- **Chamada 1 — extração** (`main.py:515-533`): `get_fact_retrieval_messages(parsed_messages, is_agent_memory)` → `USER_MEMORY_EXTRACTION_PROMPT` ou `AGENT_MEMORY_EXTRACTION_PROMPT` (`mem0/memory/utils.py:15-28`), ou `custom_fact_extraction_prompt`; `response_format={"type":"json_object"}`; saída `{"facts": [str]}` normalizada por `normalize_facts` (`utils.py`).
- **(a) similares S** (`main.py:559-582`): **para cada fato novo**, `embedding_model.embed(new_mem)` e `vector_store.search(query=new_mem, vectors=..., limit=5, filters=search_filters)`; união dos resultados com dedup por id (`unique_data`). Logo S ≤ 5 × nº de fatos. A busca é **por fato**, não pela mensagem.
- **(c) ids temporários** (`main.py:584-588`):
  ```python
  # mapping UUIDs with integers for handling UUID hallucinations
  temp_uuid_mapping = {}
  for idx, item in enumerate(retrieved_old_memory):
      temp_uuid_mapping[str(idx)] = item["id"]
      retrieved_old_memory[idx]["id"] = str(idx)
  ```
  Motivo declarado no comentário: LLMs corrompem UUIDs de 36 chars ao copiá-los; índices `"0".."n"` são copiáveis e qualquer id inventado cai fora do dicionário.
- **Chamada 2 — decisão** (`main.py:590-599`): `get_update_memory_messages(retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt)` enviado como **uma única mensagem `user`** (sem system prompt), `response_format={"type":"json_object"}`. Pulada se não há fatos (`main.py:552-553, 617-618`).
- **(b) schema do evento** — ver o prompt abaixo: `{"memory":[{"id","text","event","old_memory"}]}`; `old_memory` "Required only if the event is UPDATE".
- **Aplicação** (`main.py:620-698`):
  - `ADD` (`631-640`): `_create_memory` com embedding cacheado.
  - `UPDATE` (`641-658`): `_update_memory(memory_id=temp_uuid_mapping[resp.get("id")], ...)`; retorna `previous_memory: resp.get("old_memory")`.
  - `DELETE` (`659-667`): `_delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])`.
  - `NONE` (`668-694`): se `agent_id`/`run_id` presentes, atualiza só os ids de sessão no payload (`vector_store.update(vector=None, payload=...)`); senão "NOOP".
  - **(d) salvaguardas**: entrada sem `text` é pulada (`625-628`); qualquer exceção por item (incl. `KeyError` de id alucinado em `temp_uuid_mapping[...]`) é capturada por `except Exception` e logada, seguindo para o próximo item (`695-696`); resposta vazia/JSON inválido → `{}` (`604-616`). O changelog 2.0.0 cita #4674 trocando por `.get()` com warning, mas nessa tag ainda é indexação direta. Não há limite numérico de deletes nem verificação de que o `text` de um DELETE bate com a memória. `infer=False` (`476-511`) pula todo o pipeline (sem LLM, sem delete).
- **(g) DELETE no vector store** (`main.py:1340-1364` em v1.0.11; `mem0/mem0/memory/main.py:2100-2128` em 2.0.20): `self.vector_store.delete(vector_id=memory_id)` — **hard delete** — seguido de `self.db.add_history(memory_id, prev_value, None, "DELETE", ..., is_deleted=1)`. A tabela `history` (SQLite, `mem0/memory/storage.py:107-124`): `id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role`. `UPDATE` também grava `old_memory`/`new_memory` (`main.py:2078-2087`). Ou seja: **o vetor some; o histórico textual fica no SQLite** (`Memory.history(memory_id)`).
- **(e) `async_mode`**: não existe no OSS. No client da plataforma (`mem0-v1.0.11/mem0/client/main.py:164-166`): `if "async_mode" not in kwargs: kwargs["async_mode"] = True`. Em 2.0.x o parâmetro foi removido do client: "`async_mode` and `output_format` removed (async by default, v1.1 always)" (`docs/migration/oss-v2-to-v3.mdx`). No OSS v1.0.11, `add()` roda vector store e grafo em paralelo com `concurrent.futures.ThreadPoolExecutor` (`main.py:458-465`).
- **(f) categorias / procedural**: `custom_categories` só existe no client da plataforma (`mem0/client/project.py:179,398,722`; `client/types.py:27,116`); no OSS, `Memory.project.update(custom_categories=...)` é um shim que **sempre levanta `ValueError(_PROJECT_UPDATE_UNSUPPORTED_ERROR)`** (`mem0/mem0/memory/main.py:461-471`). Procedural memory: `add(memory_type="procedural_memory", agent_id=...)` → `_create_procedural_memory` (`main.py:1993-2036`): 1 chamada LLM com `PROCEDURAL_MEMORY_SYSTEM_PROMPT` (`prompts.py:326-404`, um resumo verbatim de passos de agente) + `"Create procedural memory of the above conversation."`, gravada como **uma** memória `ADD` com `memory_type=procedural_memory`. Sem consolidação.

#### `DEFAULT_UPDATE_MEMORY_PROMPT` na íntegra (`mem0/configs/prompts.py:176-324`; idêntico em v1.0.11 e 2.0.20 — verificado por `diff`)

```
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "User is a software engineer"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Name is John",
                    "event" : "ADD"
                }
            ]

        }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it.
If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information.
Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved facts.
Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then you do not need to update it because they convey the same information.
If the direction is to update the memory, then you have to update it.
Please keep in mind while updating you have to keep the same ID.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "I really like cheese pizza"
            },
            {
                "id" : "1",
                "text" : "User is a software engineer"
            },
            {
                "id" : "2",
                "text" : "User likes to play cricket"
            }
        ]
    - Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Loves cheese and chicken pizza",
                    "event" : "UPDATE",
                    "old_memory" : "I really like cheese pizza"
                },
                {
                    "id" : "1",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "2",
                    "text" : "Loves to play cricket with friends",
                    "event" : "UPDATE",
                    "old_memory" : "User likes to play cricket"
                }
            ]
        }


3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Dislikes cheese pizza"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "DELETE"
                }
        ]
        }

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "NONE"
                }
            ]
        }
```

Nota: o exemplo 3 (DELETE) mostra o caso "Dislikes cheese pizza" derrubando "Loves cheese pizza" **sem** gerar um ADD do fato novo — o prompt trata contradição como remoção pura, o que perde informação.

#### `get_update_memory_messages` na íntegra (`mem0/configs/prompts.py:406-463`)

```python
def get_update_memory_messages(retrieved_old_memory_dict, response_content, custom_update_memory_prompt=None):
    if custom_update_memory_prompt is None:
        global DEFAULT_UPDATE_MEMORY_PROMPT
        custom_update_memory_prompt = DEFAULT_UPDATE_MEMORY_PROMPT

    if retrieved_old_memory_dict:
        current_memory_part = f"""
    Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

    ```
    {retrieved_old_memory_dict}
    ```

    """
    else:
        current_memory_part = """
    Current memory is empty.

    """

    return f"""{custom_update_memory_prompt}

    {current_memory_part}

    The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

    ```
    {response_content}
    ```

    You must return your response in the following JSON structure only:

    {{
        "memory" : [
            {{
                "id" : "<ID of the memory>",                # Use existing ID for updates/deletes, or new ID for additions
                "text" : "<Content of the memory>",         # Content of the memory
                "event" : "<Operation to be performed>",    # Must be "ADD", "UPDATE", "DELETE", or "NONE"
                "old_memory" : "<Old memory content>"       # Required only if the event is "UPDATE"
            }},
            ...
        ]
    }}

    Follow the instruction mentioned below:
    - Do not return anything from the custom few shot prompts provided above.
    - If the current memory is empty, then you have to add the new retrieved facts to the memory.
    - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made.
    - If there is an addition, generate a new key and add the new memory corresponding to it.
    - If there is a deletion, the memory key-value pair should be removed from the memory.
    - If there is an update, the ID key should remain the same and only the value needs to be updated.

    Do not return anything except the JSON format.
    """
```

Em 2.0.20 essa função **não tem chamador** (grep `get_update_memory_messages` no pacote: só a definição em `prompts.py:406`). `FACT_RETRIEVAL_PROMPT` (`prompts.py:15-61`) e `get_fact_retrieval_messages` (`utils.py:15-33`) também ficaram sem uso em `main.py`, que importa só `ADDITIVE_EXTRACTION_PROMPT` e `generate_additive_extraction_prompt` (`main.py:20,23`).

### 2.3 mem0^g (grafo) em v1.0.11 — invalidação de relação obsoleta

`mem0-v1.0.11/mem0/memory/graph_memory.py:76-94` `MemoryGraph.add(data, filters)`:
1. `_retrieve_nodes_from_data` (LLM, tool `extract_entities`) → `entity_type_map`
2. `_establish_nodes_relations_from_data` (LLM) → `to_be_added`
3. `_search_graph_db(node_list=..., filters)` → relações existentes dos nós citados
4. `_get_delete_entities_from_search_output(search_output, data, filters)` (`347-381`): LLM com `DELETE_RELATIONS_SYSTEM_PROMPT` + **tool call** `delete_graph_memory(source, relationship, destination)` (`mem0/graphs/tools.py:342-370`); só o que vier como tool call vira `to_be_deleted`.
5. `_delete_entities` (`383-438`) — **não deleta**:
   ```cypher
   MATCH (n {source_props})-[r:{relationship}]->(m {dest_props})
   WHERE r.valid IS NULL OR r.valid = true
   SET r.valid = false, r.invalidated_at = datetime()
   ```
   com o comentário (`420-422`): "Soft-delete: mark relationship as invalid instead of removing it, enabling temporal reasoning over historical graph state. See: https://github.com/mem0ai/mem0/issues/4187".
6. `_add_entities` (merge de nós por similaridade de embedding, threshold 0.9 em `_search_source_node`/`_search_destination_node`, `662-737`).

Prompt de delete (`mem0-v1.0.11/mem0/graphs/utils.py:57-92`) — a salvaguarda anti-delete é textual:

> 2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
>    - Outdated or Inaccurate: The new information is more recent or accurate.
>    - Contradictory: The new information conflicts with or negates the existing information.
> 3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
> [...]
> 7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.
>
> For example:
> Existing Memory: alice -- loves_to_eat -- pizza
> New Information: Alice also loves to eat burger.
> Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Custo do grafo em v1.0.11: 3 chamadas LLM por `add` (entidades, relações, delete) além das 2 do vector store.

### 2.4 (h) O que mudou em 2026 (tags verificadas via `git ls-remote --tags`)

Tags: `v1.0.0`…`v1.0.11` (1.x), `v2.0.0b0..b2`, `v2.0.0` (2026-04-14/16), `v2.0.1`…`v2.0.20` (2026-09-02). Mudanças relevantes para consolidação:
- **2.0.0 (PR #4805)**: pipeline ADD-only de 1 chamada; hybrid search (semântico + BM25 + entidades, com `threshold=0.1` e `rerank=False` default); coleção `_entities`; janela rolante de 10 mensagens por escopo em SQLite (`storage.py:246-310`); remoção do grafo externo do OSS (~4000 linhas) — "graph memory is now a built-in, always-on Mem0 Platform feature"; remoção de `async_mode`, `output_format`, `enable_graph`, `force_add_only`, `immutable` do client; `custom_fact_extraction_prompt` → `custom_instructions`; default `top_k` 100→20.
- **2.0.1–2.0.20**: sem mudança no algoritmo de consolidação (changelog `docs/changelog/sdk.mdx` lista correções de vector stores, deps, Neptune, telemetria). Verificado que `v2.0.0/mem0/memory/main.py:699` já tem `# === V3 PHASED BATCH PIPELINE ===`.

Consequência prática: **no mem0 OSS atual não existe mais "fase de consolidação"**. Contradições ficam lado a lado no store e a resolução é delegada à recuperação (score temporal/relevância). O único mecanismo de deduplicação é MD5 do texto exato + a instrução "skip if semantically equivalent" dentro da extração.

---

## 3. LangMem

### 3.1 Schemas e manager

`langmem/src/langmem/knowledge/extraction.py`:
- `ExtractedMemory(typing.NamedTuple)`: `id: str`, `content: BaseModel` (`78-80`).
- `Memory(BaseModel)`: docstring "Call this tool once for each new memory you want to record. Use multi-tool calling to record multiple new memories."; único campo `content: str` (`86-92`). É literalmente o **tool schema** do insert.
- `MemoryManager.__init__(model, *, schemas=(Memory,), instructions=_MEMORY_INSTRUCTIONS, enable_inserts=True, enable_updates=True, enable_deletes=False)` (`221-236`); `create_memory_manager` repete os defaults (`536-548`).
- `ainvoke` (`238-330`): `extractor = create_extractor(self.model, tools=list(self.schemas), enable_inserts=..., enable_updates=..., enable_deletes=..., existing_schema_policy=False)` (`253-260`) — **toda a mecânica de UPDATE/DELETE é do trustcall**. Loop de `max_steps` (default 1); a partir do passo 2 adiciona a tool `Done` ("Only call this tool once you are done forming & consolidating memories", `212-218`). Identificação do resultado (`281-289`):
  ```python
  mem_id = (
      r.json_doc_id
      if hasattr(r, "__repr_name__") and r.__repr_name__() == "RemoveDoc"
      else rmeta.get("json_doc_id", str(uuid.uuid4()))
  )
  ```
  i.e. `PatchDoc` bem-sucedido traz `json_doc_id` nos metadados (= UPDATE do id existente), `RemoveDoc` carrega o id a remover, senão é INSERT com uuid novo. Memórias existentes não tocadas são mantidas (`291-293`). Em multi-step, o modelo recebe `"Memory {rid} {action}."` como tool message (`296-320`).
- `_prepare_messages` (`445-463`) — o prompt de decisão é uma mensagem `user`:
  > `{instructions}\n\nEnrich, prune, and organize memories based on any new information. If an existing memory is incorrect or outdated, update it based on the new information. All operations must be done in single parallel multi-tool call. Avoid duplicate extractions. <session_{id}>...</session_{id}>`
- `_MEMORY_INSTRUCTIONS` (`179-209`), trecho:
  > 2. **Compare & Update**
  >    - Attend to novel information that deviates from existing memories and expectations.
  >    - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.
  >    - Remove incorrect or redundant memories while maintaining internal consistency
  > [...] Prefer dense, complete memories over overlapping ones.
- `_filter_response` (`492-518`): `RemoveDoc` só é devolvido na resposta final se o id for **externo** (veio de `existing`); remoções de memórias criadas no mesmo loop são descartadas silenciosamente.

### 3.2 `RemoveDoc` / `PatchDoc` (trustcall `trustcall/trustcall/_base.py`)

- Construção do conjunto de tools (`665-696`): `new_tools = [PatchDoc] if enable_updates else []`; se `enable_inserts`, adiciona os schemas do usuário; `tool_choice = "PatchDoc" if not enable_deletes else "any"`.
- `_setup` (`700-760`): serializa `existing` como `<instance id=... schema_type=...>` dentro de `<existing>` e injeta no system prompt: "Generate JSONPatches to update the existing schema instances. If you need to extract or insert *new* instances of the schemas, call the relevant function(s)."; **só se `enable_deletes and existing`** liga o `RemoveDoc` (`748-756`).
- `RemoveDoc` (`1191-1214`): schema gerado **por conjunto de ids permitidos**, com validador:
  ```python
  class RemoveDoc(BaseModel):
      """Use this tool to remove (delete) a doc by its ID."""
      json_doc_id: str = Field(..., description=f"ID of the document to remove. Must be one of: {allowed_ids}")
      @field_validator("json_doc_id")
      def validate_doc_id(cls, v):
          if v not in allowed_ids:
              raise ValueError(f"Document ID '{v}' not found. Available IDs: {sorted(allowed_ids)}")
  ```
  Erro de validação alimenta o loop de correção do trustcall (`PatchFunctionErrors`, `1217-1245`) — o LLM recebe o erro e re-tenta.
- `_teardown` (`762-840`): para `PatchDoc`, busca o `json_doc_id` em `existing`; se não achar, `logger.error("Could not find existing schema ...")` e **descarta a tool call** (`776-799`); aplica `jsonpatch.apply_patch` (`_apply_patch`, `1691-1697`) e marca `updated_docs[tc["id"]] = json_doc_id` (`821`).
- `PatchDoc` (`1280`) usa JSON Patch (RFC 6902) — o update é um diff estrutural, não reescrita do texto.

### 3.3 `MemoryStoreManager` (persistência + busca dos similares)

`extraction.py:830-860`: `__init__(..., enable_inserts=True, enable_deletes=True, query_model=None, query_limit=5, namespace=("memories","{langgraph_user_id}"), store=None, phases=None)`. Atenção: aqui `enable_deletes=True` por default na classe, mas a factory pública `create_memory_store_manager` (`1666-1677`) usa **`enable_deletes=False`** e `query_limit=5`.

`ainvoke` (`1010-1130`):
- Busca dos similares: se há `query_model`, o LLM gera queries via tool `search_memory` (`1017-1030`); senão `utils.get_dialated_windows(messages, query_limit // 4)` (`utils.py:103-119`: janelas das últimas 1, 2, 4, … mensagens) e `store.asearch(namespace, query=...)` por janela; `_sort_results` mantém os top `query_limit` por score (`985-996`).
- `store_based = [(stable_id, kind, content)]` com `stable_id = uuid5(namespace+key)` (`982-983`).
- Chama o `MemoryManager` com `existing=store_based`; `_apply_manager_output` (`940-975`) separa `RemoveDoc` → `removed_ids`, atualizações in-place, inserts em `ephemeral`.
- `phases` extras (`1080-1094`): cada fase é outro `create_memory_manager` com `enable_deletes=True` default e instrução "You are a memory manager. Deduplicate, consolidate, and enrich these memories." (`977-985`).
- Persistência (`1096-1128`): `store.aput` só para o que mudou; `store.adelete(ns, key)` para `removed_ids` — **hard delete**.

### 3.4 Reflection (background/debounce)

`langmem/src/langmem/reflection.py`:
- `ReflectionExecutor(reflector, namespace=None, /, *, url, client, sync_client, store)` (`95-137`): string → `RemoteReflectionExecutor` (LangGraph Platform, `runs.create(..., multitask_strategy="rollback", after_seconds=after_seconds)` `180-200`); `Runnable` → `LocalReflectionExecutor`.
- `LocalReflectionExecutor.submit(payload, config, *, after_seconds=0, thread_id=SENTINEL)` (`284-337`): resolve `thread_id` do config; **se já existe task pendente para o mesmo `thread_id`, `cancel_event.set()` + `future.cancel()`** (`316-320`); enfileira em `PriorityQueue` com chave `time.time() + after_seconds` (`337`).
- Worker `_process_queue` (`400-470`): thread daemon=False; dorme até `execute_at`, pula tasks canceladas, injeta o `store` no `Runtime` e chama `self._reflector.invoke(task.payload)`.
- Doc `langmem/docs/docs/guides/delayed_processing.md`: "Wait 30 minutes before processing. If new messages arrive before then: 1. Cancel pending processing task 2. Reschedule with new messages included" e "This debouncing ensures you process complete conversation context instead of fragments."

---

## 4. Letta (branch `archive`, v0.16.8)

### 4.1 Core memory como blocos editáveis por tools

`letta-archive/letta/functions/function_sets/base.py`:
- `core_memory_append(agent_state, label, content)` (`246-260`): `new_value = current_value + "\n" + content`.
- `core_memory_replace(agent_state, label, old_content, new_content)` (`263-277`): "To delete memories, use an empty string for new_content"; `ValueError` se `old_content not in current_value`; `str.replace` (todas as ocorrências).
- `rethink_memory(agent_state, new_memory, target_block_label)` (`280-300`): reescreve o bloco; cria se não existir.
- **Sleep-time set v2** ("Based off of: anthropic-quickstarts computer-use-demo tools/edit.py", `309`):
  - `memory_replace(agent_state, label, old_string, new_string)` (`311-388`): rejeita `old_string`/`new_string` com prefixo `Line N:` ou o banner de aviso de linha (`350-361`); `expandtabs`; **exige exatamente 1 ocorrência** — 0 → `ValueError("No replacement was performed, old_string ... did not appear verbatim")`, >1 → `ValueError("Multiple occurrences ... in lines {lines}. Please ensure it is unique.")` (`366-377`); delete = `new_string=""` (docstring, `329`).
  - `memory_insert(agent_state, label, new_string, insert_line=-1)` (`391-449`): insere linha(s) na posição; `-1` = append.
  - `memory_apply_patch` (`452-485`): patch estilo codex multi-bloco; no archive levanta `NotImplementedError` (executado server-side).
  - `memory_rethink(agent_state, label, new_memory)` (`488-517`): "completely rewrite the contents of a memory block. Use this tool to make large sweeping changes (e.g. when you want to condense or reorganize the memory blocks), do NOT use this tool to make small precise edits".
  - `memory_finish_edits(agent_state)` (`520-528`): no-op que encerra o loop.
- Conjuntos (`letta/constants.py:118-143`): `BASE_MEMORY_TOOLS = ["core_memory_append", "core_memory_replace", "memory", "memory_apply_patch"]`; `BASE_SLEEPTIME_TOOLS = ["memory_replace", "memory_insert", "memory_rethink", "memory_finish_edits"]`.
- Não há versionamento/histórico do bloco no código lido: `update_block_value` sobrescreve. (Há um `GitEnabledBlockManager` em `server.py:746` — "enable_git_memory_for_agent" — **não verificado** em detalhe.)

### 4.2 Sleep-time agents: quando roda e o que faz

Criação (`letta-archive/letta/server/server.py:756-789` `create_sleeptime_agent_async`): agente `AgentType.sleeptime_agent` com `block_ids=[block.id for block in main_agent.memory.blocks]` (**os mesmos blocos, compartilhados**) + bloco próprio `memory_persona`; grupo `SleeptimeManager(manager_agent_id=main_agent.id, sleeptime_agent_frequency=5)`. `group_manager.py:94-98`: `turns_counter = -1` na criação.

Gatilho (`letta-archive/letta/groups/sleeptime_multi_agent_v4.py:132-169` `run_sleeptime_agents`):
```python
if self.group.sleeptime_agent_frequency is not None and self.group.sleeptime_agent_frequency > 0:
    turns_counter = await self.group_manager.bump_turns_counter_async(...)
if self.group.sleeptime_agent_frequency is None or (
    turns_counter is not None and turns_counter % self.group.sleeptime_agent_frequency == 0
):
    ...
    last_processed_message_id = await self.group_manager.get_last_processed_message_id_and_update_async(...)
    for sleeptime_agent_id in self.group.agent_ids:
        sleeptime_run_id = await self._issue_background_task(...)
```
Ou seja: **a cada 5 turnos do agente principal** (default), dispara em background (`safe_create_task`, `171-200`) um `Run` para cada sleeptime agent; sem frequência configurada roda a cada turno.

Execução (`202-289` `_participant_agent_step`): monta o transcript com `message_manager.list_messages(after=last_processed_message_id, before=response_messages[0].id)` + a resposta atual, e envia **uma** mensagem `user`:

> `<system-reminder>` You are a sleeptime agent - a background agent that asynchronously processes conversations after they occur. IMPORTANT: You are NOT the primary agent. [...] Your primary role is memory management. Review the conversation and use your memory tools to update any relevant memory blocks with information worth preserving. Check your memory_persona block for any additional instructions or policies. `</system-reminder>` Messages: {messages_text}

e roda `LettaAgentV3.step(...)` (`262-270`) — o agente itera chamando `memory_replace`/`memory_insert`/`memory_rethink` até `memory_finish_edits`.

System prompt do sleeptime agent (`letta-archive/letta/prompts/system_prompts/sleeptime_v2.py`), trechos:

> You run in the background, organizing and maintaining the memories of an agent assistant who chats with the user.
> [...] Use your precise tools to make narrow edits, as well as broad tools to make larger comprehensive edits. [...] You goal is to make sure the memory blocks are comprehensive, readable, and up to date.
> When writing to memory blocks, make sure to be precise when referencing dates and times (for example, do not write "today" or "recently", instead write specific dates and times, because "today" and "recently" are relative, and the memory is persisted indefinitely).
> Multi-step editing: You should continue memory editing until the blocks are organized and readable, and do not contain redundant and outdate information, then you can call a tool to finish your edits.
> Skipping memory edits: If there are no meaningful updates to make to the memory, you call the finish tool directly. Not every observation warrants a memory edit, be selective in your memory editing, but also aim to have high recall.

Não há busca de "similares": o agente vê os blocos inteiros no system prompt (com números de linha só para visualização) e edita por string-match. A "salvaguarda" é mecânica (match único e verbatim) e não semântica.

---

## 5. Graphiti / Zep

### 5.1 Modelo temporal da aresta

`graphiti/graphiti_core/edges.py:264-282` `EntityEdge`: `fact`, `fact_embedding`, `episodes`, `expired_at` ("datetime of when the node was invalidated" — i.e., quando o sistema soube), `valid_at` ("when the fact became true"), `invalid_at` ("when the fact stopped being true"), `reference_time`, `attributes`. Bi-temporal: `valid_at/invalid_at` = tempo do mundo; `expired_at` = tempo de ingestão.

### 5.2 Fluxo `resolve_extracted_edges` (`graphiti/graphiti_core/utils/maintenance/edge_operations.py:325-535`)

1. Dedup exato intra-batch por `(source_uuid, target_uuid, normalize(fact))` (`344-357`).
2. Embeddings das arestas extraídas (`362`).
3. `valid_edges_list = EntityEdge.get_between_nodes(driver, src, dst)` por aresta (`365-370`) — candidatas a **duplicata** (mesmos endpoints).
4. `related_edges` = hybrid search (`EDGE_HYBRID_SEARCH_RRF` = bm25 + cosine com RRF, `search_config_recipes.py:111-116`; `DEFAULT_SEARCH_LIMIT = 10`, `search_config.py:29`) pelo `fact`, restrita aos uuids de (3) (`395-406`).
5. `edge_invalidation_candidates` = mesma hybrid search **sem** restrição de endpoints (`410-419`), removendo as que já estão em (4) (`423-431`). Então **S ≈ até 10 duplicatas + até 10 candidatas a invalidação por aresta nova**.
6. `resolve_extracted_edge` por aresta em paralelo (`486-503`); `new_edges` = as cujo uuid resolvido == uuid extraído (`516-523`).

### 5.3 Decisão por aresta: `resolve_extracted_edge` (`623-847`)

- Sem candidatos (`656-686`): só extrai atributos custom e timestamps (`_extract_edge_timestamps`, `576-620`, chamada LLM small com `prompt_library.extract_edges.extract_timestamps`) e retorna.
- Fast path (`688-699`): mesmo par de nós + mesmo `fact` normalizado → reusa a aresta existente e só anexa o episódio. **Zero LLM.**
- Contexto com **índices contínuos** (`703-712`): `existing_edges = [{idx: i, fact}]` para duplicatas; invalidation candidates com `idx = len(related_edges) + i`.
- Chamada LLM (`733-738`): `prompt_library.dedupe_edges.resolve_edge(context)`, `response_model=EdgeDuplicate`, `model_size=ModelSize.small`.
- Validação (`742-751`): `duplicate_facts` fora de `[0, len(related_edges))` → `logger.warning` e filtrado; usa só o **primeiro** duplicado como aresta resolvida (`753-757`). `contradicted_facts` fora de `[0, max_valid_idx]` → warning e ignorado (`762-781`).
- Timestamps só para arestas novas (`820-821`).
- `if resolved_edge.invalid_at and not resolved_edge.expired_at: resolved_edge.expired_at = now` (`830-831`).
- Se algum candidato tem `valid_at` **posterior** ao da aresta nova, **a aresta nova é que expira** (`834-845`) — informação mais recente vence mesmo chegando fora de ordem.
- `resolve_edge_contradictions(resolved_edge, invalidation_candidates)` (`538-573`): para cada contradita, pula se os intervalos não se sobrepõem; senão, se `edge.valid_at < resolved_edge.valid_at`: `edge.invalid_at = resolved_edge.valid_at; edge.expired_at = edge.expired_at or utc_now()`. **Só marca; nunca deleta.** Se faltam `valid_at` em algum lado, nada é invalidado (condições exigem `is not None`).
- Persistência: `graphiti/graphiti_core/graphiti.py:1156` `entity_edges = resolved_edges + invalidated_edges` — as invalidadas são salvas de volta com `expired_at/invalid_at` (`models/edges/edge_db_queries.py:96-98`). Deleção de arestas só existe em `remove_episode` (`graphiti.py:1766-1793`). A busca não filtra expiradas por padrão; `SearchFilters.valid_at/invalid_at/expired_at` (`search/search_filters.py:62-65`) são opt-in.

### 5.4 Prompt `dedupe_edges.resolve_edge` (`graphiti/graphiti_core/prompts/dedupe_edges.py:45-101`)

Nota: **`invalidate_edges.py` não existe mais** no diretório `prompts/` (listado); a invalidação foi fundida em `resolve_edge` via `contradicted_facts`.

```python
class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int]   # 'List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.'
    contradicted_facts: list[int]  # 'List of idx values of contradicted facts (from full idx range). Empty list if none.'
```

System: "You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates."

User (trecho):

> NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.
> IMPORTANT constraints:
> - duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
> - contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
> - The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)
> [...]
> 2. CONTRADICTION DETECTION:
>    - Determine which facts the NEW FACT contradicts from either list.
>    - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
> <EXAMPLE>
> EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
> NEW FACT: "Alice works at Acme Corp as a senior engineer"
> Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)
> EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
> NEW FACT: "Bob ran 3 miles on Wednesday"
> Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
> </EXAMPLE>

Prompt de timestamps (`prompts/extract_edges.py:242-271`): "You extract temporal bounds from facts. NEVER hallucinate dates." / "If the fact is ongoing (present tense), set valid_at to REFERENCE TIME. If a change or end is expressed, set invalid_at to the relevant time. Leave both null if no time is stated or resolvable."

Resumo do critério de contradição: **o LLM diz "contradiz"; o código só converte isso em invalidação se as datas `valid_at` justificarem (antigo < novo), e a invalidação é uma data, não um delete.**

---

## 6. Claude Code auto memory + "Dream"

### 6.1 Fonte primária oficial (docs)

`https://code.claude.com/docs/en/memory` (lido em 2026-09-08):
- "Auto memory lets Claude accumulate knowledge across sessions without you writing anything. As it works, Claude saves four kinds of notes for itself" — tipos `user`, `feedback`, `project`, `reference` no frontmatter.
- Layout: `~/.claude/projects/<project>/memory/` com `MEMORY.md  # Index, one line per memory, loaded into every session` e "one topic file per memory".
- "The first 200 lines of `MEMORY.md`, or the first 25KB, whichever comes first, are loaded at the start of every conversation. Content beyond that threshold is not loaded at session start. Claude keeps `MEMORY.md` concise by moving detailed notes into separate topic files."
- "After Claude writes to `MEMORY.md`, Claude Code measures the file against the 200-line and 25KB read limits. If the file is near a limit, Claude Code reminds Claude to shorten it: keep one line per entry, move detail into topic files, and merge or drop stale entries. If the file is over a limit, the write still succeeds, but Claude Code returns an error telling Claude to rewrite the index".
- "Claude Code doesn't load topic files such as `user_role.md` or `feedback_testing.md` at startup. Claude reads them on demand".
- "When Claude writes a memory file that begins with YAML frontmatter, Claude Code records the write time in a `modified` frontmatter field" (v2.1.214+).
- **A página oficial não menciona "dream", consolidação periódica nem sub-agente.** `https://code.claude.com/docs/en/commands` não lista `/dream`. O `CHANGELOG.md` de `anthropics/claude-code` (até 2.1.263) não tem entradas com "dream"/"auto memory"/"consolidat".

### 6.2 Prompts extraídos do binário (fonte quase-primária)

Repo `https://github.com/Piebald-AI/claude-code-system-prompts` (snapshot v2.1.263, 2026-09-05; README: "extracted directly from Claude Code's compiled source code"). Três arquivos:
- `system-prompts/agent-prompt-dream-memory-consolidation.md` (1573 tokens).
- `system-prompts/system-prompt-dream-claude-md-memory-reconciliation.md` (`ccVersion: "2.1.212"`).
- `system-prompts/system-prompt-dream-team-memory-handling.md` (`ccVersion: "2.1.98"`).

Prompt principal (fetch em `https://raw.githubusercontent.com/Piebald-AI/claude-code-system-prompts/main/system-prompts/agent-prompt-dream-memory-consolidation.md`), estrutura verbatim:

> # Dream: Memory Consolidation
> You are performing a dream — a reflective pass over your memory files. Synthesize what you've learned recently into durable, well-organized memories so that future sessions can orient quickly.
> [...]
> ## Phase 1 — Orient
> - `ls` the memory directory to see what already exists
> - Read `${INDEX_FILE}` to understand the current index
> - Skim existing topic files so you improve them rather than creating duplicates
> - `ls -R logs/` — recent activity logs (one file per session under `YYYY/MM/DD/`)
> ## Phase 2 — Gather recent signal
> 1. **Session logs** [...] Read the most recent 1–3 days of sessions
> 2. **Existing memories that drifted** — facts that contradict something you see in the codebase now
> 3. **Transcript search** — [...] grep the JSONL transcripts for narrow terms
> Don't exhaustively read transcripts. Look only for things you already suspect matter.
> ## Phase 3 — Consolidate
> [...] Focus on:
> - Merging new signal into existing topic files rather than creating near-duplicates
> - Converting relative dates ("yesterday", "last week") to absolute dates so they remain interpretable after time passes
> - Deleting contradicted facts — if today's investigation disproves an old memory, fix it at the source
> ## Phase 4 — Prune and index
> Update `${INDEX_FILE}` so it stays under ${INDEX_MAX_LINES} lines AND under ~25KB. It's an **index**, not a dump — each entry should be one line under ~150 characters: `- [Title](file.md) — one-line hook`. Never write memory content directly into it.
> - Remove pointers to memories that are now stale, wrong, or superseded
> - Demote verbose entries: if an index line is over ~200 chars, it's carrying content that belongs in the topic file
> - Add pointers to newly important memories
> - Resolve contradictions — if two files disagree, fix the wrong one
> [...] Return a brief summary of what you consolidated, updated, or pruned. If nothing changed (memories are already tight), say so.

(Há uma variante `HAS_AUTOGENERATED_MEMORY_INDEX` em que o índice é montado a partir do frontmatter `name`/`description` de cada arquivo e a Phase 4 vira só "Prune".)

Reconciliação com CLAUDE.md (verbatim, ccVersion 2.1.212):

> - **Memory is stale** — CLAUDE.md and the memory describe different procedures for the same task: CLAUDE.md is the maintained, checked-in source. Delete the memory, or rewrite it to agree if it carries context worth keeping (the *why* is still useful but the *how* is wrong).
> - **CLAUDE.md may be stale** — the memory is clearly dated after CLAUDE.md and explicitly corrects it: do NOT edit CLAUDE.md during a dream. Annotate the memory with "contradicts CLAUDE.md — verify which is current" and list it in your summary so the user can update CLAUDE.md.
> - **Not a conflict** — the memory adds detail CLAUDE.md doesn't cover, or narrows a CLAUDE.md rule with a stated reason. Leave it.
> A `feedback` memory's "Why: the user corrected me" framing is not evidence it's newer than CLAUDE.md — CLAUDE.md may have been updated since.

Memória de time (verbatim, ccVersion 2.1.98):

> - **Phase 4 — be conservative pruning `team/`:**
>   - DO delete or fix a team memory that is clearly contradicted by the current code, or that a newer team memory marks as superseded.
>   - DO NOT delete a team memory just because you don't recognize it or it isn't relevant to *your* recent sessions — a teammate may rely on it.
>   - When unsure, leave it. A stale team memory costs little; deleting a teammate's load-bearing note costs a lot.
> Do not promote personal memories into `team/` during a dream — that's a deliberate choice the user makes via `/remember`, not something to do reflexively.

### 6.3 Fontes secundárias (só o que dizem; não verificadas no código)

- `https://www.implicator.ai/anthropic-adds-auto-dream-to-claude-code-fixing-memory-decay-between-sessions/`: "Developers discovered it behind a server-side flag called `tengu_onyx_plover` before Anthropic published any documentation"; gatilho: "Twenty-four hours need to pass since the last consolidation, and more than five sessions must have occurred. Both gates must open."; ativação via `/memory`, manual via `/dream`; cita `dev.to/akari_iku` e `github.com/Piebald-AI/claude-code-system-prompts`.
- `https://antoniocortes.com/en/2026/03/30/auto-memory-and-auto-dream-how-claude-code-learns-and-consolidates-its-memory/` (2026-03-30): mesmas duas condições (24h + >5 sessões), 4 fases, "Background execution. You can keep working while consolidation runs."; sem versão; cita claudefa.st.
- `https://mem0.ai/blog/how-memory-works-in-claude-code`: "200 lines maximum. If your index grows beyond that, the system silently truncates it", "25KB maximum"; dual-gate 24h + 5 sessões; diz ter lido "Claude Code's memory source code" mas não dá versão nem link.
- Busca por "services/autoDream/" ou análise de Simon Willison: **não encontrada** em `simonwillison.net`/`anthropic.com`/`code.claude.com` (WebSearch com `site:` retornou zero). **Não verificado.**

---

## 7. O que isso recomenda para um SDK Python com `enum ADD/UPDATE/DELETE/NONE` e `consolidate(new_entries, existing) -> [(op, entry)]`

### 7.1 Recomendações concretas (com a evidência)

1. **Referencie `existing` por índice inteiro no prompt, mapeie de volta no código e ignore ids desconhecidos com warning.** Todos os quatro sistemas que fazem decisão por LLM fazem isso: mem0 v1 `temp_uuid_mapping` (`mem0-v1.0.11/.../main.py:584-588`), mem0 v2 ainda cria o mapa (`main.py:936-941`), Graphiti `idx` contínuo com validação de faixa e `logger.warning` (`edge_operations.py:742-751, 762-771`), trustcall valida `json_doc_id` com `field_validator` (`_base.py:1191-1214`) e descarta `PatchDoc` órfão (`762-799`). O seu `consolidate` deve rejeitar `(UPDATE|DELETE, entry)` cujo alvo não esteja em `existing` — nunca `KeyError` escapando nem silêncio sem log (mem0 v1 engolia no `except Exception`, `main.py:695`).

2. **Não confie no LLM para DELETE; faça DELETE = invalidação com data, e reversível.** Graphiti nunca apaga: `invalid_at`/`expired_at` (`edge_operations.py:538-573`, `graphiti.py:1156`); mem0^g trocou `DELETE` por `SET r.valid=false, r.invalidated_at` explicitamente para permitir raciocínio temporal (`graph_memory.py:420-433`, issue #4187); mem0 vector store guarda `history` com `old_memory` + `is_deleted=1` (`storage.py:107-124`, `main.py:2100-2128`); LangMem e `create_memory_store_manager` vêm com `enable_deletes=False` por default (`extraction.py:544, 1675`). Recomendação: o enum `DELETE` deve produzir `entry` com `invalidated_at` (e opcionalmente `superseded_by=<id do ADD/UPDATE que o contradisse>`), e a busca filtra `invalidated_at is None` por padrão. Hard delete só por API explícita.

3. **Trate contradição como `ADD` do novo + invalidação do antigo, não como `DELETE` puro nem `UPDATE` que sobrescreve.** O prompt do mem0 v1 (exemplo 3) apaga "Loves cheese pizza" ao ver "Dislikes cheese pizza" e não guarda o fato novo — perda de informação. Graphiti separa `duplicate_facts` (reusar) de `contradicted_facts` (invalidar e manter o novo) e admite que o mesmo item seja ambos (`dedupe_edges.py:66-76`). Isso mapeia direto para o seu protocolo: `[(ADD, novo), (DELETE→invalidate, antigo)]` como par atômico.

4. **UPDATE só para enriquecimento, com `old_memory`/patch explícito e verificação de match.** mem0 v1 exige `old_memory` no evento UPDATE (`get_update_memory_messages`) e devolve `previous_memory`; Letta exige `old_string` **único e verbatim** e falha se 0 ou >1 ocorrências (`base.py:366-377`); trustcall usa JSON Patch em vez de reescrita (`_base.py:1280, 1691-1697`). Recomendação: o `(UPDATE, entry)` deve carregar `entry.previous` (texto que o LLM acredita estar substituindo); o SDK compara com o `existing[id].text` real e degrada para `NONE`+warning se não bater (ou usa isso como sinal de alucinação). Guarde `previous` no histórico.

5. **Dimensione S por fato novo, não por mensagem, e deduplique por hash antes do LLM.** mem0 v1: busca `limit=5` **por fato** (`main.py:566-576`); Graphiti: até 10 duplicatas + 10 candidatas **por aresta** (`edge_operations.py:395-419`); mem0 v2 caiu para uma busca única top-10 pela conversa inteira — mais barato, mas só funciona porque abandonou UPDATE/DELETE. Antes de qualquer LLM, dedupe exato: mem0 v2 MD5 (`main.py:1019-1031`), Graphiti `(src, dst, normalize(fact))` (`344-357`) e fast path verbatim sem LLM (`688-699`). Isso corta chamadas em re-ingestão.

6. **Use structured output com schema fechado (tool call ou Pydantic), não JSON livre.** Graphiti (`response_model=EdgeDuplicate`), LangMem/trustcall (tools `Memory`/`PatchDoc`/`RemoveDoc` com validadores que devolvem o erro ao modelo), Letta (tools). mem0 v1 usava `json_object` + parser tolerante (`remove_code_blocks`, `extract_json`) e ainda precisou de `normalize_facts` para modelos pequenos (`utils.py:96-118`). Um `Literal["ADD","UPDATE","DELETE","NONE"]` + `target_idx: int | None` validado por Pydantic elimina a classe inteira de bugs.

7. **Só ofereça DELETE ao modelo quando houver `existing`, e diga o que NÃO deletar.** trustcall só liga `RemoveDoc` se `enable_deletes and existing` (`_base.py:748-756`); mem0^g: "DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes" + exemplo pizza/burger (`graphs/utils.py:66-84`); Claude Code team memory: "When unsure, leave it. A stale team memory costs little; deleting a teammate's load-bearing note costs a lot." Graphiti só invalida se `valid_at` antigo < novo (`556-571`) — regra determinística acima do LLM. Coloque no prompt a assimetria de custo e, no código, um teto (ex.: rejeitar respostas em que >50% de `existing` vira DELETE, ou exigir `valid_at` para invalidar).

8. **Ancore datas: converta relativas em absolutas na escrita e guarde `valid_at` quando existir.** Letta: "do not write 'today' or 'recently', instead write specific dates" (`sleeptime_v2.py`); Claude Code Dream Phase 3: "Converting relative dates ... to absolute dates"; mem0 v2 tem `## Observation Date` como único âncora temporal; Graphiti extrai `valid_at/invalid_at` com "NEVER hallucinate dates ... Leave both null if no time is stated" (`extract_edges.py:242-271`). Sem `valid_at`, a regra 7 não consegue decidir ordem — então extraia-o (nullable) no mesmo schema do ADD.

9. **Separe hot path de consolidação; a consolidação é assíncrona e debounced.** Letta roda o sleeptime agent em `safe_create_task` a cada 5 turnos (`sleeptime_multi_agent_v4.py:140-160`); LangMem `ReflectionExecutor.submit(after_seconds=...)` cancela a task pendente do mesmo `thread_id` (`reflection.py:316-337`); mem0 Platform `async_mode=True` por default (v1) e "async by default" (v2); Claude Code Dream: 24h + 5 sessões (fonte secundária). O `consolidate(new_entries, existing)` deve ser puro (sem I/O) para ser chamável tanto no hot path quanto por um worker com debounce por escopo.

10. **Registre o efeito de cada op num histórico append-only.** mem0 `history(memory_id)` (SQLite `history`, `storage.py:139-181`), Graphiti mantém arestas expiradas, Claude Code usa `modified` no frontmatter. Sem isso não há como auditar um DELETE errado do LLM — e todos os sistemas assumem que ele vai acontecer.

### 7.2 O que evitar

- **Não replique o mem0 v2 "ADD-only + retrieval resolve"** a menos que você também tenha hybrid search com sinal temporal e aceite acumular contradições no store. É a decisão explícita deles ("nothing is overwritten"), tomada junto com a remoção do grafo e do `UPDATE/DELETE` do OSS; o campo `linked_memory_ids` prometido no changelog não é sequer lido no código (`main.py` grep). Para um SDK que já expõe `UPDATE/DELETE`, isso seria regressão de contrato.
- **Não use UUIDs nem textos como chave de alvo no prompt.** Todo sistema que faz isso mapeia para inteiros; textos como chave (mem0^g `source -- rel -- dest` em `_delete_entities`) dependem de normalização frágil (`_remove_spaces_from_entities`, `sanitize_relationship_for_cypher`).
- **Não faça o LLM reemitir `NONE` para tudo que não mudou** (mem0 v1 pede a lista inteira de volta com `NONE`). Custa tokens de saída e dá ao modelo a chance de "editar" o que devia ficar quieto. LangMem mantém intocado o que não foi citado (`extraction.py:291-293`); Graphiti só pede índices. Deixe `NONE` implícito: o que não aparece na saída é `NONE`.
- **Não misture extração e decisão numa chamada só se você quer UPDATE/DELETE.** A única implementação de chamada única (mem0 v2) só existe porque removeu as duas ops. Graphiti e mem0 v1 separam extração (facts/edges) da resolução contra `existing`; LangMem faz numa chamada mas via tools tipadas e com `existing` pequeno (`query_limit=5`).
- **Não hard-delete no hot path por decisão de LLM.** Nenhum dos sistemas com temporalidade faz isso; os que fazem (mem0 v1 vector store, LangMem store, Letta blocos) compensam com histórico, `enable_deletes=False` ou match verbatim.
- **Não deixe a exceção genérica engolir id alucinado** (mem0 v1 `main.py:695-696`) — o changelog 2.0.0 teve que registrar "Guard `temp_uuid_mapping` lookups against LLM-hallucinated IDs" como correção (#4674).

### 7.3 Esboço mínimo do protocolo (síntese, não código dos repos)

```python
class Op(Enum): ADD, UPDATE, DELETE, NONE

class Decision(BaseModel):            # structured output do LLM
    op: Literal["ADD", "UPDATE", "DELETE"]   # NONE implícito
    target_idx: int | None            # índice em `existing`; obrigatório p/ UPDATE/DELETE
    text: str | None                  # obrigatório p/ ADD/UPDATE
    previous: str | None              # UPDATE: texto que acredita substituir (verificado)
    valid_at: str | None              # ISO; null se não declarado

def consolidate(new_entries, existing) -> list[tuple[Op, Entry]]:
    # 0. dedup exato por hash → NONE sem LLM
    # 1. S = top-k por *entrada nova* (k≈5), união, índices 0..n
    # 2. 1 chamada structured; validar idx ∈ range, previous == existing[idx].text
    # 3. DELETE → entry.invalidated_at = now (+ superseded_by); nunca remove
    # 4. contradição → [(ADD, novo), (DELETE, antigo)] no mesmo lote
    # 5. registrar (op, old, new) no histórico
```

---

## Apêndice: comandos usados para verificação

- Clones: `git clone --depth 1` de mem0, langmem, graphiti, letta, trustcall; `git fetch --depth 1 origin tag v1.0.11|v2.0.0` + `git worktree add` para mem0; `git fetch --depth 1 origin archive` para letta.
- `diff` entre `DEFAULT_UPDATE_MEMORY_PROMPT` de v1.0.11 e 2.0.20: idêntico.
- `grep -n "uuid_mapping" mem0/mem0/memory/main.py` → só linhas 935, 937, 2597, 2599 (escrita, nunca leitura).
- `grep -rn "get_update_memory_messages" mem0/mem0` → só a definição.
- `ls graphiti/graphiti_core/prompts/` → sem `invalidate_edges.py`.
- `curl https://api.github.com/repos/Piebald-AI/claude-code-system-prompts/contents/system-prompts` → três arquivos `*dream*`.


---

## Relatório 2 — Custo, gatilhos e gate de novidade (literatura 2025–2026)

# Consolidação de memória guiada por LLM: custo, gatilhos e gate de novidade

Data da pesquisa: 2026-09-08. Fontes primárias (arXiv, docs oficiais, código, issues). Toda afirmação numérica traz URL; onde não achei, está marcado **não encontrado**.

---

## Sumário executivo

Três achados dominam o resto do relatório:

1. **O mercado saiu do "extrair + decidir ADD/UPDATE/DELETE/NOOP a cada `add()`".** O mem0 OSS (v2.0.0, 2026-04-16) abandonou o pipeline de duas chamadas do paper de 2025 por **uma chamada ADD-only + dedup por hash MD5**, e empurrou supersede/merge para o lado servidor ("Dream") — com issues abertas documentando o custo em qualidade (fatos contraditórios coexistindo). LangMem, Letta e Zep nunca fizeram o loop por turno: LangMem faz *debounce* em background, Letta roda um agente de sono a cada N passos, Zep invalida arestas em vez de deletar.
2. **O gate barato antes do LLM tem números a favor.** A-MAC (ICLR 2026): das cinco features de admissão, a única cara é a de utilidade via LLM (2.580 ms, 97,6% do custo); novidade por cosseno custa 32 ms. LightMem: pré-filtro leve (LLMLingua-2) reduz chamadas de API de ~812–987 por conversa (Mem0/A-MEM) para 18,4 mantendo/ganhando acurácia. RecMem: consolidar só quando há recorrência corta até 87% dos tokens de construção. ConsistencyGate: gate por logprob custa 23–28 ms/fato e reduz contaminação de 50% para 34%, com −1,5% de F1 em dados limpos.
3. **Deletar é o anti-padrão mais documentado; superseder é a mitigação convergente.** Memory-R1 mostra o DELETE+ADD fragmentando memória; Zep, mem0 Dream e TOKI trocam DELETE por invalidação temporal/linha de auditoria; LangMem desliga `enable_deletes` por default.

### Tabela: abordagem × gatilho × gate × custo × evidência

| Abordagem | Gatilho de extração/consolidação | Gate antes do LLM | Custo / latência conhecidos | Evidência (URL) |
|---|---|---|---|---|
| **mem0 paper (2025)** | A cada `add()` (par de mensagens), 2 chamadas LLM: extração + update (ADD/UPDATE/DELETE/NOOP) | Nenhum; recupera s=10 memórias similares e deixa o LLM decidir | Busca p50 0,148 s / p95 0,200 s; total p50 0,708 s / p95 1,44 s (busca+resposta, não `add`); ~7k tokens/conversa | https://arxiv.org/html/2504.19413 |
| **mem0 OSS v2.0.0 (2026-04)** | A cada `add()`, **1 chamada ADD-only** (sem UPDATE/DELETE) | Dedup por hash MD5 pós-extração; LLM sempre chamado se `infer=True` | "roughly half the latency" vs 2 chamadas; +20 pts LoCoMo (71,4→91,6), +26 LongMemEval (67,8→93,4) — claims do vendor | https://github.com/mem0ai/mem0/releases/tag/v2.0.0 · https://docs.mem0.ai/migration/oss-v2-to-v3 · https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py |
| **mem0 Platform + Dream** | `add` assíncrono por default (retorna `PENDING` + `event_id`); Supersede/Merge no add; Synthesis em batch (≥20 memórias; Pro a cada 7 dias, Enterprise diário; até ~24 h de atraso) | Não documentado | Hobby 10.000 adds/mês grátis; Starter $19/50k; Pro $249/500k | https://docs.mem0.ai/migration/platform-v2-to-v3 · https://docs.mem0.ai/platform/features/dream · https://mem0.ai/pricing |
| **Letta sleep-time agents** | Agente de sono roda a cada N passos do agente primário (`sleeptime_agent_frequency`; exemplo oficial 5; um repo da Letta diz "default is typically 10") ou ao compactar contexto | Nenhum (o agente de sono reescreve blocos com `rethink_memory`) | "The higher the frequency setting, the more tokens your agent will use"; paper: ~5× menos compute em test-time, +13%/+18% acurácia, custo/pergunta ÷2,5 com 10 perguntas por contexto | https://arxiv.org/html/2504.13171v1 · https://www.letta.com/blog/sleep-time-compute/ · https://docs.letta.com/guides/agents/architectures/sleeptime/ · https://github.com/letta-ai/skills/blob/main/letta/letta-api-client/sleeptime.md · https://github.com/letta-ai/ezra/blob/main/examples/gotchas/07-sleeptime-frequency/README.md |
| **LangMem `ReflectionExecutor`** | Background com `after_seconds` (default 0 no código; docs sugerem 30–60 min); nova mensagem cancela a tarefa pendente (debounce) | Nenhum explícito; `create_memory_manager` recebe `existing` e devolve insert/update/delete em 1 chamada; `enable_deletes=False` por default | Evita "redundant work", "incomplete context", "unnecessary token consumption" (sem números) | https://langchain-ai.github.io/langmem/guides/delayed_processing/ · https://github.com/langchain-ai/langmem/blob/main/src/langmem/reflection.py · https://langchain-ai.github.io/langmem/reference/memory/ |
| **Zep / Graphiti** | Por episódio (mensagem) com contexto dos últimos n=4 (paper) / `EPISODE_WINDOW_LEN = 3` (código); `add_episode_bulk` pula invalidação | Dedup de entidades por embedding + full-text antes de resolver; contradição → invalida aresta (`t_invalid`), não deleta | Latência de busca mediana 2,58 s vs 28,9 s full-context (LongMemEval); DMR 94,8%/98,2%; LongMemEval 71,2% | https://arxiv.org/html/2501.13956 · https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/graph_data_operations.py · https://help.getzep.com/graphiti/core-concepts/adding-episodes |
| **Claude Code auto memory** | Durante a sessão, a critério do modelo ("Claude doesn't save something every session"); consolidação reativa quando `MEMORY.md` se aproxima de 200 linhas / 25 KB (lembrete) ou passa (erro pedindo reescrita) | Regras de exclusão: não salvar o que é derivável do código ou já está no CLAUDE.md | Sem números públicos | https://code.claude.com/docs/en/memory |
| **ChatGPT memory** | Automático durante a conversa ("let it pick up details itself"); aviso "memories are updated" | Não público | Não público | https://openai.com/index/memory-and-new-controls-for-chatgpt/ · https://help.openai.com/en/articles/8590148-memory-faq |
| **LightMem (2025)** | Extração online por segmento de tópico (buffer 256–1024 tokens); UPDATE offline em "sleep-time" com fila top-k | Pré-filtro token-level LLMLingua-2 (retém percentil r≈0,6) | 18,43 chamadas/conversa vs 811,57 (Mem0) e 986,55 (A-MEM); 28,25k tokens vs 1.152,62k (Mem0); runtime 283,76 s vs 5.132,06 s; acurácia 68,64% vs 53,61% (Mem0) | https://arxiv.org/html/2510.18866 |
| **A-MAC (ICLR 2026)** | Por candidato extraído | Score híbrido: utilidade (LLM), confiança (ROUGE-L), novidade (1 − max cos), recência (meia-vida ~69 h), prior de tipo; θ*=0,55 admite 83% | 2.644 ms/candidato (LLM = 2.580 ms; regras < 65 ms); F1 0,583 vs A-MEM 0,541; 31% mais rápido | https://arxiv.org/abs/2603.04549v1 |
| **ConsistencyGate (2026)** | Por fato, na escrita | Auto-consistência K=5 ou logprob; limiar τ=0,7 | 264–363 ms/fato (K=5) ou 23–28 ms (logprob, 12–14×); contaminação 50%→34,1% (LoCoMo); F1 limpo 0,267 vs 0,271 (−1,5%) | https://arxiv.org/html/2607.22962v1 |
| **RecMem (2026)** | Só quando há recorrência sustentada de interações similares | Camada "subconsciente" só com embeddings | Até −87% tokens de construção de memória em 3 sistemas SOTA, com acurácia ≥ | https://arxiv.org/abs/2605.16045 |
| **Memory-R1 (2025)** | Por turno: extrai fatos, recupera similares, escolhe ADD/UPDATE/DELETE/NOOP (policy RL) | Nenhum (o gate é a própria policy) | Treinado com 152 pares QA; LoCoMo F1 45,02 vs Mem0 30,41; distribuição de ops **não reportada** | https://arxiv.org/html/2508.19828 |
| **RMM (2025)** | Fim de sessão ("prospective reflection"); top-K similares → add ou merge | Nenhum | LongMemEval 70,4% vs 63,6% (RAG); sem custo reportado | https://arxiv.org/html/2503.08026 |
| **A-MEM (2025)** | Por nova nota: 3 chamadas LLM (construção, links, evolução), k=10 vizinhos | Nenhum | ~1.200 tokens/operação (−85–93% vs 16.900 do baseline) | https://arxiv.org/html/2502.12110 |
| **MemoryBank (2023)** | Resumo diário + curva de Ebbinghaus R = e^(−t/S) | Esquecimento por decaimento, não por LLM | Sem números | https://arxiv.org/abs/2305.10250 |

---

## 1. Quando extrair/consolidar

### mem0 — do "por par de mensagens, 2 chamadas" para "por `add()`, 1 chamada, assíncrono"

- **Paper (abr/2025):** a extração recebe "m = 10 previous messages for contextual reference", um resumo da conversa e o par `(m_{t-1}, m_t)`; a fase de update recupera "s = 10 similar memories" e o LLM escolhe ADD/UPDATE/DELETE/NOOP — "Rather than using a separate classifier, we leverage the LLM's reasoning capabilities to directly select the appropriate operation". Latências reportadas são de **busca+resposta**, não de `add`: busca p50 0,148 s / p95 0,200 s; total p50 0,708 s / p95 1,440 s; ~7k tokens/conversa (Mem0^g: 1,091/2,590 s, 14k tokens; full-context 26k tokens: 9,870/17,117 s). https://arxiv.org/html/2504.19413
- **Código atual (main, set/2026):** `_add_to_vector_store` = "V3 PHASED BATCH PIPELINE": Phase 0 `get_last_messages(session_scope, limit=10)`; Phase 1 `vector_store.search(..., top_k=10)`; Phase 2 "LLM extraction (single call)" com `ADDITIVE_EXTRACTION_PROMPT`; Phase 5 dedup `hashlib.md5(text)` contra hashes dos 10 recuperados e do lote; `infer=False` pula o LLM e grava as mensagens cruas. Não existe limiar de similaridade que evite a chamada: com `infer=True` o LLM roda sempre. https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py
- **Release v2.0.0 (2026-04-16):** "One LLM call per `add()`. No separate UPDATE/DELETE pass. [...] Hash-based deduplication prevents exact duplicates; ranking at retrieval time handles the rest." e "roughly half the latency". Platform: "`add` is async by default and returns `{status: "PENDING", event_id}` — poll `/v1/event/{event_id}/`"; parâmetro `async_mode` **removido** (era o toggle). https://github.com/mem0ai/mem0/releases/tag/v2.0.0 · https://docs.mem0.ai/migration/platform-v2-to-v3
- **Changelog:** "Replaced 2-LLM-call pipeline with additive extraction [...] no more UPDATE/DELETE events"; "SQLite-based rolling window (10 messages per session scope) for LLM context". https://github.com/mem0ai/mem0/blob/main/docs/changelog/sdk.mdx
- **Serviço hospedado ("Dream"):** Supersede e Merge "are evaluated when a memory is added [...] There's no separate schedule to wait for"; Synthesis "runs as a scheduled background job per user, not on every add", exige "at least **20** memories", cadência "Pro: Every **7 days**; Enterprise: **Daily**", e "expect new pattern memories to appear within roughly 24 hours of a scheduled run". https://docs.mem0.ai/platform/features/dream
- **Latência de `add` na prática (OSS):** issue #2813 relata ~20 s com modelos locais; mantenedor responde "On my self-hosted setup, adding memory takes around 2 secs" (gpt-4o-mini) e "you should use async with the memory add operator". https://github.com/mem0ai/mem0/issues/2813
- **Custo por add (hospedado):** sem preço unitário; tiers por volume — Hobby 10.000 adds/mês grátis, Starter $19 (50.000), Pro $249 (500.000). https://mem0.ai/pricing

### Letta — sleep-time agents (a cada N passos, em background)

- **Paper:** "reducing the test-time compute needed to achieve the same accuracy by ∼5× on Stateful GSM-Symbolic and Stateful AIME"; "shifting the accuracy up by 13% [...] and 18%"; "reduce the average cost per question by 2.5× when there are 10 queries per context"; ressalva: "In settings where the queries are challenging to predict or unrelated to the context, sleep-time compute will be less effective." https://arxiv.org/html/2504.13171v1
- **Gatilho:** docs atuais — "Choose when it runs: after a set number of completed agent steps or when the context window is compacted"; opção "Agent reviews before applying [...] uses more model tokens". https://docs.letta.com/guides/agents/architectures/sleeptime/
- **Frequência:** blog — "Sleeptime agents can also be configured to run at different frequencies [...] The higher the frequency setting, the more tokens your agent will use". https://www.letta.com/blog/sleep-time-compute/ · Exemplo oficial: `sleeptime_agent_frequency=5  # Run after every 5 conversations` https://github.com/letta-ai/skills/blob/main/letta/letta-api-client/sleeptime.md · Repo `letta-ai/ezra`: "Default is typically 10 [...] Both `interval_seconds` and `min_messages` must be met for sleeptime to trigger". https://github.com/letta-ai/ezra/blob/main/examples/gotchas/07-sleeptime-frequency/README.md — **valor default canônico na página atual dos docs: não encontrado** (a página foi reescrita em torno de "dreaming").

### LangMem — `ReflectionExecutor` (debounce em background)

- Problemas que o delay resolve: "Redundant work when messages arrive in quick succession", "Incomplete context when processing mid-conversation", "Unnecessary token consumption". Mecânica: "Maintains a queue of pending memory tasks; Cancels old tasks when new messages arrive; Only processes after the specified delay". Exemplo usa `delay = 0.5` com nota "In practice would choose longer (30-60 min) depending on app context". https://langchain-ai.github.io/langmem/guides/delayed_processing/
- Código: `after_seconds: int = 0` como default de `submit()`; `LocalReflectionExecutor` cancela a tarefa anterior da mesma thread (`existing.cancel_event.set(); existing.future.cancel()`). https://github.com/langchain-ai/langmem/blob/main/src/langmem/reflection.py
- Guia conceitual: tabela Active (latência "Higher", update "Immediate") vs Background (latência "None", update "Delayed"); hot path "adds perceptible latency to user interactions". https://langchain-ai.github.io/langmem/concepts/conceptual_guide/

### Zep / Graphiti — ingestão por episódio

- "the system processes both the current message content and the last n messages to provide context for named entity recognition. For this paper and in Zep's general implementation, n=4." Contradição → "invalidates the affected edges by setting their t_invalid to the t_valid of the invalidating edge". https://arxiv.org/html/2501.13956
- Código: `EPISODE_WINDOW_LEN = 3`. https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/graph_data_operations.py
- Docs: "Use add_episode_bulk only for populating empty graphs or when edge invalidation is not required. The bulk ingestion pipeline does not perform edge invalidation operations." https://help.getzep.com/graphiti/core-concepts/adding-episodes
- Latência de ingestão por episódio: **não encontrado**.

### Claude Code — auto memory

- "Claude doesn't save something every session. It decides what's worth remembering based on whether the information would be useful in a future conversation." Consolidação é reativa ao tamanho: "After Claude writes to `MEMORY.md`, Claude Code measures the file against the 200-line and 25KB read limits. If the file is near a limit, Claude Code reminds Claude to shorten it [...] If the file is over a limit, the write still succeeds, but Claude Code returns an error telling Claude to rewrite the index". https://code.claude.com/docs/en/memory
- Um cronograma de consolidador em background: **não encontrado** nos docs. (Existe uma skill on-demand `consolidate-memory` — "merge duplicates, fix stale facts, prune the index" — listada no ambiente local desta sessão; sem URL pública.)

### ChatGPT memory

- "As you chat with ChatGPT, you can ask it to remember something specific or let it pick up details itself." e "ChatGPT now lets you know when memories are updated." https://openai.com/index/memory-and-new-controls-for-chatgpt/ · FAQ: "The 'notepad' of your saved memories is stored separately from your chat history"; "The memory summary should capture the most important details, it will not include everything that ChatGPT remembers". https://help.openai.com/en/articles/8590148-memory-faq — Mecanismo/limiares: **não público**.

---

## 2. Gate de novidade barato antes do LLM

| Trabalho | O que o gate olha | Redução de chamadas/custo | Custo em qualidade |
|---|---|---|---|
| **A-MAC** (arXiv 2603.04549, ICLR 2026) | 5 fatores: utilidade (1 chamada LLM, temp 0), confiança `max ROUGE-L(m, s)`, novidade `1 − max cos(φ(m), φ(m'))` (Sentence-BERT), recência `exp(−λτ)` com λ=0,01/h ("half-life of approximately 69 hours"), prior de tipo (regras/POS) | Admite 83% a θ*=0,55; latência 2.644 ms/candidato vs 3.831 ms (A-MEM), −31%. Decomposição: LLM 2.580 ms (97,6%); confiança 18 ms, novidade 32 ms, tipo 14 ms, recência <1 ms | F1 0,583 vs 0,541 (A-MEM); precisão 0,417 vs 0,371; recall 0,972 vs 1,0 | https://arxiv.org/abs/2603.04549v1 |
| **ConsistencyGate** (arXiv 2607.22962, 2026) | Suporte factual do candidato no contexto-fonte (K amostras 0–1 ou logprob), "scores candidates on factual support rather than utility" | Gate K=5: 264–363 ms/fato; logprob: 23–28 ms/fato (12–14×) | Contaminação 50%→34,1% (LoCoMo-Contam), 50%→36,7% (MSC), 50%→1,2% (sintético); dados limpos F1 0,267 vs 0,271 WriteAll (−1,5%) | https://arxiv.org/html/2607.22962v1 |
| **LightMem** (arXiv 2510.18866) | Sensory memory = classificação token-a-token "retain/discard" com LLMLingua-2, limiar no percentil r (0,4–0,8; ótimo ≈0,6); STM segmenta por tópico (atenção ∩ similaridade de turnos adjacentes; buffer 256–1024 tokens); LTM com "soft updates" offline em fila top-k | GPT-4o-mini: 18,43 chamadas vs 986,55 (A-MEM) / 811,57 (Mem0) / 520,62 (LangMem); tokens 28,25k vs 1.605,81k / 1.152,62k / 1.102,16k; runtime 283,76 s vs 5.132,06 s; tokens online ÷117 | Acurácia LongMemEval-S 68,64% vs 62,6% (A-MEM), 53,61% (Mem0), 37,2% (LangMem); LoCoMo +6,10–29,29% | https://arxiv.org/html/2510.18866 |
| **RecMem** (arXiv 2605.16045, 2026) | Recorrência sustentada de interações semanticamente similares (só embeddings até lá) | "reduces the memory construction token cost of three SOTA memory systems by up to 87%" | "while exceeding their accuracy" | https://arxiv.org/abs/2605.16045 |
| **D-Mem** (arXiv 2603.18631, 2026) | System 1 = retrieval vetorial; "Multi-dimensional Quality Gating" decide quando cair no System 2 (deliberação completa) | Não quantificado no abstract | F1 53,5 (LoCoMo, GPT-4o-mini) vs Mem0* 51,2; "96.7% of the Full Deliberation's performance (55.3)" | https://arxiv.org/abs/2603.18631 |
| **Memory-R1** (arXiv 2508.19828) | Nenhum pré-gate; a policy RL escolhe ADD/UPDATE/DELETE/NOOP por turno | Não reporta chamadas/turno | LoCoMo (LLaMA-3.1-8B): F1 45,02 vs 30,41 (Mem0), BLEU-1 37,51 vs 22,22, LLM-judge 62,74 vs 45,68; 152 pares QA de treino | https://arxiv.org/html/2508.19828 |
| **RMM** (arXiv 2503.08026) | Fim de sessão; para cada memória extraída, top-K similares e o LLM decide add vs merge (sem limiar explícito) | Não reporta | LongMemEval 70,4% vs 63,6%; MSC METEOR 33,4 vs 27,5 | https://arxiv.org/html/2503.08026 |
| **A-MEM** (arXiv 2502.12110) | Nenhum; k=10 vizinhos e 3 chamadas LLM por nota (construção, links, evolução) | ~1.200 tokens/operação (−85–93% vs 16.900) | LoCoMo multi-hop F1 27,02 vs 26,65 (MemGPT); temporal 45,85 vs 25,52 | https://arxiv.org/html/2502.12110 |
| **MemOS** (arXiv 2507.03724) | Ciclo Generated→Activated→Merged→Archived; transições "implicitly driven by system heuristics such as recency, contextual salience, or successful merge events" | Números de custo: **não encontrado** no texto acessível | Idem | https://arxiv.org/html/2507.03724 |
| **Mem-α** (arXiv 2509.25911) | Policy RL por chunk com `memory_insert/update/delete`; recompensa de compressão `r₃ = 1 − l_m/l_c` | Memória final média 7,9K tokens de 10,8K de entrada (≈−50% vs long-context/RAG) | Média 0,592 vs 0,461 (long-context) e 0,502 (RAG-Top2); generaliza de 30k para >400k tokens | https://arxiv.org/html/2509.25911 |
| **MemoryBank** (arXiv 2305.10250) | Curva de Ebbinghaus R = e^(−t/S); S inicia em 1 e "+1" a cada recall, t zerado; resumo diário de eventos | Sem números | Sem números | https://arxiv.org/abs/2305.10250 |
| **TraceRetain-CEM** (arXiv 2606.29178, 2026) | Retenção limitada por 7 features (success, age, access frequency, redundancy, specificity, similarity, downstream utility), evicção do menor score | "only differentiates from cache heuristics when streams contain noise" | Com 75% distratores: P@5 16,9→16,6 e 97/100 sucesso; unbounded 20,2→12,4; FIFO-K50 15,8→3,8 | https://arxiv.org/abs/2606.29178 |
| **"When to Forget"** (arXiv 2604.12007, 2026) | Memory Worth = hits⁺/(hits⁺+hits⁻) por memória, alimentado por resultado de episódios | — | ρ=0,89±0,02 com utilidade real após 10k episódios; memória obsoleta cai de MW≈0,97 para 0,17 | https://arxiv.org/html/2604.12007v1 |
| **Control-plane placement** (arXiv 2606.15903, 2026) | Compara primitivas determinísticas vs LLM na escrita vs hook na mutação | Determinístico 64–191 ms/caso; hook de mutação 2,3 s/caso, "$0.17 per 385-case run" | Determinístico: 5% em ofuscação de identificador, 0% cross-lingual; LLM na escrita: 100% canonicalização mas 0% em deleção intencional; hook de mutação: 78–85% deleção intencional, 91,7–93,2% geral | https://arxiv.org/abs/2606.15903 |

Leitura conjunta: o único componente que precisa de LLM é o juízo semântico (utilidade/contradição); novidade por cosseno, hash, recência e tipo custam dezenas de ms e removem a maior parte das chamadas quando o fluxo tem redundância (LightMem, RecMem). O custo em qualidade de um gate barato bem calibrado fica em ~1–3% (ConsistencyGate −1,5%; A-MAC recall 0,972 vs 1,0).

---

## 3. Janela de extração M

| Sistema | Janela passada ao extrator | Fonte |
|---|---|---|
| mem0 paper | par `(m_{t-1}, m_t)` + "m = 10 previous messages" + resumo | https://arxiv.org/html/2504.19413 |
| mem0 OSS v2 | mensagens do `add()` + últimas 10 do escopo (`get_last_messages(limit=10)`) + 10 memórias similares; o prompt reserva "Last k Messages (up to 20)" e "Recently Extracted Memories (up to 20)" | https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py · https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py |
| Zep/Graphiti | episódio atual + n=4 (paper) / 3 (código) anteriores | https://arxiv.org/html/2501.13956 · https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/graph_data_operations.py |
| LightMem | buffer de 256–1024 tokens segmentado por tópico | https://arxiv.org/html/2510.18866 |
| Memory-R1 | um turno por vez | https://arxiv.org/html/2508.19828 |
| RMM | sessão inteira, no fim | https://arxiv.org/html/2503.08026 |
| MemoryBank | dia inteiro (resumo diário) | https://arxiv.org/abs/2305.10250 |
| LangMem | tudo que acumulou até o debounce | https://langchain-ai.github.io/langmem/guides/delayed_processing/ |
| Letta | tudo desde o último disparo (a cada N passos) | https://www.letta.com/blog/sleep-time-compute/ |

**Evidência sobre tamanho da janela** (nenhuma varre M diretamente; a evidência mais próxima é sobre granularidade da unidade de memória):

- **SeCom** (arXiv 2502.05589): "Turn-level memory is too fine-grained, leading to fragmentary and incomplete context. Session-level memory is too coarse-grained, containing too much irrelevant information." LoCoMo GPT4Score: turno 65,58 / sessão 63,16 / segmento 71,57; Long-MT-Bench+: 84,91 / 73,38 / 88,81. Remover o denoising (LLMLingua-2, 75%) custa −9,46 GPT4Score. https://arxiv.org/html/2502.05589
- **LongMemEval** (arXiv 2410.10813): decompor sessões em rounds "significantly enhances reading performance with GPT-4o", mas não com Llama 3.1 8B; indexar com fatos extraídos como chaves: "+9.4% in recall@k and 5.4% in final accuracy". https://arxiv.org/html/2410.10813
- **Verbatim chunks vs artefatos extraídos** (arXiv 2601.00821): "Verbatim chunks win by 15.9 points on LoCoMo (43.9% vs. 28.0%) and 22.0 points on LongMemEval-S (67.4% vs. 45.4%)"; "accuracy tracks how much source text survives in the store"; recomendação: "structured memory should augment verbatim text rather than replace it". https://arxiv.org/abs/2601.00821
- **Atrasar a consolidação demais também custa**: MemoryOS default 23,2 Ans. F1 vs "delayed-flush" 20,6 vs "conservative-merge" 23,5; conclusão: "memory maintenance works best under a balanced update regime". https://arxiv.org/html/2606.24775
- Overlap entre janelas: **não encontrado** como parâmetro explícito em nenhum sistema; mem0 e Zep resolvem com contexto dos últimos k e lista de "recently extracted" como referência de dedup.

Síntese: janela = unidade de tópico (algumas centenas de tokens) com contexto de 3–10 mensagens anteriores é o ponto onde os dados convergem; janelas de turno único fragmentam, sessão inteira dilui.

---

## 4. Extração e consolidação: um prompt ou dois?

**O que cada sistema faz**

- **Dois passos:** mem0 paper (extração → update com s=10 similares) https://arxiv.org/html/2504.19413; Memory-R1 (extrai, recupera, policy escolhe op) https://arxiv.org/html/2508.19828; RMM (extrai tópicos → add/merge) https://arxiv.org/html/2503.08026; Zep (extrair → dedup → invalidar) https://arxiv.org/html/2501.13956; A-MEM (3 chamadas) https://arxiv.org/html/2502.12110; LightMem (extração online, UPDATE offline em fila) https://arxiv.org/html/2510.18866.
- **Um passo ("one-shot")**: LangMem `create_memory_manager` recebe `{"messages", "existing"}` e devolve a lista final de operações (`enable_inserts=True`, `enable_updates=True`, `enable_deletes=False` por default) https://langchain-ai.github.io/langmem/reference/memory/; mem0 OSS v2 — "One LLM call per `add()`", só ADD com `linked_memory_ids` https://github.com/mem0ai/mem0/releases/tag/v2.0.0.

**Evidência comparando custo × qualidade**

- Vendor (mem0): 1 chamada ≈ metade da latência e +20/+26 pontos em LoCoMo/LongMemEval — mas a melhora vem junto com hybrid retrieval, logo não isola o efeito de "um prompt". https://docs.mem0.ai/migration/oss-v2-to-v3
- Custo em qualidade do one-shot ADD-only: issues #4896 ("Both memories are stored as separate ADD events instead of resolving the conflict"), #4956 ("contradictory memories accumulate over time instead of the newer fact superseding the older one"), #5867 ("ADD-only memory extraction can create conflicting memories"). https://github.com/mem0ai/mem0/issues/4896 · https://github.com/mem0ai/mem0/issues/4956 · https://github.com/mem0ai/mem0/issues/5867
- Ablação MemOS "Fast Memorize" vs "Fine Memorize": LoCoMo 40,8 vs 5,0 Ans. F1; LongMemEval 30,2 vs 26,1 (o passo fino ajuda em um benchmark e destrói o outro). https://arxiv.org/html/2606.24775
- Onde colocar a decisão: LLM na escrita ("inscribe-time") canonicaliza 100% mas "cannot help intent-aware deletion (0%)"; hook na mutação recupera 78–85% e sobe o geral a 91,7–93,2% por $0,17/385 casos. https://arxiv.org/abs/2606.15903
- Estudo controlado "1 prompt vs 2 prompts" com mesma retrieval: **não encontrado**.

---

## 5. Proporções práticas

**Fração de turnos com fato memorável**

- LoCoMo (ACL 2024): 10 conversas, 27,2 sessões, 21,6 turnos/sessão, 16.618,1 tokens/conversa, 29,8 tokens/turno, 19,2 tokens/observação, 132,4 tokens/resumo de sessão; as "observations" são geradas **por turno por construção** ("a single turn of the conversation hkj is transformed into an observation okj"), então o dataset não mede a fração de turnos vazios. https://aclanthology.org/2024.acl-long.747.pdf
- Estudo de custo (arXiv 2603.04814): "103,183 extracted records" de 500 conversas de ~101k tokens (≈206 registros por conversa de ~50 sessões ≈ 4 por sessão), compressão ≈35:1; extração com GPT-5-nano "$21.76 in total ($0.0435 per conversation)"; consulta "roughly $0.0013 per query"; "At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns". https://arxiv.org/html/2603.04814v1
- A-MAC: "approximately 1,500 candidate memories" de "30 conversations [...] 15–40 turns" (≈1,2–3,3 candidatos por turno); gate ótimo admite 83%. Combinando precisão 0,417 e recall 0,972 no ponto de operação, a fração de candidatos realmente admissíveis sai em ≈36% (**derivação minha**, o paper não publica o balanço de classes). https://arxiv.org/abs/2603.04549v1
- LongMemEval: 500 perguntas; LongMemEval_S ≈115k tokens / ~50 sessões; cada pergunta depende de "up to six" sessões — ou seja, para qualquer pergunta a imensa maioria das sessões é irrelevante. https://arxiv.org/html/2410.10813
- Prompt do mem0 mostra a política: `{"facts": []}` para "Hi" e para enunciados genéricos. https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py

**Distribuição NOOP/ADD/UPDATE/DELETE:** **não encontrado**. O paper do mem0 não a publica ("No explicit distribution provided"), Memory-R1 não reporta contagem de operações nem tamanho final do banco, e mem0 v2 só emite ADD. Único dado indireto: no mem0 v2 o hash dedup exato barra apenas duplicatas literais; o restante vira ADD (issues acima).

---

## 6. Anti-padrões documentados e mitigação

| Anti-padrão | Evidência | Mitigação adotada (por sistema) |
|---|---|---|
| **Memória crescendo sem limite** | mem0 v2: "Memories accumulate; nothing is overwritten" https://github.com/mem0ai/mem0/blob/main/docs/changelog/sdk.mdx; issue #4956 https://github.com/mem0ai/mem0/issues/4956; TraceRetain: unbounded cai de P@5 20,2→12,4 com ruído https://arxiv.org/abs/2606.29178 | Claude Code: limite 200 linhas/25 KB com lembrete e erro https://code.claude.com/docs/en/memory; mem0 Dream: Merge + Synthesis (≥20 memórias, semanal/diário) https://docs.mem0.ai/platform/features/dream; TraceRetain: evicção por score; MemoryBank: decaimento e^(−t/S) https://arxiv.org/abs/2305.10250; Mem-α: recompensa de compressão (−50%) https://arxiv.org/html/2509.25911; Letta: blocos com `limit` em caracteres https://docs.letta.com/guides/core-concepts/memory/memory-blocks |
| **LLM deletando demais / fragmentando** | Memory-R1: "adopted a dog named Buddy" → "another dog named Scout" faz o baseline emitir DELETE+ADD em vez de UPDATE https://arxiv.org/html/2508.19828; hook LLM na escrita "0% on prefix-collision and compound-fact" https://arxiv.org/abs/2606.15903 | Zep: invalidação temporal em vez de delete https://arxiv.org/html/2501.13956; mem0 Dream: "Superseded memories are not deleted" + `latest_only` https://docs.mem0.ai/platform/features/dream; mem0 v2: removeu DELETE do `add()` https://github.com/mem0ai/mem0/releases/tag/v2.0.0; LangMem: `enable_deletes=False` https://langchain-ai.github.io/langmem/reference/memory/; TOKI: "preserves the losing fact in an audit row" https://arxiv.org/abs/2606.06240; Memory-R1: RL para escolher UPDATE |
| **Duplicatas parafraseadas** | mem0 v2 dedup só por MD5 exato (#4896) https://github.com/mem0ai/mem0/issues/4896; corrida TOCTOU cria duplicatas idênticas sob concorrência (#6515) https://github.com/mem0ai/mem0/issues/6515; prompt legado: "Likes cheese pizza" vs "Loves cheese pizza" → NONE https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py | A-MAC: novidade `1 − max cos` (32 ms) https://arxiv.org/abs/2603.04549v1; mem0 Dream Merge; LangMem debounce (processa o lote de uma vez) https://langchain-ai.github.io/langmem/guides/delayed_processing/; RMM merge por top-K https://arxiv.org/html/2503.08026 |
| **Fato temporário virando permanente** | issue #4956 (empregador/cidade atual) https://github.com/mem0ai/mem0/issues/4956; A-MAC motivação: "hallucinated or obsolete facts" https://arxiv.org/abs/2603.04549v1 | mem0 `expiration_date` ("Expiration hides a memory, it does not delete it [...] Malformed dates fail open") https://docs.mem0.ai/platform/features/memory-expiration; mem0 Memory Decay: fator 1,5× fresco → 0,3× obsoleto, últimos 20 acessos, "Nothing gets deleted" https://mem0.ai/blog/introducing-memory-decay-in-mem0; A-MAC recência (meia-vida ~69 h) + prior de tipo; Zep `t_valid/t_invalid`; "When to Forget" MW por resultado https://arxiv.org/html/2604.12007v1; prompt mem0 v2 exige ancorar datas relativas ("'User went to Paris last week' is useless 6 months later") https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py |
| **Memória contaminada por alucinação do extrator** | ConsistencyGate: 50% de contaminação plantada https://arxiv.org/html/2607.22962v1 | Gate de suporte factual (logprob 23–28 ms) → 34,1%; A-MAC confiança ROUGE-L (18 ms) |
| **Consolidação no hot path** | LangMem: "adds perceptible latency" https://langchain-ai.github.io/langmem/concepts/conceptual_guide/; mem0 #2813 (2–20 s por add) https://github.com/mem0ai/mem0/issues/2813 | mem0 Platform async por default; Letta sleep-time; LangMem `after_seconds`; LightMem update offline |

---

## Recomendação para um SDK Python opt-in (default = hash + cosseno sem LLM; LLM injetado)

**Gatilho default: fora do hot path, com debounce e teto de turnos.**
Nenhum sistema maduro roda LLM sincronamente por turno hoje: mem0 Platform retorna `PENDING` e processa depois, LangMem cancela e re-agenda, Letta dispara a cada N passos, LightMem faz o UPDATE offline. A evidência contrária a "só no fim da sessão" é a ablação MemoryOS (delayed-flush 20,6 vs 23,2 F1). Default sugerido: `trigger="debounce"` com `after_seconds` na casa de dezenas de segundos de inatividade **e** `max_pending_turns=10` (o próprio rolling window do mem0), o que vier primeiro; `session_end` como flush final. Manter `per_add` (síncrono) para quem quer o comportamento mem0-paper, e `manual` para quem orquestra fora.

**Gate default: três faixas por cosseno, LLM só na faixa cinza.**
Custo: novidade por cosseno 32 ms vs 2.580 ms de uma chamada de utilidade (A-MAC); LightMem/RecMem mostram que o pré-filtro barato corta 30×/87% do custo sem perder acurácia. Proposta:
- `sim ≥ dup_threshold` (hash igual ou cosseno alto) → **NOOP sem LLM** (é o caso "Likes/Loves cheese pizza").
- `sim ≤ new_threshold` → **ADD sem LLM**, com `linked_ids` vazios (mem0 v2 faz exatamente isso, e o custo em qualidade é zero para fatos realmente novos).
- faixa cinza (`new_threshold < sim < dup_threshold`) → **chamar o LLM injetado** uma vez, one-shot no estilo LangMem: `messages_novas + últimos k + top-k similares → lista de ops`. É onde vivem UPDATE e contradição, e é onde o ADD-only do mem0 falha (issues #4896/#4956/#5867).
Quando o LLM não está injetado, a faixa cinza degrada para ADD com link (comportamento mem0 v2), que é o "pior caso" já aceito em produção.

**Semântica de DELETE: superseder, nunca apagar.** Convergência de Zep, mem0 Dream, TOKI e LangMem (`enable_deletes=False`). Default: `on_contradiction="supersede"` (marca `invalid_at`, mantém linha), `delete` opt-in.

**Janela:** `context_messages=10` (mem0), `similar_top_k=10` (mem0 paper e código), unidade de extração por segmento de tópico com teto de ~512–1024 tokens (LightMem 256–1024; SeCom mostra turno único fragmentando e sessão diluindo).

**Deixar configurável (com o porquê):**
- `trigger`: `per_add | debounce(after_seconds, max_pending_turns) | every_n_turns | session_end | manual` — Letta e LangMem expõem exatamente esses eixos.
- `dup_threshold`, `new_threshold` — dependem do embedding; mem0 v2 avisa que "absolute numbers shift [...] retune any hard thresholds" ao trocar o scorer.
- `enable_updates`, `enable_deletes` (default False), `on_contradiction` — LangMem.
- `expires_at` / TTL por memória e `recency_decay` só no ranking (mem0 expiration e decay: esconder, não apagar).
- `max_memories` por escopo com evicção por score (TraceRetain) ou lembrete de compactação (Claude Code 200 linhas).
- `llm_budget_per_session` (nº máximo de chamadas) — o custo por chamada de gate é de ~0,3 s (ConsistencyGate K=5) a ~2,6 s (A-MAC utilidade); sem orçamento, uma sessão longa reintroduz o hot path pela porta dos fundos.
- Telemetria de distribuição de ops (ADD/UPDATE/NOOP/supersede por sessão) — ninguém publica esses números; medir localmente é a única forma de calibrar os limiares.


---

## Relatório 3 — DELETE temporal e avaliação da consolidação

# Pesquisa 3 — DELETE temporal e avaliação da consolidação (anchor #5)

**Data:** 2026-09-08 · **Para:** `docs/plans/2026-08-28-v020-5-consolidador-mem0.md` (decisões em aberto "DELETE é remoção ou invalidação temporal?" e "Como medir que a memória ficou melhor e não só menor?")

**Fontes primárias usadas** (tudo em `/private/tmp/claude-501/-Users-arthurgranja-github-anchor/51691894-7963-40bb-9db8-2c9902ed6c81/scratchpad/`):

| Fonte | Onde | Versão |
|---|---|---|
| Graphiti (getzep) | `repos/graphiti/` | `7db96847` 2026-09-08 |
| mem0 OSS (main, v3) | `repos/mem0/` | `dae67f74` 2026-09-04 |
| mem0 OSS v1.0.11 (último com grafo no OSS) | `repos/mem0-v1.0.11/` | clone de 2026-04-06 |
| mem0 `graph_memory.py` v0.1.100 / v0.1.115 / v1.0.11 | `repos/mem0-graph-src/` | raw do GitHub por tag |
| LangMem | `repos/langmem/` | `f8c7ebd6` 2026-09-02 |
| Letta (repo GitHub agora só tem docs/policies, 12 arquivos) | `repos/letta/` | `4511fa0b` 2026-08-23 |
| MemOS | `repos/MemOS/` | `de806942` 2026-09-08 |
| LongMemEval (repo + `longmemeval_oracle.json` baixado do HF) | `repos/LongMemEval/` | `9e0b455f` 2026-05-11 |
| LoCoMo (repo + `locomo10.json`) | `repos/locomo/` | `3eb6f2c5` 2024-08-12 |
| Zep paper arXiv 2501.13956 (PDF → `papers/zep.txt`) | `papers/` | v1 |
| mem0 paper arXiv 2504.19413 (PDF → `papers/mem0.txt`) | `papers/` | v1 |
| anchor | `/Users/arthurgranja/github/anchor/src/anchor/` | branch `dev`, `3c88e66` |

Convenção: `arquivo:linha` é relativo à raiz do repo indicado; linhas de `.txt` dos papers são do `pdftotext -layout`.

---

## Sumário executivo

**Tema A — DELETE é remoção ou invalidação temporal?** Invalidação. Todo sistema que declara raciocínio temporal como objetivo invalida em vez de apagar: Graphiti seta `invalid_at`/`expired_at` na aresta contradita (`graphiti_core/utils/maintenance/edge_operations.py:569-571`) e a busca **não** exclui arestas invalidadas por padrão — devolve-as com a janela de validade para o LLM (`search/search_helpers.py:56-57`); o mem0^g diz no paper que marca "as invalid rather than physically removing" (`papers/mem0.txt:259-260`), o código OSS fazia `DELETE r` até v0.1.115 (`repos/mem0-graph-src/graph_memory-v0.1.115.py:359`) e foi corrigido para `SET r.valid = false, r.invalidated_at = datetime()` depois da issue #4187 (`repos/mem0-v1.0.11/mem0/memory/graph_memory.py:420-428`); MemOS tem `status: "archived"` + `history` de versões (`src/memos/memories/textual/item.py:109-111,126-129`) e a busca filtra `status="activated"` (`.../retrieve/recall.py:129`). Os que apagam de fato — mem0 vector store (`vector_store.delete` + tabela `history` em SQLite, `repos/mem0/mem0/memory/main.py:2111-2122`), LangMem (`RemoveDoc` → `store.adelete`, `src/langmem/knowledge/extraction.py:1134`) e Letta (API `DELETE /v1/archives/{id}/passages/{id}` "permanently removes", docs.letta.com) — ou guardam um log paralelo (mem0) ou delegam a semântica ao usuário (LangMem docs: "you can control what removal means, be that a soft or hard delete"). Papers de 2026 convergem para bi-temporal com invalidação (Engram 2606.09900 "invalidating, never deleting"; MemStrata 2606.26511; TOKI 2606.06240 "audit row"), e as métricas novas (FAMA, stale-fact-error rate) **exigem** saber que uma memória foi invalidada para medir se o agente parou de usá-la. Dado curioso e importante: o **mem0 OSS v3 (2026) abandonou UPDATE/DELETE inteiramente** — "Single-pass ADD-only extraction — one LLM call, no UPDATE/DELETE. Memories accumulate; nothing is overwritten" (`repos/mem0/README.md:57`) e resolve conflito no retrieval. Para o anchor: soft-delete = `expires_at=now` + `metadata["invalidated_by"]` **cobre**, porque os cinco backends já filtram `expires_at` em `search`, `list_all` e `search_filtered` (tabela em A.4). Três ressalvas com evidência: o GC apaga expirados (`memory/gc.py:162-166`), logo o histórico some no primeiro `collect()`; o passo de pipeline ignora `DELETE` hoje (`pipeline/memory_steps.py:44-46`); e `GraphIndexingEntryStore.add` com o mesmo `content` não toca o grafo (`ingestion/graph_extractors.py:399-405`), então a evidência da entrada invalidada continua viva no grafo.

**Tema B — como medir que a memória ficou melhor e não só menor?** Nenhum benchmark público mede consolidação diretamente; todos medem QA depois de ingerir a conversa inteira. LoCoMo (1.986 QA em 10 conversas, 1.540 sem a categoria adversarial) é o mais citado e o menos confiável: o juiz gpt-4o-mini aceita 62,81% de respostas propositalmente erradas (Penfield Labs, 2026-04-09) e a disputa mem0×Zep de 2025 terminou com três números para o mesmo sistema (65,99% no paper do mem0; 84% → 75,14% no blog da Zep; 58,44% no re-run do mem0, `github.com/getzep/zep-papers/issues/5`). LongMemEval é o único com subconjunto reaproveitável: **78 questões `knowledge-update`** (6 de abstenção) com o par fato-antigo/fato-novo marcado por `has_answer` nos turnos (contado em `longmemeval_oracle.json`). O que mede consolidação de verdade está em 2026: MemoryAgentBench "FactConsolidation" (Mem0 18% single-hop, Zep 7%, BM25 48%), MemConflict (métricas UOCS/CRS + SEH@K/SRS, 6 sistemas), Memora/FAMA ("penalizes reliance on obsolete or invalidated memory"), ForgetEval (1.000 casos por primitiva supersede/release/purge + 385 adversariais, score determinístico por substring; mem0 88,8%/68,3%), PrecisionMemBench (89 casos, precisão de retrieval). **Dataset público rotulado com ADD/UPDATE/DELETE/NOOP: não encontrado.** O paper do mem0 reporta só F1/BLEU-1/LLM-judge por categoria + tokens e latência p50/p95 (`papers/mem0.txt:414-432, 535-566`) — nada sobre tamanho do store, contradições ou update accuracy. Para o anchor: o harness atual é de retrieval (`evaluation/golden.py` com `GoldenCase`/`evaluate_retriever`/`assert_metric_floor`); falta um `evaluation/consolidation.py` pequeno com `ConsolidationCase` (turnos + estado inicial + operações esperadas + estado final esperado) e um relatório com 3 métricas: acurácia de operação por classe, taxa de fatos obsoletos vivos no estado final, e stale-fact rate no top-k. Golden set de 30–60 casos desenhado na seção B.3.

---

## Tema A — DELETE é remoção ou invalidação temporal?

### A.1 Graphiti / Zep

**Modelo bi-temporal no código.** `EntityEdge` carrega os quatro timestamps (`repos/graphiti/graphiti_core/edges.py:271-278`):

```python
expired_at: datetime | None = Field(default=None, description='datetime of when the node was invalidated')
valid_at:   datetime | None = Field(default=None, description='datetime of when the fact became true')
invalid_at: datetime | None = Field(default=None, description='datetime of when the fact stopped being true')
```

(`created_at` vem da classe base `Edge`.) O paper descreve o mesmo par de linhas do tempo: "Zep implements a bi-temporal model, where timeline T represents the chronological ordering of events, and timeline T′ represents the transactional order of Zep's data ingestion. While the T′ timeline serves the traditional purpose of database auditing, the T timeline provides an additional dimension for modeling the dynamic nature of conversational data" (`papers/zep.txt:109-112`); e "the system tracks four timestamps: t′created and t′expired ∈ T′ monitor when facts are created or invalidated in the system, while tvalid and tinvalid ∈ T track the temporal range during which facts held true" (`papers/zep.txt:162-165`).

**Contradição invalida, não apaga.** Paper, §2.2.3: "The introduction of new edges can invalidate existing edges in the database. The system employs an LLM to compare new edges against semantically related existing edges to identify potential contradictions. When the system identifies temporally overlapping contradictions, it invalidates the affected edges by setting their tinvalid to the tvalid of the invalidating edge. Following the transactional timeline T′, Graphiti consistently prioritizes new information when determining edge invalidation" (`papers/zep.txt:166-170`). "...maintaining both current relationship states and historical records of relationship evolution over time" (`:171-172`). Código correspondente (`graphiti_core/utils/maintenance/edge_operations.py:538-573`), o núcleo:

```python
# New edge invalidates edge
elif (edge_valid_at_utc is not None and resolved_edge_valid_at_utc is not None
      and edge_valid_at_utc < resolved_edge_valid_at_utc):
    edge.invalid_at = resolved_edge.valid_at
    edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
    invalidated_edges.append(edge)
```

(linhas 563-571). Os candidatos vêm de uma busca semântica separada (`edge_invalidation_candidates`, `:407-430`) e o LLM decide contradição em `resolve_extracted_edge` (`:623-712`, o contexto `edge_invalidation_candidates` em `:704-712`). Não existe `DELETE` de aresta no fluxo de ingestão. README: "Facts have validity windows. When information changes, old facts are invalidated — not deleted. Query what's true now, or what was true at any point in time" (`repos/graphiti/README.md:120-121`) e "Explicit bi-temporal tracking with automatic fact invalidation … Automatic fact invalidation with temporal history preserved" (`README.md:145-146`).

**Como a busca trata arestas inválidas — ponto que surpreende.** A busca **não filtra por padrão**. `SearchFilters` tem `valid_at`, `invalid_at`, `created_at`, `expired_at` todos `default=None` (`graphiti_core/search/search_filters.py:62-65`); só viram cláusula Cypher se o chamador passar (`:149-178` para `valid_at`, `:180-209` para `invalid_at`, `:242-271` para `expired_at`). `grep -rn "expired_at IS NULL\|invalid_at IS NULL" graphiti_core/search graphiti_core/driver graphiti_core/graph_queries.py` → nenhuma ocorrência. O que o Graphiti faz é **devolver a janela de validade ao LLM** e deixar o raciocínio temporal para ele: `search_results_to_context_string` serializa `{'fact', 'valid_at', 'invalid_at' or 'Present'}` e instrui "Facts are considered valid between their valid_at and invalid_at dates. Facts with an invalid_at date of "Present" are considered valid" (`graphiti_core/search/search_helpers.py:29-34, 56-57`). Ou seja: o modelo é "invalidar + expor a janela", não "invalidar + esconder". O paper não descreve filtragem na busca (grep por `invalid` em `zep.txt` só retorna §2.2.3).

**Resultados reportados.** DMR: "Zep achieved 94.8% accuracy with gpt-4o-turbo … MemGPT 93.4%" (`papers/zep.txt:302-304, 325`). LongMemEval_S: Full-context 55,4% / Zep 63,8% (gpt-4o-mini); 60,2% / 71,2% (gpt-4o), 1,6k tokens de contexto vs 115k (`zep.txt:375-382`). Por tipo (Table 3, `zep.txt:390-405`): **knowledge-update 76,9% → 74,4% (gpt-4o-mini, queda) e 78,2% → 83,3% (gpt-4o)**; temporal-reasoning 36,5% → 54,1% e 45,1% → 62,4%. O próprio paper admite: "additional development may be needed to improve less capable models' understanding of Zep's temporal data" (`zep.txt:369-374`) — consistente com o design "expor a janela e confiar no LLM".

### A.2 mem0 e mem0^g

**Vector store: hard delete + tabela `history`.** Confirmado no main atual (`repos/mem0/mem0/memory/main.py:2100-2122`):

```python
def _delete_memory(self, memory_id, existing_memory=None):
    ...
    self.vector_store.delete(vector_id=memory_id)
    self.db.add_history(memory_id, prev_value, None, "DELETE", created_at=created_at,
                        updated_at=updated_at, actor_id=..., role=..., is_deleted=1)
```

`_update_memory` sobrescreve o payload no vector store e grava `add_history(memory_id, prev_value, data, "UPDATE", ...)` (`main.py:2038-2091`). O `history` é SQLite (`self.db = SQLiteManager(self.config.history_db_path)`, `main.py:500`) com schema `history(id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role)` (`repos/mem0/mem0/memory/storage.py:108-119`). Portanto: o dado sai do índice de busca, mas o par (old, new, event) fica num log consultável por `Memory.history(memory_id)` (`main.py:1946-1959`). É um "delete com trilha de auditoria", não uma invalidação que a busca conheça.

**Decisão ADD/UPDATE/DELETE/NONE.** Prompt `DEFAULT_UPDATE_MEMORY_PROMPT` (`repos/mem0/mem0/configs/prompts.py:176-185`): "ADD: Add it to the memory as a new element / UPDATE: Update an existing memory element / DELETE: Delete an existing memory element / NONE: Make no change (if the fact is already present or irrelevant)". Paper: "ADD for creation of new memories when no semantically equivalent memory exists; UPDATE for augmentation of existing memories with complementary information; DELETE for removal of memories contradicted by new information; and NOOP when the candidate fact requires no modification" (`papers/mem0.txt:186-189`); Algoritmo 1: "else if Contradicts(f, M) then return DELETE" (`mem0.txt:1075-1076`), "M ← M \ {mi} ▷ Remove contradicted information" (`:1065`).

**mem0^g: paper vs código.** Paper: "An LLM-based update resolver determines if certain relationships should be obsolete, marking them as invalid rather than physically removing them to enable temporal reasoning" (`papers/mem0.txt:259-260`). Código OSS na época do paper (v0.1.100 e v0.1.115) fazia hard delete: `MATCH (n ...)-[r:{relationship}]->(m ...) ... DELETE r` (`repos/mem0-graph-src/graph_memory-v0.1.100.py:287-300`; `graph_memory-v0.1.115.py:330-359`). A discrepância foi reportada na issue mem0ai/mem0#4187 (2026-03-02, "Graph memory uses hard DELETE instead of soft-delete — breaks temporal reasoning described in paper", https://github.com/mem0ai/mem0/issues/4187) e corrigida em v1.0.x (`repos/mem0-v1.0.11/mem0/memory/graph_memory.py:420-428`):

```python
# Soft-delete: mark relationship as invalid instead of removing it,
# enabling temporal reasoning over historical graph state.
# See: https://github.com/mem0ai/mem0/issues/4187
cypher = f"""
MATCH (n {self.node_label} {{{source_props_str}}})-[r:{relationship}]->(m {self.node_label} {{{dest_props_str}}})
WHERE r.valid IS NULL OR r.valid = true
SET r.valid = false, r.invalidated_at = datetime()
```

e o `MERGE` de relação re-ativa (`ON MATCH SET r.valid = true, ... r.invalidated_at = null`, `:499-503`). O prompt de decisão do grafo é conservador por design: "Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated … DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes" (`repos/mem0-graph-src/utils-v1.0.11.py:69-79`).

**mem0 OSS v3 (2026) abandonou UPDATE/DELETE.** No clone atual não existe `graph_memory.py` (`ls repos/mem0/mem0/memory` → `base.py main.py storage.py utils.py …`), e `_add_to_vector_store` usa `ADDITIVE_EXTRACTION_PROMPT` incondicionalmente (`repos/mem0/mem0/memory/main.py:942`; prompt em `configs/prompts.py:464-520`: "Your sole operation is ADD … When a new memory is related to an Existing Memory — same topic, overlapping entities, updated/shifted preference … include the Existing Memory's ID in the new memory's "linked_memory_ids" array"). README: "**Single-pass ADD-only extraction** -- one LLM call, no UPDATE/DELETE. Memories accumulate; nothing is overwritten" e "**Temporal Reasoning** -- time-aware retrieval that ranks the right dated instance for queries about current state, past events, and upcoming plans" (`repos/mem0/README.md:57-61`). Guia de migração (https://docs.mem0.ai/migration/oss-v2-to-v3): "The ADD-only model means memories accumulate over time. When information changes, the new fact is stored alongside the old one. Retrieval handles ranking: the most relevant, current information surfaces first"; ganho alegado "+20 point improvement on LoCoMo (71.4 → 91.6) and +26 point improvement on LongMemEval (67.8 → 93.4)". `_update_memory`/`_delete_memory` continuam existindo só para a API pública `update()`/`delete()` (`main.py:2038, 2100`). Leitura: o mem0 concluiu que a decisão DELETE feita pelo LLM no write path custava mais do que rendia e moveu a resolução de conflito para o read path com timestamps — que é exatamente o que "invalidação + janela temporal" habilita.

### A.3 Outros sistemas e papers

| Sistema | Semântica do "delete" | Evidência |
|---|---|---|
| **Letta** (archival) | Hard delete; sem invalidação. | Repo GitHub `letta-ai/letta` hoje só tem docs (12 arquivos, `git ls-files`). API: `DELETE /v1/archives/{archive_id}/passages/{passage_id}` "permanently removes the passage from both the database and vector storage" (https://docs.letta.com/api-reference/agents/passages/delete). Ferramentas do agente são `archival_memory_insert`/`archival_memory_search` (sem delete no núcleo). |
| **LangMem** | `RemoveDoc` → delete real no `BaseStore`; a semântica é do usuário na API funcional. | `store.adelete(ns, key)` para cada `removed_ids` (`repos/langmem/src/langmem/knowledge/extraction.py:1127-1135`; construção de `removed_ids` em `:951-963`). Docs: "the manager will return `RemoveDoc` objects to indicate that the memory should be removed, and a new memory will be created in its place … you can control what "removal" means, be that a soft or hard delete, or simply a down-weighting of the memory" (`repos/langmem/docs/docs/guides/extract_semantic_memories.md:79`). |
| **MemOS** | Ciclo de vida com estados; atualização **arquiva** a versão antiga e guarda `history`. | `status: Literal["activated", "resolving", "archived", "deleted"]` (`repos/MemOS/src/memos/memories/textual/item.py:109-111`); `ArchivedTextualMemory`: "When an existing memory item needs to be updated due to conflict/duplicate with new memory contents, its previous contents will be preserved, in 2 places" (`item.py:49-59`); `history: list[ArchivedTextualMemory]` (`:126-129`); feedback UPDATE seta `{"status": "archived"}` no antigo (`src/memos/mem_feedback/feedback.py:315-319`); a busca filtra `status="activated"` (`src/memos/memories/textual/tree_text_memory/retrieve/recall.py:129, 331, 342, 376`). Paper 2507.03724 (HTML): estados "Generated, Activated, Merged, and Archived" + "Frozen" ("updates are disabled and full modification histories are retained for auditing"); MemCube com "Lifespan Policy (TTL or decay rules)" e "Version Chain". |
| **Engram** (arXiv 2606.09900, 2026-06-05) | Bi-temporal, "invalidating, never deleting, so every fact keeps provenance and a supersession chain"; leitura "as-of". LongMemEval_S 83,6% vs 73,2% full-context com 9,6k vs 79k tokens. | https://arxiv.org/abs/2606.09900 |
| **MemStrata** (arXiv 2606.26511, 2026-06-25) | Regra determinística de supersessão (subject, relation, object) que "retires the stale value in a bi-temporal ledger — with no similarity threshold and no LLM call". Argumento-chave: "cosine similarity distinguishes a contradicted fact from a duplicated one with AUROC 0.59 (near chance)". Métrica: **stale-fact-error rate** — RAG serve valor obsoleto 15-40% das vezes, MemStrata ~0%. | https://arxiv.org/abs/2606.26511 |
| **TOKI** (arXiv 2606.06240, 2026-06-04) | Resolução de contradição tratada como controle de concorrência no write; quatro heurísticas (last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy) como operadores bi-temporais "with an isolation precondition and a provenance annotation that preserves the losing fact in an audit row". | https://arxiv.org/abs/2606.06240 |
| **TSM** (arXiv 2601.07468) | Memórias "durativas" numa linha do tempo semântica; recuperação "time-valid, duration-consistent". +12,2% em LongMemEval/LoCoMo. | https://arxiv.org/abs/2601.07468 |
| **Lethe / ForgetEval** (arXiv 2606.15903) | Três primitivas explícitas: **supersede** (substitui), **release** (fica "present but mute"), **purge** (apaga com recibo criptográfico). "Production failures are predominantly forgetting failures rather than recall failures, yet existing benchmarks measure only recall." | https://arxiv.org/abs/2606.15903 · https://github.com/deeplethe/lethe |
| **Survey grafo-memória** (arXiv 2602.05665) | Define bi-temporal como distinção valid time × transaction time e cita o Graphiti como implementação de referência. | https://arxiv.org/pdf/2602.05665 |

**Existe consenso?** Sim, com uma nuance. (1) Quem tem raciocínio temporal como requisito (Graphiti, mem0^g pós-#4187, MemOS, Engram, MemStrata, TOKI, TSM) invalida e mantém histórico; nenhum deles apaga no caminho de consolidação. (2) Quem apaga (mem0 vector, LangMem, Letta) ou mantém log fora do índice (mem0 `history`), ou terceiriza a decisão (LangMem docs), ou nem tenta consolidar (Letta deixa o agente editar). (3) A tendência 2026 é reduzir o papel do LLM no write path: ADD-only + ranking temporal no read (mem0 v3), supersessão determinística por chave (MemStrata, Lethe), ou "assembly" pós-retrieval que separa extração de evidência de execução da política (arXiv 2606.01435, "the best reported retrieval/memory result is 54% single-hop and all 22 reported systems score at most 7% multi-hop" no FactConsolidation).

**Argumentos a favor da invalidação** (com fonte): histórico/"o que o usuário achava antes" e queries as-of (Graphiti README:120-121; Zep §2.2.3 "historical records of relationship evolution"); auditoria (Zep: T′ "serves the traditional purpose of database auditing"; TOKI "audit row"; MemOS "Frozen … retained for auditing"); reversibilidade quando o LLM erra a decisão — a issue #4187 relata que "during batch imports, conflict detection incorrectly deletes current facts"; e mensurabilidade — FAMA (Memora) e stale-fact-error rate (MemStrata) só existem se o sistema sabe quais memórias estão inválidas.

**Argumentos contra**: busca mais complexa — mas só se você quiser expor as inválidas; filtrar por `expires_at`/`invalidated_at` é uma cláusula `WHERE` (Graphiti nem faz isso; MemOS faz com `status`); custo de storage — crescimento monotônico, precisa de política de retenção (MemOS tem TTL, Lethe tem `purge`); contexto poluído — real se as inválidas vazarem para o prompt sem a janela; o Zep paper mostra a queda em `knowledge-update` com gpt-4o-mini (76,9% → 74,4%) quando o LLM recebe a janela e não a entende (`zep.txt:396-398`).

### A.4 Para o anchor — o que já existe e o que o soft-delete implica

**Modelo.** `MemoryEntry.expires_at: datetime | None` e `is_expired` (`src/anchor/models/memory.py:60, 73-78`). `MemoryOperation` já tem `DELETE` (`src/anchor/protocols/memory.py:18-24`). Não há hoje `invalidated_at`/`valid_from` em `MemoryEntry`; o grafo já é bi-temporal completo: `GraphEdge.valid_from/valid_to/created_at/invalidated_at` (`src/anchor/models/graph.py:119-122`), `is_live(as_of)` (`:140-148`), docstring "An obsolete edge is invalidated, never deleted: `invalidated_at` is transaction time, `valid_from`/`valid_to` is world time" (`graph.py:15-16`); `invalidate_edge` em memória (`src/anchor/storage/memory_store.py:315-327`), SQLite (`storage/sqlite/_graph_store.py:215-216`) e Postgres (`storage/postgres/_graph_store.py:227-228`); busca de arestas sempre com `invalidated_at IS NULL` (`sqlite/_graph_store.py:392, 425`; `postgres/_graph_store.py:421, 453`).

**`search()` / `list_all()` filtram `expires_at`? Sim, nos cinco backends:**

| Backend | `search` | `list_all` | `search_filtered` | `list_all_unfiltered` (GC) |
|---|---|---|---|---|
| In-memory `InMemoryEntryStore` e `JsonFileMemoryStore` (ambos via `BaseEntryStoreMixin`) | `storage/_base.py:62-67` → `_filters.py:43-58` (`if entry.is_expired: continue`, `:53`) | `_base.py:70-72` → `_filters.py:61-63` | `_base.py:84-…` (docstring `:97`) → `_filters.py:108-112` (`:111`) | `_base.py:80-82` |
| SQLite (`SqliteEntryStore` sync e async) | `sqlite/_entry_store.py:51-59` com `_NON_EXPIRED_CLAUSE = "(expires_at IS NULL OR expires_at > ?)"` (`:25`); async `:201-212` | `:61-68`; async `:214-221` | `:105-125` (`clauses = [_NON_EXPIRED_CLAUSE]`, `:118`) | `:87-90`; async `:237` |
| Postgres | `postgres/_entry_store.py:79-92` (`WHERE (expires_at IS NULL OR expires_at > $1)`, `:84`) | `:94-102` (`:99`) | `:130-146` (`:143`) | `:116-119` |
| Redis (`RedisEntryStore` e `AsyncRedisEntryStore`) | `redis/_entry_store.py:83-95` (`if not e.is_expired`, `:92`); async `:297-309` | `:97-101`; async `:311-315` | `:152-…` via `_matches` (`if e.is_expired: return False`, `:199`) | `:103-106`; async `:317` |

Índices: `idx_memory_entries_expires_at` (`sqlite/_schema.py:119`; `postgres/_schema.py:160`). Cobertura de teste: `tests/test_storage/test_entry_store_shared.py:100-118` (`test_search_filters_out_expired_entries`), `:122-129` (`list_all`), `:171-176` (`list_all_unfiltered`), `:391-…` (`search_filtered`) — **parametrizado só para InMemory e JsonFile** (`test_entry_store_shared.py:1-3`). `grep -rn expire tests/test_storage/test_sqlite tests/test_storage/test_postgres tests/test_storage/test_redis` → nada: os três backends de rede/SQL não têm teste de expiração para o entry store (o diretório `test_redis/` só tem `test_context_store.py`; `test_postgres/` só grafo e pgvector).

**Quem consome `list_all`/`search`** (todos herdam o filtro): `MemoryManager.get_relevant_facts` → `persistent_store.search` (`src/anchor/memory/manager.py:197-204`), `get_all_facts` → `list_all` (`:206-213`), `add_fact` dedup por hash sobre `list_all()` (`:184-187`), `update_fact` sobre `list_all()` (`:236-240`); `ScoredMemoryRetriever.retrieve` monta candidatos de `self._store.list_all()` (`src/anchor/retrieval/memory_retriever.py:116`) — o `VectorStore` não sabe de `expires_at` (`storage/sqlite/_vector_store.py`, `_vec_store.py`, `postgres/_vector_store.py` não mencionam `expires_at`/`MemoryEntry`), mas isso é inócuo porque os ids do vetor só viram `relevance_map` e a lista final vem de `list_all` (`memory_retriever.py:107-116`); `_store_with_consolidation` usa `store.list_all()` como `existing` (`src/anchor/pipeline/memory_steps.py:42`).

**Consequências do soft-delete = `expires_at=now` + `metadata["invalidated_by"]`:**

1. **Busca**: coberto sem mudança — a entrada some de `search`/`list_all`/`search_filtered` em todos os backends no mesmo instante (o `>` em `expires_at > now` faz `expires_at=now` expirar imediatamente; `is_expired` usa `>=`, `models/memory.py:78`). Dedup por `content_hash` em `add_fact` (`manager.py:184-187`) e o `existing` do consolidador (`memory_steps.py:42`) também deixam de ver a entrada — ou seja, um fato re-afirmado depois de invalidado volta como ADD novo, o que é o comportamento certo (Graphiti re-cria; mem0^g re-ativa com `ON MATCH SET r.valid = true`).
2. **GC apaga o histórico**: `MemoryGarbageCollector.collect_expired` faz `self._store.delete(entry.id)` para todo `is_expired` (`src/anchor/memory/gc.py:161-166`), e `collect` sempre roda a fase de expiração (`:131-132`). Sem política, o "histórico" dura até o próximo `collect()`. Opções lazy, em ordem: (a) o GC pula entradas com `metadata["invalidated_by"]` a menos que `expires_at < now - retention` (um `timedelta` no construtor); (b) `on_expiry_prune` (`memory/callbacks.py:97`) já existe para quem quiser exportar antes de apagar. Não precisa de tabela `history` como no mem0: a entrada invalidada **é** a linha de histórico.
3. **Pipeline ignora DELETE**: `_store_with_consolidation` só executa `ADD`/`UPDATE` (`pipeline/memory_steps.py:44-46`: `if action in (MemoryOperation.ADD, MemoryOperation.UPDATE) and entry is not None: store.add(entry)`). Um consolidador que devolva `DELETE` hoje não tem efeito — precisa de um ramo que faça `store.add(existing.model_copy(update={"expires_at": now, "metadata": {..., "invalidated_by": new.id}}))`. Como o protocolo devolve `(op, entry | None)` (`protocols/memory.py:118-133`), para `DELETE` o `entry` devolvido deve ser a entrada **antiga** já marcada (o `SimilarityConsolidator` devolve `None` só para `NONE`, `memory/consolidator.py:123`).
4. **Grafo**: `GraphIndexingEntryStore.add` retorna cedo quando `current.content == entry.content` (`src/anchor/ingestion/graph_extractors.py:399-405`), então invalidar por `expires_at` sem mudar o texto deixa as arestas evidenciadas por essa entrada vivas. Já existe o contrato: "Removing an item is `graph.unlink_item(item_id)` — edges left without evidence are invalidated" (`graph_extractors.py:445-447`), e `clear()` já reconhece a assimetria: "Expired entries are hidden from list_all but still evidence the graph" (`:412-414`). O ramo de `DELETE` deve chamar `unlink_item(entry.id)` (como `delete` faz em `:407-410`) — assim a aresta cuja única evidência era a entrada invalidada recebe `invalidated_at` e o grafo e o store contam a mesma história, com `valid_to` opcional se o extrator souber a data do mundo.
5. **`UPDATE` também deveria preservar a versão antiga?** No mem0 o `UPDATE` sobrescreve e loga; no MemOS o antigo vira `archived` + `history`. No anchor `SimilarityConsolidator._merge_entries` sobrescreve o `existing` in place (`memory/consolidator.py:64-94`). Com o LLM decidindo, "moro em SP" → "me mudei pro Rio" é semanticamente `DELETE(antiga)+ADD(nova)` com link; tratar `UPDATE` como "invalidar antiga + adicionar nova com `metadata["supersedes"]=old.id`" dá o histórico de graça e evita o merge textual que hoje escolhe "o conteúdo mais longo" (`consolidator.py:74-78`). Reservar o `UPDATE` in-place para enriquecimento não-contraditório (o sentido do paper do mem0: "augmentation … with complementary information").

**Não é necessário** um campo novo `invalidated_at`: `expires_at` já é o transaction-time de fim para o store, `updated_at` existe, e a distinção world-time (`valid_from/valid_to`) fica no grafo, onde já está implementada. Se um dia a busca precisar de "as-of" sobre entradas (não sobre arestas), aí sim adicionar `valid_from/valid_to` a `MemoryEntry` — YAGNI hoje; nenhum caso do golden set abaixo precisa disso.

### Recomendação para o anchor (Tema A)

**Decisão sugerida:** `DELETE` = invalidação, implementada como `expires_at = now` + `metadata["invalidated_by"] = <id da entrada nova>` na entrada antiga; nada de `delete()` no caminho de consolidação. Complementos mínimos: (i) ramo `DELETE` em `_store_with_consolidation` que grava a entrada marcada e chama `graph.unlink_item` quando o store é `GraphIndexingEntryStore`; (ii) parâmetro `retention: timedelta | None` no `MemoryGarbageCollector` — expirados com `invalidated_by` só são apagados depois de `retention`; (iii) `UPDATE` contraditório modelado como invalidar-antiga + adicionar-nova com `metadata["supersedes"]`. **Evidência:** todos os stores já filtram `expires_at` (tabela A.4), o grafo já é bi-temporal (`models/graph.py:119-122`), e a literatura de 2026 tanto nos sistemas (Graphiti, mem0^g pós-#4187, MemOS, Engram, MemStrata, TOKI) quanto nas métricas (FAMA, stale-fact rate) assume invalidação com histórico. Um teste que falha se a regra quebrar: adicionar `expires_at` ao contrato compartilhado para SQLite/Postgres/Redis (hoje só InMemory/JsonFile cobrem, `test_entry_store_shared.py:1-3`).

---

## Tema B — Como medir que a memória ficou MELHOR e não só menor?

### B.1 Benchmarks e a disputa mem0 × Zep

**LoCoMo** (arXiv 2402.17753, Maharana et al., 2024-02-27; https://arxiv.org/abs/2402.17753). Repo: `repos/locomo/data/locomo10.json`. Contado localmente: 10 conversas, média de 27,2 sessões e 588 turnos por conversa, 1.986 QA; por `category`: {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}. O código oficial trata `category 1` como multi-hop (F1 parcial por sub-resposta, `task_eval/evaluation.py:212-213`), `2, 3, 4` como "single-hop, temporal, open-domain" (`:208-209`) e `5` como adversarial por seleção de opção (`:216-217`). O paper do mem0 descreve "10 extended conversations, each containing approximately 600 dialogues and 26000 tokens on average … 200 questions on an average" e exclui a categoria adversarial (`papers/mem0.txt:286-294`) → 1.540 questões avaliadas. **Limitações**: (a) tamanho — 10 conversas, uma seed; (b) o juiz — "LoCoMo's gpt-4o-mini judge accepts 62.81% of intentionally wrong answers" (Penfield Labs, 2026-04-09, https://penfieldlabs.substack.com/p/proposal-a-new-benchmark-for-long); (c) o full-context vence — no próprio paper do mem0, Full-context J = 72,90% vs Mem0 66,88% / Mem0^g 68,44% (`papers/mem0.txt:559-566`); (d) Letta mostrou 74,0% com "a filesystem and grep, no memory system at all" (gpt-4o-mini, 2025-08-12, https://www.letta.com/blog/benchmarking-ai-agent-memory/); (e) categoria 5 sem gabarito utilizável (mem0 rebuttal). Não há nada em LoCoMo sobre atualização de fato — mede recall.

**LongMemEval** (ICLR 2025, arXiv 2410.10813; repo `repos/LongMemEval`). Cinco habilidades: "Information Extraction, Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention" (`README.md:21-26`). 500 questões; `question_type` ∈ {single-session-user, single-session-assistant, single-session-preference, temporal-reasoning, knowledge-update, multi-session}, sufixo `_abs` = abstenção (`README.md:79-81`). Contado em `longmemeval_oracle.json` (baixado de https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned): temporal-reasoning 133, multi-session 133, **knowledge-update 78 (6 de abstenção)**, single-session-user 70, -assistant 56, -preference 30; abstenção total 30. `_S` ≈ 115k tokens (~40 sessões), `_M` ≈ 500 sessões (~1,5M tokens) (`README.md:75-76`). Cada questão traz `haystack_sessions` com `has_answer: true` nos turnos de evidência e `answer_session_ids` (`README.md:87-88`) — nas KU a média é exatamente 2,0 sessões de evidência (o fato antigo e o novo). Exemplo real (`question_id 6a1eabeb`): sessão 1 "I recently set a personal best time in a charity 5K run with a time of 27:12", sessão 2 "I'm hoping to beat my personal best time of 25:50 this time around", pergunta "What was my personal best time in the charity 5K run?", resposta "25:50". Juiz: GPT-4o com ">97% agreement with human experts" (paper HTML). Achados do paper para design de memória: decompor sessões em rounds, "key expansion" com fatos do usuário (+9,4% recall) e expansão temporal da query (+6,8–11,3% recall em temporal). Novidade: LongMemEval-V2 (2026-05, https://github.com/xiaowu0162/LongMemEval-V2, contexto agêntico) — não avaliado aqui.

**MemoryAgentBench** (arXiv 2507.05257, Hu/Wang/McAuley, ICLR 2026; https://github.com/HUST-AI-HYZ/MemoryAgentBench; dataset HF `ai-hyz/MemoryAgentBench`). Quatro competências: "accurate retrieval, test-time learning, long-range understanding, and selective forgetting/conflict resolution". A que interessa: **FactConsolidation** — pares contrafactuais do MQuAKE em que "newer facts have larger serial numbers", contextos de 6K a 262K tokens, single-hop (FC-SH) e multi-hop (FC-MH). Resultados (Table 3, via arxiv.org/html/2507.05257): GPT-4o-mini 128K 45,0/5,0; BM25 48,0/3,0; **Mem0 18,0/2,0; Zep 7,0/3,0**; Cognee 28,0/3,0; MemGPT 28,0/3,0. "All methods fail on the multi-hop situation". A crítica ao mem0 no paper (tempo de construção "20,000x that of BM25", perda de informação por extrair "facts") aparece em resumos secundários — não localizei o número no HTML; tratar como não verificado.

**Benchmarks 2026 focados em update/conflito** (todos arXiv, verificados pelo abstract):

| Benchmark | O que mede | Métricas | Sistemas / números |
|---|---|---|---|
| **MemConflict** (2605.20926, 2026-05-20; https://github.com/TaoZhen1110/MemConflict) | "memory validity as a query-conditioned fitness-for-use problem"; conflitos **dynamic** ("a later true update supersedes an earlier state"), **static** ("a later false contradiction should not overwrite a stable fact"), **conditional** ("multiple memories are valid under different conditions"). 12 perfis, ~52 sessões e ~124 queries por perfil (~1.500 queries). | Black-box: Answer Accuracy, **UOCS** (Update Order Consistency), **CRS** (Conflict Recognition); white-box: **SEH@K** (gold memory no top-K), **SRS** (rank do gold), EUG. | A-Mem, LangMem, Letta, MemOS, Mem0, Memobase; melhor AA médio MemOS 0,5539. Dataset JSONL público (`Data/Step4_4.jsonl`). |
| **Memora / FAMA** (2604.20006, Uddin et al., 2026-04-21) | Conversas de semanas a meses; tarefas remembering/reasoning/recommending. **FAMA** "penalizes reliance on obsolete or invalidated memory". | FAMA (LLM-judged). | 4 LLMs + 6 agentes de memória: "frequent reuse of invalid memories and failures to reconcile evolving memories. Memory agents offer marginal improvements". |
| **ForgetEval / Lethe** (2606.15903, 2026-06-14; https://github.com/deeplethe/lethe, MIT) | 1.000 casos templated em 5 famílias (supersession, decay, amnesia, purge, drift) + 385 adversariais (substring traps, prefix collisions, cross-lingual). Primitivas **supersede / release / purge**. | Pass/fail determinístico por substring no top-K, sem juiz LLM. | Lethe 99,3% template / 63,4% adversarial (91,7% com hooks LLM); **Mem0 v2.0.2 88,8% / 68,3%**; MemPalace 0% ("no deletion primitives"). Achado central: "Production failures are predominantly forgetting failures rather than recall failures". |
| **PrecisionMemBench / Tenure** (2605.11325, 2026-05-11) | "a system that dumps its entire belief store can achieve perfect recall and mask severe precision failures". 89 casos. | precision, noise isolation, session latency, **belief mutability**. | 13 configurações; baselines "precision scores clustering at 0.22 and below". |
| **Supersede** (2606.27472, 2026-06-25) | Subconjunto KU do LongMemEval: memória auto-mantida limitada vs full-context. "replacing an agent's full context with a bounded, self-maintained memory drops accuracy from 92% to 77% even on a frontier model (gpt-5.4) … The bottleneck is therefore memory maintenance, not comprehension". Ambiente RL aberto (verifiers/prime-rl). | supersession accuracy (usa valor atual, não o obsoleto). | Qwen2.5-3B 9,0% → 16,7% após GRPO. |
| **MemStrata** (2606.26511) | Seis benchmarks locais com modelo 7B; conhecimento estático vs evolutivo. | **stale-fact-error rate**; acurácia em evolving knowledge. | RAG 0,20-0,47 vs MemStrata 0,95-1,00; RAG serve valor obsoleto 15-40%. |
| BEAM (mem0, https://github.com/mem0ai/memory-benchmarks) | 10 "ability types" incl. `knowledge_update` e `contradiction_resolution`, 100K–10M tokens. | acurácia por habilidade. | Só Mem0 reportado: 64,1 (1M) / 48,6 (10M). |

**Números reportados por sistema** (com ressalva de que quase todos são auto-reportados com stacks diferentes; a página do mem0 de 2026-05-11 admite "None of these numbers were generated using the same model stack, judge model, or retrieval configuration", https://mem0.ai/blog/ai-memory-benchmarks-in-2026):

| Sistema | LoCoMo (J) | LongMemEval | Conflito/forgetting |
|---|---|---|---|
| Mem0 (paper 2025, gpt-4o-mini) | 66,88 ± 0,15; Mem0^g 68,44 ± 0,17 (`papers/mem0.txt:567-568`); por categoria single-hop J 67,13, multi-hop 51,15, open-domain 72,93, temporal 55,51 (`:429-432`) | — | MAB FC-SH 18% (2507.05257); ForgetEval 88,8%/68,3% (v2.0.2) |
| Mem0 (v3 ADD-only, 2026, plataforma) | 92,5 (`repos/mem0/README.md:49`) | 94,4 (`README.md:50`) | — |
| Zep (paper 2025) | 65,99 ± 0,16 no paper do mem0 (`mem0.txt:565`); 84% → 75,14 ± 0,17 (blog Zep, 2025-05-06, corrigido); 58,44 ± 0,20 no re-run do mem0 | 63,8 (gpt-4o-mini) / 71,2 (gpt-4o) (`zep.txt:380-382`); KU 74,4 / 83,3 (`:398, 404`) | MAB FC-SH 7% |
| LangMem | 58,10 ± 0,21, latência p50 de busca 17,99 s (`mem0.txt:564`) | — | MemConflict (avaliado; score individual não extraído) |
| Letta | 74,0 (filesystem + grep, gpt-4o-mini, blog 2025-08-12) | "no standardized LongMemEval result found" (mem0 State of Memory 2026) | MemConflict (avaliado); MAB MemGPT FC-SH 28% |
| Full-context | 72,90 (gpt-4o-mini, `mem0.txt:559`) | 55,4 / 60,2 (`zep.txt:376-381`); KU 76,9 / 78,2 | MAB GPT-4o-mini FC-SH 45% |
| MemOS | "ranks first in all categories" (paper, Fig. 1, sem tabela extraída) | — | MemConflict AA 0,5539 (melhor) |

**A disputa mem0 × Zep (2025) — o que cada lado alegou e o que ficou.**
- mem0, paper 2025-04 (arXiv 2504.19413): Zep J = 65,99% vs Mem0 66,88% / Mem0^g 68,44% (`papers/mem0.txt:565-568`), média de 10 runs com desvio.
- Zep, blog "Lies, Damn Lies, & Statistics: Is Mem0 Really SOTA in Agent Memory?" (2025-05-06, https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/): três erros de implementação alegados — "Mem0 utilized a user graph structure designed for single user-assistant interactions but assigned the user role to both participants"; timestamps "appended to messages, rather than using Zep's dedicated created_at field"; "Searches were performed sequentially instead of in parallel". Alegou 84% (depois 80%) e, com nota de correção, "75.14% +/- 0.17" — ~10% acima do "~68%" do mem0. Também criticou o LoCoMo em si: "Mem0's own results show their system being outperformed by a simple full-context baseline … ~73%, compared to Mem0's best score of ~68%".
- mem0, issue getzep/zep-papers#5 (2025-05-08, Deshraj, https://github.com/getzep/zep-papers/issues/5): a Zep "incorrectly incorporated questions from the adversarial (5th) category" no numerador sem contá-las no denominador; re-run com as 4 categorias, prompt alinhado e 10 runs: "Zep's mean accuracy is 58.44% ± 0.20"; e `created_at` "wasn't available when we ran our benchmarks … zep-cloud==2.12.1 … May 1, 2025 — well after our paper was published". A Zep editou o post reconhecendo o erro ("we erred in how we calculated Zep's LoCoMo score", conforme o ensaio "The Benchmark Theatre", 2026-05-20, https://essays.bloo-mind.ai/posts/2026-05-20-mem-eval/); sem resposta da Zep na issue.
- O que ficou: três números para a Zep no mesmo benchmark (58,44 / 65,99 / 75,14), nenhum reproduzido por terceiro neutro; o LoCoMo perdeu credibilidade como discriminador (juiz leniente, full-context vence, filesystem vence); a comunidade migrou para LongMemEval e para benchmarks de conflito/forgetting. Recomendação do ensaio de 2026: "Build baselines first … Demand multi-seed evaluation … Validate judge quality … build your own evaluation on real use cases".

### B.2 Métricas específicas de consolidação

**Há definição formal?** Sim, dispersa em 2026; nenhuma anterior a 2025 para além de QA:

- **Stale-fact-error rate** (MemStrata 2606.26511): fração de respostas em que o sistema serve o valor superado quando obrigado a responder. É a métrica mais direta de "o fato novo substituiu o velho?" e é determinística se o gabarito tem (subject, relation, old, new).
- **FAMA — Forgetting-Aware Memory Accuracy** (Memora 2604.20006): acurácia que "penalizes reliance on obsolete or invalidated memory"; requer rótulo válido/inválido por memória no gabarito.
- **UOCS / CRS** (MemConflict 2605.20926): reconhecer a ordem de atualização e reconhecer que há contradição — separam "escolheu o novo" de "percebeu o conflito".
- **SEH@K / SRS** (MemConflict): a memória de suporte correta está no top-K e em que rank — métrica de retrieval sobre o estado consolidado.
- **Precision / noise isolation / belief mutability** (PrecisionMemBench 2605.11325): pune despejar o store inteiro; mede se uma crença pode ser mutada sem deixar rastro no top-K.
- **ForgetEval pass/fail por primitiva** (2606.15903): para cada caso, depois de supersede/release/purge, o alvo antigo **não** deve aparecer no top-K (substring), e o novo deve.
- **Selective forgetting / FactConsolidation** (MAB 2507.05257): QA onde a resposta correta é o fato de maior serial.

O que **nenhum** desses mede e o anchor pode medir barato porque controla o store: taxa de contradição residual no store (pares de entradas vivas que o gabarito diz serem incompatíveis), taxa de duplicatas parafraseadas (entradas vivas que o gabarito diz serem o mesmo fato), tamanho do store por turno (entradas vivas / total, tokens vivos) e acurácia da operação (ADD/UPDATE/DELETE/NOOP contra rótulo). O paper do mem0 reporta apenas F1, BLEU-1, LLM-as-a-Judge por categoria (Table 1, `papers/mem0.txt:414-432`), tokens de memória e latência p50/p95 de busca e total (Table 2, `:535-568`; "Mem0 1764 tokens … search p50 0.148 s p95 0.200 s", `:567`). Nada de tamanho do store por turno, taxa de contradição, duplicatas ou update accuracy — confirmado por grep em `mem0.txt` (`contradict` só aparece na motivação, `:69-72`). O `memory-benchmarks` do mem0 também não: "The README does not explicitly detail evaluation of update accuracy, duplicate detection, contradiction handling, or store size metrics" (fetch de https://github.com/mem0ai/memory-benchmarks); só BEAM tem "knowledge_update" e "contradiction_resolution" como tipos de habilidade de QA.

### B.3 Golden set mínimo de consolidação (30–60 casos)

**Existe dataset público com rótulos ADD/UPDATE/DELETE/NOOP?** Não encontrado. O mais próximo: (a) ForgetEval (rótulo por primitiva supersede/release/purge + alvo que deve/não deve aparecer, 1.000 + 385 casos, MIT); (b) MemConflict (tipo de conflito + gold memory ids, mas em termos de query, não de operação: "Framework does not explicitly label individual memory items as valid/invalid. Rather, validity is query-conditioned"); (c) LongMemEval `knowledge-update` (78 pares old→new com turnos `has_answer`, sem rótulo de operação mas trivialmente mapeável para UPDATE); (d) MAB FactConsolidation (pares contrafactuais do MQuAKE — factual, não conversacional); (e) mem0 `history` table — formato `(old_memory, new_memory, event)` é um rótulo de operação, mas o mem0 não publica dataset com isso. O LongMemEval KU **dá para reaproveitar** como fonte de 10–15 casos reais de UPDATE (filtrar `question_type == "knowledge-update"`, pegar os dois turnos `has_answer`, usar `answer` como valor esperado final) — o Supersede (2606.27472) fez exatamente isso.

**Esquema proposto** (JSONL, um caso por linha; espelha `GoldenCase` em `evaluation/golden.py:29-40`):

```json
{"name": "move_city",
 "existing": [{"id": "m1", "content": "User lives in São Paulo"}],
 "turns": [{"role": "user", "content": "Me mudei pro Rio mês passado."}],
 "expected_ops": [{"op": "delete", "target": "m1"}, {"op": "add", "content_like": "Rio"}],
 "expected_state": {"live_contains": ["Rio"], "live_not_contains": ["São Paulo"]},
 "probes": [{"query": "where does the user live", "must_hit": ["Rio"], "must_not_hit": ["São Paulo"]}]}
```

`expected_ops` é opcional (rótulo de operação para acurácia por classe); `expected_state` e `probes` são o que importa para "melhor e não só menor" — são independentes de como o consolidador chegou lá (um `UPDATE` in-place e um `DELETE+ADD` são ambos aceitos pelo estado final). `content_like`/`must_hit` por substring como no ForgetEval (determinístico, sem juiz LLM); um `judge_fn` opcional no estilo dos callbacks de `evaluation/rag.py:41-46` para casos onde substring não basta.

**Cenários canônicos e contagem sugerida (total 48):**

| # | Cenário | Op esperada | Casos | Fonte de inspiração |
|---|---|---|---|---|
| 1 | Mudança de fato ("moro em SP" → "me mudei pro Rio") | DELETE(antiga)+ADD ou UPDATE — aceitar ambos via estado final | 8 (4 sintéticos + 4 do LongMemEval KU) | LongMemEval KU; MemConflict *dynamic*; Zep §2.2.3 |
| 2 | Inversão de preferência ("não gosto de café" → "passei a gostar") | idem #1; a antiga **não** pode continuar viva | 5 | mem0 prompt UPDATE exemplo; Memora |
| 3 | Paráfrase / mesmo fato com outras palavras | NOOP (zero entradas novas vivas) | 6 | MemStrata: cosseno não separa paráfrase de contradição (AUROC 0,59) — é o caso que o `SimilarityConsolidator` erra por design |
| 4 | Fato novo independente | ADD; nada invalidado | 5 | — |
| 5 | Enriquecimento não contraditório ("gosto de café" → "gosto de café sem açúcar") | UPDATE in-place, 1 entrada viva | 4 | mem0 paper: "augmentation … with complementary information" |
| 6 | Fato temporário vs permanente ("estou em Lisboa esta semana" com "moro no Rio" já no store) | ADD com `expires_at` futuro (ou tag), **sem** invalidar "moro no Rio" | 4 | MemConflict *conditional*; TSM (durativo) |
| 7 | Múltiplos fatos numa mensagem (2–3 fatos, um deles contradiz) | ADD×n + DELETE de exatamente 1 | 5 | mem0 v3 prompt ("extract each one separately") |
| 8 | Contradição parcial ("tenho dois gatos" → "adotei um terceiro gato") | invalidar a contagem antiga, manter o resto | 4 | MAB multi-hop; mem0^g "Necessity Principle" |
| 9 | Contradição falsa / ruído ("meu irmão mora em SP" com "moro no Rio" no store) | NOOP ou ADD; **nada** invalidado | 4 | MemConflict *static* ("a later false contradiction should not overwrite a stable fact"); ForgetEval adversarial (substring traps) |
| 10 | Re-afirmação após invalidação ("voltei pra SP") | ADD nova; a invalidada continua invalidada; "Rio" agora invalidado | 3 | mem0^g `ON MATCH SET r.valid = true`; Graphiti re-cria |

Métricas do relatório (todas em [0,1], gate com o mesmo padrão de `assert_metric_floor`):

- `op_accuracy` e matriz de confusão 4×4 (só nos casos com `expected_ops`).
- `stale_alive_rate`: fração de casos em que algum `live_not_contains` está vivo em `list_all()` após a consolidação — é a "update accuracy" invertida e a *stale-fact-error rate* aplicada ao store.
- `probe_stale_hit_rate`: fração de probes cujo top-k (`search`/`ScoredMemoryRetriever`) contém um `must_not_hit` — a stale-fact rate no retrieval (ForgetEval/MemStrata).
- `probe_recall`: fração de probes com todos os `must_hit` no top-k (SEH@K).
- `dup_rate`: entradas vivas além do esperado nos casos NOOP/paráfrase.
- `live_size_delta`: (vivas depois − vivas antes) − esperado; positivo persistente = "só cresce", negativo = "só encolhe".

Custo: os 48 casos são turnos únicos com 1–3 entradas pré-existentes; com um LLM mockado o teste roda em ms; com LLM real, ~48 chamadas por rodada.

### B.4 O que o harness local já tem e o que falta

**Já existe** (`src/anchor/evaluation/`):

- `models.py:10-32` `RetrievalMetrics` (precision@k, recall@k, f1, mrr, ndcg, hit_rate); `:35-54` `RAGMetrics` com `None` = não avaliado; `:57-68` `EvaluationResult`.
- `retrieval.py:18-84` `RetrievalMetricsCalculator.evaluate(retrieved, relevant, k)` — puro, sem LLM; NDCG binário e graduado (`:122-173`).
- `golden.py:29-40` `GoldenCase(query, relevant, name)`; `:53-78` `GoldenSetReport.mean/summary`; `:81-98` `load_golden_set` (JSONL com erro por linha); `:101-140` `evaluate_retriever(retriever, cases, k, scope)`; `:143-164` `assert_metric_floor(report, metric, floor)` — "The CI-gate primitive: call it from a test so any retrieval change that regresses the golden set fails the build with the actual number". Docstring do módulo: "a small (50-200 query) hand-checked golden set … Every change to chunking, embedding, sparse backend, or reranking is gated on recall@k holding or improving" (`golden.py:1-12`).
- `batch.py:124-208` `BatchEvaluator` (mean/p95/min por métrica); `ab_testing.py` (t-test pareado entre dois retrievers); `rag.py:19-93` `LLMRAGEvaluator` com callbacks `faithfulness_fn/relevancy_fn/precision_fn/recall_fn`; `evaluator.py:16-120` `PipelineEvaluator`; `human.py`.
- Testes: `tests/test_evaluation/test_golden.py:53-65` exercita `assert_metric_floor` passando em 0.4 e falhando em 0.9; `tests/test_retrieval/test_graph_ab_benchmark.py` usa o A/B para o grafo.
- **Gate de CI**: `.github/workflows/ci.yml:63` roda `uv run pytest --cov=anchor --cov-report=xml -q` — o gate existe como primitiva testada, mas **não há golden set commitado nem teste que chame `assert_metric_floor` sobre dados reais** (grep `golden` em `tests/` → só `test_exports.py`, `test_golden.py`, `tests/fixtures/graph_corpus/README.md`).

**Nada mede memória/consolidação**: tudo é `Retriever.retrieve(QueryBundle) → ContextItem` contra ids relevantes (`golden.py:129-132`, `batch.py:170-176`). `MemoryEntry`, `MemoryConsolidator` e `MemoryOperation` não aparecem em `evaluation/` (grep). O único "harness" de memória são os testes unitários do `SimilarityConsolidator` e a lista de verificação do plano #5 ("moro em SP → me mudei pro Rio deve virar UPDATE", `docs/plans/2026-08-28-v020-5-consolidador-mem0.md`, seção Verificação).

**O que precisaria de novo** (um módulo, ~120 linhas, seguindo a forma de `golden.py`):

1. `evaluation/consolidation.py`: `ConsolidationCase` (pydantic frozen: `name, existing: list[dict], turns: list[dict], expected_ops: list[dict] = [], expected_state: dict, probes: list[dict] = []`), `load_consolidation_set(path)` (mesma leitura JSONL com erro por linha), `evaluate_consolidator(extractor, consolidator, store_factory, cases, *, retriever_factory=None, k=5) -> ConsolidationReport`, `ConsolidationReport.mean(metric)`/`summary()` com as seis métricas de B.3, e reutilizar `assert_metric_floor` — hoje ele é tipado como `GoldenSetReport` (`golden.py:143-145`), então ou aceita um `Protocol` com `.mean(metric)` e `.results`, ou o `ConsolidationReport` copia os quatro campos. O runner por caso: `store = store_factory(); for e in existing: store.add(MemoryEntry(**e))`; `entries = extractor.extract(turns)`; `actions = consolidator.consolidate(entries, store.list_all())`; aplicar as ações com a mesma função do pipeline (`_store_with_consolidation`, `pipeline/memory_steps.py:28-49` — que precisa do ramo DELETE do Tema A para o teste ter o que medir); comparar `store.list_all()` com `expected_state` e rodar `probes` via `store.search` ou `ScoredMemoryRetriever`.
2. `tests/fixtures/consolidation_golden.jsonl` com os 48 casos, e um teste `tests/test_evaluation/test_consolidation.py` que roda o `SimilarityConsolidator` (baseline: deve **falhar** em #1, #2, #3 parafraseada por design) e o futuro `LLMConsolidator` com LLM mockado; o gate de CI é `assert_metric_floor(report, "stale_alive_rate_inv", 0.9)` no teste com mock, e um caso `live` opcional (`tests/live/`) com LLM real.
3. Fora de escopo agora (YAGNI): juiz LLM para `must_hit` semântico (adicionar quando um caso do golden set não for expressável por substring), benchmark externo (LongMemEval KU completo com 78 casos exigiria ingestão de sessões inteiras e um extractor real — vale como `tests/live/` depois que o `LLMExtractor` existir).

### Recomendação para o anchor (Tema B)

**Decisão sugerida:** não medir consolidação por QA externo; medir **estado do store + probes de retrieval** num golden set próprio de ~48 casos com gabarito determinístico (substring), no molde do `golden.py` existente, com três números gateáveis: `stale_alive_rate` (fato velho continua vivo), `probe_stale_hit_rate` (fato velho volta no top-k) e `dup_rate` (paráfrase virou duplicata); `op_accuracy` fica como diagnóstico, não como gate, porque UPDATE in-place e DELETE+ADD são equivalentes no estado final. Reaproveitar 4–10 casos reais do `knowledge-update` do LongMemEval (pares `has_answer`) e, mais tarde, rodar o subconjunto inteiro como teste `live`. **Evidência:** nenhum benchmark público rotula operações (B.3); os que medem consolidação em 2026 fazem exatamente isso — ForgetEval (substring no top-K, sem juiz), MemStrata (stale-fact-error rate), PrecisionMemBench (precisão do retrieval sobre o store), MemConflict (SEH@K/UOCS); o LoCoMo/LLM-judge foi desacreditado pela disputa mem0×Zep e pelo teste de leniência (62,81%); e o harness local já tem a primitiva de gate (`golden.py:143-164`) e o padrão de JSONL + relatório para copiar. Pré-requisito: o ramo `DELETE` em `pipeline/memory_steps.py:44-46`, sem o qual o golden set não tem como distinguir "invalidou" de "ignorou".


---

## Relatório 4 — Mapa do código da `dev` para a #5

# Mapa de código — frente #5 (LLMExtractor + LLMConsolidator, paridade mem0)

Repo `/Users/arthurgranja/github/anchor`, branch `dev` @ `3c88e66`. Tudo abaixo foi lido no código; nada foi modificado. Caminhos relativos a `src/anchor/` salvo indicação.

---

## 0. Resumo executivo

| Peça | Estado hoje | Onde |
|---|---|---|
| `MemoryOperation` ADD/UPDATE/DELETE/NONE | pronto | `protocols/memory.py:18-24` |
| `MemoryExtractor` / `AsyncMemoryExtractor` | prontos | `protocols/memory.py:70-109` |
| `MemoryConsolidator` | pronto, **só sync** — não existe `AsyncMemoryConsolidator` | `protocols/memory.py:113-139` |
| Único aplicador de `consolidate()` | `_store_with_consolidation` — aplica ADD/UPDATE, **ignora DELETE e NONE** | `pipeline/memory_steps.py:28-49` |
| Chamadores de `extract()` | `auto_promotion_step` e `create_eviction_promoter` | `pipeline/memory_steps.py:205,254` |
| `MemoryManager` | não conhece extractor, consolidator nem callbacks; sem hook de fim de turno | `memory/manager.py:55,57-82` |
| `MemoryCallback.on_extraction/on_consolidation` | declarados, **ninguém dispara** | `memory/callbacks.py:60-84`; só `gc.py:169,214` dispara os de prune |
| `MemoryEntryStore.search` | **substring/LIKE em todos os backends**; nenhum store de memória com embedding | §3 |
| Padrão LLM→JSON fail-soft | `LLMGraphExtractor` e `TierCompactor` | `ingestion/graph_extractors.py:257-275`, `memory/compactor.py:94-103,178-221` |
| `tool_choice` genérico no `invoke` | sim, via `**kwargs` — permite output estruturado por tool call sem `Agent` | §4 |

---

## 1. Seams do protocolo

### 1.1 O protocolo (`protocols/memory.py`)

```python
class MemoryOperation(StrEnum):            # :18-24
    ADD = "add"; UPDATE = "update"; DELETE = "delete"; NONE = "none"

class MemoryExtractor(Protocol):           # :70-88
    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...

class AsyncMemoryExtractor(Protocol):      # :92-109
    async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...

class MemoryConsolidator(Protocol):        # :113-139
    def consolidate(self, new_entries: list[MemoryEntry], existing: list[MemoryEntry])
        -> list[tuple[MemoryOperation, MemoryEntry | None]]: ...
```

- Docstring de `consolidate` (`:134-137`): "For `ADD` and `UPDATE` operations the entry must be non-None. For `DELETE` and `NONE` operations the entry may be None." E `existing` é descrito como "All entries currently in the memory store" (`:132`).
- **`AsyncMemoryConsolidator` não existe.** Confirmado: o arquivo tem `AsyncCompactionStrategy` (`:49`) e `AsyncMemoryExtractor` (`:92`), mas nenhum async para consolidator; `grep -rn AsyncMemoryConsolidator src/anchor/` retorna vazio. Exports em `protocols/__init__.py:8-22` e `anchor/__init__.py:93,344,360,363` listam só os três acima.
- `MemoryEntry` (`models/memory.py:41-87`): `id` uuid4, `content`, `content_hash` (md5, `:36-38`, preenchido por `model_validator` `:66-71` — **`model_copy` pula o validator**, por isso quem reescreve `content` deve recomputar o hash, ver commit `3c88e66` e `memory/consolidator.py:83-85`), `expires_at` + `is_expired` (`:60,73-78`), `updated_at`, `source_turns`, `links`.
- `ConversationTurn` (`:26-33`): `role: Role` (StrEnum de `llm/models.py:15-21`, aceita `"user"`), `content`, `token_count`, `timestamp`, `metadata`.

### 1.2 Todos os callers de `.consolidate(` e `.extract(` no pacote

`grep -rn "\.consolidate(\|\.extract(" src/anchor/`:

| Linha | Quem | O que faz com o resultado |
|---|---|---|
| `pipeline/memory_steps.py:43` | `_store_with_consolidation` | itera `(action, entry)`; `store.add(entry)` só se `action in (ADD, UPDATE) and entry is not None` (`:44-46`). DELETE/NONE caem no chão. |
| `pipeline/memory_steps.py:205` | `auto_promotion_step._promote` | converte `ContextItem`s MEMORY/CONVERSATION em `ConversationTurn` (`:184-203`), `extractor.extract(turns)`, depois `_store_with_consolidation(new_entries, store, consolidator)` (`:209`). Step side-effect-only, devolve items intactos. |
| `pipeline/memory_steps.py:254` | `create_eviction_promoter._on_evict` | `extractor.extract(turns)` → `_store_with_consolidation` (`:258`); `try/except Exception: logger.exception(...)` nunca propaga (`:253-262`). Usado como `SlidingWindowMemory(on_evict=...)`. |
| `ingestion/graph_extractors.py:498` | `GraphIndexer.index` | é o **outro** protocolo `GraphExtractor.extract(item) -> Extraction` (`:118-122`), não `MemoryExtractor`. Não confundir. |

`MemoryManager` **nunca** chama `extract` nem `consolidate` (grep em `memory/manager.py`: zero). `Agent` também não.

### 1.3 `_store_with_consolidation` — código integral

`pipeline/memory_steps.py:28-49`:

```python
def _store_with_consolidation(
    entries: list[MemoryEntry],
    store: MemoryEntryStore,
    consolidator: MemoryConsolidator | None,
) -> None:
    """Persist entries, optionally deduplicating via a consolidator.

    If *consolidator* is provided, new entries are consolidated against
    entries already in the store.  Only ``ADD`` and ``UPDATE`` operations
    result in a ``store.add()`` call.
    ...
    """
    if consolidator is not None:
        existing = store.list_all()                                    # :42  ← store INTEIRO
        actions = consolidator.consolidate(entries, existing)          # :43
        for action, entry in actions:
            if action in (MemoryOperation.ADD, MemoryOperation.UPDATE) and entry is not None:
                store.add(entry)                                       # :45-46
    else:
        for entry in entries:
            store.add(entry)                                           # :47-49
```

Observações que importam para a #5:

1. **DELETE é ignorado por design documentado** (`:36-37`). Nenhum teste cobre DELETE: `grep -rn "MemoryOperation.DELETE\|\"delete\"" tests/test_memory tests/test_pipeline` → vazio. `docs/docs/api/memory.md:276` documenta `SimilarityConsolidator` como "ADD / UPDATE / NONE" — DELETE nunca foi emitido por ninguém.
2. **UPDATE = `store.add(entry)`** e o protocolo de storage diz que `add` sobrescreve pelo `id` (`protocols/storage.py:231-243`; `InMemoryEntryStore.add` `storage/json_memory_store.py:31-34` é `self._entries[entry.id] = entry`; SQLite `INSERT OR REPLACE` `storage/sqlite/_entry_store.py:14-22`; Postgres `ON CONFLICT ... DO UPDATE` `storage/postgres/_entry_store.py:57`). Logo **um UPDATE só é UPDATE se a entry devolvida carrega o `id` da existente** — `SimilarityConsolidator._merge_entries` faz `existing.model_copy(update=...)` (`memory/consolidator.py:80-94`) exatamente por isso. Um `LLMConsolidator` que devolva `MemoryEntry(content=novo)` com id novo vira ADD silencioso.
3. **`existing = store.list_all()`**: o consolidador recebe o store inteiro. Para o `SimilarityConsolidator` é O(n) embeddings cacheados (`consolidator.py:55-61,116`); para um LLM é O(n) tokens no prompt. A seleção dos S similares tem de acontecer antes ou dentro do consolidador (§3).
4. Não dispara `on_extraction`/`on_consolidation` — os callbacks existem (`memory/callbacks.py:60-84`) e o helper `_fire_memory_callback` (`:129-147`) só é usado por `gc.py:169,214`. São eventos mortos hoje.
5. Callers passam ops como **string** em testes (`tests/test_pipeline/test_memory_steps.py:442-517`, `tests/test_memory/test_auto_promotion.py:165-183` devolvem `("none", None)`/`("add", entry)`); funciona porque `MemoryOperation` é `StrEnum` (`"add" == MemoryOperation.ADD`). A comparação `action in (ADD, UPDATE)` aceita ambos.

### 1.4 O furo de DELETE no protocolo e a menor correção

O protocolo diz que para DELETE "the entry may be None" (`protocols/memory.py:137`). Sem entry não há **id do alvo**: a tupla `(DELETE, None)` é inaplicável — nem o aplicador atual nem qualquer outro poderia saber o que apagar. Sim, é um furo do protocolo: a forma `(op, entry | None)` só é completa se o slot `entry` carregar o alvo nas ops que têm alvo.

**Menor correção compatível (sem mudar assinatura):** fixar a semântica do slot `entry` por operação e aplicar no único aplicador.

- `protocols/memory.py:134-137` — trocar o docstring por: *ADD → a nova entry; UPDATE → a entry existente reescrita (mesmo `id`); DELETE → a entry existente a remover (o `id` é o alvo); NONE → `None`.* Também relaxar `:132` ("All entries currently in the memory store") para "candidate entries (all, or a similarity-selected subset)".
- `pipeline/memory_steps.py:44-46` — acrescentar o ramo:

  ```python
  elif action is MemoryOperation.DELETE and entry is not None:
      store.delete(entry.id)
  ```

  `store.delete(id) -> bool` já é parte do protocolo (`protocols/storage.py:270-283`) e do wrapper de grafo (`ingestion/graph_extractors.py:407-411`).

Isso não quebra ninguém: `SimilarityConsolidator` nunca emite DELETE (`consolidator.py:107-113`), os consolidators de teste tampouco. `(DELETE, None)` continua válido e continua sendo no-op (não há o que apagar) — só passa a ser documentado como "sem alvo = nada a fazer".

Alternativa mais rica (não recomendada agora): novo tipo de retorno com `target_id` — quebra o protocolo e os 3 sites de teste que implementam `consolidate` inline.

Soft-delete (`expires_at=now`) como variante: `store.add(entry.model_copy(update={"expires_at": datetime.now(UTC)}))` — mesma quantidade de linhas, mas esbarra no wrapper de grafo (§6.1) e deixa a entry em `list_all_unfiltered` até o GC. Ver decisão em §10.

---

## 2. `MemoryManager` e o turno do `Agent`

### 2.1 O que o manager tem hoje (`memory/manager.py`)

- `__slots__ = ("_conversation", "_persistent_store", "_tokenizer")` (`:55`) — qualquer campo novo (`_extractor`, `_consolidator`) exige entrar aqui.
- Construtor `:57-82`: `conversation_tokens`, `tokenizer`, `on_evict`, `persistent_store`, `conversation_memory`, `graph`. Se `graph` e `persistent_store` → embrulha em `GraphIndexingEntryStore` (`:67-70`, import lazy). Esse é o precedente exato para uma opção opt-in: parâmetro novo no ctor + import lazy + wrap/uso condicional.
- Mensagens: `_add_message` (`:119-132`) roteia para `add_message` (Progressive/SummaryBuffer) ou `add_turn` (SlidingWindow); `add_user_message/add_assistant_message/add_system_message/add_tool_message` (`:134-148`). Todas **sync**.
- Fatos: `add_fact` (`:152-195`) com dedupe por hash **varrendo `list_all()`** (`:182-186`, O(n)); `get_relevant_facts` → `store.search(query, top_k)` (`:197-204`); `get_all_facts` (`:206-213`); `delete_fact` (`:215-223`); `update_fact` (`:225-253`) — acha por id em `list_all()`, `model_copy` com `content`, **`content_hash` recomputado**, `updated_at`, e `store.add(updated)` (mesmo id → overwrite). **`update_fact` é o modelo de como um UPDATE deve ser materializado.**
- `get_context_items` (`:257-289`): injeta **todas** as entries não expiradas do store a prioridade 8, sem top_k. (Motivo prático da #5: memória que cresce = prompt que cresce.)
- `conversation.turns` está disponível via protocolo `ConversationMemory.turns` (`protocols/memory.py:253-254`), implementado em `sliding_window.py:69`, `summary_buffer.py:204`, `progressive.py:151` → **as últimas M mensagens são `manager.conversation.turns[-M:]`**.

### 2.2 Hooks / callbacks

- `MemoryCallback` (`memory/callbacks.py:26-103`): `on_eviction`, `on_compaction`, `on_extraction(turns, entries)`, `on_consolidation(action, new_entry, existing_entry)`, `on_decay_prune`, `on_expiry_prune`. Manager **não** aceita callbacks; só `MemoryGarbageCollector(callbacks=)` (`gc.py:99-107`) dispara `on_expiry_prune`/`on_decay_prune`.
- Não há hook "turno terminou". O único ponto de eviction é `SlidingWindowMemory(on_evict=...)` (`sliding_window.py:46,141-146`, chamado fora do lock, exceções engolidas) — é onde `create_eviction_promoter` se pendura hoje. Eviction é gatilho por **orçamento de tokens**, não por turno: não serve para "extrair a cada turno".

### 2.3 Como o `Agent` toca o manager por turno (`agent/agent.py`)

| Momento | sync `stream()` | async `astream()` | Sync/async |
|---|---|---|---|
| attach | `with_memory` `:383-387` (também `pipeline.with_memory`) | idem | — |
| mensagem do usuário | `:2272-2273` `self._memory.add_user_message(message)` | `:2366-2367` (depois de `await self._ensure_mcp()`) | **sync** |
| leitura da conversa | `self._pipeline.build(message)` `:2281` → `ContextPipeline._collect_base_items` `pipeline/pipeline.py:282-285` chama `memory.get_context_items()` → formatter → `_prepare_turn` `:1691-1715` | `abuild` `:2371` | sync (`get_context_items`) |
| tool calls | `_record_tool_call` `:1606-1616` → `add_tool_message("[Tool: name] Input: … → Result: …")`, chamado de `_error_result` `:1123` e `_ok_result` `:1146` | mesmos helpers | **sync** |
| resposta final | `_finish_turn` `:2178-2191`: `add_assistant_message(final_text)` se houver texto; chamado no `finally` de `stream` (`:2346`) e de `astream` (`:2438`) | idem | **sync** |
| contrato | `_reset_turn_state` `:2160-2176`: um turno por vez por instância | | |

**O follow-up (13/5) da sessão 8** está em `docs/plans/2026-08-31-release-engineering.md:130-134` e `:288-290`: "todo o caminho async usa `add_*_message` sync (agent.py:2257/1551/2102); com `ProgressiveSummarizationMemory`, eviction roda `summarize` (HTTP sync) NO loop … Os `aadd_*` existem e ninguém chama". Confirmação no código atual:

- `astream` → `add_user_message` (`:2367`) → `_add_message` (`manager.py:119-126`) → `ProgressiveSummarizationMemory.add_message` → `_handle_eviction` (`progressive.py:272-293`) → `self._compactor.summarize(...)` (`:289`) → `TierCompactor.summarize` → `self._llm.invoke(...)` **sync** (`compactor.py:95-97`). Tudo dentro do event loop.
- `ProgressiveSummarizationMemory.aadd_message` existe (`progressive.py:221-242`) e `grep -n "aadd_\|to_thread" agent/agent.py memory/manager.py` → vazio.

Conclusão para a #5: **qualquer consolidação por LLM colocada dentro de `_finish_turn`/`add_assistant_message` repete o bug 13/5** (round-trip de LLM bloqueando o loop no caminho async) e ainda rodaria dentro de um `finally` que executa também quando o consumidor abandona o generator (`:2340-2346`).

### 2.4 Onde o fluxo "turno terminou → extrair M → buscar S → consolidar → aplicar" se encaixa

Ponto de costura mais barato: **imediatamente após o bloco `try/finally` de `stream`/`astream` e antes de `yield TurnFinished`** (`agent.py:2347` e `:2439`). Vantagens:

- Só roda em turno completo (no abandono, o `GeneratorExit` sai pelo `finally` e nunca chega ali).
- No sync: `self._memory.remember()`; no async: `await asyncio.to_thread(self._memory.remember)` — o extractor/consolidator continuam sync (sem inventar `AsyncMemoryConsolidator`) e o loop não bloqueia. Stdlib, uma linha, rung 3 da escada.
- O método vive no manager (`MemoryManager.remember()`), que já tem `conversation.turns`, `persistent_store` e pode importar `_store_with_consolidation` lazy (o import direto `memory → pipeline.memory_steps` é seguro: `pipeline/memory_steps.py:7-16` só importa `models`, `pipeline.step` e `protocols`; mas seguir o precedente lazy de `manager.py:68` evita o ciclo via `pipeline/__init__.py:3-7`).

Alternativa "zero linhas no Agent": disparar dentro de `add_assistant_message`. Rejeitada pelo motivo do 13/5.

O extrator deve filtrar `role in ("user", "assistant")`: `_record_tool_call` injeta turnos `tool` com input/resultado truncados a `_TOOL_MEMORY_TRUNCATE = 200` (`agent.py:87,1611-1616`) — ruído para extração de fatos sobre o usuário.

---

## 3. Busca dos S similares — o que cada backend faz

Protocolo `MemoryEntryStore.search(query, top_k=5)` (`protocols/storage.py:245-258`): "matching semantics (keyword, embedding, hybrid) are implementation-defined". `list_all` documentado como "every **non-expired** entry" (`:260-268`).

| Backend | Classe | `search` | `list_all` filtra `expires_at`? | `search` filtra? |
|---|---|---|---|---|
| in-memory | `InMemoryEntryStore` `storage/json_memory_store.py:11` via `BaseEntryStoreMixin` `storage/_base.py:62-73` | `_filters.search_entries` `storage/_filters.py:41-58`: **substring** `query_lower in entry.content.lower()`, ordena por `relevance_score` desc | sim `_filters.py:61-63` | sim `:53` |
| JSON file | `JsonFileMemoryStore` `storage/json_file_store.py:20` (mesmo mixin) | idem | sim | sim |
| SQLite sync | `SqliteEntryStore` `storage/sqlite/_entry_store.py:28` | `content LIKE %q% ESCAPE` + `_NON_EXPIRED_CLAUSE` `:25,51-59` | sim `:61-68` | sim |
| SQLite async | `AsyncSqliteEntryStore` `:183` | `:201-212` mesmo LIKE | sim `:214` | sim |
| Postgres (só async) | `PostgresEntryStore` `storage/postgres/_entry_store.py:23` | `content ILIKE` + `expires_at IS NULL OR > $1` `:79-92` | sim `:94-101` | sim |
| Redis sync | `RedisEntryStore` `storage/redis/_entry_store.py:14` | carrega todos por `smembers`+`mget` e filtra substring em Python `:83-95` | sim `:97-101` | sim `:92` |
| Redis async | `AsyncRedisEntryStore` `:242` | `:297-309` | sim `:311-315` | sim |
| `GraphIndexingEntryStore` | `ingestion/graph_extractors.py:375` | delega `:420-421` | delega `:423-424` | delega |

Todos filtram expiração de forma consistente em `search` e `list_all`; `list_all_unfiltered` (`GarbageCollectableStore`, `protocols/storage.py:295-314`) é a única leitura que devolve expirados.

**Não existe store de memória com embedding.** `grep MemoryEntry storage/sqlite/_vec_store.py storage/sqlite/_vector_store.py storage/postgres/_vector_store.py` → vazio: os vector stores são `VectorStore` de `(item_id, embedding)` (`protocols/storage.py:83-160`), sem ligação automática com `MemoryEntry`.

`ScoredMemoryRetriever` (`retrieval/memory_retriever.py:22-130`): usa `embed_fn` + `vector_store` **se ambos forem passados** (`:109-113`, `vector_store.search(query_embedding, top_k*3)`), senão "keyword overlap"; candidatos vêm de `store.list_all()` (`:116`) e expirados são filtrados de novo (`:123`). Ninguém escreve embeddings de `MemoryEntry` no `vector_store` — o usuário teria de indexar por fora. Não dá para contar com ele como "busca dos S similares" pronta.

**Consequência:** `store.search(fato_novo)` com substring quase nunca acha "Moro em SP" a partir de "Me mudei pro Rio". Opções, da mais barata:

1. `existing = store.list_all()` (o que `_store_with_consolidation:42` já faz) com **cap** (ex.: 50 mais recentes/relevantes) → é o "S" para stores pequenos. Zero código novo além do cap.
2. `embed_fn` opcional no `LLMConsolidator`, reaproveitando a ideia de `SimilarityConsolidator._get_embedding` + `anchor._math.cosine_similarity` (`consolidator.py:14,55-61,127-136`) para escolher top-S dentre `existing` antes do prompt. É o mem0 (busca vetorial dos S) em processo.
3. `ScoredMemoryRetriever` com `vector_store` — exige indexação externa; fora do escopo.

---

## 4. Superfície do LLM

### 4.1 `LLMProvider` (`llm/base.py`)

```python
class LLMProvider(Protocol):                                   # :24-75
    def invoke(self, messages: list[Message], *, tools: list[ToolSchema] | None = None,
               max_tokens: int | None = None, temperature: float | None = None,
               stop: list[str] | None = None, **kwargs: Any) -> LLMResponse: ...   # :33-42
    async def ainvoke(...same...) -> LLMResponse: ...                                # :55-64
```

`BaseLLMProvider.invoke/ainvoke` (`:129-136,169-178`) só repassam `tools` e `**kwargs` para `_do_invoke/_do_ainvoke` com retry. **`tool_choice` viaja em `**kwargs`** e cada provider o lê:

- Anthropic: `anthropic.py:210-211` `if kwargs.get("tool_choice") is not None and tools: call_kwargs["tool_choice"] = _convert_tool_choice(...)`; valores `"auto"|"any"|"none"` (`:76`) ou dict `{"type": "tool", "name": ...}` passthrough (`:82-92`). `_do_invoke` `:235-252` → `_parse_response` popula `tool_calls` (`:464`).
- OpenAI/Grok/OpenRouter/Ollama/LiteLLM: `_openai_compat.py:149-163,188-189` (`any`→`required`, dict→function form).
- Gemini: `gemini.py:85-99,281-282` (`tool_config`).
- `claude_cli`: **best-effort** — vira instrução no system prompt (`claude_cli.py:330-338,554-559`: "You must call the X tool before answering"); `"none"` é exato (`:224`). `_do_invoke` sync existe (`:437-450`) por uma thread com loop próprio (`:187-192`).

Models (`llm/models.py`): `Message(role, content, tool_calls, tool_result, name, raw_content)` `:59-73`; `LLMResponse(content, tool_calls: list[ToolCall] | None, usage, model, provider, stop_reason, raw_content)` `:95-108`; `ToolCall(id, name, arguments: dict)` `:34-39`; `ToolSchema(name, description, input_schema, input_examples=())` `:126-132`; `StopReason.TOOL_USE` `:87-92`.

### 4.2 Output estruturado por tool call **sem** o `Agent`

Sim, em uma chamada, sem loop:

```python
schema = ToolSchema(
    name="memory_decisions",
    description="Record the consolidation decisions.",
    input_schema=Decisions.model_json_schema(),      # ou clean_schema(...) de agent/schema.py:197-216
)
resp = llm.invoke(
    [Message(role=Role.USER, content=prompt)],
    tools=[schema],
    tool_choice={"type": "tool", "name": "memory_decisions"},
)
args = (resp.tool_calls or [ToolCall(id="", name="", arguments={})])[0].arguments
decisions = Decisions.model_validate(args)
```

É o que `Agent._round_request` monta (`agent/agent.py:1717-1757`: `schemas = [t.to_tool_schema()]`, `call_extra["tool_choice"] = {"type": "tool", "name": "final_result"}` / `"any"`), e `_make_output_tool` (`:535-551`) usa `input_schema=output_model.model_json_schema()` diretamente. `AgentTool.to_tool_schema` está em `agent/models.py:45-54`.

Ressalvas: (a) em `claude_cli` a forçagem é por prompt, não garantida; (b) a `FakeLLM` dos testes devolve `content`, não `tool_calls` (§5); (c) os dois callers LLM já existentes na base usam **prompt → JSON no `content`**.

### 4.3 O padrão a reusar: prompt → JSON → `strip_markdown_fences` → fail-soft

- `LLMGraphExtractor.extract` (`ingestion/graph_extractors.py:257-275`): monta prompt (`_EXTRACTION_PROMPT` `:218-236`), `self._llm.invoke([Message(role=Role.USER, content=prompt)])`, `json.loads(strip_markdown_fences(response.content or ""))`, valida forma (`isinstance(data, dict)`), `_parse` **dentro do try**; `except Exception: logger.warning(...); return Extraction()`. Provider injetado (`:253-255`), "never a new client".
- `TierCompactor` (`memory/compactor.py`): `summarize` `:80-103` (`invoke`, fallback truncado + `logger.exception`), `asummarize` `:105-126` (`ainvoke`), `extract_facts` `:128-132` → `_call_fact_extraction` `:158-166` (warning, devolve `"[]"`) → `_parse_facts` `:178-221` (strip fences, valida lista/dicts/tipos, **um retry com prompt mais estrito** `:59-62,207-217`), variante async `:223-267`.
- `strip_markdown_fences` em `_text.py:6-18`.

Recomendação: seguir `LLMGraphExtractor` (JSON no content, um `invoke`, fail-soft com `logger.warning`). O tool-call fica como upgrade de uma linha (`tools=`+`tool_choice=`) quando um provider errar o JSON.

### 4.4 `Agent.with_output_model` / `run()` de dentro de um consolidador?

Pesado demais, e errado na direção:

- `Agent.__init__` (`agent/agent.py:283-372`) cria `ContextPipeline` + `AnthropicFormatter` (`:368-372`), `SkillRegistry` (`:331`), 40+ slots; `run()` (`:2498-2516`) executa `stream()` inteiro: `pipeline.build`, rounds, tool phase, `_finish_turn`.
- `with_output_model` (`:486-535`) em modo `tool` exige o tool `final_result` + `tool_choice="any"` (`:1748-1757`); modo `prompted` **recusa agente com memória** (`:2467-2473`).
- Um consolidador dentro do manager que o próprio `Agent` possui criaria um segundo `Agent` por turno — e o sub-agent é o caminho "subagent" com guards de nesting (`agent/subagent.py:63-86`).

O que o `Agent` oferece de útil é só `agent.llm` (propriedade usada em `docs/docs/guides/knowledge-graph.md:103-108`: `LLMGraphExtractor(agent.llm)`). O consolidador deve receber esse `LLMProvider`, mais nada.

---

## 5. Testes

### 5.1 Fakes de LLM existentes

| Fixture | Onde | Serve para `invoke`? |
|---|---|---|
| `FakeLLM(content)` — `.invoke(messages, **kw)` devolve `SimpleNamespace(content=...)`, grava `prompts` | `tests/test_ingestion/test_llm_graph_extractor.py:54-61` | **sim** — é o fake certo para `LLMExtractor`/`LLMConsolidator`. `Boom` (`:106-110`) cobre provider error. |
| `MagicMock()` com `invoke.return_value = LLMResponse(...)` + `FakeTokenizer` | `tests/test_memory/test_compactor.py:14-31`; `FakeTokenizer` em `tests/conftest.py:32-53` | sim (permite `side_effect` para sequência/erro) |
| `FakeLLMProvider(responses: list[list[StreamChunk]])` | `tests/test_agent/test_agent.py:68-135` | **não** — `invoke`/`ainvoke` fazem `raise NotImplementedError` (`:128-135`). Só para o loop do agente. |

### 5.2 O que `tests/test_memory/test_consolidator.py` cobre hoje

`TestSimilarityConsolidatorDedup` hash exato → NONE (`:44-71`); `TestSimilarityConsolidatorUpdate` merge (conteúdo maior, tags/links/metadata, `access_count`, `relevance_score`) (`:74-171`); `TestSimilarityConsolidatorAdd` (`:174-203`); `TestSimilarityConsolidatorMultiple` (`:206-229`); validação de threshold (`:232-245`); `TestMergedEntryHash` recomputa hash (`:248-258`). **Zero DELETE, zero LLM.** Aplicação de ops só em `tests/test_pipeline/test_memory_steps.py:442-517` e `tests/test_memory/test_auto_promotion.py:150-232` (consolidators inline com strings `"none"/"add"/"update"`); `tests/test_memory/test_manager_graph.py:71-116` chama `_store_with_consolidation` direto com `None`.

### 5.3 Live

- Gate: `_CLAUDE_CLI = bool(shutil.which("claude")) and bool(importlib.util.find_spec("claude_agent_sdk"))` + `@pytest.mark.skipif(not _CLAUDE_CLI, ...)` (`tests/live/test_graph_llm_live.py:17-20`); provider `ClaudeCLIProvider(model="sonnet", max_response_tokens=600)` (`:22-24`), import dentro do teste. Providers com chave: env-gated (`tests/live/test_providers_live.py:26-35,57`). Não há marker `live` em `pyproject.toml:150-154` (só `slow`, `integration`); `tests/live/` roda com o resto e simplesmente pula.

### 5.4 Receita: "moro em SP → me mudei pro Rio vira UPDATE" com LLM fake

```python
# tests/test_memory/test_llm_consolidator.py
import json
from types import SimpleNamespace
from anchor.memory.consolidator import LLMConsolidator          # novo
from anchor.models.memory import MemoryEntry, _compute_content_hash
from anchor.pipeline.memory_steps import _store_with_consolidation
from anchor.protocols.memory import MemoryOperation
from anchor.storage.json_memory_store import InMemoryEntryStore

class FakeLLM:                                                   # cópia do de test_llm_graph_extractor.py:54
    def __init__(self, content): self.content, self.prompts = content, []
    def invoke(self, messages, **kwargs):
        self.prompts.append(messages[0].content); return SimpleNamespace(content=self.content)

def test_move_becomes_update():
    store = InMemoryEntryStore()
    old = MemoryEntry(id="m1", content="Mora em São Paulo")
    store.add(old)
    llm = FakeLLM(json.dumps([{"op": "update", "target": "m1", "content": "Mora no Rio de Janeiro"}]))
    ops = LLMConsolidator(llm).consolidate([MemoryEntry(content="Me mudei para o Rio")], store.list_all())
    assert ops[0][0] is MemoryOperation.UPDATE and ops[0][1].id == "m1"
    _store_with_consolidation([MemoryEntry(content="Me mudei para o Rio")], store, LLMConsolidator(llm))
    (only,) = store.list_all()
    assert only.id == "m1" and only.content == "Mora no Rio de Janeiro"
    assert only.content_hash == _compute_content_hash(only.content)   # o bug do 3c88e66
    assert "m1" in llm.prompts[-1] and "São Paulo" in llm.prompts[-1]  # o alvo estava no prompt
```

Mesma forma para: "não gosto de café → passei a gostar" (UPDATE), paráfrase (`op: "none"` → 1 entry), DELETE (`op: "delete"` → `store.list_all() == []` depois do ramo novo em `_store_with_consolidation`), JSON quebrado / provider `Boom` → fail-soft (todas viram ADD + `caplog` warning), e `existing == []` → nem chama o LLM (`llm.prompts == []`).

Live: `tests/live/test_memory_llm_live.py` com o mesmo gate `_CLAUDE_CLI`, `ClaudeCLIProvider(model="sonnet", max_response_tokens=300)`, o cenário SP→Rio real, asserts frouxos (`len(store.list_all()) == 1`, `"rio" in content.lower()`).

---

## 6. Interações com a #4 (grafo)

### 6.1 `GraphIndexingEntryStore` (`ingestion/graph_extractors.py:375-430`)

```python
def add(self, entry):                                             # :399-405
    current = getattr(self._inner, "get", lambda _id: None)(entry.id)
    self._inner.add(entry)
    if current is not None and current.content == entry.content:
        return                     # :402-403  mesmo texto → NÃO re-extrai
    self._indexer.graph.unlink_item(entry.id)
    self._indexer.index_entries([entry])
def delete(self, entry_id):                                       # :407-411
    deleted = self._inner.delete(entry_id)
    if deleted: self._indexer.graph.unlink_item(entry_id)
    return deleted
def clear(self):                                                  # :413-418  usa list_all_unfiltered
```

| Operação da #5 | O que acontece no grafo |
|---|---|
| ADD (`store.add` id novo) | `current is None` → `index_entries([entry])` → `GraphIndexer.index` (`:478-513`) liga nós/arestas com `evidence=(entry.id,)`. |
| UPDATE (`store.add` mesmo id, conteúdo novo) | `unlink_item` (arestas sem evidência ficam **invalidadas, não apagadas** — `:446-448`, `graph/knowledge_graph.py:198-200`) + re-extração. Testado em `tests/test_memory/test_manager_graph.py:44-47,86-89`. |
| DELETE hard (`store.delete`) | `unlink_item` imediato (`:407-411`), testado `:49-51`. |
| DELETE soft (`store.add(expires_at=now)`, conteúdo igual) | cai no early-return `:402-403` → **o grafo mantém a evidência de uma entry expirada**. `index_entries`/`index` (`:473-513`) e `entry_to_item` (`:357-372`) nunca olham `is_expired`. Confirmado pelo teste `tests/test_memory/test_manager_graph.py:97-104`: entry adicionada já expirada → `graph.items("dave") == ["e2"]`; só o GC limpa (`:104-106`). |

Se DELETE virar soft, o decorador precisa de 2 linhas em `add`: `if entry.is_expired: self._inner.add(entry); self._indexer.graph.unlink_item(entry.id); return`. Se DELETE for hard, nada muda no grafo.

Argumento a favor de hard-delete na entry: a preservação de histórico que o `mem0^g` quer **já existe no nível das relações** — `unlink_item` invalida arestas com bitemporalidade (`CHANGELOG.md:11` "Obsolete edges are invalidated, never deleted"). O fato em si pode sumir.

### 6.2 `MemoryGarbageCollector` (`memory/gc.py`)

`collect_expired` (`:147-173`): `all_entries = store.list_all_unfiltered()` (`:161`) → `expired = [e for e in all_entries if e.is_expired]` → `self._store.delete(entry.id)` (`:164-166`) → `on_expiry_prune` (`:168-171`). Como o wrapper de grafo expõe `list_all_unfiltered` por `__getattr__` (`graph_extractors.py:426-427`) e intercepta `delete`, o caminho soft-delete → GC → `unlink_item` **está fechado** (teste `test_manager_graph.py:97-106`). O GC não é chamado por ninguém automaticamente (`grep -rn "MemoryGarbageCollector(" src/anchor/` → só o exemplo do docstring em `gc.py:82`) — é o app quem agenda.

### 6.3 Idioma de injeção

`docs/docs/guides/knowledge-graph.md:103-108` já ensina `MemoryManager(persistent_store=..., graph=GraphIndexer(graph, extractors=[LLMGraphExtractor(agent.llm)]))`. A #5 deve espelhar: `MemoryManager(persistent_store=..., extractor=LLMExtractor(agent.llm), consolidator=LLMConsolidator(agent.llm))`. Ambas as fábricas de LLM opt-in então convivem no mesmo ctor.

---

## 7. `evaluation/`

`evaluation/__init__.py:1-46` exporta: `ABTestRunner/ABTestResult/AggregatedMetrics/EvaluationDataset/EvaluationSample` (`ab_testing.py:30-80`), `BatchEvaluator`, `PipelineEvaluator` (`evaluator.py:16-120` — retrieval + RAG), `GoldenCase/GoldenCaseResult/GoldenSetReport/load_golden_set/evaluate_retriever/assert_metric_floor` (`golden.py:29-164`), `HumanEvaluationCollector/HumanJudgment`, `EvaluationResult/RAGMetrics/RetrievalMetrics` (`models.py:10-68`), `LLMRAGEvaluator` (`rag.py`), `RetrievalMetricsCalculator` (`retrieval.py:37-67`). Protocolos em `protocols/evaluation.py:16-93` (`HumanEvaluator`, `RAGEvaluator`, `RetrievalEvaluator`).

Tudo é **retriever-cêntrico**: `GoldenCase(query, relevant, name)` (`golden.py:29-40`), `evaluate_retriever(retriever, cases, k)` chama `retriever.retrieve(QueryBundle, top_k)` (`:129-131`) e mede precision/recall/MRR/NDCG. Nada mede memória.

**Mínimo para um golden set de consolidação** (YAGNI-compatível): um JSONL `{"existing": ["Mora em São Paulo"], "new": "Me mudei para o Rio", "expected": "update", "name": "..."}` e um teste que instancia o consolidador (fake ou live), roda cada caso e afere `acurácia_por_op ≥ floor` com `assert` simples — sem módulo novo. Se quiser reuso: `evaluation/memory.py` com `ConsolidationCase(BaseModel)` + `evaluate_consolidator(consolidator, cases) -> dict[str, float]` (~50 linhas espelhando `golden.py:81-164`, incluindo um `assert_metric_floor`-like). Sugiro adiar até a primeira necessidade real de CI-gate.

---

## 8. Exports e docs

### 8.1 Exports

- `memory/__init__.py:1-33`: importa `SimilarityConsolidator` (`:5`) e `CallbackExtractor` (`:8`); `__all__` `:15-33`. Adicionar `LLMConsolidator`, `LLMExtractor`.
- `anchor/__init__.py`: bloco docstring "Memory Management" `:40-49` (lista `CallbackExtractor`, `SimilarityConsolidator`); import `from anchor.memory import (...)` `:251-270`; `__all__` com `"CallbackExtractor"` `:467` e `"SimilarityConsolidator"` `:632` (lista ordenada alfabeticamente).
- `protocols/__init__.py:8-22,41-70` já exporta os protocolos; nada a mudar se a assinatura não muda.
- `tests/test_exports.py:148-152` é o smoke de exports de memória.

### 8.2 Docs (mkdocs em `docs/mkdocs.yml`; fontes em `docs/docs/`)

| Página | Nav | O que diz hoje sobre consolidator/extractor |
|---|---|---|
| `docs/docs/guides/memory.md` | `docs/mkdocs.yml:81` | `## SimilarityConsolidator` `:233-264` (exemplo + "!!! warning The library never calls an LLM" `:261-263` — **precisa de ressalva** quando o LLM entrar); `## MemoryCallback and CallbackExtractor` `:320-350`; seção `## Knowledge graph` `:143`. Novo `## LLMExtractor and LLMConsolidator` cabe entre `:264` e `:266`. |
| `docs/docs/api/memory.md` | `:108` | `## SimilarityConsolidator` `:256-282` (tabela diz "ADD / UPDATE / NONE" `:276`); `## MemoryCallback` `:328-350` (`:347` lista delete); `## CallbackExtractor` `:411-450`; `## MemoryManager` `:8`. |
| `docs/docs/api/protocols.md` | — | `MemoryExtractor` `:250-263`, `AsyncMemoryExtractor` `:265-271`, `MemoryConsolidator` `:274-286` ("Operations are ADD, UPDATE, DELETE, or NONE" — sem semântica do slot `entry`). |
| `docs/docs/api/pipeline.md` | — | `auto_promotion_step` `:318-329`, `create_eviction_promoter` `:349-370` (só "deduplication"; nada sobre DELETE). `docs/docs/guides/pipeline.md:107-108` lista os steps. |
| `docs/docs/guides/knowledge-graph.md:95-108` | — | precedente do `MemoryManager(graph=...)` com `agent.llm`. |
| `docs/docs/llms.txt:99-100` | — | **drift**: `MemoryExtractor -- extract(turn) -> list[MemoryEntry]` e `MemoryConsolidator -- consolidate(entries) -> list[MemoryEntry]` (assinaturas erradas). |
| `docs/docs/cookbook/chatbot-with-memory.md` | `:97` | não menciona consolidação/extração (grep vazio). |

### 8.3 CHANGELOG (`CHANGELOG.md`)

`## [Unreleased]` `:8`; `### Breaking` `:10`; `### Added` `:27` (as entradas da #4 estão no topo, `:28-31`, no formato "**Título (roadmap #N, fase X, opt-in)**: …"); `### Changed` `:97`; `### Fixed` `:127`. A #5 entra em Added (feature) e em Fixed ("consolidation step ignored `DELETE`").

---

## 9. Riscos e furos encontrados (pré-existentes que a #5 vai esbarrar)

1. **DELETE inaplicável pelo protocolo** — `(DELETE, None)` não carrega alvo (`protocols/memory.py:137`) e o único aplicador o ignora (`pipeline/memory_steps.py:44-46`). Sem teste. (§1.4)
2. **UPDATE só é UPDATE pelo `id`** — `store.add` sobrescreve por id em todos os backends; entry nova com id novo vira ADD silencioso. E `model_copy` não recomputa `content_hash` (`models/memory.py:66-71`; o bug do `3c88e66`). O `LLMConsolidator` precisa de `existing.model_copy(update={"content", "content_hash", "updated_at"})` como `manager.update_fact` (`manager.py:245-252`). (§1.3)
3. **`on_extraction`/`on_consolidation` são eventos mortos** — nunca disparados (`callbacks.py:60-84`; grep). Se a #5 quiser observabilidade, é a primeira a disparar; senão, documentar que continuam mortos.
4. **`existing = store.list_all()` inteiro** (`memory_steps.py:42`) — para o LLM é O(n) tokens por turno. Precisa de S-selection (cap ou `embed_fn`). (§3)
5. **Busca é substring em todo backend; nenhum store de memória com embedding** (§3). "S similares" via `search(fato)` é fraca; `ScoredMemoryRetriever` exige `vector_store` que ninguém alimenta (`retrieval/memory_retriever.py:109-116`).
6. **Memória síncrona no caminho async (follow-up 13/5)** — `_finish_turn` está num `finally` compartilhado (`agent.py:2178-2191,2346,2438`); colocar LLM ali bloqueia o loop e roda em abandono. Encaixe correto: após o `finally`, com `asyncio.to_thread` no async. (§2.3-2.4)
7. **Turnos `tool` na conversa** (`agent.py:1606-1616`) entram em `conversation.turns` — extrator deve filtrar por role.
8. **Grafo mantém evidência de entry expirada** (`graph_extractors.py:402-403,473-513`; teste `test_manager_graph.py:97-104`). Só afeta a #5 se DELETE for soft. (§6.1)
9. **`claude_cli` não garante `tool_choice`** (`claude_cli.py:554-559`) — output estruturado por tool call pode falhar no live; JSON-no-content é o caminho seguro. `FakeLLMProvider` do agente não implementa `invoke` (`tests/test_agent/test_agent.py:128-135`). (§4.1, §5.1)
10. **`add_fact` dedupe é O(n) por chamada** (`manager.py:182-186`). O consolidador não passa por `add_fact` (escreve via `store.add`), então o hash-gate (NONE barato antes do LLM) tem de viver no próprio consolidador, como em `SimilarityConsolidator.consolidate:115,121-124`.
11. **`get_context_items` injeta todos os fatos sem top_k** (`manager.py:267-285`) — a consolidação reduz o crescimento, mas não há corte; não é da #5, mas é o porquê dela.
12. **Read-modify-write não atômico** em `_store_with_consolidation` (`list_all` → `consolidate` → `add`); os stores só travam por operação (`storage/_base.py:67,72`). Seguro hoje porque `Agent` roda um turno por vez (`agent.py:2160-2166`); dois agentes no mesmo store podem entrelaçar. Não corrigir agora; registrar `ponytail:`.
13. **Docs drift** — `docs/docs/llms.txt:99-100` assinaturas erradas; `api/memory.md:276` e `guides/memory.md:261-263` ("never calls an LLM") ficarão falsas para o novo consolidador.
14. **`existing` no docstring = "All entries"** (`protocols/memory.py:132`) — passar S similares contraria a letra; ajustar o texto junto com o DELETE.

---

## 10. Menor encaixe possível

Zero arquivos novos em `src/`; um arquivo novo de teste unitário e um live.

| # | Arquivo | Mudança |
|---|---|---|
| 1 | `src/anchor/protocols/memory.py:132-137` | Docstring: `existing` = candidatos (todos ou subconjunto); `DELETE` carrega a entry existente a remover; `NONE` → `None`. Sem mudar assinatura. Não criar `AsyncMemoryConsolidator`. |
| 2 | `src/anchor/pipeline/memory_steps.py:44-46` | `elif action is MemoryOperation.DELETE and entry is not None: store.delete(entry.id)` (hard delete — o grafo já invalida arestas com histórico; soft-delete fica como upgrade documentado com `ponytail:`). Atualizar docstring `:36-37`. |
| 3 | `src/anchor/memory/extractor.py` (ao lado de `CallbackExtractor`) | `class LLMExtractor(llm, *, roles=("user","assistant"), max_turns=M)`: prompt → `invoke` → `json.loads(strip_markdown_fences(...))` → lista de `{"content", "tags"?, "memory_type"?}` → `MemoryEntry(..., source_turns=[t.timestamp.isoformat()])` (reusa a convenção de `CallbackExtractor.extract:77-87`). Fail-soft: warning + `[]`. Reuso: `_text.strip_markdown_fences`, `llm.models.Message/Role`. |
| 4 | `src/anchor/memory/consolidator.py` (ao lado de `SimilarityConsolidator`) | `class LLMConsolidator(llm, *, embed_fn=None, top_k=S, max_candidates=N)`: (a) hash-gate → `(NONE, None)` (copiar `:115,121-124`); (b) S candidatos: cosseno via `_cosine_similarity` se `embed_fn`, senão os N mais recentes por `updated_at`; (c) `existing` vazio → tudo ADD sem chamar LLM; (d) um `invoke` com memórias numeradas por `id` + fatos novos → JSON `[{"op","target","content"}]`; (e) mapear: ADD → `MemoryEntry` novo; UPDATE → `target.model_copy(update={content, content_hash, updated_at})`; DELETE → `(DELETE, target)`; NONE → `(NONE, None)`; alvo desconhecido → warning + ADD. Fail-soft global: warning + ADD para todos (comportamento sem consolidador, `memory_steps.py:47-49`). |
| 5 | `src/anchor/memory/manager.py` | `__slots__` `:55` + ctor `:57-82`: `extractor: MemoryExtractor | None = None`, `consolidator: MemoryConsolidator | None = None`, `extract_window: int = 6`. Novo `remember(self) -> list[tuple[MemoryOperation, MemoryEntry | None]]`: no-op se sem extractor/store; `turns = self._conversation.turns[-window:]`; `entries = extractor.extract(turns)`; import lazy de `_store_with_consolidation` (precedente `:68`) e aplica; devolve as ops (útil para teste/observabilidade). Sync. |
| 6 | `src/anchor/agent/agent.py:2347` e `:2439` | Após o `finally`, antes de `yield TurnFinished`: sync `if self._memory is not None: self._memory.remember()`; async `await asyncio.to_thread(self._memory.remember)`. Duas linhas cada; não toca `_finish_turn`. |
| 7 | `src/anchor/memory/__init__.py:5,8,15-33`, `src/anchor/__init__.py:40-49,251-270,467,632` | Exports `LLMConsolidator`, `LLMExtractor`. `tests/test_exports.py:148-152` smoke. |
| 8 | `tests/test_memory/test_llm_consolidator.py` (novo) | Casos de §5.4: SP→Rio UPDATE (id preservado, hash recomputado), café UPDATE, paráfrase NONE, DELETE aplicado via `_store_with_consolidation`, fail-soft (JSON ruim + `Boom`), `existing == []` não chama LLM, hash-gate não chama LLM, extractor filtra `tool`. `FakeLLM` copiado de `tests/test_ingestion/test_llm_graph_extractor.py:54-61`. |
| 9 | `tests/test_pipeline/test_memory_steps.py` (perto de `:442-517`) | Um teste: consolidator inline devolve `("delete", existing)` → `store.list_all() == []`; `("delete", None)` → no-op. |
| 10 | `tests/test_memory/test_manager_persistent.py` ou `test_manager_unified.py` | `remember()` com extractor+consolidator fake e `tests/test_memory/test_manager_graph.py`: UPDATE/DELETE via `remember()` mantêm o grafo (reusa `WordPairExtractor` `:15-24`). |
| 11 | `tests/test_agent/test_agent.py` | Um teste sync + um async: após `stream/astream`, `remember` foi chamado uma vez (manager com extractor stub que grava chamadas). |
| 12 | `tests/live/test_memory_llm_live.py` (novo) | Gate `_CLAUDE_CLI` (`tests/live/test_graph_llm_live.py:17-20`), SP→Rio real. |
| 13 | Docs | `docs/docs/guides/memory.md` nova seção após `:264` + ressalva no warning `:261-263`; `docs/docs/api/memory.md` `LLMConsolidator`/`LLMExtractor` após `:282`/`:450` e `MemoryManager` (`:8`) ganha os parâmetros; `docs/docs/api/protocols.md:284-286` semântica do slot `entry`; `docs/docs/api/pipeline.md:318-370` DELETE aplicado; `docs/docs/llms.txt:99-100` assinaturas corrigidas; `CHANGELOG.md:27` Added + `:127` Fixed. |

Fora, deliberadamente: `AsyncMemoryConsolidator` (o `to_thread` cobre), gate de novidade além do hash (o cap/`embed_fn` já é o gate barato; medir custo antes de inventar), invalidação temporal de entries (hard delete; arestas já são bitemporais), `evaluation/memory.py` (um teste com JSONL basta até haver CI-gate), callbacks `on_extraction/on_consolidation` (continuam mortos; disparar é 2 linhas em `remember()` se Arthur quiser — decidir no plano).
