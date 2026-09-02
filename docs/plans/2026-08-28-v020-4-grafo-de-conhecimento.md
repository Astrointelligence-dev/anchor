# v0.2 · #4 — Grafo de conhecimento sobre memória e documentos

**Status:** em execução (sessão 11, 2026-09-02: pesquisa + 4 decisões fechadas) · **Tamanho:** grande · **Depende de:** #3 (vault/namespace) ✅
**Sessão:** abrir com pesquisa, depois implementar. Provavelmente mais de uma sessão.

---

## O que é

O modelo do `graphify` — god nodes, comunidades, `query`/`path`/`explain`,
navegação estilo vault — como primitiva nativa do anchor, sobre **memória e
documentos ao mesmo tempo**.

Vários documentos alimentam o mesmo nó. Um assunto central emerge dos tópicos
que o cercam. Você navega em vez de só buscar.

---

## Contexto (o que já existe — mais do que parece)

| Peça | Onde | Estado |
|---|---|---|
| `SimpleGraphMemory` | `memory/graph_memory.py:14` | entidades + triplas + `link_memory` + BFS até profundidade N. **Só em memória.** |
| `graph_retrieval_step` | `pipeline/memory_steps.py:51` | query → extrator → BFS → `MemoryEntry` → `ContextItem`. Já plugado. |
| `ParentChildChunker` / `ParentExpander` | `ingestion/hierarchical.py` | hierarquia de documento em 2 níveis |
| `MemoryRetrieverAdapter` | `retrieval/memory_retriever.py` | memória → `ContextItem` |
| **Branch `feature/storage-layer-gaps`** | `.worktrees/storage-layer-gaps` | **não mergeada** — protocolos `GraphStore`/`ConversationStore`, backends SQLite e Postgres. É a persistência do grafo, meio construída e parada. |

**Primeiro passo da sessão: revisar e mergear esse branch.** É a regra da
escada — está no repo, termina antes de escrever coisa nova.

> **Sessão 11:** revisado — parecer é **descartar e reescrever** usando o
> branch como referência (ver § Pesquisa abaixo). A regra da escada vale para
> o que está no repo *e serve*; este schema contradiz as decisões de agosto.

---

## Pesquisa a fazer

1. **Qual arquitetura.** Os números de 2026:
   - Microsoft GraphRAG (Leiden hierárquico): 86% vs 32% do RAG baseline
   - LightRAG: indexação dual-level, **update incremental em tempo real**
   - PathRAG: poda por fluxo, −44% de contexto mantendo acurácia
   - HippoRAG 2: multi-hop 10–30x mais barato

   PathRAG casa direto com o `TokenBudget` — vale ler antes de escolher.
2. **Extração.** Três fontes com custos muito diferentes: wikilinks `[[...]]`
   explícitos (grátis, determinístico, formato Obsidian), extração LLM
   (cara, indispensável em doc sem link), similaridade de embedding (barata,
   ruidosa). Qual combinação, e em que ordem?
3. **Proveniência.** O `graphify` marca cada aresta `EXTRACTED` / `INFERRED` /
   `AMBIGUOUS`. Sem isso não se distingue "o documento diz" de "o modelo achou",
   e grafo LLM sem proveniência apodrece. Confirmar que vale copiar.
4. **Incremental.** LightRAG é a referência. O caro não é adicionar nó — é
   **invalidar comunidade** quando o grafo muda. Como fazer sem recomputar tudo?
5. **Comunidade.** Leiden/Louvain exige `networkx` — dependência nova. O core
   tem 2. Extra opcional `[graph]`, ou implementação mínima própria?

---

## Decisões já tomadas

- **`ContextItem` é a moeda única.** O grafo liga entidade → `item_id`, não
  → `memory_id`. Se o item veio da memória ou de um PDF já está no
  `SourceType`. Uma mudança em `SimpleGraphMemory.link_memory` torna o grafo
  universal — é o que unifica memória e documentos.
- **Namespace pertence ao item, não ao grafo** (ver #3). O nó é um só,
  alimentado por várias fontes de vários namespaces.
- **Hub é calculado, não declarado.** Centralidade de grau + comunidade. O
  usuário não marca o que é central.
- **Exclusão é em nível de nó, não de evidência.** Nó sem nenhuma evidência
  sobrevivente ao escopo **não existe** para aquele escopo — não aparece em
  travessia, nem em `backlinks`, nem em `path`. Filtrar só o conteúdo vaza: a
  existência do nó já é informação.
- **Nó excluído é parede, não ponte.** `path(A, C)` não atravessa por cima de
  um `B` excluído; senão a topologia denuncia o que foi escondido.
- **Aresta tem validade temporal.** Do `mem0^g`: relação obsoleta é **marcada
  inválida, não deletada**, preservando o contexto histórico.
- Detecção de comunidade é extra opcional, não core.

## Decisões em aberto

Fechadas pelo Arthur em 2026-09-02 (sessão 11), todas na recomendação da pesquisa:

- [x] **Aresta tem tipo** — `relation: str` **rótulo livre normalizado**
      (`casefold`, espaços → `_`), sem enum; `fact` opcional; 5–8 rótulos
      sugeridos no prompt do extrator, não validados. Wikilink → `links_to`,
      estrutura → `contains`, embedding → `similar_to`, LLM → predicado. A
      premissa "dobra o custo" era falsa: o predicado sai na mesma chamada.
- [x] **Incremental já na v0.2** — upsert por chave natural `(vault, key)`;
      aresta com `evidence` (item_ids) + `invalidated_at`; remover/reindexar
      item = invalidar arestas cuja única evidência era ele. Derivados (grau,
      PPR, comunidade) **nunca persistidos**: recompute lazy cacheado por
      `graph_version`. Sem tabela `communities`. Full-rebuild = `clear()` +
      reindex.
- [x] **`GraphRetriever` de primeira classe** implementando `Retriever` —
      ganha `effective_scope` (`pipeline/step.py:101`), RRF no
      `HybridRetriever` e `ABTestRunner` de graça. `graph_retrieval_step` fica
      exportado como legado.
- [x] **Worktree `feature/storage-layer-gaps`: descartar e reescrever**,
      branch mantido como referência (forma de 3 tabelas, CTE recursivo do
      Postgres, cenários de teste). `ConversationStore` e cache backends ficam
      fora. Worktree removido quando o novo `GraphStore` entrar na dev.
- [x] **Dependência: core próprio + extra `[graph]` = igraph** (PPR, grau,
      Louvain/label-propagation próprios ≤150 linhas; igraph para Leiden/escala).
- [x] Export Obsidian: fora da v0.2 (mantido).
- [x] **Escopo do grafo = `RetrievalScope`** (reuso; o `GraphScope` do sketch
      da sessão 5 não vira tipo novo). `SubagentDefinition.scope` já cobre o
      item "`graph_scope` no `SubagentDefinition`".

---

## Escopo

- [x] ~~Mergear `feature/storage-layer-gaps`~~ → revisado; reescrever usando-o como referência (decisão sessão 11)
- [ ] `SimpleGraphMemory` → grafo sobre `ContextItem`, com vault/namespace
- [ ] Arestas com proveniência e validade temporal
- [ ] Extração: wikilinks primeiro, extrator LLM depois
- [ ] Detecção de comunidade + hubs (extra `[graph]`)
- [ ] API de navegação: `query` (BFS com orçamento), `path`, `explain`, `backlinks`
- [ ] `GraphScope` aplicado em toda travessia (de #3)
- [x] ~~`graph_scope` no `SubagentDefinition`~~ → coberto por `SubagentDefinition.scope` (mesmo `RetrievalScope`)

**Fora:** export Obsidian (a decidir), Neo4j/FalkorDB, visualização.

## Verificação — a regra que evita o desperdício

**Construir o caminho de consulta primeiro, contra um grafo pequeno feito à
mão, e medir contra o híbrido+RRF atual num golden set — antes de investir no
construtor caro.**

O anchor já tem o antídoto pronto: `RetrievalEvaluator`, `RAGMetrics`, golden
sets, `ABTestRunner`. O modo clássico de falhar aqui é entregar um grafo lindo
que ninguém consulta e um retrieval que não melhorou. Se o grafo não ganhar do
RRF, isso se descobre em uma sessão em vez de em seis.

- [ ] Golden set montado **antes** do construtor
- [ ] A/B grafo vs híbrido+RRF, com número publicado no Review
- [ ] Teste de vazamento: nó excluído invisível por qualquer caminho

## Pesquisa — sessão 11 (2026-09-02)

Doc completo: `docs/research/2026-09-02-grafo-de-conhecimento.md` — 4 agentes
(arquitetura/incremental; aresta/extração com o código-fonte do Graphiti, mem0,
LightRAG, HippoRAG, GraphRAG e graphify lido; mapa do código com `file:line`;
parecer do worktree).

**Os números da seção "Pesquisa a fazer" estavam errados.** "86% vs 32%" é o
RobustQA da Writer, não o GraphRAG (que só reporta win rates de LLM-juiz);
PathRAG é −13,7% (não −44%; −40% é a variante `-lt`); "10–30×" é HippoRAG 1
vs IRCoT, só custo online. GraphRAG-Bench (ICLR'26): RAG vence em fato simples,
grafo em multi-hop, e **RRF(híbrido, grafo) +6,4%** — o A/B precisa de 3
condições.

**Respostas às 5 perguntas:**

1. *Arquitetura* — grafo `{entidade, item}` com `ContextItem` como nó
   (HippoRAG 2); consulta = sementes → PPR (~30 linhas) ou espalhamento com
   decaimento (o *pruning* do PathRAG, que é BFS com orçamento) → score por
   item → `TokenBudget`. Zero LLM na consulta. Global search / community
   reports **não** (57× tempo, 210× tokens).
2. *Extração* — wikilinks sempre → estrutura (`parent_id` do
   `ParentChildChunker`, `doc_id`) sempre → LLM opt-in (default ligado só para
   memória; `max_gleanings=0`) → embedding só entre nomes de entidade, ≥0.8,
   `inferred`. Não existe parser de wikilink no repo.
3. *Proveniência* — confirmado copiar: `provenance` (3 tags) + `confidence`
   (rubrica discreta) + `evidence: list[item_id]`. `evidence` é o mecanismo do
   escopo E do incremental, não só auditoria.
4. *Incremental* — o caro é comunidade **com resumo LLM**; sem LLM, Louvain em
   10k nós custa 0,5–2 s. Derivados não se persistem: recompute lazy cacheado
   por `graph_version`. `graphrag update` real não re-clusteriza.
5. *Comunidade* — core sem lib (PPR + grau + Louvain/label-propagation
   próprios, ≤150 linhas); extra `[graph]` = igraph (2 MB, 1 dep, Leiden em
   C). `networkx.leiden_communities` é backend-only; graspologic tem 17 deps.
   Hub = grau (16/20 e 20/20 de sobreposição com PageRank).

**Worktree `feature/storage-layer-gaps`:** 64 commits atrás, 6 conflitos, testes
rodam (58 ok, Postgres todo pulado). Schema sem vault, sem validade/proveniência,
`UNIQUE(source,relation,target)` proíbe reafirmar aresta invalidada, travessias
sem predicado de evidência, liga `memory_id`. 70–80% do código seria reescrito.
Parecer: **descartar e reescrever**; copiar como referência a forma de 3
tabelas, o CTE recursivo do Postgres e os cenários de teste. `ConversationStore`
e cache backends ficam fora.

**Proposta para as decisões em aberto** (schema mínimo de aresta no doc de pesquisa):

- Tipo de aresta → **rótulo livre normalizado, sem enum**; `fact` opcional;
  vocabulário sugerido no prompt, não validado. O predicado sai na mesma
  chamada da tripla — não dobra o custo.
- Incremental → **sim para nós/arestas/evidência** (upsert por chave natural +
  `invalidated_at`); derivados recomputados. Com chaves naturais, full-rebuild
  é `clear()` + reindex — os dois coexistem.
- `GraphRetriever` de primeira classe → **sim**: ganha escopo
  (`step.py:101`), RRF (`hybrid.py:70`) e `ABTestRunner` de graça;
  `graph_retrieval_step` vira legado.
- Export Obsidian → fora da v0.2 (mantido).
- Validade → bi-temporal completo (`valid_from/valid_to/created_at/
  invalidated_at`), preenchimento assimétrico, `as_of` em toda travessia.

## Plano de execução (sessão 11 →)

Uma fase por commit, teste falhando antes de cada uma, ritual xhigh ao final
(padrão das frentes #1–#3). **A regra de verificação manda a ordem:** caminho
de consulta e A/B contra um grafo pequeno e determinístico *antes* do
construtor caro (LLM). O gate depois da fase B decide se a fase D (extrator
LLM) vale o investimento; C e E têm valor próprio (navegação/workspace) e
seguem de qualquer jeito.

### A. Modelo + store in-memory + navegação (sem persistência, sem LLM)

- [x] `models/graph.py`: `GraphNode(id, label, aliases, metadata)` com chave
      canônica (NFKC + casefold + strip; `normalize_node_id`), `GraphEdge`
      (schema mínimo do doc de pesquisa: `relation`, `fact`, `provenance`,
      `confidence`, `evidence ≥1`, `valid_from/valid_to/created_at/
      invalidated_at`, `metadata`), `Provenance` Literal, `is_visible(edge,
      as_of)` — um predicado, um lugar.
- [x] `protocols/storage.py`: `GraphStore` / `AsyncGraphStore`, vault na
      montagem (`validate_vault`, padrão `SqliteContextStore:20`).
      Superfície: `upsert_node`, `get_node`, `add_edge` (merge de evidência
      se a aresta viva já existe), `invalidate_edge`, `link_item(node_id,
      item_id, namespace)`, `unlink_item(item_id)` (invalida arestas sem
      evidência restante), `neighbors(node_id, *, scope, as_of, relation)`,
      `subgraph(scope, as_of)` (nós + arestas visíveis — o que PPR/comunidade
      consomem), `graph_version`, `clear`.
- [x] `storage/memory_store.py`: `InMemoryGraphStore`. Visibilidade por
      escopo = evidência ∩ itens visíveis ≠ ∅ (`RetrievalScope.matches`),
      aplicada ao nó **antes** de qualquer travessia (parede, não ponte).
- [x] `graph/algorithms.py` (Python puro): `personalized_pagerank` (power
      iteration), `bfs_budgeted` (o pruning do PathRAG: recurso decai por
      salto, poda abaixo de θ), `shortest_path`, `degree`. Sem lib.
- [x] `graph/knowledge_graph.py`: `KnowledgeGraph(store, context_store)` —
      `query(seeds, *, scope, as_of, top_k)` → itens ranqueados, `path(a,
      b)`, `explain(a, b)` (arestas + `fact` + `provenance` + evidência),
      `backlinks(node)`, `hubs(k)` (grau). `SimpleGraphMemory` vira fachada
      fina sobre `InMemoryGraphStore` com `link_item` (breaking:
      `link_memory` sai; CHANGELOG).
- [x] Testes com decoy: nó excluído invisível por travessia, `backlinks`,
      `path` (não atravessa por cima) e `explain`; `as_of` esconde aresta
      invalidada; subagente com escopo ∩ não alarga.

**Fase A entregue** (sessão 11). Desvios: `AsyncGraphStore` fica para a
fase C, com o primeiro backend async (protocolo com uma implementação é
YAGNI); `bfs_budgeted` (pruning do PathRAG) descartado — o PPR é a versão
iterada do mesmo espalhamento e um único spreader basta; `SimpleGraphMemory`
**deletado**, não fachada — `KnowledgeGraph()` sem store já é o in-memory,
duas classes para um conceito seria o "misrouted" da arquitetura de memória;
`evidence` é opcional no modelo (aresta manual sem item existe só sem escopo;
o indexador da fase B sempre cita ≥1). Suíte 2986 verdes; ruff 151 (baseline
154), mypy 140 = baseline.

### B. Golden set + extração determinística + `GraphRetriever` + A/B (o gate)

- [x] Corpus versionado em `tests/fixtures/graph_corpus/`: ~40 notas markdown
      com wikilinks (3 estratos: fato simples / multi-hop / síntese-explicação)
      + golden JSONL (`evaluation/golden.py` formato). Feito à mão,
      determinístico.
- [x] `ingestion/graph_extractors.py`: `WikilinkExtractor` (gramática Obsidian
      completa: `[[note]]`, `|alias`, `#heading`, `#^block`, `![[embed]]`;
      resolução NFC + casefold, basename vault-wide, ambíguo → `ambiguous`)
      e `StructureExtractor` (`parent_id` → `contains`, `doc_id` →
      `belongs_to`). `MarkdownParser` passa a expor `metadata["wikilinks"]`.
      `GraphIndexer(store, extractors)` idempotente (reindexar = mesma
      evidência, zero duplicata).
- [x] `retrieval/graph.py`: `GraphRetriever(Retriever)` — sementes =
      `entity_extractor(query)` ∪ (opcional) top-k do `VectorStore` → nós;
      PPR sobre `subgraph(effective_scope)` → score por item →
      `ContextStore.get` → `ContextItem` com **id canônico** (funde no RRF).
- [x] `evaluation`: `evaluate_retriever` e `ABTestRunner.run` ganham `scope`
      (buraco achado na pesquisa).
- [x] `tests/test_retrieval/test_graph_ab_benchmark.py` no molde do
      `test_scope_recall_benchmark.py`: 3 condições — híbrido+RRF, grafo só,
      RRF(híbrido, grafo) — por estrato, número publicado no Review.
      **Gate:** grafo (ou a fusão) ganha em multi-hop/explain → D vale;
      senão D vira opt-in mínimo e o Review diz por quê.

**Fase B entregue** (sessão 11, `17748f7`). Números do A/B (recall@5 / MRR,
30 queries, 40 notas, híbrido = BM25 + bag-of-words hasheado como "denso" —
sem modelo no CI):

| Condição | fato | multi-hop | explain | **geral** |
|---|---|---|---|---|
| híbrido+RRF | 1.00 / 0.94 | 0.96 / 0.72 | 0.78 / 0.92 | **0.94 / 0.85** |
| grafo só | 0.92 / 0.64 | 0.83 / 0.68 | 0.75 / 0.66 | 0.85 / 0.66 |
| RRF(híbrido, grafo) | 1.00 / 0.80 | 0.96 / 0.66 | 0.78 / 0.72 | 0.94 / 0.73 |
| RRF 2:1 | 1.00 / 0.84 | 0.96 / 0.71 | 0.78 / 0.75 | 0.94 / 0.77 |

**Resultado do gate: o grafo de wikilinks NÃO ganha do híbrido+RRF em
retrieval neste corpus.** O híbrido já está no teto (recall 0.94; k=5 sobre
40 notas de um chunk, queries com vocabulário do alvo); a fusão empata em
recall e perde MRR porque o PPR põe o chunk da própria semente em primeiro
(certo para fato, errado para multi-hop, onde a query nomeia a origem e não
o alvo). Os asserts do benchmark são **pisos de regressão**, não
reivindicação de vitória. Consequência para a fase D, como o plano previa:
extrator LLM vira **opt-in mínimo**; o valor do grafo nesta v0.2 é
navegação (path/explain/backlinks/escopo) e memória sem wikilink, não
substituir o retrieval híbrido. Ressalva honesta: o corpus é pequeno e
keyword-friendly — um corpus onde o alvo não compartilha termos com a
query discriminaria mais; não foi "ajustado" para o grafo ganhar.

Desvios: `MarkdownParser` não ganhou `metadata["wikilinks"]` (o extrator
lê o conteúdo do chunk, que é a evidência; a lista por documento era
informativa e YAGNI); `parent_id → contains` descartado (chunks são
evidência, não nós — a hierarquia pai/filho já é servida pelo
`ParentExpander`); ambiguidade de basename (duas notas com o mesmo nome em
pastas diferentes) fica como `ponytail:` no extrator (mergem num nó; índice
de notas quando um vault real tiver duplicatas); `GraphIndexer` é add-only
(reindexar reforça; remover é `unlink_item`) — consistente com o
`ContextStore`, que também não remove itens obsoletos sozinho.
`MemoryRetrieverAdapter` passou a devolver `id=MemoryEntry.id` (moeda
única). Suíte 3017 verdes; ruff/mypy no baseline.

### C. Persistência + CLI

- [x] `storage/sqlite/_graph_store.py` (`SqliteGraphStore` + twin async):
      `graph_nodes`, `graph_edges`, `graph_node_items`, `graph_edge_items`
      em `_TABLES` com PK `(vault, …)`, índice único parcial `WHERE
      invalidated_at IS NULL`, `_SCOPED_TABLES`; visibilidade por
      `EXISTS` + `scope_sql_clauses(scope, "namespace")` (`_where.py:156`).
      `subgraph` carrega o visível de uma vez (`ponytail:` teto ~50k arestas,
      paginação/CTE quando doer).
- [x] `storage/postgres/_graph_store.py`: mesmo schema, `COLLATE "C"`,
      integração com docker (`ANCHOR_TEST_POSTGRES_DSN`) + fake asyncpg.
- [x] Fixture `make_graph_store(vault)` em `tests/test_storage/conftest.py`;
      teste de isolamento entre vaults num só `.db`.
- [x] CLI: `anchor index --graph` (roda `GraphIndexer` com os extratores
      determinísticos) e `anchor graph query|path|explain|hubs|backlinks`.
- [x] Remover o worktree `.worktrees/storage-layer-gaps` (branch fica).

**Fase C entregue** (sessão 11). `SqliteGraphStore` (tabelas `graph_nodes`,
`graph_edges`, `graph_items`, `graph_node_items`, `graph_edge_items`,
`graph_meta`, PK `(vault, …)`, índice único parcial `WHERE invalidated_at IS
NULL`, `graph_items(vault, namespace)` para o range de escopo) e
`PostgresGraphStore` (mesmo schema, `COLLATE "C"`, `BIGSERIAL seq` no lugar
do rowid, JSONB) com a regra de visibilidade escrita **uma vez** em
`storage/_graph_sql.py` e aplicada em Python sobre linhas buscadas por
índice — o in-memory é a referência e os três backends respondem igual
(teste de contrato compara `subgraph`/`edges_of`/`node_items` sob 4 escopos ×
3 instantes). Postgres rodado de verdade (`supabase/postgres:17.6.1.084` via
docker, 7 testes: pgvector + grafo). CLI: `anchor index --graph` e `anchor
graph query|path|explain|hubs|backlinks` com `--vault/--include/--exclude`.
Worktree `.worktrees/storage-layer-gaps` removido (branch mantido como
referência).

Desvios: `AsyncSqliteGraphStore` delega ao store síncrono via
`asyncio.to_thread` (é o que o aiosqlite faz por baixo; conexões do manager
são thread-local) — `ponytail:` twin nativo quando o grafo estiver num
caminho async quente; `PostgresGraphStore.version` é contador in-process
(caches derivados vivem in-process; `graph_meta` quando outro processo
precisar ver); sem teste com fake asyncpg — a integração real substituiu
(fake que emula SQL seria um segundo motor); **`KnowledgeGraph` é síncrono
e o Postgres é async** — a API de navegação sobre Postgres precisa de um
`AsyncKnowledgeGraph` (follow-up; o store cumpre o protocolo e é testado
direto). Ordem de `edges_of` fixada no protocolo: saídas primeiro, depois
entradas, inserção dentro de cada grupo.

### D. Extrator LLM + resolução de entidade (opt-in; default ligado para memória)

- [x] `LLMGraphExtractor`: uma chamada por chunk, saída estruturada
      (entidades + triplas com `relation`, `fact`, `confidence` na rubrica),
      `max_gleanings=0`, vocabulário sugerido no prompt; provider injetado
      (o do agente/manager), nunca cliente novo.
- [x] Resolução: chave canônica + tabela de aliases (o `|alias` do wikilink
      entra de graça); `EmbeddingSynonymLinker` opt-in (`similar_to`,
      `inferred`, cosseno ≥ 0.8, só entre nomes de entidade).
- [ ] Descrição de entidade = fragmentos por `item_id`; resumo LLM só com ≥8
      (regra LightRAG).
- [x] `MemoryManager`: extrator ligado por default para `MemoryEntry` (não tem
      wikilink); invalidação por re-ingestão; passada de contradição opt-in.
- [x] Um caso live com `claude_cli`.

**Fase D entregue** (sessão 11) como **opt-in mínimo**, a consequência do
gate da fase B. `LLMGraphExtractor(llm)`: uma chamada por chunk, JSON com
entidades (nome, tipo, aliases) e relações (`relation` livre normalizada,
`fact`, `provenance` extracted/inferred, `confidence`; inválido →
`ambiguous` ≤ 0.3), vocabulário sugerido no prompt (`DEFAULT_RELATIONS`),
`max_gleanings=0`, fail-soft com warning (contrato do `TierCompactor`).
Provider injetado, nunca cliente novo. `GraphIndexer.index_entries` põe
memória no grafo com `item id = MemoryEntry.id`; `MemoryManager(graph=
GraphIndexer(...))` mantém o grafo em passo: `add_fact` indexa,
`update_fact` = `unlink_item` + reindexa (aresta cuja única evidência era a
versão antiga é invalidada), `delete_fact` desliga, `clear` desliga tudo.
**Live com `claude_cli` (sonnet) passou**: 4 arestas de um fato de memória
(`leads` 0.95 extracted, `member_of` 0.7 inferred, `on_call_for` 0.95,
`uses` 0.85), todas com `evidence=("mem-1",)`, em 12 s.

Desvios (todos por YAGNI sob o gate): `EmbeddingSynonymLinker`
(`similar_to`) **não construído** — a resolução da v0.2 é chave canônica +
aliases (o modelo devolve aliases; o matcher de menções os usa); regra de
merge de descrição do LightRAG **não construída** (nós não têm descrição
ainda; `metadata` guarda `type`); passada de contradição **não construída**
(invalidação vem de re-ingestão e de `valid_to`/`invalidate_edge`);
"ligado por default para memória" virou **opt-in por construção** —
o `MemoryManager` só indexa quando recebe `graph=`; sem `aextract` (o
indexador é síncrono). Suíte 3048 verdes; ruff/mypy no baseline.

### E. Comunidades + hubs + docs

- [ ] `graph/algorithms.py`: `label_propagation` ou Louvain próprio (≤150
      linhas), testado contra `networkx` (dev-dep) por modularidade; cache
      por `(vault, scope, graph_version)`; `KnowledgeGraph.communities()`,
      rótulo de comunidade em `explain`/`hubs`.
- [ ] Extra `[graph]` = `igraph`: Leiden + `personalized_pagerank` em C
      quando instalado (mesma API, troca de motor).
- [ ] Docs (`guides/memory.md`, `api/memory.md`, `api/pipeline.md`, `faq.md`,
      `cookbook/production-patterns.md` — lista no doc de pesquisa),
      CHANGELOG (Breaking: `link_memory` → `link_item`; Added), `llms.txt`.
- [ ] Ritual xhigh do diff da frente + juiz; Review preenchido.

## Review

_(preencher ao final)_
