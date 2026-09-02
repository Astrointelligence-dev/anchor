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

- [ ] `models/graph.py`: `GraphNode(id, label, aliases, metadata)` com chave
      canônica (NFKC + casefold + strip; `normalize_node_id`), `GraphEdge`
      (schema mínimo do doc de pesquisa: `relation`, `fact`, `provenance`,
      `confidence`, `evidence ≥1`, `valid_from/valid_to/created_at/
      invalidated_at`, `metadata`), `Provenance` Literal, `is_visible(edge,
      as_of)` — um predicado, um lugar.
- [ ] `protocols/storage.py`: `GraphStore` / `AsyncGraphStore`, vault na
      montagem (`validate_vault`, padrão `SqliteContextStore:20`).
      Superfície: `upsert_node`, `get_node`, `add_edge` (merge de evidência
      se a aresta viva já existe), `invalidate_edge`, `link_item(node_id,
      item_id, namespace)`, `unlink_item(item_id)` (invalida arestas sem
      evidência restante), `neighbors(node_id, *, scope, as_of, relation)`,
      `subgraph(scope, as_of)` (nós + arestas visíveis — o que PPR/comunidade
      consomem), `graph_version`, `clear`.
- [ ] `storage/memory_store.py`: `InMemoryGraphStore`. Visibilidade por
      escopo = evidência ∩ itens visíveis ≠ ∅ (`RetrievalScope.matches`),
      aplicada ao nó **antes** de qualquer travessia (parede, não ponte).
- [ ] `graph/algorithms.py` (Python puro): `personalized_pagerank` (power
      iteration), `bfs_budgeted` (o pruning do PathRAG: recurso decai por
      salto, poda abaixo de θ), `shortest_path`, `degree`. Sem lib.
- [ ] `graph/knowledge_graph.py`: `KnowledgeGraph(store, context_store)` —
      `query(seeds, *, scope, as_of, top_k)` → itens ranqueados, `path(a,
      b)`, `explain(a, b)` (arestas + `fact` + `provenance` + evidência),
      `backlinks(node)`, `hubs(k)` (grau). `SimpleGraphMemory` vira fachada
      fina sobre `InMemoryGraphStore` com `link_item` (breaking:
      `link_memory` sai; CHANGELOG).
- [ ] Testes com decoy: nó excluído invisível por travessia, `backlinks`,
      `path` (não atravessa por cima) e `explain`; `as_of` esconde aresta
      invalidada; subagente com escopo ∩ não alarga.

### B. Golden set + extração determinística + `GraphRetriever` + A/B (o gate)

- [ ] Corpus versionado em `tests/fixtures/graph_corpus/`: ~40 notas markdown
      com wikilinks (3 estratos: fato simples / multi-hop / síntese-explicação)
      + golden JSONL (`evaluation/golden.py` formato). Feito à mão,
      determinístico.
- [ ] `ingestion/graph_extractors.py`: `WikilinkExtractor` (gramática Obsidian
      completa: `[[note]]`, `|alias`, `#heading`, `#^block`, `![[embed]]`;
      resolução NFC + casefold, basename vault-wide, ambíguo → `ambiguous`)
      e `StructureExtractor` (`parent_id` → `contains`, `doc_id` →
      `belongs_to`). `MarkdownParser` passa a expor `metadata["wikilinks"]`.
      `GraphIndexer(store, extractors)` idempotente (reindexar = mesma
      evidência, zero duplicata).
- [ ] `retrieval/graph.py`: `GraphRetriever(Retriever)` — sementes =
      `entity_extractor(query)` ∪ (opcional) top-k do `VectorStore` → nós;
      PPR sobre `subgraph(effective_scope)` → score por item →
      `ContextStore.get` → `ContextItem` com **id canônico** (funde no RRF).
- [ ] `evaluation`: `evaluate_retriever` e `ABTestRunner.run` ganham `scope`
      (buraco achado na pesquisa).
- [ ] `tests/test_retrieval/test_graph_ab_benchmark.py` no molde do
      `test_scope_recall_benchmark.py`: 3 condições — híbrido+RRF, grafo só,
      RRF(híbrido, grafo) — por estrato, número publicado no Review.
      **Gate:** grafo (ou a fusão) ganha em multi-hop/explain → D vale;
      senão D vira opt-in mínimo e o Review diz por quê.

### C. Persistência + CLI

- [ ] `storage/sqlite/_graph_store.py` (`SqliteGraphStore` + twin async):
      `graph_nodes`, `graph_edges`, `graph_node_items`, `graph_edge_items`
      em `_TABLES` com PK `(vault, …)`, índice único parcial `WHERE
      invalidated_at IS NULL`, `_SCOPED_TABLES`; visibilidade por
      `EXISTS` + `scope_sql_clauses(scope, "namespace")` (`_where.py:156`).
      `subgraph` carrega o visível de uma vez (`ponytail:` teto ~50k arestas,
      paginação/CTE quando doer).
- [ ] `storage/postgres/_graph_store.py`: mesmo schema, `COLLATE "C"`,
      integração com docker (`ANCHOR_TEST_POSTGRES_DSN`) + fake asyncpg.
- [ ] Fixture `make_graph_store(vault)` em `tests/test_storage/conftest.py`;
      teste de isolamento entre vaults num só `.db`.
- [ ] CLI: `anchor index --graph` (roda `GraphIndexer` com os extratores
      determinísticos) e `anchor graph query|path|explain|hubs|backlinks`.
- [ ] Remover o worktree `.worktrees/storage-layer-gaps` (branch fica).

### D. Extrator LLM + resolução de entidade (opt-in; default ligado para memória)

- [ ] `LLMGraphExtractor`: uma chamada por chunk, saída estruturada
      (entidades + triplas com `relation`, `fact`, `confidence` na rubrica),
      `max_gleanings=0`, vocabulário sugerido no prompt; provider injetado
      (o do agente/manager), nunca cliente novo.
- [ ] Resolução: chave canônica + tabela de aliases (o `|alias` do wikilink
      entra de graça); `EmbeddingSynonymLinker` opt-in (`similar_to`,
      `inferred`, cosseno ≥ 0.8, só entre nomes de entidade).
- [ ] Descrição de entidade = fragmentos por `item_id`; resumo LLM só com ≥8
      (regra LightRAG).
- [ ] `MemoryManager`: extrator ligado por default para `MemoryEntry` (não tem
      wikilink); invalidação por re-ingestão; passada de contradição opt-in.
- [ ] Um caso live com `claude_cli`.

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
