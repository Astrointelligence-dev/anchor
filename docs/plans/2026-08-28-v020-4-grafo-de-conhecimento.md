# v0.2 · #4 — Grafo de conhecimento sobre memória e documentos

**Status:** não iniciado · **Tamanho:** grande · **Depende de:** #3 (vault/namespace)
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

- [ ] **Aresta tem tipo além de validade?** `(auth) --[depende_de]--> (db)` ou
      ligação simples? Tipo é o que deixa `path` explicar *por que* dois
      conceitos se conectam — mas dobra o custo de extração. **Arthur precisa
      responder.**
- [ ] **Incremental já na v0.2, ou full-rebuild primeiro?** **Arthur precisa
      responder.**
- [ ] Grafo é `GraphRetriever` de primeira classe, ou continua um step opcional?
- [ ] Export pra vault Obsidian de verdade (abrir e navegar com o olho)?

---

## Escopo

- [ ] Mergear `feature/storage-layer-gaps` (persistência do grafo)
- [ ] `SimpleGraphMemory` → grafo sobre `ContextItem`, com vault/namespace
- [ ] Arestas com proveniência e validade temporal
- [ ] Extração: wikilinks primeiro, extrator LLM depois
- [ ] Detecção de comunidade + hubs (extra `[graph]`)
- [ ] API de navegação: `query` (BFS com orçamento), `path`, `explain`, `backlinks`
- [ ] `GraphScope` aplicado em toda travessia (de #3)
- [ ] `graph_scope` no `SubagentDefinition`

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

## Review

_(preencher ao final)_
