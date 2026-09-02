# Grafo de conhecimento (frente #4) — pesquisa dirigida

**Data:** 2026-09-02 · **Baseline:** `dev` @ `6cbedfd` · **Item:** `docs/plans/2026-08-28-v020-4-grafo-de-conhecimento.md`
**Método:** 4 agentes em paralelo — (1) arquitetura de consulta e construção
incremental (fontes primárias + medições locais de networkx/igraph); (2) modelo
de aresta e fontes de extração (código-fonte do Graphiti, mem0, LightRAG,
HippoRAG, GraphRAG e graphify baixado e lido, não recordado); (3) mapa do
código da `dev` com `arquivo:linha`; (4) parecer sobre o worktree
`feature/storage-layer-gaps`. Os quatro relatórios completos estão abaixo do
sumário, na íntegra.

---

## Sumário executivo

### Os números do doc de entrada estavam errados

| Afirmação no doc | O que a fonte diz | Fonte |
|---|---|---|
| "GraphRAG: 86% vs 32% do RAG baseline" | É o RobustQA da **Writer** (86,31% vs 32,74–75,89% de vendors de vector RAG). O paper do GraphRAG só reporta win rates de LLM-juiz (72–83%), sem acurácia | [writer.com/rag-benchmark](https://writer.com/engineering/rag-benchmark/) · [2404.16130](https://arxiv.org/html/2404.16130v2) |
| "PathRAG: −44% de contexto" | Tabela 6: **−13,69%** (PathRAG) e −40,41% (PathRAG-lt) vs LightRAG | [2502.14902](https://arxiv.org/html/2502.14902) |
| "HippoRAG 2: multi-hop 10–30× mais barato" | É **HippoRAG 1 vs IRCoT**, só custo online. HippoRAG 2 indexa em 99,5 min vs 277 (GraphRAG) e ganha só +2,3% em QA simples | [2405.14831](https://arxiv.org/abs/2405.14831) · [2502.14802](https://arxiv.org/html/2502.14802) |

O que 2026 acrescenta: GraphRAG-Bench (ICLR'26) — RAG vence em fato simples
(60,9 vs 49,3), grafo vence em multi-hop; e **RRF(híbrido, grafo) +6,4%** sobre
qualquer um sozinho. O A/B do doc precisa dessa terceira condição.

### Bloqueio 1 — aresta tem tipo? → **rótulo livre normalizado, sem enum** (confiança alta)

Nenhum sistema de extração por LLM usa enum fechado como único caminho:
GraphRAG e LightRAG nem têm rótulo (descrição + peso); mem0 e HippoRAG usam
predicado livre; Graphiti deriva `relation_type` em SCREAMING_SNAKE_CASE e só
oferece tipos Pydantic **opt-in** com fallback `RELATES_TO`; graphify sugere
vocabulário no prompt e não valida. **"Dobra o custo" não confirma**: o
predicado sai na mesma chamada que a tripla em todos os prompts reais. O que
dobra é *gleaning* (2ª passada "faltou algo?") — default `max_gleanings=0`.
Wikilink → `links_to`; estrutura pai/filho → `contains`; embedding →
`similar_to`; LLM → predicado normalizado (`casefold`, espaços → `_`) com 5–8
rótulos sugeridos no prompt. Se um dia precisar de tipos fechados, o caminho
do Graphiti encaixa por cima sem migração porque a coluna já é string.

### Bloqueio 2 — incremental ou full-rebuild? → **incremental para nós/arestas/evidência; derivados recomputados, nunca persistidos** (confiança alta)

A camada cara de invalidar é *comunidade com resumo por LLM* — e o anchor
decidiu não ter isso ("hub é calculado"). Sem LLM, Louvain global em 10k nós /
54k arestas custa **0,49–2,17 s em Python puro** (medido). É o que LightRAG,
HippoRAG 2 e Graphiti fazem; o `graphrag update` real **não re-clusteriza**
(roda Leiden só no delta e concatena com offset de ID). O mínimo sem dívida:
upsert por chave natural `(vault, entity_key)`; aresta com `evidence`
(item_ids) e `invalidated_at`; remover documento = invalidar arestas cuja
evidência era só aquele item; `graph_version` = contagem de arestas vivas;
hubs/comunidades `@cache` por `(vault, scope, graph_version)`. Sem tabela
`communities`. Com chaves naturais, "full-rebuild" é `clear()` + reindex — os
dois modos coexistem sem migração. **A pergunta desaparece se as chaves forem
certas desde o início.**

### Worktree `feature/storage-layer-gaps` → **descartar e reescrever, usando o branch como referência**

64 commits atrás da dev; `git merge-tree` dá 6 conflitos (4 triviais, 2
médios em `_schema.py`). Os testes rodam (58 passed, 10 skipped — todo o
Postgres pulado). Mas o schema contradiz as decisões de agosto em três eixos
que tocam *toda* query: sem `vault` (nenhuma das 3 tabelas), sem validade nem
proveniência (`UNIQUE(source,relation,target)` proíbe reafirmar aresta
invalidada; `remove_edge` é DELETE), e as 6 travessias (4 BFS + 2 CTE) leem só
`graph_edges`, sem predicado de evidência por salto. Liga `memory_id`, não
`item_id`. Em DDL tudo é ALTER; em código, 70–80% do GraphStore é reescrito.
Merge (1,5 h) publicaria uma API sem vault que o #4 trocaria na sessão
seguinte. Cherry-pick não tem candidato. **O que copiar como referência:**
forma de 3 tabelas + auto-criação de nó, esqueleto do CTE recursivo do
Postgres, cenários de teste (ciclo, depth-2, `relation_filter`, cascata).
`ConversationStore` e cache backends ficam fora (nenhuma frente pede; HITL
durável está fora da v0.2).

### Arquitetura de consulta → grafo `{entidade, item}`, PPR/fluxo com decaimento, zero LLM na consulta

`ContextItem` como nó do grafo (padrão HippoRAG 2 — casa com "moeda única").
Sementes = entidades da query + wikilinks + embedding do item → PPR (~30
linhas, 0,09 s em 10k nós) *ou* espalhamento com decaimento/limiar (o
*pruning* do PathRAG, que é literalmente BFS com orçamento) → score por
`ContextItem` → `TokenBudget` corta por `SourceType`. Nó excluído removido
*antes* do espalhamento (parede, não ponte). Descartar do PathRAG o pareamento
O(N²) e o path-prompting. **Não implementar global search / community
reports** (57× tempo, 210× tokens por pergunta; até 331k tokens no
GraphRAG-Bench). Comunidade é rótulo para `explain`/hubs. Hub = grau
(sobreposição 16/20 e 20/20 com PageRank).

**`GraphRetriever` de primeira classe** implementando `Retriever`
(`protocols/retriever.py:20`, já com `scope`): ganha de graça a interseção de
escopo do `retriever_step` (`pipeline/step.py:101`), a fusão RRF dentro do
`HybridRetriever` (`hybrid.py:70`) e o `ABTestRunner`. O `graph_retrieval_step`
atual não tem nenhuma das três (sem escopo, `items + new_items`, full scan do
store por query) — fica como legado exportado.

### Dependência → core sem lib; extra `[graph]` = **igraph**, não networkx nem graspologic

PPR + grau + Louvain/label-propagation próprios (≤150 linhas; o Louvain do
networkx tem 141 linhas de Python puro). `networkx.leiden_communities` é
*backend-only* (`implemented_by_nx=False`) — a razão óbvia para depender dele
não existe. igraph: 2 MB, 1 dep, Leiden + `personalized_pagerank` em C (é o
que HippoRAG usa). graspologic: 17 deps, scipy 28 MB, quebra em 3.13 no
graphify. networkx só faz sentido como `to_networkx()` de export. Confiança
média-alta: "próprio vs networkx" é gosto (networkx tem 0 deps); graspologic é
o único indefensável.

### Extração → ordem por custo

1. **Wikilinks — sempre** (grátis, determinístico, `extracted` 1.0,
   `links_to`): gramática `[[note]]`, `[[note|alias]]`, `[[note#heading]]`,
   `[[note#^block]]`, `![[embed]]`; resolução NFC + casefold, basename
   vault-wide, caminho desambigua, ambíguo → `ambiguous`. **Não existe parser
   de wikilink no repo** (zero ocorrências em `src/`); `MarkdownParser` já
   extrai frontmatter (`tags`/`aliases` de graça) e headings.
2. **Estrutura já existente — sempre**: `metadata["parent_id"]` do
   `ParentChildChunker` (content-hash, estável) vira `contains`; `doc_id` +
   `chunk_index` dão `belongs_to_doc`/`next`.
3. **LLM — opt-in por fonte**: default ligado para memória (não tem wikilink),
   desligado para docs com ≥1 wikilink; `max_gleanings=0`; saída com
   `provenance` + rubrica de confiança. Regra de merge de descrição do
   LightRAG: lista de fragmentos por `item_id`, resumo LLM só com ≥8.
4. **Embedding — opt-in, só entre nomes de entidade**: `similar_to`,
   `inferred`, cosseno ≥ 0.8 (HippoRAG), nunca item×item.

Entidade: chave canônica NFKC+casefold, tabela de aliases (o `|alias` do
wikilink de graça), embedding só sugere `same_as`, merge só confirmado.

### Schema mínimo de aresta

```python
class Edge(BaseModel):
    source: str; target: str
    relation: str                                # rótulo livre normalizado
    fact: str | None = None                      # frase-evidência p/ explain()
    provenance: Literal["extracted", "inferred", "ambiguous"]
    confidence: float                            # extracted=1.0; inferred ∈ {.95...55}; ambiguous ≤ .3
    evidence: list[str]                          # item_ids, ≥1 — é a chave do escopo E do incremental
    valid_from: datetime | None = None           # tempo do mundo
    valid_to: datetime | None = None
    created_at: datetime                         # tempo de transação
    invalidated_at: datetime | None = None       # None = viva
    metadata: dict = {}
```

Sem `weight` (= `len(evidence)`). Bi-temporal completo custa uma coluna
nullable a mais e dá `as_of` em toda travessia. Predicado único de
visibilidade: `invalidated_at IS NULL AND (valid_from IS NULL OR valid_from <=
:as_of) AND (valid_to IS NULL OR valid_to > :as_of)`. Índice único parcial
`WHERE invalidated_at IS NULL` no lugar do `UNIQUE(source, relation, target)`.
Evidência em tabela própria `graph_edge_evidence(edge_id, item_id, vault,
namespace)` — desnormalizada, porque itens de memória não vivem em
`context_items`. Exclusão por escopo = `EXISTS` sobre a evidência com
`scope_sql_clauses(scope, "namespace")` (`storage/_where.py:156`), aplicado a
cada salto.

Não copiar do Graphiti agora: chamada LLM de contradição a cada inserção.
Invalidação na v0.2 vem de re-ingestão (determinística) e de uma passada
opcional de contradição no extrator, opt-in.

### Seams do código (`dev` @ `6cbedfd`)

| Seam | Onde | Nota |
|---|---|---|
| Mudança de moeda | `memory/graph_memory.py:87` `link_memory(entity_id, memory_id)` | `memory_id` → `item_id`; tudo em memória, sem vault/namespace/persistência |
| Único contato grafo↔pipeline | `pipeline/memory_steps.py:51` | sem `scope`; `store.list_all()` por query (`:104`) |
| Predicado de escopo | `storage/_where.py:156` `scope_sql_clauses` + `_prefix_range:147` | range boundary-aware; nunca `LIKE` |
| Escopo ativo | `models/scope.py:142` **`ACTIVE_SCOPE`** (sem underscore), `effective_scope:152`, `scope_kwargs:162`, `same_vault:172` | |
| Onde um retriever pluga | `pipeline/step.py:101` `effective_scope(scope)`; `protocols/retriever.py:20` | `GraphRetriever` entra sem código novo |
| Schema SQLite | `storage/sqlite/_schema.py:153` `ensure_tables`; helper real é `rebuild_with_scope_key:91` (não existe `add_scope_columns`) | `_TABLES` dict + `_SCOPED_TABLES` |
| Fixture a copiar | `tests/test_storage/conftest.py:118` `make_context_store` (factory por vault) | permite teste de vazamento entre vaults num só `.db` |
| Benchmark a copiar | `tests/test_storage/test_scope_recall_benchmark.py` | molde do A/B |
| Aresta grátis | `ingestion/hierarchical.py:140` `parent_id = sha256(parent_text)[:16]` | content-hash, deduplica parents idênticos |
| Campo dormente | `models/memory.py:64` `MemoryEntry.links` | persistido em todos os backends, ninguém lê |

**Surpresas:** `MemoryRetrieverAdapter` gera `ContextItem` com uuid novo — o
`memory_id` só sobrevive em `metadata` (`retrieval/memory_retriever.py:290`),
e `rrf_fuse` funde por `item.id` (`_rrf.py:54`) — sem a moeda única o grafo
nunca funde com o denso. `ABTestRunner.run` e `evaluate_retriever` chamam
`retrieve(query, top_k)` **sem `scope`** — a bancada não sabe A/B-testar
retriever escopado (o benchmark da #3 contornou com adaptador). **Não existe
golden set real** — só fixtures sintéticas. `networkx` está no `uv.lock` só
como transitiva de `torch`.

### Verificação (A/B) — 3 condições, golden set estratificado

Golden set com 3 estratos (fato simples / multi-hop / síntese-explicação);
condições: híbrido+RRF, grafo só, **RRF(híbrido, grafo)**. Critério de
investir no construtor LLM = ganho em multi-hop/explain, não na média
(GraphRAG-Bench mostra RAG vencendo em fato simples).

---
---


# Parte 1 — Arquitetura e construção incremental


**Escopo:** perguntas 1, 4 e 5 do doc `docs/plans/2026-08-28-v020-4-grafo-de-conhecimento.md`.
**Método:** só fontes primárias (arXiv, repositórios oficiais, docs oficiais) + medições locais reproduzíveis (wheels baixados do PyPI, benchmark em Python 3.14.5 / networkx 3.6.1 sem scipy). Data: 2026-09-02.

**Contexto do anchor usado como referência:** `TokenBudget` aloca `max_tokens` + `priority` por `SourceType` com pool compartilhado (`src/anchor/models/budget.py`); `SimpleGraphMemory` faz BFS não ponderada até `max_depth` (`src/anchor/memory/graph_memory.py`); `graph_retrieval_step` = extrator → BFS → `MemoryEntry` → `ContextItem(priority=6)` com corte por `max_items`, não por tokens (`src/anchor/pipeline/memory_steps.py:51`); protocolo `GraphStore` no worktree já tem `add_node`/`add_edge` com semântica de upsert (merge de metadata) e `get_neighbors(max_depth, relation_filter)` (`.worktrees/storage-layer-gaps/src/anchor/protocols/storage.py:384`).

---

## 1. Arquitetura de consulta

### 1.1 Tabela comparativa

| Sistema | Índice (o que é construído) | Chamadas LLM na indexação, por unidade | Caminho de CONSULTA (o que entra no contexto do LLM) | Chamadas LLM por consulta | Usa comunidades? |
|---|---|---|---|---|---|
| **MS GraphRAG** (2404.16130) | entidades+relações+claims por chunk de 600 tok → resumo por elemento → Leiden hierárquico (graspologic-native, `max_cluster_size=10`) → *community report* por comunidade em cada nível | por chunk: 1 extração + `max_gleanings=1` (re-prompt "faltou algo?"); + 1 resumo por entidade e por relação com múltiplas descrições; + 1 report por comunidade por nível; claims opcionais. Podcast (1 M tokens, 8.564 nós / 20.691 arestas): **281 min** de indexação | **Local:** top‑k entidades por embedding (`top_k_entities=10`, `top_k_relationships=10`) → relações, covariates, reports e text units, com proporções `text_unit_prop=0.5`, `community_prop=0.15`, `max_context_tokens=12 000`. **Global:** map‑reduce sobre *todos* os reports de um nível (`data_max_tokens=12 000` por lote), respostas intermediárias pontuadas 0–100, reduce. **DRIFT:** primer com top‑K reports → perguntas de follow‑up → local search | Local: 1. Global: ≥2 (1 por lote de reports no map + 1 reduce) — no MultihopSum, **57× o tempo e 210× os tokens** do VanillaRAG (~9 min, ~300 K tokens por pergunta) | Sim — é o núcleo do global search |
| **LightRAG** (2410.05779) | entidades+relações por chunk de 1200 tok, com "profiling" (par chave‑valor por nó/aresta) e dedup por nome; grafo + 2 índices vetoriais (entidades, relações) | "total tokens / chunk size" chamadas (1 por chunk de extração; gleanings opcionais no código). Merge de entidade repetida: descrições concatenadas com `<SEP>`; LLM só resume quando ≥ `force_llm_summary_on_merge=8` fragmentos ou > `summary_context_size=12 000` tokens | 1 chamada extrai *keywords* low‑level (entidades específicas) e high‑level (temas) → busca vetorial em entidades e relações → **vizinhos a 1 hop** → concatena descrições de entidades/relações + chunks de origem. Retrieval: **<100 tokens e 1 chamada** vs GraphRAG 610 K tokens / centenas de chamadas (dataset legal) | 1 (keywords) + 1 (resposta) | **Não** |
| **PathRAG** (2502.14902) | "construímos o grafo seguindo o método do GraphRAG" — na prática o repo é um fork do LightRAG (mesmos módulos `operate.py/base.py/storage.py/prompt.py`, funções `kg_query`, `_find_most_related_text_unit_from_entities`) | igual ao LightRAG (herdado) | keywords → N=40 nós por embedding → **poda por fluxo** entre pares de nós: recurso 1 no nó semente, cada vizinho recebe α·S(v)/grau(v) (α=0,7 no paper; **0,8 no código**), para quando S(v)/grau < θ; caminho pontuado pela média do recurso nas arestas; caminhos viram texto e entram no prompt em **ordem crescente de confiabilidade** (o melhor no fim) | 1 (keywords) + 1 (resposta) | **Não** |
| **HippoRAG 2** (2502.14802) | OpenIE por passagem (triplas sem esquema) → nós de frase + **nós de passagem** ("contains") + arestas de sinônimo por embedding (cos > 0,8); grafo igraph em pickle | **1 chamada OpenIE por passagem** (+ embeddings). MuSiQue (11.656 passagens): **99,5 min** vs GraphRAG 277 min, LightRAG 235 min, RAPTOR 100,5 min | query → top triplas por embedding → **filtro de reconhecimento** (1 chamada LLM sobre top‑5 triplas) → **PPR** (igraph `personalized_pagerank`) com reset nas entidades‑semente e nas passagens (peso 0,05 × similaridade) → **top‑5 passagens** vão para o LLM | 1 (filtro) + 1 (resposta); 1,2 s/pergunta vs GraphRAG 10,7 s | **Não** |
| **Graphiti/Zep** (2501.13956) | por episódio: extração de entidades (com janela dos últimos n=4 episódios) → resolução contra nós existentes (busca cos + BM25, depois LLM) → extração de fatos/arestas → dedup de arestas → invalidação temporal (t_invalid = t_valid da aresta contraditória) → atributos/resumo do nó | 5 chamadas LLM por episódio (`extract_nodes`, `resolve_extracted_nodes`, `extract_edges`, `resolve_extracted_edges`, `extract_attributes_from_nodes`) + 1 por comunidade tocada se `update_communities=True` (**default False**) | busca híbrida: cos + BM25 + BFS a partir de nós recentes; rerank por RRF, MMR, menções de episódio, distância no grafo, cross‑encoder; devolve fatos/arestas e resumos de nó | 0 na busca (só rerankers) | Sim, mas opcional — label propagation, resumos via LLM |

Fontes: GraphRAG paper [arXiv:2404.16130](https://arxiv.org/html/2404.16130v2); dataflow [microsoft.github.io/graphrag/index/default_dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/); defaults [`config/defaults.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py); local/global/DRIFT [query docs](https://microsoft.github.io/graphrag/query/local_search/), [global](https://microsoft.github.io/graphrag/query/global_search/), [drift](https://microsoft.github.io/graphrag/query/drift_search/); custo 57×/210× [arXiv:2503.04338](https://arxiv.org/html/2503.04338). LightRAG [arXiv:2410.05779](https://arxiv.org/html/2410.05779v2), merge em [`lightrag/operate.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py) e [`constants.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py). PathRAG [arXiv:2502.14902](https://arxiv.org/html/2502.14902), código [`PathRAG/operate.py`](https://github.com/BUPT-GAMMA/PathRAG/blob/main/PathRAG/operate.py) (`bfs_weighted_paths`, `alpha = 0.8`). HippoRAG 2 [arXiv:2502.14802](https://arxiv.org/html/2502.14802), código [`HippoRAG.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py). Zep [arXiv:2501.13956](https://arxiv.org/html/2501.13956), código [`graphiti.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/graphiti.py).

### 1.2 Essencial vs acessório

| Sistema | Essencial (sem isso não é o método) | Acessório (dá para tirar sem perder a ideia) |
|---|---|---|
| GraphRAG | *community reports* + map‑reduce (é o que responde pergunta "global"/de síntese) | gleanings, claims, DRIFT, resumo de descrições, Leiden hierárquico com vários níveis (o próprio paper mostra C0 — nível raiz — com 97 % menos tokens e win rate parecido) |
| LightRAG | keywords em dois níveis + índice vetorial de **relações** (não só entidades) + expansão 1 hop | profiling K‑V, índice incremental (é união de conjuntos — trivial) |
| PathRAG | poda por fluxo com decaimento + limiar (é uma BFS com orçamento) | caminhos entre *pares* (O(N²) pares), ordenação crescente no prompt, N=40 |
| HippoRAG 2 | **nós de passagem no mesmo grafo das entidades** + PPR | filtro de reconhecimento por LLM (a ablação mostra ganho pequeno), OpenIE sem esquema |
| Graphiti | resolução de entidade contra o grafo existente + bitemporalidade (t_valid/t_invalid) | comunidades (default off), resumos de nó |

### 1.3 O que casa com `TokenBudget` + BFS

O `TokenBudget` do anchor é: um teto de tokens por `SourceType`, prioridade inteira, pool compartilhado. O que o grafo precisa devolver é **uma lista ordenada de `ContextItem`** que o orçamento corta. Logo o encaixe é com métodos que produzem um *ranking* de itens, não com métodos que produzem *um texto* (community report).

- **PathRAG — confirmado, com uma correção.** A poda por fluxo é literalmente uma BFS com orçamento: o nó semente recebe recurso 1, cada vizinho recebe `α·S(v)/grau(v)`, a travessia para quando `S(v)/grau(v) < θ`. Trocar "recurso" por "tokens alocados ao `SourceType`" dá o mesmo algoritmo: o orçamento se divide entre os vizinhos, decai por salto e o nó cujo quinhão não paga nem o menor item é podado. O que **não** casa é a parte de "caminhos entre pares de nós": ela exige N sementes e O(N²) buscas de caminho, e serve para o prompt de *narrativa* (path‑based prompting), que o anchor não precisa porque quem monta o prompt é o `ContextItem`. Fica: o *pruning*, sem o *pairing*.
- **HippoRAG 2 (PPR) — casa igualmente bem e é mais robusto.** PPR é o mesmo espalhamento, mas iterado até convergência em vez de 1 passe; com **nós de passagem no grafo** (= `ContextItem` como nó, exatamente a decisão "ContextItem é a moeda única"), o resultado do PPR já é um score por item, que o `TokenBudget` corta. Custo: power iteration de ~30 linhas, 0,09 s em 10k nós / 54k arestas em Python puro (medido aqui, ver §3). PPR também dá "hub" de graça (ver §3.3).
- **LightRAG** — a expansão 1‑hop + índice de relações é o que o `graph_retrieval_step` já faz com `max_depth`; a única peça que vale copiar é a **regra de merge de descrições** (§2).
- **GraphRAG global/DRIFT — não casa.** Global search produz texto, não itens; precisa de reports resumidos por LLM (a parte cara e a parte que precisa de invalidação), e os números de 2025/26 mostram que é o pior custo/benefício (§4). Comunidades no anchor servem para *navegação* (hubs, `explain`), não para responder — o doc já decidiu isso ("hub é calculado, não declarado").

**Arquitetura de consulta recomendada:** sementes (wikilink/entidade da query + embedding do item) → PPR *ou* fluxo‑com‑decaimento sobre o grafo `{entidade, item}` restrito ao `GraphScope` (nó excluído removido *antes* do espalhamento, para ser parede e não ponte) → score por `ContextItem` → `TokenBudget` corta por `SourceType`. Zero chamadas LLM na consulta. Comunidades entram só em `explain`/`backlinks` como rótulo.

---

## 2. Incremental vs full‑rebuild

### 2.1 Como cada um faz

| Sistema | Nós/arestas | Comunidades | Evidência |
|---|---|---|---|
| **LightRAG** | Novo documento passa pelo mesmo φ; grafo final = **união** de nós e arestas. Entidade de mesmo nome: descrições concatenadas com `<SEP>` (dedup exato), `entity_type` = o mais frequente, `weight` da aresta = soma (piso = nº de chunks‑fonte); LLM só re‑resume quando ≥ 8 fragmentos (`DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8`) ou > 12 000 tokens | Não tem — é o argumento do paper: GraphRAG precisaria regenerar "1 399 comunidades × 2 × 5 000 tokens" no dataset legal | [paper §"Fast Adaptation"](https://arxiv.org/html/2410.05779v2); [`operate.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py) (`_merge_nodes_then_upsert`, `_handle_entity_relation_summary`); [`constants.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py) |
| **MS GraphRAG ≥ 1.0** (`graphrag update`, métodos `standard-update`/`fast-update`) | O delta roda o **pipeline padrão inteiro** (`["load_update_documents", *_standard_workflows, *_update_workflows]`), depois `update_entities_relationships` agrupa por título, junta descrições em lista e re‑roda `summarize_descriptions` nas entidades mescladas; `degree` fica "first" com um `# todo: re‑compute` | **Não re‑clusteriza.** Leiden roda só no grafo do delta; `_update_and_merge_communities` desloca os IDs do delta por `old_max + 1` e faz `pd.concat`; reports idem. As comunidades antigas nunca absorvem entidades novas. O plano de 2024 (colocar entidades novas em comunidades existentes, re‑rodar Leiden por limiar) segue como intenção no issue #741 | [`workflows/factory.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/workflows/factory.py); [`index/update/communities.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/update/communities.py); [`index/update/entities.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/update/entities.py); [CLI](https://microsoft.github.io/graphrag/cli/); [issue #741](https://github.com/microsoft/graphrag/issues/741) |
| **Graphiti** | Incremental por construção: cada episódio resolve entidades contra o grafo existente (busca híbrida + LLM) e arestas contra arestas do mesmo par; contradição → `t_invalid` marcado, nunca delete | Label propagation (não Leiden). Nó novo é atribuído à comunidade da **pluralidade dos vizinhos** (`determine_entity_community`), e o resumo daquela comunidade é regenerado por LLM (`update_community`). O paper admite que "as comunidades divergem gradualmente" e "refreshes periódicos continuam necessários" (`build_communities()` = recompute total). `update_communities` é **False por default** | [paper](https://arxiv.org/html/2501.13956); [`community_operations.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/community_operations.py); [`graphiti.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/graphiti.py) |
| **HippoRAG 2** | `index()` carrega OpenIE já feito, roda OpenIE só nos chunks novos (`load_existing_openie`, `chunk_keys_to_process`), e chama `add_fact_edges`/`add_passage_edges`/`add_synonymy_edges` sobre o grafo igraph existente | Não tem | [`HippoRAG.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py) |
| **graphify** (modelo do doc) | `--update`: "re‑extract only changed files, merge into existing graph" | README não diz se re‑roda comunidade; Leiden via graspologic é extra opcional (`[leiden]`, quebra em Python 3.13 — issue #290) | [README](https://github.com/Graphify-Labs/graphify), [issue #290](https://github.com/Graphify-Labs/graphify/issues/290) |

### 2.2 Estratégias para comunidades quando o grafo muda

| Estratégia | Quem usa | Custo quando muda | Dívida |
|---|---|---|---|
| **Recompute global** (Leiden/Louvain no grafo inteiro) | GraphRAG `index` (não `update`); Graphiti `build_communities()` | O(E) em Python puro: **0,5–2,2 s em 10k nós / 50k arestas** medidos aqui — barato **se não houver resumo por LLM**. Com reports por LLM é o cenário "1 399 × 2 × 5 000 tokens" | nenhuma; comunidade = dado derivado |
| **Recompute local** (só comunidades tocadas) | plano do GraphRAG (#741), nunca entregue | precisa de índice comunidade→nós e regra de "quando re‑rodar"; IDs de comunidade viram estado | alta: IDs estáveis, drift, limiares |
| **Atribuição por vizinhança** (nó novo vai para a comunidade da maioria dos vizinhos) | Graphiti (`update_communities=True`) | O(grau) por nó; mas admitem divergência e refresh periódico | média: acumula drift; precisa do recompute mesmo assim |
| **Append de partição** (delta clusterizado sozinho e concatenado) | GraphRAG `update` (o que existe hoje) | zero | alta: entidades novas nunca entram em comunidades antigas; a hierarquia vira "uma floresta por lote" |
| **Lazy na consulta** (sem persistir; calcular quando pedirem e cachear por versão do grafo) | LazyGraphRAG (grafo de co‑ocorrência NLP + comunidades sem resumo prévio; indexação = 0,1 % do GraphRAG, consulta = 4 % do global search) | custo do recompute global, pago só quando alguém chama `explain`/hubs | nenhuma, desde que exista um `graph_version` (contador de arestas ou hash) para invalidar o cache |
| **Sem comunidades** | LightRAG, PathRAG, HippoRAG 2 | zero | nenhuma; perde só `explain`/hubs por agrupamento |

### 2.3 O que é "incremental" de verdade e o que custa

O custo incremental se divide em três camadas que os sistemas acima tratam de forma diferente:

1. **Nó/aresta/evidência** — em todos os sistemas é união + upsert por chave (nome normalizado). É *gratuito* se a chave primária do store já for `(vault, entity_key)` e a aresta carregar `item_id` de origem. O `GraphStore` do worktree já tem `add_node` com merge de metadata e `add_edge` que cria nós implicitamente — é upsert. A única regra que evita dívida aqui é a do LightRAG: **descrição acumula em lista; resume (LLM) só ao passar um limiar (8)**, senão o nó‑hub vira um parágrafo infinito.
2. **Atributos derivados** (grau, PageRank, hub) — são funções do grafo; recomputar é O(E) sem LLM. Não persistir como coluna "verdade"; se persistir, é cache com versão.
3. **Comunidades com resumo por LLM** — é a única camada em que "invalidar" custa dinheiro. **O anchor não tem essa camada:** o doc fecha "hub é calculado, não declarado" e "comunidade é extra opcional". Portanto o problema que o doc chama de "o caro" não existe na v0.2; ele só aparece se um dia houver `community report` gerado por LLM.

### 2.4 Recomendação para o bloqueio

**"Incremental já na v0.2" para nós, arestas e evidências; "full‑rebuild" (recompute global, lazy, cacheado por versão) para tudo que é derivado — grau, PageRank, comunidades.** Não é meio‑termo: é o que LightRAG, HippoRAG 2 e Graphiti fazem na prática, e é o que o GraphRAG *finge* fazer (o `update` dele é append de partição com drift).

O mínimo viável sem dívida:

- Upsert por chave natural: `entity_key` normalizado (lowercase + strip; a resolução por embedding/LLM do Graphiti é a fase 2, não a 1) — chave já é o que `GraphStore.add_node` faz.
- Aresta `(src, rel, dst, item_id, provenance, valid_from, valid_to)`: adicionar é `INSERT OR IGNORE`; remover documento = marcar `valid_to` nas arestas daquele `item_id` (exclusão por nó cai fora naturalmente: nó sem evidência viva não existe para o escopo).
- Descrição de entidade = lista de fragmentos por `item_id` (não string concatenada); resumo por LLM só sob demanda e por limiar.
- `graph_version` = `count(*)` de arestas vivas ou `max(rowid)`; qualquer derivado (hubs, comunidades) é `@cache` por `(vault, scope, graph_version)`.
- Nenhuma tabela `communities`. `rebuild_communities()` é uma função, não um estado.

**O que "full‑rebuild primeiro" custaria depois:** se o builder for escrito assumindo grafo vazio (IDs sequenciais de entidade, grau/comunidade como coluna, descrição como string única), migrar para incremental exige re‑chavear entidades e reprocessar descrições — é exatamente a dívida que o GraphRAG carrega (`degree: "first"  # todo`). Se as chaves forem naturais desde o início, "full‑rebuild" é só `clear()` + reindex, e "incremental" é o comportamento default do upsert — os dois coexistem sem migração. Ou seja: o custo de migrar não está em *escolher* incremental, está em *escolher chaves erradas*; escolhendo certo, a pergunta desaparece.

---

## 3. Detecção de comunidade e hubs

### 3.1 Bibliotecas (medido: PyPI JSON em 2026‑09‑02, wheels macOS arm64 quando existem)

| Lib | Versão | Wheel | Deps obrigatórias | Louvain | Leiden | Label prop. | Obs. |
|---|---|---|---|---|---|---|---|
| **networkx** | 3.6.1 | 2,07 MB (7,8 MB instalado) | **0** | `louvain_communities` — **Python puro**, presente desde a tag `networkx-2.7.1` (ausente em 2.6.3) | `leiden_communities` existe desde 3.5, mas é `@_dispatchable(implemented_by_nx=False)`: **"does not have a default NetworkX implementation… may only be run with an installable backend such as 'cugraph'"** — chama `NotImplementedError` | `label_propagation_communities`, `asyn_lpa_communities` (Python puro) | `pagerank` padrão exige scipy; existe `_pagerank_python` (fallback interno) |
| **igraph** | 1.0.0 | 2,05 MB | texttable | `community_multilevel` | `community_leiden` | `community_label_propagation` | C; `personalized_pagerank` (é o que HippoRAG usa); sem numpy |
| **leidenalg** | 0.12.0 | 1,93 MB | igraph | — | sim (referência de Traag) | — | só se quiser Leiden "de verdade" |
| **graspologic** | 3.4.4 | 5,2 MB | **17** (numpy, scipy 28,7 MB, scikit‑learn, matplotlib, seaborn, statsmodels, umap‑learn, gensim, POT, hyppo…) | — | `hierarchical_leiden` (é o que o GraphRAG usa, via `graspologic-native`) | — | GraphRAG pina `graspologic-native>=1.2,<1.3` e `networkx~=3.6` |
| **graspologic-native** | 1.3.1 | 0,73 MB | numpy, scipy | — | Leiden em Rust | — | ainda arrasta scipy |
| **cdlib** | 0.4.1 | 0,31 MB | **20** (scikit‑learn, python‑louvain, pulp, demon, networkx…) | via python‑louvain | via leidenalg (opcional) | sim | meta‑biblioteca; pesada por transitividade |

Fontes: [networkx louvain docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html), [networkx leiden docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.leiden.leiden_communities.html) e fonte [`leiden.py`](https://github.com/networkx/networkx/blob/main/networkx/algorithms/community/leiden.py); tags via GitHub API (`networkx-2.6.3`, `2.7.1`, `3.4.2`, `3.5`); PyPI [networkx](https://pypi.org/project/networkx/), [igraph](https://pypi.org/project/igraph/), [leidenalg](https://pypi.org/project/leidenalg/), [graspologic](https://pypi.org/project/graspologic/), [graspologic-native](https://pypi.org/project/graspologic-native/), [cdlib](https://pypi.org/project/cdlib/); GraphRAG [`pyproject.toml`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/pyproject.toml) e [`cluster_graph.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/operations/cluster_graph.py); igraph [API](https://python.igraph.org/en/stable/api/igraph.Graph.html).

### 3.2 "Cabe em <150 linhas de Python puro?" — sim, e o networkx é a prova

Contagem no wheel 3.6.1 (docstrings removidas via `ast`): `louvain.py` = **141 linhas de código** (384 com docs), `label_propagation.py` = **105 linhas** (338 com docs). Nenhum dos dois importa numpy/scipy.

Benchmark local (Python 3.14.5, networkx 3.6.1, **sem scipy**, Apple Silicon):

| Grafo | Algoritmo | Tempo | Resultado |
|---|---|---|---|
| planted partition 10 000 nós / 54 112 arestas (50 grupos) | `louvain_communities` | **0,49 s** | 50 comunidades, modularidade 0,702 |
| idem | `label_propagation_communities` | 0,36 s | 36 comunidades, mod. 0,668 |
| idem | `asyn_lpa_communities` | 0,34 s | 52 comunidades, mod. 0,708 |
| idem | `_pagerank_python` (networkx, sem scipy) | 0,25 s | — |
| idem | power iteration própria, 30 linhas | **0,09 s** | top‑20 por PageRank ∩ top‑20 por grau = 16/20 |
| Barabási–Albert 10 000 nós / 49 975 arestas | `louvain_communities` | 2,17 s | 15 comunidades |
| idem | PageRank próprio vs grau | — | top‑20 ∩ = **20/20** |

Conclusão: para ≤10k nós, Louvain em Python puro é sub‑segundo a poucos segundos; não há motivo de performance para igraph/graspologic nessa escala. O que igraph compra é Leiden garantidamente conexo e 10–100× de velocidade quando passar de ~100k nós.

### 3.3 Grau basta, ou PageRank?

- Nos dois grafos testados, o top‑20 por grau e por PageRank coincidem em 80–100 %. Para "hub = quem está no centro", grau (uma query `GROUP BY`) resolve; é o que GraphRAG usa como `rank` de entidade e o que o graphify chama de god node ("highest‑degree concepts").
- PageRank vale **não pelo hub, mas pela consulta**: a mesma power iteration com vetor de reset personalizado é o PPR do HippoRAG (§1.3). Então PageRank entra no core como algoritmo de *retrieval* (~30 linhas, 0,09 s) e o hub global sai de graça como PageRank sem personalização — ou simplesmente grau. Não são duas peças.

### 3.4 Recomendação

**Core (sem dependência nova):** PPR/power iteration própria + grau para hub + label propagation ou Louvain próprio (≤150 linhas, cacheado por `graph_version`). Critério: o algoritmo é menor que o custo de explicar a dependência, roda em <1 s na escala‑alvo, e o `leiden_communities` do networkx — a razão óbvia para depender dele — **não existe sem backend**.

**Extra `[graph]` = `igraph` (não networkx, não graspologic):** 2 MB, 1 dep, Leiden + Louvain + label propagation + `personalized_pagerank` em C; é o que HippoRAG usa. Ativa quando `nodes > ~50k` ou quando alguém pedir Leiden hierárquico. graspologic é 5 MB + scipy + sklearn + matplotlib para fazer a mesma coisa e já quebra em Python 3.13 no graphify (#290). networkx só faz sentido se o anchor quiser expor `to_networkx()` para o usuário desenhar — isso é feature de export, não de core.

---

## 4. Os números de 2026 (verificação das fontes do doc)

| Claim no doc | O que a fonte primária diz | Condição em que vale | Veredito |
|---|---|---|---|
| "GraphRAG: 86 % vs 32 % do RAG baseline" | **Não é do GraphRAG da Microsoft.** É o benchmark RobustQA da **Writer**: Writer Knowledge Graph 86,31 % vs 8 soluções de vector RAG entre 32,74 % e 75,89 % (Azure Cognitive Search+GPT‑4, Pinecone Canopy, LangChain…) — 50 000 perguntas, 32 M documentos, produto proprietário | benchmark do vendor sobre o próprio produto; o "32 %" é o *pior* concorrente | **Errado por atribuição.** O paper do GraphRAG reporta win rates LLM‑as‑judge de 72–83 % em *comprehensiveness/diversity* vs naive RAG, em 2 corpora de ~1–1,7 M tokens, e **nenhuma métrica de acurácia**. Fonte: [writer.com/engineering/rag-benchmark](https://writer.com/engineering/rag-benchmark/), [arXiv:2404.16130](https://arxiv.org/html/2404.16130v2) |
| "PathRAG: −44 % de contexto mantendo acurácia" | Tabela 6 do paper: LightRAG 16 728 tokens → PathRAG 14 438 (**−13,69 %**) → PathRAG‑lt 9 968 (**−40,41 %**); win rate médio 59,93 % vs GraphRAG e 57,09 % vs LightRAG, LLM‑as‑judge | vs LightRAG, num dataset; "‑lt" é a variante com menos caminhos | **Número inexistente no paper**; o mais próximo é −40 % da variante reduzida. [arXiv:2502.14902](https://arxiv.org/html/2502.14902) |
| "HippoRAG 2: multi‑hop 10–30× mais barato" | O "10–30× mais barato e 6–13× mais rápido" é do **HippoRAG 1** (2405.14831), na fase **online**, vs **IRCoT** (retrieval iterativo multi‑passo), não vs RAG simples. HippoRAG 2 reporta: indexação 99,5 min vs GraphRAG 277 / LightRAG 235 (≈2,4–2,8×), consulta 1,2 s vs 10,7 s (≈9×), +7 % em memória associativa vs NV‑Embed‑v2, e **só +2,3 % em QA simples (NQ)** | comparação com um baseline caro; ganho em QA simples é marginal | **Atribuição e escala erradas.** [arXiv:2405.14831](https://arxiv.org/abs/2405.14831), [arXiv:2502.14802](https://arxiv.org/html/2502.14802) |
| "LightRAG: update incremental em tempo real" | União de conjuntos + merge por nome; sem comunidade. Verdadeiro, mas trivial | — | Correto. [arXiv:2410.05779](https://arxiv.org/html/2410.05779v2) |

### 4.1 O que 2025–2026 acrescenta (e por que o A/B do doc é obrigatório)

| Fonte | Achado | Implicação para o anchor |
|---|---|---|
| **GraphRAG‑Bench / "When to use Graphs in RAG"** (ICLR'26) [arXiv:2506.05690](https://arxiv.org/html/2506.05690) | "recent studies report that GraphRAG frequently underperforms vanilla RAG on many real‑world tasks". Fact retrieval: RAG 60,92 % vs MS‑GraphRAG 49,29 % (novels). Complex reasoning: HippoRAG2 53,38 % vs RAG 42,93 %. Summarização: HippoRAG 90,95 % evidence recall vs RAG 73,38 %. MS‑GraphRAG global: 7 800 → 40 000 tokens de prompt conforme a dificuldade, até 331 375 tokens médios no corpus médico vs 1 020 do HippoRAG2 | Grafo ganha em multi‑hop e síntese, **perde em fato simples**; o modelo que ganha (HippoRAG2) é PPR sobre passagens, sem comunidades |
| **RAG vs GraphRAG** [arXiv:2502.11371](https://arxiv.org/html/2502.11371v3) | NQ: RAG 64,78 F1 vs GraphRAG‑local 63,01; HotpotQA: HippoRAG2 63,01 vs RAG 60,04; MultiHop‑RAG: KG‑GraphRAG **41,24 %** vs RAG 67,02 % vs GraphRAG‑local 69,01 %. **Integração RAG+GraphRAG: +6,4 %**; seleção por roteamento: +1,1 % | "Grafo *ou* híbrido" é a pergunta errada; o ganho está em *fundir* os dois rankings — o anchor já tem RRF |
| **Unified framework** [arXiv:2503.04338](https://arxiv.org/html/2503.04338) | Global search: 57× tempo, 210× tokens por pergunta vs VanillaRAG; RAPTOR/HippoRAG ~1 200 tokens por pergunta; **CheapRAG** (filtrar comunidades por vetor antes do LLM) corta 100× | Comunidade só é útil quando filtrada por similaridade; não use global search |
| **Unbiased evaluation** [arXiv:2506.06331](https://arxiv.org/abs/2506.06331) | Perguntas do GraphRAG são pouco ligadas ao corpus e o LLM‑juiz é enviesado; "performance gains are much more moderate than reported" | Win rates de comprehensiveness/diversity (a base do "GraphRAG ganha") não valem como acurácia |
| **BM25 Wins at Scale** (jul/2026) [arXiv:2607.26497](https://arxiv.org/abs/2607.26497) | 28 tiers de corpus (450×): "graph‑based RAG encounters construction walls before deployment scale"; BM25 lidera acima de ~10 M tokens; "lexical retrieval is the strongest scalable default" | Para corpus pequeno o grafo *pode* competir, mas só se a construção for barata |
| **Dynamic community selection** (MS Research) [blog](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/) | Podar comunidades com um LLM barato antes do map‑reduce: **−77 % de custo** sem diferença estatística de qualidade | A própria Microsoft trata global search "cheio" como desperdício |
| **LazyGraphRAG** (MS Research) [blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) | Grafo de co‑ocorrência de sintagmas nominais (NLP, sem LLM), comunidades sem resumo prévio; indexação = 0,1 % do GraphRAG (= vector RAG), consulta = 4 % do global search, qualidade comparável; "700× lower query cost" vs global | **Comunidade lazy, sem resumo, é o estado da arte da própria Microsoft** — confirma §2.2 |
| **Core‑based Hierarchies** (mar/2026) [arXiv:2603.05207](https://arxiv.org/abs/2603.05207) | Substitui Leiden por k‑core (determinístico, linear, sem aleatoriedade) + amostragem com orçamento de tokens; melhora comprehensiveness/diversity com menos tokens | Hierarquia por k‑core é ~20 linhas e determinística — alternativa a Louvain para hub/nível |
| **SPRIG / "Democratizing GraphRAG"** (fev/2026) [arXiv:2602.23372](https://arxiv.org/html/2602.23372), **LinearRAG** [arXiv:2510.10114](https://arxiv.org/abs/2510.10114) | Grafos por NER + co‑ocorrência, sem LLM, PPR na consulta; SPRIG no HotpotQA: Recall@10 0,844 (grafo+denso) vs **0,851 do RRF léxico+denso** e 0,742 BM25; LinearRAG "no extra token consumption" | Em 2026 a tendência é construir grafo **sem LLM** — e mesmo assim o RRF híbrido empata ou ganha em single‑hop |
| **GraphRAG‑Router** (abr/2026) [arXiv:2604.16401](https://arxiv.org/abs/2604.16401) | Roteia por query entre variantes de GraphRAG e LLM puro; −30 % de uso de LLM grande | Confirma que "um retriever para tudo" perde para roteamento — o `ABTestRunner` do anchor deve medir por tipo de pergunta, não só média |

**Síntese:** em corpus pequeno e perguntas de fato, híbrido+RRF empata ou ganha; o grafo compensa em multi‑hop e em "explique o que liga A a B" — exatamente `path`/`explain`. Isso torna o A/B do doc não só prudente mas necessário, e diz *como* montá‑lo: golden set estratificado (fato simples / multi‑hop / síntese), e a condição vencedora provável é **RRF(híbrido, grafo)**, não "grafo vs híbrido".

---

## Recomendações

| # | Recomendação | Confiança | Base |
|---|---|---|---|
| R1 | **Bloqueio:** incremental já na v0.2 para nós/arestas/evidência (upsert por chave natural + `valid_to`); comunidades e derivados **não persistidos** — recompute global lazy, cacheado por `graph_version`. Sem tabela `communities`. | **Alta** | §2: LightRAG/HippoRAG/Graphiti fazem exatamente isso; `graphrag update` não re‑clusteriza; LazyGraphRAG; Louvain 10k nós em 0,5–2 s sem LLM. [2410.05779](https://arxiv.org/html/2410.05779v2), [graphrag factory.py](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/workflows/factory.py), [graphiti community_operations.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/community_operations.py), [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) |
| R2 | **Consulta:** grafo `{entidade, item}` com `ContextItem` como nó (padrão HippoRAG 2), espalhamento com decaimento/limiar (PathRAG) *ou* PPR (30 linhas), score por item → `TokenBudget`. Sem LLM na consulta. PathRAG "casa" pelo pruning; descartar o pareamento O(N²) e o path‑prompting. | **Alta** | §1.3; [2502.14902](https://arxiv.org/html/2502.14902) (`bfs_weighted_paths`), [2502.14802](https://arxiv.org/html/2502.14802), GraphRAG‑Bench (HippoRAG2 vence com 1 020 tokens) |
| R3 | **Não implementar global search / community reports por LLM.** Comunidade é rótulo para `explain`/hubs. | **Alta** | 57×/210× ([2503.04338](https://arxiv.org/html/2503.04338)), −77 % com poda ([MS blog](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)), 331 K tokens ([2506.05690](https://arxiv.org/html/2506.05690)) |
| R4 | **Dependência:** core sem lib de grafo (PPR + grau + Louvain/label‑propagation próprios, ≤150 linhas, com teste de modularidade contra o networkx no CI de dev). Extra `[graph]` = **igraph** (2 MB, 1 dep) para Leiden/escala, não networkx (Leiden sem backend) nem graspologic (17 deps). | **Média‑alta** | §3.1–3.2: medições locais; [networkx leiden.py](https://github.com/networkx/networkx/blob/main/networkx/algorithms/community/leiden.py) `implemented_by_nx=False`; PyPI. Média porque "próprio vs networkx" é gosto: networkx tem 0 deps e 2 MB — se Arthur preferir, `[graph]=networkx` também é defensável; o que **não** é defensável é graspologic |
| R5 | **Hub = grau** (query SQL) na v0.2; PageRank global só se já existir PPR para a consulta (mesma função). | **Alta** | §3.3: 16/20 e 20/20 de sobreposição; GraphRAG usa grau como `rank`; graphify usa grau |
| R6 | **Corrigir o doc:** 86 %/32 % é Writer RobustQA; PathRAG é −13,7 %/−40,4 % vs LightRAG; 10–30× é HippoRAG 1 vs IRCoT, online. | **Alta** | §4 |
| R7 | **A/B:** golden set estratificado (fato / multi‑hop / síntese‑explicação); condições: híbrido+RRF, grafo só, **RRF(híbrido, grafo)**. Critério de investir no construtor LLM = ganho em multi‑hop/explain, não na média. | **Alta** | [2502.11371](https://arxiv.org/html/2502.11371v3) (+6,4 % integração), [2506.05690](https://arxiv.org/html/2506.05690), [2607.26497](https://arxiv.org/abs/2607.26497) |
| R8 | **Regra de merge de descrição** copiada do LightRAG: lista de fragmentos por `item_id`, resumo LLM só com ≥ 8 fragmentos ou estouro de tokens. | **Média** | [`constants.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py) `DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8`; o limiar certo para o anchor é empírico |
| R9 | Se quiser hierarquia determinística sem aleatoriedade (testes reproduzíveis), k‑core (2026) em vez de Louvain. | **Baixa** | [2603.05207](https://arxiv.org/abs/2603.05207) — só abstract verificado, sem números |

### Reprodução das medições

```
# tamanhos: curl https://pypi.org/pypi/<pkg>/json  (2026-09-02)
# linhas: pip download networkx==3.6.1 --no-deps; unzip; ast-strip docstrings
# benchmark: python3.14, networkx 3.6.1 sem scipy, planted_partition_graph(50, 200, 0.04, 0.0003) e barabasi_albert_graph(10000, 5)
```

---

# Parte 2 — Modelo de aresta e fontes de extração


**Para:** anchor v0.2 · frente #4 (grafo de conhecimento sobre memória e documentos)
**Entrada:** `docs/plans/2026-08-28-v020-4-grafo-de-conhecimento.md` (perguntas 2 e 3 + bloqueio "aresta tem tipo?"), `docs/plans/2026-08-28-v020-roadmap.md` § Contexto herdado
**Método:** fontes primárias (código nos repos oficiais lido via `gh api`, papers em arXiv, docs oficiais, e o pacote `graphifyy 0.9.39` instalado em `~/.local/share/uv/tools/graphifyy/...`). Nada respondido de memória; cada afirmação de design tem URL ou caminho local.

**Aviso importante sobre o mem0:** a graph memory do mem0 (`mem0/memory/graph_memory.py`, `mem0/graphs/**`) **foi removida do repositório** no PR #4805 (2026-04-14) — confirmado em [issue #6591](https://github.com/mem0ai/mem0/issues/6591). Todo o código citado abaixo foi lido no último SHA que ainda a continha: [`e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/memory/graph_memory.py). O paper continua válido como referência de design, mas o "mem0^g" não é mais um sistema mantido para copiar.

---

## O que o anchor já tem (linha de base para a escada)

- `GraphStore` (`src/anchor/protocols/storage.py:384`) já modela **`add_edge(source, relation: str, target, metadata)`** e `get_neighbors(..., relation_filter)`. A aresta já é tipada por string.
- SQLite (`src/anchor/storage/sqlite/_schema.py:60`): `graph_edges(id, source, relation, target, metadata_json, UNIQUE(source, relation, target))`. Postgres espelha.
- `SimpleGraphMemory` (`src/anchor/memory/graph_memory.py`) guarda triplas `(source, relation, target)`.

Conclusão de escada (degrau 2): **o "tipo" já existe no schema**. A decisão real não é "adicionar tipo", é "o que colocar na coluna `relation` e se ela é validada contra um enum".

---

## 1. Tipo de aresta

### 1.1 Como cada sistema modela

| Sistema | Tipo fechado? | Rótulo | Outros campos da aresta | Fonte |
|---|---|---|---|---|
| **Microsoft GraphRAG** | Não | Nenhum rótulo curto; `relationship_description` (frase livre) + `relationship_strength` (numérico) | `weight` (= strength, **somado** ao mesclar duplicatas por `(source,target)`), `description` (lista de descrições), `text_unit_ids`, `rank`, `attributes` | [prompt `extract_graph.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/prompts/index/extract_graph.py) · [`relationship.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/data_model/relationship.py) · [`_merge_relationships`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/operations/extract_graph/extract_graph.py) |
| **LightRAG** | Não | `relationship_keywords` (lista livre separada por vírgula) + `relationship_description` | `weight` (somado), `source_id` (chunk ids unidos por `<SEP>`), `file_path`, `timestamp` | [`prompt.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/prompt.py) · [`operate.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py) · [`constants.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py) |
| **Graphiti / Zep** | **Opcional** | `name` = `relation_type` em SCREAMING_SNAKE_CASE derivado do predicado (livre); se o usuário declarar `edge_types` (Pydantic) e `edge_type_map`, o LLM classifica; sem match → `RELATES_TO` | `fact` (frase), `fact_embedding`, `episodes: list[str]`, `valid_at`, `invalid_at`, `created_at`, `expired_at`, `attributes` | [`edges.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py) · [`extract_edges.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py) · [docs custom edge types](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types) |
| **mem0^g** (removido) | Não | `relationship` livre, normalizado `lower().replace(" ", "_")` e usado **como TYPE Cypher** (`MERGE (s)-[r:{relationship}]->(d)`) | `valid` (bool), `invalidated_at`; nós têm `embedding`, `created`, `mentions` | [`graph_memory.py@e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/memory/graph_memory.py) · [`memory/utils.py@e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/memory/utils.py) (`remove_spaces_from_entities`) |
| **HippoRAG 2** | Não | Predicado OpenIE livre (`["Radio City", "located in", "India"]`) | No grafo, a aresta vira peso = nº de fatos; `edge_kind` ∈ {fact, synonym, fact+synonym}, `synonym_score`, `fact_source_counts` | [`triple_extraction.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/prompts/templates/triple_extraction.py) · [`HippoRAG.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py) |
| **graphify 0.9.39** | **Vocabulário sugerido, não validado** | `relation` obrigatório; o prompt lista `calls\|implements\|references\|cites\|conceptually_related_to\|shares_data_with\|semantically_similar_to\|rationale_for`, mas `validate.py` só checa presença do campo, não o valor; extratores AST emitem outros (`imports`, `contains`, `mixes_in`, `requires_env`) | `confidence` ∈ {EXTRACTED, INFERRED, AMBIGUOUS} (**validado**), `confidence_score`, `source_file`, `source_location`, `weight` | local: `graphify/llm.py:452-478`, `graphify/validate.py:5-8`, `graphify/mcp_ingest.py:195-264`; `~/.claude/skills/graphify/references/extraction-spec.md` |

**Resposta à pergunta central:** nenhum dos sistemas de extração por LLM usa enum fechado como único caminho. O padrão dominante é **rótulo livre em texto normalizado** (mem0, Graphiti, HippoRAG, graphify na prática), e o único que oferece tipos fechados (Graphiti) os torna opt-in com fallback genérico `RELATES_TO`. GraphRAG e LightRAG nem têm rótulo curto — só descrição e peso — e compensam com a descrição embutida.

### 1.2 Custo real de pedir o predicado

O doc diz que tipo "dobra o custo de extração". **Não confirma.** Em todos os prompts reais, o predicado/descrição sai **na mesma chamada** que a tripla:

- GraphRAG: `("relationship"<|>src<|>tgt<|>relationship_description<|>relationship_strength)` — a descrição é uma frase (~20–40 tokens); um rótulo de 2–5 tokens seria uma fração disso ([prompt](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/prompts/index/extract_graph.py)).
- LightRAG: `relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description` — keywords + descrição na mesma linha ([prompt](https://github.com/HKUDS/LightRAG/blob/main/lightrag/prompt.py)).
- Graphiti: `Edge{source_entity_name, target_entity_name, relation_type, fact, valid_at, invalid_at, episode_indices}` — um único structured output para tudo ([extract_edges.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py)).
- mem0: a tool `establish_relations` devolve `{source, relationship, destination}` — a relação **é** o payload da chamada; a chamada existe com ou sem rótulo ([tools.py@e44b46e](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/graphs/tools.py)).

O que de fato dobra custo é **gleaning** (2ª passada "faltou algo?"): a continuação reenvia o histórico inteiro (prompt + chunk + saída anterior) — foi exatamente o bug/fix do GraphRAG [issue #615](https://github.com/microsoft/graphrag/issues/615) / [PR #734](https://github.com/microsoft/graphrag/pull/734). Default `max_gleanings = 1` no GraphRAG ([defaults.py](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py)) e `DEFAULT_MAX_GLEANING = 1` no LightRAG ([constants.py](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py)). Ou seja: **rótulo custa ~0; gleaning custa ~2×**. São decisões independentes.

### 1.3 O que `path()`/`explain()` precisam

- graphify `path` imprime `{label} --{relation}--> [{confidence}]` e `explain` imprime `--{rel}--> {neighbor} [{conf}] ({src_file})` — só rótulo + tag de proveniência + arquivo-fonte (`~/.claude/skills/graphify/references/query.md`).
- Graphiti devolve o `fact` (frase) e `valid_at/invalid_at` na busca; é a frase que explica ([paper Zep §retrieval](https://arxiv.org/abs/2501.13956)).

Um enum não melhora nenhuma dessas saídas: elas renderizam texto. O enum só serviria para `relation_filter` — e o `GraphStore` já filtra por string. Um vocabulário sugerido no prompt (como o graphify faz) dá a consistência que o filtro precisa sem custo de validação.

### 1.4 Recomendação (bloqueio)

**Rótulo livre, normalizado (`casefold`, espaços → `_`), + `fact` opcional + evidência. Sem enum.** Confiança: alta.

- Wikilink → `relation="links_to"` (determinístico). Embedding → `"similar_to"`. Estrutura pai/filho já existente → `"contains"`. LLM → predicado normalizado (regra do mem0: "consistent, general, and timeless… prefer 'professor' over 'became_professor'" — [utils.py@e44b46e](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/graphs/utils.py)).
- Vocabulário sugerido no prompt (5–8 rótulos: `depends_on`, `part_of`, `uses`, `defines`, `related_to`…) para conter a dispersão, **não validado** — igual ao graphify.
- Se um dia precisar de tipos fechados por domínio, o caminho do Graphiti (`edge_types` Pydantic opcionais com fallback) encaixa por cima sem migração, porque a coluna já é string.

---

## 2. Proveniência

### 2.1 Modelo do graphify (confirmado no código)

- `confidence` ∈ `{"EXTRACTED", "INFERRED", "AMBIGUOUS"}` — **validado** em `graphify/validate.py:5` (`VALID_CONFIDENCES`) e obrigatório em toda aresta (`REQUIRED_EDGE_FIELDS = {source, target, relation, confidence, source_file}`).
- `confidence_score`: EXTRACTED sempre 1.0; INFERRED em **rubrica discreta** 0.95/0.85/0.75/0.65/0.55; AMBIGUOUS 0.1–0.3. A justificativa está no próprio prompt: "the bimodal distribution observed in production (>50% at 0.5, >40% at 0.85+) shows the range guidance is being collapsed to a binary" (`~/.claude/skills/graphify/references/extraction-spec.md`). Default de export quando ausente: `{EXTRACTED: 1.0, INFERRED: 0.5, AMBIGUOUS: 0.2}` (`graphify/export.py:159`).
- `source_file` verbatim é o que permite "replace-on-re-extract" no `--update` incremental (extraction-spec, regra `source_file RULE`).
- Extratores determinísticos (AST, manifests, MCP) emitem sempre `EXTRACTED` (`cargo_introspect.py:9`, `manifest_ingest.py:103`, `ruby_resolution.py:134`); o builder rebaixa recuperação por alias para `INFERRED` (`build.py:198-206`).
- Auditoria: recurso MCP `graphify://audit` com % por tag (`serve.py:1891`).

### 2.2 Comparação

| Sistema | Proveniência por aresta | Evidência | Confiança |
|---|---|---|---|
| graphify | `confidence` (3 tags) | `source_file` + `source_location` | `confidence_score` (rubrica) |
| Graphiti | — (tudo é "extraído" de episódio) | `episodes: list[uuid]` (**cresce** quando o mesmo fato reaparece: `resolved.episodes.append(episode.uuid)` em [`edge_operations.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py)) | — |
| GraphRAG | — | `text_unit_ids: list[str]` | `weight` = soma dos strengths (paper: "the number of duplicates for a given relationship becomes edge weights" — [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)) |
| LightRAG | — | `source_id` (`<SEP>`-joined) + `file_path` | `weight` somado |
| mem0^g | — | — (só `mentions` no nó) | — |
| HippoRAG 2 | `edge_kind` (fact / synonym / fact+synonym) | `fact_source_counts` | `synonym_score` |

### 2.3 Mínimo viável

`provenance: Literal["extracted","inferred","ambiguous"]` + `evidence: list[item_id]` + `confidence: float` — **sim, e é o que o anchor precisa por razões próprias**, não só por cópia:

1. **`evidence` é o mecanismo do escopo.** A decisão "nó sem evidência sobrevivente ao escopo não existe" e "nó excluído é parede" (doc §Decisões) exige saber quais `item_id` sustentam cada aresta. Sem lista de evidência, `GraphScope` não tem como filtrar.
2. **`evidence` é o mecanismo do incremental.** Re-ingerir um documento = invalidar as arestas cuja única evidência era aquele item (é o "replace-on-re-extract" do graphify, em versão temporal).
3. **`provenance` é o que `explain()` mostra** para o usuário distinguir "o documento diz" de "o modelo achou".

`confidence` como float separado do enum: manter os dois (como graphify). A tag responde "de onde veio", o número responde "quanto acreditar", e a rubrica discreta evita o colapso bimodal observado.

### 2.4 Grafo sem proveniência "apodrece" — há evidência?

Há, com nuance. A evidência publicada aponta menos para *alucinação* e mais para **drift e duplicação**:

- [**Toward Robust GraphRAG** (arXiv 2603.14828)](https://arxiv.org/abs/2603.14828): "spurious noise causes retrieval drift toward plausible but unsupported triples" e informação incompleta "leads to retrieval hallucination by forcing continuation through under-supported graph structure". Remédio proposto: verificação de suficiência e *fallback* para o texto — ou seja, precisar da evidência textual atrás da aresta.
- [**KGGen** (arXiv 2502.09956)](https://arxiv.org/abs/2502.09956): "automatically extracted KGs are of questionable quality"; a contribuição é *clustering* de entidades para reduzir esparsidade — o problema é duplicata, não invenção.
- [**Are LLMs Effective KG Constructors?** (arXiv 2510.11297)](https://arxiv.org/abs/2510.11297): LLMs "can generally produce relevant and document-faithful triples with limited hallucination", mas com "substantially different extraction behaviors across the construction stages".
- O próprio GraphRAG admite: "our analysis uses exact string matching for entity matching" ([arXiv 2404.16130](https://arxiv.org/abs/2404.16130)) — e o graphify documenta a classe de bug "ID drift" (#811, #550, #1033, #1104) em `graphify/ids.py` como motivo de centralizar normalização.

Leitura: o que apodrece é (a) entidades duplicadas que fragmentam o grafo e (b) arestas inferidas indistinguíveis das explícitas, que o traversal segue como se fossem fato. Proveniência + evidência resolvem (b) e dão o material para corrigir (a).

---

## 3. Validade temporal

### 3.1 Os dois modelos

**Graphiti (bi-temporal):** "t′created and t′expired ∈ T′ monitor when facts are created or invalidated in the system, while tvalid and tinvalid ∈ T track the temporal range during which facts held true" ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)). No código ([`edges.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py)): `created_at`, `expired_at`, `valid_at`, `invalid_at`. `valid_at/invalid_at` são extraídos pelo LLM a partir do texto usando `REFERENCE_TIME` (regras em [`extract_edges.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py): "If the fact is ongoing (present tense), set valid_at to the timestamp of the episode… Leave both fields null if no explicit or resolvable time is stated"). Invalidação por contradição em `resolve_edge_contradictions` ([`edge_operations.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py)):

```python
elif edge_valid_at < resolved_edge_valid_at:
    edge.invalid_at = resolved_edge.valid_at
    edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
```

Só invalida se **ambas** as datas de validade forem conhecidas e a antiga for anterior; o LLM (`dedupe_edges.resolve_edge`) é quem aponta `contradicted_facts`.

**mem0^g (invalidação simples):** paper: "An LLM-based update resolver determines if certain relationships should be obsolete, marking them as invalid rather than physically removing them to enable temporal reasoning" ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413)). Código ([`graph_memory.py@e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/memory/graph_memory.py) `_delete_entities`):

```cypher
WHERE r.valid IS NULL OR r.valid = true
SET r.valid = false, r.invalidated_at = datetime()
```

e toda busca filtra `WHERE r.valid IS NULL OR r.valid = true`. Uma linha do tempo só (transação); a decisão de obsolescência vem do prompt `DELETE_RELATIONS_SYSTEM_PROMPT` ("DO NOT DELETE if there is a possibility of same type of relationship but different destination nodes").

### 3.2 Para o anchor

A pergunta é "basta `valid_from`/`valid_to` + `created_at`, ou bi-temporal desde já?". Observações:

- A decisão "marcada inválida, não deletada" **já exige** um timestamp de invalidação (`invalidated_at`, como o mem0). Sem ele não se sabe *quando* o sistema desistiu da aresta.
- `valid_from`/`valid_to` são outra coisa: o intervalo em que o fato *vale no mundo*, extraído do texto (Graphiti) ou herdado do item (memória tem timestamp; documento tem data de ingestão).
- A diferença de custo entre "3 campos" e "4 campos" é **uma coluna nullable**. A diferença de capacidade é poder responder "o que o grafo sabia em T" (replay/debug/A-B) separadamente de "o que era verdade em T".

**Recomendação:** os 4 campos desde já, mas com preenchimento assimétrico — confiança: alta.

| Campo | Quem preenche | Default |
|---|---|---|
| `created_at` | sistema, sempre | `now()` |
| `invalidated_at` | sistema, ao invalidar (contradição, re-ingestão sem o link, exclusão do item) | `None` |
| `valid_from` | extração (LLM com `reference_time`; wikilink/estrutura = timestamp do item) | `None` = "desde sempre" |
| `valid_to` | extração quando o texto diz que acabou; ou = `valid_from` da aresta que a contradiz (regra Graphiti) | `None` = "ainda vale" |

**Quem consome:** toda travessia (`get_neighbors`, `path`, `backlinks`, `explain`) recebe `as_of: datetime | None = None` e aplica o predicado

```sql
invalidated_at IS NULL
AND (valid_from IS NULL OR valid_from <= :as_of)
AND (valid_to   IS NULL OR valid_to   >  :as_of)
```

com `as_of = now()` por default. Índice em `(source, invalidated_at)` e `(target, invalidated_at)`. Detalhe de schema: o `UNIQUE(source, relation, target)` atual impede re-afirmar uma aresta invalidada; trocar por **índice único parcial** `WHERE invalidated_at IS NULL` (SQLite e Postgres suportam).

O que **não** copiar do Graphiti agora: a chamada extra ao LLM para `contradicted_facts` em toda inserção. Para v0.2, invalidação vem de (1) re-ingestão do item (determinística) e (2) uma passada opcional de contradição no extrator LLM, opt-in.

---

## 4. Fontes de extração e ordem

### 4.1 Wikilinks (formato Obsidian) — gramática completa

Fonte oficial: [obsidian.md/help/links](https://obsidian.md/help/links) e [obsidian.md/help/embeds](https://obsidian.md/help/embeds).

| Forma | Exemplo | Nota |
|---|---|---|
| Nota | `[[Three laws of motion]]` ou `[[Three laws of motion.md]]` | extensão opcional |
| Alias | `[[Example\|Custom name]]` | "Use a vertical bar (\|) to change the display text" |
| Heading na mesma nota | `[[#Preview a linked file]]` | file vazio |
| Heading em outra nota | `[[About Obsidian#Links are first-class citizens]]` | |
| Sub-heading | `[[Help and support#Questions and advice#Report bugs…]]` | `#` encadeado |
| Bloco | `[[2023-01-01#^37066d]]` / `[[2023-01-01#^quote-of-the-day]]` | "Block identifiers can only consist of Latin letters, numbers, and dashes" |
| Caminho | `[[Projects/Three laws of motion]]` | "Folder paths start at the vault root and use forward slashes (/), even on Windows" |
| Embed | `![[Figure 1.png]]`, `![[Note#Heading]]`, `![[Note#^b15695]]`, `![[Engelbart.jpg\|100x145]]`, `![[Document.pdf#page=3]]` | "Prefixing an internal link with an exclamation mark (!) allows you to embed" |
| Alvo inexistente | — | "If the link points to a note that doesn't exist yet, Obsidian creates the note at that folder path" |

**Resolução nome → arquivo.** O Obsidian não documenta o algoritmo; a API oficial expõe `MetadataCache.getFirstLinkpathDest(linkpath, sourcePath): TFile | null` — "Get the best match for a linkpath" ([docs.obsidian.md](https://docs.obsidian.md/Reference/TypeScript+API/MetadataCache/getFirstLinkpathDest)). O comportamento observável:

- **Case-insensitive**: `[[Log]]`, `[[log]]`, `[[lOg]]`, `[[LOG]]` resolvem para o mesmo arquivo (confirmado em macOS e Windows no [fórum](https://forum.obsidian.md/t/case-sensitivity/52331)).
- **Basename único em qualquer pasta resolve; caminho só desambigua.** A opção "Shortest path when possible" é sobre como *novos* links são escritos: "will use only the note name, unless there's two notes with the same names. Then it uses as little of the file path as possible" ([fórum, ryanjamurphy](https://forum.obsidian.md/t/settings-new-link-format-what-is-shortest-path-when-possible/6748)). Isso implica que a resolução aceita basename vault-wide.
- Parser de referência (obsidian-export, Rust): [`references.rs`](https://github.com/zoni/obsidian-export/blob/main/src/references.rs) divide em `file` / `#section` / `|label` (bloco `^id` cai em `section`); [`lib.rs` `lookup_filename_in_vault`](https://github.com/zoni/obsidian-export/blob/main/src/lib.rs) normaliza **NFC**, compara `ends_with` no caminho, com e sem `.md`, exato e `to_lowercase()`. Testes: `[[NESTED/notea]]` → `nested/NoteA.md`, `[[Note.1]]` → `Note.1.md`, `Note\u{61}\u{308}` → `Noteä.md`.
- graphify: regex `(?<!\!)\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]` (`graphify/extractors/markdown.py:15`) — exclui embeds, descarta alias e âncora, e resolve **só como irmão relativo** (`source_dir / target + ".md"`), não vault-wide. É uma limitação em relação ao Obsidian; não copiar essa parte.

**Gramática recomendada para o anchor (stdlib `re`):**

```python
WIKILINK = re.compile(r"(?P<embed>!)?\[\[(?P<target>[^\]|#]*)(?:#(?P<anchor>[^\]|]*))?(?:\|(?P<alias>[^\]]*))?\]\]")
```

Resolução: chave = `unicodedata.normalize("NFC", target).casefold()`, `.md` removido; índice do vault por basename casefolded → lista de caminhos; se `target` contém `/`, casar por sufixo de caminho; um único candidato → EXTRACTED 1.0; vários → escolher o da mesma pasta do documento-fonte se houver, senão marcar **AMBIGUOUS** com todos os candidatos em `metadata`; nenhum → nó placeholder `unresolved=True` (o Obsidian trata link pendente como nota futura). `anchor` (heading/bloco) e `alias` vão para `metadata`; o alias alimenta a tabela de aliases da §5. Embeds (`!`) são links com `metadata.embed=True` — a relação semântica é a mesma.

### 4.2 Extração por LLM — prompts reais

| Sistema | Chamadas por chunk | Formato de saída | Gleaning |
|---|---|---|---|
| **GraphRAG** | 1 (+1 por gleaning) | Texto delimitado: `("entity"<\|>NAME<\|>TYPE<\|>desc)` `##` `("relationship"<\|>SRC<\|>TGT<\|>desc<\|>strength)` … `<\|COMPLETE\|>`. Tipos de **entidade** são fechados (`{entity_types}`); relação não tem tipo | `CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction…"` + `LOOP_PROMPT` ("Answer Y … or N"); paper: logit bias 100 para forçar Y/N; `max_gleanings=1` |
| **LightRAG** | 1 (+1 por gleaning) | `entity<\|#\|>name<\|#\|>type<\|#\|>desc` / `relation<\|#\|>src<\|#\|>tgt<\|#\|>keywords<\|#\|>desc`, `<\|COMPLETE\|>`; alternativa JSON `{name,type,description}` / `{source,target,keywords,description}` | `entity_continue_extraction_user_prompt`: "identify and extract any missed or incorrectly formatted entities and relationships… do not re-output…"; `DEFAULT_MAX_GLEANING=1` |
| **mem0^g** | 2 no `add` (`extract_entities` → `establish_relations`) + 1 de invalidação (`delete_graph_memory`/`noop`) + buscas por embedding | **Tool calls** JSON: `extract_entities{entities:[{entity, entity_type}]}`, `establish_relations{entities:[{source, relationship, destination}]}`; relação normalizada `lower().replace(" ","_")` | Não |
| **Graphiti** | 1 para nós, 1 para arestas (+ dedup/contradição/timestamps/atributos, cada uma sua chamada) | **Structured output** Pydantic `ExtractedEdges{edges:[{source_entity_name, target_entity_name, relation_type, fact, valid_at, invalid_at, episode_indices}]}` | Opcional (`reflexion`) |
| **HippoRAG 2** | 2 (NER → triplas) | JSON: `{"named_entities":[…]}` depois `{"triples":[[s,p,o],…]}`, one-shot | Não |
| **graphify** | 1 por chunk (subagente) | JSON único `{nodes, edges, hyperedges}` com `confidence` + `confidence_score` + `source_file` | `--mode deep` (mais INFERRED), não gleaning |

**Gleaning vale o custo?** O paper do GraphRAG justifica como forma de usar chunks maiores (2400 tokens) sem perder recall ([arXiv 2404.16130](https://arxiv.org/abs/2404.16130)); o custo é ~1 chamada extra com todo o histórico. O anchor já chunka pequeno (`ParentChildChunker`), onde a perda de recall por chunk é menor. **Recomendação:** `max_gleanings=0` por default, parâmetro exposto; medir no golden set antes de ligar.

**Formato para o anchor:** structured output JSON (uma chamada, sem delimitadores frágeis), no espírito do Graphiti/graphify:

```json
{"entities":[{"name":"…","type":"…"}],
 "edges":[{"source":"…","relation":"snake_case","target":"…","fact":"frase",
           "provenance":"extracted|inferred|ambiguous","confidence":0.85,
           "valid_from":null,"valid_to":null}]}
```

com a rubrica discreta de confiança do graphify e a regra de nomes do mem0 no prompt.

### 4.3 Similaridade de embedding como aresta

- **HippoRAG 2** é quem usa: "synonym edges between phrase pairs within the KG, detecting those with vector similarity above a predefined threshold" — threshold **0.8** (paper Tabela 13 e `synonymy_edge_sim_threshold=0.8`, `synonymy_edge_topk=2047` em [`config_utils.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/utils/config_utils.py)). Detalhes que importam: os nós são **frases curtas (nomes de entidade)**, não passagens; a aresta recebe `edge_kind="synonym"` e `weight = max(contagem de fatos, synonym_score)` (`HippoRAG.py:683-689`); no incremental só compara nós novos contra todos (`HippoRAG.py:1305`). O paper afirma que a abordagem "introduce[s] less LLM-generated noise", não que sinônimos por cosseno sejam limpos.
- **Graphiti** e **mem0** usam embedding para **candidatos de dedup**, não como aresta: Graphiti `NODE_DEDUP_COSINE_MIN_SCORE = 0.6` seguido de LLM ([`node_operations.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/node_operations.py)); mem0 `threshold` 0.7 default (0.9 na busca de nó fonte/destino) para reaproveitar nó antes do `MERGE`.
- **graphify** não usa embedding: `semantically_similar_to` é julgado pelo LLM, INFERRED 0.6–0.95, "only when the similarity is genuinely non-obvious and cross-cutting" (extraction-spec).

**Ruído:** entre *nomes de entidade* curtos, cosseno ≥ 0.8 é razoavelmente preciso (é o regime do HippoRAG). Entre *ContextItems* inteiros, cosseno mede tópico, não relação — e o anchor já tem isso no vector store; duplicar no grafo faz `path()` atravessar por afinidade temática, que é exatamente o vazamento que "nó excluído é parede" quer evitar.

### 4.4 Ordem recomendada

1. **Wikilinks — sempre** (grátis, determinístico, `extracted`, 1.0, `links_to`). Inclui embeds e aliases.
2. **Estrutura já existente — sempre**: pai→filho do `ParentChildChunker` como `contains`; item→entidades declaradas em frontmatter/tags se houver. Também `extracted` 1.0.
3. **LLM — opt-in por fonte**: ligado por default para **memória** (não tem wikilink) e desligado para documentos que já têm ≥1 wikilink; `max_gleanings=0`. Saída com `provenance` e rubrica de confiança. Uma chamada por chunk.
4. **Embedding — opt-in, só entre nomes de entidade**: `similar_to`, `inferred`, `confidence = cosseno`, threshold 0.8, top-k pequeno, incremental (novos × todos). Nunca item×item.

---

## 5. Resolução de entidade

| Sistema | Normalização de string | Fuzzy/embedding | LLM decide? | Fonte |
|---|---|---|---|---|
| **GraphRAG** | `clean_str(name.upper())` (unescape HTML, strip, remove controle); merge exato por `(title, type)` | Não | Não. Paper: "exact string matching for entity matching" | [`graph_extractor.py:146-158`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/index/operations/extract_graph/graph_extractor.py) |
| **LightRAG** | `normalize_entity_name` → `sanitize_and_normalize_extracted_text` (HTML, aspas, CJK, largura); **não** dobra caixa | Não | Não | [`utils.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/utils.py) |
| **Graphiti** | `_normalize_string_exact` = `lower()` + colapsar espaços | (1) MinHash/LSH sobre 3-gramas, Jaccard ≥ **0.9**, com *entropy gate* (nome ≥ 6 chars ou ≥ 2 tokens, entropia ≥ 1.5); (2) candidatos por cosseno ≥ **0.6** + full-text | Sim, para o que sobrar: prompt `dedupe_nodes` com `duplicate_candidate_id = -1` quando incerto; "NEVER fabricate entity names or mark distinct entities as duplicates" | [`dedup_helpers.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/dedup_helpers.py) · [`dedupe_nodes.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/dedupe_nodes.py) |
| **mem0^g** | `lower().replace(" ", "_")` | Cosseno ≥ 0.7 (config) / 0.9 (busca de nó) para reaproveitar nó existente | Não (o threshold decide) | [`graph_memory.py@e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/memory/graph_memory.py) |
| **HippoRAG 2** | Nenhuma (nós = frases como extraídas) | Não dedupa; cria aresta de sinonímia ≥ 0.8 | Não | [`HippoRAG.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/HippoRAG.py) |
| **graphify** | `normalize_id`: NFKC → não-`\w` vira `_` → colapsa `_` → `casefold` (`ids.py`) | `dedup.py`: exato → entropia ≥ 2.5 → MinHash LSH 0.7 → Jaro-Winkler ≥ 92 → boost de mesma comunidade → union-find | Não | local `graphify/ids.py`, `graphify/dedup.py:244-246` |

**"auth" / "Auth" / "autenticação" no mesmo nó:**

- `auth` ↔ `Auth`: normalização de string resolve (todos os sistemas que dobram caixa).
- `auth` ↔ `autenticação`: nenhuma normalização de string resolve; Jaro-Winkler/Jaccard também não (strings diferentes). Resolve-se por (a) **alias explícito** — `[[auth|autenticação]]` já entrega o par de graça; (b) embedding de nome + confirmação (Graphiti); (c) LLM que já vê a lista de entidades existentes (mem0 "List of entities: […]").

**Mínimo viável para o anchor** (stdlib, sem dependência nova) — confiança: alta:

1. **Chave canônica** = `normalize_id` do graphify (NFKC + casefold + não-alfanumérico → `_`), 4 linhas de `unicodedata` + `re`. Guardar `label` original no nó.
2. **Tabela de aliases** `graph_aliases(alias_key, node_id)` alimentada por (i) alias de wikilink, (ii) frontmatter `aliases:` se existir, (iii) `same_as` confirmado. Resolução de qualquer nome passa por ela antes de criar nó.
3. **Fuzzy só com gate**: Jaro-Winkler não está na stdlib; `difflib.SequenceMatcher` está. Se usar, copiar o *entropy gate* (nomes curtos nunca casam por fuzzy — é a lição explícita do Graphiti e do graphify).
4. **Embedding ≥ 0.8 entre nomes** gera aresta `same_as`/`similar_to` **inferred**, não merge automático. Merge só com confirmação (LLM opt-in ou humano). mem0 mergeava por threshold puro; Graphiti não — seguir o Graphiti.
5. Passar a lista de entidades já conhecidas no prompt do extrator LLM (custa tokens de input, evita a maior parte das duplicatas na origem).

---

## Recomendações (com confiança e URLs)

| # | Recomendação | Confiança | Evidência principal |
|---|---|---|---|
| R1 | **Tipo = rótulo livre normalizado, sem enum**; `fact` opcional; vocabulário sugerido no prompt, não validado | Alta | Graphiti [`extract_edges.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py) + [docs edge types](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types); mem0 [`utils.py@e44b46e`](https://github.com/mem0ai/mem0/blob/e44b46ef2ea559953190d0bfb5a58bf10ad799bb/mem0/graphs/utils.py); graphify `validate.py:5-8` |
| R2 | Rótulo **não dobra custo**; gleaning sim (~2×). Default `max_gleanings=0` | Alta | [GraphRAG #615](https://github.com/microsoft/graphrag/issues/615) / [PR #734](https://github.com/microsoft/graphrag/pull/734); [`defaults.py`](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py); [LightRAG `constants.py`](https://github.com/HKUDS/LightRAG/blob/main/lightrag/constants.py) |
| R3 | Proveniência = `provenance` (3 tags) + `confidence` (rubrica discreta) + `evidence: list[item_id]` | Alta | graphify `validate.py`, `extraction-spec.md` (rubrica); GraphRAG `text_unit_ids`; Graphiti `episodes`; [arXiv 2603.14828](https://arxiv.org/abs/2603.14828) |
| R4 | Bi-temporal completo (4 timestamps), preenchimento assimétrico; `as_of` em toda travessia; índice único parcial `WHERE invalidated_at IS NULL` | Alta | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956); Graphiti [`edge_operations.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py); mem0 `_delete_entities` |
| R5 | Ordem: wikilinks → estrutura → LLM opt-in (default on p/ memória) → embedding só entre nomes de entidade, ≥ 0.8, `inferred` | Alta | [obsidian.md/help/links](https://obsidian.md/help/links); HippoRAG [`config_utils.py`](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/utils/config_utils.py) |
| R6 | Wikilink: regex com embed/anchor/alias; resolução NFC + casefold, basename vault-wide, caminho desambigua, ambíguo → `ambiguous` | Alta | [obsidian-export `lib.rs`](https://github.com/zoni/obsidian-export/blob/main/src/lib.rs); [fórum case-insensitivity](https://forum.obsidian.md/t/case-sensitivity/52331); [`getFirstLinkpathDest`](https://docs.obsidian.md/Reference/TypeScript+API/MetadataCache/getFirstLinkpathDest) |
| R7 | Entidade: chave canônica NFKC+casefold, tabela de aliases, embedding só sugere (`same_as` inferred), merge só confirmado | Alta | graphify `ids.py`; Graphiti [`dedup_helpers.py`](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/dedup_helpers.py) |
| R8 | Não copiar do Graphiti agora: chamada LLM de contradição a cada inserção; não copiar do graphify: resolução de wikilink só-irmão | Média | `graphify/extractors/markdown.py:19-53` |

### Schema mínimo de aresta (pseudo-Pydantic)

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class Edge(BaseModel):
    source: str                                  # node_id canônico (normalize_id)
    target: str
    relation: str                                # rótulo livre, casefold + "_"; "links_to" | "contains" | "similar_to" | <predicado LLM>
    fact: str | None = None                      # frase-evidência p/ explain(); LLM preenche, wikilink não

    provenance: Literal["extracted", "inferred", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)    # extracted=1.0; inferred ∈ {.95,.85,.75,.65,.55}; ambiguous ≤ .3
    evidence: list[str] = Field(min_length=1)    # item_ids (ContextItem) que sustentam a aresta — é a chave do GraphScope

    valid_from: datetime | None = None           # tempo do mundo (LLM/item); None = desde sempre
    valid_to: datetime | None = None             # None = ainda vale
    created_at: datetime                         # tempo de transação
    invalidated_at: datetime | None = None       # None = viva; set em contradição / re-ingestão / exclusão

    metadata: dict = {}                          # extractor, anchor, alias, embed, candidatos ambíguos…
```

Notas de implementação (ponytail):

- `weight` **não** entra: é `len(evidence)` (é assim que GraphRAG e HippoRAG chegam ao peso).
- `extractor` ("wikilink" | "llm" | "embedding" | "structure") fica em `metadata`, não é campo — só auditoria.
- Persistência: manter `graph_edges` e **adicionar colunas** `provenance, confidence, valid_from, valid_to, created_at, invalidated_at`; `evidence` numa tabela `graph_edge_evidence(edge_id, item_id)` no mesmo molde da `graph_memory_links` existente; trocar `UNIQUE(source, relation, target)` por índice único parcial `WHERE invalidated_at IS NULL`.
- Predicado de travessia único, em um lugar (`_filters.py` já existe): `invalidated_at IS NULL AND (valid_from IS NULL OR valid_from <= :as_of) AND (valid_to IS NULL OR valid_to > :as_of)`.
- Exclusão por escopo = aresta cuja `evidence` ∩ itens visíveis = ∅ é invisível; nó sem aresta/evidência visível não existe. Uma regra, aplicada antes do BFS.

---

## Anexos — trechos verificados

- GraphRAG `graph_extractor.py:146-158`: `entity_name = clean_str(record_attributes[1].upper())`, `source = clean_str(record_attributes[1].upper())`, `weight = float(record_attributes[-1])` (fallback 1.0).
- GraphRAG `_merge_relationships`: `groupby(["source","target"]).agg(description=list, text_unit_ids=list, weight=("weight","sum"))`.
- Graphiti `edges.py`: `name: 'name of the edge, relation name'`, `fact: 'fact representing the edge and nodes that it connects'`, `episodes: 'list of episode ids that reference these entity edges'`, `expired_at: 'datetime of when the node was invalidated'`, `valid_at: 'datetime of when the fact became true'`, `invalid_at: 'datetime of when the fact stopped being true'`.
- Graphiti `resolve_extracted_edges`: dedup exato por `(source_uuid, target_uuid, _normalize_string_exact(fact))`; candidatos de duplicata restritos a arestas entre o **mesmo par** de nós (paper: "constrained to edges existing between the same entity pairs"); candidatos de invalidação por busca híbrida no fato.
- mem0 `EXTRACT_RELATIONS_PROMPT`: "Extract only explicitly stated information… Use consistent, general, and timeless relationship types. Example: Prefer 'professor' over 'became_professor.'"
- mem0 `DELETE_RELATIONS_SYSTEM_PROMPT`: "DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes" (exemplo pizza/burger).
- HippoRAG `HippoRAG.py:1327`: `if score < self.global_config.synonymy_edge_sim_threshold or num_nns >= 100: break`.
- graphify `extraction-spec.md`: "confidence_score is REQUIRED on every edge - never omit it, never use 0.5 as a default".
- graphify `ids.py` docstring: lista os bugs de drift de ID (#811, #550, #1033, #1104) como razão de centralizar `normalize_id`.
- obsidian-export `lib.rs:823-826`: `ends_with(filename)` || `ends_with(filename + ".md")` || versões `to_lowercase()`.

---

# Parte 3 — Mapa do código (dev @ 6cbedfd)


Repo `/Users/arthurgranja/github/anchor`, branch `dev`, HEAD `6cbedfd`. Todos os caminhos são absolutos a partir dessa raiz.

## 1. Grafo existente

### `src/anchor/memory/graph_memory.py` — `SimpleGraphMemory`

Classe única, 238 linhas, **zero dependências** (só `collections.deque`). Definida em `:14`.

Estrutura de dados (`__init__`, `:47-52`):

```
_nodes: dict[str, dict[str, Any]]            # entity_id -> metadata
_edges: list[tuple[str, str, str]]           # (source, relation, target)
_entity_to_memories: dict[str, list[str]]    # entity_id -> [memory_id]
_adjacency: dict[str, set[str]]              # cache bidirecional
_adjacency_dirty: bool
```

API pública completa:

| Assinatura | Linha | Nota |
|---|---|---|
| `add_entity(entity_id: str, metadata: dict \| None = None) -> None` | `:54` | merge de metadata se já existe |
| `add_relationship(source: str, relation: str, target: str) -> None` | `:69` | auto-cria nós; **permite duplicata** (append em lista, sem dedupe) |
| `link_memory(entity_id: str, memory_id: str) -> None` | `:87` | `KeyError` se o nó não existe; **append sem dedupe** |
| `get_related_entities(entity_id, max_depth: int = 2) -> list[str]` | `:110` | BFS, exclui o nó inicial |
| `get_memory_ids_for_entity(entity_id) -> list[str]` | `:146` | |
| `get_related_memory_ids(entity_id, max_depth=2) -> list[str]` | `:158` | entidade + vizinhança, dedup por ordem |
| `remove_entity(entity_id) -> None` | `:182` | rebuild de `_edges` por list-comp O(E) |
| `clear() -> None` | `:197` | |
| `entities -> list[str]` (property) | `:204` | |
| `relationships -> list[tuple[str,str,str]]` (property) | `:210` | |
| `get_entity_metadata(entity_id) -> dict` | `:214` | `KeyError` se ausente |
| `__repr__` / `__len__` | `:231` / `:237` | |

Detalhes do BFS (`:124-144`): retorna `[]` se o nó não existe; `_rebuild_adjacency` (`:102`) **colapsa a direção** — `_adjacency` recebe `src→tgt` e `tgt→src`, então a travessia é sempre não-direcionada e a `relation` é descartada. Não há filtro por tipo de aresta, nem por profundidade ponderada, nem orçamento.

**Limitações relevantes para a frente #4:**
- Só em memória. Nenhuma persistência, nenhum `GraphStore`.
- Liga **`memory_id`** (documentado em `:92` como "``MemoryEntry.id``"), não `item_id`.
- Sem `vault`, sem `namespace`, sem escopo em nenhuma superfície.
- Sem proveniência de aresta, sem validade temporal, sem tipo consultável (a `relation` existe mas nenhum método a usa para filtrar).
- Sem `path`, sem `explain`, sem `backlinks`, sem centralidade.
- Arestas duplicadas acumulam em `_edges` (`:84`) — `relationships` devolve repetidos.

### `src/anchor/pipeline/memory_steps.py:51` — `graph_retrieval_step`

```python
def graph_retrieval_step(
    graph: SimpleGraphMemory,
    store: MemoryEntryStore,
    entity_extractor: Callable[[str], list[str]],
    max_depth: int = 2,
    max_items: int = 5,
    name: str = "graph_retrieval",
    on_error: Literal["raise", "skip"] = "skip",
) -> PipelineStep:
```

Fluxo real (closure `_retrieve`, `:85-129`):
1. `entity_ids = entity_extractor(query.query_str)` (`:86`) — extrator é **100% fornecido pelo usuário**, não há implementação no repo.
2. Para cada entidade, `graph.get_related_memory_ids(entity_id, max_depth=max_depth)` (`:94`), dedup ordenado.
3. `all_entries = store.list_all()` e `entry_map = {e.id: e for e in all_entries}` (`:104-105`) — **full scan do store a cada query**; não há `get(id)` em lote.
4. Constrói `ContextItem(content=entry.content, source=SourceType.MEMORY, score=entry.relevance_score, priority=6, metadata={"memory_id", "memory_type", "tags", "source": "graph_retrieval"})` (`:115-126`).
5. Retorna `items + new_items` (append, não fusão/rerank).

**Sem `scope`.** Ao contrário de `retriever_step` (`pipeline/step.py:85`), este step não lê `effective_scope` nem repassa escopo — é um vazamento hoje, se o grafo passar a carregar itens escopados.

Registro / uso:
- `src/anchor/pipeline/__init__.py:5` (import) e `:33` (`__all__`).
- `src/anchor/__init__.py:18` (comentário de agrupamento), `:293` (import), `:670` (`__all__`).
- `src/anchor/memory/__init__.py:10` e `:31` para `SimpleGraphMemory`; `src/anchor/__init__.py:224` e `:622`.
- Testes: `tests/test_pipeline/test_memory_steps.py:15,96,108,120,132,152,176,194` e `tests/test_memory/test_graph_memory.py` (≈60 instanciações, cobertura densa da API atual).
- **Nenhum uso em `src/` fora do próprio registro** — o step é opcional e nunca montado por default (nem na CLI, nem em `Agent`).

### Protocolo `GraphMemory`/`GraphStore` em `src/anchor/protocols/`?

**Não existe na `dev`.** Os protocolos de storage (`src/anchor/protocols/storage.py`) são:

| Protocolo | Linha |
|---|---|
| `ContextStore` | `:17` |
| `VectorStore` | `:81` |
| `DocumentStore` | `:162` |
| `MemoryEntryStore` | `:221` |
| `GarbageCollectableStore` | `:293` |
| `AsyncContextStore` | `:332` |
| `AsyncVectorStore` | `:347` |
| `AsyncDocumentStore` | `:374` |
| `AsyncMemoryEntryStore` | `:390` |
| `AsyncGarbageCollectableStore` | `:405` |

Nem `protocols/memory.py` (que tem `CompactionStrategy:28`, `MemoryExtractor:70`, `MemoryConsolidator:113`, `MemoryDecay`, `MemoryOperation:18`) traz grafo.

### O branch `feature/storage-layer-gaps` (o que a tabela do plano promete)

`git worktree list` → `/Users/arthurgranja/github/anchor/.worktrees/storage-layer-gaps` em `1f0afb0`.

**Estado: divergido, não mergeável como está.** 17 commits à frente da `dev`, ~20 atrás. `git diff --stat dev feature/storage-layer-gaps` → `229 files changed, 10848 insertions(+), 24243 deletions(-)`. O diff **apaga** artefatos da frente #3 e posteriores:

```
D src/anchor/models/scope.py
D src/anchor/storage/_where.py
D src/anchor/evaluation/golden.py
D src/anchor/agent/subagent.py
D src/anchor/agent/events.py
D src/anchor/agent/hooks.py
D src/anchor/embeddings/*  (todo o pacote)
D src/anchor/storage/sqlite/_vec_store.py
D src/anchor/llm/providers/claude_cli.py
```

O que ele adiciona e vale colher (cherry-pick, não merge):

```
A src/anchor/storage/sqlite/_graph_store.py        (341 linhas)
A src/anchor/storage/postgres/_graph_store.py      (183 linhas, WITH RECURSIVE :70,:90)
A src/anchor/storage/sqlite/_conversation_store.py
A src/anchor/storage/postgres/_conversation_store.py
A src/anchor/cache/sqlite_backend.py
A src/anchor/cache/redis_backend.py
A tests/test_storage/test_graph_store.py           (329 linhas)
```

Protocolo `GraphStore` do branch (`.worktrees/storage-layer-gaps/src/anchor/protocols/storage.py:384`), API: `add_node`, `add_edge(source, relation, target, metadata)`, `get_neighbors(node_id, max_depth=1, relation_filter=None)`, `get_edges`, `get_node_metadata`, `link_memory`, `get_memory_ids(node_id, max_depth=1)`, `remove_node`, `remove_edge`, `list_nodes`, `list_edges`, `clear`. Gêmeo `AsyncGraphStore` em `:473`. `ConversationStore` em `:532`.

Schema SQLite do branch (`.worktrees/.../storage/sqlite/_schema.py:56-72`):

```sql
graph_nodes(node_id TEXT PRIMARY KEY, metadata_json TEXT)
graph_edges(id INTEGER PK AUTOINCREMENT, source, relation, target, metadata_json,
            UNIQUE(source, relation, target))
graph_memory_links(node_id, memory_id, PRIMARY KEY(node_id, memory_id))
```

Índices: `idx_edges_source`, `idx_edges_target`, `idx_edges_relation`, `idx_memory_links_node` (`:103-106`). **Nenhuma coluna `vault`/`namespace`, e a tabela de links chama-se `memory_id`** — precisa virar `item_id` + colunas de escopo antes de servir à frente #4.

`SimpleGraphMemory` do branch (`.worktrees/.../memory/graph_memory.py:33`) recebe `store: GraphStore | None = None` e delega cada método com `if self._store is not None: ... else: <fallback dict>`. Duplicação de lógica em cada método — vale reescrever em vez de colher tal qual.

Doc de design correspondente: `docs/plans/2026-03-14-storage-layer-gaps-design.md` (spec completa do `GraphStore`) e `docs/plans/2026-03-14-storage-layer-gaps.md`.

## 2. Moeda única

### `ContextItem` — `src/anchor/models/context.py:52`

```python
class ContextItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))   # :59
    content: str                                                  # :60
    source: SourceType                                            # :61
    score: float = Field(default=0.0, ge=0.0, le=1.0)             # :62
    priority: int = Field(default=5, ge=1, le=10)                 # :63
    token_count: int = Field(default=0, ge=0)                     # :64
    metadata: dict[str, Any] = Field(default_factory=dict)        # :65
    created_at: datetime = ...                                    # :66
    vault: str = DEFAULT_VAULT                                    # :69
    namespace: str = ROOT_NAMESPACE                               # :70
    model_config = ConfigDict(frozen=True)                        # :72
```

- **Não há campo `embedding`.** Vetores vivem no `VectorStore`, chaveados por `item_id` (`protocols/storage.py:87`).
- Validadores: `_validate_vault` (`:74-77` → `validate_vault`) e `_validate_namespace` (`:79-82` → `normalize_namespace`). Vault e namespace são checados **na construção**, não na leitura.
- `frozen=True` — toda mutação é `model_copy(update=...)` (padrão usado em `_rrf.py:75`, `cli.py:205`, `hierarchical.py:237`).

`SourceType` (`:41-49`), StrEnum: `RETRIEVAL="retrieval"`, `MEMORY="memory"`, `SYSTEM="system"`, `USER="user"`, `TOOL="tool"`, `CONVERSATION="conversation"`.

Também no arquivo: `ContextWindow` (`:85`), `ContextResult` (`:129`), `PipelineDiagnostics` TypedDict (`:24`), `StepDiagnostic` (`:16`).

### Geração de `id`

O default é `uuid4()` — **não determinístico**. O caminho de ingestão sobrescreve:

- `src/anchor/ingestion/metadata.py:30` — `generate_chunk_id(doc_id, chunk_index) -> f"{doc_id}-chunk-{chunk_index}"` (`:40`). Determinístico por `(doc_id, índice)`, **não por conteúdo**: reordenar chunks reatribui ids.
- Usado em `src/anchor/ingestion/ingester.py:296` (`_build_items`) e `:341` (`_build_items_with_metadata`), com `metadata` de `extract_chunk_metadata` (`metadata.py:43`).
- Contraste: `ParentChildChunker` usa `sha256(parent_text)[:16]` para `parent_id` (`ingestion/hierarchical.py:140`) — esse **é** content-hash. Duas convenções de id convivem no repo.

### `MemoryEntry` — `src/anchor/models/memory.py:41`

```python
id: str = uuid4                       # :47
content: str                          # :48
relevance_score: float [0,1] = 0.5    # :49
access_count: int = 0                 # :50
last_accessed / created_at / updated_at: datetime   # :51,:52,:61
tags: list[str]                       # :53
metadata: dict[str, Any]              # :54
memory_type: MemoryType = SEMANTIC    # :57
user_id / session_id: str | None      # :58,:59
expires_at: datetime | None           # :60
content_hash: str = ""                # :62  (md5, _compute_content_hash :35)
source_turns: list[str]               # :63
links: list[str]                      # :64  ← CAMPO DORMENTE
```

`MemoryEntry.links` só aparece em: `memory/consolidator.py:66` (merge no dedupe), `storage/_serialization.py:65,:92` (`links_json`), `storage/postgres/_entry_store.py:60,:76,:238`. **Ninguém lê para navegar.** É a aresta memória→memória que já está persistida em todos os backends e nunca foi usada — candidata natural a virar aresta de grafo sem migração de schema.

Sem `vault`/`namespace` em `MemoryEntry` — memória vive na raiz (ver adaptador abaixo). Também no arquivo: `ConversationTurn:26`, `MemoryType:17`, `FactType:90`, `KeyFact:101`, `SummaryTier:112`, `TierConfig:124`.

### `MemoryEntry` → `ContextItem`

`src/anchor/retrieval/memory_retriever.py:247` — `MemoryRetrieverAdapter`:

```python
def retrieve(self, query: QueryBundle, top_k: int = 10, *,
             scope: RetrievalScope | None = None) -> list[ContextItem]:   # :265
    if scope is not None and not scope.matches(ROOT_NAMESPACE):           # :286
        return []
    entries = self._retriever.retrieve(query.query_str, top_k=top_k)      # :288
    return [ContextItem(content=e.content, source=SourceType.MEMORY,
                        priority=7, score=e.relevance_score,
                        metadata={"memory_type": e.memory_type,
                                  "memory_id": e.id}) ...]                # :290-297
```

**Regra que o grafo herda:** memória não tem namespace; toda entrada é tratada como `/` e um escopo que não vê a raiz esconde tudo (`:286`). O `id` do `ContextItem` gerado **é um uuid novo** — o `memory_id` só sobrevive em `metadata`. Isso é exatamente o problema que a decisão "grafo liga `item_id`" resolve.

`ScoredMemoryRetriever` (`:22`, `retrieve` em `:85`, `as_retriever()` em `:222`) é o produtor; ele **não** aceita `scope` — o gate está só no adaptador.

### Documento → `ContextItem` (caminho `anchor index`)

`src/anchor/cli.py:158` — comando `index(path, db, embeddings, chunk_size, vault, namespace)`:

1. `DocumentIngester(chunker=RecursiveCharacterChunker(chunk_size))` (`:187`).
2. `ingester.ingest_file(path)` / `ingest_directory(path)` (`:190-193`) → `list[ContextItem]` via `_build_items` (`ingester.py:284`).
3. `ns = normalize_namespace(namespace)` (`:200`), depois `items = [item.model_copy(update={"namespace": ns}) for item in items]` (`:205`) — **o namespace é carimbado depois da ingestão**, não durante.
4. `context_store = _open_context_store(db, vault)` (`:207`, helper em `cli.py:116-123` → `SqliteContextStore(manager, vault=vault)`), loop `context_store.add(item)` (`:208-209`).
5. Se houver provider de embeddings: `provider.embed_documents([...])`, `_open_vector_store(db, dim, vault)` e `vector_store.add_embedding(item.id, vector, item.metadata, namespace=ns)` (`:216-218`).

Ou seja: **`ContextItem.id` (= `{doc_id}-chunk-{i}`) é a chave que amarra ContextStore e VectorStore.** É essa a chave que o grafo deve referenciar; a chave real de linha é `(vault, id)` (ver §4).

`SqliteContextStore` (`storage/sqlite/_context_store.py:15`): vault-bound (`__init__:20`, `validate_vault:24`), `add` reescreve o vault do item para o do mount (`:32-33`), `get`/`get_all`/`delete`/`clear` sempre filtram `WHERE vault = ?` (`:49,:59,:66,:75`). **`ContextStore` não tem método com `scope`** — o filtro por namespace vive só no `VectorStore`.

## 3. Escopo da frente #3 (shippada)

### `src/anchor/models/scope.py` (177 linhas)

| Símbolo | Linha | Assinatura / valor |
|---|---|---|
| `DEFAULT_VAULT` | `:29` | `"__default__"` |
| `ROOT_NAMESPACE` | `:33` | `"/"` |
| `validate_vault(vault: str) -> str` | `:36` | rejeita vazio, `/` e `:` |
| `normalize_namespace(ns: str) -> str` | `:57` | força `/` inicial, tira `/` final, `ValueError` em vazio |
| `is_under(namespace, prefix) -> bool` | `:74` | `ns == prefix or ns.startswith(prefix + "/")`; `prefix == "/"` → sempre True |
| `RetrievalScope(BaseModel, frozen=True)` | `:85` | `include: tuple[str,...] = ()`, `exclude: tuple[str,...] = ()` |
| `._normalize` (validator) | `:93` | normaliza ambas as tuplas |
| `.matches(namespace) -> bool` | `:98` | exclude primeiro, include vazio = tudo |
| `.intersect(child) -> RetrievalScope` | `:107` | só estreita; disjunto → `exclude=("/",)` (`:132`) |
| **`ACTIVE_SCOPE`** (ContextVar) | `:142` | nome **sem** underscore inicial |
| `active_scope() -> RetrievalScope \| None` | `:147` | |
| `effective_scope(local) -> RetrievalScope \| None` | `:152` | `active.intersect(local)` |
| `scope_kwargs(scope) -> dict` | `:162` | `{}` ou `{"scope": scope}` — compat com stores pré-#3 |
| `same_vault(*stores) -> None` | `:172` | levanta se mounts divergem |

Docstring do módulo (`:1-20`) fixa a doutrina: **vault nunca faz parte do escopo** (é mount, ligado na construção do store); escopo é só navegação por namespace.

### `Agent` — `src/anchor/agent/agent.py`

- Import: `:33` `from anchor.models.scope import ACTIVE_SCOPE, RetrievalScope`.
- Slots: `_active_scope` (`:235`), `_scope` (`:266`); init em `:327-328`.
- `with_scope(self, scope: RetrievalScope) -> Agent` em **`:441`** — grava `self._scope`, retorna `self` (builder).
- `_pool_entry(self) -> bool` em **`:2132`**. O cálculo do escopo efetivo está em `:2144-2148`:
  ```python
  inherited_scope = ACTIVE_SCOPE.get()
  if inherited_scope is not None and self._scope is not None:
      self._active_scope = inherited_scope.intersect(self._scope)
  else:
      self._active_scope = self._scope or inherited_scope
  ```
- **Publicação na janela da tool call**: `:1470` `scope_token = ACTIVE_SCOPE.set(self._active_scope)` … `:1474` `ACTIVE_SCOPE.reset(scope_token)`, dentro do `try/finally` que envolve `self._execute_call(tc)` (`:1471`). Comentário em `:1465-1467` explica a doutrina: set e reset no mesmo frame, nunca ambient por turno.
- Outros pontos de publicação: `:1537` (caminho async da tool call), `:2281`/`:2285` (`build` do pipeline do turno), `:2371`/`:2375` (variante async). O commit `e863427` fez o escopo alcançar o build do pipeline, não só as tools.
- `SubagentDefinition.scope` aplicado em `:711-714`: `if definition.scope is not None: sub.with_scope(definition.scope)`.

### `SubagentDefinition` — `src/anchor/agent/subagent.py:56`

```python
scope: RetrievalScope | None = None
```
Comentário `:54-55`: "Narrower retrieval scope for this child; the parent's published scope still applies on top (intersection) — it only ever narrows." Modelo frozen (`:42`).

### `scope_sql_clauses` — `src/anchor/storage/_where.py`

Arquivo: `src/anchor/storage/_where.py` (179 linhas). Docstring `:8-13` explica a escolha: prefixo compila para **range**, nunca `LIKE`, porque é a forma que o `vec0` do sqlite-vec consegue empurrar para baixo e que um btree serve; assume ordenação byte-wise (BINARY no SQLite, `COLLATE "C"` no Postgres).

```python
_AFTER_SLASH = "0"                                        # :35 (char após '/')

def _prefix_range(ns_col: str, prefix: str) -> tuple[str, list[Any]]:   # :147
    if prefix == ROOT_NAMESPACE:
        return "1=1", []
    return (f"({ns_col} = ? OR ({ns_col} >= ? AND {ns_col} < ?))",
            [prefix, prefix + "/", prefix + _AFTER_SLASH])

def scope_sql_clauses(scope: RetrievalScope | None, ns_col: str
                     ) -> tuple[list[str], list[Any]]:     # :156
```
Includes viram um único `(... OR ...)` (`:168-174`); cada exclude vira `NOT <range>` em AND (`:175-177`) — exclude ganha por construção.

Consumidores: `storage/sqlite/_vector_store.py:19,:64`, `storage/sqlite/_vec_store.py:27,:246`, `storage/postgres/_vector_store.py:22,:174` (via `_numbered` para `$n`). Também no arquivo: `SQL_OPS:23`, `SET_OPS:31`, `is_op_dict:38`, `check_operator:44`, `matches_where:77` (avaliação pura em Python, para backends in-memory), `sql_where_clauses:93`.

### Como um `GraphStore` novo consome isso

O que reusar, na ordem:

1. **`scope_sql_clauses(scope, "namespace")`** (`_where.py:156`) na tabela de *evidências* (`graph_item_links`), não na de nós. É a única função que produz o prefixo boundary-aware; escrever `LIKE` de novo quebraria a paridade `/campanha-1` vs `/campanha-10`.
2. **`matches_where` / `RetrievalScope.matches`** (`scope.py:98`) para o backend in-memory — mesma semântica, avaliação Python.
3. **`validate_vault`** (`scope.py:36`) no `__init__` do store, e vault como coluna da PK — o padrão de `SqliteContextStore.__init__:20-24`.
4. **`effective_scope`** (`scope.py:152`) em toda entrada pública de travessia, e **`scope_kwargs`** (`:162`) ao repassar a stores.

Para a regra "nó existe para este escopo": a consulta é `EXISTS (SELECT 1 FROM graph_item_links l WHERE l.node_id = n.node_id AND l.vault = ? AND <scope_sql_clauses(scope, "l.namespace")>)`. Isso satisfaz de uma vez as duas decisões do plano — exclusão em nível de nó, e nó excluído é parede: o mesmo `EXISTS` aplicado a cada salto da BFS/CTE impede a travessia por cima.

**Nota:** `graph_item_links` precisa carregar `vault` + `namespace` como colunas *desnormalizadas* (copiadas do `ContextItem` na indexação). Um JOIN com `context_items` não funciona para itens de memória, que não vivem lá.

## 4. Storage

### Arquivos (`src/anchor/storage/`, 4055 linhas)

| Arquivo | Linhas | O que é |
|---|---|---|
| `__init__.py` | 21 | reexports |
| `_base.py` | 170 | `BaseEntryStoreMixin:28` — search/list/get/`search_filtered`/`delete_by_user` compartilhados por stores de `MemoryEntry` in-memory e JSON; exige `self._entries` + `self._lock`; hook `_after_mutation:51` |
| `_filters.py` | 128 | `matches_filters`, `search_entries`, `list_all_entries`, `search_filtered_entries` — lógica de filtro de `MemoryEntry` |
| `_serialization.py` | 101 | `context_item_to_row` / `row_to_context_item`, `memory_entry_to_row` / `row_to_memory_entry` (`links_json` em `:65,:92`) |
| `_where.py` | 179 | compilador único de `where` + escopo (§3) |
| `json_file_store.py` | 191 | `JsonFileMemoryStore` — `MemoryEntry` em arquivo JSON |
| `json_memory_store.py` | 47 | `InMemoryEntryStore` |
| `memory_store.py` | 204 | `InMemoryContextStore:29` (vault-bound, `:39-52`), `InMemoryVectorStore:74` (`add_embedding:104`, `search:118`, `_cosine_similarity:159`), `InMemoryDocumentStore:171` |
| `postgres/_connection.py` | 94 | `PostgresConnectionManager:14` — **pool asyncpg**; `initialize():43` obrigatório antes do uso, `acquire():83`, `close():87`; `_get_pool():76` levanta se não inicializado |
| `postgres/_context_store.py` | 121 | `PostgresContextStore` |
| `postgres/_document_store.py` | 64 | |
| `postgres/_entry_store.py` | 239 | não reusa `_serialization` (asyncpg já devolve JSONB parseado — nota em `:1-5`) |
| `postgres/_schema.py` | 189 | `ensure_tables(conn, *, embedding_dim):11`; `_migrate_scoped_table:134` (ADD COLUMN + recriação de índice) |
| `postgres/_vector_store.py` | 199 | pgvector; escopo em `:174` |
| `redis/_connection.py` | 84 | manager sync + async |
| `redis/_context_store.py` | 224 | layout de chave `ctxv:` (commit `ef80238`) |
| `redis/_entry_store.py` | 366 | |
| `sqlite/__init__.py` | 36 | reexports, incl. `ensure_tables`, `ensure_tables_async` |
| `sqlite/_connection.py` | 134 | `SqliteConnectionManager:18`; `get_connection():49` (sync), `get_async_connection(_attempts=3):69` (aiosqlite, com retry), `close():116`, `aclose():127`; WAL por default (`:38`) |
| `sqlite/_context_store.py` | 152 | `SqliteContextStore:15` + `AsyncSqliteContextStore` |
| `sqlite/_document_store.py` | 112 | |
| `sqlite/_entry_store.py` | 254 | |
| `sqlite/_schema.py` | 172 | DDL + migração (abaixo) |
| `sqlite/_vec_store.py` | 313 | `SqliteVecVectorStore` — KNN real via extensão `sqlite-vec`; escopo empurrado em `:246` |
| `sqlite/_vector_store.py` | 220 | brute-force cosseno; escopo em `:64` |

Cache é pacote separado: `src/anchor/cache/backend.py` — só `InMemoryCacheBackend:12` (`get:39`, `set:59`, `invalidate:84`, `clear:92`). Protocolo em `src/anchor/protocols/cache.py:9`. **Não há backend Redis/SQLite de cache na `dev`** (estão só no branch parado).

### Como o SQLite cria schema

`src/anchor/storage/sqlite/_schema.py`:

- `_TABLES: dict[str, str]` (`:13-58`) — DDL de `context_items` (`:14`), `embeddings` (`:27`), `documents` (`:35`), `memory_entries` (`:40`). **`context_items` e `embeddings` têm `PRIMARY KEY (vault, id)`** (`:25`, `:33`); `documents` e `memory_entries` ainda são chaveadas por id simples.
- `_SCOPED_TABLES = ("context_items", "embeddings")` (`:61`).
- `_INDEXES` (`:63-71`) — `idx_context_items_scope ON context_items(vault, namespace)` (`:64`), `idx_embeddings_scope` (`:65`), mais 5 em `memory_entries`.
- **Não existe `add_scope_columns`.** O helper real é `rebuild_with_scope_key(conn, table, ddl) -> bool` (`:91`): inspeciona `PRAGMA index_list`/`index_info`, e se nenhum índice único inclui `vault`, copia as linhas para `{table}__new` preservando rowid (`_rebuild_plan:74`, comentário em `:75-77` — vec0 aponta para o rowid), dropa e renomeia, tudo em uma transação com ROLLBACK (`:108-121`). Idempotente. Gêmeo async: `rebuild_with_scope_key_async` (`:124`).
- `ensure_tables(conn)` (`:153`): cria todas as DDL → roda `rebuild_with_scope_key` nas duas scoped → cria índices → commit. `ensure_tables_async(conn)` (`:164`) é o espelho literal, 8 linhas.

**Não há tabela de migrations, nem número de versão.** A migração é detectada por introspecção (`PRAGMA`), a cada `ensure_tables`. Um `SqliteGraphStore` novo entra adicionando entradas em `_TABLES` + `_INDEXES` e, se precisar de chave por vault, listando-se em `_SCOPED_TABLES`.

### Como o Postgres faz

`src/anchor/storage/postgres/_schema.py`: `async def ensure_tables(conn: asyncpg.Connection, *, embedding_dim: int)` (`:11`). Valida `embedding_dim > 0` (`:26`), `CREATE EXTENSION IF NOT EXISTS vector` (`:30`), depois `CREATE TABLE IF NOT EXISTS` para `context_items` (`:32`), `embeddings` (`:48`), `documents` (`:59`), `memory_entries` (`:67`). Namespace é `TEXT COLLATE "C"` (`:42`, `:53`) — indispensável para o range de prefixo funcionar. PK `(vault, id)` / `(vault, item_id)`.

Migração: `_migrate_scoped_table` (`:134`) — `ALTER TABLE ... ADD COLUMN namespace TEXT COLLATE "C"` (`:154`), e se o índice existente usa `text_pattern_ops` (`:163`), recria com opclass simples numa coluna `COLLATE "C"` (`:125` explica: `text_pattern_ops` só serve colação não-C). Conexão via pool (`_connection.py`), sempre async — **não existe twin sync no Postgres**.

### Padrão sync/async twins

Convenção firme: mesma classe duas vezes, prefixo `Async`, no mesmo arquivo.

```
SqliteContextStore  / AsyncSqliteContextStore    (sqlite/_context_store.py)
SqliteVectorStore   / AsyncSqliteVectorStore     (sqlite/_vector_store.py)
SqliteEntryStore    / AsyncSqliteEntryStore      (sqlite/_entry_store.py)
SqliteDocumentStore / AsyncSqliteDocumentStore   (sqlite/_document_store.py)
ensure_tables       / ensure_tables_async        (sqlite/_schema.py:153/:164)
rebuild_with_scope_key / rebuild_with_scope_key_async  (:91/:124)
```

Exceções: `SqliteVecVectorStore` é só sync; Postgres é só async; Redis tem os dois no mesmo módulo. Protocolos espelham (`protocols/storage.py:332-405`).

### Fixtures de teste — `tests/test_storage/conftest.py` (140 linhas)

Não existe fixture chamada `context_store_impl`. As que existem:

| Fixture | Linha | Params |
|---|---|---|
| `entry_store` | `:29` | `["memory", "json_file", "sqlite"]` |
| `sqlite_conn_manager` | `:49` | — (chama `ensure_tables` em `:53`) |
| `sqlite_context_store` | `:57` | — |
| `sqlite_vector_store` | `:65` | — |
| `sqlite_document_store` | `:73` | — |
| `DIM = 128` | `:81` | constante importada por outros testes |
| **`make_vector_store`** | `:84` | `["memory", "sqlite", "vec"]` — **factory** `maker(vault) -> store` sobre **um mesmo backing** (`:86`); `vec` faz `importorskip("sqlite_vec")` (`:104`); teardown fecha os abertos (`:114-115`) |
| **`make_context_store`** | `:118` | `["memory", "sqlite"]` — mesma forma de factory por vault |

**O padrão a copiar para um `GraphStore`:** as duas últimas. A factory-por-vault (em vez de fixture-de-instância) é o que permite testar isolamento entre vaults sobre um único arquivo `.db` — exatamente o teste de vazamento que o plano exige ("nó excluído invisível por qualquer caminho").

Um `SqliteGraphStore` se encaixaria assim: DDL em `sqlite/_schema.py:_TABLES` + índices, classe `SqliteGraphStore` / `AsyncSqliteGraphStore` em `sqlite/_graph_store.py`, exports em `sqlite/__init__.py`, protocolo em `protocols/storage.py`, e fixture `make_graph_store(vault)` em `tests/test_storage/conftest.py`.

## 5. Avaliação (`src/anchor/evaluation/`, 1348 linhas)

| Arquivo | Linhas | Símbolos-chave |
|---|---|---|
| `retrieval.py` | 178 | **`RetrievalMetricsCalculator:18`** (`__init__(k=10):27`, `evaluate:37`); privados `_precision_at_k:91`, `_recall_at_k:99`, `_f1:107`, `_mrr:114`, `_ndcg:122`, `_graded_ndcg:149`, `_hit_rate:176` |
| `models.py` | 68 | `RetrievalMetrics:10`, **`RAGMetrics:35`**, `EvaluationResult:57` |
| `evaluator.py` | 120 | `PipelineEvaluator:17` — `evaluate_retrieval(retrieved, relevant, k=10):40`, `evaluate_rag:58` |
| `rag.py` | 93 | `LLMRAGEvaluator:18` (`__init__:38`, `evaluate:58`) |
| `golden.py` | 161 | `GoldenCase:33`, `GoldenCaseResult:47`, `GoldenSetReport:57`, `load_golden_set:85`, `evaluate_retriever:105`, `assert_metric_floor:140` |
| `ab_testing.py` | 287 | `AggregatedMetrics:30`, `EvaluationSample:52`, `EvaluationDataset:68`, `ABTestResult:84`, `_normal_cdf:115`, `_t_test_paired:127`, `ABTestRunner:157` |
| `human.py` | 187 | `HumanEvaluationCollector`, `HumanJudgment` |
| `batch.py` | 208 | `BatchEvaluator` |

**Atenção ao nome:** não existe `RetrievalEvaluator`. O `__init__.py` exporta `RetrievalMetricsCalculator` (`:23,:41`) e `PipelineEvaluator` (`:12,:38`). `RAGMetrics` existe (`models.py:35`), produzido por `LLMRAGEvaluator`.

### Golden set

Formato (`golden.py:33-44`): JSONL, uma linha por caso.
```json
{"query": "...", "relevant": ["id1", "id2"]}
{"query": "...", "relevant": {"id1": 2.0}, "name": "..."}
```
`relevant: list[str] | dict[str, float]` — binário ou graduado (NDCG graduado via `_graded_ndcg`).

Loader: `load_golden_set(path) -> list[GoldenCase]` (`:85`), linha a linha, erro com `path:line_num` (`:100`).
Runner: `evaluate_retriever(retriever, cases, k=10, calculator=None) -> GoldenSetReport` (`:105`).
Gate de CI: `assert_metric_floor(report, metric, floor)` (`:140`) — mensagem inclui os 3 piores casos (`:150-156`).
`GoldenSetReport.mean(metric):63`, `.summary():70` (precision/recall/f1/mrr/ndcg/hit_rate).

**Existe corpus real? Não.** Zero arquivos `.jsonl` versionados no repo (busca por `*.jsonl` fora de `.venv` retorna nada). As únicas fontes:
- `tests/test_evaluation/test_golden.py:42,:51` — escreve JSONL em `tmp_path` dentro do teste.
- `tests/test_storage/test_scope_recall_benchmark.py:97` — constrói `GoldenCase` em memória.
- Docstring `golden.py:3-5` descreve a prática ("50-200 queries checadas à mão") mas nenhum corpus foi montado.

Isso confirma o item do plano "Golden set montado **antes** do construtor" — ele não existe, é trabalho novo.

### Como se roda um A/B retriever X vs Y hoje

`ABTestRunner` (`ab_testing.py:157`):
```python
runner = ABTestRunner(PipelineEvaluator(), dataset)     # :167
result = runner.run(retriever_a, retriever_b, k=10, significance_level=0.05)   # :171
```
Loop em `:206-215`: para cada `sample`, `retriever_a.retrieve(QueryBundle(query_str=sample.query), top_k=k)` e idem para B, métricas por `evaluate_retrieval`. Agrega (`_aggregate:247`), t-test pareado **sobre precision@k** (`:220-222`, `_t_test_paired:127`, `_normal_cdf:115` — implementação própria, sem scipy), decide `winner` `"a"/"b"/"tie"` (`:225-228`), monta `per_metric_comparison` (`_build_per_metric_comparison:262`).

Exemplos de teste existentes: `tests/test_evaluation/test_ab_testing.py` — `TestABTestRunner:62` (12 cenários, `:98`–`:315`) e `TestABTestRunnerIntegration:397` (`:417` "Human judgments → EvaluationDataset → ABTestRunner end-to-end", `:449`).

**Buraco relevante para a frente #4:** nem `ABTestRunner.run` (`:210-211`) nem `evaluate_retriever` (`golden.py:128`) passam `scope`. O protocolo local `_Retriever` em `golden.py:29-30` é `retrieve(self, query: QueryBundle, top_k: int = 10)` — **sem `scope`**, divergindo do `protocols/retriever.py:20`. Para A/B-testar um `GraphRetriever` escopado será preciso ou estender essas assinaturas, ou embrulhar o retriever num adaptador que fixa o escopo (é o que `test_scope_recall_benchmark.py:49` faz com `_StoreRetriever`).

### `tests/test_storage/test_scope_recall_benchmark.py` (113 linhas) — a referência de benchmark

Docstring (`:1-9`) declara a intenção: fixar o número de recall pré-filtro vs pós-filtro, "run with `-s` to see the table; the plan doc records it".

Anatomia a copiar:
- Constantes no topo: `N_ITEMS=500`, `N_NAMESPACES=20`, `N_QUERIES=30`, `K=10` (`:22-25`); dicionário `SCOPES` com 3 seletividades rotuladas (`:27-33`).
- Corpus sintético determinístico: `random.Random(3)` (`:82`), vetores gaussianos normalizados `_unit` (`:36`) — com um comentário (`:37-39`) explicando por que **não** usa o helper do conftest (a família `sin(φ+i)` colapsa num plano 2-D e produz quase-empates).
- Ground truth calculado por força bruta: `sorted(in_scope, key=lambda i: -_dot(...))[:K]` (`:96-97`).
- Adaptador `_StoreRetriever` (`:49`) implementando o `retrieve(QueryBundle, top_k)` que `evaluate_retriever` espera, com flag `post_filter` para materializar a referência ruim (`:65-69`).
- `print` da tabela (`:109`) + **dois asserts com número**: `pre >= 0.99` (`:111`) e `post <= 0.6` para os casos include (`:113`).

É exatamente o molde para o A/B "grafo vs híbrido+RRF" que o plano exige — inclusive o teste de vazamento (variar `SCOPES` e assertar que o nó excluído não aparece por travessia, backlink ou path).

## 6. Retrievers e pipeline (`src/anchor/retrieval/`)

### Protocolo — `src/anchor/protocols/retriever.py`

```python
@runtime_checkable
class Retriever(Protocol):                                   # :17
    def retrieve(self, query: QueryBundle, top_k: int = 10, *,
                 scope: RetrievalScope | None = None) -> list[ContextItem]: ...   # :20-26
```
Docstring `:33-35`: "a retriever that cannot honor it must raise" — escopo não pode ser silenciosamente descartado. `AsyncRetriever.aretrieve` em `:53` (mesma forma).

### `HybridRetriever` + RRF

- `src/anchor/retrieval/hybrid.py:17` — classe; `__init__:32`; `retrieve:57` com `scope` kw-only em `:62`; repassa a cada sub-retriever com `**scope_kwargs(scope)` em `:70`.
- **RRF: `src/anchor/retrieval/_rrf.py:14`** — `rrf_fuse(ranked_lists, weights=None, k=60, top_k=None, retrieval_method="rrf") -> list[ContextItem]`. Fórmula em `:54`: `score[id] += weight / (k + rank)`. Normalização min-max em `:65-73` (porque `ContextItem.score` é `[0,1]`), score bruto preservado em `metadata["rrf_raw_score"]` (`:81`). Fusão **por `item.id`** (`:54`) — se o grafo emitir `ContextItem`s com uuid novo em vez do `item_id` canônico, o RRF não consegue fundir com o denso/esparso. Mais um argumento para a moeda única.

### `DenseRetriever` — `src/anchor/retrieval/dense.py:19`

`retrieve:91`, `scope` kw-only em `:97`; monta kwargs condicionalmente (`:117-123`) para não quebrar `VectorStore` custom; importa `same_vault` (`:12`) para validar mounts.

Outros: `SparseRetriever` (`sparse.py`), `AsyncDenseRetriever`/`AsyncHybridRetriever` (`async_retriever.py`), `RoutedRetriever` + roteadores (`router.py`), rerankers (`rerankers.py`, `async_reranker.py`), `SharedSpaceRetriever` (`cross_modal.py`). Todos listados em `src/anchor/retrieval/__init__.py`.

### `retriever_step` — `src/anchor/pipeline/step.py:85`

```python
def retriever_step(name: str, retriever: Retriever, top_k: int = 10, *,
                   scope: RetrievalScope | None = None) -> PipelineStep:   # :85-91
    def _retrieve(items, query):
        retrieved = retriever.retrieve(
            query, top_k=top_k, **scope_kwargs(effective_scope(scope)))    # :100-102
        return items + retrieved
```

**`effective_scope(scope)` na linha `:101` é o ponto exato onde `ACTIVE_SCOPE ∩ estático` acontece.** Docstring `:94-96`: "pipeline retrieval can only narrow — the same doctrine as the tools". Gêmeo async: `async_retriever_step:108`, com `effective_scope` em `:119`. Vizinhos: `postprocessor_step:126`, `reranker_step:144`, `filter_step:189`.

### Onde um `GraphRetriever` entra

**Primeira classe é o caminho mais barato**, e por razões concretas, não estéticas:

- Implementar `Retriever` (`protocols/retriever.py:20`) dá de graça: `retriever_step` com intersecção de escopo (`step.py:101`), fusão via `rrf_fuse` dentro do `HybridRetriever` (`hybrid.py:70`), e o A/B do `ABTestRunner.run` (`ab_testing.py:171`) — que é o gate de verificação que o plano exige.
- O `graph_retrieval_step` atual **não** tem nenhuma dessas três coisas: sem escopo, sem fusão (faz `items + new_items`, `memory_steps.py:129`), sem forma testável por `evaluate_retriever`.
- Manter o step significa reimplementar `effective_scope` + fusão dentro dele.

Custo do caminho de primeira classe: um `GraphRetriever.retrieve(query, top_k, *, scope)` precisa mapear `entity_extractor(query_str)` → travessia → `item_id`s → `ContextItem`s. O passo "id → ContextItem" precisa de um `ContextStore.get(id)` (existe, `protocols/storage.py:34`, e o SQLite filtra por vault em `_context_store.py:49`), então não há peça faltando.

Recomendação: `GraphRetriever` implementando `Retriever`; manter `graph_retrieval_step` como está por compat (está exportado no `__init__` público, `src/anchor/__init__.py:670`) e marcá-lo como legado na doc.

## 7. Ingestão hierárquica

### `src/anchor/ingestion/hierarchical.py` (251 linhas)

**`ParentChildChunker:24`**, `__init__:56` (`parent_chunk_size=1024`, `child_chunk_size=256`, `parent_overlap=100`, `child_overlap=25`), valida `child < parent` (`:64`). Usa dois `FixedSizeChunker` (`:78`, `:83`).

`chunk_with_metadata(text, metadata) -> list[tuple[str, dict]]` (`:113`), o método que interessa. Metadata gravada por filho (`:145-150`):
```python
{"parent_id": <sha256(parent_text)[:16]>,   # :140
 "parent_index": parent_idx,
 "child_index": child_idx,
 "is_child_chunk": True}
```
Comentário em `:137-139` explica a escolha do content-hash: o antigo `"parent-{idx}"` colidia entre documentos, e o hash **deduplica parents idênticos de graça**. O texto do parent fica UMA vez em `self._parent_texts` (`:141`), recuperável por `get_parent(parent_id)` (`:157`) — não é copiado para cada filho (`:126-128`).

**`ParentExpander:168`** (`PostProcessor`): `process:195` troca o conteúdo do filho pelo texto do pai, dedup por `parent_id` (`:214`, `:222-225`), resolve por `parent_lookup` (`:229`) com fallback ao legado `metadata["parent_text"]` (`:231`).

**Aproveitamento para o grafo:** `metadata["parent_id"]` já é uma aresta `part_of` materializada e **estável por conteúdo** — child_item → parent_id. Duas propriedades úteis:
1. Determinística e content-addressed: reindexar o mesmo documento reproduz o mesmo `parent_id`, então a aresta não duplica (ao contrário de `ContextItem.id`, que é posicional — `{doc_id}-chunk-{i}`).
2. Já deduplica parents idênticos entre documentos, o que é literalmente "vários documentos alimentam o mesmo nó" do plano, em pequena escala.

Custo: zero extração, zero LLM. É a aresta mais barata que existe no repo hoje. Outros metadados de `extract_chunk_metadata` (`ingestion/metadata.py:43`) — `doc_id`, `chunk_index`, `total_chunks` — dão as arestas `next`/`prev` e `belongs_to_doc` no mesmo preço.

### Parser de markdown / wikilink

**Não há parser de wikilink.** Buscas em `src/`:
- `grep -rn '\[\[' src/anchor --include="*.py"` → só falsos positivos de `Callable[[...]]` (assinaturas de tipo).
- `grep -rni "wikilink|obsidian" src/anchor` → **uma única ocorrência**, e é um comentário: `src/anchor/models/scope.py:7` ("Obsidian's `path:` operator never names a vault").

`MarkdownParser` (`src/anchor/ingestion/parsers.py:70`) extrai apenas:
- frontmatter YAML via `_FRONTMATTER_RE:79` → parseado com `yaml.safe_load` (`:112`) e mergeado em metadata com `setdefault` (`:116`);
- headings via `_HEADING_RE:78` (`^(#{1,6})\s+(.+)$`) → `metadata["title"]` (primeiro h1, `:126`) e `metadata["headings"] = [{"level", "text"}]` (`:130`).

Nenhuma extração de link — nem `[[wikilink]]`, nem `[md](link)`. Outros parsers no arquivo: `PlainTextParser:36`, `_HTMLTextExtractor:138`, `HTMLParser:177`, `PDFParser:223`, `CSVParser:322`, `JSONParser:373`, `DocxParser:414`.

**Escrever o extrator de wikilink é trabalho novo**, mas trivial: um `re.findall(r"\[\[([^\]|#]+)", text)` no `MarkdownParser.parse`, gravando `metadata["wikilinks"]`. `yaml` já é dependência do core, então frontmatter com `tags:`/`aliases:` também já chega parseado — mais arestas de graça.

## 8. Dependências

`pyproject.toml`:

**Core** (`dependencies`, 2 pacotes): `pydantic>=2.8,<3`, `pyyaml>=6.0,<7`. `requires-python = ">=3.11"`. Nome do pacote: `astro-anchor`, versão `0.1.1`.

**Extras** (`[project.optional-dependencies]`, 24 no total): `tiktoken`, `bm25`, `flashrank`, `cli` (typer+rich), `anthropic`, `claude-cli`, `agents`, `all-providers`, `gemini`, `grok`, `litellm`, `mcp`, `ollama`, `openai`, `pdf`, `pricing` (genai-prices), `sqlite` (aiosqlite), `sqlite-vec`, `postgres` (asyncpg), `voyage`, `local-embeddings` (sentence-transformers), `redis`, `all-storage`, `otlp`, `docs`, `all`.

Padrão de agregador a seguir se houver `[graph]`: `all-storage = ["astro-anchor[sqlite,sqlite-vec,postgres,redis]"]` — extras compostos referenciam o próprio pacote.

**dev** (`[dependency-groups]`): pytest, pytest-asyncio, pytest-cov, ruff, mypy, rank-bm25, typer, rich, tiktoken, genai-prices, mkdocs-material, fakeredis.

**`networkx`:** **não é dependência declarada** — nem core, nem extra, nem dev. Aparece em `uv.lock:2542` apenas como transitiva de `torch` (`uv.lock:4546`), que por sua vez vem de `sentence-transformers` (extra `local-embeddings`). Ou seja: ele *está* no `.venv` de dev, mas um usuário que instale `astro-anchor` puro não o tem. Adicionar `[graph]` continua sendo uma dependência nova de verdade.

Config: ruff `line-length=100`, target `py311`, lint com `["E","F","W","I","N","UP","B","A","SIM","RUF","C90","PT","S"]`; mypy `strict = true`; pytest `asyncio_mode = "auto"`, `addopts = "--strict-markers -v"`; coverage `fail_under = 80`.

**Contagem:**
- `src/anchor`: **30 931 linhas** em 155 arquivos `.py`.
- Testes: **162 arquivos** `tests/**/test_*.py`, **42 544 linhas** de `.py` em `tests/`.
- `src/anchor/cli.py`: 359 linhas; `src/anchor/storage/`: 4055; `src/anchor/evaluation/`: 1348.

## 9. Docs a atualizar

Menções a graph/grafo/`SimpleGraphMemory` em `docs/docs/`:

| Arquivo:linha | O que diz | Ação |
|---|---|---|
| `docs/docs/guides/memory.md:143` | seção `## SimpleGraphMemory` | **reescrever** — exemplo completo em `:149-166` usa `add_entity`/`add_relationship`/`link_memory("alice", "mem-001")`/`get_related_memory_ids`. Todo o snippet muda com item_id |
| `docs/docs/guides/memory.md:16-17` | "3. **Graph memory** — an optional entity-relationship graph (`SimpleGraphMemory`)" | atualizar a introdução |
| `docs/docs/api/memory.md:129-134` | `## SimpleGraphMemory` + assinatura `SimpleGraphMemory()` | **reescrever** — a assinatura ganha store/vault |
| `docs/docs/api/pipeline.md:331-338` | `### graph_retrieval_step(graph, store, entity_extractor, ...)` + tabela de params | atualizar ou marcar legado |
| `docs/docs/guides/pipeline.md:108` | lista `graph_retrieval_step` entre os steps de memória | atualizar |
| `docs/docs/faq.md:156-158` | `??? question "What is graph memory?"` | reescrever a resposta |
| `docs/docs/guides/index.md:18` | "Sliding window, summary buffer, and graph memory" | frase |
| `docs/docs/getting-started/first-pipeline.md:471` | link "Memory Guide — ... and graph memory" | frase |
| `docs/docs/changelog.md:106` | "Simple graph memory with BFS traversal..." | histórico, **não mexer**; adicionar entrada nova |
| `docs/docs/cookbook/production-patterns.md:480` | `async_retriever_step("graph-db", async_graph_retriever, top_k=5)` | exemplo com retriever de grafo externo — **candidato a virar o exemplo do `GraphRetriever` nativo** |
| `docs/docs/index.md:186` | `graph LR` | falso positivo (Mermaid) |
| `docs/docs/llms.txt` | menciona graph | regenerar |

Docs que já falam de escopo/vault e precisarão da seção de grafo: `docs/docs/api/agent.md`, `api/pipeline.md`, `api/storage.md`, `api/models.md`, `api/protocols.md`. **Não existe `docs/docs/api/scope.md`** — o escopo está espalhado por esses cinco.

Planos correlatos em `docs/plans/`: `2026-08-28-v020-3-vault-namespace.md` (a frente #3, com a semântica canônica), `2026-08-28-v020-roadmap.md`, `2026-03-14-storage-layer-gaps-design.md` e `-storage-layer-gaps.md` (spec do `GraphStore`, escrita antes da frente #3 — o `GraphStore` de lá **não conhece vault nem namespace**).

---

# Parte 4 — Parecer sobre o worktree feature/storage-layer-gaps


**Data:** 2026-09-02 · **Revisor:** subagente (leitura apenas; nenhuma árvore foi alterada)
**Merge-base real:** `41383063` (o prompt dizia `4133806` — dígito trocado; `git merge-base HEAD dev` confirma).
**Distância:** branch 17 commits à frente, **64 atrás** da dev (`git rev-list --count`).

> **Parecer em uma linha:** descartar e reescrever `GraphStore` na dev usando o branch como referência (esqueleto de 3 tabelas + CTE recursivo + cenários de teste). Nada vale cherry-pick hoje. Custo: 8–12 h, que é o que a frente #4 já orça.

---

## 1. Inventário

`git diff --stat 41383063 1f0afb0` → 24 arquivos, +2700 / −156.

| Arquivo | Δ | O que faz |
|---|---|---|
| `CHANGELOG.md` | +17 | Entrada "Unreleased" para os 3 protocolos |
| `docs/superpowers/specs/2026-03-14-storage-layer-gaps-design.md` | ±1 | Edição de 1 linha; a dev moveu o doc para `docs/plans/` |
| `src/anchor/__init__.py` | +16 | Exporta `GraphStore`, `ConversationStore`, `Async*`, `InMemoryGraphStore`, `InMemoryConversationStore` |
| `src/anchor/cache/__init__.py` | +12 | Exporta `SqliteCacheBackend`; `RedisCacheBackend` opcional via try/except |
| `src/anchor/cache/redis_backend.py` | +69 | `RedisCacheBackend`: get/set/invalidate/clear, SETEX p/ TTL, `scan_iter` p/ clear |
| `src/anchor/cache/sqlite_backend.py` | +64 | `SqliteCacheBackend` sobre tabela `cache_entries` |
| `src/anchor/memory/graph_memory.py` | 260 mod | `SimpleGraphMemory(store=None)`: delega ao `GraphStore` **ou** mantém fallback em memória; ganha `relation_filter` e `remove_edge` |
| `src/anchor/memory/manager.py` | +63 | `conversation_store` / `session_id` / `auto_persist`; `save()` / `load()` |
| `src/anchor/protocols/storage.py` | +244 | `GraphStore`, `AsyncGraphStore`, `ConversationStore`, `AsyncConversationStore` |
| `src/anchor/storage/_serialization.py` | +50 | `conversation_turn_to_row` / `row_to_conversation_turn`, `summary_tier_to_row` / `row_to_summary_tier` |
| `src/anchor/storage/memory_store.py` | +225 | `InMemoryGraphStore`, `InMemoryConversationStore` |
| `src/anchor/storage/postgres/__init__.py` | +4 | Exports |
| `src/anchor/storage/postgres/_conversation_store.py` | +135 | `PostgresConversationStore` (async) |
| `src/anchor/storage/postgres/_graph_store.py` | +183 | `PostgresGraphStore` com `WITH RECURSIVE` |
| `src/anchor/storage/postgres/_schema.py` | +56 | DDL `graph_nodes`, `graph_edges`, `graph_memory_links`, `conversation_turns`, `summary_tiers` |
| `src/anchor/storage/sqlite/__init__.py` | +9 | Exports |
| `src/anchor/storage/sqlite/_conversation_store.py` | +244 | `SqliteConversationStore` + `AsyncSqliteConversationStore` |
| `src/anchor/storage/sqlite/_graph_store.py` | +341 | `SqliteGraphStore` + `AsyncSqliteGraphStore` (BFS em Python) |
| `src/anchor/storage/sqlite/_schema.py` | +50 | Mesmas 5 tabelas + `cache_entries` + índices |
| `tests/test_cache/test_redis_backend.py` | +71 | Redis mockado (`MagicMock`) |
| `tests/test_cache/test_sqlite_backend.py` | +64 | SQLite em tmp |
| `tests/test_integration/test_persistent_memory.py` | +102 | `MemoryManager` + `InMemoryConversationStore` (save/load, auto_persist, tiers) |
| `tests/test_storage/test_conversation_store.py` | +246 | InMemory + SQLite + Postgres (skip sem DSN) |
| `tests/test_storage/test_graph_store.py` | +329 | InMemory + SQLite + compat `SimpleGraphMemory` + Postgres (skip sem DSN) |

Distribuição aproximada das 2700 linhas: **GraphStore ≈ 1460** · **ConversationStore ≈ 1010** · **Cache ≈ 280** · exports/changelog ≈ 50.

### 1.1 `GraphStore` — API (`protocols/storage.py:384-469`, async espelhado em 473-527)

```python
add_node(node_id: str, metadata: dict | None = None) -> None          # merge de metadata se existe
add_edge(source, relation, target, metadata: dict | None = None) -> None  # auto-cria nós
get_neighbors(node_id, max_depth=1, relation_filter: str|list[str]|None=None) -> list[str]
get_edges(node_id) -> list[tuple[str, str, str]]                       # (source, relation, target)
get_node_metadata(node_id) -> dict | None
link_memory(node_id, memory_id) -> None                                # KeyError se nó não existe
get_memory_ids(node_id, max_depth=1) -> list[str]
remove_node(node_id) -> None                                           # cascata: arestas + links
remove_edge(source, relation, target) -> bool
list_nodes() -> list[str]; list_edges() -> list[tuple]; clear() -> None
```

**Modelo de nó** — `graph_nodes(node_id TEXT PRIMARY KEY, metadata_json TEXT)` (`sqlite/_schema.py:56-59`; Postgres `metadata JSONB`, `postgres/_schema.py:93-96`). O `node_id` é a *string da entidade* (ex.: `"alice"`). **Sem** `label`, **sem** `type` (só dentro do JSON), **sem** `vault`, **sem** `namespace`.

**Modelo de aresta** — `graph_edges(id AUTOINCREMENT, source, relation, target, metadata_json, UNIQUE(source, relation, target))` (`sqlite/_schema.py:60-67`; `postgres/_schema.py:99-106`). Tem **tipo** (`relation`). **Não tem** `weight`, `valid_from`/`valid_to`, `provenance`, `vault`. O `metadata_json` da aresta é **write-only**: nenhum método lê (`get_edges`/`list_edges` devolvem 3-tuplas).

**O que liga** — `graph_memory_links(node_id, memory_id) PRIMARY KEY` (`sqlite/_schema.py:68-72`; `postgres/_schema.py:109-113`). Alvo é **`MemoryEntry.id`**, sem FK, sem vault. O consumidor é `graph_retrieval_step` (dev `pipeline/memory_steps.py:94`), que resolve os ids via `MemoryEntryStore.list_all()`.

**Travessia** —
- `InMemoryGraphStore`: reconstrói o dict de adjacência **a cada chamada** (`memory_store.py:193-205`, chamado em `:214`), depois BFS (`:216-228`). Há uma segunda cópia `_get_neighbors_unlocked` (`:262-280`) que ignora `relation_filter`.
- `SqliteGraphStore`: BFS em Python, **uma ida ao SQL por nó visitado** (`UNION` de saída/entrada) — `sqlite/_graph_store.py:64-98` (sync) e `:225-261` (async, cópia idêntica).
- `PostgresGraphStore`: `WITH RECURSIVE reachable(node, depth, path)` + `CROSS JOIN LATERAL` unindo `e.target` e `e.source`, ciclo evitado por `NOT neighbor = ANY(r.path)` — `postgres/_graph_store.py:69-86` (com filtro) e `:89-106` (sem filtro; só difere pelo predicado `relation = ANY($3)`). `SELECT DISTINCT node … WHERE node != $1` (`:85`, `:105`). Observação: array de caminho enumera **todos os caminhos simples**, não um conjunto de visitados — em grafo denso com `max_depth ≥ 3` explode.

### 1.2 `ConversationStore` — API (`protocols/storage.py:531-585`, async 589-621)

```python
append_turn(session_id, turn: ConversationTurn) -> None
load_turns(session_id, limit: int | None = None) -> list[ConversationTurn]   # cronológico
save_summary_tiers(session_id, tiers: dict[int, SummaryTier | None]) -> None # substitui tudo
load_summary_tiers(session_id) -> dict[int, SummaryTier | None]              # {1,2,3}
truncate_turns(session_id, keep_last: int) -> None
delete_session(session_id) -> bool; list_sessions() -> list[str]; clear() -> None
```

**Modelo de turno** — `ConversationTurn(role: Role, content, token_count, timestamp, metadata)` (dev `models/memory.py:26-33`): **texto puro**; não há `tool_call_id`, `tool_calls`, resultado estruturado. Tabela `conversation_turns(id, session_id, turn_index, role, content, token_count, metadata_json, created_at, UNIQUE(session_id, turn_index))` (`sqlite/_schema.py:73-83`). **Postgres não tem o UNIQUE** (`postgres/_schema.py:120-131`). `turn_index` = `SELECT MAX+1` seguido de `INSERT` (`sqlite/_conversation_store.py:35-47`; `postgres/_conversation_store.py:24-37`) → corrida: SQLite levanta `IntegrityError`, Postgres duplica em silêncio.

**Tiers** — `summary_tiers(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) PK(session_id, tier_level)` (`sqlite/_schema.py:84-93`).

**Onde o Agent a consumiria** — **não consome.** O único consumidor é `MemoryManager` (branch `manager.py:41-73`, `:126-133`, `:292-330`). O `Agent` da dev só fala com `MemoryManager.add_user_message/add_assistant_message/add_tool_message` (`agent/agent.py:2273`, `:2190`, `:1616`). Tool calls chegam à memória como **texto achatado e truncado** `"[Tool: name] Input: … → Result: …"` (`agent.py:1606-1616`). Logo, com `auto_persist=True` o store recebe turnos de texto — **não é** um log estruturado de tool calls cross-turn.

**HITL durável** — na dev a aprovação é callback inline que **bloqueia o turno** ("an approval may stay pending indefinitely — the turn waits", `agent.py:1025-1045`; `with_approval` em `:470-483`). Não há estado durável. O roadmap v0.2 declara "Execução durável estilo Temporal" **fora de escopo** (`docs/plans/2026-08-28-v020-roadmap.md:38-42`). O design do branch empurrou "Agent state checkpointing" para uma "Spec 2" (`2026-03-14-storage-layer-gaps-design.md:539-547`) que **nunca foi escrita** (nenhum arquivo na dev). Conclusão: `ConversationStore` é storage isolado, ligado só ao `MemoryManager`; não é o caminho do HITL nem do histórico de tool calls.

### 1.3 O resto

- `RedisCacheBackend` (`cache/redis_backend.py:14-69`): chave `{prefix}cache:{key}`; independente do layout `ctxv:` da dev (não colide). Não é vault-aware — se a chave do passo de pipeline não incluir vault, cache de um vault serve outro (problema da chave, não do backend).
- `SqliteCacheBackend` (`cache/sqlite_backend.py:25-26`) chama `storage.sqlite._schema.ensure_tables` → cria **todo o schema de storage** (e na dev rodaria a migração de vault) só para ter uma tabela de cache.
- Nenhum dos três (cache Redis/SQLite, ConversationStore) aparece em qualquer frente do roadmap (`grep -rn ConversationStore docs/plans/2026-08-28-*` → só a menção do #4 ao branch).

---

## 2. Drift contra a dev

### 2.1 Conflitos reais

`git merge-tree --write-tree --name-only 1f0afb0 dev` → árvore `6fb945e`, **6 conflitos**:

| Arquivo | Hunk (linhas na árvore mesclada) | Natureza | Esforço |
|---|---|---|---|
| `CHANGELOG.md` | 25–90 | Duas seções "Unreleased" | trivial |
| `src/anchor/protocols/storage.py` | 12–17 | Só a linha de import (`ConversationTurn, SummaryTier` × `ROOT_NAMESPACE, RetrievalScope`) | trivial |
| `src/anchor/storage/memory_store.py` | 17–28 | Só imports (branch × `scope`/`_where` da dev) | trivial |
| `src/anchor/storage/sqlite/__init__.py` | 40–44 | Uma linha do `__all__` (`SqliteGraphStore` × `SqliteVecVectorStore`) | trivial |
| `src/anchor/storage/sqlite/_schema.py` | 58–109 | Branch acrescenta 5 entradas a uma **lista**; dev converteu `_TABLES` em **dict** (`:13-58`), adicionou `_SCOPED_TABLES` (`:61`) e `rebuild_with_scope_key` (`:91-121`) | médio: reescrever o DDL como entradas do dict |
| `src/anchor/storage/postgres/_schema.py` | 104–252 | Bloco inteiro do branch × bloco HNSW/GIN/`_migrate_scoped_table` da dev (`:104-189`) | médio |

Os outros **18 arquivos aplicam limpos** — 4 porque a dev não os tocou (`graph_memory.py`, `manager.py`, `cache/__init__.py`, `postgres/__init__.py`: `git diff --stat 41383063 dev -- …` vazio) e 14 porque são novos. Resolução textual: ~1 h. Mas isso é o menor problema.

### 2.2 O que a dev ganhou e o branch não sabe

`git diff --stat 41383063 dev -- src/anchor/{storage,protocols,models,memory,cache}` → **29 arquivos, +1641/−260**. Nos arquivos que o branch toca: `protocols/storage.py` +41, `memory_store.py` +67, `postgres/_schema.py` +114, `sqlite/_schema.py` +126, `_serialization.py` +4, `__init__.py` +89. Novos na dev: `models/scope.py` (177: `DEFAULT_VAULT`, `validate_vault`, `RetrievalScope`, `ACTIVE_SCOPE`), `storage/_where.py` (179: `matches_where`, `sql_where_clauses`, `scope_sql_clauses`), `sqlite/_vec_store.py` (313).

Padrão que **todo store da dev** segue agora e o branch ignora:
- vault ligado na construção: `SqliteContextStore(conn_manager, *, vault=DEFAULT_VAULT)`, `validate_vault` no `__init__`, propriedade `.vault`, carimbo no `add`, `AND vault = ?` em todo SELECT/DELETE (`sqlite/_context_store.py:20-77`; `memory_store.py` dev `:29-51`).
- PK composta `(vault, id)` e migração de shape legado (`sqlite/_schema.py:91-121`; `postgres/_schema.py:134-189`).
- `PostgresConnectionManager.initialize(**kwargs)` com `server_settings` (`postgres/_connection.py:43-69`) — a API `acquire()` que o branch usa **ainda existe** (`:83`), então o código do branch *importa e roda*; só não respeita o vault.
- `anchor migrate` (`cli.py:232`), Redis `ctxv:{vault}:{id}` (`redis/_context_store.py:42-67`).

Nota honesta: na dev, `_entry_store.py` e `_document_store.py` (SQLite e Postgres) **também ainda não** são vault-bound (`grep vault` vazio). A regra "vault em todo store" foi aplicada a `ContextStore`/`VectorStore`; o #4 exige que o grafo nasça com ela.

### 2.3 Quanto do branch sobrevive a vault + item_id

Contei os SQL strings do grafo: ~25 no `SqliteGraphStore`, duplicados no async (~50), ~20 no Postgres. **Todos** ganham `vault` (INSERT, WHERE, chaves, `INSERT OR IGNORE` dos nós). `graph_memory_links.memory_id` vira `item_id` e os 2 métodos do protocolo + 3 backends + `SimpleGraphMemory` + `graph_retrieval_step` mudam de nome/alvo. As 3 tabelas trocam de PK. As 6 travessias (4 BFS Python + 2 CTE) precisam de um predicado de visibilidade por salto (ver §4e). O `InMemoryGraphStore` guarda dicts sem vault e arestas como `list[tuple]` + `_edge_metadata` por 3-tupla (`memory_store.py:162-165`) — não comporta vault nem duas versões da mesma tripla.

Estimativa: **20–30 % das ~1460 linhas do GraphStore sobrevivem verbatim** (esqueleto de classe, uso do connection manager, auto-criação de nó, esqueleto do CTE, cenários de teste). Do ConversationStore, ~80 % sobreviveria *se* fosse desejado — mas não é pedido por nenhuma frente.

---

## 3. Qualidade

### 3.1 Testes

```
cd .worktrees/storage-layer-gaps && uv run pytest tests/test_storage/test_graph_store.py \
  tests/test_storage/test_conversation_store.py -q -x --no-header -p no:cacheprovider
# (uv criou .venv do worktree em CPython 3.11.15, 55 pacotes)
58 passed, 10 skipped in 0.07s
```

Rodam e passam. Os 10 skips são as classes Postgres (`postgres_only`, `test_graph_store.py:280`; `ANCHOR_TEST_POSTGRES_DSN` ausente) — ou seja, **o CTE recursivo, a única peça de valor real, nunca é exercido** em CI. A dev tem um fake de conexão asyncpg em processo (`tests/test_storage/test_review_fixes.py`, commit `3f2c8d9`) que o branch não usa.

### 3.2 Lente ponytail

1. **Três grafos em memória convivendo.** Fallback dentro de `SimpleGraphMemory` (`graph_memory.py:35-40`, e cada método vira `if self._store: … else: <cópia>`), `InMemoryGraphStore` (`memory_store.py:155-321`) e a BFS duplicada `_get_neighbors_unlocked` (`:262-280`). O design dizia "SimpleGraphMemory *becomes* the InMemory GraphStore" (`design.md:125`); a implementação manteve os dois.
2. **BFS escrita 4× em Python** (`memory_store.py:207-228`, `:262-280`; `sqlite/_graph_store.py:64-98`, `:225-261`) **+ CTE 2×** (`postgres/_graph_store.py:69-86`, `:89-106`) diferindo por um predicado.
3. **Regressão de performance no caminho sem store:** `get_related_entities` agora chama `_rebuild_adjacency(relation_filter)` **incondicionalmente** (`graph_memory.py:75`); antes era guardado por `_adjacency_dirty`. A flag continua sendo escrita (`:63`, `:126`, `:135`, `:146`) e nunca lida — código morto. `InMemoryGraphStore` faz o mesmo O(E) por consulta com o lock preso (`memory_store.py:211-214`).
4. **Metadata de aresta é write-only** — abstração especulativa; o que #4 quer (validade, proveniência, peso) são colunas, não JSON.
5. **Semântica inconsistente entre backends:** `InMemoryGraphStore.add_edge` atualiza metadata ao re-inserir (`memory_store.py:188-189`); SQLite/PG usam `INSERT OR IGNORE` / `ON CONFLICT DO NOTHING` e nunca atualizam (`sqlite/_graph_store.py:57-61`; `postgres/_graph_store.py:53-58`). `conversation_turns` tem `UNIQUE` no SQLite (`sqlite/_schema.py:82`) e não no Postgres (`postgres/_schema.py:120-131`).
6. **Atomicidade:** `PostgresGraphStore.remove_node` são 3 statements em autocommit, sem `conn.transaction()` (`postgres/_graph_store.py:148-152`); `delete_session` idem (`postgres/_conversation_store.py:111-119`); só `save_summary_tiers` usa transação (`:66-78`). O contrato do protocolo promete cascata atômica (`protocols/storage.py:395`, `:543`).
7. **Async SQLite nunca cria as tabelas:** `AsyncSqliteGraphStore._ensure_tables` existe (`sqlite/_graph_store.py:183-185`) e ninguém chama; `AsyncSqliteConversationStore` não tem nem isso. Primeiro uso em DB novo falha, a menos que um store sync tenha rodado antes.
8. **`MemoryManager.save()` duplica turnos** a cada chamada quando `auto_persist=False` (`manager.py:301-303`, branch). **`load()` escreve campo privado de outra classe** (`self._conversation._tiers = tiers`, `:325`) e reproduz turnos via `_add_message` — re-tokeniza e pode disparar compactação (chamada de LLM) durante o *restore*.
9. **`SqliteCacheBackend` acopla cache ao schema de storage** (`cache/sqlite_backend.py:25-26`).
10. **Decisões que contradizem o roadmap:** liga `memory_id`, não `item_id`; sem vault; aresta sem validade/proveniência; `remove_edge` é DELETE (`sqlite/_graph_store.py:145-152`) quando #4 quer "inválida ≠ deletada".

Nada disso é "ruim de má fé" — é um branch de março escrito antes das decisões de agosto. Mas não é um branch que se conserta com um ALTER.

---

## 4. Decisões da frente #4 vs. o schema do branch

| Decisão (#4 `:57-75`) | O que o schema do branch tem | DDL | Código | Veredito |
|---|---|---|---|---|
| **a. `ContextItem` é a moeda única** (aresta → `item_id`) | `graph_memory_links.memory_id` → `MemoryEntry` (`sqlite/_schema.py:68-72`) | `RENAME COLUMN` trivial | renomear `link_memory`/`get_memory_ids` no protocolo (`:439-449`), 3 backends, `SimpleGraphMemory`, e trocar o consumidor de `MemoryEntryStore.list_all()` por `ContextStore` em `memory_steps.py:94-108` | moderado — mas o #4 já prevê exatamente essa mudança |
| **b. Vault fronteira dura em todo store** | nenhuma tabela tem `vault`; PKs são `node_id`, `(source,relation,target)`, `(node_id,memory_id)` | 3 `ADD COLUMN` + 3 PKs compostas (nada shippou → sem migração) | **todo** SQL string dos 3 backends (~70), construtores `vault=`, `InMemory` reindexado por vault | **reescrever os backends** |
| **c. Namespace pertence ao item** | nó não tem namespace — correto | nada | nada no nó; mas o link precisa ser joinável a `context_items(vault, id)` → coberto por (a)+(b) | ok |
| **d. Aresta tem validade temporal** (inválida ≠ deletada) | `graph_edges` sem `valid_from`/`valid_to`/`provenance`; `UNIQUE(source,relation,target)` | 3 `ADD COLUMN`, **mas** a UNIQUE proíbe reafirmar uma aresta invalidada (chave precisa de `valid_from` ou unique parcial `WHERE valid_to IS NULL`) | `remove_edge` vira `UPDATE valid_to`; todo SELECT de travessia ganha `valid_to IS NULL` (ou "as-of"); `InMemory` com `list[tuple]` não comporta duas versões da mesma tripla | **reescrever a aresta** |
| **e. Exclusão em nível de nó** (nó sem evidência visível não existe; nó excluído é parede) | travessia lê só `graph_edges` (`SELECT target FROM graph_edges WHERE source=?`); nenhum join com evidência | nada além de (a)+(b) | cada salto (4 BFS + 2 CTE) precisa de `EXISTS (links ⋈ context_items WHERE <scope_sql_clauses(namespace)>)`; `path` não pode atravessar nó invisível. A dev já tem o predicado pronto (`_where.py:156-179`) | **reescrever a travessia** — motivo central |
| **f. Hub calculado** | nada | nada | `COUNT(*) GROUP BY node` sobre arestas **visíveis** (depende de e) | trivial depois de (e) |

Em DDL, tudo cabe em "ALTER pequeno". Em código, (b), (d) e (e) tocam **cada query e cada travessia** — é 2/3 do GraphStore reescrito. Isso é reescrita, não ajuste.

---

## 5. Parecer

### Opções, custo e risco

| Opção | Custo | Resultado | Risco |
|---|---|---|---|
| **Merge** | ~1,5 h (6 conflitos + `_TABLES` como dict + suíte) **+ a reescrita inevitável (8–12 h)** | Publica em `anchor.__init__` um `GraphStore`/`link_memory` sem vault, contra a regra "vault validado em toda montagem" (`ef80238`), e o #4 troca a API na sessão seguinte. Leva junto `ConversationStore` com o bug do `save()` e dois cache backends que nenhuma frente pede | médio: churn de API pública + um store que ignora a fronteira de isolamento |
| **Rebase** (17 commits) | 3–4 h | Os 6 arquivos conflitam repetidas vezes (`5bd681c`, `8995c4d`, `16843fc`, `1798bef`, `e6ee3a7`, `62e935f`, `8c2cf36`, `1f0afb0`) e o resultado é idêntico ao merge | dominado pelo merge |
| **Cherry-pick** | — | Cache (`d900e6c`, `bab27ee`): YAGNI, fora do roadmap, e o SQLite acopla ao schema. ConversationStore (`16843fc`, `2b5c220`, `6485f1b`, `e8b8848`): fora do roadmap, HITL durável declarado fora da v0.2, e a integração (`f8fbee5`, `552c727`) tem o bug do `save()` e o poke em `_tiers`. GraphStore: nenhum arquivo é pickável — todos precisam de vault | **nada vale um commit hoje** |
| **Descartar e reescrever usando o branch como referência** | **8–12 h (1–2 sessões)** — o que o #4 já orça como "grande" | Grafo nasce no padrão da dev (vault na montagem, `_TABLES` dict, `scope_sql_clauses`), com `item_id`, validade e proveniência desde a primeira linha | baixo: greenfield sobre padrão consolidado; as duas decisões abertas do #4 (aresta tipada? incremental?) têm de ser respondidas antes de qualquer opção |

### Recomendação: **descartar e reescrever**

**O que copiar do branch (referência, não cherry-pick):**
- A forma de 3 tabelas — nós / arestas / links nó↔item — e a auto-criação de nós no `add_edge` via `INSERT OR IGNORE` (`sqlite/_graph_store.py:52-56`).
- O esqueleto do CTE recursivo do Postgres (`postgres/_graph_store.py:69-86`): `WITH RECURSIVE reachable(node, depth, path)` + `CROSS JOIN LATERAL` unindo `target`/`source`. Trocar o array de caminho por semântica de visitados (ou limitar `max_depth`) — `ponytail:` ceiling exponencial em grafo denso.
- Os cenários de `tests/test_storage/test_graph_store.py`: ciclo (`:108`, `:266`), profundidade 2 (`:82`), `relation_filter` str e lista (`:89-103`), link + vizinhança (`:132`), cascata do `remove_node` (`:142`).
- A ideia de `relation_filter` na travessia.

**O que fazer diferente:**
- Vault na construção (padrão `sqlite/_context_store.py:20-28`), entradas em `_TABLES` (dict) + `_SCOPED_TABLES`, PKs `(vault, …)`.
- Aresta com colunas `valid_from`, `valid_to NULL`, `provenance` (EXTRACTED/INFERRED/AMBIGUOUS) e, se o Arthur escolher aresta tipada, `relation NOT NULL`; `remove_edge` invalida.
- `link_item(node, item_id)` / `get_item_ids`; o step passa a resolver via `ContextStore`.
- **Uma** BFS em Python compartilhada por InMemory e SQLite (helper que recebe `neighbors(node) -> iterable`), **um** CTE parametrizado, predicado de visibilidade por salto vindo de `scope_sql_clauses`.
- `SimpleGraphMemory` vira fachada fina sobre um `GraphStore` (default `InMemoryGraphStore`) — a intenção original do design, sem o caminho duplo.
- Testar o Postgres com o fake asyncpg da dev, não com skip.
- Deixar `ConversationStore` e os cache backends **fora**; se alguma frente futura pedir persistência de sessão, a forma da tabela e os 4 helpers de serialização (`_serialization.py` do branch, +50 linhas) são 30 minutos de cópia.

**Antes de escrever:** responder as duas decisões abertas do #4 (`:79-84`) — aresta tipada além de validade, e incremental ou full-rebuild. A primeira muda a coluna `relation`; a segunda muda se há tabela de comunidade/versão.

**Sobre o worktree:** manter o branch como referência (não deletar), remover o worktree quando o novo `GraphStore` estiver na dev. Não fiz nenhuma dessas ações.
