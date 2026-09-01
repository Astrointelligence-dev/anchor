# v0.2 · #3 — Vault, namespace e filtragem

**Status:** em implementação (sessão 9, 2026-09-01) · **Tamanho:** médio ·
**Depende de:** nada · **Bloqueia:** #4 (grafo de conhecimento)

## Pesquisa (2026-09-01) — resultados

Cinco agentes (pré-filtro nos backends, SOTA de tenancy, mapa do código,
evidência das decisões, referências Anthropic/Obsidian). O que fechou:

- **Pré-filtro verificado por execução** (sqlite-vec 0.1.9 do venv, decoy
  em k=2 + EXPLAIN): partition key do vec0 pré-filtra; metadata com
  `>=`/`<` pré-filtra (o truque de range dá prefixo); `rowid IN
  (subquery)` faz pushdown real; **`OR` no WHERE do KNN degrada
  silenciosamente para pós-filtro** — multi-include/exclude vai sempre
  pela subquery. Pin sobe para `>=0.1.9` (bug de DELETE com TEXT >12
  chars nos 0.1.6–0.1.8).
- **pgvector**: colunas reais + btree `(vault, namespace
  text_pattern_ops)`; `hnsw.iterative_scan=relaxed_order` (0.8.0+, guard
  de versão degrada limpo); asyncpg faz RESET ALL no release do pool.
  Nenhum backend precisa de over-fetch.
- **SOTA**: partition=isolamento, filtro=atributo (todas as fontes);
  hierarquia como caminho consultado por prefixo (LangGraph BaseStore é
  o prior art em produção — btree text_pattern_ops); operadores
  convergentes `eq ne in nin gt gte lt lte` + and/or; ninguém tem
  include/exclude com exclude-vencendo nem estreitamento obrigatório —
  diferencial do anchor.
- **Referências**: Anthropic (docs + leak do source map de 31/03/2026,
  que confirma o design publicado) — fronteira dura FLAT fora da
  hierarquia (projeto/repo/handler), índice+folhas (MEMORY.md 200
  linhas + arquivos-tópico sob demanda), "isolation lives where the
  storage lives". Obsidian — vault=pasta com config própria, zero
  cross-vault, `path:` como operador de query, grafo ortogonal às
  pastas.
- **Correção do roadmap**: retrofit de vault NÃO força reescrita de ids
  (UUIDs já são globais; a colisão real é o doc_id determinístico em
  store compartilhada → chave composta ou vault-bound-store). O "#3
  antes do #4" segue válido pelo motivo certo: migração de schema
  completa + rebuild do vec0 + o grafo referenciando ids.

## Decisões do Arthur (sessão 9)

1. **Namespace = coluna TEXT indexada** (materialized path, trailing
   normalizado): range scan no SQLite, btree text_pattern_ops no
   Postgres; no sqlite-vec vive na vec_items e entra no KNN via
   subquery. (Padrão LangGraph/Obsidian `path:`.)
2. **Vault = mount ligado no construtor do store** (padrão do prefixo
   Redis já no repo): o store emite o filtro por dentro — partition key
   no vec0, coluna indexada no Postgres, prefixo de chave no Redis.
   Caller nunca escreve `WHERE vault`. Arquivo-por-vault continua
   possível no SQLite como modo de deploy (db_path), não como design do
   protocolo.
3. **Modelo de MOUNTS (decisão nova — refina o sketch da sessão 5).**
   O vault é um workspace que o agente habita (navegar nodes, pesquisar
   como no Obsidian, escrever em áreas abertas) e que também ancora o
   corpus vetorizado (ex.: documentos jurídicos). Um agente carrega um
   **conjunto de mounts** (nome → store/retriever já ligado no seu
   vault); a LLM escolhe **entre os montados pelo nome** — nunca um
   vault arbitrário. `RetrievalScope` NÃO carrega vault: é só navegação
   (`include`/`exclude` de namespaces, exclude vence). Subagente herda
   subconjunto dos mounts do pai + escopo interseccionado — "só
   estreita" vale para os dois. `GraphScope` vira alias no #4.
   Follow-up anotado (não construir agora): política de ESCRITA por
   prefixo de namespace ("nodes abertos" editáveis pela LLM vs áreas
   máquina-only do pipeline) — reusa a mesma máquina de prefixo.
4. **Migração**: sentinela `__default__` (precedente Pinecone; NULL
   quebra UNIQUE, "default" colide com vault legítimo); versionamento
   `PRAGMA user_version` / tabela `schema_version`; `anchor migrate`
   (auto no `index` sqlite, explícito no Postgres); rebuild do vec0 por
   cópia de blob (sem re-embedding); `UNIQUE(vault, id)` adiado até
   multi-vault-no-mesmo-db shipar.

## Plano de implementação (fases; teste falhando antes de cada uma)

- **Fase 1 — Modelos**: `RetrievalScope` frozen em `models/`
  (`include`/`exclude`, `matches()` com exclude-vence, `intersect()`
  que só estreita, normalização de caminho); `ContextItem` ganha
  `vault="__default__"` e `namespace="/"` (defaults ⇒ ~zero churn nos
  255 usos de teste).
- **Fase 2 — Storage**: operadores no clause builder compartilhado do
  sqlite (`$prefix`, `$in`, `$gt/$gte/$lt/$lte`, `$ne`, `$nin`) — a
  vec_store passa a usá-lo (mata a duplicação do finding 38); colunas +
  vault no construtor nos backends (in-memory, sqlite, sqlite-vec com
  partition key, postgres com iterative-scan guard, redis via prefixo);
  teste de spoilers na suíte compartilhada (exclude some de busca,
  listagem e get) + regressão do OR-trap com decoy.
- **Fase 3 — Migração**: versionamento + `anchor migrate` + rebuild do
  vec0; auto-run no `index`.
- **Fase 4 — Agente**: mounts nomeados na camada de RAG tools
  (`search` valida vault ∈ mounts), `Agent.with_scope()`,
  `SubagentDefinition.scope`/`mounts` com interseção obrigatória no
  `with_subagents` (mesmo hook do usage_limits); `where`/escopo
  atravessando `retriever_step`/`rag_tools` (hoje o where morre acima
  do Dense).
- **Fase 5 — CLI + docs**: `--vault`/`--namespace` no index,
  `--include`/`--exclude` no query; guides/API; CHANGELOG.
- **Verificação final**: spoilers em todos os backends; subagente não
  alarga; benchmark de recall pré vs pós-filtro com
  `RetrievalEvaluator`; ritual xhigh.

---

## O que é

Dois níveis de escopo sobre tudo que o anchor recupera — memória e documentos —
mais o filtro que os agentes usam pra enxergar só um pedaço.

| Nível | Análogo Obsidian | O que faz | Atravessável? |
|---|---|---|---|
| **Vault** | vault | fronteira dura de isolamento | não |
| **Namespace** | pasta dentro do vault | organização e filtro, hierárquico | sim |

O ponto que define o desenho: **namespace não pode ser a fronteira de
isolamento.** Se fosse, documentos de pastas diferentes não conseguiriam
alimentar o mesmo nó de conhecimento — que é justamente o objetivo. A fronteira
mora um nível acima, no vault.

```
vault "dnd"                        ← nada aqui toca o vault "trabalho"
  /regras/combate        ─┐
  /campanha-1/sessoes    ─┼─→ todos alimentam  [Dragão Vermelho]   (um nó só)
  /bestiario             ─┘
```

---

## Contexto (o que já existe)

| Peça | Onde | Estado |
|---|---|---|
| `VectorStore.search(where=...)` | `protocols/storage.py:105` | **só igualdade**, pré-filtro |
| `ContextItem.metadata` | `models/context.py:63` | dict livre |
| Prefixo de chave | `storage/redis/_connection.py:19` | única coisa parecida com namespace |
| Backends | `storage/{sqlite,postgres,redis}` | in-memory, sqlite-vec, pgvector |

Não existe namespace, vault, tenant nem partição em lugar nenhum.

---

## Pesquisa a fazer

1. **Pré-filtro vs pós-filtro.** No Pinecone o filtro de metadata é aplicado
   *pós-retrieval*, e degrada recall quando é seletivo. O Qdrant resolve com
   pré-filtro HNSW (ACORN), mas exige projetar o índice de filtro antes.
   **Como fazer isso em sqlite-vec e pgvector?** Essa é a pergunta central da
   sessão — o `where` de igualdade de hoje não serve para namespace hierárquico.
2. **Namespace vs metadata filter.** A orientação do mercado é: namespace para
   partição dura, filtro para atributo. Nosso vault é a partição dura e o
   namespace é... qual dos dois? Provavelmente um índice dedicado, não
   metadata solta.
3. **Escala.** Pinecone: um namespace por tenant funciona até poucos milhares.
   Qdrant: payload filtering numa coleção escala melhor. Que ordem de grandeza
   o anchor precisa suportar?
4. **Operadores.** `$in`/`$nin` têm limite de 10k valores no Pinecone. Que
   conjunto mínimo de operadores vale implementar — `$in`, `$gt/$lt`, `$and/$or`,
   prefixo hierárquico?

---

## Decisões já tomadas

- **Vault é raso e burro**, namespace é hierárquico e esperto. O vault é quase
  de graça (uma coluna, um argumento de construtor) mas **não dá pra retrofitar
  barato** — adicionar fronteira de isolamento depois obriga a reescrever todo
  id e migrar o que já foi indexado. Por isso entra desde o começo.
- Namespace é **prefixo hierárquico**: `/campanha-1` pega `/campanha-1/sessoes/...`.
- `GraphScope` é **um objeto**, não três APIs:

```python
GraphScope(vault="dnd")                                # vault inteiro
GraphScope(vault="dnd", include=["/campanha-1"])       # namespace específico
GraphScope(vault="dnd", exclude=["/spoilers"])         # vault menos exclusões
```

- **`exclude` sempre ganha de `include`.** Sem essa regra,
  `include=["/campanha-1"], exclude=["/campanha-1/spoilers"]` é ambíguo.
- Escopo em subagente **só estreita**: o efetivo é a interseção com o do pai.

## Decisões em aberto

- [ ] Namespace vira coluna indexada, tabela própria, ou metadata com índice?
- [ ] Um vault por store, ou vault como coluna numa store compartilhada?
- [ ] `GraphScope` vive em `models/` (é escopo de recuperação, não só de grafo)?
      Talvez o nome certo seja `RetrievalScope`.
- [ ] Migração: o que acontece com dados já indexados sem vault?

---

## Escopo

- [x] `vault` e `namespace` no `ContextItem` e nos stores (Fases 1-2)
- [x] Filtro hierárquico por prefixo, **pré-filtro no índice** (ranges
      boundary-aware; vec0 via subquery pushdown — OR direto no KNN é
      pós-filtro silencioso, verificado)
- [x] Operadores no `where`: core convergente eq/ne/in/nin/gt/gte/lt/lte
- [x] `RetrievalScope` com include/exclude, exclude vence, intersect()
      só estreita (GraphScope vira alias no #4)
- [x] Escopo por agente (`with_scope`, publicado na janela da tool call)
      e por `SubagentDefinition.scope`, interseção obrigatória provada
      por mutação
- [x] Backends: in-memory, sqlite, sqlite-vec (partition key), pgvector
      (iterative scan 0.8+ c/ guard), redis (vault na chave)
- [x] Migração in-place (`ensure_tables` + rebuild do vec0 por blob copy
      + `anchor migrate`); sentinela `__default__`
- [x] Mounts nomeados no `rag_tools` (LLM escolhe entre os montados)
- [x] CLI `--vault/--namespace/--include/--exclude` + smoke end-to-end

**Fora:** ACL por usuário, criptografia por vault, cotas.

## Verificação

- Teste que um item em `/spoilers` **não aparece** em nenhuma consulta com
  `exclude=["/spoilers"]` — nem por busca, nem por listagem, nem por id.
- Teste que subagente não consegue alargar o escopo do pai.
- Benchmark de recall com filtro seletivo (é o risco que a pesquisa aponta):
  medir pré-filtro vs pós-filtro num golden set com `RetrievalEvaluator`.

## Review

_(preencher ao final)_
