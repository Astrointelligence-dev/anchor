# v0.2 · #3 — Vault, namespace e filtragem

**Status:** não iniciado · **Tamanho:** médio · **Depende de:** nada
**Bloqueia:** #4 (grafo de conhecimento)
**Sessão:** abrir com pesquisa, depois implementar.

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

- [ ] `vault` e `namespace` no `ContextItem` e nos stores
- [ ] Filtro hierárquico por prefixo, **pré-filtro no índice** (não pós)
- [ ] Operadores além de igualdade no `where` (conjunto definido pela pesquisa)
- [ ] `GraphScope`/`RetrievalScope` com `include`/`exclude` e precedência
- [ ] Escopo por agente e por `SubagentDefinition`, com interseção obrigatória
- [ ] Implementação nos backends: in-memory, sqlite-vec, pgvector
- [ ] Migração dos dados existentes

**Fora:** ACL por usuário, criptografia por vault, cotas.

## Verificação

- Teste que um item em `/spoilers` **não aparece** em nenhuma consulta com
  `exclude=["/spoilers"]` — nem por busca, nem por listagem, nem por id.
- Teste que subagente não consegue alargar o escopo do pai.
- Benchmark de recall com filtro seletivo (é o risco que a pesquisa aponta):
  medir pré-filtro vs pós-filtro num golden set com `RetrievalEvaluator`.

## Review

_(preencher ao final)_
