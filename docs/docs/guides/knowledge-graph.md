# Knowledge Graph

anchor keeps a knowledge graph over **memory and documents at the same
time**: entities and notes are nodes, relations are typed edges, and every
node or edge is backed by the `ContextItem`s that evidence it. You navigate
(`neighbors`, `backlinks`, `path`, `explain`) instead of only searching, and
the same `RetrievalScope` that governs retrieval governs the graph — a node
without visible evidence does not exist for that scope, and is never
crossed.

```python
from anchor import KnowledgeGraph

graph = KnowledgeGraph()                       # in-memory store by default
graph.add_node("Alice", metadata={"type": "person"})
graph.link_item("Alice", "mem-001")            # the item id is the currency
graph.add_edge("Alice", "works on", "Project X", evidence=["mem-001"],
               fact="Alice leads Project X")

graph.neighbors("alice", max_depth=2)          # ['project_x']
graph.related_items("Project X")               # ['mem-001']
graph.explain("alice", "project x")            # [GraphEdge(relation='works_on', ...)]
graph.query(["alice"], top_k=5)                # [('mem-001', 0.58)]  ← PPR scores on items
```

## The model

| Piece | What it is |
|---|---|
| `GraphNode` | `id` is a canonical key (`normalize_key`: NFKC + casefold + whitespace → `_`, so `"Project X"` and `"project x"` are one node), `label` the display name, `aliases`, `metadata`. |
| `GraphEdge` | `source → target` with a free normalized `relation` (`"Works On"` → `works_on`; no closed enum), an optional `fact`, `provenance` (`extracted` / `inferred` / `ambiguous`), `confidence`, `evidence` (item ids) and bi-temporal validity: `valid_from`/`valid_to` (world time), `created_at`/`invalidated_at` (transaction time). |
| Evidence | The `ContextItem` ids that support a node or edge. For documents these are chunk ids; for memory the item id **is** `MemoryEntry.id`. |

An obsolete edge is **invalidated, never deleted** — `invalidate_edge()`, or
`valid_to` from the text. Every read takes `as_of` and only walks edges live
at that instant.

## Scope: the node is the unit of exclusion

```python
from anchor import RetrievalScope

no_secret = RetrievalScope(exclude=("/secret",))
graph.nodes(scope=no_secret)          # the villain evidenced only by /secret is gone
graph.path("hero", "city", scope=no_secret)   # None if the only route crossed the villain
```

Rules, identical on every backend:

- an item is visible when the scope matches its namespace;
- a node is visible when at least one of its items is (a node without any
  evidence exists only unscoped);
- an edge is visible when it is live at `as_of`, both endpoints are visible
  and at least one evidence item is visible.

A hidden node is a wall, not a bridge: `path` and `explain` refuse to cross
it, `backlinks` and `hubs` never list it, `query` never scores its items.
Because subagents inherit the parent's scope intersected with their own
(`Agent.with_scope`, `SubagentDefinition.scope`), a child can never see more
of the graph than its parent.

## Building the graph

### From documents — free, deterministic

`GraphIndexer` runs extractors over `ContextItem`s and writes into the graph.
The two default extractors cost nothing:

- `StructureExtractor` — the chunk's note (its file name) becomes a node
  evidenced by the chunk; frontmatter `aliases` become node aliases and
  `tags` become `tagged` edges.
- `WikilinkExtractor` — every `[[target]]` (also `|alias`, `#heading`,
  `#^block`, `![[embed]]`) becomes a `links_to` edge from the chunk's note,
  and the target is linked to the chunk as a mention. Resolution is by
  basename, case-insensitive, like Obsidian.

```python
from anchor import DocumentIngester, GraphIndexer, KnowledgeGraph

items = DocumentIngester().ingest_directory("vault/")
graph = KnowledgeGraph()
GraphIndexer(graph).index(items)     # add-only, idempotent: reindexing merges evidence
graph.unlink_item(items[0].id)       # removal: edges left without evidence are invalidated
```

### From memory — opt-in, one model call per fact

`LLMGraphExtractor(llm)` extracts entities (name, type, aliases) and typed
relations (`relation`, `fact`, `provenance`, `confidence`) in a single call,
with a suggested vocabulary (`DEFAULT_RELATIONS`) and no second "gleaning"
pass. Give it to a `MemoryManager` and the graph follows the persistent
store: added facts are indexed, updated facts re-extracted, deleted facts
take their evidence with them.

```python
from anchor import GraphIndexer, KnowledgeGraph, LLMGraphExtractor, MemoryManager

graph = KnowledgeGraph()
manager = MemoryManager(
    persistent_store=store,
    graph=GraphIndexer(graph, extractors=[LLMGraphExtractor(agent.llm)]),
)
manager.add_fact("Ana Lima leads Team Payments and is on call for Billing.")
graph.explain("ana lima", "billing")   # [on_call_for · extracted · 0.95 · evidence=('<entry id>',)]
```

The extractor is opt-in on purpose: on a 40-note wiki the wikilink graph did
not beat hybrid BM25+dense retrieval (recall@5 0.94 for both, hybrid ahead on
MRR — the numbers live in the plan doc), so the graph's value is navigation,
scope and memory without links, not replacing your retriever.

## Retrieval and fusion

`GraphRetriever` is a first-class `Retriever`: seeds are the nodes the query
mentions (an alias-aware n-gram matcher, or your `entity_extractor`) plus,
optionally, the nearest passages from a vector store; personalized PageRank
over the entity–item graph scores every visible item. Items keep their
canonical ids, so `HybridRetriever` fuses them with dense and sparse results
by RRF.

```python
from anchor import GraphRetriever, HybridRetriever

graph_ret = GraphRetriever(graph, context_store, vector_store=vectors, embeddings=emb)
fused = HybridRetriever([hybrid, graph_ret], weights=[1.0, 0.5])
```

`graph_retrieval_step(graph, memory_store, entity_extractor)` remains for the
memory-only pipeline shape; it honours the published scope like every other
step.

## Communities and hubs

`hubs()` ranks visible nodes by degree. `communities()` partitions the visible
subgraph — a deterministic Louvain in the core, Leiden through igraph
when the `[graph]` extra is installed — and caches the result on the store's
`version`, so derived data is never persisted and never stale.

```bash
pip install "astro-anchor[graph]"      # Leiden + C-speed algorithms via igraph
```

## Persistence and the CLI

`SqliteGraphStore` (+ `AsyncSqliteGraphStore`) and `PostgresGraphStore`
persist the graph next to the index, vault-bound at construction like every
store. The CLI builds and navigates it:

```bash
anchor index vault/ --db anchor.db --graph --namespace /wiki
anchor graph path checkout ledger --db anchor.db
anchor graph explain checkout ledger --db anchor.db --exclude /incidents
anchor graph hubs --db anchor.db -k 10
anchor graph communities --db anchor.db
anchor graph backlinks payments --db anchor.db
anchor graph query "who is on call for checkout?" --db anchor.db
```

## What's next

- [Memory](memory.md) — the store the graph follows
- [Retrieval](retrieval.md) — hybrid retrieval and RRF
- [API: Knowledge Graph](../api/graph.md)
