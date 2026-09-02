# Knowledge Graph

`anchor.graph`, `anchor.models.graph`, the graph stores and extractors.

## Models (`anchor.models.graph`)

### `normalize_key(text) -> str`

Canonical key for node ids and relation labels: NFKC + casefold, then every run
of whitespace, hyphens and underscores becomes one `_` (`auth-service`, `Auth Service`
and `AUTH_SERVICE` are one node; `C++`/`C#`/`.NET` stay distinct). Raises `ValueError`
on empty input.

### `GraphNode`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | (required) | Canonical key (normalized on construction). |
| `label` | `str` | the raw `id` | Display name. |
| `aliases` | `tuple[str, ...]` | `()` | Other spellings; the mention matcher uses them. |
| `metadata` | `dict[str, Any]` | `{}` | Free metadata (`kind`, `type`, ...). |

### `GraphEdge`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | uuid4 | |
| `source`, `target` | `str` | (required) | Node ids (normalized). |
| `relation` | `str` | (required) | Free label, normalized (`works_on`). |
| `fact` | `str \| None` | `None` | Evidence sentence for `explain()`. |
| `provenance` | `"extracted" \| "inferred" \| "ambiguous"` | `"extracted"` | Where it came from. |
| `confidence` | `float` | `1.0` | `[0, 1]`. |
| `evidence` | `tuple[str, ...]` | `()` | `ContextItem` ids supporting the edge (deduplicated). |
| `valid_from`, `valid_to` | `datetime \| None` | `None` | World time. |
| `created_at`, `invalidated_at` | `datetime` / `datetime \| None` | now / `None` | Transaction time. |
| `metadata` | `dict[str, Any]` | `{}` | |

`weight` (property) is `max(1, len(evidence))`; `is_live(as_of=None)` is
`invalidated_at is None and valid_from <= as_of < valid_to`.

### `Subgraph`

`nodes: dict[str, GraphNode]`, `edges: list[GraphEdge]`, `items: dict[str, tuple[str, ...]]`
— everything visible under a scope at a time; what the algorithms consume.

## `KnowledgeGraph` (`anchor.graph`)

```python
KnowledgeGraph(store: GraphStore | None = None)   # default: InMemoryGraphStore()
```

Every read takes `scope: RetrievalScope | None` and, where time matters,
`as_of: datetime | None` (default now). Names are normalized on entry.

| Method | Returns | Description |
|---|---|---|
| `add_node(name, *, label=None, aliases=(), metadata=None)` | `GraphNode` | Upsert; aliases/metadata merge, first label wins. |
| `add_edge(source, relation, target, *, evidence=(), fact=None, provenance="extracted", confidence=1.0, valid_from=None, valid_to=None, metadata=None)` | `GraphEdge` | Creates endpoints; reinforces a live edge with the same triple. Evidence must be linked first. |
| `link_item(name, item_id, namespace="/")` | `None` | The item evidences the node under its namespace. `KeyError` for unknown node. |
| `unlink_item(item_id)` | `int` | Forget the item; edges left without evidence are invalidated. |
| `invalidate_edge(edge_id, *, at=None)` | `bool` | Mark obsolete (kept in history). |
| `remove_node(name)` | `bool` | Drop the node, invalidate its edges. |
| `node(name)` / `nodes(*, scope)` | `GraphNode \| None` / `list[str]` | |
| `items(name, *, scope)` | `list[str]` | Visible evidence of a node. |
| `edges(name, *, direction="both", scope, as_of)` | `list[GraphEdge]` | Outgoing first, then incoming. |
| `backlinks(name, *, scope, as_of)` | `list[GraphEdge]` | Edges pointing into the node. |
| `neighbors(name, *, max_depth=1, scope, as_of)` | `list[str]` | BFS both ways, start excluded. |
| `related_items(name, *, max_depth=2, scope, as_of)` | `list[str]` | Items of the node and its neighbourhood. |
| `path(a, b, *, scope, as_of)` | `list[str] \| None` | Fewest hops; never through a hidden node. |
| `explain(a, b, *, scope, as_of)` | `list[GraphEdge]` | The edges along `path`. |
| `mentions(text, *, scope, max_words=4, subgraph=None)` | `list[str]` | Nodes whose id/alias appears in the text. |
| `query(seeds, *, item_seeds=(), top_k=10, scope, as_of, damping=0.5, subgraph=None)` | `list[tuple[str, float]]` | Items ranked by personalized PageRank. |
| `hubs(*, k=10, scope, as_of)` | `list[tuple[str, int]]` | Degree ranking. |
| `communities(*, scope, as_of)` | `dict[str, int]` | Partition, cached on `store.version`; Leiden with `[graph]`, else Louvain. |

## Algorithms (`anchor.graph.algorithms`)

Pure Python: `adjacency(pairs)`, `bfs(adj, start, *, max_depth)`,
`shortest_path(adj, a, b)`, `degree(adj)`,
`personalized_pagerank(adj, seeds, *, damping=0.5, max_iter=100, tol=1e-6)`,
`louvain(adj, *, resolution=1.0, max_levels=10)`, `modularity(adj, communities)`.

## Stores

`GraphStore` / `AsyncGraphStore` protocols (see [Protocols](protocols.md))
with `InMemoryGraphStore(vault=...)`, `SqliteGraphStore(conn_manager, vault=...)`
and the async `PostgresGraphStore(conn_manager, vault=...)`. All
are vault-bound at construction; `version` increments on every write.

## Extraction (`anchor.ingestion`)

| Name | Description |
|---|---|
| `GraphExtractor` (protocol) | `extract(item) -> Extraction` |
| `Extraction` | `nodes` (linked to the item as evidence) and `edges` (evidence defaults to the item). |
| `StructureExtractor()` | The chunk's note as a node; frontmatter aliases/tags. |
| `WikilinkExtractor()` | `[[target]]` → `links_to`. Helpers: `wikilinks(text)`, `parse_wikilink(inner)`, `Wikilink`. |
| `LLMGraphExtractor(llm, *, relations=DEFAULT_RELATIONS)` | Entities + typed relations in one model call; fail-soft. |
| `GraphIndexer(graph, extractors=None)` | `index(items) -> IndexStats`, `index_entries(entries)`; default extractors = structure + wikilinks. |
| `GraphIndexingEntryStore(inner, indexer)` | A `MemoryEntryStore` wrapper: `add` indexes (re-extracts on changed content), `delete`/`clear` unlink; everything else forwards. What `MemoryManager(graph=...)` wraps the store in. |

## Retrieval

`GraphRetriever(graph, context_store, entity_extractor=None, *, vector_store=None, embeddings=None, seed_k=5, damping=0.5, tokenizer=None)`
— see [Retrieval](retrieval.md). `graph_retrieval_step(graph, store, entity_extractor, ..., scope=None)`
— see [Pipeline](pipeline.md).
