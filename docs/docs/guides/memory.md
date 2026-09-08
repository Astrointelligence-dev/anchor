# Memory Management

anchor provides a layered memory system that tracks conversation
history, stores persistent facts, and integrates both into the context
pipeline. This guide covers every component -- from the coordinator down
to decay functions and garbage collection.

## Architecture Overview

The memory system has three tiers:

1. **Conversation memory** -- a token-budgeted window of recent turns
   (`SlidingWindowMemory` or `SummaryBufferMemory`).
2. **Persistent facts** -- long-term entries in a `MemoryEntryStore`,
   managed through `MemoryManager`.
3. **Knowledge graph** -- an optional graph over memory *and* documents
   (`KnowledgeGraph`), see the [Knowledge Graph guide](knowledge-graph.md).

`MemoryManager` sits on top, producing `ContextItem` objects for the pipeline.

```
MemoryManager
  |-- conversation: SlidingWindowMemory | SummaryBufferMemory
  |-- persistent_store: MemoryEntryStore (optional)
  `-- get_context_items() --> ContextItem[]
```

## MemoryManager -- The Coordinator

`MemoryManager` is the single entry point for adding messages and
retrieving context.

```python
from anchor import MemoryManager

manager = MemoryManager(conversation_tokens=4096)
manager.add_user_message("What is context engineering?")
manager.add_assistant_message("It is the practice of assembling...")

items = manager.get_context_items()
print(len(items))  # 2 context items from conversation history
```

Pass a custom conversation backend to switch strategies:

```python
from anchor import MemoryManager, SummaryBufferMemory

def compact(turns):
    return "; ".join(t.content[:40] for t in turns)

summary_mem = SummaryBufferMemory(max_tokens=2048, compact_fn=compact)
manager = MemoryManager(conversation_memory=summary_mem)
```

!!! tip
    When `conversation_memory` is provided, the `conversation_tokens`,
    `tokenizer`, and `on_evict` parameters are ignored.

## SlidingWindowMemory

A rolling window of conversation turns within a token budget. When a new
turn would exceed the limit, oldest turns are evicted first.

```python
from anchor import SlidingWindowMemory

window = SlidingWindowMemory(max_tokens=1024)
window.add_turn("user", "Hello!")
window.add_turn("assistant", "Hi there!")
print(window.total_tokens)  # current token usage
```

### Eviction Callback

React to evicted turns (log them, archive them):

```python
def on_evict(evicted_turns):
    for t in evicted_turns:
        print(f"Evicted: {t.role}: {t.content[:50]}")

window = SlidingWindowMemory(max_tokens=512, on_evict=on_evict)
```

### Custom Eviction Policy and Recency Scorer

```python
from anchor import (
    SlidingWindowMemory, ImportanceEviction, ExponentialRecencyScorer,
)

window = SlidingWindowMemory(
    max_tokens=1024,
    eviction_policy=ImportanceEviction(
        importance_fn=lambda turn: len(turn.content) / 100.0
    ),
    recency_scorer=ExponentialRecencyScorer(decay_rate=3.0),
)
```

## SummaryBufferMemory

A two-tier memory: recent turns verbatim plus a running summary of
evicted content. Exactly one of `compact_fn` or `progressive_compact_fn`
must be provided.

### Simple Compaction

```python
from anchor import SummaryBufferMemory

def compact(turns):
    return "Summary: " + "; ".join(t.content[:60] for t in turns)

mem = SummaryBufferMemory(max_tokens=1024, compact_fn=compact)
mem.add_message("user", "Tell me about Python.")
mem.add_message("assistant", "Python is a high-level language...")
print(mem.summary)  # None until eviction occurs
```

### Progressive Compaction

`progressive_compact_fn` also receives the previous summary, enabling
incremental refinement:

```python
from anchor import SummaryBufferMemory

def progressive(turns, previous_summary):
    new_content = "; ".join(t.content[:40] for t in turns)
    if previous_summary:
        return f"{previous_summary} | {new_content}"
    return new_content

mem = SummaryBufferMemory(max_tokens=512, progressive_compact_fn=progressive)
```

!!! note
    If the compaction function raises an exception, the raw turn content
    is used as a fallback so evicted data is never lost.

## Knowledge graph

Memory facts can feed the same knowledge graph as your documents. Give the
`MemoryManager` a `GraphIndexer` and every fact added is indexed (the item
id is the entry id), an updated fact is re-extracted, and a deleted fact
takes its evidence with it:

```python
from anchor import GraphIndexer, KnowledgeGraph, LLMGraphExtractor, MemoryManager

graph = KnowledgeGraph()
manager = MemoryManager(
    persistent_store=store,
    graph=GraphIndexer(graph, extractors=[LLMGraphExtractor(llm)]),
)
manager.add_fact("Alice works on Project X with Bob.")

graph.neighbors("alice", max_depth=2)      # ['project_x', 'bob']
graph.related_items("Project X")           # [<entry id>]
```

`graph_retrieval_step(graph, store, entity_extractor)` pulls the memory
entries a query's entities lead to into the pipeline. The full API, the
scope rules and the wikilink extractors for documents are in the
[Knowledge Graph guide](knowledge-graph.md).

## Eviction Policies

Three built-in policies implement the `EvictionPolicy` protocol.

**FIFOEviction** -- evicts oldest turns first (matches the default):

```python
from anchor import FIFOEviction, SlidingWindowMemory

window = SlidingWindowMemory(max_tokens=1024, eviction_policy=FIFOEviction())
```

**ImportanceEviction** -- evicts lowest-scoring turns first:

```python
from anchor import ImportanceEviction

policy = ImportanceEviction(importance_fn=lambda turn: len(turn.content) / 100.0)
```

**PairedEviction** -- evicts user+assistant pairs together so no orphaned
questions remain:

```python
from anchor import PairedEviction, SlidingWindowMemory

window = SlidingWindowMemory(max_tokens=1024, eviction_policy=PairedEviction())
```

## Decay Functions and Recency Scorers

Decay functions compute retention scores for persistent memory entries.
The garbage collector uses them to prune forgotten memories.

**EbbinghausDecay** -- forgetting curve `R = e^(-t/S)` where strength
grows with access count:

```python
from anchor import EbbinghausDecay
decay = EbbinghausDecay(base_strength=1.0, reinforcement_factor=0.5)
```

**LinearDecay** -- linear interpolation from 1.0 to 0.0. At
`half_life_hours` the retention is 0.5:

```python
from anchor import LinearDecay
decay = LinearDecay(half_life_hours=168.0)  # 7 days
```

**Recency scorers** control position-based scores within a sliding window:

```python
from anchor import ExponentialRecencyScorer, LinearRecencyScorer

exp_scorer = ExponentialRecencyScorer(decay_rate=2.0)
print(exp_scorer.score(0, 10))   # ~0.0 (oldest)
print(exp_scorer.score(9, 10))   # 1.0  (newest)

lin_scorer = LinearRecencyScorer(min_score=0.5)
print(lin_scorer.score(0, 10))   # 0.5
print(lin_scorer.score(9, 10))   # 1.0
```

## SimilarityConsolidator

Determines whether new memory entries should be added, merged, or
skipped based on content hashing and cosine similarity.

```python
import math
from anchor import SimilarityConsolidator, MemoryEntry

def embed_fn(text: str) -> list[float]:
    return [math.sin(i + len(text)) for i in range(8)]

consolidator = SimilarityConsolidator(
    embed_fn=embed_fn,
    similarity_threshold=0.85,
)

existing = [MemoryEntry(content="User prefers dark mode")]
new_entries = [
    MemoryEntry(content="User likes dark themes"),   # similar -> UPDATE
    MemoryEntry(content="User prefers dark mode"),    # exact dup -> NONE
    MemoryEntry(content="User works at Acme Corp"),   # new -> ADD
]

results = consolidator.consolidate(new_entries, existing)
for action, entry in results:
    print(action, entry.content if entry else "(skipped)")
```

!!! warning
    `SimilarityConsolidator` never calls an LLM — you provide the `embed_fn`
    (OpenAI, Cohere, local models, etc.). It also cannot tell a
    contradiction from a paraphrase: "moved to Rio" is *similar* to "lives
    in São Paulo" and gets merged, not corrected. For that, opt into the
    LLM pair below.

## LLMExtractor and LLMConsolidator

The two-phase update of the mem0 paper, opt-in: an `LLMExtractor` turns the
recent turns into facts, an `LLMConsolidator` decides `ADD` / `UPDATE` /
`DELETE` / `NONE` for each fact against the most similar memories. Both take
the agent's own provider — never a new client — and fail soft (a bad answer
logs a warning and adds the facts as-is).

```python
from anchor import Agent, LLMConsolidator, LLMExtractor, MemoryManager, InMemoryEntryStore

agent = Agent(llm=...)
memory = MemoryManager(
    persistent_store=InMemoryEntryStore(),
    extractor=LLMExtractor(agent.llm),
    consolidator=LLMConsolidator(agent.llm, embed_fn=my_embed),  # embed_fn optional
    remember_every=1,  # run after every completed turn (5 = Letta-style batching)
)
agent.with_memory(memory)

list(agent.chat("Me mudei pro Rio de Janeiro mês passado."))
memory.get_all_facts()  # "User lives in São Paulo" became "User lives in Rio de Janeiro"
```

What happens on `remember()` (the `Agent` calls `after_turn()` after every
completed turn — not on an abandoned stream — and in `astream` off the event
loop):

1. The extractor reads the last `extract_window` user/assistant turns (tool
   turns are noise) and returns facts.
2. Cheap gates first: an exact content-hash match is `NONE` without a model
   call; an empty store makes everything `ADD`; with `embed_fn`, a fact whose
   best cosine against the store is below `new_threshold` is `ADD` without a
   call. **High similarity never decides `NONE` on its own** — cosine cannot
   separate a contradiction from a paraphrase, so that band goes to the model.
3. One model call with the `top_k` most similar memories per fact (the
   `max_candidates` most recent ones without `embed_fn`), referenced by index.
   `UPDATE` keeps the memory's id, recomputes the hash and records
   `metadata["previous_content"]`; `DELETE` is a **soft delete** — the entry
   gets `expires_at=now` (`MemoryEntry.invalidate`), drops out of
   `search`/`list_all` in every backend, and stays readable through
   `list_all_unfiltered` until the garbage collector's `retention` elapses.
   An unknown `update` target degrades to `ADD`, an unknown `delete` target
   is ignored, a fact the model leaves out is `ADD`.

Call `memory.remember()` yourself for a manual flush (end of session, or
with `remember_every=0`). `MemoryCallback.on_extraction` /
`on_consolidation` observe every step. The consolidation golden set in
`tests/fixtures/consolidation_golden.jsonl` (48 cases: fact changes,
preference flips, paraphrases, temporary facts, false contradictions,
re-affirmations) measures the pair — see the [evaluation guide](evaluation.md).

## MemoryGarbageCollector and GCStats

Prunes expired and decayed entries from a `GarbageCollectableStore`.

```python
from anchor import MemoryGarbageCollector, EbbinghausDecay, InMemoryEntryStore

store = InMemoryEntryStore()
gc = MemoryGarbageCollector(store=store, decay=EbbinghausDecay())
# retention=timedelta(days=30) keeps invalidated (soft-deleted) facts as history that long

stats = gc.collect(retention_threshold=0.1)
print(stats)  # GCStats(applied: expired_pruned=0, decayed_pruned=0, ...)

# Dry run -- identify what would be pruned without deleting
stats = gc.collect(retention_threshold=0.1, dry_run=True)
print(stats.total_pruned)
```

The collector works in two phases:

1. **Expiry phase** -- removes entries whose `is_expired` property is `True`.
2. **Decay phase** -- computes retention and removes entries below the threshold.

Both phases fire `MemoryCallback` hooks for observability.

## Persistent Facts

`MemoryManager` manages persistent facts through a `MemoryEntryStore`.
Content-hash deduplication prevents storing the same content twice.

```python
from anchor import MemoryManager, MemoryType, InMemoryEntryStore

store = InMemoryEntryStore()
manager = MemoryManager(persistent_store=store)

# Add facts (returns existing entry if duplicate)
entry = manager.add_fact(
    "User prefers dark mode",
    tags=["preference"],
    memory_type=MemoryType.SEMANTIC,
)

# Search, update, delete
results = manager.get_relevant_facts("theme preference", top_k=3)
manager.update_fact(entry.id, "User prefers dark mode with blue accent")
manager.delete_fact(entry.id)  # hard delete; a consolidator's DELETE is the soft one
all_facts = manager.get_all_facts()
```

!!! warning
    Calling `add_fact` without a configured `persistent_store` raises
    `StorageError`. Always pass a `MemoryEntryStore` to the constructor.

## MemoryCallback and CallbackExtractor

`MemoryCallback` is a protocol for observing memory lifecycle events.
Implement only the methods you care about:

```python
class MyCallback:
    def on_eviction(self, turns, remaining_tokens):
        print(f"Evicted {len(turns)} turns, {remaining_tokens} tokens left")

    def on_decay_prune(self, pruned_entries, threshold):
        print(f"Pruned {len(pruned_entries)} entries below {threshold}")
```

`CallbackExtractor` delegates memory extraction to a user function:

```python
from anchor import CallbackExtractor, MemoryType
from anchor.models.memory import ConversationTurn

def my_extractor(turns):
    return [
        {"content": "User prefers dark mode", "tags": ["preference"]},
        {"content": "User's name is Alice", "memory_type": "semantic"},
    ]

extractor = CallbackExtractor(extract_fn=my_extractor)
turns = [ConversationTurn(role="user", content="I prefer dark mode")]
entries = extractor.extract(turns)
print(entries[0].content)  # "User prefers dark mode"
```

## Complete Example

A memory-aware pipeline with summarization, persistent facts, and
garbage collection:

```python
from anchor import (
    ContextPipeline, MemoryManager, SummaryBufferMemory,
    MemoryGarbageCollector, EbbinghausDecay, InMemoryEntryStore,
    QueryBundle,
)

store = InMemoryEntryStore()

def compact(turns):
    return "Previously: " + "; ".join(t.content[:50] for t in turns)

conversation = SummaryBufferMemory(max_tokens=2048, compact_fn=compact)
manager = MemoryManager(conversation_memory=conversation, persistent_store=store)

manager.add_fact("User prefers Python over JavaScript", tags=["preference"])
manager.add_user_message("How do I sort a list in Python?")
manager.add_assistant_message("Use sorted() or list.sort().")

pipeline = (
    ContextPipeline(max_tokens=4096)
    .with_memory(manager)
    .add_system_prompt("You are a helpful coding assistant.")
)
result = pipeline.build(QueryBundle(query_str="Show me an example"))
print(f"Context items: {len(result.window.items)}")

gc = MemoryGarbageCollector(store=store, decay=EbbinghausDecay())
stats = gc.collect(retention_threshold=0.1)
print(stats)
```

## Next Steps

- [Memory API Reference](../api/memory.md) -- constructor signatures,
  parameter tables, and method details for every memory class.
- [Pipeline Guide](../guides/pipeline.md) -- how memory integrates with
  the context pipeline.
