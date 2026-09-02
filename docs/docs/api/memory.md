# Memory API Reference

All classes below are importable from `anchor` directly.
For the conceptual guide, see [Memory Guide](../guides/memory.md).

---

## MemoryManager

Coordinates conversation memory and persistent facts.

```python
MemoryManager(
    conversation_tokens: int = 4096,
    tokenizer: Tokenizer | None = None,
    on_evict: Callable[[list[ConversationTurn]], None] | None = None,
    persistent_store: MemoryEntryStore | None = None,
    conversation_memory: ConversationMemory | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `conversation_tokens` | `int` | `4096` | Token budget for the default sliding window. Ignored when `conversation_memory` is provided. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. Falls back to the built-in counter. |
| `on_evict` | `Callable \| None` | `None` | Callback for evicted turns. Ignored when `conversation_memory` is provided. |
| `persistent_store` | `MemoryEntryStore \| None` | `None` | Store for long-term facts. |
| `conversation_memory` | `ConversationMemory \| None` | `None` | Custom conversation backend. Overrides the default sliding window. |

| Property | Type | Description |
|---|---|---|
| `conversation` | `ConversationMemory` | The underlying conversation memory instance. |
| `conversation_type` | `str` | Returns `"sliding_window"`, `"summary_buffer"`, or the class name. |
| `persistent_store` | `MemoryEntryStore \| None` | The persistent store, if configured. |

| Method | Returns | Description |
|---|---|---|
| `add_user_message(content)` | `None` | Add a user turn. |
| `add_assistant_message(content)` | `None` | Add an assistant turn. |
| `add_system_message(content)` | `None` | Add a system turn. |
| `add_tool_message(content)` | `None` | Add a tool turn. |
| `add_fact(content, tags, memory_type, metadata)` | `MemoryEntry` | Store a persistent fact with content-hash deduplication. Raises `StorageError` without a store. |
| `get_relevant_facts(query, top_k=5)` | `list[MemoryEntry]` | Search persistent store. Returns `[]` without a store. |
| `get_all_facts()` | `list[MemoryEntry]` | Return all persistent entries. |
| `delete_fact(entry_id)` | `bool` | Delete a fact by ID. Returns `False` if not found. |
| `update_fact(entry_id, content)` | `MemoryEntry \| None` | Update fact content. Returns `None` if not found. |
| `get_context_items(priority=7)` | `list[ContextItem]` | Assemble context items. Facts at priority 8, conversation at given priority. |
| `clear()` | `None` | Clear conversation history and persistent store. |

---

## SlidingWindowMemory

Token-aware sliding window. Thread-safe.

```python
SlidingWindowMemory(
    max_tokens: int = 4096,
    tokenizer: Tokenizer | None = None,
    on_evict: Callable[[list[ConversationTurn]], None] | None = None,
    eviction_policy: EvictionPolicy | None = None,
    recency_scorer: RecencyScorer | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `4096` | Maximum token budget. Must be positive. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. |
| `on_evict` | `Callable \| None` | `None` | Called with evicted turns list. |
| `eviction_policy` | `EvictionPolicy \| None` | `None` | Custom eviction strategy. Default is FIFO. |
| `recency_scorer` | `RecencyScorer \| None` | `None` | Custom recency scoring. Default is linear 0.5--1.0. |

| Property | Type | Description |
|---|---|---|
| `turns` | `list[ConversationTurn]` | Current turns (copy). |
| `total_tokens` | `int` | Tokens currently used. |
| `max_tokens` | `int` | The configured budget. |

| Method | Returns | Description |
|---|---|---|
| `add_turn(role, content, **metadata)` | `ConversationTurn` | Add a turn, evicting old turns if needed. Truncates if a single turn exceeds budget. |
| `to_context_items(priority=7)` | `list[ContextItem]` | Convert to context items with recency-weighted scores. Role in metadata, not content. |
| `clear()` | `None` | Remove all turns and reset token count. |

---

## SummaryBufferMemory

Two-tier memory: recent turns verbatim plus a running summary. Exactly one compaction function required.

```python
SummaryBufferMemory(
    max_tokens: int,
    compact_fn: Callable[[list[ConversationTurn]], str] | None = None,
    progressive_compact_fn: Callable[[list[ConversationTurn], str | None], str] | None = None,
    tokenizer: Tokenizer | None = None,
    summary_priority: int = 6,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | *(required)* | Token budget for the internal sliding window. |
| `compact_fn` | `Callable \| None` | `None` | Simple compaction: receives evicted turns, returns summary. |
| `progressive_compact_fn` | `Callable \| None` | `None` | Progressive: receives evicted turns and previous summary. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. |
| `summary_priority` | `int` | `6` | Priority for the summary context item. |

!!! warning
    Providing both or neither compaction function raises `ValueError`.

| Property | Type | Description |
|---|---|---|
| `summary` | `str \| None` | Running summary, or `None` before first eviction. |
| `summary_tokens` | `int` | Token count of the current summary. |
| `turns` | `list[ConversationTurn]` | Live turns in the sliding window. |
| `total_tokens` | `int` | Tokens in the live window (excludes summary). |

| Method | Returns | Description |
|---|---|---|
| `add_turn(turn)` | `None` | Add a pre-built `ConversationTurn`. |
| `add_message(role, content, **metadata)` | `ConversationTurn` | Add a message by role and content. |
| `to_context_items(priority=7)` | `list[ContextItem]` | Summary item (if present) followed by live window items. |
| `clear()` | `None` | Clear both window and summary. |

---

## KnowledgeGraph

The entity graph moved to `anchor.graph` and now spans memory and
documents: see [Knowledge Graph](graph.md). `MemoryManager(graph=GraphIndexer(...))`
keeps it in step with the persistent store.

---

## FIFOEviction

Evicts oldest turns first. Matches the built-in default.

```python
FIFOEviction()
```

| Method | Returns | Description |
|---|---|---|
| `select_for_eviction(turns, tokens_to_free)` | `list[int]` | Indices of oldest turns until enough tokens freed. |

---

## ImportanceEviction

Evicts turns with the lowest importance scores first.

```python
ImportanceEviction(importance_fn: Callable[[ConversationTurn], float])
```

| Parameter | Type | Description |
|---|---|---|
| `importance_fn` | `Callable[[ConversationTurn], float]` | Scoring function. Lower scores evicted first. |

| Method | Returns | Description |
|---|---|---|
| `select_for_eviction(turns, tokens_to_free)` | `list[int]` | Indices of least-important turns until enough tokens freed. |

---

## PairedEviction

Evicts user+assistant turn pairs together. Pairs evicted oldest-first.

```python
PairedEviction()
```

| Method | Returns | Description |
|---|---|---|
| `select_for_eviction(turns, tokens_to_free)` | `list[int]` | Indices of oldest turn-pairs until enough tokens freed. |

---

## ExponentialRecencyScorer

Exponential recency scoring: `(e^(rate * x) - 1) / (e^(rate) - 1)`.

```python
ExponentialRecencyScorer(decay_rate: float = 2.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `decay_rate` | `float` | `2.0` | Controls curve steepness. Must be positive. |

| Method | Returns | Description |
|---|---|---|
| `score(index, total)` | `float` | Score in [0.0, 1.0]. `index=0` is oldest. |

---

## LinearRecencyScorer

Linear recency scoring from `min_score` to 1.0.

```python
LinearRecencyScorer(min_score: float = 0.5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_score` | `float` | `0.5` | Score for oldest turn. Must be in [0.0, 1.0). |

| Method | Returns | Description |
|---|---|---|
| `score(index, total)` | `float` | Score in [min_score, 1.0]. |

---

## EbbinghausDecay

Ebbinghaus forgetting curve: `R = e^(-t/S)` where `S = base_strength + access_count * reinforcement_factor`.

```python
EbbinghausDecay(base_strength: float = 1.0, reinforcement_factor: float = 0.5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_strength` | `float` | `1.0` | Initial memory strength in hours. Must be positive. |
| `reinforcement_factor` | `float` | `0.5` | Strength added per access. Must be non-negative. |

| Method | Returns | Description |
|---|---|---|
| `compute_retention(entry)` | `float` | Retention in [0.0, 1.0] based on time and access count. |

---

## LinearDecay

Linear decay from 1.0 to 0.0 over twice the half-life.

```python
LinearDecay(half_life_hours: float = 168.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `half_life_hours` | `float` | `168.0` | Hours until retention reaches 0.5 (default 7 days). Must be positive. |

| Method | Returns | Description |
|---|---|---|
| `compute_retention(entry)` | `float` | Retention in [0.0, 1.0]. 0.5 at half-life, 0.0 at twice half-life. |

---

## SimilarityConsolidator

Merges similar memories via embedding cosine similarity and content-hash deduplication.

```python
SimilarityConsolidator(
    embed_fn: Callable[[str], list[float]],
    similarity_threshold: float = 0.85,
    max_cache_size: int = 1000,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_fn` | `Callable[[str], list[float]]` | *(required)* | Embedding function. The library never calls an LLM. |
| `similarity_threshold` | `float` | `0.85` | Cosine similarity above which entries are merged. In [0.0, 1.0]. |
| `max_cache_size` | `int` | `1000` | Max cached embeddings before cache is cleared. |

| Method | Returns | Description |
|---|---|---|
| `consolidate(new_entries, existing)` | `list[tuple[MemoryOperation, MemoryEntry \| None]]` | `ADD` (new), `UPDATE` (merged), or `NONE` (duplicate) for each entry. |

!!! note
    Merged entries keep the longer content, combine tags/links/metadata,
    increment `access_count`, and use the higher `relevance_score`.

---

## MemoryGarbageCollector

Prunes expired and decayed entries from a `GarbageCollectableStore`.

```python
MemoryGarbageCollector(
    store: GarbageCollectableStore,
    decay: MemoryDecay | None = None,
    callbacks: list[MemoryCallback] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `store` | `GarbageCollectableStore` | *(required)* | Store to prune. Must support `list_all_unfiltered()`. |
| `decay` | `MemoryDecay \| None` | `None` | Decay function. Without it, only expiry pruning runs. |
| `callbacks` | `list[MemoryCallback] \| None` | `None` | Callbacks notified of pruning events. |

| Method | Returns | Description |
|---|---|---|
| `collect(retention_threshold=0.1, dry_run=False)` | `GCStats` | Full GC (expiry + decay). |
| `collect_expired(dry_run=False)` | `list[MemoryEntry]` | Remove only expired entries. |
| `collect_decayed(retention_threshold=0.1, dry_run=False)` | `list[MemoryEntry]` | Remove only decayed entries. Raises `ValueError` without decay. |

---

## GCStats

Statistics from a garbage collection run.

```python
GCStats(expired_pruned: int, decayed_pruned: int, total_remaining: int, dry_run: bool)
```

| Attribute | Type | Description |
|---|---|---|
| `expired_pruned` | `int` | Entries pruned due to expiration. |
| `decayed_pruned` | `int` | Entries pruned due to low retention. |
| `total_remaining` | `int` | Entries remaining after collection. |
| `dry_run` | `bool` | Whether this was a dry run. |
| `total_pruned` | `int` | *(property)* Sum of expired + decayed. |

---

## MemoryCallback

Protocol for observing memory lifecycle events. All methods default to no-ops.

```python
class MemoryCallback(Protocol):
    def on_eviction(self, turns: list[ConversationTurn], remaining_tokens: int) -> None: ...
    def on_compaction(self, evicted_turns: list[ConversationTurn], summary: str, previous_summary: str | None) -> None: ...
    def on_extraction(self, turns: list[ConversationTurn], entries: list[MemoryEntry]) -> None: ...
    def on_consolidation(self, action: str, new_entry: MemoryEntry | None, existing_entry: MemoryEntry | None) -> None: ...
    def on_decay_prune(self, pruned_entries: list[MemoryEntry], threshold: float) -> None: ...
    def on_expiry_prune(self, pruned_entries: list[MemoryEntry]) -> None: ...
```

| Method | Description |
|---|---|
| `on_eviction` | Turns evicted from sliding window. |
| `on_compaction` | Evicted turns compacted into summary. |
| `on_extraction` | Memories extracted from conversation. |
| `on_consolidation` | Consolidation decision (add/update/delete/none). |
| `on_decay_prune` | Entries pruned by low retention score. |
| `on_expiry_prune` | Expired entries removed. |

---

## ProgressiveSummarizationMemory

Four-tier progressive summarization memory. Satisfies the `ConversationMemory` protocol.
Pluggable into `MemoryManager` and `ContextPipeline`.

```python
ProgressiveSummarizationMemory(
    max_tokens: int = 8192,
    llm: LLMProvider | str = "anthropic/claude-haiku-4-5-20251001",
    tier_config: list[TierConfig] | None = None,
    max_facts: int = 50,
    fact_token_budget: int = 500,
    tokenizer: Tokenizer | None = None,
    callbacks: list[ProgressiveSummarizationCallback] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `8192` | Total token budget across all tiers. Must be positive. |
| `llm` | `LLMProvider \| str` | `"anthropic/claude-haiku-4-5-20251001"` | LLM for summarization. Accepts a provider instance or a `"provider/model"` string. |
| `tier_config` | `list[TierConfig] \| None` | `None` | Custom tier allocation. Defaults to 4 tiers (verbatim → bullet → paragraph → archive). |
| `max_facts` | `int` | `50` | Maximum extracted facts to retain. |
| `fact_token_budget` | `int` | `500` | Token budget reserved for facts. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. Falls back to the built-in counter. |
| `callbacks` | `list \| None` | `None` | Lifecycle callbacks for summarization events. |

| Method | Returns | Description |
|---|---|---|
| `add_turn(role, content, **metadata)` | `ConversationTurn` | Add a turn, triggering tier promotion when budget is exceeded. |
| `to_context_items(priority=7)` | `list[ContextItem]` | Assemble context items from all tiers and facts. |
| `clear()` | `None` | Clear all tiers and facts. |

---

## TierCompactor

Handles LLM calls for summarization and fact extraction. Used internally by `ProgressiveSummarizationMemory`.

```python
TierCompactor(
    llm: LLMProvider,
    tokenizer: Tokenizer | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLMProvider` | *(required)* | LLM provider for generating summaries. |
| `tokenizer` | `Tokenizer \| None` | `None` | Custom tokenizer. Falls back to the built-in counter. |

| Method | Returns | Description |
|---|---|---|
| `summarize(content, target_tier, target_tokens, existing_summary=None)` | `str` | Synchronously summarize content for a target tier. Falls back to truncation on LLM failure. |
| `extract_facts(content)` | `list[str]` | Extract key facts from content via LLM. Returns empty list on failure. |

---

## CallbackExtractor

Delegates memory extraction to a user-provided function.

```python
CallbackExtractor(
    extract_fn: Callable[[list[ConversationTurn]], list[dict[str, Any]]],
    default_type: MemoryType = MemoryType.SEMANTIC,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `extract_fn` | `Callable` | *(required)* | Receives turns, returns dicts with required `"content"` key. Optional: `"tags"`, `"memory_type"`, `"metadata"`, `"relevance_score"`, `"user_id"`, `"session_id"`. |
| `default_type` | `MemoryType` | `MemoryType.SEMANTIC` | Default type when not specified in the dict. |

| Method | Returns | Description |
|---|---|---|
| `extract(turns)` | `list[MemoryEntry]` | Build entries from user function output. Raises `ValueError` if `"content"` missing. |

**Example**

```python
from anchor import CallbackExtractor, MemoryType
from anchor.models.memory import ConversationTurn

def my_extractor(turns):
    return [{"content": f"Discussed: {turns[0].content[:40]}", "tags": ["topic"]}]

extractor = CallbackExtractor(extract_fn=my_extractor)
turns = [ConversationTurn(role="user", content="Tell me about Python async")]
entries = extractor.extract(turns)
print(entries[0].content)       # "Discussed: Tell me about Python async"
print(entries[0].memory_type)   # MemoryType.SEMANTIC
```
