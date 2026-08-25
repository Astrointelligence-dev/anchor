# Models API Reference

API reference for all core data models in anchor. Every model listed here
is a Pydantic `BaseModel` (unless otherwise noted) and can be imported directly
from `anchor`.

---

## `ContextItem`

The atomic unit of context flowing through the pipeline. Items are **frozen**
(immutable after creation) to prevent context poisoning bugs.

```python
from anchor import ContextItem

class ContextItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: SourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | Auto-generated UUID | Unique identifier. |
| `content` | `str` | (required) | The text content of this context item. |
| `source` | `SourceType` | (required) | Origin type (retrieval, memory, system, etc.). |
| `score` | `float` | `0.0` | Relevance score between 0.0 and 1.0. |
| `priority` | `int` | `5` | Priority rank from 1 (lowest) to 10 (highest). |
| `token_count` | `int` | `0` | Pre-computed token count. Filled by the pipeline if zero. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata. |
| `created_at` | `datetime` | `datetime.now(UTC)` | Timestamp of creation. |

!!! tip
    Since `ContextItem` is frozen, use `item.model_copy(update={"score": 0.9})`
    to create a modified copy.

### Example

```python
from anchor import ContextItem, SourceType

item = ContextItem(
    content="Context engineering is the discipline of building dynamic systems.",
    source=SourceType.RETRIEVAL,
    score=0.95,
    priority=5,
    token_count=9,
)
print(item.id)        # auto-generated UUID
print(item.content)   # "Context engineering is..."
```

---

## `ContextWindow`

A complete context window ready for formatting. Token-aware and priority-ranked.

| Field | Type | Default | Description |
|---|---|---|---|
| `items` | `list[ContextItem]` | `[]` | The context items in this window. |
| `max_tokens` | `int` | `8192` | Maximum token capacity. Must be > 0. |
| `used_tokens` | `int` | `0` | Tokens currently consumed. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata. |

### Properties

| Property | Type | Description |
|---|---|---|
| `remaining_tokens` | `int` | `max(0, max_tokens - used_tokens)` |
| `utilization` | `float` | Fraction of budget used (0.0 to 1.0). |

### Methods

#### `add_item(item) -> bool`

Add an item if it fits within the token budget. Returns `True` if added,
`False` if it would exceed the budget.

#### `add_items_by_priority(items) -> list[ContextItem]`

Add items sorted by priority (highest first, ties broken by score). Returns the
overflow list -- items that did not fit.

---

## `ContextResult`

The final output of the context pipeline.

| Field | Type | Default | Description |
|---|---|---|---|
| `window` | `ContextWindow` | (required) | The assembled context window. |
| `formatted_output` | `str \| dict[str, Any]` | `""` | Formatted output for the target LLM provider. |
| `format_type` | `str` | `"generic"` | Name of the formatter used. |
| `overflow_items` | `list[ContextItem]` | `[]` | Items that exceeded the token budget. |
| `diagnostics` | `PipelineDiagnostics` | `{}` | Build diagnostics (see below). |
| `build_time_ms` | `float` | `0.0` | Total pipeline build time in milliseconds. |

---

## `QueryBundle`

Encapsulates a query as it flows through the pipeline.

| Field | Type | Default | Description |
|---|---|---|---|
| `query_str` | `str` | (required) | The query text. |
| `embedding` | `list[float] \| None` | `None` | Optional pre-computed embedding vector. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary query metadata (user ID, session ID, etc.). |
| `chat_history` | `list[ConversationTurn]` | `[]` | Optional conversation history for context. |

!!! tip
    `ContextPipeline.build()` accepts a plain string and wraps it in a
    `QueryBundle` automatically.

---

## `TokenBudget`

Manages token budget across all context sources with per-source allocations.
Unallocated tokens form a shared pool.

| Field | Type | Default | Description |
|---|---|---|---|
| `total_tokens` | `int` | (required) | Total token budget. Must be > 0. |
| `allocations` | `list[BudgetAllocation]` | `[]` | Per-source-type token caps. |
| `reserve_tokens` | `int` | `0` | Tokens reserved (e.g., for the LLM response). |

**Validation:** The sum of all `allocations[].max_tokens` plus `reserve_tokens`
must not exceed `total_tokens`.

### Properties

| Property | Type | Description |
|---|---|---|
| `shared_pool` | `int` | Tokens not explicitly allocated to any source. |

### Methods

#### `get_allocation(source) -> int`

Get the max tokens for a source type. Falls back to the shared pool if the
source has no explicit allocation.

#### `get_overflow_strategy(source) -> OverflowStrategy`

Get the overflow strategy for a source type. Returns `"truncate"` for sources
without an explicit allocation.

### Example

```python
from anchor import TokenBudget, BudgetAllocation, SourceType

budget = TokenBudget(
    total_tokens=8192,
    reserve_tokens=1200,
    allocations=[
        BudgetAllocation(source=SourceType.SYSTEM, max_tokens=800, priority=10),
        BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=3000, priority=5),
    ],
)
print(budget.shared_pool)                              # 3192
print(budget.get_allocation(SourceType.RETRIEVAL))     # 3000
print(budget.get_allocation(SourceType.MEMORY))        # 3192 (shared pool)
```

---

## `BudgetAllocation`

Token allocation for a specific source type within a `TokenBudget`.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `SourceType` | (required) | The source type this allocation applies to. |
| `max_tokens` | `int` | (required) | Maximum tokens for this source. Must be > 0. |
| `priority` | `int` | `5` | Priority rank (1--10). |
| `overflow_strategy` | `OverflowStrategy` | `"truncate"` | `"truncate"` keeps items up to cap; `"drop"` drops all if cap exceeded. |

---

## Budget Factory Functions

Three factory functions return pre-configured `TokenBudget` instances. All
accept a single `max_tokens: int` parameter and raise `ValueError` if it is
not positive.

| Factory | System | Memory | Conversation | Retrieval | Tool | Reserve | Shared |
|---|---|---|---|---|---|---|---|
| `default_chat_budget` | 10% | 10% | 20% | 25% | -- | 15% | 20% |
| `default_rag_budget` | 10% | 5% | 10% | 40% | -- | 15% | 20% |
| `default_agent_budget` | 15% | 10% | 15% | 20% | 15% | 15% | 10% |

---

## Enums

### `SourceType`

The origin type of a context item. A `StrEnum`.

```python
from anchor import SourceType
```

| Value | String | Description |
|---|---|---|
| `SourceType.RETRIEVAL` | `"retrieval"` | Retrieved from a knowledge base. |
| `SourceType.MEMORY` | `"memory"` | Persistent long-term memory. |
| `SourceType.SYSTEM` | `"system"` | System prompts and instructions. |
| `SourceType.USER` | `"user"` | User-provided content. |
| `SourceType.TOOL` | `"tool"` | Tool call results. |
| `SourceType.CONVERSATION` | `"conversation"` | Conversation history turns. |

### `OverflowStrategy`

Type alias for `Literal["truncate", "drop"]`. Controls behavior when a source
exceeds its budget allocation.

- `"truncate"` -- include items up to the cap (sorted by priority/score).
- `"drop"` -- if total for that source exceeds the cap, all items are dropped.

### `Role`

Message role for conversation turns. A `StrEnum` (imported from `anchor.llm.models`).

```python
from anchor.llm import Role
```

| Value | String | Description |
|---|---|---|
| `Role.SYSTEM` | `"system"` | System prompt messages. |
| `Role.USER` | `"user"` | User messages. |
| `Role.ASSISTANT` | `"assistant"` | Assistant responses. |
| `Role.TOOL` | `"tool"` | Tool call results. |

!!! tip
    Since `Role` is a `StrEnum`, `str(Role.USER)` returns `"user"` and
    `Role.USER == "user"` is `True`.

### `MemoryType`

Classification of memory entries by cognitive type. A `StrEnum`.

| Value | String |
|---|---|
| `MemoryType.SEMANTIC` | `"semantic"` |
| `MemoryType.EPISODIC` | `"episodic"` |
| `MemoryType.PROCEDURAL` | `"procedural"` |
| `MemoryType.CONVERSATION` | `"conversation"` |

---

## Memory Models

### `ConversationTurn`

A single turn in a conversation.

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `Role` | (required) | `"user"`, `"assistant"`, `"system"`, or `"tool"`. |
| `content` | `str` | (required) | The message content. |
| `token_count` | `int` | `0` | Pre-computed token count. |
| `timestamp` | `datetime` | `datetime.now(UTC)` | When this turn occurred. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata. |

### `MemoryEntry`

A persistent memory entry with relevance tracking.

```python
from anchor import MemoryEntry

class MemoryEntry(BaseModel):
    id: str                          # Auto-generated UUID
    content: str                     # (required) Memory content text
    relevance_score: float = 0.5     # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime           # Auto-set to now(UTC)
    created_at: datetime              # Auto-set to now(UTC)
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    memory_type: MemoryType = MemoryType.SEMANTIC
    user_id: str | None = None       # Optional user scope
    session_id: str | None = None    # Optional session scope
    expires_at: datetime | None = None
    updated_at: datetime              # Auto-set to now(UTC)
    content_hash: str = ""           # Auto-computed MD5
    source_turns: list[str] = []     # IDs of source conversation turns
    links: list[str] = []            # IDs of linked memory entries
```

**Property:** `is_expired -> bool` -- `True` if `expires_at` is set and in the past.

**Method:** `touch() -> MemoryEntry` -- returns a copy with incremented `access_count`
and refreshed `last_accessed`.

---

## Diagnostics Models

### `PipelineDiagnostics`

A `TypedDict` (with `total=False`) describing the diagnostics dict produced by
the pipeline. All keys are optional.

| Key | Type | Description |
|---|---|---|
| `steps` | `list[StepDiagnostic]` | Per-step timing and item counts. |
| `memory_items` | `int` | Memory items injected before steps. |
| `total_items_considered` | `int` | Items entering the assembly phase. |
| `items_included` | `int` | Items that fit in the window. |
| `items_overflow` | `int` | Items that exceeded the budget. |
| `token_utilization` | `float` | Fraction of budget used (0.0--1.0). |
| `token_usage_by_source` | `dict[str, int]` | Tokens used per source type. |
| `query_enriched` | `bool` | Whether query enrichment was applied. |
| `skipped_steps` | `list[str]` | Steps skipped due to errors. |
| `failed_step` | `str` | Step that caused a fatal error. |
| `budget_overflow_by_source` | `dict[str, int]` | Items overflowed per source. |
| `shared_pool_usage` | `int` | Tokens used by unallocated sources. |

### `StepDiagnostic`

A `TypedDict` for a single pipeline step.

| Key | Type | Description |
|---|---|---|
| `name` | `str` | Step name. |
| `items_after` | `int` | Number of items after the step executed. |
| `time_ms` | `float` | Execution time in milliseconds. |

---

## See Also

- [Pipeline Guide](../guides/pipeline.md) -- how models fit into the pipeline flow
- [Pipeline API Reference](pipeline.md) -- `ContextPipeline` and step factories
- [Exceptions Reference](exceptions.md) -- error hierarchy
