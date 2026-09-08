# Token Budget Management

Token budgets give you fine-grained control over how tokens are allocated
across different context sources. Instead of a single `max_tokens` cap
where all sources compete, you can assign dedicated portions to system
prompts, memory, retrieval, tools, and other sources.

## Why Token Budgets?

Without budgets, all `ContextItem` objects compete for the same token pool.
A large retrieval result can crowd out conversation history. A verbose
system prompt can leave no room for RAG context.

Token budgets solve this by:

- **Allocating per-source caps** -- guarantee each source gets its share.
- **Reserving tokens** -- hold back tokens for the LLM's response.
- **Defining overflow strategies** -- control what happens when a source
  exceeds its cap.
- **Tracking shared pool usage** -- diagnostics show how tokens flow.

## Budget vs. Usage Limits

`TokenBudget` governs **what enters the context window** when the
pipeline assembles a turn: per-source allocations, priorities, and
overflow strategies. It does not cap what an agent's tool loop spends
across rounds — that is `Agent.with_usage_limits(UsageLimits(...))`,
which stops the turn gracefully (`stopped_by="usage_limit"`) when a
token, tool-call, or USD limit is crossed. The limit is a shared pool
for the whole run: subagents spawned during the turn debit the same
budget. See the [Agent Guide](../guides/agent.md#usage-limits).

## Core Models

### TokenBudget

The top-level budget model defines the total token capacity and how it is
divided.

```python
from anchor import TokenBudget, BudgetAllocation, SourceType

budget = TokenBudget(
    total_tokens=8192,
    reserve_tokens=1200,  # Hold back for the LLM response
    allocations=[
        BudgetAllocation(source=SourceType.SYSTEM, max_tokens=800, priority=10),
        BudgetAllocation(source=SourceType.MEMORY, max_tokens=800, priority=8),
        BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=3200, priority=5),
    ],
)
```

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | `int` | Total token budget (must be > 0) |
| `allocations` | `list[BudgetAllocation]` | Per-source allocations |
| `reserve_tokens` | `int` | Tokens reserved for the LLM response (default: 0) |

The model validates that `sum(allocations) + reserve_tokens <= total_tokens`.
If the sum exceeds the total, a `ValueError` is raised at construction time.

### BudgetAllocation

Defines how many tokens a single source type may consume.

```python
from anchor import BudgetAllocation, SourceType

alloc = BudgetAllocation(
    source=SourceType.RETRIEVAL,
    max_tokens=3200,
    priority=5,
    overflow_strategy="truncate",  # or "drop"
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `SourceType` | -- | The source type this allocation applies to |
| `max_tokens` | `int` | -- | Maximum tokens for this source (must be > 0) |
| `priority` | `int` | 5 | Priority used for ordering (1--10) |
| `overflow_strategy` | `"truncate" \| "drop"` | `"truncate"` | What to do when the source exceeds its cap |

## Overflow Strategies

When a source produces more items than its allocation allows, the overflow
strategy determines what happens.

### Truncate (default)

Items are sorted by `(-priority, -score)`. Items are kept until the cap is
reached; the rest overflow.

```
Source "retrieval" cap: 2000 tokens

Item A (800 tokens, score=0.95) --> KEPT     (800 / 2000)
Item B (700 tokens, score=0.85) --> KEPT     (1500 / 2000)
Item C (600 tokens, score=0.70) --> OVERFLOW (would exceed 2000)
Item D (400 tokens, score=0.60) --> OVERFLOW
```

### Drop

If the total tokens for the source exceed the cap, **all** items for that
source are dropped. This is useful when partial retrieval context is worse
than no retrieval context.

```
Source "retrieval" cap: 2000 tokens

Total items: 2500 tokens --> ALL DROPPED (exceeds cap)
```

!!! warning "Drop strategy"
    Use `"drop"` only when your application requires all-or-nothing behavior
    for a source. In most cases, `"truncate"` is the safer choice.

## Reserve Tokens

The `reserve_tokens` field subtracts tokens from the effective `max_tokens`
of the pipeline. This guarantees space for the LLM's response.

```python
from anchor import ContextPipeline, TokenBudget

budget = TokenBudget(total_tokens=8192, reserve_tokens=1200)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
# Effective context window = 8192 - 1200 = 6992 tokens
```

The pipeline will raise a `PipelineExecutionError` if `reserve_tokens >=
max_tokens` (leaving zero or negative space for context).

## Shared Pool

Tokens not explicitly allocated to any source form the **shared pool**.
Sources without an allocation compete for this pool during window assembly.

```python
budget = TokenBudget(
    total_tokens=8192,
    reserve_tokens=1200,          # 1200
    allocations=[
        BudgetAllocation(source=SourceType.SYSTEM, max_tokens=800),    # 800
        BudgetAllocation(source=SourceType.RETRIEVAL, max_tokens=3200), # 3200
    ],
)
print(budget.shared_pool)  # 8192 - 1200 - 800 - 3200 = 2992
```

Items from sources with no explicit allocation (e.g., `SourceType.MEMORY`,
`SourceType.CONVERSATION`, `SourceType.USER` in the example above) draw
from the shared pool.

The `get_allocation()` method returns the per-source cap if one exists, or
the shared pool size as a fallback:

```python
print(budget.get_allocation(SourceType.SYSTEM))     # 800
print(budget.get_allocation(SourceType.RETRIEVAL))   # 3200
print(budget.get_allocation(SourceType.MEMORY))      # 2992 (shared pool)
```

## Preset Factories

Three factory functions provide sensible defaults for common application
types. Each accepts a `max_tokens` parameter and returns a configured
`TokenBudget`.

### default_chat_budget

Optimized for conversational applications with moderate retrieval.

```python
from anchor import default_chat_budget

budget = default_chat_budget(max_tokens=8192)
```

| Source | Allocation | Percentage |
|--------|-----------|------------|
| System | 819 | 10% |
| Memory | 819 | 10% |
| Conversation | 1638 | 20% |
| Retrieval | 2048 | 25% |
| Reserve | 1228 | 15% |
| Shared pool | -- | 20% |

### default_rag_budget

Optimized for RAG-heavy applications where retrieval dominates.

```python
from anchor import default_rag_budget

budget = default_rag_budget(max_tokens=8192)
```

| Source | Allocation | Percentage |
|--------|-----------|------------|
| System | 819 | 10% |
| Memory | 409 | 5% |
| Conversation | 819 | 10% |
| Retrieval | 3276 | 40% |
| Reserve | 1228 | 15% |
| Shared pool | -- | 20% |

### default_agent_budget

Optimized for agentic applications with tool usage.

```python
from anchor import default_agent_budget

budget = default_agent_budget(max_tokens=8192)
```

| Source | Allocation | Percentage |
|--------|-----------|------------|
| System | 1228 | 15% |
| Memory | 819 | 10% |
| Conversation | 1228 | 15% |
| Retrieval | 1638 | 20% |
| Tool | 1228 | 15% |
| Reserve | 1228 | 15% |
| Shared pool | -- | 10% |

!!! tip "Custom budgets"
    The presets are a starting point. For production workloads, construct
    a `TokenBudget` directly with allocations tuned to your application's
    data distribution.

## Using Budgets with the Pipeline

Attach a budget to the pipeline with `.with_budget()`:

```python
from anchor import ContextPipeline, default_rag_budget

budget = default_rag_budget(max_tokens=8192)
pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_budget(budget)
    .add_system_prompt("You are a helpful assistant.")
)
result = pipeline.build("What is context engineering?")
```

You can also pass the budget directly to the constructor:

```python
pipeline = ContextPipeline(max_tokens=8192, budget=budget)
```

## Budget Diagnostics

When a budget is configured, the pipeline's diagnostics include extra
fields that track how tokens were spent:

```python
result = pipeline.build("What is context engineering?")
d = result.diagnostics

# Tokens used per source type
print(d.get("token_usage_by_source"))
# e.g. {"system": 45, "retrieval": 1200, "memory": 300}

# Tokens used by sources without explicit allocations
print(d.get("shared_pool_usage"))
# e.g. 300

# Items dropped because a source exceeded its cap
print(d.get("budget_overflow_by_source"))
# e.g. {"retrieval": 3}  -- 3 retrieval items were dropped
```

!!! note "Overflow vs window overflow"
    Budget overflow happens during per-source cap enforcement (Stage 4a).
    Window overflow happens when total items still exceed `max_tokens` after
    budget filtering (Stage 4b). Both are tracked in diagnostics.

## Source Types

The `SourceType` enum defines the valid source categories:

| Value | Description |
|-------|-------------|
| `SourceType.SYSTEM` | System prompts and instructions |
| `SourceType.MEMORY` | Persistent memory entries |
| `SourceType.CONVERSATION` | Conversation history turns |
| `SourceType.RETRIEVAL` | RAG / search results |
| `SourceType.TOOL` | Tool or function call outputs |
| `SourceType.USER` | Direct user-provided context |

## See Also

- [Architecture](architecture.md) -- how budgets integrate into the pipeline
- [Protocols](protocols.md) -- protocol-based plugin system
- [Pipeline Guide](../guides/pipeline.md) -- practical pipeline examples
