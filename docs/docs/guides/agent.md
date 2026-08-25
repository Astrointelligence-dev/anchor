# Agent Guide

The Agent module provides a high-level, batteries-included interface that combines
the context pipeline with any LLM provider. It handles streaming chat, automatic
tool use loops, memory management, and agentic RAG -- all through a fluent builder
API.

## Overview

The `Agent` class wraps three systems into a single entry point:

1. **ContextPipeline** -- assembles system prompts, memory, and retrieval context
2. **LLM Provider** -- streams responses via the [`LLMProvider`](../api/llm.md#llmprovider-protocol) protocol with automatic retries
3. **Tool loop** -- executes tools and feeds results back to the model

```
User message
    |
    v
ContextPipeline.build()  -->  system + messages
    |
    v
LLM Provider (streaming)
    |
    v
Tool use?  -- yes --> execute tools --> feed back --> loop
    |
    no
    v
Yield text chunks
```

## Quick Start

```python
from anchor import Agent

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_system_prompt("You are a helpful assistant.")
)

for chunk in agent.chat("What is context engineering?"):
    print(chunk, end="", flush=True)
```

!!! note
    The Agent requires at least one LLM provider SDK. The default is Anthropic:
    `pip install astro-anchor[anthropic]`. See the
    [LLM Providers Guide](llm-providers.md) for all supported providers.

## Constructor

```python
Agent(
    model: str = "claude-haiku-4-5-20251001",
    *,
    api_key: str | None = None,
    llm: LLMProvider | None = None,
    fallbacks: list[str] | None = None,
    max_tokens: int = 16384,
    max_response_tokens: int = 1024,
    max_rounds: int = 10,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-haiku-4-5-20251001"` | Model string in `"provider/model"` format. No prefix defaults to `anthropic/`. |
| `api_key` | `str \| None` | `None` | API key (falls back to provider-specific env var) |
| `llm` | `LLMProvider \| None` | `None` | Pre-built provider instance. Overrides `model`/`api_key`. |
| `fallbacks` | `list[str] \| None` | `None` | Fallback model strings (e.g. `["openai/gpt-4o"]`) |
| `max_tokens` | `int` | `16384` | Token budget for the context pipeline |
| `max_response_tokens` | `int` | `1024` | Maximum tokens in each API response |
| `max_rounds` | `int` | `10` | Maximum tool-use rounds per chat call |

## Fluent Configuration

All configuration methods return `self` for chaining:

```python
from anchor import Agent, MemoryManager, InMemoryEntryStore

memory = MemoryManager(store=InMemoryEntryStore())

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_system_prompt("You are a helpful coding assistant.")
    .with_memory(memory)
)
```

### with_system_prompt

```python
agent.with_system_prompt(prompt: str) -> Agent
```

Sets the system prompt. Clears any previous system prompt and registers
the new one with the underlying pipeline.

### with_memory

```python
agent.with_memory(memory: MemoryManager) -> Agent
```

Attaches a `MemoryManager` for conversation history and persistent facts.
The agent automatically records user messages, assistant responses, and tool
calls in memory.

### with_tools

```python
agent.with_tools(tools: list[AgentTool]) -> Agent
```

Adds tools (additive -- multiple calls accumulate tools). Each `AgentTool`
is exposed to the model during the tool-use loop.

### with_skill / with_skills

```python
agent.with_skill(skill: Skill) -> Agent
agent.with_skills(skills: list[Skill]) -> Agent
```

Registers one or more [Skills](#skills-system). Always-loaded skills have
their tools available immediately. On-demand skills are advertised in a
discovery prompt and activated via the `activate_skill` meta-tool.

## Chat Methods

### Synchronous Streaming

```python
for chunk in agent.chat("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

`chat()` returns an `Iterator[str]` that yields text chunks as they arrive.
If the model calls tools, the agent executes them and feeds results back
automatically, continuing until a final text response or `max_rounds` is
reached.

### Async Streaming

```python
async for chunk in agent.achat("Explain quantum computing"):
    print(chunk, end="", flush=True)
```

`achat()` is the async counterpart. It uses `pipeline.abuild()` and async
iteration over the streaming API.

### Event Stream

`chat()`/`achat()` are text-only projections of a richer stream.
`stream()`/`astream()` expose the full tool-use loop as one ordered
sequence of typed events (`AgentEvent`, discriminated by `.type`) —
built for UIs that need more than text:

```python
from anchor import (
    RoundStarted, TextDelta, ToolStarted, ToolFinished,
    RoundFinished, TurnFinished,
)

async for event in agent.astream("Find and summarize the report"):
    match event:
        case RoundStarted(round=r, max_rounds=n):
            status.update(f"round {r + 1}/{n}")
        case TextDelta(text=text):
            log.write(text)
        case ToolStarted(tool_call_id=cid, name=name):
            spinners[cid] = show_spinner(name)
        case ToolFinished(tool_call_id=cid, is_error=err):
            spinners.pop(cid).done(error=err)
        case RoundFinished(usage=usage):
            meter.add(usage.prompt_tokens, usage.completion_tokens)
        case TurnFinished(diagnostics=diag):
            status.update(f"done ({diag.stopped_by})")
```

The event vocabulary: `TurnStarted`, `RoundStarted`, `TextDelta`,
`ToolStarted`/`ToolFinished` (correlated by `tool_call_id` — tool calls
run concurrently in the async path and finish events arrive live, in
completion order), `CompactionStarted`/`CompactionFinished`,
`RoundFinished` (with per-round `RoundUsage`), and a terminal
`TurnFinished` carrying the final text and `TurnDiagnostics`.

Tool failures surface as `ToolFinished(is_error=True)`, not exceptions;
only turn-level failures (provider, MCP) raise. Events forwarded from a
subagent's run arrive flat in the same stream with `parent_tool_call_id`
set to the invoking tool call — top-level events leave it `None`, which
is also how the text projections filter subagent text out.

### Accessing the Last Result

After calling `chat()` or `achat()`, the full `ContextResult` is available:

```python
for chunk in agent.chat("Hello"):
    pass

result = agent.last_result
print(result.diagnostics)
```

## The @tool Decorator

The `@tool` decorator converts a plain function into an `AgentTool` with
auto-generated JSON Schema from type hints:

```python
from anchor import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22 {units}"
```

The decorator extracts:

- **name** from `fn.__name__` (override with `name=`)
- **description** from the first docstring paragraph (override with `description=`)
- **input_schema** from type hints (override with `input_model=`)

### Parameterized Usage

```python
from pydantic import BaseModel
from anchor import tool

class SearchInput(BaseModel):
    query: str
    max_results: int = 5

@tool(name="search", description="Search the knowledge base", input_model=SearchInput)
def search_kb(query: str, max_results: int = 5) -> str:
    """Search the knowledge base."""
    return f"Found {max_results} results for: {query}"
```

!!! tip
    When you provide an `input_model`, validation uses full Pydantic validation
    instead of basic JSON Schema type checking. This gives you richer constraints
    like `ge=`, `le=`, `pattern=`, etc.

### Three Tiers of Tool Creation

| Tier | Approach | Schema Source |
|---|---|---|
| 1 | `@tool` bare decorator | Auto-generated from type hints |
| 2 | `@tool(input_model=MyModel)` | Explicit Pydantic model |
| 3 | Direct `AgentTool(...)` construction | Raw `input_schema` dict |

## AgentTool Model

`AgentTool` is a frozen Pydantic model:

```python
from anchor import AgentTool

my_tool = AgentTool(
    name="lookup",
    description="Look up a value",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    fn=lambda key: f"Value for {key}",
)
```

Key methods:

- `to_tool_schema()` -- Provider-agnostic [`ToolSchema`](../api/llm.md#toolschema)
- `validate_input(tool_input)` -- Returns `(bool, str)` tuple

## Skills System

Skills group related tools into discoverable units with optional on-demand
activation. This implements **progressive tool disclosure** -- the model
starts with a small tool set and can activate more capabilities as needed.

### Skill Model

```python
from anchor import Skill

my_skill = Skill(
    name="data_analysis",
    description="Tools for analyzing datasets",
    instructions="Use these tools when the user asks about data trends.",
    tools=(analyze_tool, summarize_tool),
    activation="on_demand",  # or "always"
    tags=("analytics",),
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique identifier |
| `description` | `str` | required | Shown in discovery prompt |
| `instructions` | `str` | `""` | Injected when skill is activated |
| `tools` | `tuple[AgentTool, ...]` | `()` | Tools this skill provides |
| `activation` | `Literal["always", "on_demand"]` | `"on_demand"` | When tools become available |
| `tags` | `tuple[str, ...]` | `()` | Grouping/filtering tags |

### Activation Modes

**Always-loaded** skills have their tools available from the first API round:

```python
agent.with_skill(Skill(
    name="utils",
    description="Utility tools",
    tools=(calc_tool,),
    activation="always",
))
```

**On-demand** skills are advertised in a discovery prompt. The agent calls
the auto-generated `activate_skill` meta-tool to make their tools available:

```python
agent.with_skill(Skill(
    name="advanced_search",
    description="Advanced search with filters and facets",
    tools=(faceted_search_tool, filter_tool),
    activation="on_demand",
))
```

### SkillRegistry

The `SkillRegistry` manages skill registration and activation state:

```python
from anchor import SkillRegistry, Skill

registry = SkillRegistry()
registry.register(my_skill)

# Check status
registry.is_active("data_analysis")  # False (on_demand, not yet activated)

# Activate
skill = registry.activate("data_analysis")

# Get all active tools
tools = registry.active_tools()
```

## Built-in Skills

### memory_skill

Creates a skill with four CRUD tools for persistent user facts:

```python
from anchor import Agent, MemoryManager, InMemoryEntryStore, memory_skill

memory = MemoryManager(store=InMemoryEntryStore())

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_memory(memory)
    .with_skill(memory_skill(memory))
)
```

The memory skill provides:

| Tool | Description |
|---|---|
| `save_fact` | Save a new fact about the user |
| `search_facts` | Search previously saved facts |
| `update_fact` | Update an existing fact by ID |
| `delete_fact` | Delete an outdated fact by ID |

!!! warning
    The memory skill's `activation` is `"always"` by default. All four tools
    are available from the first round.

### rag_skill

Creates an on-demand skill with a `search_docs` tool for agentic RAG:

```python
from anchor import Agent, rag_skill

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_skill(rag_skill(retriever=my_retriever, embed_fn=my_embed_fn))
)
```

The model decides when to activate the skill and search documentation,
making this **agentic RAG** -- retrieval timing is model-controlled.

| Parameter | Type | Description |
|---|---|---|
| `retriever` | object | Any object with `retrieve(query, top_k)` |
| `embed_fn` | `Callable[[str], list[float]] \| None` | Optional embedding function |

!!! note
    The RAG skill's `activation` is `"on_demand"`. The agent must call
    `activate_skill("rag")` before `search_docs` becomes available.

## Putting It All Together

```python
from anchor import (
    Agent, MemoryManager, InMemoryEntryStore,
    memory_skill, rag_skill, tool,
)

# Custom tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Setup
memory = MemoryManager(store=InMemoryEntryStore())

agent = (
    Agent(model="claude-haiku-4-5-20251001", max_rounds=5)
    .with_system_prompt("You are a helpful assistant with memory and tools.")
    .with_memory(memory)
    .with_tools([calculate])
    .with_skill(memory_skill(memory))
)

# Chat
for chunk in agent.chat("Remember that my favorite color is blue"):
    print(chunk, end="", flush=True)
```

## Error Handling and Retries

The underlying `BaseLLMProvider` retries on transient errors with exponential
backoff. Transient errors include:

- `RateLimitError` -- 429 responses (respects `retry_after` header)
- `ServerError` -- 5xx responses
- `TimeoutError` -- request timeouts

Non-transient errors (`AuthenticationError`, `ModelNotFoundError`,
`ContentFilterError`) raise immediately.

Tool execution errors are caught and returned as error messages to the model,
allowing it to recover gracefully.

See the [LLM Providers Guide](llm-providers.md) for details on fallback
chains and the error hierarchy.

## See Also

- [LLM Providers Guide](llm-providers.md) -- multi-provider setup and fallbacks
- [Pipeline Guide](../guides/pipeline.md) -- underlying context assembly
- [Memory Guide](../guides/memory.md) -- memory management details
- [Agent API Reference](../api/agent.md) -- complete API signatures
- [LLM API Reference](../api/llm.md) -- provider protocol, models, and errors
- [Formatters Guide](../guides/formatters.md) -- output formatting
