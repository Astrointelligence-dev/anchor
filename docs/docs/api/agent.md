# Agent API Reference

The agent module provides the `Agent` class, `AgentTool` model, `@tool`
decorator, and the skills system for progressive tool disclosure.

All classes are importable from `anchor`:

```python
from anchor import Agent, AgentTool, tool, Skill, SkillRegistry
from anchor import memory_skill, rag_skill, memory_tools, rag_tools
from anchor.llm import LLMProvider, create_provider
```

---

## Agent

High-level agent combining the context pipeline with any LLM provider via
the [`LLMProvider`](llm.md#llmprovider-protocol) protocol. Provides streaming chat
with automatic tool use, memory management, and agentic RAG.

### Constructor

```python
class Agent:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        llm: LLMProvider | None = None,
        fallbacks: list[str] | None = None,
        max_tokens: int = 16384,
        max_response_tokens: int = 1024,
        max_rounds: int = 10,
        tool_timeout: float | None = None,
        tool_result_max_tokens: int | None = 10_000,
    ) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"claude-haiku-4-5-20251001"` | Model string in `"provider/model"` format. No prefix defaults to `anthropic/`. |
| `api_key` | `str \| None` | `None` | API key (falls back to provider-specific env var) |
| `llm` | `LLMProvider \| None` | `None` | Pre-built provider instance. Overrides `model` and `api_key` when set. |
| `fallbacks` | `list[str] \| None` | `None` | Fallback model strings (e.g. `["openai/gpt-4o"]`). Creates a `FallbackProvider`. |
| `max_tokens` | `int` | `16384` | Token budget for the context pipeline |
| `max_response_tokens` | `int` | `1024` | Max tokens in each API response |
| `max_rounds` | `int` | `10` | Max tool-use rounds per `chat()` call |
| `tool_timeout` | `float \| None` | `None` | Default timeout for async tool calls (per-tool `AgentTool.timeout` overrides); sync tools run without one by design |
| `tool_result_max_tokens` | `int \| None` | `10_000` | Head+tail cap applied to every tool result entering the messages (`None` disables; per-tool `AgentTool.max_result_tokens` overrides) |

!!! tip
    See the [LLM Providers Guide](../guides/llm-providers.md) for supported
    providers, installation, and fallback chain configuration.

### Methods

#### with_system_prompt

```python
def with_system_prompt(self, prompt: str) -> Agent
```

Set the system prompt. Clears any previous system prompt. Returns `self`.

#### with_memory

```python
def with_memory(self, memory: MemoryManager) -> Agent
```

Attach a `MemoryManager` for conversation history and persistent facts.
Returns `self`.

#### with_tools

```python
def with_tools(self, tools: list[AgentTool]) -> Agent
```

Add tools (additive). Returns `self`.

#### with_skill

```python
def with_skill(self, skill: Skill) -> Agent
```

Register a single skill. Returns `self`.

#### with_skills

```python
def with_skills(self, skills: list[Skill]) -> Agent
```

Register multiple skills. Returns `self`.

#### chat

```python
def chat(self, message: str) -> Iterator[str]
```

Send a message and stream the response synchronously. Handles the full
tool-use loop: if the model calls tools, they are executed and results fed
back until a final text response or `max_rounds` is reached.

**Yields:** Text chunks as they arrive from the API.

#### achat

```python
async def achat(self, message: str) -> AsyncIterator[str]
```

Async variant of `chat()`. Uses `pipeline.abuild()` and async streaming.

**Yields:** Text chunks as they arrive from the API.

### Properties

| Property | Type | Description |
|---|---|---|
| `memory` | `MemoryManager \| None` | The attached memory manager |
| `pipeline` | `ContextPipeline` | The underlying context pipeline |
| `last_result` | `ContextResult \| None` | Result from the most recent `chat()` call |

### Example

```python
from anchor import Agent, tool

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

agent = (
    Agent(model="claude-haiku-4-5-20251001")
    .with_system_prompt("You are friendly.")
    .with_tools([greet])
)

for chunk in agent.chat("Please greet Alice"):
    print(chunk, end="", flush=True)
```

---

## AgentTool

A frozen Pydantic model representing a tool the Agent can use during
conversation.

### Constructor

```python
class AgentTool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]
    input_model: type[BaseModel] | None = None
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Tool name (exposed to the model) |
| `description` | `str` | required | Tool description (exposed to the model) |
| `input_schema` | `dict[str, Any]` | required | JSON Schema for inputs |
| `fn` | `Callable[..., str]` | required | Callable that executes the tool |
| `input_model` | `type[BaseModel] \| None` | `None` | Optional Pydantic model for validation |
| `read_only` | `bool` | `False` | Declares the tool side-effect free: the async loop runs consecutive read-only calls concurrently; undeclared (write) tools run alone |
| `max_result_tokens` | `int \| None` | `None` | Per-tool override of the agent's tool-result cap (`None` inherits `tool_result_max_tokens`) |

### Methods

#### to_tool_schema

```python
def to_tool_schema(self) -> ToolSchema
```

Convert to a provider-agnostic [`ToolSchema`](llm.md#toolschema). Returns
a `ToolSchema` with `name`, `description`, and `input_schema` fields.

#### validate_input

```python
def validate_input(self, tool_input: dict[str, Any]) -> tuple[bool, str]
```

Validate tool input against the schema. Returns `(True, "")` when valid,
`(False, error_message)` otherwise.

When `input_model` is set, uses full Pydantic validation. Otherwise falls
back to basic JSON Schema type checking.

---

## tool (decorator)

Creates an `AgentTool` from a decorated function with auto-generated JSON
Schema from type hints.

### Signature

```python
@overload
def tool(fn: Callable[..., str]) -> AgentTool: ...

@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., str]], AgentTool]: ...
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fn` | `Callable[..., str] \| None` | `None` | Function to wrap (bare `@tool` usage) |
| `name` | `str \| None` | `None` | Override tool name (defaults to `fn.__name__`) |
| `description` | `str \| None` | `None` | Override description (defaults to first docstring paragraph) |
| `input_model` | `type[BaseModel] \| None` | `None` | Explicit Pydantic input model |

### Examples

```python
from anchor import tool

# Bare usage
@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

# Parameterized usage
@tool(name="custom_add", description="Add two numbers together")
def add_numbers(a: int, b: int) -> str:
    return str(a + b)
```

---

## Skill

A frozen Pydantic model representing a named group of tools with optional
on-demand activation.

### Constructor

```python
class Skill(BaseModel):
    name: str
    description: str
    instructions: str = ""
    tools: tuple[AgentTool, ...] = ()
    activation: Literal["always", "on_demand"] = "always"
    tags: tuple[str, ...] = ()
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique skill identifier |
| `description` | `str` | required | Shown in discovery prompt |
| `instructions` | `str` | `""` | Detailed usage guide injected on activation |
| `tools` | `tuple[AgentTool, ...]` | `()` | Tools this skill provides |
| `activation` | `Literal["always", "on_demand"]` | `"always"` | When tools become available |
| `tags` | `tuple[str, ...]` | `()` | Optional grouping tags |

---

## SkillRegistry

Manages skill registration and activation state.

### Constructor

```python
class SkillRegistry:
    def __init__(self) -> None
```

### Methods

#### register

```python
def register(self, skill: Skill) -> None
```

Register a skill. Raises `ValueError` on duplicate name.

#### activate

```python
def activate(self, name: str) -> Skill
```

Mark an on-demand skill as active. Returns the skill.
Raises `KeyError` if not registered.

#### deactivate

```python
def deactivate(self, name: str) -> None
```

Remove a skill from the active set.

#### reset

```python
def reset(self) -> None
```

Clear all activation state (keeps registrations).

#### get

```python
def get(self, name: str) -> Skill | None
```

Look up a skill by name, or `None` if not found.

#### is_active

```python
def is_active(self, name: str) -> bool
```

Return `True` if the skill's tools should be available now.
Always-loaded skills are always active.

#### active_tools

```python
def active_tools(self) -> list[AgentTool]
```

Return all tools from currently-active skills. Raises `ValueError` if
two active skills provide tools with the same name.

#### on_demand_skills

```python
def on_demand_skills(self) -> list[Skill]
```

Return skills that require activation.

#### skill_discovery_prompt

```python
def skill_discovery_prompt(self) -> str
```

Build the Tier-1 discovery text for the system prompt.
Returns an empty string when there are no on-demand skills.

---

## memory_skill

Factory function that creates a `Skill` wrapping memory CRUD tools.

### Signature

```python
def memory_skill(memory: MemoryManager) -> Skill
```

**Returns** a skill with four tools: `save_fact`, `search_facts`,
`update_fact`, `delete_fact`. Activation is `"always"`.

---

## rag_skill

Factory function that creates a `Skill` wrapping document search tools.

### Signature

```python
def rag_skill(
    retriever: object,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> Skill
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `retriever` | `object` | Any object with a `retrieve(query, top_k)` method |
| `embed_fn` | `Callable[[str], list[float]] \| None` | Optional embedding function |

**Returns** a skill with one tool: `search_docs`. Activation is `"on_demand"`.

---

## memory_tools

Factory function that creates memory CRUD tools directly (without wrapping
in a Skill).

### Signature

```python
def memory_tools(memory: MemoryManager) -> list[AgentTool]
```

**Returns** four tools: `save_fact`, `search_facts`, `update_fact`, `delete_fact`.

---

## rag_tools

Factory function that creates RAG search tools directly (without wrapping
in a Skill).

### Signature

```python
def rag_tools(
    retriever: Any,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[AgentTool]
```

**Returns** a list containing one tool: `search_docs`.

---

## See Also

- [Agent Guide](../guides/agent.md) -- usage guide with examples
- [LLM Providers Guide](../guides/llm-providers.md) -- multi-provider setup and fallbacks
- [LLM API Reference](llm.md) -- provider protocol, models, and errors
- [Pipeline API Reference](../api/pipeline.md) -- underlying pipeline
- [Protocols Reference](../api/protocols.md) -- extension point protocols
