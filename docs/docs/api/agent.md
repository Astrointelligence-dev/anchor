# Agent API Reference

The agent module provides the `Agent` class, `AgentTool` model, `@tool`
decorator, the typed event stream, usage limits, subagents, hooks and
approval, structured output, and the skills system for progressive tool
disclosure.

All classes are importable from `anchor`:

```python
from anchor import Agent, AgentTool, tool, Skill, SkillRegistry
from anchor import memory_skill, rag_skill, memory_tools, rag_tools
from anchor import (
    AgentEvent, TurnStarted, RoundStarted, TextDelta, ToolStarted,
    ToolFinished, CompactionStarted, CompactionFinished, RoundFinished,
    UsageLimitReached, TurnFinished,
    UsageLimits, RoundUsage, TurnDiagnostics, ChildTurn,
    SubagentDefinition, HookResult, ApprovalRequest, ApprovalDecision,
    AgentCallback, FileMemoryBackend,
)
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

#### with_skills_directory / with_skill_from_path

```python
def with_skills_directory(self, path: str | Path) -> Agent
def with_skill_from_path(self, path: str | Path) -> Agent
```

Load `SKILL.md` skills from a directory tree (or a single skill directory).
Returns `self`.

#### with_mcp_servers

```python
def with_mcp_servers(self, servers: list[str | MCPServerConfig]) -> Agent
```

Connect to external MCP servers. Accepts `MCPServerConfig` objects or
convenience strings (URLs for HTTP, commands for STDIO). Connections are
lazy and async-only — use `astream()`/`achat()` and `aclose()` (or
`async with agent:`). See the [MCP guide](../guides/mcp.md). Returns `self`.

#### with_budget

```python
def with_budget(self, budget: TokenBudget) -> Agent
```

Attach a `TokenBudget` to the context pipeline — governs what enters the
context window. To cap what a run may *spend*, see
[`with_usage_limits`](#with_usage_limits). Returns `self`.

#### with_usage_limits

```python
def with_usage_limits(self, limits: UsageLimits) -> Agent
```

Enforce [`UsageLimits`](#usagelimits) across the turn **and its subagents**:
the turn runs against a shared run-wide pool that every subagent spawned
during it (`task`/`as_tool`) debits, so a child's spend counts against the
orchestrator's budget. Crossing a limit emits
[`UsageLimitReached`](#usagelimitreached), grants the model one wrap-up
round (final-round notice; `tool_choice="none"`, or the `final_result` tool
when structured output is still pending) and ends the turn with
`stopped_by="usage_limit"` — no exception is raised. A child cut mid-run
wraps up the same way and returns a marked partial result. Returns `self`.

#### with_scope

```python
def with_scope(self, scope: RetrievalScope) -> Agent
```

Set the agent's retrieval scope — **namespaces only** (the vault is a
store mount, bound at construction, never part of a query-time object).
The scope is published for the duration of each tool call, so
scope-aware tools (`rag_tools`, custom tools via `current_scope()`) and
subagent turns see it; a subagent's effective scope is the intersection
with its own — a child can only narrow, never widen. Returns `self`.

#### with_hooks

```python
def with_hooks(
    self,
    *,
    pre_tool_use: list[PreToolHook] | None = None,
    post_tool_use: list[PostToolHook] | None = None,
) -> Agent
```

Add veto hooks around tool execution (additive). A pre-hook
`(tool_name, tool_input) -> HookResult | None` may deny a call (the reason
is fed back to the model as an `is_error` tool result), rewrite its input,
or answer `"ask"` to route the call to the approval callback; a post-hook
`(tool_name, tool_input, output) -> HookResult | None` may replace the
output. A pre-hook that raises fails closed: the call is denied. Returns
`self`.

#### with_approval

```python
def with_approval(self, callback: ApprovalCallback) -> Agent
```

Set the inline human-in-the-loop approval callback:
`(ApprovalRequest) -> ApprovalDecision`, sync or async. Called for tools
marked `requires_approval=True` and for calls a pre-hook answered `"ask"` —
the tool call pauses until the callback returns (an async callback may stay
pending indefinitely; timeouts are the application's choice). Deny becomes
an `is_error` tool result carrying the reason; an approval may rewrite the
input. Without a callback, approval-gated calls fail closed. Parallel tool
calls run their approval callbacks concurrently — serialize inside the
callback if your UI needs one prompt at a time. Returns `self`.

#### with_output_model

```python
def with_output_model(
    self,
    output_model: type[BaseModel],
    *,
    mode: str = "tool",          # "tool" | "prompted"
    max_output_retries: int = 1,
) -> Agent
```

Require schema-validated structured output; consume it via
[`run()`/`arun()`](#run--arun), `TurnFinished.output`, or
`agent.last_output`.

- `mode="tool"` (default, portable): a synthetic `final_result` tool
  carries the schema and `tool_choice="any"` keeps the model from stopping
  in plain text; invalid arguments come back as an error tool result — the
  loop's own retry mechanic — bounded by `max_output_retries`, after which
  the turn ends with `stopped_by="output_missing"` and `run()` raises.
- `mode="prompted"`: the schema is appended to the prompt and the reply is
  validated, with self-contained retry turns (the subagent mechanic).
  Requires an agent without `with_memory`.

Returns `self`.

#### with_callbacks

```python
def with_callbacks(self, callbacks: list[AgentCallback]) -> Agent
```

Add observer callbacks for loop events (additive). Fire-and-forget:
exceptions are swallowed and logged. See [`AgentCallback`](#agentcallback).
Returns `self`.

#### with_context_management

```python
def with_context_management(self, config: dict[str, Any]) -> Agent
```

Attach an Anthropic `context_management` config, passed through with every
request; the Anthropic provider routes to the beta API with the required
flags, other providers ignore it. `compaction` blocks returned by the API
round-trip verbatim within the turn. For a provider-agnostic alternative,
see [`with_compaction`](#with_compaction). Returns `self`.

#### with_compaction

```python
def with_compaction(
    self,
    trigger_tokens: int,
    *,
    keep_last: int = 4,
    compact_fn: Callable[[str], str] | None = None,
) -> Agent
```

Enable client-side compaction of the tool loop (works with every provider).
When the turn's working messages exceed `trigger_tokens`, older messages
are summarized into a single `[Conversation summary]` user message; the
last `keep_last` messages are kept intact. `compact_fn` receives the
flattened transcript and returns the summary; the default uses
`TierCompactor` with this agent's own LLM. Progress is visible as
`CompactionStarted`/`CompactionFinished` events. Returns `self`.

#### with_memory_tool

```python
def with_memory_tool(self, backend: FileMemoryBackend | str | Path) -> Agent
```

Attach the client-side memory tool (`memory_20250818`-compatible, any
provider). Accepts a `FileMemoryBackend` or a base path; registers the
`memory` tool (view/create/str_replace/insert/delete/rename with strict
`/memories` containment) and appends the memory protocol to the system
prompt. Returns `self`.

#### with_subagents

```python
def with_subagents(self, definitions: list[SubagentDefinition]) -> Agent
```

Register subagents and the `task(agent_name, task)` meta-tool. Each
[`SubagentDefinition`](#subagentdefinition) becomes an isolated
sub-`Agent` (own system prompt, restricted tools, no memory); a discovery
listing is appended to the system prompt. Subagents cannot have subagent
tools of their own (no nesting — enforced at registration and re-checked
at call time). Returns `self`.

#### as_tool

```python
def as_tool(
    self,
    name: str,
    description: str,
    *,
    output_model: type[BaseModel] | None = None,
    max_output_retries: int = 1,
) -> AgentTool
```

Expose this agent as a subagent tool for an orchestrator. The tool takes a
single `task` string — the subagent starts with a clean context. With
`output_model`, the subagent must return schema-valid JSON; invalid output
is retried `max_output_retries` times with the validation error. Agents
that already have subagent tools cannot be wrapped (no nesting).

#### stream / astream

```python
def stream(self, message: str) -> Iterator[AgentEvent]
async def astream(self, message: str) -> AsyncGenerator[AgentEvent, None]
```

Send a message and stream the full tool-use loop as one ordered stream of
typed [events](#events): `TurnStarted`, per-round
`RoundStarted`/`RoundFinished` (with `RoundUsage`), `TextDelta`,
`ToolStarted`/`ToolFinished` (correlated by `tool_call_id`), compaction and
usage-limit events, and a terminal `TurnFinished` with the final text and
`TurnDiagnostics`. `chat`/`achat` are the text-only projections of the same
loop.

`astream` additionally schedules tools by side effect: consecutive
`read_only` calls run concurrently (cap 10), writes run alone;
`ToolFinished` events are emitted live in completion order. Subagent events
are forwarded flat into the parent stream with `parent_tool_call_id` set.
Diagnostics persist (`agent.last_turn`) even when the consumer abandons the
generator. MCP servers require the async variant.

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

#### run / arun

```python
def run(self, message: str) -> BaseModel
async def arun(self, message: str) -> BaseModel
```

Run a turn and return the validated structured-output instance. Requires
[`with_output_model`](#with_output_model). Raises when the turn ends
without valid output (`stopped_by="output_missing"`); `chat`/`stream`
consumers see the verdict instead of an exception.

#### aclose

```python
async def aclose(self) -> None
```

Clean up MCP connections and other async resources. The agent is also an
async context manager: `async with Agent(...).with_mcp_servers([...]) as agent:`.

### Properties

| Property | Type | Description |
|---|---|---|
| `memory` | `MemoryManager \| None` | The attached memory manager |
| `pipeline` | `ContextPipeline` | The underlying context pipeline |
| `last_result` | `ContextResult \| None` | Result from the most recent `chat()` call |
| `last_turn` | `TurnDiagnostics \| None` | Per-round accounting and outcome of the most recent turn (reset at turn start) |
| `last_output` | `str \| None` | Normalized structured-output JSON from the most recent turn |

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
    timeout: float | None = None
    defer_loading: bool = False
    input_examples: tuple[dict[str, Any], ...] = ()
    requires_approval: bool = False
    read_only: bool = False
    max_result_tokens: int | None = None
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Tool name (exposed to the model) |
| `description` | `str` | required | Tool description (exposed to the model) |
| `input_schema` | `dict[str, Any]` | required | JSON Schema for inputs |
| `fn` | `Callable[..., str]` | required | Callable that executes the tool |
| `input_model` | `type[BaseModel] \| None` | `None` | Optional Pydantic model for validation |
| `timeout` | `float \| None` | `None` | Per-call timeout on async tool callers (overrides `Agent(tool_timeout=...)`); sync tools run without one by design |
| `defer_loading` | `bool` | `False` | Keep the schema out of the prompt until the auto-registered `search_tools` meta-tool loads it |
| `input_examples` | `tuple[dict, ...]` | `()` | Example inputs forwarded to providers that support them (Anthropic) |
| `requires_approval` | `bool` | `False` | Route every call through the [approval callback](#with_approval); fails closed without one |
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

## Events

`stream()`/`astream()` yield `AgentEvent` — a discriminated union (by the
`type` literal) of frozen Pydantic models. Every event carries
`parent_tool_call_id: str | None`: `None` for top-level events, the parent
`task`/subagent tool call's id for events forwarded flat from a subagent.

| Event | `type` | Fields | Meaning |
|---|---|---|---|
| `TurnStarted` | `"turn_started"` | — | Context built; rounds about to run |
| `RoundStarted` | `"round_started"` | `round`, `max_rounds` | A model round is starting (0-based) |
| `TextDelta` | `"text_delta"` | `text` | Incremental assistant text (the projection target of `chat`) |
| `ToolStarted` | `"tool_started"` | `tool_call_id`, `name`, `tool_input` | A tool call is about to execute (a pre-hook may still rewrite the input) |
| `ToolFinished` | `"tool_finished"` | `tool_call_id`, `name`, `result`, `is_error` | Tool call completed — tool failure is `is_error=True`, not an exception |
| `CompactionStarted` | `"compaction_started"` | — | Client-side compaction is summarizing older messages (LLM call) |
| `CompactionFinished` | `"compaction_finished"` | `tokens_before`, `tokens_after` | Compaction replaced the head with a summary |
| `RoundFinished` | `"round_finished"` | `round`, `usage: RoundUsage` | A round completed, with its token accounting |
| `UsageLimitReached` | `"usage_limit_reached"` | `kind`, `used`, `limit`, `scope` | A usage limit was crossed; the next round is the wrap-up round |
| `TurnFinished` | `"turn_finished"` | `text`, `diagnostics: TurnDiagnostics`, `output` | Terminal event: final text, diagnostics, structured-output JSON |

`UsageLimitReached.kind` is `"total_tokens" | "tool_calls" | "cost"` (for
`"cost"`, `used`/`limit` are USD); `scope="run"` means the shared pool
spanning subagents tripped, `scope="turn"` one agent's own per-turn limit.

```python
for event in agent.stream("Summarize the report"):
    match event.type:
        case "text_delta":
            print(event.text, end="", flush=True)
        case "tool_started":
            print(f"\n[{event.name}...]")
        case "turn_finished":
            print(f"\n({event.diagnostics.total_tokens} tokens)")
```

---

## UsageLimits

```python
class UsageLimits(BaseModel, frozen=True):
    total_tokens_limit: int | None = None   # > 0
    tool_calls_limit: int | None = None     # >= 0
    cost_limit: float | None = None         # USD, > 0
```

Limits enforced by the agent loop, shared across subagents: the agent that
starts a turn with limits creates a run-wide pool that every subagent
spawned during it debits — the effective budget only ever narrows (a child
may carry its own narrower per-turn limits on top). Crossing a limit grants
one wrap-up round and stops with `stopped_by="usage_limit"`; no exception.
Checks are post-hoc, so a run may overshoot by the rounds in flight plus
the bounded wrap-up calls.

`cost_limit` is priced per round from, in order: provider-reported billed
cost (`claude_cli` sends it), the runtime-overridable
`anchor.llm.pricing.MODEL_PRICING` table, and genai-prices
(`pip install astro-anchor[pricing]`); an unpriced model warns once and
debits $0 (its tokens still count).

---

## RoundUsage

Per-round token accounting (frozen). Fields: `round`, `prompt_tokens`,
`completion_tokens`, `tool_schema_tokens`, `tool_result_tokens`,
`cache_creation_tokens`, `cache_read_tokens`, `tool_calls`, `cost_usd`,
plus the `total_tokens` property (full input + output, as billed).
`prompt_tokens`/`completion_tokens` come from provider-reported usage;
when the provider reports none on the stream they are estimated with the
agent's tokenizer. `tool_schema_tokens`/`tool_result_tokens` are
visibility subsets of the prompt, never added on top of it.

---

## TurnDiagnostics

Accounting and outcome for one full turn (frozen), available as
`agent.last_turn` and on `TurnFinished.diagnostics`.

| Member | Type | Description |
|---|---|---|
| `rounds` | `tuple[RoundUsage, ...]` | This agent's own model rounds |
| `stopped_by` | `Literal["stop", "max_rounds", "max_tokens", "usage_limit", "output_missing", "stuck"]` | Why the turn ended |
| `children` | `tuple[ChildTurn, ...]` | Subagent turns observed while this turn ran |
| `total_prompt_tokens` / `total_completion_tokens` / `total_tool_result_tokens` / `total_tokens` / `total_tool_calls` / `total_cost_usd` | properties | Sums over `rounds` |
| `run_total_tokens` / `run_total_tool_calls` / `run_total_cost_usd` | properties | This turn **plus every subagent turn under it**, recursively |

`stopped_by` values: `"stop"` — the model finished; `"max_rounds"` — the
round cap cut the turn; `"max_tokens"` — the provider cut the response;
`"usage_limit"` — a `UsageLimits` breach ended the turn after wrap-up;
`"output_missing"` — structured output never validated within
`max_output_retries`; `"stuck"` — the loop detected identical repeated
tool calls and ended the turn after a nudge and wrap-up.

`ChildTurn` (frozen): `tool_call_id`, `name`, `diagnostics` — one entry
per subagent turn; a child's own `stopped_by` makes partial results
machine-visible, and a child that died mid-turn still appears with the
accounting it accrued.

---

## SubagentDefinition

```python
class SubagentDefinition(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    model: str | None = None            # None inherits the orchestrator's provider
    tools: tuple[AgentTool, ...] = ()
    output_model: type[BaseModel] | None = None
    max_rounds: int = 6
    usage_limits: UsageLimits | None = None   # narrower per-turn limits; the run pool still applies
    scope: RetrievalScope | None = None       # narrower namespace scope; the parent's still intersects
```

Declarative description of a subagent for
[`with_subagents`](#with_subagents). Each definition becomes an isolated
sub-`Agent` with a clean context; the model delegates via the
`task(agent_name, task)` meta-tool.

---

## RetrievalScope

```python
class RetrievalScope(BaseModel, frozen=True):
    include: tuple[str, ...] = ()   # namespace prefixes; empty = whole vault
    exclude: tuple[str, ...] = ()   # exclude ALWAYS wins
```

Navigation scope over hierarchical namespaces (`/campanha-1/sessoes`).
Prefix matching is boundary-aware (`/campanha-1` never matches
`/campanha-10`). `matches(namespace)` evaluates one path;
`intersect(child)` produces the effective child scope — excludes union,
includes keep the deeper prefix per pair, and disjoint non-empty
includes yield a scope that matches nothing. Importable from `anchor`;
`current_scope()` (from `anchor.agent`) returns the scope published for
the current tool call.

Vector stores accept `search(..., scope=...)` as a pre-filter, and
`rag_tools`/`rag_skill`/`retriever_step` thread it through. Passing a
**dict of mounts** to `rag_tools({"juridico": r1, "notas": r2})` gives
the search tool a `vault` argument restricted to the mounted names —
an unmounted vault is an error result, never a lookup.

---

## Hooks and approval

### HookResult

```python
class HookResult(BaseModel, frozen=True):
    decision: Literal["allow", "deny", "ask"] = "allow"
    reason: str | None = None
    updated_input: dict[str, Any] | None = None    # pre-hooks
    updated_output: str | None = None              # post-hooks
```

Decision returned by a tool hook. `decision`/`reason` apply to pre-hooks
only — the reason is fed back to the model on deny; `"ask"` routes the
call to the approval callback. Hook signatures:
`PreToolHook = (tool_name, tool_input) -> HookResult | None` and
`PostToolHook = (tool_name, tool_input, output) -> HookResult | None`
(`None` = allow unchanged).

### ApprovalRequest / ApprovalDecision

```python
class ApprovalRequest(BaseModel, frozen=True):
    tool_call_id: str
    name: str
    tool_input: dict[str, Any]

class ApprovalDecision(BaseModel, frozen=True):
    approved: bool
    reason: str | None = None                     # fed back to the model on deny
    updated_input: dict[str, Any] | None = None   # replaces the input on approve

ApprovalCallback = Callable[[ApprovalRequest],
                            ApprovalDecision | Awaitable[ApprovalDecision]]
```

The inline human-in-the-loop seam — see [`with_approval`](#with_approval).

### AgentCallback

Observer protocol for loop events; all methods are optional no-ops and
exceptions are swallowed and logged: `on_round_start(round_index)`,
`on_round_end(round_index)`, `on_tool_start(name, tool_input)`,
`on_tool_end(name, tool_input, result)`,
`on_tool_error(name, tool_input, error)`. Register with
[`with_callbacks`](#with_callbacks); `TracingAgentCallback` (observability)
implements it.

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
- [MCP Guide](../guides/mcp.md) -- consuming and exposing MCP servers
- [LLM API Reference](llm.md) -- provider protocol, models, and errors
- [MCP API Reference](mcp.md) -- bridge classes and configuration
- [Pipeline API Reference](../api/pipeline.md) -- underlying pipeline
- [Protocols Reference](../api/protocols.md) -- extension point protocols
