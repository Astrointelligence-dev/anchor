# Agent with Tools

Build an agent with custom tools, skills, and memory using the `@tool`
decorator and `SkillRegistry`.

---

!!! warning "LLM Provider Required"
    The `Agent` class requires at least one LLM provider SDK. The default
    provider is Anthropic: `pip install astro-anchor[anthropic]`. See the
    [LLM Providers Guide](../guides/llm-providers.md) for all supported
    providers. The tool creation and skill registration sections below run
    without any provider SDK. The `agent.chat()` call at the end requires
    a configured provider (e.g. `ANTHROPIC_API_KEY`).

## Overview

This example demonstrates:

- Creating custom tools with the `@tool` decorator
- Inspecting auto-generated tool schemas
- Registering skills with `SkillRegistry`
- Building an `Agent` with tools, memory, and skills
- Using built-in `memory_tools()` and `rag_tools()`

## Tool Creation (No API Key Needed)

```python
from anchor import tool, AgentTool

# ---------------------------------------------------------------
# 1. Basic @tool decorator -- auto-generates schema from type hints
# ---------------------------------------------------------------
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Only supports basic arithmetic operations.
    """
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is supported."
    try:
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# The @tool decorator returns an AgentTool, not a function
print(f"Type: {type(calculate)}")  # <class 'AgentTool'>
print(f"Name: {calculate.name}")
print(f"Description: {calculate.description}")
print(f"Schema: {calculate.input_schema}")

# ---------------------------------------------------------------
# 2. @tool with explicit name and description
# ---------------------------------------------------------------
@tool(name="get_weather", description="Look up current weather for a city.")
def weather(city: str, units: str = "celsius") -> str:
    """Get weather information."""
    # In a real app, this would call a weather API
    return f"Weather in {city}: 22 degrees {units}, partly cloudy."

print(f"\nName: {weather.name}")       # "get_weather"
print(f"Schema: {weather.input_schema}")

# ---------------------------------------------------------------
# 3. Validate tool inputs
# ---------------------------------------------------------------
valid, error = calculate.validate_input({"expression": "2 + 3"})
print(f"\nValid input: {valid}, error: '{error}'")

valid, error = calculate.validate_input({})  # missing required field
print(f"Missing field: {valid}, error: '{error}'")

# ---------------------------------------------------------------
# 4. Execute tools directly
# ---------------------------------------------------------------
result = calculate.fn(expression="(10 + 5) * 2")
print(f"\nDirect call: {result}")

result = weather.fn(city="Tokyo")
print(f"Direct call: {result}")

# ---------------------------------------------------------------
# 5. Export to different provider formats
# ---------------------------------------------------------------
print("\n=== Provider Schemas ===\n")
schema = calculate.to_tool_schema()
print(f"ToolSchema: name={schema.name}, description={schema.description}")
print(f"Input schema: {schema.input_schema}")
```

## Skills and SkillRegistry (No API Key Needed)

```python
from anchor import (
    tool,
    AgentTool,
    Skill,
    SkillRegistry,
)

# ---------------------------------------------------------------
# 1. Create tools for a "data analysis" skill
# ---------------------------------------------------------------
@tool
def count_words(text: str) -> str:
    """Count the number of words in a text."""
    count = len(text.split())
    return f"Word count: {count}"

@tool
def summarize_numbers(numbers: str) -> str:
    """Compute basic statistics for a comma-separated list of numbers."""
    try:
        nums = [float(n.strip()) for n in numbers.split(",")]
        total = sum(nums)
        mean = total / len(nums)
        return (
            f"Count: {len(nums)}, Sum: {total:.2f}, "
            f"Mean: {mean:.2f}, Min: {min(nums):.2f}, Max: {max(nums):.2f}"
        )
    except ValueError:
        return "Error: provide comma-separated numbers."

# ---------------------------------------------------------------
# 2. Bundle tools into a Skill
# ---------------------------------------------------------------
data_skill = Skill(
    name="data_analysis",
    description="Tools for analyzing text and numerical data.",
    instructions="Use these tools when the user asks about data analysis.",
    tools=(count_words, summarize_numbers),
    activation="always",  # available from the start
)

# ---------------------------------------------------------------
# 3. Create an on-demand skill (loaded only when needed)
# ---------------------------------------------------------------
@tool
def translate(text: str, target_language: str) -> str:
    """Translate text to a target language (demo: just wraps the text)."""
    return f"[{target_language}] {text}"

translation_skill = Skill(
    name="translation",
    description="Translate text between languages.",
    instructions="Use when the user asks for translation.",
    tools=(translate,),
    activation="on_demand",  # must be activated by the agent
)

# ---------------------------------------------------------------
# 4. Register skills in SkillRegistry
# ---------------------------------------------------------------
registry = SkillRegistry()
registry.register(data_skill)
registry.register(translation_skill)

print(f"Data analysis active: {registry.is_active('data_analysis')}")
print(f"Translation active:   {registry.is_active('translation')}")

# Active tools = tools from always-loaded skills
active = registry.active_tools()
print(f"Active tools: {[t.name for t in active]}")

# Discovery prompt for on-demand skills
prompt = registry.skill_discovery_prompt()
print(f"\nDiscovery prompt:\n{prompt}")

# ---------------------------------------------------------------
# 5. Activate the on-demand skill
# ---------------------------------------------------------------
registry.activate("translation")
print(f"\nTranslation active: {registry.is_active('translation')}")
active = registry.active_tools()
print(f"Active tools: {[t.name for t in active]}")
```

## Built-in Tools (No API Key Needed)

```python
import math

from anchor import (
    MemoryManager,
    DenseRetriever,
    InMemoryContextStore,
    InMemoryEntryStore,
    InMemoryVectorStore,
    ContextItem,
    SourceType,
    memory_tools,
    rag_tools,
)

# ---------------------------------------------------------------
# 1. memory_tools: CRUD for persistent facts
# ---------------------------------------------------------------
memory = MemoryManager(
    conversation_tokens=2048,
    persistent_store=InMemoryEntryStore(),
)

mem_tools = memory_tools(memory)
print("Memory tools:")
for t in mem_tools:
    print(f"  {t.name}: {t.description[:60]}...")

# Use the tools directly
save_tool = next(t for t in mem_tools if t.name == "save_fact")
result = save_tool.fn(fact="User prefers Python over JavaScript")
print(f"\nSave result: {result}")

search_tool = next(t for t in mem_tools if t.name == "search_facts")
result = search_tool.fn(query="programming language preference")
print(f"Search result: {result}")

# ---------------------------------------------------------------
# 2. rag_tools: search documentation
# ---------------------------------------------------------------
def embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw

retriever = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn,
)
retriever.index([
    ContextItem(
        content="anchor supports hybrid retrieval with RRF.",
        source=SourceType.RETRIEVAL,
    ),
    ContextItem(
        content="Use ContextPipeline to assemble context from multiple sources.",
        source=SourceType.RETRIEVAL,
    ),
])

search_tools = rag_tools(retriever, embed_fn=embed_fn)
print("\nRAG tools:")
for t in search_tools:
    print(f"  {t.name}: {t.description[:60]}...")

# Use the tool directly
search = search_tools[0]
result = search.fn(query="How does hybrid retrieval work?")
print(f"\nSearch result:\n{result}")
```

## Full Agent Setup (Requires API Key)

```python
import os

from anchor import (
    Agent,
    MemoryManager,
    InMemoryEntryStore,
    memory_tools,
    Skill,
    tool,
)

# Only run this section if you have an API key
if os.environ.get("ANTHROPIC_API_KEY"):
    # Create memory
    memory = MemoryManager(
        conversation_tokens=4096,
        persistent_store=InMemoryEntryStore(),
    )

    # Create custom tools
    @tool
    def current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    # Build the agent
    agent = (
        Agent(model="claude-haiku-4-5-20251001")
        .with_system_prompt(
            "You are a helpful assistant. Remember important user facts."
        )
        .with_memory(memory)
        .with_tools(memory_tools(memory))
        .with_tools([current_time])
    )

    # Stream a response
    for chunk in agent.chat("Hello! My name is Alice and I love hiking."):
        print(chunk, end="", flush=True)
    print()

    # Memory is automatically updated
    turns = memory.conversation.turns
    print(f"\nConversation turns: {len(turns)}")

    # Access the pipeline diagnostics
    if agent.last_result:
        print(f"Token utilization: "
              f"{agent.last_result.diagnostics.get('token_utilization', 0):.1%}")
else:
    print("Set ANTHROPIC_API_KEY to run the Agent example.")
```

## Key Concepts

### Tool Creation Tiers

| Tier | Method | Schema Source |
|------|--------|--------------|
| 1 | `@tool` decorator | Auto-generated from type hints |
| 2 | `@tool(input_model=MyModel)` | Explicit Pydantic model |
| 3 | `AgentTool(...)` constructor | Manual JSON Schema dict |

### Skill Activation Modes

| Mode | Behavior |
|------|----------|
| `"always"` | Tools available from the first API round |
| `"on_demand"` | Advertised in discovery prompt, loaded when activated |

!!! tip "Tool Function Requirements"
    Tool functions must return `str`. The `@tool` decorator extracts the
    function name, docstring, and type hints to auto-generate the JSON
    Schema used by the LLM.

!!! note "Multi-Provider Agent"
    The `Agent` class supports any LLM provider via the `LLMProvider`
    protocol. Use `Agent(model="openai/gpt-4o")` or any other supported
    provider. Tools are converted to a provider-agnostic `ToolSchema` via
    `to_tool_schema()`. See the [LLM Providers Guide](../guides/llm-providers.md)
    for details.

## Next Steps

- [Chatbot with Memory](chatbot-with-memory.md) -- memory management in depth
- [RAG Pipeline](rag-pipeline.md) -- add retrieval to your agent
- [Custom Retriever](custom-retriever.md) -- implement custom data sources
