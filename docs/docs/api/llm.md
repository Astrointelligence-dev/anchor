# LLM API Reference

The `anchor.llm` module provides a unified interface for multiple LLM
providers. It defines the `LLMProvider` protocol, a `BaseLLMProvider` base
class with retry logic, a provider registry with automatic routing, and
structured data models for messages, responses, and errors.

All classes are importable from `anchor.llm`:

```python
from anchor.llm import (
    LLMProvider, BaseLLMProvider,
    create_provider, register_provider,
    FallbackProvider,
    Message, Role, LLMResponse, StreamChunk, Usage, StopReason,
    ContentBlock, ToolCall, ToolCallDelta, ToolResult, ToolSchema,
    ProviderError, AuthenticationError, RateLimitError, ServerError,
    TimeoutError, ModelNotFoundError, ContentFilterError,
    ProviderNotInstalledError,
    MODEL_PRICING, calculate_cost,
)
```

---

## LLMProvider (Protocol)

A runtime-checkable protocol that all LLM providers must satisfy. Use this
as the type hint when accepting any provider.

```python
@runtime_checkable
class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def provider_name(self) -> str: ...
    def invoke(self, messages, *, tools=None, max_tokens=None, temperature=None, stop=None, **kwargs) -> LLMResponse: ...
    def stream(self, messages, *, tools=None, max_tokens=None, temperature=None, stop=None, **kwargs) -> Iterator[StreamChunk]: ...
    async def ainvoke(self, messages, *, tools=None, max_tokens=None, temperature=None, stop=None, **kwargs) -> LLMResponse: ...
    def astream(self, messages, *, tools=None, max_tokens=None, temperature=None, stop=None, **kwargs) -> AsyncIterator[StreamChunk]: ...
```

### Properties

| Property | Type | Description |
|---|---|---|
| `model_id` | `str` | Fully qualified model identifier (e.g. `"openai/gpt-4o"`) |
| `provider_name` | `str` | Provider name (e.g. `"anthropic"`, `"openai"`) |

### Methods

#### invoke

```python
def invoke(
    self,
    messages: list[Message],
    *,
    tools: list[ToolSchema] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> LLMResponse
```

Send messages and return a complete response synchronously.

#### stream

```python
def stream(
    self,
    messages: list[Message],
    *,
    tools: list[ToolSchema] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[StreamChunk]
```

Send messages and yield response chunks as they arrive.

#### ainvoke

```python
async def ainvoke(
    self,
    messages: list[Message],
    *,
    tools: list[ToolSchema] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> LLMResponse
```

Async variant of `invoke()`.

#### astream

```python
def astream(
    self,
    messages: list[Message],
    *,
    tools: list[ToolSchema] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]
```

Async variant of `stream()`.

**Common Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | required | Conversation messages |
| `tools` | `list[ToolSchema] \| None` | `None` | Tool definitions available to the model |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `stop` | `list[str] \| None` | `None` | Stop sequences |

---

## BaseLLMProvider

Abstract base class that implements retry logic, error mapping, and the
`LLMProvider` protocol. Subclass this to add a new provider.

### Constructor

```python
class BaseLLMProvider(ABC):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model name within the provider |
| `api_key` | `str \| None` | `None` | API key (resolved by `_resolve_api_key` if not set) |
| `base_url` | `str \| None` | `None` | Override the provider's default base URL |
| `max_retries` | `int` | `2` | Number of retries on transient errors |
| `timeout` | `float` | `60.0` | Request timeout in seconds |

### Properties

| Property | Type | Description |
|---|---|---|
| `model_id` | `str` | Returns `"{provider_name}/{model}"` |
| `provider_name` | `str` | Set by each subclass (e.g. `"anthropic"`) |

### Public Methods

The public methods `invoke`, `stream`, `ainvoke`, and `astream` wrap the
abstract `_do_*` methods with retry logic for transient errors.

### Abstract Methods (for subclasses)

| Method | Description |
|---|---|
| `_resolve_api_key() -> str` | Resolve the API key from environment or config |
| `_do_invoke(messages, **kwargs) -> LLMResponse` | Provider-specific invoke |
| `_do_stream(messages, **kwargs) -> Iterator[StreamChunk]` | Provider-specific stream |
| `_do_ainvoke(messages, **kwargs) -> LLMResponse` | Provider-specific async invoke |
| `_do_astream(messages, **kwargs) -> AsyncIterator[StreamChunk]` | Provider-specific async stream |

---

## create_provider

Factory function that creates a provider from a `"provider/model"` string.

### Signature

```python
def create_provider(
    model: str,
    *,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    **kwargs: Any,
) -> LLMProvider
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Provider-prefixed model string (e.g. `"openai/gpt-4o"`) |
| `api_key` | `str \| None` | `None` | API key passed to the provider |
| `fallbacks` | `list[str] \| None` | `None` | Fallback model strings; wraps result in `FallbackProvider` |

A model string without a prefix defaults to `"anthropic"` for backward
compatibility.

### Example

```python
from anchor.llm import create_provider

# Single provider
provider = create_provider("openai/gpt-4o")
response = provider.invoke(messages)

# With fallbacks
provider = create_provider(
    "anthropic/claude-sonnet-4-20250514",
    fallbacks=["openai/gpt-4o"],
)
```

---

## register_provider

Register a custom provider class in the global registry.

### Signature

```python
def register_provider(name: str, cls: type[BaseLLMProvider]) -> None
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Prefix used in model strings (e.g. `"mycloud"`) |
| `cls` | `type[BaseLLMProvider]` | Provider class to instantiate |

### Example

```python
from anchor.llm import register_provider, BaseLLMProvider

class MyProvider(BaseLLMProvider):
    provider_name = "mycloud"
    ...

register_provider("mycloud", MyProvider)
provider = create_provider("mycloud/my-model")
```

---

## FallbackProvider

Wraps a primary provider with one or more fallback providers. On transient
errors, automatically retries with the next provider in the chain.

### Constructor

```python
class FallbackProvider:
    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
    ) -> None
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `primary` | `LLMProvider` | The preferred provider tried first |
| `fallbacks` | `list[LLMProvider]` | Providers tried in order on transient failure |

### Properties

| Property | Type | Description |
|---|---|---|
| `model_id` | `str` | Delegates to the primary provider |
| `provider_name` | `str` | Delegates to the primary provider |

### Fallback Behavior

- **`invoke` / `ainvoke`:** On a transient `ProviderError`, tries each
  fallback in order until one succeeds.
- **`stream` / `astream`:** Fallback is attempted only before the first
  chunk is yielded. Mid-stream errors propagate to the caller.

---

## Supported Providers

| Provider | Model Prefix | SDK Package | Install Extra |
|---|---|---|---|
| Anthropic | `anthropic/` | `anthropic` | `pip install astro-anchor[anthropic]` |
| Claude Code CLI | `claude_cli/` | `claude-agent-sdk` | `pip install astro-anchor[claude-cli]` |
| OpenAI | `openai/` | `openai` | `pip install astro-anchor[openai]` |
| Google Gemini | `gemini/` | `google-genai` | `pip install astro-anchor[gemini]` |
| Grok (xAI) | `grok/` | `openai` | `pip install astro-anchor[openai]` |
| Ollama | `ollama/` | `ollama` | `pip install astro-anchor[ollama]` |
| OpenRouter | `openrouter/` | `openai` | `pip install astro-anchor[openai]` |
| LiteLLM | `litellm/` | `litellm` | `pip install astro-anchor[litellm]` |

---

## Data Models

All models are frozen Pydantic `BaseModel` instances unless otherwise noted.

### Role

```python
class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
```

### Message

```python
class Message(BaseModel):
    role: Role
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    name: str | None = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `Role` | required | Message role |
| `content` | `str \| list[ContentBlock] \| None` | `None` | Text or multimodal content |
| `tool_calls` | `list[ToolCall] \| None` | `None` | Tool calls requested by the assistant |
| `tool_result` | `ToolResult \| None` | `None` | Result of a tool invocation |
| `name` | `str \| None` | `None` | Optional participant name |

### ContentBlock

```python
class ContentBlock(BaseModel):
    type: str
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    media_type: str | None = None
```

### ToolCall

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict
```

### ToolCallDelta

```python
class ToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str | None = None
```

### ToolResult

```python
class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False
```

### ToolSchema

```python
class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict
```

### Usage

```python
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float | None = None
```

### StopReason

```python
class StopReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    model: str | None = None
    provider: str | None = None
    stop_reason: StopReason | None = None
```

### StreamChunk

```python
class StreamChunk(BaseModel):
    content: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: Usage | None = None
    stop_reason: StopReason | None = None
```

---

## Pricing

### MODEL_PRICING

A dictionary mapping model name patterns to per-token costs. Used by
`calculate_cost` to estimate request costs.

### calculate_cost

```python
def calculate_cost(
    model: str,
    usage: Usage,
) -> float | None
```

Returns the estimated cost in USD for the given model and token usage, or
`None` if the model is not in `MODEL_PRICING`.

---

## Errors

All runtime provider errors inherit from `ProviderError`. Each error
carries a `provider` string and an `is_transient` flag that controls
retry behavior.

### ProviderError

```python
class ProviderError(Exception):
    provider: str
    is_transient: bool
```

Base class for all provider errors.

### Error Subclasses

| Error | `is_transient` | Description |
|---|---|---|
| `AuthenticationError` | `False` | Invalid or missing API key |
| `RateLimitError` | `True` | Rate limit exceeded; has optional `retry_after: float \| None` |
| `ServerError` | `True` | Provider returned a 5xx error |
| `TimeoutError` | `True` | Request timed out |
| `ModelNotFoundError` | `False` | Requested model does not exist |
| `ContentFilterError` | `False` | Content blocked by the provider's safety filter |

### ProviderNotInstalledError

```python
class ProviderNotInstalledError(Exception)
```

Raised at setup time when a required SDK package is not installed. This is
not a subclass of `ProviderError` since it is a configuration issue, not a
runtime error.

---

## See Also

- [Agent API Reference](../api/agent.md) -- the Agent that consumes LLM providers
- [Exceptions Reference](../api/exceptions.md) -- other framework exceptions
- [Protocols Reference](../api/protocols.md) -- extension point protocols
