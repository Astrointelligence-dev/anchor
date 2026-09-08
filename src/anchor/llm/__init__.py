"""anchor.llm: Unified multi-provider LLM interface.

Base / Protocol:
    LLMProvider, BaseLLMProvider

Registry:
    create_provider, register_provider

Fallback:
    FallbackProvider

Models:
    Role, ContentBlock, ToolCall, ToolCallDelta, ToolResult,
    Message, Usage, StopReason, LLMResponse, StreamChunk, ToolSchema

Errors:
    ProviderError, RateLimitError, ServerError, LLMTimeoutError,
    AuthenticationError, ModelNotFoundError, ContentFilterError,
    ProviderNotInstalledError

Pricing:
    MODEL_PRICING, calculate_cost
"""

from anchor.llm.base import BaseLLMProvider, LLMProvider
from anchor.llm.errors import (
    AuthenticationError,
    ContentFilterError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    ServerError,
    LLMTimeoutError,
)
from anchor.llm.fallback import FallbackProvider
from anchor.llm.models import (
    ContentBlock,
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    ToolSchema,
    Usage,
)
from anchor.llm.pricing import MODEL_PRICING, calculate_cost
from anchor.llm.registry import create_provider, register_provider

__all__ = [
    # Base / Protocol
    "BaseLLMProvider",
    "LLMProvider",
    # Registry
    "create_provider",
    "register_provider",
    # Fallback
    "FallbackProvider",
    # Models
    "ContentBlock",
    "LLMResponse",
    "Message",
    "Role",
    "StopReason",
    "StreamChunk",
    "ToolCall",
    "ToolCallDelta",
    "ToolResult",
    "ToolSchema",
    "Usage",
    # Errors
    "AuthenticationError",
    "ContentFilterError",
    "ModelNotFoundError",
    "ProviderError",
    "ProviderNotInstalledError",
    "RateLimitError",
    "ServerError",
    "LLMTimeoutError",
    # Pricing
    "MODEL_PRICING",
    "calculate_cost",
]
