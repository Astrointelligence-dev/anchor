"""MCP data models.

All models use Pydantic v2 with frozen=True, matching anchor conventions.
Field names use snake_case (Pythonic) rather than MCP spec's camelCase.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator


class MCPServerConfig(BaseModel, frozen=True):
    """Configuration for connecting to an external MCP server.

    Accepts both command (STDIO) and url (HTTP) — transport is
    auto-inferred from which field is provided.
    """

    name: str
    """Unique server name. Used for tool name prefixing."""

    command: str | None = None
    """STDIO transport: command to spawn (e.g., 'npx', 'python')."""

    args: list[str] | None = None
    """STDIO transport: command arguments."""

    url: str | None = None
    """HTTP/SSE transport: server URL."""

    env: dict[str, str] | None = None
    """Environment variables passed to the server process."""

    headers: dict[str, str] | None = None
    """HTTP headers for authentication (e.g., Bearer tokens)."""

    cache_tools: bool = True
    """Cache the tool list after first discovery."""

    prefix_tools: bool = True
    """Prefix tool names with server name to prevent collisions."""

    defer_tools: bool = False
    """Register this server's tools with ``defer_loading=True``: their
    schemas stay out of the prompt until the ``search_tools`` meta-tool
    loads them."""

    timeout: float | None = 30.0
    """Timeout in seconds for MCP server operations. None disables timeout."""

    @model_validator(mode="after")
    def _check_transport(self) -> Self:
        if self.command is None and self.url is None:
            msg = "MCPServerConfig requires either 'command' (STDIO) or 'url' (HTTP)"
            raise ValueError(msg)
        if self.command is not None and self.url is not None:
            msg = "MCPServerConfig accepts 'command' or 'url', not both"
            raise ValueError(msg)
        if not self.name:
            msg = "MCPServerConfig 'name' must be non-empty"
            raise ValueError(msg)
        return self


class MCPResource(BaseModel, frozen=True):
    """Descriptor for an MCP resource."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None


class MCPPromptArgument(BaseModel, frozen=True):
    """A single argument for an MCP prompt template."""

    name: str
    description: str | None = None
    required: bool = False


class MCPPrompt(BaseModel, frozen=True):
    """Descriptor for an MCP prompt template."""

    name: str
    description: str | None = None
    arguments: list[MCPPromptArgument] | None = None
