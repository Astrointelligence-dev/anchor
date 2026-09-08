"""MCP protocol definitions (PEP 544).

Any object with matching methods can be used — no inheritance required.
Follows anchor's established pattern from protocols/retriever.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, Self, runtime_checkable

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.models import MCPPrompt, MCPResource


@runtime_checkable
class MCPClient(Protocol):
    """Protocol for connecting to and consuming external MCP servers."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def list_tools(self) -> list[ToolSchema]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def list_resources(self) -> list[MCPResource]: ...

    async def read_resource(self, uri: str) -> str: ...

    async def list_prompts(self) -> list[MCPPrompt]: ...

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...


@runtime_checkable
class MCPServer(Protocol):
    """Protocol for exposing anchor capabilities as an MCP server."""

    def expose_tool(self, tool: AgentTool) -> None: ...

    def expose_tools(self, tools: list[AgentTool]) -> None: ...

    def expose_resource(
        self, uri: str, handler: Callable[..., Any],
    ) -> None: ...

    def expose_prompt(
        self, name: str, handler: Callable[..., Any],
    ) -> None: ...

    async def run(
        self, transport: Literal["stdio", "sse", "http"] = "stdio",
    ) -> None: ...
