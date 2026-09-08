"""Adapters between MCP tools and anchor AgentTool.

Provides conversion utilities and config string parsing.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import urlparse

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPError
from anchor.mcp.models import MCPServerConfig


def mcp_tool_to_agent_tool(
    *,
    schema: ToolSchema,
    async_caller: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
    server_name: str,
    prefix: bool,
    defer_loading: bool = False,
) -> AgentTool:
    """Convert an MCP ToolSchema into an anchor AgentTool.

    The returned AgentTool's ``fn`` raises MCPError if called
    synchronously — MCP tools require ``achat()`` (async path).
    The ``_mcp_async_caller`` attribute is used by ``Agent._aexecute_tool()``.
    """
    tool_name = f"{server_name}_{schema.name}" if prefix else schema.name
    original_name = schema.name

    def _sync_sentinel(**_kwargs: Any) -> str:
        msg = (
            f"MCP tool '{tool_name}' requires async execution. "
            f"Use agent.achat() instead of agent.chat()."
        )
        raise MCPError(msg)

    tool = AgentTool(
        name=tool_name,
        description=schema.description,
        input_schema=schema.input_schema,
        fn=_sync_sentinel,
        defer_loading=defer_loading,
    )

    # Attach async caller for _aexecute_tool() to use
    object.__setattr__(tool, "_mcp_async_caller", async_caller)
    object.__setattr__(tool, "_mcp_original_name", original_name)

    return tool


def parse_server_string(server: str) -> MCPServerConfig:
    """Parse a convenience string into MCPServerConfig.

    - Strings starting with http:// or https:// -> URL config
    - Other strings -> command + args (split by whitespace)
    """
    stripped = server.strip()
    if not stripped:
        msg = "Server string must not be empty"
        raise MCPError(msg)

    if stripped.startswith(("http://", "https://")):
        parsed = urlparse(stripped)
        name = parsed.hostname or "mcp-server"
        return MCPServerConfig(name=name, url=stripped)

    parts = stripped.split()
    command = parts[0]
    args = parts[1:] if len(parts) > 1 else []
    return MCPServerConfig(name=command, command=command, args=args)
