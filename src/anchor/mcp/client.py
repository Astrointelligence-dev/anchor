"""FastMCP-backed client bridge for consuming external MCP servers.

Uses FastMCP's Client class with auto-inferred transport.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any, Self

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPConnectionError, MCPError, MCPToolError
from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)
from anchor.mcp.tools import mcp_tool_to_agent_tool

try:
    from fastmcp.client import Client
except ImportError as exc:
    _import_error = exc

    class Client:  # type: ignore[no-redef]
        """Placeholder when fastmcp is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MCP bridge requires the 'fastmcp' package. "
                "Install with: pip install astro-anchor[mcp]"
            ) from _import_error

logger = logging.getLogger(__name__)


class FastMCPClientBridge:
    """Connects to external MCP servers using FastMCP Client.

    Implements the MCPClient protocol. Adapts MCP tools to
    anchor's AgentTool format for seamless Agent integration.
    """

    __slots__ = ("_cached_tools", "_config", "_fastmcp_client")

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._fastmcp_client: Client | None = None  # type: ignore[type-arg]
        self._cached_tools: list[ToolSchema] | None = None

    def _not_connected_error(self) -> MCPConnectionError:
        """Build a 'not connected' error for guard checks."""
        return MCPConnectionError(
            "Not connected. Call connect() or use async context manager first.",
            server_name=self._config.name,
            transport="unknown",
        )

    async def connect(self) -> None:
        """Connect using auto-inferred transport."""
        try:
            kwargs: dict[str, Any] = {}
            if self._config.timeout is not None:
                kwargs["timeout"] = self._config.timeout

            if self._config.url is not None:
                if self._config.headers:
                    kwargs["headers"] = dict(self._config.headers)
                client = Client(self._config.url, **kwargs)
            elif self._config.command is not None:
                cmd = self._config.command
                if self._config.args:
                    cmd = f"{cmd} {' '.join(shlex.quote(a) for a in self._config.args)}"
                if self._config.env:
                    kwargs["env"] = dict(self._config.env)
                client = Client(cmd, **kwargs)
            else:
                msg = "No command or URL configured"
                raise MCPConnectionError(
                    msg,
                    server_name=self._config.name,
                    transport="unknown",
                )

            self._fastmcp_client = await client.__aenter__()  # type: ignore[no-untyped-call]
        except MCPConnectionError:
            raise
        except Exception as exc:
            msg = f"Failed to connect to MCP server '{self._config.name}': {exc}"
            raise MCPConnectionError(
                msg,
                server_name=self._config.name,
                transport="http" if self._config.url else "stdio",
            ) from exc

    async def disconnect(self) -> None:
        """Close the FastMCP Client connection."""
        if self._fastmcp_client is not None:
            try:
                await self._fastmcp_client.__aexit__(None, None, None)  # type: ignore[no-untyped-call]
            except (OSError, ConnectionError) as exc:
                logger.error(
                    "Error disconnecting from MCP server '%s': %s",
                    self._config.name,
                    exc,
                )
            finally:
                self._fastmcp_client = None
                self._cached_tools = None

    async def list_tools(self) -> list[ToolSchema]:
        """List tools, with optional caching."""
        if self._config.cache_tools and self._cached_tools is not None:
            return self._cached_tools

        if self._fastmcp_client is None:
            raise self._not_connected_error()
        mcp_tools = await self._fastmcp_client.list_tools()
        schemas = [
            ToolSchema(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in mcp_tools
        ]

        if self._config.cache_tools:
            self._cached_tools = schemas

        return schemas

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute tool via FastMCP Client, return result as string."""
        if self._fastmcp_client is None:
            raise self._not_connected_error()
        try:
            result = await self._fastmcp_client.call_tool(name, arguments)
            if hasattr(result, "content") and isinstance(result.content, list):
                texts = [
                    block.text for block in result.content
                    if hasattr(block, "text") and block.text
                ]
                return " ".join(texts) if texts else str(result)
            return str(result)
        except Exception as exc:
            raise MCPToolError(
                f"MCP tool '{name}' failed: {exc}",
                tool_name=name,
                server_name=self._config.name,
                cause=exc,
            ) from exc

    async def list_resources(self) -> list[MCPResource]:
        """List resources from the MCP server."""
        if self._fastmcp_client is None:
            raise self._not_connected_error()
        mcp_resources = await self._fastmcp_client.list_resources()
        return [
            MCPResource(
                uri=str(r.uri),
                name=r.name,
                description=r.description,
                mime_type=r.mimeType if hasattr(r, "mimeType") else None,
            )
            for r in mcp_resources
        ]

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        if self._fastmcp_client is None:
            raise self._not_connected_error()
        result = await self._fastmcp_client.read_resource(uri)
        if isinstance(result, list) and result:
            first = result[0]
            if hasattr(first, "content"):
                return str(first.content)
        return str(result)

    async def list_prompts(self) -> list[MCPPrompt]:
        """List prompt templates from the MCP server."""
        if self._fastmcp_client is None:
            raise self._not_connected_error()
        mcp_prompts = await self._fastmcp_client.list_prompts()
        return [
            MCPPrompt(
                name=p.name,
                description=p.description,
                arguments=[
                    MCPPromptArgument(
                        name=a.name,
                        description=a.description,
                        required=a.required or False,
                    )
                    for a in (p.arguments or [])
                ],
            )
            for p in mcp_prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """Get a rendered prompt."""
        if self._fastmcp_client is None:
            raise self._not_connected_error()
        result = await self._fastmcp_client.get_prompt(name, arguments)
        if hasattr(result, "messages") and result.messages:
            texts = []
            for msg in result.messages:
                if hasattr(msg.content, "text"):
                    texts.append(msg.content.text)
            return " ".join(texts) if texts else str(result)
        return str(result)

    async def as_agent_tools(self) -> list[AgentTool]:
        """Convert all MCP tools to AgentTool instances."""
        schemas = await self.list_tools()
        return [
            mcp_tool_to_agent_tool(
                schema=schema,
                async_caller=self.call_tool,
                server_name=self._config.name,
                prefix=self._config.prefix_tools,
                defer_loading=self._config.defer_tools,
            )
            for schema in schemas
        ]

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()


class MCPClientPool:
    """Manages connections to multiple MCP servers.

    Convenience class for connecting to several servers at once
    and collecting all their tools as AgentTool instances.
    """

    __slots__ = ("_clients", "_configs")

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        self._clients: list[FastMCPClientBridge] = []

    async def connect_all(self) -> None:
        """Connect to all configured servers concurrently.

        If any connection fails, already-connected bridges are cleaned up
        before re-raising the first error.
        """
        bridges = [FastMCPClientBridge(cfg) for cfg in self._configs]
        results = await asyncio.gather(
            *(b.connect() for b in bridges),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # Clean up any successfully connected bridges
            connected = [
                b for b, r in zip(bridges, results, strict=True)
                if not isinstance(r, BaseException)
            ]
            cleanup_results = await asyncio.gather(
                *(b.disconnect() for b in connected),
                return_exceptions=True,
            )
            for r in cleanup_results:
                if isinstance(r, Exception):
                    logger.error("MCP cleanup error during failed connect_all: %s", r)
            raise errors[0]
        self._clients = bridges

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        results = await asyncio.gather(
            *(c.disconnect() for c in self._clients),
            return_exceptions=True,
        )
        for client, result in zip(self._clients, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Error disconnecting MCP server '%s': %s",
                    client._config.name,
                    result,
                )
        self._clients = []

    async def all_agent_tools(self) -> list[AgentTool]:
        """Collect AgentTool instances from all connected servers."""
        tool_lists = await asyncio.gather(
            *(c.as_agent_tools() for c in self._clients),
        )
        return [tool for tools in tool_lists for tool in tools]

    def _bridge(self, server: str) -> FastMCPClientBridge:
        for client in self._clients:
            if client._config.name == server:
                return client
        available = ", ".join(c._config.name for c in self._clients) or "none"
        msg = f"Unknown MCP server: '{server}'. Connected: {available}"
        raise MCPError(msg)

    async def all_prompts(self) -> dict[str, list[MCPPrompt]]:
        """Prompt templates per connected server, keyed by server name."""
        prompt_lists = await asyncio.gather(
            *(c.list_prompts() for c in self._clients),
        )
        return {
            client._config.name: prompts
            for client, prompts in zip(self._clients, prompt_lists, strict=True)
        }

    async def get_prompt(
        self, server: str, name: str, arguments: dict[str, Any] | None = None,
    ) -> str:
        """Render a prompt template from *server* as message text."""
        return await self._bridge(server).get_prompt(name, arguments)

    async def all_resources(self) -> dict[str, list[MCPResource]]:
        """Resources per connected server, keyed by server name."""
        resource_lists = await asyncio.gather(
            *(c.list_resources() for c in self._clients),
        )
        return {
            client._config.name: resources
            for client, resources in zip(
                self._clients, resource_lists, strict=True,
            )
        }

    async def read_resource(self, server: str, uri: str) -> str:
        """Read a resource from *server* by URI."""
        return await self._bridge(server).read_resource(uri)

    async def __aenter__(self) -> Self:
        await self.connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect_all()
