"""MCP Bridge — bidirectional Model Context Protocol integration.

Exposes anchor capabilities as MCP tools/resources/prompts and
consumes external MCP servers from anchor's Agent.
"""

from __future__ import annotations

from anchor.mcp.client import FastMCPClientBridge, MCPClientPool
from anchor.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)
from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)
from anchor.mcp.protocols import MCPClient, MCPServer
from anchor.mcp.server import FastMCPServerBridge
from anchor.mcp.tools import mcp_tool_to_agent_tool, parse_server_string

__all__ = [
    "FastMCPClientBridge",
    "FastMCPServerBridge",
    "MCPClient",
    "MCPClientPool",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPResource",
    "MCPServer",
    "MCPServerConfig",
    "MCPTimeoutError",
    "MCPToolError",
    "mcp_tool_to_agent_tool",
    "parse_server_string",
]
