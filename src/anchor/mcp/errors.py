"""MCP error hierarchy.

All errors inherit from AstroContextError for consistent
exception handling across the anchor toolkit.
"""

from __future__ import annotations

from anchor.exceptions import AstroContextError


class MCPError(AstroContextError):
    """Base error for all MCP operations."""


class MCPConnectionError(MCPError):
    """Failed to connect to an MCP server."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str,
        transport: str = "",
    ) -> None:
        full_msg = f"[{server_name}] {message}" if server_name else message
        super().__init__(full_msg)
        self.server_name = server_name
        self.transport = transport


class MCPToolError(MCPError):
    """Error executing an MCP tool."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        server_name: str,
        cause: BaseException | None = None,
    ) -> None:
        full_msg = f"[{server_name}/{tool_name}] {message}"
        super().__init__(full_msg)
        self.tool_name = tool_name
        self.server_name = server_name
        self.cause = cause


class MCPTimeoutError(MCPError):
    """MCP server operation timed out."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str = "",
        operation: str = "",
    ) -> None:
        super().__init__(message)
        self.server_name = server_name
        self.operation = operation


class MCPConfigError(MCPError):
    """Invalid MCP server configuration."""
