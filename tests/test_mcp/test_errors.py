"""Tests for MCP error hierarchy."""

from __future__ import annotations

from anchor.exceptions import AstroContextError
from anchor.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)


class TestMCPErrorHierarchy:
    """All MCP errors must inherit from AstroContextError."""

    def test_mcp_error_is_astro_context_error(self) -> None:
        err = MCPError("test")
        assert isinstance(err, AstroContextError)

    def test_mcp_connection_error_is_mcp_error(self) -> None:
        err = MCPConnectionError("failed", server_name="fs", transport="stdio")
        assert isinstance(err, MCPError)
        assert err.server_name == "fs"
        assert err.transport == "stdio"

    def test_mcp_tool_error_is_mcp_error(self) -> None:
        cause = RuntimeError("boom")
        err = MCPToolError("failed", tool_name="read", server_name="fs", cause=cause)
        assert isinstance(err, MCPError)
        assert err.tool_name == "read"
        assert err.server_name == "fs"
        assert err.cause is cause

    def test_mcp_timeout_error_is_mcp_error(self) -> None:
        err = MCPTimeoutError("timed out", server_name="fs", operation="list_tools")
        assert isinstance(err, MCPError)
        assert err.server_name == "fs"
        assert err.operation == "list_tools"

    def test_mcp_timeout_error_defaults(self) -> None:
        err = MCPTimeoutError("timed out")
        assert err.server_name == ""
        assert err.operation == ""

    def test_mcp_config_error_is_mcp_error(self) -> None:
        err = MCPConfigError("bad config")
        assert isinstance(err, MCPError)

    def test_all_errors_catchable_as_base(self) -> None:
        errors = [
            MCPConnectionError("x", server_name="s", transport="t"),
            MCPToolError("x", tool_name="t", server_name="s"),
            MCPTimeoutError("x"),
            MCPConfigError("x"),
        ]
        for err in errors:
            try:
                raise err
            except MCPError:
                pass  # All should be caught

    def test_connection_error_includes_server_name_in_message(self) -> None:
        err = MCPConnectionError("refused", server_name="myserver", transport="stdio")
        assert "myserver" in str(err)

    def test_tool_error_includes_server_and_tool_in_message(self) -> None:
        err = MCPToolError("failed", tool_name="read", server_name="myserver")
        assert "myserver" in str(err)
        assert "read" in str(err)
