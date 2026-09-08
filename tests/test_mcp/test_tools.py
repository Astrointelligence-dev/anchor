"""Tests for MCP tool <-> AgentTool conversion."""

from __future__ import annotations

import pytest

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPError
from anchor.mcp.tools import mcp_tool_to_agent_tool, parse_server_string


class TestMCPToolToAgentTool:
    """Convert ToolSchema + async caller into AgentTool."""

    def test_basic_conversion(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

        async def caller(name: str, args: dict) -> str:
            return "content"

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=True,
        )

        assert isinstance(tool, AgentTool)
        assert tool.name == "fs_read_file"
        assert tool.description == "Read a file"
        assert "path" in tool.input_schema.get("properties", {})

    def test_no_prefix(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=False,
        )
        assert tool.name == "read_file"

    def test_sync_fn_raises_mcp_error(self) -> None:
        schema = ToolSchema(
            name="test_tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="s",
            prefix=False,
        )
        with pytest.raises(MCPError, match="achat"):
            tool.fn()

    def test_async_caller_and_original_name_attached(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return "content"

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=True,
        )
        assert tool._mcp_async_caller is caller
        assert tool._mcp_original_name == "read_file"

    def test_empty_description(self) -> None:
        schema = ToolSchema(
            name="noop",
            description="",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="s",
            prefix=False,
        )
        assert tool.description == ""


class TestParseServerString:
    """Parse convenience strings into MCPServerConfig."""

    def test_http_url(self) -> None:
        cfg = parse_server_string("http://localhost:8080/mcp")
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.command is None
        assert cfg.name == "localhost"

    def test_https_url(self) -> None:
        cfg = parse_server_string("https://api.example.com/mcp")
        assert cfg.url == "https://api.example.com/mcp"
        assert cfg.name == "api.example.com"

    def test_command_string(self) -> None:
        cfg = parse_server_string("npx -y @mcp/server-filesystem /data")
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@mcp/server-filesystem", "/data"]
        assert cfg.url is None
        assert cfg.name == "npx"

    def test_single_command(self) -> None:
        cfg = parse_server_string("python")
        assert cfg.command == "python"
        assert cfg.args == []
        assert cfg.name == "python"

    def test_defaults_applied(self) -> None:
        cfg = parse_server_string("http://localhost:8080/mcp")
        assert cfg.cache_tools is True
        assert cfg.prefix_tools is True
        assert cfg.timeout == 30.0
        assert cfg.env is None
        assert cfg.headers is None

    def test_empty_string_raises(self) -> None:
        with pytest.raises(MCPError, match="must not be empty"):
            parse_server_string("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(MCPError, match="must not be empty"):
            parse_server_string("   ")
