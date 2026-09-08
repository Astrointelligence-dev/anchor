"""Tests for FastMCPServerBridge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastmcp", reason="fastmcp required for MCP server tests")

from anchor.agent.tool_decorator import tool
from anchor.mcp.protocols import MCPServer
from anchor.mcp.server import FastMCPServerBridge


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello {name}"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


class TestFastMCPServerBridge:
    def test_satisfies_protocol(self) -> None:
        server = FastMCPServerBridge("test")
        assert isinstance(server, MCPServer)

    def test_expose_tool(self) -> None:
        server = FastMCPServerBridge("test")
        server.expose_tool(greet)
        assert len(server._registered_tools) == 1

    def test_expose_tools(self) -> None:
        server = FastMCPServerBridge("test")
        server.expose_tools([greet, add])
        assert len(server._registered_tools) == 2

    def test_expose_resource(self) -> None:
        server = FastMCPServerBridge("test")

        def handler() -> str:
            return "resource data"

        server.expose_resource("data://config", handler)
        assert len(server._registered_resources) == 1

    def test_expose_prompt(self) -> None:
        server = FastMCPServerBridge("test")

        def handler(topic: str) -> str:
            return f"Analyze {topic}"

        server.expose_prompt("analyze", handler)
        assert len(server._registered_prompts) == 1

    def test_from_agent(self) -> None:
        from anchor.agent.agent import Agent

        mock_llm = MagicMock()
        mock_llm.model_id = "mock/test"
        mock_llm.provider_name = "mock"
        agent = Agent(llm=mock_llm).with_tools([greet, add])
        server = FastMCPServerBridge.from_agent(agent)
        assert isinstance(server, FastMCPServerBridge)
        assert len(server._registered_tools) >= 2
        assert "context://pipeline" in server._registered_resources
        assert "context://memory" in server._registered_resources
        assert "chat" in server._registered_prompts

    def test_from_pipeline(self) -> None:
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.formatted_output = "pipeline output"
        mock_pipeline.build.return_value = mock_result

        server = FastMCPServerBridge.from_pipeline(mock_pipeline)
        assert isinstance(server, FastMCPServerBridge)
        assert "query" in server._registered_tools
        assert "context://result" in server._registered_resources

    @pytest.mark.asyncio
    async def test_run_unknown_transport_raises(self) -> None:
        server = FastMCPServerBridge("test")
        with pytest.raises(ValueError, match="Unknown transport"):
            await server.run(transport="invalid")
