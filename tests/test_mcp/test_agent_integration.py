"""Tests for Agent MCP integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from anchor.agent.agent import Agent
from anchor.agent.models import AgentTool
from anchor.mcp.errors import MCPError
from anchor.mcp.models import MCPServerConfig


class TestAgentWithMCPServers:
    def test_with_mcp_servers_accepts_strings(self) -> None:
        agent = Agent(llm=MagicMock())
        result = agent.with_mcp_servers([
            "npx -y @mcp/server-filesystem /data",
            "http://localhost:8080/mcp",
        ])
        assert result is agent
        assert len(agent._mcp_configs) == 2

    def test_with_mcp_servers_accepts_configs(self) -> None:
        cfg = MCPServerConfig(name="test", command="echo")
        agent = Agent(llm=MagicMock()).with_mcp_servers([cfg])
        assert len(agent._mcp_configs) == 1
        assert agent._mcp_configs[0] is cfg

    def test_with_mcp_servers_accepts_mixed(self) -> None:
        cfg = MCPServerConfig(name="test", command="echo")
        agent = Agent(llm=MagicMock()).with_mcp_servers([
            cfg,
            "http://localhost:8080",
        ])
        assert len(agent._mcp_configs) == 2

    def test_chat_raises_when_mcp_configured(self) -> None:
        agent = Agent(llm=MagicMock()).with_mcp_servers([
            "http://localhost:8080",
        ])
        with pytest.raises(TypeError, match="achat"):
            list(agent.chat("hello"))

    def test_with_mcp_servers_is_additive(self) -> None:
        agent = Agent(llm=MagicMock())
        agent.with_mcp_servers(["http://a.com"])
        agent.with_mcp_servers(["http://b.com"])
        assert len(agent._mcp_configs) == 2


class TestAgentAsyncMCPExecution:
    @pytest.mark.asyncio
    async def test_aexecute_mcp_tool_uses_async_caller(self) -> None:
        agent = Agent(llm=MagicMock())

        async_caller = AsyncMock(return_value="mcp result")

        def sentinel(**_kwargs: Any) -> str:
            raise MCPError("sync")

        tool = AgentTool(
            name="server_read",
            description="Read",
            input_schema={"type": "object", "properties": {}},
            fn=sentinel,
        )
        object.__setattr__(tool, "_mcp_async_caller", async_caller)
        object.__setattr__(tool, "_mcp_original_name", "read")

        agent._mcp_tools = [tool]
        result = await agent._aexecute_tool("server_read", {"path": "/data"})

        assert result == "mcp result"
        async_caller.assert_called_once_with("read", {"path": "/data"})

    @pytest.mark.asyncio
    async def test_aexecute_regular_tool_uses_sync_fn(self) -> None:
        from anchor.agent.tool_decorator import tool

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        agent = Agent(llm=MagicMock()).with_tools([greet])
        result = await agent._aexecute_tool("greet", {"name": "World"})
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_aexecute_mcp_tool_returns_error_on_failure(self) -> None:
        agent = Agent(llm=MagicMock())

        async_caller = AsyncMock(side_effect=RuntimeError("boom"))

        def sentinel(**_kwargs: Any) -> str:
            raise MCPError("sync")

        tool = AgentTool(
            name="broken",
            description="Broken tool",
            input_schema={"type": "object", "properties": {}},
            fn=sentinel,
        )
        object.__setattr__(tool, "_mcp_async_caller", async_caller)
        object.__setattr__(tool, "_mcp_original_name", "broken")

        agent._mcp_tools = [tool]
        result = await agent._aexecute_tool("broken", {})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_aexecute_unknown_tool_returns_unknown(self) -> None:
        agent = Agent(llm=MagicMock())
        result = await agent._aexecute_tool("nonexistent", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_all_active_tools_includes_mcp(self) -> None:
        agent = Agent(llm=MagicMock())
        mcp_tool = AgentTool(
            name="mcp_tool",
            description="MCP",
            input_schema={"type": "object", "properties": {}},
            fn=lambda: "",
        )
        agent._mcp_tools = [mcp_tool]
        all_tools = agent._all_active_tools()
        assert any(t.name == "mcp_tool" for t in all_tools)

    @pytest.mark.asyncio
    async def test_aexecute_mcp_tool_error_includes_details(self) -> None:
        agent = Agent(llm=MagicMock())
        async_caller = AsyncMock(side_effect=RuntimeError("connection reset"))

        def sentinel(**_kwargs: Any) -> str:
            raise MCPError("sync")

        tool = AgentTool(
            name="broken",
            description="Broken tool",
            input_schema={"type": "object", "properties": {}},
            fn=sentinel,
        )
        object.__setattr__(tool, "_mcp_async_caller", async_caller)
        object.__setattr__(tool, "_mcp_original_name", "broken")
        agent._mcp_tools = [tool]
        result = await agent._aexecute_tool("broken", {})
        assert "connection reset" in result

    @pytest.mark.asyncio
    async def test_aclose_disconnects_pool(self) -> None:
        agent = Agent(llm=MagicMock())
        mock_pool = AsyncMock()
        agent._mcp_pool = mock_pool
        agent._mcp_tools = [MagicMock()]

        await agent.aclose()
        mock_pool.disconnect_all.assert_awaited_once()
        assert agent._mcp_pool is None
        assert agent._mcp_tools == []

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        mock_pool = AsyncMock()
        async with Agent(llm=MagicMock()) as agent:
            agent._mcp_pool = mock_pool
            agent._mcp_tools = [MagicMock()]
        mock_pool.disconnect_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_noop_without_pool(self) -> None:
        agent = Agent(llm=MagicMock())
        await agent.aclose()  # should not raise
