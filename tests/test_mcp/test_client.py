"""Tests for FastMCPClientBridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.llm.models import ToolSchema
from anchor.mcp.client import FastMCPClientBridge
from anchor.mcp.errors import MCPConnectionError, MCPToolError
from anchor.mcp.models import MCPServerConfig
from anchor.mcp.protocols import MCPClient


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    return MCPServerConfig(name="test", command="echo", args=["hello"])


@pytest.fixture
def http_config() -> MCPServerConfig:
    return MCPServerConfig(name="api", url="http://localhost:8080/mcp")


class TestFastMCPClientBridge:
    async def test_init(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        assert bridge._config is stdio_config
        assert bridge._fastmcp_client is None

    async def test_connect_creates_client_for_stdio(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            assert bridge._fastmcp_client is not None

    async def test_connect_creates_client_for_http(
        self, http_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(http_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            assert bridge._fastmcp_client is not None

    async def test_disconnect(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.disconnect()
            assert bridge._fastmcp_client is None

    async def test_list_tools_returns_tool_schemas(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "read_file"
        mock_mcp_tool.description = "Read a file"
        mock_mcp_tool.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}
        mock_client.list_tools.return_value = [mock_mcp_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            tools = await bridge.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "read_file"
        assert isinstance(tools[0], ToolSchema)

    async def test_list_tools_caching(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.list_tools()
            await bridge.list_tools()

        assert mock_client.list_tools.call_count == 1

    async def test_call_tool(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="file content")]
        mock_client.call_tool.return_value = mock_result

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            result = await bridge.call_tool("read_file", {"path": "/data/x"})

        assert result == "file content"

    async def test_as_agent_tools(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "greet"
        mock_mcp_tool.description = "Greet someone"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_client.list_tools.return_value = [mock_mcp_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            tools = await bridge.as_agent_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_greet"

    async def test_satisfies_protocol(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        assert isinstance(bridge, MCPClient)

    async def test_connect_failure_raises_connection_error(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with (
            patch("anchor.mcp.client.Client", side_effect=OSError("refused")),
            pytest.raises(MCPConnectionError, match="refused"),
        ):
            await bridge.connect()

    async def test_call_tool_failure_raises_tool_error(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.call_tool.side_effect = RuntimeError("tool broke")

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            with pytest.raises(MCPToolError, match="tool broke"):
                await bridge.call_tool("bad_tool", {})

    async def test_list_tools_no_caching(self, stdio_config: MCPServerConfig) -> None:
        no_cache_config = MCPServerConfig(
            name="test", command="echo", args=["hello"], cache_tools=False,
        )
        bridge = FastMCPClientBridge(no_cache_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.list_tools()
            await bridge.list_tools()

        assert mock_client.list_tools.call_count == 2

    async def test_list_resources(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resource = MagicMock()
        mock_resource.uri = "file:///data/test"
        mock_resource.name = "test"
        mock_resource.description = "A test resource"
        mock_resource.mimeType = "text/plain"
        mock_client.list_resources.return_value = [mock_resource]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            resources = await bridge.list_resources()

        assert len(resources) == 1
        assert resources[0].uri == "file:///data/test"

    async def test_list_prompts(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_prompt = MagicMock()
        mock_prompt.name = "analyze"
        mock_prompt.description = "Analyze code"
        mock_prompt.arguments = []
        mock_client.list_prompts.return_value = [mock_prompt]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            prompts = await bridge.list_prompts()

        assert len(prompts) == 1
        assert prompts[0].name == "analyze"

    async def test_context_manager(self, stdio_config: MCPServerConfig) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            async with FastMCPClientBridge(stdio_config) as bridge:
                assert bridge._fastmcp_client is not None
            assert bridge._fastmcp_client is None

    async def test_list_tools_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.list_tools()

    async def test_call_tool_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.call_tool("test", {})

    async def test_list_resources_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.list_resources()

    async def test_read_resource_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.read_resource("file:///test")

    async def test_list_prompts_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.list_prompts()

    async def test_get_prompt_not_connected_raises(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await bridge.get_prompt("test", {})
