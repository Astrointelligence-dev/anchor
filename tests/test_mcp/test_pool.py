"""Tests for MCPClientPool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.mcp.client import MCPClientPool
from anchor.mcp.models import MCPServerConfig


@pytest.fixture
def two_configs() -> list[MCPServerConfig]:
    return [
        MCPServerConfig(name="fs", command="echo", args=["fs"]),
        MCPServerConfig(name="db", command="echo", args=["db"]),
    ]


class TestMCPClientPool:
    async def test_connect_all(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            assert len(pool._clients) == 2

    async def test_disconnect_all(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            await pool.disconnect_all()
            assert len(pool._clients) == 0

    async def test_all_agent_tools_collects_from_all(
        self, two_configs: list[MCPServerConfig],
    ) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "A tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            tools = await pool.all_agent_tools()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "fs_tool1" in names
        assert "db_tool1" in names

    async def test_context_manager(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            async with MCPClientPool(two_configs) as pool:
                assert len(pool._clients) == 2
            assert len(pool._clients) == 0

    async def test_connect_all_partial_failure_cleans_up(
        self, two_configs: list[MCPServerConfig],
    ) -> None:
        """If one server fails to connect, successfully connected ones are cleaned up."""
        call_count = 0

        async def mock_connect(bridge: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("connection refused")

        with patch(
            "anchor.mcp.client.FastMCPClientBridge.connect",
            side_effect=mock_connect,
            autospec=True,
        ), patch(
            "anchor.mcp.client.FastMCPClientBridge.disconnect",
            new_callable=AsyncMock,
        ) as mock_disconnect:
            pool = MCPClientPool(two_configs)
            with pytest.raises(OSError, match="connection refused"):
                await pool.connect_all()
            # The successfully connected bridge should have been cleaned up
            assert mock_disconnect.call_count == 1
