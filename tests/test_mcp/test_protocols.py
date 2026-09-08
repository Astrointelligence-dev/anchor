"""Tests for MCP protocol conformance."""

from __future__ import annotations

from typing import Any, Self

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.models import MCPPrompt, MCPResource
from anchor.mcp.protocols import MCPClient, MCPServer


class _FakeMCPClient:
    """Minimal class that structurally conforms to MCPClient."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def list_tools(self) -> list[ToolSchema]:
        return []
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return ""
    async def list_resources(self) -> list[MCPResource]:
        return []
    async def read_resource(self, uri: str) -> str:
        return ""
    async def list_prompts(self) -> list[MCPPrompt]:
        return []
    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        return ""
    async def __aenter__(self) -> Self:
        return self
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...


class _FakeMCPServer:
    """Minimal class that structurally conforms to MCPServer."""

    def expose_tool(self, tool: AgentTool) -> None: ...
    def expose_tools(self, tools: list[AgentTool]) -> None: ...
    def expose_resource(self, uri: str, handler: Any) -> None: ...
    def expose_prompt(self, name: str, handler: Any) -> None: ...
    async def run(self, transport: str = "stdio") -> None: ...


class TestMCPClientProtocol:
    def test_fake_client_satisfies_protocol(self) -> None:
        client = _FakeMCPClient()
        assert isinstance(client, MCPClient)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), MCPClient)


class TestMCPServerProtocol:
    def test_fake_server_satisfies_protocol(self) -> None:
        server = _FakeMCPServer()
        assert isinstance(server, MCPServer)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), MCPServer)
