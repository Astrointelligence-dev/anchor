"""MCP bridge gaps: defer_tools wire, name collisions, prompts/resources.

Covers the 2026-08-26 round: MCPServerConfig.defer_tools flows into
AgentTool.defer_loading; MCP tool names that shadow existing tools fail
loudly at connect; pool-level prompt/resource aggregation is app-facing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.agent import Agent
from anchor.llm.models import ToolSchema
from anchor.mcp.client import MCPClientPool
from anchor.mcp.errors import MCPError
from anchor.mcp.models import MCPServerConfig
from anchor.mcp.tools import mcp_tool_to_agent_tool
from tests.test_agent.test_agent import FakeLLMProvider, _Tok
from tests.test_agent.test_phase4_loop import _echo_tool


def _schema(name: str = "lookup") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="looks things up",
        input_schema={"type": "object", "properties": {}},
    )


async def _noop_caller(_name: str, _input: dict[str, Any]) -> str:
    return "ok"


def _mock_fastmcp_client(tools: list[Any] | None = None) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.list_tools.return_value = tools or []
    return client


def _mcp_tool_stub(name: str) -> MagicMock:
    stub = MagicMock()
    stub.name = name
    stub.description = "stub"
    stub.inputSchema = {"type": "object", "properties": {}}
    return stub


# ---------------------------------------------------------------------------
# defer_tools wire
# ---------------------------------------------------------------------------


def test_defer_loading_flows_from_converter():
    tool = mcp_tool_to_agent_tool(
        schema=_schema(),
        async_caller=_noop_caller,
        server_name="srv",
        prefix=True,
        defer_loading=True,
    )
    assert tool.defer_loading is True
    assert tool.name == "srv_lookup"


async def test_defer_tools_config_reaches_agent_tools():
    config = MCPServerConfig(name="srv", command="echo", defer_tools=True)
    mock_client = _mock_fastmcp_client([_mcp_tool_stub("lookup")])

    with patch("anchor.mcp.client.Client", return_value=mock_client):
        pool = MCPClientPool([config])
        await pool.connect_all()
        (tool,) = await pool.all_agent_tools()

    assert tool.defer_loading is True


# ---------------------------------------------------------------------------
# Collision check at connect
# ---------------------------------------------------------------------------


async def test_mcp_name_shadowing_fails_loudly():
    config = MCPServerConfig(
        name="srv", command="echo", prefix_tools=False,
    )
    mock_client = _mock_fastmcp_client([_mcp_tool_stub("echo")])

    agent = Agent(llm=FakeLLMProvider([]), tokenizer=_Tok())
    agent.with_tools([_echo_tool()])  # direct tool named "echo"
    agent.with_mcp_servers([config])

    with (
        patch("anchor.mcp.client.Client", return_value=mock_client),
        pytest.raises(ValueError, match="collision"),
    ):
        await agent._ensure_mcp()
    # Failed connect leaves no half-attached pool.
    assert agent._mcp_pool is None
    assert agent._mcp_tools == []


async def test_two_servers_with_same_tool_name_collide():
    configs = [
        MCPServerConfig(name="a", command="echo", prefix_tools=False),
        MCPServerConfig(name="b", command="echo", prefix_tools=False),
    ]
    agent = Agent(llm=FakeLLMProvider([]), tokenizer=_Tok())
    agent.with_mcp_servers(configs)

    def _client(*_args: object, **_kwargs: object) -> AsyncMock:
        return _mock_fastmcp_client([_mcp_tool_stub("dup")])

    with (
        patch("anchor.mcp.client.Client", side_effect=_client),
        pytest.raises(ValueError, match="collision"),
    ):
        await agent._ensure_mcp()


# ---------------------------------------------------------------------------
# Prompts / resources aggregation (app-facing)
# ---------------------------------------------------------------------------


def _client_with_prompts_and_resources() -> AsyncMock:
    client = _mock_fastmcp_client()
    prompt = MagicMock()
    prompt.name = "review"
    prompt.description = "review a PR"
    prompt.arguments = None
    client.list_prompts.return_value = [prompt]

    message = MagicMock()
    message.content = MagicMock()
    message.content.text = "Please review PR 42"
    prompt_result = MagicMock()
    prompt_result.messages = [message]
    client.get_prompt.return_value = prompt_result

    resource = MagicMock()
    resource.uri = "file://readme"
    resource.name = "readme"
    resource.description = None
    resource.mimeType = "text/plain"
    client.list_resources.return_value = [resource]

    block = MagicMock()
    block.content = "hello resource"
    client.read_resource.return_value = [block]
    return client


async def test_pool_aggregates_prompts_and_resources():
    configs = [MCPServerConfig(name="srv", command="echo")]
    with patch(
        "anchor.mcp.client.Client",
        return_value=_client_with_prompts_and_resources(),
    ):
        pool = MCPClientPool(configs)
        await pool.connect_all()

        prompts = await pool.all_prompts()
        assert list(prompts) == ["srv"]
        assert prompts["srv"][0].name == "review"

        resources = await pool.all_resources()
        assert resources["srv"][0].uri == "file://readme"

        text = await pool.read_resource("srv", "file://readme")
        assert "hello resource" in text


async def test_pool_unknown_server_raises():
    configs = [MCPServerConfig(name="srv", command="echo")]
    with patch(
        "anchor.mcp.client.Client",
        return_value=_client_with_prompts_and_resources(),
    ):
        pool = MCPClientPool(configs)
        await pool.connect_all()
        with pytest.raises(MCPError, match="Unknown MCP server"):
            await pool.read_resource("ghost", "file://x")


async def test_agent_accessors_delegate_to_pool():
    config = MCPServerConfig(name="srv", command="echo")
    agent = Agent(llm=FakeLLMProvider([]), tokenizer=_Tok())
    agent.with_mcp_servers([config])

    with patch(
        "anchor.mcp.client.Client",
        return_value=_client_with_prompts_and_resources(),
    ):
        prompts = await agent.mcp_prompts()
        assert prompts["srv"][0].name == "review"
        resources = await agent.mcp_resources()
        assert resources["srv"][0].name == "readme"
        text = await agent.mcp_read_resource("srv", "file://readme")
        assert "hello resource" in text
    await agent.aclose()


def test_agent_accessor_without_servers_raises():
    agent = Agent(llm=FakeLLMProvider([]), tokenizer=_Tok())
    with pytest.raises(RuntimeError, match="No connected MCP servers"):
        agent._require_mcp_pool()
