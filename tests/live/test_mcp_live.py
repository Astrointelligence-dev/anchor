"""Live MCP smoke test — a real fastmcp server over stdio, no mocks.

Needs no credentials: the server is a local subprocess
(tests/live/_mcp_server.py), so this suite runs in CI. It is the only
place the MCP bridge talks to an actual server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from anchor.agent import Agent
from anchor.mcp.client import MCPClientPool
from anchor.mcp.models import MCPServerConfig
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import _tool_results_of

_SERVER = Path(__file__).parent / "_mcp_server.py"


def _config(**overrides: object) -> MCPServerConfig:
    return MCPServerConfig(
        name="live",
        command=sys.executable,
        args=[str(_SERVER)],
        **overrides,  # type: ignore[arg-type]
    )


async def test_pool_against_real_server() -> None:
    async with MCPClientPool([_config()]) as pool:
        tools = await pool.all_agent_tools()
        names = [t.name for t in tools]
        assert "live_add" in names

        prompts = await pool.all_prompts()
        assert any(p.name == "greet" for p in prompts["live"])
        rendered = await pool.get_prompt("live", "greet", {"name": "Arthur"})
        assert "Arthur" in rendered

        resources = await pool.all_resources()
        assert any("readme" in r.uri for r in resources["live"])
        content = await pool.read_resource("live", "data://readme")
        assert "live resource content" in content


async def test_agent_executes_real_mcp_tool() -> None:
    provider = FakeLLMProvider([
        _tool_use_response("tu_1", "live_add", {"a": 17, "b": 25}),
        _text_response("42"),
    ])
    agent = Agent(llm=provider, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    agent.with_mcp_servers([_config()])

    chunks = [c async for c in agent.achat("Add 17 and 25")]
    await agent.aclose()

    assert "".join(chunks) == "42"
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is False
    assert "42" in result.content


async def test_defer_tools_against_real_server() -> None:
    provider = FakeLLMProvider([_text_response("hi")])
    agent = Agent(llm=provider, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    agent.with_mcp_servers([_config(defer_tools=True)])

    [c async for c in agent.achat("Hello")]
    await agent.aclose()

    names = [t.name for t in provider.seen_tools[0] or []]
    # Deferred: the real server's tools stay out of the prompt; only
    # the auto-registered search_tools meta-tool is sendable.
    assert "live_add" not in names
    assert "search_tools" in names
