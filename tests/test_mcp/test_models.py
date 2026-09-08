"""Tests for MCP data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)


class TestMCPServerConfig:
    """MCPServerConfig validation and behavior."""

    def test_stdio_config(self) -> None:
        cfg = MCPServerConfig(name="fs", command="npx", args=["-y", "server"])
        assert cfg.name == "fs"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "server"]
        assert cfg.url is None
        assert cfg.cache_tools is True
        assert cfg.prefix_tools is True
        assert cfg.timeout == 30.0

    def test_http_config(self) -> None:
        cfg = MCPServerConfig(name="api", url="http://localhost:8080/mcp")
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.command is None

    def test_requires_command_or_url(self) -> None:
        with pytest.raises(ValidationError, match=r"command.*STDIO.*url.*HTTP"):
            MCPServerConfig(name="bad")

    def test_frozen(self) -> None:
        cfg = MCPServerConfig(name="fs", command="npx")
        with pytest.raises(ValidationError):
            cfg.name = "other"  # type: ignore[misc]

    def test_with_headers(self) -> None:
        cfg = MCPServerConfig(
            name="api",
            url="http://example.com",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_with_env(self) -> None:
        cfg = MCPServerConfig(
            name="db",
            command="python",
            args=["server.py"],
            env={"DB_URL": "postgres://localhost"},
        )
        assert cfg.env == {"DB_URL": "postgres://localhost"}

    def test_custom_timeout(self) -> None:
        cfg = MCPServerConfig(name="slow", url="http://slow.example.com", timeout=120.0)
        assert cfg.timeout == 120.0

    def test_rejects_both_command_and_url(self) -> None:
        with pytest.raises(ValidationError, match=r"command.*or.*url.*not both"):
            MCPServerConfig(name="bad", command="echo", url="http://example.com")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match=r"name.*non-empty"):
            MCPServerConfig(name="", command="echo")


class TestMCPResource:
    """MCPResource model."""

    def test_basic(self) -> None:
        r = MCPResource(uri="file:///tmp/data.txt", name="data")
        assert r.uri == "file:///tmp/data.txt"
        assert r.name == "data"
        assert r.description is None
        assert r.mime_type is None

    def test_with_metadata(self) -> None:
        r = MCPResource(
            uri="db://users",
            name="users",
            description="User table",
            mime_type="application/json",
        )
        assert r.description == "User table"
        assert r.mime_type == "application/json"

    def test_frozen(self) -> None:
        r = MCPResource(uri="x", name="y")
        with pytest.raises(ValidationError):
            r.uri = "z"  # type: ignore[misc]


class TestMCPPrompt:
    """MCPPrompt and MCPPromptArgument models."""

    def test_basic_prompt(self) -> None:
        p = MCPPrompt(name="analyze")
        assert p.name == "analyze"
        assert p.description is None
        assert p.arguments is None

    def test_prompt_with_arguments(self) -> None:
        p = MCPPrompt(
            name="summarize",
            description="Summarize content",
            arguments=[
                MCPPromptArgument(name="text", required=True),
                MCPPromptArgument(name="max_length", description="Max words"),
            ],
        )
        assert len(p.arguments) == 2
        assert p.arguments[0].required is True
        assert p.arguments[1].required is False

    def test_frozen(self) -> None:
        p = MCPPrompt(name="x")
        with pytest.raises(ValidationError):
            p.name = "y"  # type: ignore[misc]
