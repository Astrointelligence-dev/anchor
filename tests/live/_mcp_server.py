"""Minimal real fastmcp server for the live MCP smoke test (stdio)."""

from fastmcp import FastMCP

mcp = FastMCP("live")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.prompt
def greet(name: str) -> str:
    """Build a greeting prompt."""
    return f"Say hello to {name}"


@mcp.resource("data://readme")
def readme() -> str:
    """A tiny resource."""
    return "live resource content"


if __name__ == "__main__":
    mcp.run()
