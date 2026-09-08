"""FastMCP-backed server bridge for exposing anchor as MCP server."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anchor.agent.models import AgentTool

try:
    from fastmcp import FastMCP
except ImportError as exc:
    _import_error = exc

    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MCP bridge requires the 'fastmcp' package. "
                "Install with: pip install astro-anchor[mcp]"
            ) from _import_error

if TYPE_CHECKING:
    from anchor.agent.agent import Agent
    from anchor.pipeline.pipeline import ContextPipeline

logger = logging.getLogger(__name__)


class FastMCPServerBridge:
    """Exposes anchor capabilities as an MCP server via FastMCP."""

    __slots__ = (
        "_mcp",
        "_registered_prompts",
        "_registered_resources",
        "_registered_tools",
    )

    def __init__(self, name: str = "anchor") -> None:
        self._mcp = FastMCP(name)
        self._registered_tools: list[str] = []
        self._registered_resources: list[str] = []
        self._registered_prompts: list[str] = []

    def expose_tool(self, tool: AgentTool) -> None:
        """Register an AgentTool as an MCP tool."""
        fn = tool.fn
        self._mcp.tool(name=tool.name, description=tool.description)(fn)
        self._registered_tools.append(tool.name)

    def expose_tools(self, tools: list[AgentTool]) -> None:
        """Register multiple tools."""
        for t in tools:
            self.expose_tool(t)

    def expose_resource(
        self,
        uri: str,
        handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a resource at the given URI pattern."""
        self._mcp.resource(uri)(handler)
        self._registered_resources.append(uri)

    def expose_prompt(
        self,
        name: str,
        handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a prompt template."""
        self._mcp.prompt(name=name)(handler)
        self._registered_prompts.append(name)

    @classmethod
    def from_agent(cls, agent: Agent) -> FastMCPServerBridge:
        """Create server from an Agent, exposing all its tools.

        Also exposes:
        - resource ``context://pipeline`` -> pipeline config
        - resource ``context://memory`` -> memory state
        - prompt ``chat`` -> send a message through the agent
        """
        server = cls(name="anchor-agent")
        server.expose_tools(agent._tools)

        def _pipeline_resource() -> str:
            """Return the agent's pipeline configuration."""
            return str(getattr(agent, "_pipeline", "No pipeline configured"))

        server.expose_resource("context://pipeline", _pipeline_resource)

        def _memory_resource() -> str:
            """Return the agent's memory state."""
            return str(getattr(agent, "_memory", "No memory configured"))

        server.expose_resource("context://memory", _memory_resource)

        def _chat_prompt(message: str) -> str:
            """Send a message through the agent."""
            return message

        server.expose_prompt("chat", _chat_prompt)

        return server

    @classmethod
    def from_pipeline(cls, pipeline: ContextPipeline) -> FastMCPServerBridge:
        """Create server exposing pipeline as retrieval tools.

        Exposes:
        - tool ``query`` -> run pipeline.build() and return results
        - resource ``context://result`` -> last pipeline result
        """
        server = cls(name="anchor-pipeline")
        _last_result: list[str] = []

        def query(text: str) -> str:
            """Run the context pipeline and return formatted results."""
            result = pipeline.build(text)
            output = str(result.formatted_output)
            _last_result.clear()
            _last_result.append(output)
            return output

        server._mcp.tool(name="query", description="Query the context pipeline")(query)
        server._registered_tools.append("query")

        def _result_resource() -> str:
            """Return the last pipeline result."""
            return _last_result[0] if _last_result else "No results yet"

        server.expose_resource("context://result", _result_resource)

        return server

    async def run(self, transport: str = "stdio") -> None:
        """Start the MCP server."""
        if transport == "stdio":
            await self._mcp.run_async(transport="stdio")
        elif transport == "http":
            await self._mcp.run_async(transport="streamable-http")
        elif transport == "sse":
            await self._mcp.run_async(transport="sse")
        else:
            msg = f"Unknown transport: {transport!r}. Use 'stdio', 'http', or 'sse'."
            raise ValueError(msg)
