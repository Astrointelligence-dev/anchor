"""Meta-tool for discovering deferred tools by keyword.

Tools registered with ``defer_loading=True`` are excluded from the
schemas sent to the model until discovered through ``search_tools``.
This keeps large tool sets out of the prompt (and out of the token
bill) until they are actually needed — the client-side analog of the
Anthropic Tool Search Tool. Once loaded, a tool stays loaded for the
rest of the session.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from anchor.agent.models import AgentTool
from anchor.agent.tool_decorator import tool

if TYPE_CHECKING:
    from anchor.agent.agent import Agent


def _make_search_tools_tool(agent: Agent) -> AgentTool:
    """Create the ``search_tools`` meta-tool bound to *agent*.

    Matches the query as a case-insensitive regex (falling back to a
    literal substring on invalid patterns) against each deferred tool's
    name and description. Matches become available on the next round.
    """

    @tool(
        name="search_tools",
        description=(
            "Search deferred tools by keyword or regex. Matching tools "
            "become available for use on your next response."
        ),
        read_only=True,
    )
    def search_tools(query: str) -> str:
        """Search deferred tools and load the matches.

        Args:
            query: Keyword or regex matched against tool names and
                descriptions.
        """
        deferred = [
            t for t in agent._all_active_tools() if t.defer_loading
        ]
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        matches = [
            t for t in deferred if pattern.search(f"{t.name} {t.description}")
        ]
        if not matches:
            names = ", ".join(t.name for t in deferred) or "none"
            return f"No tools matching '{query}'. Deferred tools: {names}"

        for t in matches:
            agent._deferred_loaded.add(t.name)
        listing = "\n".join(f"  - {t.name}: {t.description}" for t in matches)
        return f"Loaded tools (available next round):\n{listing}"

    return search_tools
