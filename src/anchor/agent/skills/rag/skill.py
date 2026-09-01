"""RAG skill factory."""

from __future__ import annotations

from collections.abc import Callable

from anchor.agent.skills.models import Skill
from anchor.models.scope import RetrievalScope

from .tools import rag_tools


def rag_skill(
    retriever: object,
    embed_fn: Callable[[str], list[float]] | None = None,
    *,
    scope: RetrievalScope | None = None,
) -> Skill:
    """Create a Skill that wraps :func:`rag_tools`.

    Returns a skill with one tool: ``search_docs``.

    Parameters
    ----------
    retriever:
        Any object with a ``retrieve(query, top_k)`` method.
    embed_fn:
        Optional embedding function for the query bundle.
    """
    tools = rag_tools(retriever, embed_fn, scope=scope)
    return Skill(
        name="rag",
        description="Search documentation for relevant information.",
        instructions=(
            "RAG skill provides document search.\n"
            "- Use search_docs when the user asks about features, APIs, or concepts.\n"
            "- Results are ranked by relevance; top 5 are returned."
        ),
        tools=tuple(tools),
        activation="on_demand",
        tags=("core", "retrieval"),
    )
