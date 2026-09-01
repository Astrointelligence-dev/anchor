"""RAG search tool for the agent — mount-aware since front #3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anchor.agent.models import _ACTIVE_SCOPE, AgentTool
from anchor.agent.tool_decorator import tool
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope


def current_scope() -> RetrievalScope | None:
    """The retrieval scope published for the current tool call.

    ``None`` outside a scoped agent turn. Scope-aware tools intersect
    this with their own scope — the published scope already carries the
    parent∩child narrowing for subagent turns.
    """
    return _ACTIVE_SCOPE.get()


def _combined_scope(tool_scope: RetrievalScope | None) -> RetrievalScope | None:
    active = current_scope()
    if active is None:
        return tool_scope
    if tool_scope is None:
        return active
    return active.intersect(tool_scope)


def _retrieve(
    retriever: Any,
    q: QueryBundle,
    scope: RetrievalScope | None,
) -> list[Any]:
    # Pass scope only when set, so retrievers written before front #3
    # keep working unchanged (the where= compatibility pattern).
    if scope is not None:
        return list(retriever.retrieve(q, top_k=5, scope=scope))
    return list(retriever.retrieve(q, top_k=5))


def _format_results(results: list[Any]) -> str:
    if not results:
        return "No relevant documents found."
    parts: list[str] = []
    for item in results:
        section = item.metadata.get("section", "")
        prefix = f"[{section}] " if section else ""
        parts.append(f"{prefix}{item.content[:500]}")
    return "\n\n---\n\n".join(parts)


def rag_tools(
    retriever: Any,
    embed_fn: Callable[[str], list[float]] | None = None,
    *,
    scope: RetrievalScope | None = None,
) -> list[AgentTool]:
    """Create a ``search_docs`` tool for agentic RAG.

    The model decides when to search documentation, making this
    agentic RAG -- the model controls retrieval timing.

    Parameters
    ----------
    retriever:
        Any object with a ``retrieve(query, top_k)`` method — one mount.
        For multiple named mounts (vaults), pass a **dict**
        ``{"juridico": retriever_a, "notas": retriever_b}``: the tool
        gains a ``vault`` argument and the model chooses **among the
        mounted names only** — an unmounted vault is an error result,
        never a lookup.
    embed_fn:
        Optional embedding function.  If the retriever needs
        embeddings in the QueryBundle, provide this.
    scope:
        Optional namespace scope bound to this tool. The scope
        published by the running agent (which already carries subagent
        narrowing) is intersected on top — the effective scope only
        ever narrows.
    """
    if isinstance(retriever, dict):
        return _mounted_rag_tools(retriever, embed_fn, scope)

    @tool(
        description=(
            "Search documentation for relevant information. Use when the user "
            "asks about features, APIs, concepts, or anything that might be in the docs."
        ),
    )
    def search_docs(query: str) -> str:
        """Search documentation for relevant information.

        Args:
            query: Search query for finding relevant documentation.
        """
        q = QueryBundle(query_str=query)
        if embed_fn is not None:
            q = q.model_copy(update={"embedding": embed_fn(query)})
        results = _retrieve(retriever, q, _combined_scope(scope))
        return _format_results(results)

    return [search_docs]


def _mounted_rag_tools(
    mounts: dict[str, Any],
    embed_fn: Callable[[str], list[float]] | None,
    scope: RetrievalScope | None,
) -> list[AgentTool]:
    if not mounts:
        msg = "rag_tools mounts dict must not be empty"
        raise ValueError(msg)
    names = sorted(mounts)
    listing = ", ".join(names)

    @tool(
        description=(
            "Search the mounted document vaults for relevant information. "
            f"Available vaults: {listing}. Use when the user asks about "
            "anything that might be in those documents."
        ),
    )
    def search_docs(query: str, vault: str = "") -> str:
        """Search mounted document vaults for relevant information.

        Args:
            query: Search query for finding relevant documents.
            vault: Which mounted vault to search. Omit to search the
                only mount when a single one exists.
        """
        if not vault:
            if len(mounts) == 1:
                vault = names[0]
            else:
                return (
                    f"Error: choose a vault to search. Mounted: {listing}."
                )
        target = mounts.get(vault)
        if target is None:
            # Bounded choice: only what the app mounted is reachable.
            return f"Error: vault '{vault}' is not mounted. Mounted: {listing}."
        q = QueryBundle(query_str=query)
        if embed_fn is not None:
            q = q.model_copy(update={"embedding": embed_fn(query)})
        results = _retrieve(target, q, _combined_scope(scope))
        return _format_results(results)

    return [search_docs]
