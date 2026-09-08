"""LLMGraphExtractor against the real Claude Code CLI (no API key).

Skips unless the ``claude`` CLI and ``claude-agent-sdk`` are installed.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest

from anchor.graph import KnowledgeGraph
from anchor.ingestion import GraphIndexer, LLMGraphExtractor
from anchor.models.context import ContextItem, SourceType

_CLAUDE_CLI = bool(shutil.which("claude")) and bool(importlib.util.find_spec("claude_agent_sdk"))


@pytest.mark.skipif(not _CLAUDE_CLI, reason="claude CLI or claude-agent-sdk missing")
def test_llm_graph_extractor_builds_a_navigable_graph() -> None:
    from anchor.llm.providers.claude_cli import ClaudeCLIProvider

    llm = ClaudeCLIProvider(model="sonnet", max_response_tokens=600)
    graph = KnowledgeGraph()
    item = ContextItem(
        id="mem-1",
        content=(
            "Ana Lima is the tech lead of Team Payments and the on-call engineer for "
            "the Billing service, which records every charge in the Ledger service."
        ),
        source=SourceType.MEMORY,
    )
    stats = GraphIndexer(graph, extractors=[LLMGraphExtractor(llm)]).index([item])
    assert stats.nodes >= 3
    assert stats.edges >= 2
    for edge in graph.store.subgraph().edges:
        assert edge.provenance in ("extracted", "inferred", "ambiguous")
        assert edge.evidence == ("mem-1",)
    assert graph.mentions("who is on call for the billing service?")
    print(graph.store.subgraph().edges)
