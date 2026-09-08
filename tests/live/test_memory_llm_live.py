"""LLMExtractor + LLMConsolidator against the real Claude Code CLI (no API key).

Skips unless the ``claude`` CLI and ``claude-agent-sdk`` are installed. The
golden-set run is ``slow`` (~100 model calls); the single scenario is the
plan's own acceptance check.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from anchor.evaluation.consolidation import evaluate_consolidator, load_consolidation_set
from anchor.evaluation.golden import assert_metric_floor
from anchor.memory.consolidator import LLMConsolidator
from anchor.memory.extractor import LLMExtractor
from anchor.memory.manager import MemoryManager
from anchor.models.memory import MemoryEntry
from anchor.storage.json_memory_store import InMemoryEntryStore

_CLAUDE_CLI = bool(shutil.which("claude")) and bool(importlib.util.find_spec("claude_agent_sdk"))
_GOLDEN = Path(__file__).parents[1] / "fixtures" / "consolidation_golden.jsonl"

pytestmark = pytest.mark.skipif(not _CLAUDE_CLI, reason="claude CLI or claude-agent-sdk missing")


class _Counting:
    """Counts provider calls: the cost the plan asks to measure."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return self._llm.invoke(messages, **kwargs)


def _llm() -> _Counting:
    from anchor.llm.providers.claude_cli import ClaudeCLIProvider

    return _Counting(ClaudeCLIProvider(model="sonnet", max_response_tokens=400))


def test_moving_city_becomes_an_update_not_a_contradiction() -> None:
    llm = _llm()
    store = InMemoryEntryStore()
    store.add(MemoryEntry(id="m1", content="User lives in São Paulo"))
    manager = MemoryManager(
        persistent_store=store, extractor=LLMExtractor(llm), consolidator=LLMConsolidator(llm)
    )
    manager.add_user_message("Me mudei pro Rio de Janeiro mês passado.")
    manager.add_assistant_message("Que legal! Como está sendo a mudança?")

    ops = manager.remember()

    live = store.list_all()
    print(ops, [e.content for e in live], f"calls={llm.calls}")
    assert len(live) == 1
    assert "rio" in live[0].content.lower()
    assert "são paulo" not in live[0].content.lower()
    assert llm.calls == 2  # one extraction, one consolidation


@pytest.mark.slow
def test_golden_set_with_the_llm_pair() -> None:
    llm = _llm()
    cases = load_consolidation_set(_GOLDEN)
    report = evaluate_consolidator(LLMExtractor(llm), LLMConsolidator(llm), cases)
    print(report.summary(), f"calls={llm.calls} cases={len(cases)}")
    for r in report.results:
        if not r.metrics.passed:
            print("FAIL", r.case.name, r.live, r.operations)
    assert_metric_floor(report, "passed", 0.75)
