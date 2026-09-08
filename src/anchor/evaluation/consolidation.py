"""Golden-set consolidation evaluation: does memory get better, not just smaller?

No public benchmark labels memory operations, and the ones that measure
consolidation (ForgetEval, MemStrata, MemConflict — 2026) score the *state
of the store* and whether a stale fact still comes back. So does this: a
case is a store, a conversation (in one or more steps), and the expected
result — substrings that must stay live, substrings that must not, how
many entries stay live, and probe searches that must or must not hit.
Deterministic substring checks, no judge; an in-place ``UPDATE`` and a
``DELETE`` + ``ADD`` are the same final state.

Usage::

    cases = load_consolidation_set("consolidation_golden.jsonl")
    report = evaluate_consolidator(extractor, consolidator, cases)
    assert_metric_floor(report, "passed", 0.8)   # in a CI test
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anchor.memory.manager import MemoryManager
from anchor.models.memory import ConversationTurn, MemoryEntry
from anchor.protocols.memory import MemoryConsolidator, MemoryExtractor
from anchor.protocols.storage import MemoryEntryStore
from anchor.storage.json_memory_store import InMemoryEntryStore


class Probe(BaseModel):
    """A search against the consolidated store: what its top-k must and must not contain."""

    model_config = ConfigDict(frozen=True)

    query: str
    must_hit: list[str] = Field(default_factory=list)
    must_not_hit: list[str] = Field(default_factory=list)


class ConsolidationCase(BaseModel):
    """One golden case: the store before, the conversation, the state expected after.

    ``turns`` is a list of steps, each a list of turns; every step is
    remembered on its own, so a case can replay "moved to Rio" and then
    "moved back". A flat list of turns is a single step. ``expected_live``
    is a count or an inclusive ``[min, max]`` range of live entries.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    existing: list[str] = Field(default_factory=list)
    turns: list[list[ConversationTurn]]
    live_contains: list[str] = Field(default_factory=list)
    live_not_contains: list[str] = Field(default_factory=list)
    expected_live: int | tuple[int, int]
    probes: list[Probe] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _one_step_when_flat(cls, data: Any) -> Any:
        turns = data.get("turns") if isinstance(data, dict) else None
        if turns and isinstance(turns[0], dict):
            data = {**data, "turns": [turns]}
        return data


class ConsolidationMetrics(BaseModel):
    """Per-case checks as 1.0/0.0, so a report mean is a pass rate."""

    model_config = ConfigDict(frozen=True)

    state_ok: float
    size_ok: float
    probes_ok: float

    @property
    def passed(self) -> float:
        return float(self.state_ok and self.size_ok and self.probes_ok)


class ConsolidationCaseResult(BaseModel):
    """What the store looked like after the case, and how it scored."""

    model_config = ConfigDict(frozen=True)

    case: ConsolidationCase
    metrics: ConsolidationMetrics
    live: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()


class ConsolidationReport(BaseModel):
    """Aggregate report over a consolidation golden set."""

    results: tuple[ConsolidationCaseResult, ...] = ()
    k: int = Field(default=5, ge=1)

    def mean(self, metric: str) -> float:
        """Mean of a ``ConsolidationMetrics`` field (``passed`` included)."""
        if not self.results:
            return 0.0
        values = [float(getattr(r.metrics, metric)) for r in self.results]
        return sum(values) / len(values)

    def summary(self) -> dict[str, float]:
        return {m: self.mean(m) for m in ("passed", "state_ok", "size_ok", "probes_ok")}

    def failures(self) -> list[str]:
        """Names of the cases that did not pass."""
        return [r.case.name for r in self.results if not r.metrics.passed]


def load_consolidation_set(path: str | Path) -> list[ConsolidationCase]:
    """Load cases from a JSONL file, one ``ConsolidationCase`` per line."""
    cases: list[ConsolidationCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(ConsolidationCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Invalid consolidation case at {path}:{line_num}: {e}"
            raise ValueError(msg) from e
    return cases


def _contains(live: Sequence[str], needle: str) -> bool:
    return any(needle in content for content in live)


def _size_ok(count: int, expected: int | tuple[int, int]) -> bool:
    if isinstance(expected, int):
        return count == expected
    low, high = expected
    return low <= count <= high


def _probe_ok(store: MemoryEntryStore, probe: Probe, k: int) -> bool:
    hits = [e.content for e in store.search(probe.query, top_k=k)]
    return all(_contains(hits, s) for s in probe.must_hit) and not any(
        _contains(hits, s) for s in probe.must_not_hit
    )


def evaluate_consolidator(
    extractor: MemoryExtractor,
    consolidator: MemoryConsolidator | None,
    cases: Sequence[ConsolidationCase],
    *,
    store_factory: Callable[[], MemoryEntryStore] = InMemoryEntryStore,
    k: int = 5,
) -> ConsolidationReport:
    """Replay every case through ``MemoryManager.remember()`` and score the store.

    Each case gets a fresh store from *store_factory* seeded with its
    ``existing`` entries (ids ``e0``, ``e1``, ...); each step of its
    conversation is added to a manager over that store and remembered.
    *consolidator* ``None`` is the "add everything" baseline.
    """
    results: list[ConsolidationCaseResult] = []
    for case in cases:
        store = store_factory()
        for i, content in enumerate(case.existing):
            store.add(MemoryEntry(id=f"e{i}", content=content))
        operations: list[str] = []
        for step in case.turns:
            manager = MemoryManager(
                persistent_store=store,
                extractor=extractor,
                consolidator=consolidator,
                extract_window=len(step),
            )
            for turn in step:
                manager._add_message(turn.role, turn.content)
            operations.extend(str(op) for op, _ in manager.remember())
        live = [e.content for e in store.list_all()]
        metrics = ConsolidationMetrics(
            state_ok=float(
                all(_contains(live, s) for s in case.live_contains)
                and not any(_contains(live, s) for s in case.live_not_contains)
            ),
            size_ok=float(_size_ok(len(live), case.expected_live)),
            probes_ok=float(all(_probe_ok(store, p, k) for p in case.probes)),
        )
        results.append(
            ConsolidationCaseResult(
                case=case, metrics=metrics, live=tuple(live), operations=tuple(operations)
            )
        )
    return ConsolidationReport(results=tuple(results), k=k)
