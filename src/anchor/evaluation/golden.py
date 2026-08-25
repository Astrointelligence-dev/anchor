"""Golden-set retrieval evaluation: the CI gate for retrieval changes.

The 2026 practice: a small (50-200 query) hand-checked golden set of real
queries with known-relevant chunk ids. Every change to chunking, embedding,
sparse backend, or reranking is gated on recall@k holding or improving.

Usage::

    cases = load_golden_set("golden.jsonl")
    report = evaluate_retriever(retriever, cases, k=10)
    assert_metric_floor(report, "recall_at_k", 0.8)   # in a CI test
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from anchor.evaluation.models import RetrievalMetrics
from anchor.evaluation.retrieval import RetrievalMetricsCalculator
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle


class _Retriever(Protocol):
    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...


class GoldenCase(BaseModel):
    """One golden-set entry: a real query and its relevant document ids.

    ``relevant`` is either a list of ids (binary) or an ``{id: grade}``
    map (graded NDCG).
    """

    model_config = ConfigDict(frozen=True)

    query: str
    relevant: list[str] | dict[str, float]
    name: str = ""


class GoldenCaseResult(BaseModel):
    """Metrics for a single golden case."""

    model_config = ConfigDict(frozen=True)

    case: GoldenCase
    metrics: RetrievalMetrics
    retrieved_ids: tuple[str, ...] = ()


class GoldenSetReport(BaseModel):
    """Aggregate report over a golden set."""

    results: tuple[GoldenCaseResult, ...] = ()
    k: int = Field(default=10, ge=1)

    def mean(self, metric: str) -> float:
        """Mean of a ``RetrievalMetrics`` field across all cases."""
        if not self.results:
            return 0.0
        values = [float(getattr(r.metrics, metric)) for r in self.results]
        return sum(values) / len(values)

    def summary(self) -> dict[str, float]:
        """Means for every metric — the numbers to log and gate on."""
        return {
            metric: self.mean(metric)
            for metric in (
                "precision_at_k",
                "recall_at_k",
                "f1_at_k",
                "mrr",
                "ndcg",
                "hit_rate",
            )
        }


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    """Load golden cases from a JSONL file.

    Each line: ``{"query": "...", "relevant": ["id1", ...]}`` or
    ``{"query": "...", "relevant": {"id1": 2.0, ...}, "name": "..."}``.
    """
    cases: list[GoldenCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(GoldenCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Invalid golden-set entry at {path}:{line_num}: {e}"
            raise ValueError(msg) from e
    return cases


def evaluate_retriever(
    retriever: _Retriever,
    cases: Sequence[GoldenCase],
    k: int = 10,
    calculator: RetrievalMetricsCalculator | None = None,
) -> GoldenSetReport:
    """Run every golden case through *retriever* and score it.

    Parameters
    ----------
    retriever:
        Anything with ``retrieve(QueryBundle, top_k)`` — dense, sparse,
        hybrid, or a full pipeline wrapper.
    cases:
        The golden set.
    k:
        Retrieval cutoff (both for ``top_k`` and the @k metrics).
    calculator:
        Optional custom metrics calculator.
    """
    calc = calculator or RetrievalMetricsCalculator(k=k)
    results: list[GoldenCaseResult] = []
    for case in cases:
        retrieved = retriever.retrieve(QueryBundle(query_str=case.query), top_k=k)
        metrics = calc.evaluate(retrieved, case.relevant, k=k)
        results.append(
            GoldenCaseResult(
                case=case,
                metrics=metrics,
                retrieved_ids=tuple(item.id for item in retrieved),
            )
        )
    return GoldenSetReport(results=tuple(results), k=k)


def assert_metric_floor(
    report: GoldenSetReport, metric: str, floor: float
) -> None:
    """Raise ``AssertionError`` when a mean metric falls below *floor*.

    The CI-gate primitive: call it from a test so any retrieval change
    that regresses the golden set fails the build with the actual number.
    """
    value = report.mean(metric)
    if value < floor:
        worst = sorted(
            report.results, key=lambda r: getattr(r.metrics, metric)
        )[:3]
        detail = "; ".join(
            f"{r.case.name or r.case.query[:40]!r}={getattr(r.metrics, metric):.3f}"
            for r in worst
        )
        msg = (
            f"Golden-set {metric} {value:.3f} below floor {floor:.3f} "
            f"(k={report.k}). Worst cases: {detail}"
        )
        raise AssertionError(msg)
