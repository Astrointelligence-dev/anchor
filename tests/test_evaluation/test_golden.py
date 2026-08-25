"""Tests for the golden-set eval harness and graded NDCG (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anchor.evaluation.golden import (
    GoldenCase,
    assert_metric_floor,
    evaluate_retriever,
    load_golden_set,
)
from anchor.evaluation.retrieval import RetrievalMetricsCalculator
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle


def _item(item_id: str) -> ContextItem:
    return ContextItem(id=item_id, content=item_id, source=SourceType.RETRIEVAL)


class _StaticRetriever:
    """Returns a fixed ranking regardless of the query."""

    def __init__(self, ranking: dict[str, list[str]]) -> None:
        self._ranking = ranking

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return [_item(i) for i in self._ranking.get(query.query_str, [])[:top_k]]


class TestGoldenSet:
    def test_load_from_jsonl(self, tmp_path: Path) -> None:
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            '{"query": "q1", "relevant": ["a", "b"]}\n'
            '\n'
            '{"query": "q2", "relevant": {"c": 2.0, "d": 1.0}, "name": "graded"}\n'
        )
        cases = load_golden_set(golden)
        assert len(cases) == 2
        assert cases[0].relevant == ["a", "b"]
        assert cases[1].name == "graded"

    def test_invalid_line_reports_location(self, tmp_path: Path) -> None:
        golden = tmp_path / "bad.jsonl"
        golden.write_text('{"query": "q1", "relevant": ["a"]}\nnot json\n')
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            load_golden_set(golden)

    def test_evaluate_and_gate(self) -> None:
        retriever = _StaticRetriever({"q1": ["a", "x"], "q2": ["y", "z"]})
        cases = [
            GoldenCase(query="q1", relevant=["a"]),
            GoldenCase(query="q2", relevant=["c"], name="miss"),
        ]
        report = evaluate_retriever(retriever, cases, k=2)
        assert report.mean("recall_at_k") == pytest.approx(0.5)
        assert report.summary()["hit_rate"] == pytest.approx(0.5)

        assert_metric_floor(report, "recall_at_k", 0.4)  # passes
        with pytest.raises(AssertionError, match="miss"):
            assert_metric_floor(report, "recall_at_k", 0.9)


class TestGradedNDCG:
    def test_graded_beats_binary_ordering(self) -> None:
        calc = RetrievalMetricsCalculator(k=2)
        grades = {"high": 3.0, "low": 1.0}

        good = calc.evaluate([_item("high"), _item("low")], grades)
        bad = calc.evaluate([_item("low"), _item("high")], grades)
        assert good.ndcg == pytest.approx(1.0)
        assert bad.ndcg < good.ndcg

    def test_binary_list_still_works(self) -> None:
        calc = RetrievalMetricsCalculator(k=2)
        m = calc.evaluate([_item("a"), _item("x")], ["a"])
        assert m.ndcg == pytest.approx(1.0)
