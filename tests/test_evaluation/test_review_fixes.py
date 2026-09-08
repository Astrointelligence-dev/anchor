"""Regression test for the 2026-08-31 review, finding 43: NDCG's ideal
ranking was truncated at len(retrieved) instead of k — a retriever that
returned fewer than k items was graded against a shrunken ideal and
scored an inflated (even perfect) NDCG, so the CI gate passed on real
regressions that shrank the result set.
"""

from __future__ import annotations

from anchor.evaluation.retrieval import RetrievalMetricsCalculator
from anchor.models.context import ContextItem, SourceType


def _item(doc_id: str) -> ContextItem:
    return ContextItem(
        id=doc_id, content=f"doc {doc_id}", source=SourceType.RETRIEVAL,
    )


class TestNdcgIdealTruncatesAtK:
    def test_graded_short_result_set_is_not_perfect(self):
        calc = RetrievalMetricsCalculator(k=10)
        grades = {f"d{i}": 3.0 for i in range(5)}
        retrieved = [_item("d0"), _item("d1")]  # broken: only 2 of 5

        metrics = calc.evaluate(retrieved, grades)
        assert metrics.ndcg < 1.0  # pre-fix: exactly 1.0

    def test_binary_short_result_set_is_not_perfect(self):
        calc = RetrievalMetricsCalculator(k=10)
        relevant = {f"d{i}" for i in range(5)}
        retrieved = [_item("d0"), _item("d1")]

        metrics = calc.evaluate(retrieved, relevant)
        assert metrics.ndcg < 1.0

    def test_full_perfect_ranking_still_scores_one(self):
        calc = RetrievalMetricsCalculator(k=5)
        grades = {f"d{i}": 3.0 for i in range(5)}
        retrieved = [_item(f"d{i}") for i in range(5)]

        metrics = calc.evaluate(retrieved, grades)
        assert metrics.ndcg == 1.0
