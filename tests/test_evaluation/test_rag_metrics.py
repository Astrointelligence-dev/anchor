"""Tests for LLMRAGEvaluator."""

from __future__ import annotations

import pytest

from anchor.evaluation.models import RAGMetrics
from anchor.evaluation.rag import LLMRAGEvaluator
from anchor.protocols.evaluation import RAGEvaluator


class TestLLMRAGEvaluator:
    """Test LLM-based RAG evaluation with mock callbacks."""

    def test_protocol_compliance(self) -> None:
        evaluator = LLMRAGEvaluator(relevancy_fn=lambda q, a: 0.5)
        assert isinstance(evaluator, RAGEvaluator)

    def test_all_callbacks_provided(self) -> None:
        evaluator = LLMRAGEvaluator(
            faithfulness_fn=lambda _a, _c: 0.9,
            relevancy_fn=lambda _q, _a: 0.8,
            precision_fn=lambda _q, _c: 0.7,
            recall_fn=lambda _q, _c, _gt: 0.6,
        )

        m = evaluator.evaluate(
            query="What is X?",
            answer="X is Y.",
            contexts=["context 1"],
            ground_truth="X is indeed Y.",
        )

        assert m.faithfulness == pytest.approx(0.9)
        assert m.answer_relevancy == pytest.approx(0.8)
        assert m.context_precision == pytest.approx(0.7)
        assert m.context_recall == pytest.approx(0.6)

    def test_no_callbacks_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one metric callback"):
            LLMRAGEvaluator()

    def test_zero_score_distinct_from_not_evaluated(self) -> None:
        """A genuine 0.0 from a callback must not look like 'not configured'."""
        evaluator = LLMRAGEvaluator(relevancy_fn=lambda q, a: 0.0)
        m = evaluator.evaluate(query="Q", answer="A", contexts=["c"])
        assert m.answer_relevancy == 0.0
        assert m.faithfulness is None

    def test_partial_callbacks(self) -> None:
        evaluator = LLMRAGEvaluator(
            faithfulness_fn=lambda _a, _c: 0.85,
        )

        m = evaluator.evaluate(query="Q", answer="A", contexts=["c"])

        assert m.faithfulness == pytest.approx(0.85)
        assert m.answer_relevancy is None  # not configured -> not evaluated
        assert m.context_precision is None
        assert m.context_recall is None

    def test_recall_requires_ground_truth(self) -> None:
        """Recall callback is skipped when ground_truth is None."""
        recall_called = False

        def recall_fn(_q: str, _c: list[str], _gt: str) -> float:
            nonlocal recall_called
            recall_called = True
            return 1.0

        evaluator = LLMRAGEvaluator(recall_fn=recall_fn)

        m = evaluator.evaluate(query="Q", answer="A", contexts=["c"], ground_truth=None)

        assert not recall_called
        assert m.context_recall is None

    def test_recall_called_with_ground_truth(self) -> None:
        """Recall callback fires when ground_truth is provided."""
        evaluator = LLMRAGEvaluator(recall_fn=lambda _q, _c, _gt: 0.75)

        m = evaluator.evaluate(query="Q", answer="A", contexts=["c"], ground_truth="GT")

        assert m.context_recall == pytest.approx(0.75)

    def test_callbacks_receive_correct_arguments(self) -> None:
        """Verify that callbacks receive the right arguments."""
        captured: dict[str, tuple[object, ...]] = {}

        def faithfulness_fn(answer: str, contexts: list[str]) -> float:
            captured["faithfulness"] = (answer, contexts)
            return 0.5

        def relevancy_fn(query: str, answer: str) -> float:
            captured["relevancy"] = (query, answer)
            return 0.5

        def precision_fn(query: str, contexts: list[str]) -> float:
            captured["precision"] = (query, contexts)
            return 0.5

        def recall_fn(query: str, contexts: list[str], ground_truth: str) -> float:
            captured["recall"] = (query, contexts, ground_truth)
            return 0.5

        evaluator = LLMRAGEvaluator(
            faithfulness_fn=faithfulness_fn,
            relevancy_fn=relevancy_fn,
            precision_fn=precision_fn,
            recall_fn=recall_fn,
        )

        evaluator.evaluate(
            query="the query",
            answer="the answer",
            contexts=["ctx1", "ctx2"],
            ground_truth="the truth",
        )

        assert captured["faithfulness"] == ("the answer", ["ctx1", "ctx2"])
        assert captured["relevancy"] == ("the query", "the answer")
        assert captured["precision"] == ("the query", ["ctx1", "ctx2"])
        assert captured["recall"] == ("the query", ["ctx1", "ctx2"], "the truth")

    def test_returns_frozen_model(self) -> None:
        evaluator = LLMRAGEvaluator(relevancy_fn=lambda q, a: 0.5)
        m = evaluator.evaluate(query="Q", answer="A", contexts=[])
        assert isinstance(m, RAGMetrics)
        with pytest.raises(Exception):  # noqa: B017
            m.faithfulness = 0.5  # type: ignore[misc]
