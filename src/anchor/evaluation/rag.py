"""LLM-based RAG evaluation via user-supplied callback functions.

The evaluator delegates each metric dimension to an optional callback.
This keeps the evaluation logic free of any specific LLM SDK while allowing
users to plug in their own judge models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from anchor.evaluation.models import RAGMetrics

logger = logging.getLogger(__name__)


class LLMRAGEvaluator:
    """RAGAS-style RAG evaluator driven by callback functions.

    Each callback computes a single metric dimension and returns a float
    in [0.0, 1.0].  Missing callbacks default to ``0.0``.

    Parameters:
        faithfulness_fn: ``(answer, contexts) -> float``
        relevancy_fn: ``(query, answer) -> float``
        precision_fn: ``(query, contexts) -> float``
        recall_fn: ``(query, contexts, ground_truth) -> float``
    """

    __slots__ = (
        "_faithfulness_fn",
        "_precision_fn",
        "_recall_fn",
        "_relevancy_fn",
    )

    def __init__(
        self,
        *,
        faithfulness_fn: Callable[[str, list[str]], float] | None = None,
        relevancy_fn: Callable[[str, str], float] | None = None,
        precision_fn: Callable[[str, list[str]], float] | None = None,
        recall_fn: Callable[[str, list[str], str], float] | None = None,
    ) -> None:
        if not any((faithfulness_fn, relevancy_fn, precision_fn, recall_fn)):
            msg = (
                "LLMRAGEvaluator requires at least one metric callback -- "
                "an evaluator with none configured can only produce "
                "meaningless results"
            )
            raise ValueError(msg)
        self._faithfulness_fn = faithfulness_fn
        self._relevancy_fn = relevancy_fn
        self._precision_fn = precision_fn
        self._recall_fn = recall_fn

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGMetrics:
        """Evaluate RAG output quality using the registered callbacks.

        Parameters:
            query: The original user query.
            answer: The generated answer.
            contexts: Context strings fed to the generator.
            ground_truth: Optional reference answer for recall computation.

        Returns:
            A ``RAGMetrics`` instance.  Dimensions without callbacks are
            ``None`` (not evaluated), never a silent 0.0.
        """
        faithfulness = (
            self._faithfulness_fn(answer, contexts) if self._faithfulness_fn is not None else None
        )
        relevancy = self._relevancy_fn(query, answer) if self._relevancy_fn is not None else None
        precision = self._precision_fn(query, contexts) if self._precision_fn is not None else None
        recall = (
            self._recall_fn(query, contexts, ground_truth)
            if self._recall_fn is not None and ground_truth is not None
            else None
        )

        return RAGMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )
