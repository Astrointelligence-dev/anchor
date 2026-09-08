"""A/B testing framework for comparing retrieval strategies.

Provides statistical comparison of two retrievers on a shared evaluation
dataset, using a paired t-test to determine whether one significantly
outperforms the other.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anchor.evaluation.evaluator import PipelineEvaluator
from anchor.evaluation.models import RetrievalMetrics
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope, scope_kwargs
from anchor.protocols.retriever import Retriever

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic result models
# ------------------------------------------------------------------


class AggregatedMetrics(BaseModel):
    """Aggregated retrieval metrics across multiple evaluation samples.

    Parameters:
        mean_precision: Mean precision@k across all samples.
        mean_recall: Mean recall@k across all samples.
        mean_f1: Mean F1@k across all samples.
        mean_mrr: Mean MRR across all samples.
        mean_ndcg: Mean NDCG across all samples.
        num_samples: Number of samples evaluated.
    """

    model_config = ConfigDict(frozen=True)

    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    num_samples: int = 0


class EvaluationSample(BaseModel):
    """A single evaluation sample with a query and its relevant document IDs.

    Parameters:
        query: The query string.
        relevant_ids: IDs of documents that are relevant to the query.
        metadata: Arbitrary metadata for the sample.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """A collection of evaluation samples.

    Parameters:
        samples: The list of evaluation samples.
        name: Optional name for the dataset.
        metadata: Arbitrary metadata for the dataset.
    """

    model_config = ConfigDict(frozen=True)

    samples: list[EvaluationSample] = Field(default_factory=list)
    name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ABTestResult(BaseModel):
    """Result of an A/B test comparing two retrievers.

    Parameters:
        metrics_a: Aggregated metrics for retriever A.
        metrics_b: Aggregated metrics for retriever B.
        winner: Which retriever won: ``"a"``, ``"b"``, or ``"tie"``.
        p_value: The p-value from the paired t-test.
        is_significant: Whether the result is statistically significant.
        significance_level: The threshold for significance (default 0.05).
        per_metric_comparison: Per-metric deltas and details.
        metadata: Arbitrary metadata for the test result.
    """

    model_config = ConfigDict(frozen=True)

    metrics_a: AggregatedMetrics
    metrics_b: AggregatedMetrics
    winner: str  # "a", "b", or "tie"
    p_value: float
    is_significant: bool
    significance_level: float = 0.05
    per_metric_comparison: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Statistical helpers (stdlib-only, no scipy)
# ------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function.

    Parameters:
        x: The z-score value.

    Returns:
        Approximate probability P(Z <= x).
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_test_paired(values_a: list[float], values_b: list[float]) -> float:
    """Compute approximate p-value for a paired t-test using stdlib only.

    Parameters:
        values_a: Per-sample metric values for retriever A.
        values_b: Per-sample metric values for retriever B.

    Returns:
        Approximate two-sided p-value.
    """
    n = len(values_a)
    if n < 2:
        return 1.0
    diffs = [a - b for a, b in zip(values_a, values_b, strict=True)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    if std_diff == 0:
        return 0.0 if mean_diff != 0 else 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    # Approximate p-value using normal distribution for large n.
    # For small n this is conservative but avoids scipy dependency.
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return p_value


# ------------------------------------------------------------------
# ABTestRunner
# ------------------------------------------------------------------


class ABTestRunner:
    """Run an A/B test comparing two retrievers on a shared dataset.

    Parameters:
        evaluator: The pipeline evaluator used to compute retrieval metrics.
        dataset: The evaluation dataset containing queries and relevance labels.
    """

    __slots__ = ("_dataset", "_evaluator")

    def __init__(self, evaluator: PipelineEvaluator, dataset: EvaluationDataset) -> None:
        self._evaluator = evaluator
        self._dataset = dataset

    def run(
        self,
        retriever_a: Retriever,
        retriever_b: Retriever,
        k: int = 10,
        significance_level: float = 0.05,
        *,
        scope: RetrievalScope | None = None,
    ) -> ABTestResult:
        """Execute the A/B test and return statistical results.

        Both retrievers are evaluated on every sample in the dataset.
        A paired t-test on precision@k determines significance.

        Parameters:
            retriever_a: The first retriever ("A").
            retriever_b: The second retriever ("B").
            k: Top-k cutoff for retrieval evaluation.
            significance_level: Threshold below which p-value is significant.
            scope: Namespace scope forwarded to both retrievers (only when set).

        Returns:
            An ``ABTestResult`` with aggregated metrics, p-value, and winner.
        """
        samples = self._dataset.samples
        if not samples:
            empty_agg = AggregatedMetrics()
            return ABTestResult(
                metrics_a=empty_agg,
                metrics_b=empty_agg,
                winner="tie",
                p_value=1.0,
                is_significant=False,
                significance_level=significance_level,
            )

        metrics_a_list: list[RetrievalMetrics] = []
        metrics_b_list: list[RetrievalMetrics] = []

        for sample in samples:
            query = QueryBundle(query_str=sample.query)
            retrieved_a = retriever_a.retrieve(query, top_k=k, **scope_kwargs(scope))
            retrieved_b = retriever_b.retrieve(query, top_k=k, **scope_kwargs(scope))
            m_a = self._evaluator.evaluate_retrieval(retrieved_a, sample.relevant_ids, k=k)
            m_b = self._evaluator.evaluate_retrieval(retrieved_b, sample.relevant_ids, k=k)
            metrics_a_list.append(m_a)
            metrics_b_list.append(m_b)

        agg_a = self._aggregate(metrics_a_list)
        agg_b = self._aggregate(metrics_b_list)

        # Paired t-test on precision@k
        precisions_a = [m.precision_at_k for m in metrics_a_list]
        precisions_b = [m.precision_at_k for m in metrics_b_list]
        p_value = _t_test_paired(precisions_a, precisions_b)

        is_significant = p_value < significance_level
        if is_significant:
            winner = "a" if agg_a.mean_precision > agg_b.mean_precision else "b"
        else:
            winner = "tie"

        per_metric = self._build_per_metric_comparison(agg_a, agg_b)

        return ABTestResult(
            metrics_a=agg_a,
            metrics_b=agg_b,
            winner=winner,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=significance_level,
            per_metric_comparison=per_metric,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(metrics_list: list[RetrievalMetrics]) -> AggregatedMetrics:
        """Aggregate a list of per-sample metrics into means."""
        n = len(metrics_list)
        if n == 0:
            return AggregatedMetrics()
        return AggregatedMetrics(
            mean_precision=statistics.mean(m.precision_at_k for m in metrics_list),
            mean_recall=statistics.mean(m.recall_at_k for m in metrics_list),
            mean_f1=statistics.mean(m.f1_at_k for m in metrics_list),
            mean_mrr=statistics.mean(m.mrr for m in metrics_list),
            mean_ndcg=statistics.mean(m.ndcg for m in metrics_list),
            num_samples=n,
        )

    @staticmethod
    def _build_per_metric_comparison(
        agg_a: AggregatedMetrics,
        agg_b: AggregatedMetrics,
    ) -> dict[str, dict[str, Any]]:
        """Build a per-metric comparison dict with deltas."""
        metric_pairs = [
            ("precision", agg_a.mean_precision, agg_b.mean_precision),
            ("recall", agg_a.mean_recall, agg_b.mean_recall),
            ("f1", agg_a.mean_f1, agg_b.mean_f1),
            ("mrr", agg_a.mean_mrr, agg_b.mean_mrr),
            ("ndcg", agg_a.mean_ndcg, agg_b.mean_ndcg),
        ]
        result: dict[str, dict[str, Any]] = {}
        for name, val_a, val_b in metric_pairs:
            result[name] = {
                "a": val_a,
                "b": val_b,
                "delta": val_a - val_b,
            }
        return result

    def __repr__(self) -> str:
        return (
            f"ABTestRunner(evaluator={self._evaluator!r}, "
            f"dataset_size={len(self._dataset.samples)})"
        )
