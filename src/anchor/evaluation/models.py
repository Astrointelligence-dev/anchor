"""Pydantic models for evaluation results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMetrics(BaseModel):
    """Metrics for a single retrieval evaluation.

    All metrics are bounded in [0.0, 1.0].

    Parameters:
        precision_at_k: Fraction of retrieved items that are relevant.
        recall_at_k: Fraction of relevant items that were retrieved.
        f1_at_k: Harmonic mean of precision and recall.
        mrr: Mean Reciprocal Rank -- reciprocal of the rank of the first
            relevant item.
        ndcg: Normalized Discounted Cumulative Gain.
        hit_rate: Whether at least one relevant item was retrieved (0 or 1).
    """

    model_config = ConfigDict(frozen=True)

    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    f1_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)


class RAGMetrics(BaseModel):
    """RAGAS-style metrics for RAG evaluation.

    All metrics are bounded in [0.0, 1.0]. ``None`` means the dimension was
    NOT evaluated (no callback configured) — previously this silently
    scored 0.0, indistinguishable from a genuine zero.

    Parameters:
        faithfulness: How faithful the answer is to the provided contexts.
        answer_relevancy: How relevant the answer is to the query.
        context_precision: Precision of the retrieved contexts for the query.
        context_recall: Recall of the retrieved contexts against ground truth.
    """

    model_config = ConfigDict(frozen=True)

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    context_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    context_recall: float | None = Field(default=None, ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Complete evaluation result combining retrieval and RAG metrics.

    Parameters:
        retrieval_metrics: Retrieval-quality metrics, if computed.
        rag_metrics: RAG-quality metrics, if computed.
        metadata: Arbitrary metadata attached to the evaluation run.
    """

    retrieval_metrics: RetrievalMetrics | None = None
    rag_metrics: RAGMetrics | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
