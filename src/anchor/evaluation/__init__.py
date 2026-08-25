"""Evaluation framework for retrieval and RAG quality assessment."""

from .ab_testing import (
    ABTestResult,
    ABTestRunner,
    AggregatedMetrics,
    EvaluationDataset,
    EvaluationSample,
)
from .batch import BatchEvaluator
from .evaluator import PipelineEvaluator
from .golden import (
    GoldenCase,
    GoldenCaseResult,
    GoldenSetReport,
    assert_metric_floor,
    evaluate_retriever,
    load_golden_set,
)
from .human import HumanEvaluationCollector, HumanJudgment
from .models import EvaluationResult, RAGMetrics, RetrievalMetrics
from .rag import LLMRAGEvaluator
from .retrieval import RetrievalMetricsCalculator

__all__ = [
    "ABTestResult",
    "ABTestRunner",
    "AggregatedMetrics",
    "BatchEvaluator",
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationSample",
    "GoldenCase",
    "GoldenCaseResult",
    "GoldenSetReport",
    "HumanEvaluationCollector",
    "HumanJudgment",
    "LLMRAGEvaluator",
    "PipelineEvaluator",
    "RAGMetrics",
    "RetrievalMetrics",
    "RetrievalMetricsCalculator",
    "assert_metric_floor",
    "evaluate_retriever",
    "load_golden_set",
]
