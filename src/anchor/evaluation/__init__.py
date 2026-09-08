"""Evaluation framework for retrieval and RAG quality assessment."""

from .ab_testing import (
    ABTestResult,
    ABTestRunner,
    AggregatedMetrics,
    EvaluationDataset,
    EvaluationSample,
)
from .batch import BatchEvaluator
from .consolidation import (
    ConsolidationCase,
    ConsolidationCaseResult,
    ConsolidationMetrics,
    ConsolidationReport,
    Probe,
    evaluate_consolidator,
    load_consolidation_set,
)
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
    "ConsolidationCase",
    "ConsolidationCaseResult",
    "ConsolidationMetrics",
    "ConsolidationReport",
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
    "Probe",
    "RAGMetrics",
    "RetrievalMetrics",
    "RetrievalMetricsCalculator",
    "assert_metric_floor",
    "evaluate_consolidator",
    "evaluate_retriever",
    "load_consolidation_set",
    "load_golden_set",
]
