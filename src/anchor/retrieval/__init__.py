"""Retrieval strategies for anchor."""

from ._rrf import rrf_fuse
from .async_reranker import AsyncCohereReranker, AsyncCrossEncoderReranker
from .async_retriever import AsyncDenseRetriever, AsyncHybridRetriever
from .cross_modal import CrossModalEncoder, SharedSpaceRetriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .late_interaction import (
    LateInteractionRetriever,
    LateInteractionScorer,
    MaxSimScorer,
)
from .memory_retriever import MemoryRetrieverAdapter, ScoredMemoryRetriever
from .rerankers import (
    CohereReranker,
    CrossEncoderReranker,
    FlashRankReranker,
    RerankerPipeline,
    RoundRobinReranker,
)
from .router import CallbackRouter, KeywordRouter, MetadataRouter, RoutedRetriever
from .sparse import SparseRetriever

__all__ = [
    "AsyncCohereReranker",
    "AsyncCrossEncoderReranker",
    "AsyncDenseRetriever",
    "AsyncHybridRetriever",
    "CallbackRouter",
    "CohereReranker",
    "CrossEncoderReranker",
    "CrossModalEncoder",
    "DenseRetriever",
    "FlashRankReranker",
    "HybridRetriever",
    "KeywordRouter",
    "LateInteractionRetriever",
    "LateInteractionScorer",
    "MaxSimScorer",
    "MemoryRetrieverAdapter",
    "MetadataRouter",
    "RerankerPipeline",
    "RoundRobinReranker",
    "RoutedRetriever",
    "ScoredMemoryRetriever",
    "SharedSpaceRetriever",
    "SparseRetriever",
    "rrf_fuse",
]
