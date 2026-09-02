"""Retrieval strategies for anchor."""

from ._rrf import rrf_fuse
from .async_reranker import AsyncCohereReranker, AsyncCrossEncoderReranker
from .async_retriever import AsyncDenseRetriever, AsyncHybridRetriever
from .cross_modal import CrossModalEncoder, SharedSpaceRetriever
from .dense import DenseRetriever
from .graph import GraphRetriever
from .hybrid import HybridRetriever
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
    "GraphRetriever",
    "HybridRetriever",
    "KeywordRouter",
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
