"""Protocol definitions for anchor's pluggable architecture."""

from .cache import CacheBackend
from .classifier import QueryClassifier
from .embeddings import EmbeddingProvider
from .evaluation import HumanEvaluator, RAGEvaluator, RetrievalEvaluator
from .ingestion import Chunker, DocumentParser
from .memory import (
    AsyncCompactionStrategy,
    AsyncMemoryExtractor,
    CompactionStrategy,
    ConversationMemory,
    EvictionPolicy,
    MemoryConsolidator,
    MemoryDecay,
    MemoryExtractor,
    MemoryOperation,
    MemoryProvider,
    MemoryQueryEnricher,
    QueryEnricher,
    RecencyScorer,
)
from .multimodal import ModalityEncoder, TableExtractor
from .observability import MetricsCollector, SpanExporter
from .postprocessor import AsyncPostProcessor, PostProcessor
from .query_transform import AsyncQueryTransformer, QueryTransformer
from .reranker import AsyncReranker, Reranker
from .retriever import AsyncRetriever, Retriever
from .router import QueryRouter
from .storage import (
    AsyncGraphStore,
    ContextStore,
    DocumentStore,
    GarbageCollectableStore,
    GraphStore,
    MemoryEntryStore,
    VectorStore,
)
from .tokenizer import Tokenizer

__all__ = [
    "AsyncCompactionStrategy",
    "AsyncGraphStore",
    "AsyncMemoryExtractor",
    "AsyncPostProcessor",
    "AsyncQueryTransformer",
    "AsyncReranker",
    "AsyncRetriever",
    "CacheBackend",
    "Chunker",
    "CompactionStrategy",
    "ContextStore",
    "ConversationMemory",
    "DocumentParser",
    "DocumentStore",
    "EmbeddingProvider",
    "EvictionPolicy",
    "GarbageCollectableStore",
    "GraphStore",
    "HumanEvaluator",
    "MemoryConsolidator",
    "MemoryDecay",
    "MemoryEntryStore",
    "MemoryExtractor",
    "MemoryOperation",
    "MemoryProvider",
    "MemoryQueryEnricher",
    "MetricsCollector",
    "ModalityEncoder",
    "PostProcessor",
    "QueryClassifier",
    "QueryEnricher",
    "QueryRouter",
    "QueryTransformer",
    "RAGEvaluator",
    "RecencyScorer",
    "Reranker",
    "RetrievalEvaluator",
    "Retriever",
    "SpanExporter",
    "TableExtractor",
    "Tokenizer",
    "VectorStore",
]
