"""Tests for top-level package exports."""

from __future__ import annotations


class TestTopLevelExports:
    """Verify all expected symbols are importable from the top-level package."""

    def test_pipeline_exports(self) -> None:
        from anchor import (
            ContextPipeline,
            PipelineStep,
            async_postprocessor_step,
            async_reranker_step,
            async_retriever_step,
            filter_step,
            postprocessor_step,
            query_transform_step,
            reranker_step,
            retriever_step,
        )

        assert ContextPipeline is not None
        assert PipelineStep is not None
        assert async_postprocessor_step is not None
        assert async_reranker_step is not None
        assert async_retriever_step is not None
        assert filter_step is not None
        assert postprocessor_step is not None
        assert query_transform_step is not None
        assert reranker_step is not None
        assert retriever_step is not None

    def test_model_exports(self) -> None:
        from anchor import (
            BudgetAllocation,
            ContextItem,
            ContextResult,
            ContextWindow,
            ConversationTurn,
            MemoryEntry,
            OverflowStrategy,
            PipelineDiagnostics,
            QueryBundle,
            Role,
            SourceType,
            StepDiagnostic,
            TokenBudget,
        )

        assert ContextItem is not None
        assert ContextResult is not None
        assert ContextWindow is not None
        assert QueryBundle is not None
        assert SourceType is not None
        assert TokenBudget is not None
        assert BudgetAllocation is not None
        assert ConversationTurn is not None
        assert MemoryEntry is not None
        assert OverflowStrategy is not None
        assert PipelineDiagnostics is not None
        assert Role is not None
        assert StepDiagnostic is not None

    def test_protocol_exports(self) -> None:
        from anchor import (
            AsyncPostProcessor,
            AsyncReranker,
            AsyncRetriever,
            BaseFormatter,
            ContextStore,
            Formatter,
            PostProcessor,
            Reranker,
            Retriever,
            Tokenizer,
            VectorStore,
        )

        assert AsyncPostProcessor is not None
        assert AsyncReranker is not None
        assert AsyncRetriever is not None
        assert BaseFormatter is not None
        assert Formatter is not None
        assert BaseFormatter is Formatter  # backward-compat alias
        assert ContextStore is not None
        assert PostProcessor is not None
        assert Reranker is not None
        assert Retriever is not None
        assert Tokenizer is not None
        assert VectorStore is not None

    def test_exception_exports(self) -> None:
        from anchor import (
            AstroContextError,
            FormatterError,
            RetrieverError,
            StorageError,
        )

        assert AstroContextError is not None
        assert FormatterError is not None
        assert RetrieverError is not None
        assert StorageError is not None

    def test_retrieval_exports(self) -> None:
        from anchor import (
            AsyncCohereReranker,
            AsyncCrossEncoderReranker,
            AsyncDenseRetriever,
            AsyncHybridRetriever,
            CohereReranker,
            CrossEncoderReranker,
            CrossModalEncoder,
            DenseRetriever,
            FlashRankReranker,
            HybridRetriever,
            RerankerPipeline,
            RoundRobinReranker,
            SharedSpaceRetriever,
            SparseRetriever,
            rrf_fuse,
        )

        assert AsyncCohereReranker is not None
        assert AsyncCrossEncoderReranker is not None
        assert AsyncDenseRetriever is not None
        assert AsyncHybridRetriever is not None
        assert CohereReranker is not None
        assert CrossEncoderReranker is not None
        assert CrossModalEncoder is not None
        assert DenseRetriever is not None
        assert FlashRankReranker is not None
        assert HybridRetriever is not None
        assert RerankerPipeline is not None
        assert RoundRobinReranker is not None
        assert SharedSpaceRetriever is not None
        assert SparseRetriever is not None
        assert rrf_fuse is not None

    def test_formatter_exports(self) -> None:
        from anchor import AnthropicFormatter, GenericTextFormatter, OpenAIFormatter

        assert AnthropicFormatter is not None
        assert GenericTextFormatter is not None
        assert OpenAIFormatter is not None

    def test_memory_exports(self) -> None:
        from anchor import MemoryManager, SlidingWindowMemory

        assert MemoryManager is not None
        assert SlidingWindowMemory is not None

    def test_storage_exports(self) -> None:
        from anchor import InMemoryContextStore, InMemoryVectorStore

        assert InMemoryContextStore is not None
        assert InMemoryVectorStore is not None

    def test_ingestion_exports(self) -> None:
        from anchor import (
            Chunker,
            CodeChunker,
            DocumentIngester,
            DocumentParser,
            FixedSizeChunker,
            HTMLParser,
            IngestionError,
            MarkdownParser,
            MetadataEnricher,
            ParentChildChunker,
            ParentExpander,
            PDFParser,
            PlainTextParser,
            RecursiveCharacterChunker,
            SemanticChunker,
            SentenceChunker,
            TableAwareChunker,
            extract_chunk_metadata,
            generate_chunk_id,
            generate_doc_id,
        )

        assert Chunker is not None
        assert CodeChunker is not None
        assert DocumentIngester is not None
        assert DocumentParser is not None
        assert FixedSizeChunker is not None
        assert HTMLParser is not None
        assert IngestionError is not None
        assert MarkdownParser is not None
        assert MetadataEnricher is not None
        assert PDFParser is not None
        assert ParentChildChunker is not None
        assert ParentExpander is not None
        assert PlainTextParser is not None
        assert RecursiveCharacterChunker is not None
        assert SemanticChunker is not None
        assert SentenceChunker is not None
        assert TableAwareChunker is not None
        assert extract_chunk_metadata is not None
        assert generate_chunk_id is not None
        assert generate_doc_id is not None

    def test_token_exports(self) -> None:
        from anchor import TiktokenCounter

        assert TiktokenCounter is not None

    def test_query_transform_exports(self) -> None:
        from anchor import (
            AsyncQueryTransformer,
            DecompositionTransformer,
            HyDETransformer,
            MultiQueryTransformer,
            QueryTransformer,
            QueryTransformPipeline,
            StepBackTransformer,
            query_transform_step,
        )

        assert AsyncQueryTransformer is not None
        assert DecompositionTransformer is not None
        assert HyDETransformer is not None
        assert MultiQueryTransformer is not None
        assert QueryTransformPipeline is not None
        assert QueryTransformer is not None
        assert StepBackTransformer is not None
        assert query_transform_step is not None

    def test_evaluation_exports(self) -> None:
        from anchor import (
            AggregatedMetrics,
            BatchEvaluator,
            EvaluationDataset,
            EvaluationResult,
            EvaluationSample,
            LLMRAGEvaluator,
            PipelineEvaluator,
            RAGEvaluator,
            RAGMetrics,
            RetrievalEvaluator,
            RetrievalMetrics,
            RetrievalMetricsCalculator,
        )

        assert AggregatedMetrics is not None
        assert BatchEvaluator is not None
        assert EvaluationDataset is not None
        assert EvaluationResult is not None
        assert EvaluationSample is not None
        assert LLMRAGEvaluator is not None
        assert PipelineEvaluator is not None
        assert RAGEvaluator is not None
        assert RAGMetrics is not None
        assert RetrievalEvaluator is not None
        assert RetrievalMetrics is not None
        assert RetrievalMetricsCalculator is not None

    def test_multimodal_exports(self) -> None:
        from anchor import (
            CompositeEncoder,
            HTMLTableParser,
            ImageDescriptionEncoder,
            MarkdownTableParser,
            ModalityEncoder,
            ModalityType,
            MultiModalContent,
            MultiModalConverter,
            MultiModalItem,
            TableEncoder,
            TableExtractor,
            TextEncoder,
        )

        assert CompositeEncoder is not None
        assert HTMLTableParser is not None
        assert ImageDescriptionEncoder is not None
        assert MarkdownTableParser is not None
        assert ModalityEncoder is not None
        assert ModalityType is not None
        assert MultiModalContent is not None
        assert MultiModalConverter is not None
        assert MultiModalItem is not None
        assert TableEncoder is not None
        assert TableExtractor is not None
        assert TextEncoder is not None

    def test_observability_exports(self) -> None:
        from anchor import (
            ConsoleSpanExporter,
            CostEntry,
            CostSummary,
            CostTracker,
            CostTrackingCallback,
            FileSpanExporter,
            InMemoryMetricsCollector,
            InMemorySpanExporter,
            LoggingMetricsCollector,
            MetricPoint,
            MetricsCollector,
            OTLPMetricsExporter,
            OTLPSpanExporter,
            Span,
            SpanExporter,
            SpanKind,
            Tracer,
            TraceRecord,
            TracingCallback,
        )

        assert ConsoleSpanExporter is not None
        assert CostEntry is not None
        assert CostSummary is not None
        assert CostTracker is not None
        assert CostTrackingCallback is not None
        assert FileSpanExporter is not None
        assert InMemoryMetricsCollector is not None
        assert InMemorySpanExporter is not None
        assert LoggingMetricsCollector is not None
        assert MetricPoint is not None
        assert MetricsCollector is not None
        assert OTLPMetricsExporter is not None
        assert OTLPSpanExporter is not None
        assert Span is not None
        assert SpanExporter is not None
        assert SpanKind is not None
        assert TraceRecord is not None
        assert Tracer is not None
        assert TracingCallback is not None

    def test_router_exports(self) -> None:
        from anchor import (
            CallbackRouter,
            KeywordRouter,
            MetadataRouter,
            QueryRouter,
            RoutedRetriever,
        )

        assert CallbackRouter is not None
        assert KeywordRouter is not None
        assert MetadataRouter is not None
        assert QueryRouter is not None
        assert RoutedRetriever is not None

    def test_cache_exports(self) -> None:
        from anchor import (
            CacheBackend,
            InMemoryCacheBackend,
        )

        assert CacheBackend is not None
        assert InMemoryCacheBackend is not None

    def test_version(self) -> None:
        from anchor import __version__

        assert __version__ == "0.1.1"
