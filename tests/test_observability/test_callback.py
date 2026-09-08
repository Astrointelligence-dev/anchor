"""Tests for the TracingCallback pipeline integration."""

from __future__ import annotations

from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.observability.callback import TracingCallback, _infer_span_kind
from anchor.observability.exporters import InMemorySpanExporter
from anchor.observability.metrics import InMemoryMetricsCollector
from anchor.observability.models import SpanKind
from anchor.observability.tracer import Tracer
from anchor.pipeline.callbacks import PipelineCallback
from anchor.pipeline.pipeline import ContextPipeline
from anchor.pipeline.step import PipelineStep
from tests.conftest import FakeTokenizer


def _make_item(content: str = "test", score: float = 0.5) -> ContextItem:
    """Create a simple test context item."""
    return ContextItem(
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        token_count=1,
    )


class TestTracingCallbackProtocol:
    """Verify TracingCallback satisfies PipelineCallback."""

    def test_is_pipeline_callback(self) -> None:
        callback = TracingCallback()
        assert isinstance(callback, PipelineCallback)


class TestTracingCallbackLifecycle:
    """Test the callback through manual lifecycle invocations."""

    def test_pipeline_start_creates_trace(self) -> None:
        callback = TracingCallback()
        query = QueryBundle(query_str="hello")
        callback.on_pipeline_start(query)

        assert callback.tracer.get_trace(callback.last_trace.trace_id) is not None  # type: ignore[union-attr]

    def test_step_start_and_end_creates_span(self) -> None:
        exporter = InMemorySpanExporter()
        callback = TracingCallback(exporters=[exporter])

        query = QueryBundle(query_str="test")
        callback.on_pipeline_start(query)
        callback.on_step_start("retrieval", [_make_item()])
        callback.on_step_end("retrieval", [_make_item(), _make_item()], 10.5)

        # Build a fake result to trigger export
        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        result = pipeline.build(query)
        callback.on_pipeline_end(result)

        spans = exporter.get_spans()
        # Should have: pipeline span + retrieval span
        assert len(spans) == 2

    def test_step_error_records_error_status(self) -> None:
        exporter = InMemorySpanExporter()
        callback = TracingCallback(exporters=[exporter])

        query = QueryBundle(query_str="test")
        callback.on_pipeline_start(query)
        callback.on_step_start("bad-step", [])
        callback.on_step_error("bad-step", RuntimeError("boom"))

        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        result = pipeline.build(query)
        callback.on_pipeline_end(result)

        spans = exporter.get_spans()
        error_spans = [s for s in spans if s.status == "error"]
        assert len(error_spans) == 1
        assert "boom" in error_spans[0].attributes.get("error", "")

    def test_metrics_recorded(self) -> None:
        metrics = InMemoryMetricsCollector()
        callback = TracingCallback(metrics_collector=metrics)

        query = QueryBundle(query_str="test")
        callback.on_pipeline_start(query)
        callback.on_step_start("retrieval", [])
        callback.on_step_end("retrieval", [_make_item()], 15.0)

        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        result = pipeline.build(query)
        callback.on_pipeline_end(result)

        step_metrics = metrics.get_metrics("step.duration_ms")
        assert len(step_metrics) == 1
        assert step_metrics[0].value == 15.0
        assert step_metrics[0].tags["step"] == "retrieval"

        pipeline_metrics = metrics.get_metrics("pipeline.build_time_ms")
        assert len(pipeline_metrics) == 1


class TestTracingCallbackWithPipeline:
    """Integration test: TracingCallback wired into a real ContextPipeline."""

    def test_full_pipeline_trace(self) -> None:
        exporter = InMemorySpanExporter()
        metrics = InMemoryMetricsCollector()
        callback = TracingCallback(exporters=[exporter], metrics_collector=metrics)

        items = [_make_item("result")]

        def retriever_fn(
            existing: list[ContextItem], query: QueryBundle
        ) -> list[ContextItem]:
            return existing + items

        step = PipelineStep(name="search", fn=retriever_fn)
        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        pipeline.add_step(step)
        pipeline.add_callback(callback)

        pipeline.build("What is context?")

        # Verify spans exported
        spans = exporter.get_spans()
        assert len(spans) >= 2  # pipeline + search

        span_names = {s.name for s in spans}
        assert "pipeline" in span_names
        assert "search" in span_names

        # Verify the search span has correct attributes
        search_spans = [s for s in spans if s.name == "search"]
        assert len(search_spans) == 1
        assert search_spans[0].kind == SpanKind.RETRIEVAL
        assert search_spans[0].status == "ok"

        # Verify metrics
        assert len(metrics.get_metrics("step.duration_ms")) == 1
        assert len(metrics.get_metrics("pipeline.build_time_ms")) == 1
        assert len(metrics.get_metrics("pipeline.items_included")) == 1

    def test_pipeline_with_multiple_steps(self) -> None:
        exporter = InMemorySpanExporter()
        callback = TracingCallback(exporters=[exporter])

        def step1(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return [*items, _make_item("a")]

        def step2(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            return items

        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        pipeline.add_step(PipelineStep(name="fetch", fn=step1))
        pipeline.add_step(PipelineStep(name="rerank", fn=step2))
        pipeline.add_callback(callback)

        pipeline.build("test")

        spans = exporter.get_spans()
        span_names = {s.name for s in spans}
        assert "pipeline" in span_names
        assert "fetch" in span_names
        assert "rerank" in span_names

    def test_pipeline_with_failing_step_skip(self) -> None:
        exporter = InMemorySpanExporter()
        callback = TracingCallback(exporters=[exporter])

        def bad_step(items: list[ContextItem], q: QueryBundle) -> list[ContextItem]:
            msg = "intentional failure"
            raise RuntimeError(msg)

        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        pipeline.add_step(PipelineStep(name="bad", fn=bad_step, on_error="skip"))
        pipeline.add_callback(callback)

        pipeline.build("test")

        spans = exporter.get_spans()
        error_spans = [s for s in spans if s.status == "error"]
        assert len(error_spans) == 1
        assert error_spans[0].name == "bad"


class TestInferSpanKind:
    """Test the span kind inference heuristic."""

    def test_retrieval_keywords(self) -> None:
        assert _infer_span_kind("search") == SpanKind.RETRIEVAL
        assert _infer_span_kind("retriever") == SpanKind.RETRIEVAL
        assert _infer_span_kind("fetch-docs") == SpanKind.RETRIEVAL

    def test_reranking_keywords(self) -> None:
        assert _infer_span_kind("rerank") == SpanKind.RERANKING
        assert _infer_span_kind("score-ranker") == SpanKind.RERANKING

    def test_formatting_keywords(self) -> None:
        assert _infer_span_kind("format-output") == SpanKind.FORMATTING

    def test_memory_keywords(self) -> None:
        assert _infer_span_kind("memory-lookup") == SpanKind.MEMORY
        assert _infer_span_kind("history") == SpanKind.MEMORY

    def test_ingestion_keywords(self) -> None:
        assert _infer_span_kind("ingest-docs") == SpanKind.INGESTION
        assert _infer_span_kind("index-builder") == SpanKind.INGESTION

    def test_query_transform_keywords(self) -> None:
        assert _infer_span_kind("transform-query") == SpanKind.QUERY_TRANSFORM
        assert _infer_span_kind("expand-terms") == SpanKind.QUERY_TRANSFORM

    def test_default_fallback(self) -> None:
        assert _infer_span_kind("custom-step") == SpanKind.PIPELINE


class TestTracingCallbackEdgeCases:
    """Edge case tests."""

    def test_step_end_without_start(self) -> None:
        callback = TracingCallback()
        query = QueryBundle(query_str="test")
        callback.on_pipeline_start(query)
        # Call on_step_end without on_step_start — should not raise
        callback.on_step_end("unknown", [], 1.0)

    def test_step_error_without_start(self) -> None:
        callback = TracingCallback()
        query = QueryBundle(query_str="test")
        callback.on_pipeline_start(query)
        # Call on_step_error without on_step_start — should not raise
        callback.on_step_error("unknown", RuntimeError("x"))

    def test_pipeline_end_without_start(self) -> None:
        callback = TracingCallback()
        pipeline = ContextPipeline(max_tokens=1000, tokenizer=FakeTokenizer())
        result = pipeline.build("test")
        # Calling on_pipeline_end without on_pipeline_start — should not raise
        callback.on_pipeline_end(result)

    def test_custom_tracer(self) -> None:
        tracer = Tracer()
        callback = TracingCallback(tracer=tracer)
        assert callback.tracer is tracer

    def test_last_trace_initially_none(self) -> None:
        callback = TracingCallback()
        assert callback.last_trace is None
