"""Pipeline callback that automatically traces pipeline execution."""

from __future__ import annotations

import logging

from anchor.models.context import ContextItem, ContextResult
from anchor.models.query import QueryBundle
from anchor.observability.models import MetricPoint, Span, SpanKind, TraceRecord
from anchor.observability.tracer import Tracer
from anchor.protocols.observability import MetricsCollector, SpanExporter

logger = logging.getLogger(__name__)


class TracingCallback:
    """A ``PipelineCallback`` that traces pipeline execution via spans.

    Hooks into the pipeline lifecycle to automatically create spans for the
    overall pipeline execution and each individual step.  Completed spans
    are forwarded to any registered ``SpanExporter`` implementations, and
    timing metrics are recorded on the optional ``MetricsCollector``.

    Usage::

        from anchor.observability import TracingCallback, InMemorySpanExporter

        exporter = InMemorySpanExporter()
        callback = TracingCallback(exporters=[exporter])
        pipeline.add_callback(callback)

        result = pipeline.build("What is context engineering?")
        spans = exporter.get_spans()  # inspect recorded spans

    Parameters:
        tracer: An optional ``Tracer`` instance.  A new one is created if
            not provided.
        exporters: Span exporters to receive completed spans.
        metrics_collector: Optional collector for timing metrics.
    """

    __slots__ = (
        "_active_trace",
        "_exporters",
        "_metrics_collector",
        "_pipeline_span",
        "_step_spans",
        "_tracer",
    )

    def __init__(
        self,
        tracer: Tracer | None = None,
        exporters: list[SpanExporter] | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._tracer = tracer or Tracer()
        self._exporters: list[SpanExporter] = exporters or []
        self._metrics_collector = metrics_collector
        self._active_trace: TraceRecord | None = None
        self._pipeline_span: Span | None = None
        self._step_spans: dict[str, Span] = {}

    @property
    def tracer(self) -> Tracer:
        """The underlying tracer instance."""
        return self._tracer

    @property
    def last_trace(self) -> TraceRecord | None:
        """The most recently completed trace, or ``None``."""
        return self._active_trace

    # -- PipelineCallback interface --

    def on_pipeline_start(self, query: QueryBundle) -> None:
        """Begin a new trace and root span for the pipeline execution.

        Parameters:
            query: The query being processed.
        """
        trace = self._tracer.start_trace(
            name="pipeline.build",
            attributes={"query": query.query_str},
        )
        self._active_trace = trace
        self._step_spans.clear()

        self._pipeline_span = self._tracer.start_span(
            trace_id=trace.trace_id,
            name="pipeline",
            kind=SpanKind.PIPELINE,
            attributes={"query": query.query_str},
        )

    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None:
        """Create a span for a pipeline step.

        Parameters:
            step_name: The name of the step being started.
            items: The context items entering the step.
        """
        if self._active_trace is None:
            return

        parent_id = self._pipeline_span.span_id if self._pipeline_span else None
        kind = _infer_span_kind(step_name)

        span = self._tracer.start_span(
            trace_id=self._active_trace.trace_id,
            name=step_name,
            kind=kind,
            parent_span_id=parent_id,
            attributes={"items_in": len(items)},
        )
        self._step_spans[step_name] = span

    def on_step_end(
        self, step_name: str, items: list[ContextItem], time_ms: float
    ) -> None:
        """End the span for a completed step and record metrics.

        Parameters:
            step_name: The name of the completed step.
            items: The context items after the step.
            time_ms: The step execution time in milliseconds.
        """
        span = self._step_spans.pop(step_name, None)
        if span is None:
            return

        self._tracer.end_span(
            span,
            status="ok",
            attributes={"items_out": len(items), "time_ms": time_ms},
        )

        if self._metrics_collector is not None:
            self._metrics_collector.record(
                MetricPoint(
                    name="step.duration_ms",
                    value=time_ms,
                    tags={"step": step_name},
                ),
            )

    def on_step_error(self, step_name: str, error: Exception) -> None:
        """End the span for a failed step with error status.

        Parameters:
            step_name: The name of the failed step.
            error: The exception that was raised.
        """
        span = self._step_spans.pop(step_name, None)
        if span is None:
            return

        self._tracer.end_span(
            span,
            status="error",
            attributes={"error": str(error)},
        )

    def on_pipeline_end(self, result: ContextResult) -> None:
        """End the pipeline span and trace, then export all spans.

        Parameters:
            result: The pipeline result with diagnostics.
        """
        if self._active_trace is None:
            return

        trace_id = self._active_trace.trace_id

        # End any remaining step spans (shouldn't happen normally)
        for _step_name, span in list(self._step_spans.items()):
            self._tracer.end_span(span, status="error", attributes={"error": "orphaned"})
        self._step_spans.clear()

        # End the root pipeline span
        if self._pipeline_span is not None:
            self._tracer.end_span(
                self._pipeline_span,
                status="ok",
                attributes={
                    "items_included": result.diagnostics.get("items_included", 0),
                    "build_time_ms": result.build_time_ms,
                    "token_utilization": result.diagnostics.get("token_utilization", 0.0),
                },
            )
            self._pipeline_span = None

        # Fetch the latest trace state (with all accumulated spans) before ending
        current_trace = self._tracer.get_trace(trace_id) or self._active_trace
        ended_trace = self._tracer.end_trace(current_trace)
        self._active_trace = ended_trace

        # Export spans
        if ended_trace.spans:
            for exporter in self._exporters:
                try:
                    exporter.export(ended_trace.spans)
                except Exception:
                    logger.warning(
                        "SpanExporter %r failed", exporter, exc_info=True,
                    )

        # Record pipeline-level metrics
        if self._metrics_collector is not None:
            self._metrics_collector.record(
                MetricPoint(
                    name="pipeline.build_time_ms",
                    value=result.build_time_ms,
                ),
            )
            self._metrics_collector.record(
                MetricPoint(
                    name="pipeline.items_included",
                    value=float(result.diagnostics.get("items_included", 0)),
                ),
            )
            self._metrics_collector.flush()


class TracingAgentCallback:
    """An ``AgentCallback`` that records one span per tool execution.

    Attach with ``agent.with_callbacks([TracingAgentCallback(...)])``.
    Each tool call gets its own short trace, ended and exported on
    completion — nothing accumulates on long-lived agents.
    """

    __slots__ = ("_exporters", "_tool_spans", "_tracer")

    def __init__(
        self,
        tracer: Tracer | None = None,
        exporters: list[SpanExporter] | None = None,
    ) -> None:
        self._tracer = tracer or Tracer()
        self._exporters: list[SpanExporter] = exporters or []
        self._tool_spans: dict[str, tuple[TraceRecord, Span]] = {}

    @property
    def tracer(self) -> Tracer:
        """The underlying tracer instance."""
        return self._tracer

    def on_tool_start(self, name: str, tool_input: dict[str, object]) -> None:
        trace = self._tracer.start_trace(name=f"tool.{name}")
        span = self._tracer.start_span(
            trace_id=trace.trace_id,
            name=f"tool.{name}",
            kind=SpanKind.TOOL,
            attributes={"tool_input_keys": sorted(tool_input)},
        )
        self._tool_spans[name] = (trace, span)

    def on_tool_end(
        self, name: str, tool_input: dict[str, object], result: str,
    ) -> None:
        self._finish(name, status="ok", attributes={"result_chars": len(result)})

    def on_tool_error(
        self, name: str, tool_input: dict[str, object], error: str,
    ) -> None:
        self._finish(name, status="error", attributes={"error": error})

    def _finish(
        self, name: str, *, status: str, attributes: dict[str, object],
    ) -> None:
        # ponytail: spans keyed by tool name — parallel calls to the SAME tool
        # share one span; key by call id if that ever matters.
        entry = self._tool_spans.pop(name, None)
        if entry is None:
            return
        trace, span = entry
        completed = self._tracer.end_span(span, status=status, attributes=attributes)
        self._tracer.end_trace(trace)
        for exporter in self._exporters:
            try:
                exporter.export([completed])
            except Exception:
                logger.exception("Span exporter failed")


def _infer_span_kind(step_name: str) -> SpanKind:
    """Infer the ``SpanKind`` from a step name using simple heuristics.

    Parameters:
        step_name: The pipeline step name.

    Returns:
        The most likely ``SpanKind`` for the step.
    """
    lower = step_name.lower()
    if "retriev" in lower or "search" in lower or "fetch" in lower:
        return SpanKind.RETRIEVAL
    if "rerank" in lower or "rank" in lower:
        return SpanKind.RERANKING
    if "format" in lower:
        return SpanKind.FORMATTING
    if "memory" in lower or "history" in lower:
        return SpanKind.MEMORY
    if "ingest" in lower or "index" in lower:
        return SpanKind.INGESTION
    if "transform" in lower or "expand" in lower or "query" in lower:
        return SpanKind.QUERY_TRANSFORM
    return SpanKind.PIPELINE
