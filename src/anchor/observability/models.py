"""Pydantic models for observability: spans, traces, and metrics."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpanKind(StrEnum):
    """The kind of operation being traced."""

    PIPELINE = "pipeline"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    FORMATTING = "formatting"
    MEMORY = "memory"
    INGESTION = "ingestion"
    QUERY_TRANSFORM = "query_transform"
    TOOL = "tool"


class Span(BaseModel):
    """A single traced operation within a pipeline execution.

    Spans form a tree structure via ``parent_span_id`` and capture timing,
    status, and arbitrary attributes for a discrete unit of work.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    trace_id: str
    name: str
    kind: SpanKind
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class TraceRecord(BaseModel):
    """A complete trace of a pipeline execution.

    Groups all spans belonging to a single pipeline ``build()`` or
    ``abuild()`` invocation under a shared ``trace_id``.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[Span] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    """A single metric measurement at a point in time.

    Parameters:
        name: The metric name (e.g. ``"pipeline.build_time_ms"``).
        value: The numeric measurement value.
        timestamp: When the measurement was taken.
        tags: Arbitrary key-value labels for filtering and grouping.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: dict[str, str] = Field(default_factory=dict)
