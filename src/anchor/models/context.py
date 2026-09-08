"""Core context models for anchor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from anchor.models.scope import DEFAULT_VAULT, ROOT_NAMESPACE, normalize_namespace, validate_vault


class StepDiagnostic(TypedDict):
    """Diagnostics for a single pipeline step."""

    name: str
    items_after: int
    time_ms: float


class PipelineDiagnostics(TypedDict, total=False):
    """Typed schema for the diagnostics dict produced by the context pipeline."""

    steps: list[StepDiagnostic]
    memory_items: int
    total_items_considered: int
    items_included: int
    items_overflow: int
    token_utilization: float
    token_usage_by_source: dict[str, int]
    query_enriched: bool
    skipped_steps: list[str]
    failed_step: str
    budget_overflow_by_source: dict[str, int]
    shared_pool_usage: int


class SourceType(StrEnum):
    """The origin type of a context item."""

    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    SYSTEM = "system"
    USER = "user"
    TOOL = "tool"
    CONVERSATION = "conversation"


class ContextItem(BaseModel):
    """A single unit of context to be included in an LLM prompt.

    This is the atomic unit that flows through the pipeline.
    Items are immutable after creation to prevent context poisoning bugs.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: SourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Front #3: vault = mount the item lives in (hard boundary, stamped
    # by the store layer); namespace = hierarchical path inside it.
    vault: str = DEFAULT_VAULT
    namespace: str = ROOT_NAMESPACE

    model_config = ConfigDict(frozen=True)

    @field_validator("vault")
    @classmethod
    def _validate_vault(cls, v: str) -> str:
        return validate_vault(v)

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, ns: str) -> str:
        return normalize_namespace(ns)


class ContextWindow(BaseModel):
    """A complete context window ready for formatting.

    Token-aware and priority-ranked. Items are added respecting the
    token budget, with higher-priority items placed first.
    """

    items: list[ContextItem] = Field(default_factory=list)
    max_tokens: int = Field(default=8192, gt=0)
    used_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def utilization(self) -> float:
        """Fraction of token budget used (0.0 to 1.0)."""
        if self.max_tokens == 0:
            return 0.0
        return min(1.0, self.used_tokens / self.max_tokens)

    def add_item(self, item: ContextItem) -> bool:
        """Add an item if it fits within the token budget. Returns True if added."""
        if item.token_count > self.remaining_tokens:
            return False
        self.items.append(item)
        self.used_tokens += item.token_count
        return True

    def __len__(self) -> int:
        return len(self.items)

    def add_items_by_priority(self, items: list[ContextItem]) -> list[ContextItem]:
        """Add items sorted by priority (highest first), return items that didn't fit."""
        sorted_items = sorted(items, key=lambda x: (-x.priority, -x.score))
        overflow: list[ContextItem] = []
        for item in sorted_items:
            if not self.add_item(item):
                overflow.append(item)
        return overflow


class ContextResult(BaseModel):
    """The final output of the context pipeline.

    Contains the assembled window, formatted output, and diagnostics.
    """

    window: ContextWindow
    formatted_output: str | dict[str, Any] = ""
    format_type: str = "generic"
    overflow_items: list[ContextItem] = Field(default_factory=list)
    diagnostics: PipelineDiagnostics = Field(default_factory=dict)  # type: ignore[assignment]
    build_time_ms: float = Field(default=0.0, ge=0.0)
