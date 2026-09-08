"""Custom exceptions for anchor."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AstroContextError",
    "FormatterError",
    "IngestionError",
    "PipelineExecutionError",
    "RetrieverError",
    "StorageError",
]


class AstroContextError(Exception):
    """Base exception for all anchor errors."""


class PipelineExecutionError(AstroContextError):
    """Error during pipeline execution with partial diagnostics attached."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}



class RetrieverError(AstroContextError):
    """Raised when a retriever encounters an error."""


class StorageError(AstroContextError):
    """Raised when a storage backend encounters an error."""


class FormatterError(AstroContextError):
    """Raised when formatting context fails."""


class IngestionError(AstroContextError):
    """Raised when document ingestion fails."""
