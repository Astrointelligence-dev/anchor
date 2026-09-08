"""Error hierarchy for the multi-provider LLM layer.

Each provider adapter maps SDK-specific exceptions to these.
The `is_transient` flag drives retry and fallback behavior.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for all provider runtime failures."""

    def __init__(self, message: str, *, provider: str, is_transient: bool = False):
        super().__init__(message)
        self.provider = provider
        self.is_transient = is_transient


class AuthenticationError(ProviderError):
    """Invalid or missing API key."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class RateLimitError(ProviderError):
    """Rate limit exceeded (429). Transient — retry after backoff."""

    def __init__(
        self, message: str, *, provider: str, retry_after: float | None = None
    ):
        super().__init__(message, provider=provider, is_transient=True)
        self.retry_after = retry_after


class ServerError(ProviderError):
    """Provider server error (5xx). Transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=True)


class LLMTimeoutError(ProviderError):
    """Request timed out. Transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=True)


class ModelNotFoundError(ProviderError):
    """Model does not exist or is not available."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class ContentFilterError(ProviderError):
    """Response blocked by content filter. Not transient."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, is_transient=False)


class ProviderNotInstalledError(Exception):
    """SDK for a provider is not installed. Setup error, not runtime."""

    def __init__(self, provider: str, package: str, extra: str):
        super().__init__(
            f"{provider} provider requires the '{package}' package. "
            f"Install with: pip install anchor[{extra}]"
        )
