"""Built-in model pricing for cost tracking.

Prices are best-effort and ship with the package. Users can override
at runtime via MODEL_PRICING["model"] = {"input": X, "output": Y}.

Last updated: 2026-03-13
Prices in USD per 1M tokens.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # xAI (Grok)
    "grok-3": {"input": 3.0, "output": 15.0},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
}


def calculate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Calculate USD cost. Returns None if model pricing unknown.

    Tries exact match first, then strips date suffixes for alias matching.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        normalized = _normalize_model_name(model)
        pricing = MODEL_PRICING.get(normalized)
    if pricing is None:
        return None
    return (
        prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    ) / 1_000_000


def _normalize_model_name(model: str) -> str:
    """Strip trailing date suffixes for alias matching.

    'gpt-4o-2024-08-06' -> 'gpt-4o'
    'model-20240806' -> 'model'
    """
    return re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", model)


def estimate_round_cost(
    model_id: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    """USD price of one model round, or None when no source knows the model.

    The single price entrypoint for the agent loop's ``cost_limit``.
    ``model_id`` is the provider convention ``"provider/model"`` (a bare
    model name works too). Resolution order:

    1. :data:`MODEL_PRICING` (runtime-overridable) — cache tokens are
       priced at the full input rate, a conservative approximation.
    2. genai-prices, when installed (``astro-anchor[pricing]``) — full
       cache-aware rates, tried with the provider id first and bare
       second.

    Never raises: any lookup failure means None. Note genai-prices
    counts cache tokens as a subset of ``input_tokens``, while anchor's
    counts (Anthropic convention) exclude them — the mapping here adds
    them back so cached rounds price correctly instead of erroring.
    """
    provider, sep, model = model_id.partition("/")
    if not sep:
        provider, model = "", model_id
    priced = calculate_cost(
        model,
        prompt_tokens + cache_creation_tokens + cache_read_tokens,
        completion_tokens,
    )
    if priced is not None:
        return priced
    try:
        from genai_prices import Usage as PriceUsage
        from genai_prices import calc_price
    except ImportError:
        return None
    usage = PriceUsage(
        input_tokens=prompt_tokens + cache_creation_tokens + cache_read_tokens,
        output_tokens=completion_tokens,
        cache_write_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )

    def attempt(provider_id: str | None) -> float | None:
        try:
            return float(
                calc_price(usage, model_ref=model, provider_id=provider_id)
                .total_price,
            )
        except Exception as exc:  # a price lookup must never kill a turn
            logger.debug("genai-prices lookup failed for %r: %s", model_id, exc)
            return None

    cost = attempt(provider) if provider else None
    if cost is None:
        cost = attempt(None)
    return cost
