"""Tests for anchor.llm.pricing."""

from __future__ import annotations

import pytest

from anchor.llm.pricing import (
    MODEL_PRICING,
    _normalize_model_name,
    calculate_cost,
    estimate_round_cost,
)


class TestCalculateCost:
    def test_known_model(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == 2.50  # $2.50 per 1M input tokens

    def test_known_model_output(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == 10.0  # $10 per 1M output tokens

    def test_mixed_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=500_000, completion_tokens=100_000)
        assert cost == pytest.approx(1.25 + 1.0)

    def test_unknown_model_returns_none(self):
        cost = calculate_cost("unknown-model-xyz", prompt_tokens=100, completion_tokens=50)
        assert cost is None

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_alias_normalization(self):
        """Model with date suffix should match base model pricing."""
        cost = calculate_cost("gpt-4o-2024-08-06", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == 2.50

    def test_anthropic_model(self):
        cost = calculate_cost(
            "claude-haiku-4-5-20251001",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert cost == 0.80


class TestNormalizeModelName:
    def test_strips_date_suffix_dashes(self):
        assert _normalize_model_name("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_date_suffix_no_dashes(self):
        assert _normalize_model_name("model-20240806") == "model"

    def test_preserves_non_date_suffix(self):
        assert _normalize_model_name("gpt-4o-mini") == "gpt-4o-mini"

    def test_preserves_canonical_anthropic(self):
        result = _normalize_model_name("claude-sonnet-4-20250514")
        assert isinstance(result, str)


class TestEstimateRoundCost:
    """estimate_round_cost — the agent loop's single price entrypoint."""

    def test_model_pricing_table_wins(self):
        cost = estimate_round_cost(
            "openai/gpt-4o", prompt_tokens=1_000_000, completion_tokens=0,
        )
        assert cost == 2.50

    def test_cache_tokens_price_and_never_raise(self):
        # genai-prices counts cache tokens as a subset of input_tokens;
        # anchor's counts exclude them. The mapping must reconcile the
        # two instead of raising ValueError on any cached round.
        cost = estimate_round_cost(
            "anthropic/claude-haiku-4-5",
            prompt_tokens=100,
            completion_tokens=100,
            cache_read_tokens=20_000,
        )
        assert cost is not None
        assert cost > 0

    def test_multi_segment_model_id_uses_provider(self):
        # litellm/openrouter/... ids must not collapse to a $0 miss.
        cost = estimate_round_cost(
            "litellm/openrouter/anthropic/claude-3.5-sonnet",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self):
        cost = estimate_round_cost(
            "mock/model-that-cannot-exist-xyz",
            prompt_tokens=100,
            completion_tokens=100,
        )
        assert cost is None

    def test_runtime_override_reaches_the_estimate(self):
        MODEL_PRICING["my-house-model"] = {"input": 1.0, "output": 2.0}
        try:
            cost = estimate_round_cost(
                "custom/my-house-model",
                prompt_tokens=1_000_000,
                completion_tokens=0,
            )
        finally:
            del MODEL_PRICING["my-house-model"]
        assert cost == 1.0
