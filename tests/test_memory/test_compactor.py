"""Tests for anchor.memory.compactor (TierCompactor)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.compactor import TierCompactor
from anchor.models.memory import FactType
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_compactor(response_content: str = "Summary text") -> tuple[TierCompactor, MagicMock]:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_llm_response(response_content)
    mock_llm.model_id = "test/model"
    mock_llm.provider_name = "test"
    compactor = TierCompactor(llm=mock_llm, tokenizer=FakeTokenizer())
    return compactor, mock_llm


class TestTierCompactorSummarize:
    def test_summarize_tier1(self) -> None:
        compactor, mock_llm = _make_compactor("Detailed summary of conversation")
        result = compactor.summarize("user: hello\nassistant: hi", target_tier=1, target_tokens=500)
        assert result == "Detailed summary of conversation"
        mock_llm.invoke.assert_called_once()
        # Check prompt contains target tokens
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        assert any("500" in str(m.content) for m in messages)

    def test_summarize_tier2(self) -> None:
        compactor, _mock_llm = _make_compactor("Compact summary")
        result = compactor.summarize("Detailed summary text", target_tier=2, target_tokens=100)
        assert result == "Compact summary"

    def test_summarize_tier3(self) -> None:
        compactor, _mock_llm = _make_compactor("Headline")
        result = compactor.summarize("Compact text", target_tier=3, target_tokens=20)
        assert result == "Headline"

    def test_progressive_merge(self) -> None:
        compactor, mock_llm = _make_compactor("Merged summary")
        result = compactor.summarize(
            "New content",
            target_tier=1,
            target_tokens=500,
            existing_summary="Old summary",
        )
        assert result == "Merged summary"
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        prompt_text = str(messages[0].content)
        assert "Old summary" in prompt_text
        assert "New content" in prompt_text

    def test_summarize_fallback_on_error(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.invoke.side_effect = Exception("LLM error")
        result = compactor.summarize("Some content here", target_tier=1, target_tokens=500)
        # Falls back to raw content (possibly truncated)
        assert "Some content here" in result


class TestTierCompactorFactExtraction:
    def test_extract_valid_facts(self) -> None:
        json_response = (
            '[{"type": "decision", "content": "Use FastAPI"},'
            ' {"type": "number", "content": "Budget: $50k"}]'
        )
        compactor, _mock_llm = _make_compactor(json_response)
        facts = compactor.extract_facts("Some conversation", source_tier=0)
        assert len(facts) == 2
        assert facts[0].fact_type == FactType.DECISION
        assert facts[0].content == "Use FastAPI"
        assert facts[0].source_tier == 0
        assert facts[1].fact_type == FactType.NUMBER

    def test_extract_empty_array(self) -> None:
        compactor, _ = _make_compactor("[]")
        facts = compactor.extract_facts("Some text", source_tier=1)
        assert facts == []

    def test_extract_invalid_json_retries(self) -> None:
        compactor, mock_llm = _make_compactor("not json")
        # First call returns bad JSON, retry also returns bad JSON
        mock_llm.invoke.side_effect = [
            _make_llm_response("not json"),
            _make_llm_response("still not json"),
        ]
        facts = compactor.extract_facts("text", source_tier=0)
        assert facts == []

    def test_extract_filters_invalid_fact_types(self) -> None:
        json_response = (
            '[{"type": "invalid_type", "content": "x"}, {"type": "decision", "content": "y"}]'
        )
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        assert len(facts) == 1
        assert facts[0].fact_type == FactType.DECISION

    def test_extract_strips_markdown_fences(self) -> None:
        json_response = '```json\n[{"type": "entity", "content": "FastAPI"}]\n```'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        assert len(facts) == 1

    def test_source_tier_injected(self) -> None:
        json_response = '[{"type": "date", "content": "March 2026"}]'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=2)
        assert facts[0].source_tier == 2

    def test_token_count_computed(self) -> None:
        json_response = '[{"type": "decision", "content": "Use FastAPI over Flask"}]'
        compactor, _ = _make_compactor(json_response)
        facts = compactor.extract_facts("text", source_tier=0)
        # FakeTokenizer counts words: "Use FastAPI over Flask" = 4 tokens
        assert facts[0].token_count == 4

    def test_extract_llm_failure_returns_empty(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.invoke.side_effect = Exception("LLM down")
        facts = compactor.extract_facts("text", source_tier=0)
        assert facts == []


class TestTierCompactorAsync:
    def test_asummarize(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response("Async summary"))
        result = asyncio.run(
            compactor.asummarize("content", target_tier=1, target_tokens=500)
        )
        assert result == "Async summary"
        mock_llm.ainvoke.assert_awaited_once()

    def test_aextract_facts(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_llm_response('[{"type": "decision", "content": "test"}]')
        )
        facts = asyncio.run(
            compactor.aextract_facts("content", source_tier=1)
        )
        assert len(facts) == 1
        mock_llm.ainvoke.assert_awaited_once()

    def test_asummarize_fallback(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("fail"))
        result = asyncio.run(
            compactor.asummarize("fallback content", target_tier=1, target_tokens=500)
        )
        assert "fallback" in result
