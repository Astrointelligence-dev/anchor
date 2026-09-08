"""Integration tests for ProgressiveSummarizationMemory with MemoryManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.manager import MemoryManager
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.memory import TierConfig
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.invoke.side_effect = lambda msgs, **kw: _make_llm_response(
        "[]" if "Extract key facts" in str(msgs[0].content) else "Summary"
    )
    mock.model_id = "test/model"
    mock.provider_name = "test"
    return mock


class TestMemoryManagerIntegration:
    def test_add_messages_via_manager(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        assert len(mem.turns) == 2

    def test_conversation_type(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        assert manager.conversation_type == "progressive_summarization"

    def test_get_context_items(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        manager.add_user_message("Hello")
        items = manager.get_context_items()
        assert len(items) >= 1

    def test_full_conversation_with_cascade(self) -> None:
        mock_llm = _make_mock_llm()
        config = [
            TierConfig(level=0, max_tokens=10),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        manager = MemoryManager(conversation_memory=mem, tokenizer=FakeTokenizer())
        # Add 20 turns
        for i in range(10):
            manager.add_user_message(f"User message number {i}")
            manager.add_assistant_message(f"Reply to message {i}")
        # Should have cascaded
        items = manager.get_context_items()
        assert len(items) > 0
