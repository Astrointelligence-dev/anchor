"""Tests for progressive summarization data models and memory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.callbacks import ProgressiveSummarizationCallback
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.context import SourceType
from anchor.models.memory import FactType, KeyFact, SummaryTier, TierConfig
from anchor.protocols.memory import ConversationMemory
from tests.conftest import FakeTokenizer


class TestFactType:
    def test_all_values(self) -> None:
        assert FactType.DECISION == "decision"
        assert FactType.ENTITY == "entity"
        assert FactType.NUMBER == "number"
        assert FactType.DATE == "date"
        assert FactType.PREFERENCE == "preference"
        assert FactType.CONSTRAINT == "constraint"


class TestKeyFact:
    def test_create_minimal(self) -> None:
        fact = KeyFact(fact_type=FactType.DECISION, content="Use FastAPI", source_tier=0)
        assert fact.fact_type == FactType.DECISION
        assert fact.content == "Use FastAPI"
        assert fact.source_tier == 0
        assert fact.id  # auto-generated UUID
        assert fact.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            KeyFact(fact_type=FactType.NUMBER, content="x", source_tier=0, token_count=-1)


class TestSummaryTier:
    def test_create(self) -> None:
        tier = SummaryTier(level=1, content="Summary text", source_turn_count=5)
        assert tier.level == 1
        assert tier.content == "Summary text"
        assert tier.source_turn_count == 5
        assert tier.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SummaryTier(level=1, content="x", source_turn_count=1, token_count=-1)


class TestTierConfig:
    def test_create_default(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        assert config.level == 0
        assert config.max_tokens == 4096
        assert config.target_tokens == 0
        assert config.priority == 7

    def test_frozen(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        with pytest.raises(AttributeError):
            config.level = 1  # type: ignore[misc]


class TestProgressiveSummarizationCallback:
    def test_protocol_exists(self) -> None:
        assert hasattr(ProgressiveSummarizationCallback, 'on_tier_cascade')
        assert hasattr(ProgressiveSummarizationCallback, 'on_facts_extracted')
        assert hasattr(ProgressiveSummarizationCallback, 'on_compaction_error')

    def test_satisfies_protocol(self) -> None:
        class MyCallback:
            def on_tier_cascade(self, from_tier, to_tier, tokens_in, tokens_out):
                pass
            def on_facts_extracted(self, facts, source_tier):
                pass
            def on_compaction_error(self, tier, error):
                pass

        assert isinstance(MyCallback(), ProgressiveSummarizationCallback)


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_mock_llm(summary: str = "Summary", facts: str = "[]") -> MagicMock:
    mock = MagicMock()
    mock.invoke.side_effect = lambda msgs, **kw: _make_llm_response(
        facts if "Extract key facts" in str(msgs[0].content) else summary
    )
    mock.model_id = "test/model"
    mock.provider_name = "test"
    return mock


class TestProgressiveConstruction:
    def test_default_config(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert mem.turns == []
        assert mem.total_tokens == 0
        assert mem.summary is None
        assert mem.facts == []

    def test_custom_tier_config(self) -> None:
        mock_llm = _make_mock_llm()
        config = [
            TierConfig(level=0, max_tokens=100),
            TierConfig(level=1, max_tokens=50, target_tokens=25),
            TierConfig(level=2, max_tokens=20, target_tokens=10),
            TierConfig(level=3, max_tokens=10, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        assert mem.tier_tokens == {0: 0, 1: 0, 2: 0, 3: 0}

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            ProgressiveSummarizationMemory(max_tokens=-1, llm=_make_mock_llm())

    def test_satisfies_conversation_memory_protocol(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert isinstance(mem, ConversationMemory)

    def test_clear_resets_everything(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=100, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        mem.add_message("user", "hello world")
        assert len(mem.turns) == 1
        mem.clear()
        assert mem.turns == []
        assert mem.facts == []
        assert mem.summary is None


class TestProgressiveCascade:
    def test_tier1_created_on_overflow(self) -> None:
        """When Tier 0 overflows, evicted turns should create a Tier 1 summary."""
        mock_llm = _make_mock_llm(summary="Tier 1 summary")
        # FakeTokenizer: 1 token per word. max_tokens=10 for Tier 0.
        config = [
            TierConfig(level=0, max_tokens=10),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Add messages until overflow (each ~3 tokens: "word1 word2 word3")
        for i in range(5):
            mem.add_message("user", f"message number {i}")
        # Tier 1 should now have content
        assert mem.tiers[1] is not None
        assert mem.tiers[1].content == "Tier 1 summary"

    def test_tier2_created_on_tier1_overflow(self) -> None:
        """When Tier 1 overflows, content should cascade to Tier 2."""
        call_count = 0
        def smart_response(msgs, **kw):
            nonlocal call_count
            call_count += 1
            prompt = str(msgs[0].content)
            if "Extract key facts" in prompt:
                return _make_llm_response("[]")
            # Return progressively longer summaries to trigger overflow
            return _make_llm_response("word " * 20)  # 20 tokens

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = smart_response
        mock_llm.model_id = "test/model"
        mock_llm.provider_name = "test"

        config = [
            TierConfig(level=0, max_tokens=5),   # very small window
            TierConfig(level=1, max_tokens=15, target_tokens=10),  # overflow at 15
            TierConfig(level=2, max_tokens=100, target_tokens=50),
            TierConfig(level=3, max_tokens=50, target_tokens=10),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Add many messages to trigger multiple evictions
        for i in range(10):
            mem.add_message("user", f"word{i} word{i}")
        # Since Tier 1 summary is 20 tokens and max is 15, it should cascade
        # Tier 1 should be cleared (set to None) after cascade
        # Tier 2 should have content
        assert mem.tiers[2] is not None

    def test_facts_extracted_during_cascade(self) -> None:
        facts_json = '[{"type": "decision", "content": "Use Python"}]'
        mock_llm = _make_mock_llm(summary="Summary", facts=facts_json)
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")
        assert len(mem.facts) > 0
        assert mem.facts[0].content == "Use Python"

    def test_facts_fifo_eviction(self) -> None:
        facts_json = '[{"type": "decision", "content": "fact"}]'
        mock_llm = _make_mock_llm(summary="Summary", facts=facts_json)
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=1000, target_tokens=50),
            TierConfig(level=2, max_tokens=500, target_tokens=20),
            TierConfig(level=3, max_tokens=100, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=2000, llm=mock_llm, tier_config=config,
            max_facts=3, tokenizer=FakeTokenizer(),
        )
        # Trigger many evictions to produce many facts
        for i in range(20):
            mem.add_message("user", f"word{i} word{i}")
        assert len(mem.facts) <= 3


class TestProgressiveContextOutput:
    def test_context_items_priorities(self) -> None:
        mock_llm = _make_mock_llm(summary="Summary text here")
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        # Trigger a cascade
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")

        items = mem.to_context_items(priority=7)
        # Should have tier 1 summary + verbatim turns
        summary_items = [i for i in items if i.metadata.get("summary")]
        verbatim_items = [
            i
            for i in items
            if not i.metadata.get("summary") and i.source == SourceType.CONVERSATION
        ]
        assert verbatim_items
        [i for i in items if not i.metadata.get("summary") and i.source == SourceType.CONVERSATION]

        # Tier 1 should be at priority 6
        if summary_items:
            assert summary_items[0].priority == 6

    def test_empty_tiers_omitted(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        mem.add_message("user", "hello")
        items = mem.to_context_items()
        summary_items = [i for i in items if i.metadata.get("summary")]
        assert len(summary_items) == 0  # No tiers populated yet

    def test_relative_priority(self) -> None:
        mock_llm = _make_mock_llm(summary="Summary")
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        for i in range(5):
            mem.add_message("user", f"word{i} word{i}")
        # Call with custom priority
        items = mem.to_context_items(priority=5)
        summary_items = [i for i in items if i.metadata.get("summary")]
        if summary_items:
            # Tier 1 at priority=5-1=4
            assert summary_items[0].priority == 4


class TestProgressiveAddTurn:
    def test_add_turn_with_conversation_turn(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        from anchor.models.memory import ConversationTurn
        turn = ConversationTurn(role="user", content="hello world", token_count=2)
        mem.add_turn(turn)
        assert len(mem.turns) == 1
        assert mem.turns[0].content == "hello world"


class TestProgressiveAsync:
    def test_aadd_message_uses_async_compaction(self) -> None:
        """aadd_message should use async LLM calls for compaction."""
        import asyncio
        from unittest.mock import AsyncMock

        mock_llm = _make_mock_llm(summary="Async summary")
        mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response("Async summary"))
        config = [
            TierConfig(level=0, max_tokens=5),
            TierConfig(level=1, max_tokens=100, target_tokens=50),
            TierConfig(level=2, max_tokens=50, target_tokens=20),
            TierConfig(level=3, max_tokens=20, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )

        async def run():
            for i in range(5):
                await mem.aadd_message("user", f"word{i} word{i}")

        asyncio.run(run())
        # Should have used ainvoke (async) not invoke (sync) for compaction
        mock_llm.ainvoke.assert_awaited()

    def test_aadd_turn_delegates(self) -> None:
        import asyncio
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        from anchor.models.memory import ConversationTurn
        turn = ConversationTurn(role="user", content="async turn", token_count=2)
        asyncio.run(mem.aadd_turn(turn))
        assert len(mem.turns) == 1


class TestProgressiveThreadSafety:
    def test_concurrent_add_message_no_corruption(self) -> None:
        """Concurrent add_message calls should not corrupt state."""
        import threading

        mock_llm = _make_mock_llm(summary="Summary")
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        errors: list[Exception] = []

        def add_messages(start: int) -> None:
            try:
                for i in range(10):
                    mem.add_message("user", f"thread-{start}-msg-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mem.turns) == 30
