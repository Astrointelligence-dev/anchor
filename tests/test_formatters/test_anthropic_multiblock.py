"""Tests for AnthropicFormatter multi-block system support.

Verifies that multiple system items produce individual content blocks
with correct cache_control placement on the last block only.
"""

from __future__ import annotations

from anchor.formatters.anthropic import AnthropicFormatter
from anchor.models.context import ContextItem, ContextWindow, SourceType


class TestMultiBlockSystemBlocks:
    """Multiple system items produce separate content blocks."""

    def test_single_system_item_produces_single_block(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="You are a helpful assistant.",
                source=SourceType.SYSTEM,
                token_count=5,
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert len(output["system"]) == 1
        assert output["system"][0] == {"type": "text", "text": "You are a helpful assistant."}

    def test_multiple_system_items_produce_multiple_blocks(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System instruction A.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="System instruction B.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="System instruction C.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert len(output["system"]) == 3
        assert output["system"][0] == {"type": "text", "text": "System instruction A."}
        assert output["system"][1] == {"type": "text", "text": "System instruction B."}
        assert output["system"][2] == {"type": "text", "text": "System instruction C."}

    def test_empty_system_parts_produce_empty_system_list(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert output["system"] == []

    def test_system_blocks_preserve_order(self) -> None:
        window = ContextWindow(max_tokens=10000)
        texts = ["First", "Second", "Third"]
        for t in texts:
            window.add_item(
                ContextItem(content=t, source=SourceType.SYSTEM, token_count=1)
            )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        result_texts = [b["text"] for b in output["system"]]
        assert result_texts == texts


class TestMultiBlockCaching:
    """Cache control placed on last system block only when caching enabled."""

    def test_cache_control_on_last_block_only(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Block A", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Block B", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Block C", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        assert len(output["system"]) == 3
        # First two blocks should NOT have cache_control
        assert "cache_control" not in output["system"][0]
        assert "cache_control" not in output["system"][1]
        # Last block should have cache_control
        assert output["system"][2]["cache_control"] == {"type": "ephemeral"}

    def test_single_block_gets_cache_control_when_caching(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Only block", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        assert len(output["system"]) == 1
        assert output["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_when_caching_disabled(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Block A", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Block B", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=False)
        output = formatter.format(window)

        assert len(output["system"]) == 2
        assert "cache_control" not in output["system"][0]
        assert "cache_control" not in output["system"][1]

    def test_empty_system_no_cache_control(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        assert output["system"] == []

    def test_cache_control_with_messages(self) -> None:
        """Multi-block system with caching still works alongside messages."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System A", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="System B", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.CONVERSATION,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        # System: 2 blocks, cache_control on last only
        assert len(output["system"]) == 2
        assert "cache_control" not in output["system"][0]
        assert output["system"][1]["cache_control"] == {"type": "ephemeral"}

        # Messages: context (user) and conversation (user) merged into one
        assert len(output["messages"]) == 1
        block = output["messages"][0]["content"][0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert "Here is relevant context:" in block["text"]
        assert "Hello" in block["text"]
