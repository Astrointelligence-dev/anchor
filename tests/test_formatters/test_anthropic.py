"""Tests for anchor.formatters.anthropic."""

from __future__ import annotations

from anchor.formatters.anthropic import AnthropicFormatter
from anchor.models.context import ContextItem, ContextWindow, SourceType


class TestAnthropicFormatter:
    """AnthropicFormatter produces correct dict structure."""

    def test_format_type(self) -> None:
        formatter = AnthropicFormatter()
        assert formatter.format_type == "anthropic"

    def test_system_prompt(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert isinstance(output, dict)
        assert output["system"] == [{"type": "text", "text": "You are helpful."}]
        assert output["messages"] == []

    def test_multiple_system_prompts_separate_blocks(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Be concise.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["system"] == [
            {"type": "text", "text": "Be helpful."},
            {"type": "text", "text": "Be concise."},
        ]

    def test_memory_items_become_messages(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Hi there!",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][0]["content"] == "Hello"
        assert output["messages"][1]["role"] == "assistant"

    def test_retrieval_items_become_context_message(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert "doc about Python" in output["messages"][0]["content"]

    def test_context_message_merged_with_first_user_memory(self) -> None:
        """Context block (user) + first memory (user) are merged to avoid
        consecutive user messages that the Anthropic API would reject."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="user msg",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        # Both user messages merged into one
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert "user msg" in output["messages"][0]["content"]

    def test_context_before_assistant_memory_stays_separate(self) -> None:
        """Context block (user) + first memory (assistant) remain separate
        since they have different roles."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="assistant reply",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][0]["content"].startswith("Here is relevant context:")
        assert output["messages"][1]["role"] == "assistant"

    def test_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output == {"system": [], "messages": []}

    def test_memory_default_role_is_user(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="no role specified",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={},  # No role key
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["messages"][0]["role"] == "user"

    def test_full_structure(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System prompt", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="User says hi",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="A relevant document", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert output["system"] == [{"type": "text", "text": "System prompt"}]
        # Context block and user memory merged (both role="user")
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert "User says hi" in output["messages"][0]["content"]


class TestAnthropicFormatterCaching:
    """AnthropicFormatter with enable_caching=True adds cache_control."""

    def test_caching_adds_cache_control_to_system(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        assert output["system"] == [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_caching_adds_cache_control_to_context_message(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        # Valid wire format: cache_control on a content block, not the message
        block = output["messages"][0]["content"][0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert block["type"] == "text"

    def test_caching_disabled_no_cache_control(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=False)
        output = formatter.format(window)
        assert "cache_control" not in output["system"][0]
        assert "cache_control" not in output["messages"][0]

    def test_caching_full_structure(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System prompt", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="User says hi",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="A relevant document", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        # System block has cache_control
        assert output["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Context and user memory merged into one message (both user role);
        # cache_control on the merged message's content block.
        assert len(output["messages"]) == 1
        block = output["messages"][0]["content"][0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert "Here is relevant context:" in block["text"]
        assert "User says hi" in block["text"]

    def test_caching_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        assert output == {"system": [], "messages": []}

    def test_caching_no_context_only_system(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        # System has cache_control
        assert output["system"][0]["cache_control"] == {"type": "ephemeral"}
        # User message does NOT have cache_control (it's not a context message)
        assert "cache_control" not in output["messages"][0]


class TestAnthropicFormatterRoleAlternation:
    """AnthropicFormatter enforces strict user/assistant alternation."""

    def test_context_block_and_user_memory_merged(self) -> None:
        """Context block (user) + first user memory message are merged."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert "Hello" in output["messages"][0]["content"]

    def test_consecutive_user_memory_items_merged(self) -> None:
        """Two user memory items without an assistant in between are merged."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="First question",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Follow up",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Answer",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert "First question" in output["messages"][0]["content"]
        assert "Follow up" in output["messages"][0]["content"]
        assert output["messages"][1]["role"] == "assistant"

    def test_proper_alternation_unchanged(self) -> None:
        """user -> assistant -> user passes through without merging."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Hi!",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        window.add_item(
            ContextItem(
                content="How are you?",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 3
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][1]["role"] == "assistant"
        assert output["messages"][2]["role"] == "user"

    def test_empty_window_returns_empty_messages(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["messages"] == []
