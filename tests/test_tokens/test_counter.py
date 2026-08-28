"""Tests for anchor.tokens.counter.

Since tiktoken requires network access to download encoding data (which is
unavailable in this test environment), we mock the tiktoken encoding to
test TiktokenCounter's logic, and we verify the Tokenizer protocol
compliance using FakeTokenizer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tiktoken", reason="tiktoken required for TiktokenCounter tests")

from unittest.mock import patch  # noqa: E402

from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import TiktokenCounter
from tests.conftest import FakeTokenizer


class _MockEncoding:
    """Mock tiktoken encoding for offline testing."""

    def encode(self, text: str) -> list[int]:
        """Encode text as one token per whitespace-separated word."""
        if not text or not text.strip():
            return []
        return list(range(len(text.split())))

    def decode(self, tokens: list[int]) -> str:
        """Decode is not perfectly invertible but sufficient for testing truncation."""
        # This is used after encoding then slicing, so we approximate
        return " ".join(f"tok{t}" for t in tokens)


def _make_counter() -> TiktokenCounter:
    """Create a TiktokenCounter with a mocked tiktoken encoding."""
    with patch("tiktoken.get_encoding", return_value=_MockEncoding()):
        counter = TiktokenCounter()
    return counter


class TestTiktokenCounterCountTokens:
    """count_tokens method."""

    def test_returns_positive_int_for_non_empty_text(self) -> None:
        counter = _make_counter()
        result = counter.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_zero_for_empty_string(self) -> None:
        counter = _make_counter()
        assert counter.count_tokens("") == 0

    def test_longer_text_has_more_tokens(self) -> None:
        counter = _make_counter()
        short = counter.count_tokens("Hi")
        long = counter.count_tokens("This is a much longer sentence with many words in it.")
        assert long > short

    def test_consistent_results(self) -> None:
        counter = _make_counter()
        text = "Deterministic token counting"
        assert counter.count_tokens(text) == counter.count_tokens(text)


class TestTiktokenCounterTruncate:
    """truncate_to_tokens method."""

    def test_truncation_produces_shorter_output(self) -> None:
        counter = _make_counter()
        text = "This is a somewhat long sentence that should definitely exceed five tokens."
        truncated = counter.truncate_to_tokens(text, max_tokens=5)
        assert counter.count_tokens(truncated) <= 5

    def test_large_limit_returns_original(self) -> None:
        counter = _make_counter()
        text = "Short text."
        truncated = counter.truncate_to_tokens(text, max_tokens=1000)
        assert truncated == text

    def test_truncate_to_zero_tokens(self) -> None:
        counter = _make_counter()
        text = "Some content"
        truncated = counter.truncate_to_tokens(text, max_tokens=0)
        # With 0 max_tokens, decode([]) returns empty or very short
        assert counter.count_tokens(truncated) == 0

    def test_truncate_exact_boundary(self) -> None:
        counter = _make_counter()
        text = "Hello world"
        token_count = counter.count_tokens(text)
        truncated = counter.truncate_to_tokens(text, max_tokens=token_count)
        assert truncated == text

    def test_split_head_tail_returns_whole_text_when_it_fits(self) -> None:
        counter = _make_counter()
        head, tail, total = counter.split_head_tail("a b c", 2, 2)
        assert (head, tail, total) == ("a b c", "", 3)

    def test_split_head_tail_splits_oversized_text(self) -> None:
        counter = _make_counter()
        text = " ".join(f"w{i}" for i in range(10))
        head, tail, total = counter.split_head_tail(text, 2, 3)
        assert total == 10
        # Mock encoding maps word i to token id i.
        assert head == "tok0 tok1"
        assert tail == "tok7 tok8 tok9"

    def test_split_head_tail_zero_tail_budget(self) -> None:
        counter = _make_counter()
        head, tail, total = counter.split_head_tail("a b c d", 2, 0)
        assert tail == ""
        assert total == 4
        assert head == "tok0 tok1"


class TestTiktokenCounterProtocol:
    """TiktokenCounter satisfies the Tokenizer protocol."""

    def test_isinstance_check(self) -> None:
        counter = _make_counter()
        assert isinstance(counter, Tokenizer)


class TestFakeTokenizerProtocol:
    """FakeTokenizer also satisfies the Tokenizer protocol."""

    def test_isinstance_check(self) -> None:
        counter = FakeTokenizer()
        assert isinstance(counter, Tokenizer)


class TestGetDefaultCounter:
    """get_default_counter singleton."""

    def test_returns_tiktoken_counter(self) -> None:
        from anchor.tokens.counter import get_default_counter

        get_default_counter.cache_clear()
        with patch("tiktoken.get_encoding", return_value=_MockEncoding()):
            try:
                c = get_default_counter()
                assert isinstance(c, TiktokenCounter)
            finally:
                get_default_counter.cache_clear()

    def test_returns_same_instance(self) -> None:
        from anchor.tokens.counter import get_default_counter

        get_default_counter.cache_clear()
        with patch("tiktoken.get_encoding", return_value=_MockEncoding()):
            try:
                c1 = get_default_counter()
                c2 = get_default_counter()
                assert c1 is c2
            finally:
                get_default_counter.cache_clear()
