"""Tests for built-in chunking strategies."""

from __future__ import annotations

import pytest

from anchor.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from anchor.protocols.ingestion import Chunker
from tests.conftest import FakeTokenizer


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = FixedSizeChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, fixed_chunker: FixedSizeChunker) -> None:
        assert fixed_chunker.chunk("") == []
        assert fixed_chunker.chunk("   ") == []

    def test_short_text_single_chunk(self, fixed_chunker: FixedSizeChunker) -> None:
        text = "hello world"
        chunks = fixed_chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_splitting_respects_chunk_size(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        text = "one two three four five six seven eight nine ten"
        chunks = chunker.chunk(text)
        for c in chunks:
            assert fake_tokenizer.count_tokens(c) <= 5

    def test_overlap_produces_shared_content(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=2, tokenizer=fake_tokenizer)
        text = "one two three four five six seven eight"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # With overlap, adjacent chunks should share some words
        if len(chunks) >= 2:
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1, "Adjacent chunks should share words due to overlap"

    def test_invalid_chunk_size(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0, tokenizer=fake_tokenizer)

    def test_invalid_overlap(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            FixedSizeChunker(overlap=-1, tokenizer=fake_tokenizer)

    def test_overlap_exceeds_chunk_size(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match=r"overlap.*must be less than chunk_size"):
            FixedSizeChunker(chunk_size=5, overlap=5, tokenizer=fake_tokenizer)

    def test_repr(self, fixed_chunker: FixedSizeChunker) -> None:
        assert "FixedSizeChunker" in repr(fixed_chunker)
        assert "chunk_size=10" in repr(fixed_chunker)

    def test_metadata_parameter_accepted(self, fixed_chunker: FixedSizeChunker) -> None:
        """Metadata param is accepted but unused."""
        chunks = fixed_chunker.chunk("hello world", metadata={"lang": "en"})
        assert len(chunks) == 1


class TestRecursiveCharacterChunker:
    """Tests for RecursiveCharacterChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = RecursiveCharacterChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, recursive_chunker: RecursiveCharacterChunker) -> None:
        assert recursive_chunker.chunk("") == []
        assert recursive_chunker.chunk("   ") == []

    def test_short_text_single_chunk(self, recursive_chunker: RecursiveCharacterChunker) -> None:
        chunks = recursive_chunker.chunk("hello world")
        assert len(chunks) == 1

    def test_splits_on_paragraph_boundary(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        text = "one two three\n\nfour five six"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_falls_back_to_finer_separator(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        # No paragraph breaks, should split on newline or space
        text = "one two three four five six seven eight"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert fake_tokenizer.count_tokens(c) <= 5

    def test_invalid_args(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=0, tokenizer=fake_tokenizer)
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(overlap=-1, tokenizer=fake_tokenizer)

    def test_repr(self, recursive_chunker: RecursiveCharacterChunker) -> None:
        assert "RecursiveCharacterChunker" in repr(recursive_chunker)


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SentenceChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, sentence_chunker: SentenceChunker) -> None:
        assert sentence_chunker.chunk("") == []
        assert sentence_chunker.chunk("   ") == []

    def test_single_sentence(self, sentence_chunker: SentenceChunker) -> None:
        chunks = sentence_chunker.chunk("Hello world.")
        assert len(chunks) == 1

    def test_multiple_sentences_grouped(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SentenceChunker(chunk_size=20, overlap=0, tokenizer=fake_tokenizer)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        # All sentences should fit in one chunk (under 20 words)
        assert len(chunks) == 1

    def test_sentences_split_when_exceeding_budget(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SentenceChunker(chunk_size=5, overlap=0, tokenizer=fake_tokenizer)
        text = "One two three. Four five six. Seven eight nine."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_sentence_overlap(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SentenceChunker(chunk_size=5, overlap=1, tokenizer=fake_tokenizer)
        text = "One two three. Four five six. Seven eight nine."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_invalid_args(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=0, tokenizer=fake_tokenizer)
        with pytest.raises(ValueError):
            SentenceChunker(overlap=-1, tokenizer=fake_tokenizer)

    def test_repr(self, sentence_chunker: SentenceChunker) -> None:
        assert "SentenceChunker" in repr(sentence_chunker)
