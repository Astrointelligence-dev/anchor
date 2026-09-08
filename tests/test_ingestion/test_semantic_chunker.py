"""Tests for SemanticChunker."""

from __future__ import annotations

import math

import pytest

from anchor.ingestion.chunkers import SemanticChunker
from anchor.protocols.ingestion import Chunker
from tests.conftest import FakeTokenizer


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Generate deterministic embeddings based on text content.

    Similar texts get similar embeddings, different texts get different ones.
    Uses a hash-based approach for determinism.
    """
    embeddings: list[list[float]] = []
    for text in texts:
        seed = hash(text) % 10000
        raw = [math.sin(seed * 1000 + i) for i in range(64)]
        norm = math.sqrt(sum(x * x for x in raw))
        embeddings.append([x / norm for x in raw] if norm > 0 else raw)
    return embeddings


def _contrasting_embed_fn(texts: list[str]) -> list[list[float]]:
    """Embeddings where texts about different topics are very different.

    Texts containing 'dog' map to one region, 'cat' to another, etc.
    This makes boundary detection easier to test.
    """
    topic_vectors: dict[str, list[float]] = {
        "dog": [1.0] + [0.0] * 63,
        "cat": [0.0, 1.0] + [0.0] * 62,
        "fish": [0.0, 0.0, 1.0] + [0.0] * 61,
        "bird": [0.0, 0.0, 0.0, 1.0] + [0.0] * 60,
    }

    embeddings: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        matched = False
        for keyword, vec in topic_vectors.items():
            if keyword in lower:
                embeddings.append(vec)
                matched = True
                break
        if not matched:
            # Default: hash-based embedding
            seed = hash(text) % 10000
            raw = [math.sin(seed * 1000 + i) for i in range(64)]
            norm = math.sqrt(sum(x * x for x in raw))
            embeddings.append([x / norm for x in raw] if norm > 0 else raw)
    return embeddings


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(embed_fn=_fake_embed_fn, tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, semantic_chunker: SemanticChunker) -> None:
        assert semantic_chunker.chunk("") == []
        assert semantic_chunker.chunk("   ") == []

    def test_short_text_single_chunk(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(
            embed_fn=_fake_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=50,
        )
        text = "This is a short sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short sentence."

    def test_splits_at_semantic_boundaries(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(
            embed_fn=_contrasting_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=15,
            min_chunk_size=0,
            threshold=0.5,
        )
        # These sentences have very different topics => orthogonal embeddings
        # cosine similarity between orthogonal vectors is 0.0 < 0.5 threshold
        # Total ~20 words > chunk_size=15 forces the semantic path
        text = (
            "The dog runs in the park. The dog plays fetch. "
            "The cat sleeps on a mat. The cat purrs softly."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}: {chunks}"

    def test_respects_chunk_size(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(
            embed_fn=_fake_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=10,
            min_chunk_size=0,
        )
        text = (
            "First sentence here. Second sentence here. "
            "Third sentence here. Fourth sentence here. "
            "Fifth sentence here. Sixth sentence here. "
            "Seventh sentence here. Eighth sentence here."
        )
        chunks = chunker.chunk(text)
        for chunk in chunks:
            token_count = fake_tokenizer.count_tokens(chunk)
            assert token_count <= 10, f"Chunk exceeds chunk_size: {token_count} tokens in '{chunk}'"

    def test_min_chunk_size_merges_small_chunks(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(
            embed_fn=_contrasting_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=50,
            min_chunk_size=5,
            threshold=0.5,
        )
        # Each sentence is short (3-5 words). With min_chunk_size=5,
        # small chunks should be merged with neighbors.
        text = "The dog runs. The cat sleeps. The fish swims. The bird flies."
        chunks = chunker.chunk(text)
        for chunk in chunks:
            # Each merged chunk should meet or exceed the min_chunk_size
            # (unless the total text is smaller)
            token_count = fake_tokenizer.count_tokens(chunk)
            assert token_count >= 5 or len(chunks) == 1, (
                f"Chunk too small: {token_count} tokens in '{chunk}'"
            )

    def test_threshold_sensitivity(self, fake_tokenizer: FakeTokenizer) -> None:
        text = (
            "The dog runs in the park. The dog plays fetch. "
            "The cat sleeps on a mat. The cat purrs softly."
        )
        # Lower threshold => fewer splits (more similarity accepted)
        low_thresh = SemanticChunker(
            embed_fn=_contrasting_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=50,
            min_chunk_size=0,
            threshold=0.0,
        )
        # Higher threshold => more splits
        high_thresh = SemanticChunker(
            embed_fn=_contrasting_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=50,
            min_chunk_size=0,
            threshold=0.99,
        )
        low_chunks = low_thresh.chunk(text)
        high_chunks = high_thresh.chunk(text)
        assert len(low_chunks) <= len(high_chunks), (
            f"Lower threshold should produce fewer or equal splits: "
            f"{len(low_chunks)} vs {len(high_chunks)}"
        )

    def test_repr(self, semantic_chunker: SemanticChunker) -> None:
        r = repr(semantic_chunker)
        assert "SemanticChunker" in r

    def test_invalid_args(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(
                embed_fn=_fake_embed_fn,
                tokenizer=fake_tokenizer,
                chunk_size=0,
            )

    def test_metadata_parameter_accepted(self, semantic_chunker: SemanticChunker) -> None:
        """Metadata param is accepted but unused."""
        chunks = semantic_chunker.chunk("hello world", metadata={"lang": "en"})
        assert len(chunks) == 1

    def test_single_sentence(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = SemanticChunker(
            embed_fn=_fake_embed_fn,
            tokenizer=fake_tokenizer,
            chunk_size=50,
        )
        chunks = chunker.chunk("Just one sentence here.")
        assert len(chunks) == 1
