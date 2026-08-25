"""Tests for two-level hierarchical chunking and parent expansion."""

from __future__ import annotations

import pytest

from anchor.ingestion.hierarchical import ParentChildChunker, ParentExpander
from anchor.models.context import ContextItem, SourceType
from anchor.protocols.ingestion import Chunker
from anchor.protocols.postprocessor import PostProcessor
from tests.conftest import FakeTokenizer


class TestParentChildChunker:
    """Tests for ParentChildChunker."""

    def test_protocol_compliance(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = ParentChildChunker(tokenizer=fake_tokenizer)
        assert isinstance(chunker, Chunker)

    def test_empty_input(self, parent_child_chunker: ParentChildChunker) -> None:
        assert parent_child_chunker.chunk("") == []
        assert parent_child_chunker.chunk("   ") == []

    def test_short_text_single_parent_single_child(
        self, parent_child_chunker: ParentChildChunker,
    ) -> None:
        text = "hello world"
        chunks = parent_child_chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_child_chunks_fit_within_parent(
        self, parent_child_chunker: ParentChildChunker,
    ) -> None:
        # 30 words => should produce multiple parents (parent_chunk_size=20)
        text = " ".join(f"word{i}" for i in range(30))
        pairs = parent_child_chunker.chunk_with_metadata(text)

        # Collect unique parent ids; text lives on the chunker, not metadata
        parent_ids = {m["parent_id"] for _, m in pairs}
        assert len(parent_ids) >= 2

        # Every child text should be a substring of its parent
        for child_text, meta in pairs:
            parent_text = parent_child_chunker.get_parent(meta["parent_id"])
            assert parent_text is not None
            # Each word in the child should appear in the parent
            for word in child_text.split():
                assert word in parent_text

    def test_chunk_with_metadata_includes_parent_info(
        self, parent_child_chunker: ParentChildChunker,
    ) -> None:
        text = " ".join(f"w{i}" for i in range(15))
        pairs = parent_child_chunker.chunk_with_metadata(text)
        assert len(pairs) >= 1

        for _child_text, meta in pairs:
            assert "parent_id" in meta
            assert "parent_text" not in meta  # stored once on the chunker
            assert "parent_index" in meta
            assert "child_index" in meta
            assert meta["is_child_chunk"] is True
            assert parent_child_chunker.get_parent(meta["parent_id"]) is not None

    def test_chunk_returns_strings_only(
        self, parent_child_chunker: ParentChildChunker,
    ) -> None:
        text = " ".join(f"w{i}" for i in range(15))
        chunks = parent_child_chunker.chunk(text)
        assert all(isinstance(c, str) for c in chunks)

    def test_parent_overlap(self, fake_tokenizer: FakeTokenizer) -> None:
        chunker = ParentChildChunker(
            parent_chunk_size=10,
            child_chunk_size=5,
            parent_overlap=2,
            child_overlap=1,
            tokenizer=fake_tokenizer,
        )
        text = " ".join(f"w{i}" for i in range(25))
        pairs = chunker.chunk_with_metadata(text)

        # Collect parent texts by index
        parents_by_idx: dict[int, str] = {}
        for _, meta in pairs:
            text_val = chunker.get_parent(meta["parent_id"])
            assert text_val is not None
            parents_by_idx[meta["parent_index"]] = text_val

        if len(parents_by_idx) >= 2:
            p0_words = set(parents_by_idx[0].split())
            p1_words = set(parents_by_idx[1].split())
            assert p0_words & p1_words, "Adjacent parents should share words due to overlap"

    def test_invalid_args(self, fake_tokenizer: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match=r"child_chunk_size.*must be less than"):
            ParentChildChunker(
                parent_chunk_size=10,
                child_chunk_size=10,
                tokenizer=fake_tokenizer,
            )
        with pytest.raises(ValueError, match=r"child_chunk_size.*must be less than"):
            ParentChildChunker(
                parent_chunk_size=10,
                child_chunk_size=20,
                tokenizer=fake_tokenizer,
            )

    def test_repr(self, parent_child_chunker: ParentChildChunker) -> None:
        r = repr(parent_child_chunker)
        assert "ParentChildChunker" in r
        assert "parent_chunk_size=20" in r
        assert "child_chunk_size=5" in r

    def test_chunk_with_metadata_empty_input(
        self, parent_child_chunker: ParentChildChunker,
    ) -> None:
        assert parent_child_chunker.chunk_with_metadata("") == []
        assert parent_child_chunker.chunk_with_metadata("   ") == []

    def test_metadata_propagation(self, fake_tokenizer: FakeTokenizer) -> None:
        """Document-level metadata should propagate to child chunks."""
        chunker = ParentChildChunker(
            parent_chunk_size=20,
            child_chunk_size=5,
            parent_overlap=2,
            child_overlap=1,
            tokenizer=fake_tokenizer,
        )
        doc_meta = {"source": "test_doc", "lang": "en"}
        pairs = chunker.chunk_with_metadata("one two three four five", doc_meta)
        assert len(pairs) >= 1
        for _, meta in pairs:
            assert meta["source"] == "test_doc"
            assert meta["lang"] == "en"


class TestParentExpander:
    """Tests for ParentExpander."""

    def test_protocol_compliance(self) -> None:
        expander = ParentExpander()
        assert isinstance(expander, PostProcessor)

    def test_expands_child_to_parent(self, parent_expander: ParentExpander) -> None:
        item = ContextItem(
            content="child text",
            source=SourceType.RETRIEVAL,
            metadata={
                "is_child_chunk": True,
                "parent_id": "parent-0",
                "parent_text": "full parent text here",
            },
        )
        result = parent_expander.process([item])
        assert len(result) == 1
        assert result[0].content == "full parent text here"

    def test_deduplicates_by_parent_id(
        self, parent_expander: ParentExpander,
    ) -> None:
        items = [
            ContextItem(
                content=f"child {i}",
                source=SourceType.RETRIEVAL,
                metadata={
                    "is_child_chunk": True,
                    "parent_id": "parent-0",
                    "parent_text": "same parent text",
                },
            )
            for i in range(3)
        ]
        result = parent_expander.process(items)
        assert len(result) == 1
        assert result[0].content == "same parent text"

    def test_passthrough_non_child_items(
        self, parent_expander: ParentExpander,
    ) -> None:
        item = ContextItem(
            content="regular item",
            source=SourceType.RETRIEVAL,
            metadata={"some_key": "some_value"},
        )
        result = parent_expander.process([item])
        assert len(result) == 1
        assert result[0].content == "regular item"
        assert result[0] is item  # Should be the same object

    def test_mixed_items(self, parent_expander: ParentExpander) -> None:
        regular = ContextItem(
            content="regular item",
            source=SourceType.RETRIEVAL,
        )
        child_a = ContextItem(
            content="child a",
            source=SourceType.RETRIEVAL,
            metadata={
                "is_child_chunk": True,
                "parent_id": "parent-0",
                "parent_text": "parent 0 text",
            },
        )
        child_b = ContextItem(
            content="child b",
            source=SourceType.RETRIEVAL,
            metadata={
                "is_child_chunk": True,
                "parent_id": "parent-1",
                "parent_text": "parent 1 text",
            },
        )
        child_a2 = ContextItem(
            content="child a2",
            source=SourceType.RETRIEVAL,
            metadata={
                "is_child_chunk": True,
                "parent_id": "parent-0",
                "parent_text": "parent 0 text",
            },
        )

        result = parent_expander.process([regular, child_a, child_b, child_a2])
        assert len(result) == 3
        assert result[0].content == "regular item"
        assert result[1].content == "parent 0 text"
        assert result[2].content == "parent 1 text"

    def test_keep_child_option(self) -> None:
        expander = ParentExpander(keep_child=True)
        item = ContextItem(
            content="child text",
            source=SourceType.RETRIEVAL,
            metadata={
                "is_child_chunk": True,
                "parent_id": "parent-0",
                "parent_text": "full parent text",
            },
        )
        result = expander.process([item])
        assert len(result) == 1
        assert result[0].content == "full parent text"
        assert result[0].metadata["original_child_content"] == "child text"

    def test_empty_input(self, parent_expander: ParentExpander) -> None:
        result = parent_expander.process([])
        assert result == []

    def test_repr(self) -> None:
        assert "ParentExpander" in repr(ParentExpander())
        assert "keep_child=True" in repr(ParentExpander(keep_child=True))
