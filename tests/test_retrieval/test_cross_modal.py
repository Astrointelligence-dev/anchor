"""Tests for cross-modal retrieval."""

from __future__ import annotations

import pytest

from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.retriever import Retriever
from anchor.retrieval.cross_modal import CrossModalEncoder, SharedSpaceRetriever


def _text_encoder(text: str) -> list[float]:
    return [len(text) / 100.0, 0.5, 0.3]


def _image_encoder(desc: str) -> list[float]:
    return [0.2, len(desc) / 100.0, 0.7]


def _make_item(
    content: str,
    item_id: str | None = None,
    modality: str | None = None,
) -> ContextItem:
    kwargs: dict = {"content": content, "source": SourceType.RETRIEVAL}
    if item_id is not None:
        kwargs["id"] = item_id
    if modality is not None:
        kwargs["metadata"] = {"modality": modality}
    return ContextItem(**kwargs)


class TestCrossModalEncoder:
    def test_cross_modal_encoder_basic(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        result = encoder.encode("hello", "text")
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == len("hello") / 100.0

    def test_cross_modal_encoder_unknown_modality(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        with pytest.raises(ValueError, match="Unknown modality"):
            encoder.encode("data", "audio")

    def test_cross_modal_encoder_modalities_property(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        assert encoder.modalities == ["image", "text"]


class TestSharedSpaceRetriever:
    def test_shared_space_retriever_protocol(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)
        assert isinstance(retriever, Retriever)

    def test_shared_space_retriever_index_and_retrieve(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder, query_modality="text")

        items = [
            _make_item("short", item_id="a"),
            _make_item("a much longer piece of text content", item_id="b"),
        ]
        retriever.index(items, modality="text")

        query = QueryBundle(query_str="a somewhat long query string")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2
        # Both items returned, sorted by similarity

    def test_shared_space_retriever_empty_index(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)
        query = QueryBundle(query_str="test")
        results = retriever.retrieve(query, top_k=5)
        assert results == []

    def test_shared_space_retriever_top_k(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)

        items = [_make_item(f"item number {i}", item_id=str(i)) for i in range(5)]
        retriever.index(items, modality="text")

        query = QueryBundle(query_str="test query")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2

    def test_shared_space_retriever_modality_from_metadata(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder, query_modality="text")

        items = [
            _make_item("a text document", item_id="t1", modality="text"),
            _make_item("an image description", item_id="i1", modality="image"),
        ]
        # Index without explicit modality -- should use metadata
        retriever.index(items)

        query = QueryBundle(query_str="find similar content")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2

    def test_shared_space_custom_similarity(self) -> None:
        def dot_product(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=False))

        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder, similarity_fn=dot_product)

        items = [_make_item("hello world", item_id="x")]
        retriever.index(items, modality="text")

        query = QueryBundle(query_str="hello")
        results = retriever.retrieve(query, top_k=1)
        assert len(results) == 1


class TestCrossModalEncoderEdgeCases:
    """Edge cases for CrossModalEncoder."""

    def test_single_modality_encoder(self) -> None:
        """Encoder with only one modality should work for that modality."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        assert encoder.modalities == ["text"]
        result = encoder.encode("hello", "text")
        assert isinstance(result, list)

    def test_encoding_same_content_twice_is_deterministic(self) -> None:
        """Encoding the same content twice should produce identical results."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        result1 = encoder.encode("test content", "text")
        result2 = encoder.encode("test content", "text")
        assert result1 == result2

        img_result1 = encoder.encode("a cat photo", "image")
        img_result2 = encoder.encode("a cat photo", "image")
        assert img_result1 == img_result2


class TestSharedSpaceRetrieverEdgeCases:
    """Edge cases for SharedSpaceRetriever."""

    def test_mixed_modalities_in_index(self) -> None:
        """Text and image items indexed together in a shared space."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder, query_modality="text")

        text_item = _make_item("a document about cats", item_id="t1", modality="text")
        image_item = _make_item("photo of a cat", item_id="i1", modality="image")

        retriever.index([text_item, image_item])

        query = QueryBundle(query_str="find cats")
        results = retriever.retrieve(query, top_k=10)
        assert len(results) == 2
        result_ids = {r.id for r in results}
        assert result_ids == {"t1", "i1"}

    def test_query_modality_different_from_index_modality(self) -> None:
        """Text query against image-only indexed items in shared space.

        Both modalities map into the same vector space, so cross-modal
        retrieval should still return results.
        """
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder, query_modality="text")

        # Index only image items
        items = [
            _make_item("sunset over ocean", item_id="img1", modality="image"),
            _make_item("mountain landscape", item_id="img2", modality="image"),
        ]
        retriever.index(items)

        query = QueryBundle(query_str="beautiful scenery")
        results = retriever.retrieve(query, top_k=5)
        assert len(results) == 2
        # Both image items are returned for a text query
        result_ids = {r.id for r in results}
        assert result_ids == {"img1", "img2"}

    def test_reindexing_appends(self) -> None:
        """Calling index() twice should append items, not overwrite."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)

        batch1 = [_make_item("first batch item", item_id="b1")]
        batch2 = [_make_item("second batch item", item_id="b2")]

        retriever.index(batch1, modality="text")
        retriever.index(batch2, modality="text")

        query = QueryBundle(query_str="batch item")
        results = retriever.retrieve(query, top_k=10)
        assert len(results) == 2
        result_ids = {r.id for r in results}
        assert result_ids == {"b1", "b2"}

    def test_large_number_of_items(self) -> None:
        """Index 150 items and retrieve top_k correctly."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)

        items = [_make_item(f"document number {i}", item_id=str(i)) for i in range(150)]
        retriever.index(items, modality="text")

        query = QueryBundle(query_str="document")
        results = retriever.retrieve(query, top_k=10)
        assert len(results) == 10
        # All results should have unique IDs
        result_ids = [r.id for r in results]
        assert len(set(result_ids)) == 10

    def test_large_number_top_k_exceeds_index(self) -> None:
        """top_k larger than indexed items returns all items."""
        encoder = CrossModalEncoder(encoders={"text": _text_encoder})
        retriever = SharedSpaceRetriever(encoder=encoder)

        items = [_make_item(f"item {i}", item_id=str(i)) for i in range(5)]
        retriever.index(items, modality="text")

        query = QueryBundle(query_str="item")
        results = retriever.retrieve(query, top_k=100)
        assert len(results) == 5


class TestRepr:
    def test_repr_all(self) -> None:
        encoder = CrossModalEncoder(encoders={"text": _text_encoder, "image": _image_encoder})
        assert "CrossModalEncoder" in repr(encoder)
        assert "image" in repr(encoder)
        assert "text" in repr(encoder)

        retriever = SharedSpaceRetriever(encoder=encoder)
        assert "SharedSpaceRetriever" in repr(retriever)
