"""Two-level hierarchical chunking and parent expansion.

Provides ``ParentChildChunker`` for creating large parent / small child
chunk pairs, and ``ParentExpander`` for replacing child content with
the full parent text after retrieval.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

from anchor.ingestion.chunkers import FixedSizeChunker
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

logger = logging.getLogger(__name__)


class ParentChildChunker:
    """Two-level hierarchical chunker.

    Produces large parent chunks for context and small child chunks
    for retrieval.

    First splits text into large parent chunks, then subdivides each parent
    into smaller child chunks. Each child chunk's metadata includes its
    parent_id for later expansion.

    Implements a variant of the ``Chunker`` protocol that returns
    ``(text, metadata)`` tuples via ``chunk_with_metadata()``.

    Parameters:
        parent_chunk_size: Token size for parent chunks. Default 1024.
        child_chunk_size: Token size for child chunks. Default 256.
        parent_overlap: Token overlap between parent chunks. Default 100.
        child_overlap: Token overlap between child chunks. Default 25.
        tokenizer: Token counter. Uses default if not provided.
    """

    __slots__ = (
        "_child_chunk_size",
        "_child_chunker",
        "_child_overlap",
        "_parent_chunk_size",
        "_parent_chunker",
        "_parent_overlap",
        "_parent_texts",
        "_tokenizer",
    )

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
        parent_overlap: int = 100,
        child_overlap: int = 25,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if child_chunk_size >= parent_chunk_size:
            msg = (
                f"child_chunk_size ({child_chunk_size}) must be less than "
                f"parent_chunk_size ({parent_chunk_size})"
            )
            raise ValueError(msg)

        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size
        self._parent_overlap = parent_overlap
        self._child_overlap = child_overlap
        self._tokenizer = tokenizer or get_default_counter()
        self._parent_texts: dict[str, str] = {}

        self._parent_chunker = FixedSizeChunker(
            chunk_size=parent_chunk_size,
            overlap=parent_overlap,
            tokenizer=self._tokenizer,
        )
        self._child_chunker = FixedSizeChunker(
            chunk_size=child_chunk_size,
            overlap=child_overlap,
            tokenizer=self._tokenizer,
        )

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text into child chunk texts only.

        This satisfies the ``Chunker`` protocol by returning plain strings.

        Parameters:
            text: The text to chunk.
            metadata: Unused; accepted for protocol compliance.

        Returns:
            A list of child chunk text strings.
        """
        if not text or not text.strip():
            return []

        parent_chunks = self._parent_chunker.chunk(text)
        child_texts: list[str] = []

        for parent_text in parent_chunks:
            children = self._child_chunker.chunk(parent_text)
            child_texts.extend(children)

        return child_texts

    def chunk_with_metadata(
        self, text: str, metadata: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Split text into child chunks with parent metadata attached.

        Parameters:
            text: The text to chunk.
            metadata: Optional document-level metadata to propagate.

        Returns:
            A list of ``(child_text, child_metadata)`` tuples. Each
            metadata dict includes ``parent_id``, ``parent_index``,
            ``child_index``, and ``is_child_chunk``. The parent text is
            stored ONCE on the chunker (see :meth:`get_parent`) instead of
            being duplicated into every child's metadata — a 1024-token
            parent copied into 4-8 children was pure storage bloat.
        """
        if not text or not text.strip():
            return []

        parent_chunks = self._parent_chunker.chunk(text)
        results: list[tuple[str, dict[str, Any]]] = []

        for parent_idx, parent_text in enumerate(parent_chunks):
            # Content-hash id: globally unique across documents (the old
            # "parent-{idx}" collided between docs) and dedupes identical
            # parents for free.
            parent_id = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()[:16]
            self._parent_texts[parent_id] = parent_text
            children = self._child_chunker.chunk(parent_text)

            for child_idx, child_text in enumerate(children):
                child_meta: dict[str, Any] = {
                    "parent_id": parent_id,
                    "parent_index": parent_idx,
                    "child_index": child_idx,
                    "is_child_chunk": True,
                }
                if metadata:
                    child_meta.update(metadata)
                results.append((child_text, child_meta))

        return results

    def get_parent(self, parent_id: str) -> str | None:
        """Look up a parent chunk's text by id (for ``ParentExpander``)."""
        return self._parent_texts.get(parent_id)

    def __repr__(self) -> str:
        return (
            f"ParentChildChunker(parent_chunk_size={self._parent_chunk_size}, "
            f"child_chunk_size={self._child_chunk_size})"
        )


class ParentExpander:
    """PostProcessor that expands child chunks back to their parent text.

    When child chunks are retrieved, this expander replaces the child
    content with the full parent content, deduplicating by parent_id
    to avoid repeating the same parent multiple times.

    Implements the ``PostProcessor`` protocol.

    Parameters:
        keep_child: If True, keep the original child content in metadata
            under ``original_child_content``. Default False.
        parent_lookup: Callable resolving a ``parent_id`` to its text —
            typically ``chunker.get_parent``. Falls back to legacy
            ``metadata["parent_text"]`` when not provided.
    """

    __slots__ = ("_keep_child", "_parent_lookup")

    def __init__(
        self,
        keep_child: bool = False,
        parent_lookup: Callable[[str], str | None] | None = None,
    ) -> None:
        self._keep_child = keep_child
        self._parent_lookup = parent_lookup

    def process(
        self,
        items: list[ContextItem],
        query: QueryBundle | None = None,
    ) -> list[ContextItem]:
        """Expand child chunks to parent text, deduplicating by parent_id.

        For each item with ``is_child_chunk`` in its metadata, the content
        is replaced with the parent text. Multiple children from the same
        parent are deduplicated (first occurrence wins). Items without
        ``is_child_chunk`` pass through unchanged.

        Parameters:
            items: The context items to post-process.
            query: The original query bundle (unused).

        Returns:
            A list of ``ContextItem`` objects with child chunks expanded.
        """
        seen_parents: set[str] = set()
        result: list[ContextItem] = []

        for item in items:
            if not item.metadata.get("is_child_chunk"):
                result.append(item)
                continue

            parent_id = item.metadata.get("parent_id", "")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent_text: str | None = None
            if self._parent_lookup is not None:
                parent_text = self._parent_lookup(parent_id)
            if parent_text is None:
                parent_text = item.metadata.get("parent_text", item.content)
            new_metadata = dict(item.metadata)

            if self._keep_child:
                new_metadata["original_child_content"] = item.content

            expanded = ContextItem(
                id=item.id,
                content=parent_text,
                source=item.source,
                score=item.score,
                priority=item.priority,
                token_count=item.token_count,
                metadata=new_metadata,
            )
            result.append(expanded)

        return result

    def __repr__(self) -> str:
        return f"ParentExpander(keep_child={self._keep_child})"
