"""Built-in chunking strategies for document ingestion.

All chunkers implement the ``Chunker`` protocol and use the ``Tokenizer``
protocol for token-aware splitting.  Zero external dependencies required.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from anchor._math import cosine_similarity
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

logger = logging.getLogger(__name__)


class FixedSizeChunker:
    """Split text into fixed-size chunks by token count.

    Implements the ``Chunker`` protocol.
    """

    __slots__ = ("_chunk_size", "_overlap", "_tokenizer")

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if overlap < 0:
            msg = f"overlap must be non-negative, got {overlap}"
            raise ValueError(msg)
        if overlap >= chunk_size:
            msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = tokenizer or get_default_counter()

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text into fixed-size token chunks with overlap.

        Parameters:
            text: The text to chunk.
            metadata: Unused; accepted for protocol compliance.

        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        # Per-word token counts computed once: O(n) instead of re-encoding
        # the growing chunk for every appended word (O(n^2)). Exact for
        # whitespace tokenizers; a close approximation for BPE encodings
        # (chunk_size is a soft target, not a hard contract).
        word_tokens = [self._tokenizer.count_tokens(w) for w in words]
        chunks: list[str] = []
        start = 0

        while start < len(words):
            # Build a chunk up to chunk_size tokens
            end = start
            running = 0
            while end < len(words):
                next_total = running + word_tokens[end]
                if next_total > self._chunk_size and end > start:
                    break
                running = next_total
                end += 1

            if end > start:
                chunks.append(" ".join(words[start:end]))

            # If this chunk reached the end of the text, we're done
            if end >= len(words):
                break

            # Advance by (chunk_size - overlap) worth of words
            if end == start:
                # Single word exceeds chunk_size; skip it to avoid infinite loop
                start = end + 1
            else:
                # Calculate how many words to step back for overlap
                chunk_word_count = end - start
                step = max(
                    1, chunk_word_count - self._overlap_words(word_tokens, start, end)
                )
                start += step

        return chunks

    def _overlap_words(self, word_tokens: list[int], start: int, end: int) -> int:
        """Calculate number of trailing words that fit in overlap tokens."""
        if self._overlap == 0:
            return 0
        count = 0
        running = 0
        idx = end - 1
        while idx >= start and count < (end - start):
            running += word_tokens[idx]
            if running > self._overlap:
                break
            count += 1
            idx -= 1
        return count

    def __repr__(self) -> str:
        return (
            f"FixedSizeChunker(chunk_size={self._chunk_size}, "
            f"overlap={self._overlap})"
        )


class RecursiveCharacterChunker:
    """Split text using a hierarchy of separators, falling back to finer splits.

    Separator hierarchy: ``"\\n\\n"`` → ``"\\n"`` → ``". "`` → ``" "``.

    Defaults follow Chroma's chunking evaluation: well-parameterized
    recursive splitting at 200-400 tokens with zero overlap matches
    semantic chunkers within ~2 recall points at far lower cost.

    Implements the ``Chunker`` protocol.
    """

    __slots__ = ("_chunk_size", "_overlap", "_separators", "_tokenizer")

    _DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(
        self,
        chunk_size: int = 384,
        overlap: int = 0,
        separators: tuple[str, ...] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if overlap < 0:
            msg = f"overlap must be non-negative, got {overlap}"
            raise ValueError(msg)
        if overlap >= chunk_size:
            msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._separators = separators or self._DEFAULT_SEPARATORS
        self._tokenizer = tokenizer or get_default_counter()

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Recursively split text using separator hierarchy.

        Parameters:
            text: The text to chunk.
            metadata: Unused; accepted for protocol compliance.

        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            return []
        # Overlap is applied exactly once, over the final flat chunk list.
        # _split is recursive; applying overlap inside it would re-apply
        # overlap to sub-chunks at every recursion level.
        return self._apply_overlap(self._split(text, 0))

    def _split(self, text: str, sep_idx: int) -> list[str]:
        """Recursively split text, falling back to finer separators."""
        if self._tokenizer.count_tokens(text) <= self._chunk_size:
            stripped = text.strip()
            return [stripped] if stripped else []

        if sep_idx >= len(self._separators):
            # No more separators; truncate to chunk_size
            return [self._tokenizer.truncate_to_tokens(text, self._chunk_size)]

        separator = self._separators[sep_idx]
        parts = text.split(separator)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = separator.join([current, part]) if current else part

            if self._tokenizer.count_tokens(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If the part itself exceeds chunk_size, split it further
                if self._tokenizer.count_tokens(part) > self._chunk_size:
                    sub_chunks = self._split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply token-based overlap between adjacent chunks."""
        if self._overlap == 0 or len(chunks) <= 1:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            # Walk backwards accumulating per-word counts: O(n) per boundary
            # instead of re-encoding a growing suffix string per word.
            running = 0
            cut = len(prev_words)
            for j in range(len(prev_words) - 1, -1, -1):
                running += self._tokenizer.count_tokens(prev_words[j])
                if running > self._overlap:
                    break
                cut = j
            overlap_text = " ".join(prev_words[cut:])

            if overlap_text:
                merged = overlap_text + " " + chunks[i]
                # Only add overlap if it doesn't exceed chunk_size
                if self._tokenizer.count_tokens(merged) <= self._chunk_size:
                    result.append(merged)
                else:
                    result.append(chunks[i])
            else:
                result.append(chunks[i])

        return result

    def __repr__(self) -> str:
        return (
            f"RecursiveCharacterChunker(chunk_size={self._chunk_size}, "
            f"overlap={self._overlap})"
        )


class SentenceChunker:
    """Split text at sentence boundaries, grouping sentences to fill chunks.

    Uses regex-based sentence boundary detection.  Overlap is measured
    in sentences rather than tokens.

    Implements the ``Chunker`` protocol.
    """

    __slots__ = ("_chunk_size", "_overlap", "_sentence_pattern", "_tokenizer")

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 1,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if overlap < 0:
            msg = f"overlap must be non-negative, got {overlap}"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = tokenizer or get_default_counter()
        self._sentence_pattern = self._SENTENCE_RE

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text into sentence-based chunks.

        Parameters:
            text: The text to chunk.
            metadata: Unused; accepted for protocol compliance.

        Returns:
            A list of text chunks, each containing one or more sentences.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(sentences):
            end = start
            current = ""

            while end < len(sentences):
                candidate = " ".join(sentences[start : end + 1])
                if self._tokenizer.count_tokens(candidate) > self._chunk_size and end > start:
                    break
                current = candidate
                end += 1

            if current:
                chunks.append(current)

            if end == start:
                # Single sentence exceeds chunk_size; include it anyway
                chunks.append(sentences[start])
                start = end + 1
            else:
                step = max(1, (end - start) - self._overlap)
                start += step

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(chunk_size={self._chunk_size}, "
            f"overlap={self._overlap})"
        )


class SemanticChunker:
    """Split text at semantic boundaries using embedding similarity.

    Segments the text into sentences, computes embeddings, and splits
    where cosine similarity between adjacent segments drops below a
    threshold.

    Implements the ``Chunker`` protocol.

    Parameters:
        embed_fn: A callable that takes a list of strings and returns
            a list of float lists (embeddings). Signature:
            ``(list[str]) -> list[list[float]]``
        tokenizer: Token counter for enforcing chunk size limits.
        threshold: Cosine similarity threshold. Adjacent segments with
            similarity below this are split. Default 0.5.
        chunk_size: Maximum tokens per chunk. Default 512.
        min_chunk_size: Minimum tokens per chunk. Chunks smaller than
            this are merged with neighbors. Default 50.
    """

    __slots__ = ("_chunk_size", "_embed_fn", "_min_chunk_size", "_threshold", "_tokenizer")

    _SENTENCE_RE = SentenceChunker._SENTENCE_RE

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        tokenizer: Tokenizer | None = None,
        threshold: float = 0.5,
        chunk_size: int = 512,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if min_chunk_size < 0:
            msg = f"min_chunk_size must be non-negative, got {min_chunk_size}"
            raise ValueError(msg)
        self._embed_fn = embed_fn
        self._tokenizer = tokenizer or get_default_counter()
        self._threshold = threshold
        self._chunk_size = chunk_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text into semantically coherent chunks.

        Parameters:
            text: The text to chunk.
            metadata: Unused; accepted for protocol compliance.

        Returns:
            A list of text chunks split at semantic boundaries.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # If everything fits in one chunk, return as-is
        joined = " ".join(sentences)
        if self._tokenizer.count_tokens(joined) <= self._chunk_size:
            return [joined]

        # Compute embeddings for all sentences
        embeddings = self._embed_fn(sentences)

        # Identify split points where similarity drops below threshold
        split_indices: list[int] = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                split_indices.append(i + 1)

        # Group sentences between split points
        groups = self._group_sentences(sentences, split_indices)

        # Merge small chunks with neighbors
        groups = self._merge_small_chunks(groups)

        # Split oversized chunks using word-based fallback
        result: list[str] = []
        fallback = FixedSizeChunker(
            chunk_size=self._chunk_size, overlap=0, tokenizer=self._tokenizer
        )
        for group in groups:
            if self._tokenizer.count_tokens(group) > self._chunk_size:
                result.extend(fallback.chunk(group))
            else:
                result.append(group)

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = self._SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(
        self, sentences: list[str], split_indices: list[int]
    ) -> list[str]:
        """Group sentences between split points into chunks.

        Parameters:
            sentences: All sentences from the text.
            split_indices: Indices where semantic breaks occur.

        Returns:
            A list of text chunks formed by joining sentence groups.
        """
        groups: list[str] = []
        prev = 0
        for idx in split_indices:
            group = " ".join(sentences[prev:idx])
            if group.strip():
                groups.append(group.strip())
            prev = idx
        # Remaining sentences
        tail = " ".join(sentences[prev:])
        if tail.strip():
            groups.append(tail.strip())
        return groups

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks smaller than min_chunk_size with their neighbors.

        Parameters:
            chunks: List of text chunks.

        Returns:
            A list of chunks with small ones merged into neighbors.
        """
        if self._min_chunk_size <= 0 or len(chunks) <= 1:
            return chunks

        merged: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            prev_tokens = self._tokenizer.count_tokens(merged[-1])
            cur_tokens = self._tokenizer.count_tokens(chunk)
            if prev_tokens < self._min_chunk_size or cur_tokens < self._min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        return merged

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(threshold={self._threshold}, "
            f"chunk_size={self._chunk_size})"
        )


class MarkdownHeaderChunker:
    """Structure-aware chunking for Markdown: split at headings first.

    Splits the document into sections at markdown headings, chunks each
    section with an inner chunker, and stamps every chunk with its header
    path (``"H1 > H2"``) — both in metadata (``headers``) and, by default,
    prepended to the chunk content so embeddings carry the document
    structure. Structure-aware splitting is the evidence-backed win over
    semantic-similarity splitting for structured documents.

    Implements the ``Chunker`` protocol plus ``chunk_with_metadata`` (the
    ``DocumentIngester`` hook).

    Parameters:
        inner: Chunker applied to each section's body. Defaults to
            ``RecursiveCharacterChunker`` with its standard defaults.
        include_headers_in_content: Prepend the header path to each chunk's
            text (default True).
        tokenizer: Token counter passed to the default inner chunker.
    """

    __slots__ = ("_include_headers", "_inner")

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

    def __init__(
        self,
        inner: Any = None,
        include_headers_in_content: bool = True,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._inner = inner or RecursiveCharacterChunker(tokenizer=tokenizer)
        self._include_headers = include_headers_in_content

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split markdown into header-scoped chunks (text only)."""
        return [c for c, _ in self.chunk_with_metadata(text, metadata)]

    def chunk_with_metadata(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Split markdown into ``(chunk, {"headers": path})`` tuples."""
        if not text or not text.strip():
            return []

        sections = self._split_sections(text)
        results: list[tuple[str, dict[str, Any]]] = []
        for header_path, body in sections:
            if not body.strip():
                continue
            for chunk in self._inner.chunk(body):
                content = (
                    f"{header_path}\n\n{chunk}"
                    if header_path and self._include_headers
                    else chunk
                )
                chunk_meta: dict[str, Any] = {}
                if header_path:
                    chunk_meta["headers"] = header_path
                results.append((content, chunk_meta))
        return results

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split text at headings, tracking the ``H1 > H2`` path per section."""
        sections: list[tuple[str, str]] = []
        # (level, title) stack for the current header path
        stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            path = " > ".join(title for _, title in stack)
            sections.append((path, body))
            current_lines.clear()

        for line in text.splitlines():
            match = self._HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, match.group(2).strip()))
            else:
                current_lines.append(line)
        flush()
        return sections

    def __repr__(self) -> str:
        return f"MarkdownHeaderChunker(inner={self._inner!r})"
