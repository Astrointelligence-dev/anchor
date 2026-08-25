"""Sparse (BM25) retrieval.

Requires the 'bm25' extra: pip install astro-anchor[bm25]

The default backend is **bm25s** (numpy sparse scoring — up to 500x faster
than rank_bm25, correct Lucene-variant scoring, Snowball stemming and
stopword support via PyStemmer). When a custom ``tokenize_fn`` is supplied,
or bm25s is not installed, the legacy rank_bm25 path is used.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anchor.exceptions import RetrieverError
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Snowball language name -> bm25s stopword code
_STOPWORD_CODES = {
    "english": "en",
    "portuguese": "pt",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
}


class SparseRetriever:
    """BM25-based retrieval over tokenized documents.

    Parameters
    ----------
    tokenize_fn:
        Custom tokenizer ``(text) -> tokens``. Supplying one forces the
        legacy rank_bm25 backend (bm25s owns its tokenization pipeline).
    tokenizer:
        Token counter for ``token_count`` on returned items.
    min_score:
        Optional raw-BM25-score threshold.
    language:
        Snowball language for stemming + stopwords on the bm25s backend
        (e.g. ``"english"``, ``"portuguese"``). Stemming measurably
        improves BM25; the old whitespace tokenizer was the worst case
        for morphology-rich languages.

    Implements the Retriever protocol.
    """

    __slots__ = (
        "_backend",
        "_bm25",
        "_bm25s",
        "_items",
        "_language",
        "_min_score",
        "_stemmer",
        "_tokenize_fn",
        "_tokenizer",
    )

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]] | None = None,
        tokenizer: Tokenizer | None = None,
        min_score: float | None = None,
        language: str = "english",
    ) -> None:
        self._tokenize_fn = tokenize_fn or self._default_tokenize
        self._bm25: BM25Okapi | None = None
        self._bm25s: Any = None
        self._items: list[ContextItem] = []
        self._tokenizer = tokenizer or get_default_counter()
        self._min_score = min_score
        self._language = language
        self._stemmer: Any = None
        # Custom tokenize_fn implies the caller controls tokenization ->
        # legacy backend. Otherwise prefer bm25s when installed.
        self._backend = "legacy" if tokenize_fn is not None else "auto"

    def __repr__(self) -> str:
        return (
            f"SparseRetriever(indexed_items={len(self._items)}, "
            f"backend={self._resolved_backend()!r})"
        )

    def _resolved_backend(self) -> str:
        if self._backend == "legacy":
            return "rank_bm25"
        if self._bm25s is not None:
            return "bm25s"
        if self._bm25 is not None:
            return "rank_bm25"
        return "auto"

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization (legacy backend)."""
        return text.lower().split()

    def _get_stemmer(self) -> Any:
        if self._stemmer is None:
            try:
                import Stemmer

                self._stemmer = Stemmer.Stemmer(self._language)
            except ImportError:
                logger.warning(
                    "PyStemmer not installed; bm25s will index without stemming. "
                    "Install it with: pip install astro-anchor[bm25]"
                )
                self._stemmer = False  # sentinel: tried and unavailable
            except KeyError:
                logger.warning(
                    "PyStemmer has no stemmer for language %r; indexing without "
                    "stemming.",
                    self._language,
                )
                self._stemmer = False
        return self._stemmer or None

    def _stopwords(self) -> str | None:
        return _STOPWORD_CODES.get(self._language)

    def index(self, items: list[ContextItem]) -> int:
        """Build the BM25 index from context items (full rebuild)."""
        self._items = list(items)
        texts = [item.content for item in self._items]

        if self._backend != "legacy":
            try:
                import bm25s
            except ImportError:
                bm25s = None  # type: ignore[assignment]
            if bm25s is not None:
                corpus_tokens = bm25s.tokenize(
                    texts,
                    stopwords=self._stopwords(),
                    stemmer=self._get_stemmer(),
                    show_progress=False,
                )
                retriever = bm25s.BM25()
                retriever.index(corpus_tokens, show_progress=False)
                self._bm25s = retriever
                self._bm25 = None
                return len(self._items)

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            msg = (
                "A BM25 backend is required for SparseRetriever. "
                "Install one with: pip install astro-anchor[bm25]"
            )
            raise RetrieverError(msg) from e

        tokenized_corpus = [self._tokenize_fn(item.content) for item in self._items]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25s = None
        return len(self._items)

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items using BM25 scoring."""
        if self._bm25s is not None:
            return self._retrieve_bm25s(query, top_k)
        if self._bm25 is not None:
            return self._retrieve_legacy(query, top_k)
        msg = "Must call index() before retrieve()"
        raise RetrieverError(msg)

    def _retrieve_bm25s(self, query: QueryBundle, top_k: int) -> list[ContextItem]:
        import bm25s

        query_tokens = bm25s.tokenize(
            query.query_str,
            stopwords=self._stopwords(),
            stemmer=self._get_stemmer(),
            show_progress=False,
        )
        k = min(top_k, len(self._items))
        if k == 0:
            return []
        try:
            doc_indices, scores = self._bm25s.retrieve(
                query_tokens, k=k, show_progress=False
            )
        except (ValueError, IndexError):
            # e.g. query reduced to zero tokens after stopword removal
            return []

        pairs = [
            (float(score), int(idx))
            for idx, score in zip(doc_indices[0], scores[0], strict=True)
        ]
        return self._build_items(pairs)

    def _retrieve_legacy(self, query: QueryBundle, top_k: int) -> list[ContextItem]:
        if self._bm25 is None:  # pragma: no cover
            msg = "Must call index() before retrieve()"
            raise RetrieverError(msg)
        tokenized_query = self._tokenize_fn(query.query_str)
        scores = self._bm25.get_scores(tokenized_query)

        if len(scores) == 0:
            return []

        scored_indices = [(float(s), i) for i, s in enumerate(scores)]
        top_entries = heapq.nlargest(top_k, scored_indices)
        return self._build_items(top_entries)

    def _build_items(self, raw_pairs: list[tuple[float, int]]) -> list[ContextItem]:
        """Turn ``(raw_score, index)`` pairs into scored ContextItems.

        The ``score`` field is max-normalized (ContextItem.score is [0, 1]);
        the raw BM25 score is preserved in ``metadata["raw_score"]``.
        """
        positive = [(s, i) for s, i in raw_pairs if s > 0]
        if not positive:
            return []
        max_score = max(s for s, _ in positive)

        items: list[ContextItem] = []
        for raw_score, idx in positive:
            if self._min_score is not None and raw_score < self._min_score:
                continue
            item = self._items[idx]
            scored_item = item.model_copy(update={
                "source": SourceType.RETRIEVAL,
                "score": raw_score / max_score,
                "token_count": item.token_count or self._tokenizer.count_tokens(item.content),
                "metadata": {
                    **item.metadata,
                    "retrieval_method": "sparse_bm25",
                    "raw_score": raw_score,
                },
            })
            items.append(scored_item)
        items.sort(key=lambda x: x.score, reverse=True)
        return items
