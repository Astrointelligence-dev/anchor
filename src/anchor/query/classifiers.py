"""Built-in query classification strategies.

Classifiers inspect a ``QueryBundle`` and return a string label
indicating the query category.  They implement the ``QueryClassifier``
protocol and can be used with ``classified_retriever_step`` to route
queries to specialised retrievers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from anchor._math import cosine_similarity as _cosine_similarity
from anchor.models.query import QueryBundle

logger = logging.getLogger(__name__)


class KeywordClassifier:
    """Classifies queries by matching keywords in the query string.

    Rules are evaluated in insertion order; the first matching rule
    wins.  If no rule matches, the ``default`` label is returned.

    Parameters:
        rules: A mapping from class label to a list of keywords.
            A query matches a rule if any keyword is found in the
            query string.
        default: The fallback label when no rule matches.
        case_sensitive: Whether keyword matching is case-sensitive.
    """

    __slots__ = ("_case_sensitive", "_default", "_rules")

    def __init__(
        self,
        rules: dict[str, list[str]],
        default: str,
        case_sensitive: bool = False,
    ) -> None:
        self._rules = rules
        self._default = default
        self._case_sensitive = case_sensitive

    def __repr__(self) -> str:
        labels = list(self._rules.keys())
        return (
            f"KeywordClassifier(labels={labels!r}, "
            f"default={self._default!r}, "
            f"case_sensitive={self._case_sensitive})"
        )

    def classify(self, query: QueryBundle) -> str:
        """Classify by scanning for keyword matches.

        Parameters:
            query: The query bundle to classify.

        Returns:
            The label of the first matching rule, or the default label.
        """
        text = query.query_str if self._case_sensitive else query.query_str.lower()
        for label, keywords in self._rules.items():
            for kw in keywords:
                check = kw if self._case_sensitive else kw.lower()
                if check in text:
                    logger.debug("KeywordClassifier matched label=%r via keyword=%r", label, kw)
                    return label
        logger.debug("KeywordClassifier fell back to default=%r", self._default)
        return self._default


class CallbackClassifier:
    """Classifies queries by delegating to a user-supplied callback.

    Parameters:
        classify_fn: A callable ``(QueryBundle) -> str`` that returns
            the class label for the given query.
    """

    __slots__ = ("_classify_fn",)

    def __init__(self, classify_fn: Callable[[QueryBundle], str]) -> None:
        self._classify_fn = classify_fn

    def __repr__(self) -> str:
        return "CallbackClassifier()"

    def classify(self, query: QueryBundle) -> str:
        """Classify by delegating to the callback.

        Parameters:
            query: The query bundle to classify.

        Returns:
            The string label returned by the callback.
        """
        label = self._classify_fn(query)
        logger.debug("CallbackClassifier returned label=%r", label)
        return label


class EmbeddingClassifier:
    """Classifies queries by comparing embeddings to labelled centroids.

    Assigns the query to the class whose centroid has the highest
    cosine similarity (or custom distance) to the query embedding.

    Parameters:
        centroids: A mapping from class label to centroid embedding.
        distance_fn: An optional callable ``(list[float], list[float]) -> float``
            that returns a similarity score (higher is more similar).
            Defaults to cosine similarity.
    """

    __slots__ = ("_centroids", "_distance_fn")

    def __init__(
        self,
        centroids: dict[str, list[float]],
        distance_fn: Callable[[list[float], list[float]], float] | None = None,
    ) -> None:
        self._centroids = centroids
        self._distance_fn = distance_fn or _cosine_similarity

    def __repr__(self) -> str:
        labels = list(self._centroids.keys())
        return f"EmbeddingClassifier(labels={labels!r})"

    def classify(self, query: QueryBundle) -> str:
        """Classify by embedding similarity to centroids.

        Parameters:
            query: The query bundle to classify. Must have a non-None
                ``embedding``.

        Returns:
            The label of the closest centroid.

        Raises:
            ValueError: If ``query.embedding`` is ``None``.
        """
        if query.embedding is None:
            msg = "EmbeddingClassifier requires query.embedding to be set"
            raise ValueError(msg)

        best_label = ""
        best_score = float("-inf")
        for label, centroid in self._centroids.items():
            score = self._distance_fn(query.embedding, centroid)
            if score > best_score:
                best_score = score
                best_label = label

        logger.debug(
            "EmbeddingClassifier selected label=%r with score=%.4f",
            best_label,
            best_score,
        )
        return best_label
