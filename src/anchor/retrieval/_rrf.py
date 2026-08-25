"""Reciprocal Rank Fusion (RRF) utility for combining ranked result lists."""

from __future__ import annotations

import logging

from anchor.models.context import ContextItem

logger = logging.getLogger(__name__)

__all__ = ["rrf_fuse"]


def rrf_fuse(
    ranked_lists: list[list[ContextItem]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int | None = None,
    retrieval_method: str = "rrf",
) -> list[ContextItem]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = sum(weight_i / (k + rank_i(d))) for each list i

    Parameters:
        ranked_lists: Lists of items ranked by relevance.
        weights: Per-list weights (defaults to equal weight 1.0).
        k: Smoothing constant (default 60, from original RRF paper).
        top_k: Maximum number of items to return. If None, return all.
        retrieval_method: Label recorded in each item's metadata.

    Returns:
        Fused list of items sorted by RRF score. The ``score`` field is
        min-max normalized across the returned set (ContextItem.score is
        constrained to [0, 1]); the raw RRF score is preserved in
        ``metadata["rrf_raw_score"]``.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        msg = "weights must have same length as ranked_lists"
        raise ValueError(msg)

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, ContextItem] = {}

    for ranking, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + weight / (k + rank)
            if item.id not in item_map or item.score > item_map[item.id].score:
                item_map[item.id] = item

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    if not sorted_ids:
        return []

    max_rrf = rrf_scores[sorted_ids[0]]
    min_rrf = rrf_scores[sorted_ids[-1]] if len(sorted_ids) > 1 else 0.0
    score_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

    fused_results: list[ContextItem] = []
    for item_id in sorted_ids:
        original = item_map[item_id]
        normalized_score = (
            (rrf_scores[item_id] - min_rrf) / score_range if score_range > 0 else 1.0
        )
        fused_item = original.model_copy(
            update={
                "score": min(1.0, max(0.0, normalized_score)),
                "metadata": {
                    **original.metadata,
                    "retrieval_method": retrieval_method,
                    "rrf_raw_score": rrf_scores[item_id],
                },
            }
        )
        fused_results.append(fused_item)
    return fused_results
