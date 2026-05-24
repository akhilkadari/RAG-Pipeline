"""Reciprocal Rank Fusion (RRF) of dense and sparse result lists.

  rrf_score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

The constant `k` (default 60) dampens the effect of very high ranks; weights
let you bias toward semantic vs keyword retrieval.
"""
from __future__ import annotations

from typing import Any

from src.config import settings


class RRFFuser:
    def __init__(
        self,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        k_constant: int | None = None,
    ) -> None:
        self.dense_weight = (
            dense_weight if dense_weight is not None else settings.rrf_dense_weight
        )
        self.sparse_weight = (
            sparse_weight if sparse_weight is not None else settings.rrf_sparse_weight
        )
        self.k_constant = k_constant if k_constant is not None else settings.rrf_k_constant

    def fuse(
        self,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for hit in dense_hits:
            cid = hit["id"]
            entry = merged.setdefault(cid, _blank_entry(hit))
            entry["dense_rank"] = hit.get("dense_rank")
            entry["dense_score"] = hit.get("dense_score", hit.get("similarity", 0.0))

        for hit in sparse_hits:
            cid = hit["id"]
            entry = merged.setdefault(cid, _blank_entry(hit))
            entry["sparse_rank"] = hit.get("sparse_rank")
            entry["sparse_score"] = hit.get("sparse_score", hit.get("score", 0.0))

        for entry in merged.values():
            score = 0.0
            if entry.get("dense_rank") is not None:
                score += self.dense_weight / (self.k_constant + entry["dense_rank"])
            if entry.get("sparse_rank") is not None:
                score += self.sparse_weight / (self.k_constant + entry["sparse_rank"])
            entry["rrf_score"] = score

        ranked = sorted(merged.values(), key=lambda e: e["rrf_score"], reverse=True)
        for i, entry in enumerate(ranked, start=1):
            entry["fused_rank"] = i
        return ranked


def _blank_entry(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": hit["id"],
        "text": hit.get("text", ""),
        "metadata": hit.get("metadata", {}),
        "dense_rank": None,
        "dense_score": None,
        "sparse_rank": None,
        "sparse_score": None,
    }
