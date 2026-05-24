"""Sparse keyword retriever (BM25)."""
from __future__ import annotations

from typing import Any

from src.indexing import BM25Index


class SparseRetriever:
    def __init__(self, bm25_index: BM25Index | None = None) -> None:
        self.bm25_index = bm25_index or BM25Index()
        self.bm25_index.load()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        hits = self.bm25_index.query(query, top_k=top_k)
        for rank, hit in enumerate(hits, start=1):
            hit["sparse_rank"] = rank
            hit["sparse_score"] = hit.get("score", 0.0)
            hit["source"] = "sparse"
        return hits
