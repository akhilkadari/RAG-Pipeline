"""Hybrid retriever orchestrating dense + sparse + fusion + rerank."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import settings

from .dense import DenseRetriever
from .fusion import RRFFuser
from .reranker import LLMReranker
from .sparse import SparseRetriever


@dataclass
class RetrievalResult:
    query: str
    final_chunks: list[dict[str, Any]]
    dense_hits: list[dict[str, Any]] = field(default_factory=list)
    sparse_hits: list[dict[str, Any]] = field(default_factory=list)
    fused: list[dict[str, Any]] = field(default_factory=list)
    used_rerank: bool = False
    avg_rerank_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "final_chunks": self.final_chunks,
            "dense_hits": self.dense_hits,
            "sparse_hits": self.sparse_hits,
            "fused": self.fused,
            "used_rerank": self.used_rerank,
            "avg_rerank_score": self.avg_rerank_score,
        }


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        fuser: RRFFuser | None = None,
        reranker: LLMReranker | None = None,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.fuser = fuser or RRFFuser()
        self._reranker = reranker  # lazy

    def retrieve(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rerank_input: int | None = None,
        rerank_output: int | None = None,
        use_rerank: bool = True,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """`mode` ∈ {"hybrid", "dense", "sparse"}. Useful for ablations."""
        dense_top_k = dense_top_k or settings.dense_top_k
        sparse_top_k = sparse_top_k or settings.sparse_top_k
        rerank_input = rerank_input or settings.rerank_input
        rerank_output = rerank_output or settings.rerank_output

        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []

        if mode in ("hybrid", "dense"):
            dense_hits = self.dense.retrieve(query, top_k=dense_top_k)
        if mode in ("hybrid", "sparse"):
            sparse_hits = self.sparse.retrieve(query, top_k=sparse_top_k)

        if mode == "dense":
            fused = [
                {
                    **h,
                    "rrf_score": h.get("dense_score", 0.0),
                    "fused_rank": i + 1,
                }
                for i, h in enumerate(dense_hits)
            ]
        elif mode == "sparse":
            fused = [
                {
                    **h,
                    "rrf_score": h.get("sparse_score", 0.0),
                    "fused_rank": i + 1,
                }
                for i, h in enumerate(sparse_hits)
            ]
        else:
            fused = self.fuser.fuse(dense_hits, sparse_hits)

        candidates = fused[:rerank_input]

        if use_rerank and candidates:
            if self._reranker is None:
                self._reranker = LLMReranker()
            final = self._reranker.rerank(query, candidates, top_k=rerank_output)
        else:
            final = candidates[:rerank_output]
            for i, c in enumerate(final, start=1):
                c["final_rank"] = i
                c["rerank_score"] = c.get("rrf_score", 0.0)

        avg = (
            sum(c.get("rerank_score", 0.0) for c in final) / len(final)
            if final
            else 0.0
        )

        return RetrievalResult(
            query=query,
            final_chunks=final,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused=fused,
            used_rerank=use_rerank,
            avg_rerank_score=avg,
        )
