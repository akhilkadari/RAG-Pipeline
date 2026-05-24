"""Tests for Reciprocal Rank Fusion."""
from __future__ import annotations

from src.retrieval.fusion import RRFFuser


def _hit(cid: str, *, dense_rank=None, sparse_rank=None) -> dict:
    return {
        "id": cid,
        "text": cid,
        "metadata": {},
        "dense_rank": dense_rank,
        "sparse_rank": sparse_rank,
        "dense_score": None if dense_rank is None else 1.0 / dense_rank,
        "sparse_score": None if sparse_rank is None else 10.0 / sparse_rank,
    }


def test_rrf_combines_lists() -> None:
    fuser = RRFFuser(dense_weight=1.0, sparse_weight=1.0, k_constant=60)
    dense = [_hit("a", dense_rank=1), _hit("b", dense_rank=2)]
    sparse = [_hit("b", sparse_rank=1), _hit("c", sparse_rank=2)]
    fused = fuser.fuse(dense, sparse)
    ids = [f["id"] for f in fused]
    # b appears in both -> highest score.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
    for entry in fused:
        assert "rrf_score" in entry
        assert "fused_rank" in entry


def test_rrf_respects_weights() -> None:
    # Heavy dense weighting should rank a > c (only in sparse list at rank 1).
    fuser = RRFFuser(dense_weight=10.0, sparse_weight=0.01, k_constant=60)
    dense = [_hit("a", dense_rank=1)]
    sparse = [_hit("c", sparse_rank=1)]
    fused = fuser.fuse(dense, sparse)
    assert fused[0]["id"] == "a"


def test_rrf_handles_empty_inputs() -> None:
    fuser = RRFFuser()
    assert fuser.fuse([], []) == []
    assert fuser.fuse([_hit("a", dense_rank=1)], [])[0]["id"] == "a"
