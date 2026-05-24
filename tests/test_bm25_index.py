"""BM25 index unit tests."""
from __future__ import annotations

from pathlib import Path

from src.indexing import BM25Index


def test_bm25_returns_relevant_chunk(tmp_path: Path) -> None:
    idx = BM25Index(persist_path=tmp_path / "bm25.pkl")
    idx.build(
        ids=["a", "b", "c"],
        texts=[
            "The Helix API rate limit is 600 requests per minute.",
            "Vacation policy at Helix is unlimited.",
            "Records that fail validation are returned with their index.",
        ],
        metadatas=[{}, {}, {}],
    )
    hits = idx.query("rate limit per minute", top_k=2)
    assert hits
    assert hits[0]["id"] == "a"


def test_bm25_save_load_roundtrip(tmp_path: Path) -> None:
    """BM25 needs multiple docs for IDF to be informative; this test uses a small corpus."""
    idx = BM25Index(persist_path=tmp_path / "bm25.pkl")
    idx.build(
        ids=["x", "y", "z"],
        texts=[
            "unique zzz keyword present here",
            "completely different content here",
            "another unrelated document text",
        ],
        metadatas=[{}, {}, {}],
    )
    idx.save()
    fresh = BM25Index(persist_path=tmp_path / "bm25.pkl")
    assert fresh.load()
    hits = fresh.query("zzz", top_k=1)
    assert hits and hits[0]["id"] == "x"
