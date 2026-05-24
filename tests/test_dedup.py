"""Tests for the deduplicator."""
from __future__ import annotations

from src.chunking import Chunk
from src.indexing import Deduplicator


def test_exact_duplicates_dropped() -> None:
    dedup = Deduplicator(threshold=0.95)
    chunks = [Chunk(text="hello world"), Chunk(text="hello world")]
    embeddings = [[1.0, 0.0], [0.99, 0.01]]
    kept, _, dropped = dedup.filter(chunks, embeddings)
    assert len(kept) == 1
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "exact_hash"


def test_near_duplicates_dropped_by_cosine() -> None:
    dedup = Deduplicator(threshold=0.95)
    chunks = [Chunk(text="hello world"), Chunk(text="completely different text")]
    embeddings = [[1.0, 0.0], [0.99, 0.01]]
    kept, _, dropped = dedup.filter(chunks, embeddings)
    assert len(kept) == 1
    assert dropped and dropped[0]["reason"] == "near_duplicate"


def test_distinct_chunks_all_kept() -> None:
    dedup = Deduplicator(threshold=0.95)
    chunks = [Chunk(text="alpha"), Chunk(text="beta")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    kept, _, dropped = dedup.filter(chunks, embeddings)
    assert len(kept) == 2
    assert dropped == []
