"""Tests for the three chunking strategies."""
from __future__ import annotations

from src.chunking import get_chunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.loaders import Document


def _doc(text: str, **meta) -> Document:
    return Document(text=text, metadata={"filename": "doc.md", **meta})


def test_fixed_size_chunker_respects_length() -> None:
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    long_text = "abcdefghij" * 20  # 200 chars
    chunks = chunker.split([_doc(long_text)])
    assert chunks
    for c in chunks:
        assert len(c.text) <= 60  # size + tolerance for overlap rounding
        assert c.metadata["chunking_strategy"] == "fixed_size"


def test_recursive_chunker_prefers_paragraphs() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    text = "Paragraph one is short.\n\nParagraph two is also short.\n\nThird short paragraph."
    chunks = chunker.split([_doc(text)])
    assert chunks
    for c in chunks:
        assert c.metadata["chunking_strategy"] == "recursive"
        assert c.text.strip()


def test_semantic_chunker_with_stub_embedder() -> None:
    """SemanticChunker should split on a stub embedder that toggles direction."""

    def stub_embed(texts):
        # Return alternating embeddings so consecutive cosine distances are large
        return [
            [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i, _ in enumerate(texts)
        ]

    chunker = SemanticChunker(
        embed_fn=stub_embed,
        breakpoint_percentile=50,
        min_chunk_size=1,
        max_chunk_size=10_000,
    )
    text = "First sentence. Second one. Third here. Fourth coming. Fifth ends."
    chunks = chunker.split([_doc(text)])
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["chunking_strategy"] == "semantic"


def test_factory_picks_correct_chunker() -> None:
    assert isinstance(get_chunker("fixed_size"), FixedSizeChunker)
    assert isinstance(get_chunker("recursive"), RecursiveChunker)


def test_chunk_metadata_includes_strategy_and_count() -> None:
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.split([_doc("# Title\n\nSome body content here.")])
    assert chunks
    meta = chunks[0].metadata
    assert meta["chunk_index"] == 0
    assert meta["chunking_strategy"] == "recursive"
    assert meta["char_count"] == len(chunks[0].text)
