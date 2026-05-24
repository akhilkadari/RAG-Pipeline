"""Factory selecting a chunker by name."""
from __future__ import annotations

from typing import Literal

from src.config import settings

from .base import BaseChunker
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

ChunkerName = Literal["fixed_size", "recursive", "semantic"]


def get_chunker(
    name: ChunkerName | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embed_fn=None,
) -> BaseChunker:
    """Return a chunker instance by name.

    `embed_fn` is required when `name == "semantic"`.
    """
    name = (name or settings.default_chunker).lower()  # type: ignore[assignment]
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if name == "fixed_size":
        return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    if name == "recursive":
        return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
    if name == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunker requires embed_fn")
        return SemanticChunker(embed_fn=embed_fn)
    raise ValueError(f"Unknown chunker: {name!r}")
