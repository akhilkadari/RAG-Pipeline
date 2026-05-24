"""Fixed-size chunker with character overlap (the simple baseline)."""
from __future__ import annotations

from langchain_text_splitters import CharacterTextSplitter

from src.loaders import Document

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # separator="" forces hard length-based splitting (no respect for boundaries).
        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = self._splitter.split_text(doc.text)
            for idx, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                chunks.append(self._make_chunk(piece, doc, idx))
        return chunks
