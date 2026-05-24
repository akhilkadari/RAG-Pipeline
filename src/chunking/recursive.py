"""Recursive character chunker that prefers structural boundaries.

Splits first on heading-like markers, then paragraphs, then sentences,
then words, then chars — falling back only when needed to fit chunk_size.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loaders import Document

from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    name = "recursive"

    # Order matters: try strongest boundaries first.
    _SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            separators=self._SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
        )

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = self._splitter.split_text(doc.text)
            for idx, piece in enumerate(pieces):
                cleaned = piece.strip()
                if not cleaned:
                    continue
                chunks.append(self._make_chunk(cleaned, doc, idx))
        return chunks
