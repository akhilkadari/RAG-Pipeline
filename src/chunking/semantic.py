"""Semantic chunker: splits on topic boundaries via embedding similarity.

Algorithm:
  1. Split each document into sentences.
  2. Group every sentence with its small neighbourhood (buffer) and embed it.
  3. Compute cosine distance between consecutive grouped embeddings.
  4. Place a chunk boundary wherever distance exceeds the percentile threshold.
  5. Merge resulting groups into chunks bounded by max_chunk_size.

This keeps chunks coherent (inside one topic) without over-splitting.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from src.loaders import Document

from .base import BaseChunker, Chunk

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        embed_fn: EmbedFn,
        buffer_size: int = 1,
        breakpoint_percentile: float = 90.0,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
    ) -> None:
        self.embed_fn = embed_fn
        self.buffer_size = buffer_size
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in documents:
            for idx, text in enumerate(self._split_document(doc.text)):
                cleaned = text.strip()
                if not cleaned:
                    continue
                out.append(self._make_chunk(cleaned, doc, idx))
        return out

    # ─── internals ─────────────────────────────────────────

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]

    def _grouped_sentences(self, sentences: list[str]) -> list[str]:
        # Concatenate each sentence with `buffer_size` neighbours on each side.
        grouped = []
        n = len(sentences)
        for i in range(n):
            lo = max(0, i - self.buffer_size)
            hi = min(n, i + self.buffer_size + 1)
            grouped.append(" ".join(sentences[lo:hi]))
        return grouped

    def _split_document(self, text: str) -> list[str]:
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 2:
            return [text]

        grouped = self._grouped_sentences(sentences)
        embeddings = np.asarray(self.embed_fn(grouped), dtype=np.float32)

        # Cosine distance between consecutive grouped embeddings.
        distances = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
            cosine_sim = float(np.dot(a, b) / denom)
            distances.append(1.0 - cosine_sim)

        if not distances:
            return [text]

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints = [i for i, d in enumerate(distances) if d >= threshold]

        groups: list[str] = []
        start = 0
        for bp in breakpoints:
            end = bp + 1
            groups.append(" ".join(sentences[start:end]))
            start = end
        groups.append(" ".join(sentences[start:]))

        # Enforce max_chunk_size with a hard fallback to char windows;
        # merge tiny groups under min_chunk_size.
        merged: list[str] = []
        buf = ""
        for g in groups:
            if not buf:
                buf = g
                continue
            if len(buf) < self.min_chunk_size:
                buf = (buf + " " + g).strip()
                continue
            merged.append(buf)
            buf = g
        if buf:
            merged.append(buf)

        final: list[str] = []
        for piece in merged:
            if len(piece) <= self.max_chunk_size:
                final.append(piece)
            else:
                # Hard split if a single semantic group is too long.
                step = self.max_chunk_size
                for i in range(0, len(piece), step):
                    final.append(piece[i : i + step])
        return final
