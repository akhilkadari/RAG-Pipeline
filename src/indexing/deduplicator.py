"""Near-duplicate detection over chunks via cosine similarity."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.chunking import Chunk
from src.config import settings


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return matrix / norms


class Deduplicator:
    """Holds a growing matrix of chunk embeddings to test new chunks against.

    Two layers:
      * exact-hash dedup (cheap, catches 1:1 duplicates by content).
      * cosine-similarity threshold (catches near-duplicates above 0.95).
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else settings.dedup_threshold
        self._matrix: np.ndarray | None = None
        self._hashes: set[str] = set()

    def seed(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        """Bootstrap with already-indexed chunks (so future calls dedup against them)."""
        if not chunks:
            return
        for c in chunks:
            self._hashes.add(c.content_hash())
        new = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def filter(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> tuple[list[Chunk], list[list[float]], list[dict]]:
        """Return (kept_chunks, kept_embeddings, dropped_records)."""
        kept_chunks: list[Chunk] = []
        kept_embs: list[list[float]] = []
        dropped: list[dict] = []

        new_arr = (
            _normalize(np.asarray(embeddings, dtype=np.float32))
            if embeddings
            else np.empty((0, 0), dtype=np.float32)
        )

        for i, chunk in enumerate(chunks):
            chash = chunk.content_hash()
            if chash in self._hashes:
                dropped.append({"chunk_id": chunk.id, "reason": "exact_hash"})
                continue

            vec = new_arr[i] if new_arr.size else None
            max_sim = 0.0

            if vec is not None and self._matrix is not None and len(self._matrix) > 0:
                sims = self._matrix @ vec
                if sims.size:
                    max_sim = float(sims.max())

            if max_sim >= self.threshold:
                dropped.append(
                    {
                        "chunk_id": chunk.id,
                        "reason": "near_duplicate",
                        "max_similarity": max_sim,
                    }
                )
                continue

            kept_chunks.append(chunk)
            kept_embs.append(embeddings[i])
            self._hashes.add(chash)
            if vec is not None:
                self._matrix = (
                    vec.reshape(1, -1)
                    if self._matrix is None
                    else np.vstack([self._matrix, vec.reshape(1, -1)])
                )

        return kept_chunks, kept_embs, dropped
