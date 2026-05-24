"""Sparse keyword index over chunks using BM25."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi

from src.config import settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization. Good enough for technical docs."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    """In-memory BM25 with pickle persistence under settings.index_dir."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self.persist_path = persist_path or (settings.index_dir / "bm25.pkl")
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []

    # ─── lifecycle ─────────────────────────────────────────
    def build(
        self,
        ids: Iterable[str],
        texts: Iterable[str],
        metadatas: Iterable[dict[str, Any]],
    ) -> None:
        self._ids = list(ids)
        self._texts = list(texts)
        self._metadatas = list(metadatas)
        self._tokenized = [tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("wb") as f:
            pickle.dump(
                {
                    "ids": self._ids,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                    "tokenized": self._tokenized,
                },
                f,
            )

    def load(self) -> bool:
        if not self.persist_path.exists():
            return False
        with self.persist_path.open("rb") as f:
            payload = pickle.load(f)
        self._ids = payload["ids"]
        self._texts = payload["texts"]
        self._metadatas = payload["metadatas"]
        self._tokenized = payload["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        return True

    # ─── queries ───────────────────────────────────────────
    def query(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self._bm25 or not self._ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": self._ids[i],
                "text": self._texts[i],
                "metadata": self._metadatas[i],
                "score": float(scores[i]),
            }
            for i in order
            if scores[i] > 0
        ]

    def count(self) -> int:
        return len(self._ids)
