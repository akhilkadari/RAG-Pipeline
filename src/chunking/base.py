"""Base abstractions for chunking strategies."""
from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.loaders import Document


class Chunk(BaseModel):
    """A single retrieval unit produced from a parent Document."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash of the chunk's text for dedup / change detection."""
        return hashlib.sha1(self.text.strip().encode("utf-8")).hexdigest()


class BaseChunker(ABC):
    """Contract every chunker must satisfy."""

    name: str = "base"

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError

    def _make_chunk(
        self,
        text: str,
        parent: Document,
        chunk_index: int,
        extra: dict[str, Any] | None = None,
    ) -> Chunk:
        metadata = {
            **parent.metadata,
            "chunk_index": chunk_index,
            "chunking_strategy": self.name,
            "char_count": len(text),
            "parent_source": parent.metadata.get("source"),
            "section_heading": _nearest_heading(parent, text),
        }
        if extra:
            metadata.update(extra)
        return Chunk(text=text, metadata=metadata)


def _nearest_heading(parent: Document, chunk_text: str) -> str | None:
    """Best-effort: pick the nearest preceding heading from parent metadata."""
    sections = parent.metadata.get("sections")
    if not sections:
        # HTML loader stores headings under a different key
        headings = parent.metadata.get("headings")
        if headings:
            return headings[0]
        return None
    # Find the latest section whose heading text appears in the chunk.
    last_match = None
    for section in sections:
        text = section.get("text") if isinstance(section, dict) else str(section)
        if text and text in chunk_text:
            last_match = text
    if last_match:
        return last_match
    first = sections[0]
    return first.get("text") if isinstance(first, dict) else str(first)
