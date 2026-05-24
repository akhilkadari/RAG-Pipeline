"""Grounded answer generator with bracketed citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config import settings

from .llm_client import LLMClient
from .prompts import GROUNDED_SYSTEM_PROMPT, USER_TEMPLATE, format_context

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NO_INFO_MARKER = "i don't have enough information"


@dataclass
class Answer:
    question: str
    answer_text: str
    citations: list[int] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    is_idk: bool = False
    raw_llm_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer_text,
            "citations": self.citations,
            "chunks": self.chunks,
            "is_idk": self.is_idk,
        }


class GroundedGenerator:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient(model=settings.generation_model)

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> Answer:
        if not chunks:
            return Answer(
                question=question,
                answer_text=(
                    "I don't have enough information in the provided documentation "
                    "to answer that. The retriever returned no relevant context."
                ),
                citations=[],
                chunks=[],
                is_idk=True,
            )

        context_block = format_context(chunks)
        prompt = USER_TEMPLATE.format(question=question, context=context_block)
        text = self._client.chat(
            prompt, system=GROUNDED_SYSTEM_PROMPT, temperature=0.0, max_tokens=1024
        )

        citations = _extract_citations(text, max_index=len(chunks))
        is_idk = _NO_INFO_MARKER in text.lower()

        return Answer(
            question=question,
            answer_text=text,
            citations=citations,
            chunks=chunks,
            is_idk=is_idk,
            raw_llm_output=text,
        )


def _extract_citations(text: str, *, max_index: int) -> list[int]:
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for piece in match.group(1).split(","):
            try:
                n = int(piece.strip())
            except ValueError:
                continue
            if 1 <= n <= max_index and n not in seen:
                seen.append(n)
    return seen
