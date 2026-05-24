"""Citation verifier: parses each (claim, citation) pair and judges support."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.config import settings

from .llm_client import LLMClient
from .prompts import CITATION_JUDGE_PROMPT

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass
class CitationCheck:
    claim: str
    cited_indexes: list[int]
    supported: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "cited_indexes": self.cited_indexes,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass
class CitationReport:
    checks: list[CitationCheck]
    sentences_with_citation: int
    total_sentences: int

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.checks if c.supported)

    @property
    def coverage(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.sentences_with_citation / self.total_sentences

    @property
    def accuracy(self) -> float:
        if not self.checks:
            return 0.0
        return self.supported_count / len(self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "supported_count": self.supported_count,
            "total_checks": len(self.checks),
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "sentences_with_citation": self.sentences_with_citation,
            "total_sentences": self.total_sentences,
        }


class CitationVerifier:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient(model=settings.judge_model)

    def verify(self, answer_text: str, chunks: list[dict[str, Any]]) -> CitationReport:
        sentences = _split_sentences(answer_text)
        sentences_with_citation = 0
        checks: list[CitationCheck] = []

        for sentence in sentences:
            citations = _citations_in(sentence, max_index=len(chunks))
            if citations:
                sentences_with_citation += 1
                claim = _CITATION_RE.sub("", sentence).strip()
                if not claim:
                    continue
                passages = "\n\n---\n\n".join(
                    f"[{i}] " + (chunks[i - 1].get("text") or "") for i in citations
                )
                judged = self._client.json(
                    CITATION_JUDGE_PROMPT.format(claim=claim, passage=passages),
                    temperature=0.0,
                    max_tokens=200,
                )
                checks.append(
                    CitationCheck(
                        claim=claim,
                        cited_indexes=citations,
                        supported=bool(judged.get("supported", False)),
                        reason=str(judged.get("reason", "")),
                    )
                )

        return CitationReport(
            checks=checks,
            sentences_with_citation=sentences_with_citation,
            total_sentences=len(sentences),
        )


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _citations_in(sentence: str, *, max_index: int) -> list[int]:
    out: list[int] = []
    for match in _CITATION_RE.finditer(sentence):
        for p in match.group(1).split(","):
            try:
                n = int(p.strip())
            except ValueError:
                continue
            if 1 <= n <= max_index and n not in out:
                out.append(n)
    return out
