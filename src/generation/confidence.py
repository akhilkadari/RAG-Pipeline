"""Composite confidence scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import settings

from .citation_verifier import CitationReport
from .llm_client import LLMClient
from .prompts import COMPLETENESS_PROMPT


@dataclass
class ConfidenceBreakdown:
    retrieval: float
    citation_coverage: float
    citation_accuracy: float
    completeness: float
    composite: float
    completeness_missing: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval,
            "citation_coverage": self.citation_coverage,
            "citation_accuracy": self.citation_accuracy,
            "completeness": self.completeness,
            "composite": self.composite,
            "completeness_missing": self.completeness_missing,
        }


class ConfidenceScorer:
    """Combine retrieval, citation, and completeness signals into one score."""

    WEIGHTS = {
        "retrieval": 0.30,
        "citation_coverage": 0.20,
        "citation_accuracy": 0.30,
        "completeness": 0.20,
    }

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient(model=settings.judge_model)

    def score(
        self,
        question: str,
        answer_text: str,
        chunks: list[dict[str, Any]],
        citation_report: CitationReport,
    ) -> ConfidenceBreakdown:
        retrieval = _retrieval_confidence(chunks)
        completeness, missing = self._completeness(question, answer_text)

        composite = (
            self.WEIGHTS["retrieval"] * retrieval
            + self.WEIGHTS["citation_coverage"] * citation_report.coverage
            + self.WEIGHTS["citation_accuracy"] * citation_report.accuracy
            + self.WEIGHTS["completeness"] * completeness
        )

        return ConfidenceBreakdown(
            retrieval=retrieval,
            citation_coverage=citation_report.coverage,
            citation_accuracy=citation_report.accuracy,
            completeness=completeness,
            composite=composite,
            completeness_missing=missing,
        )

    def _completeness(self, question: str, answer: str) -> tuple[float, list[str]]:
        if not answer.strip():
            return 0.0, []
        result = self._client.json(
            COMPLETENESS_PROMPT.format(question=question, answer=answer),
            temperature=0.0,
            max_tokens=200,
        )
        score = result.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        missing = result.get("missing_parts") or []
        if not isinstance(missing, list):
            missing = [str(missing)]
        return score, [str(m) for m in missing]


def _retrieval_confidence(chunks: list[dict[str, Any]]) -> float:
    """Average rerank score (or RRF score, or similarity) of the final chunks."""
    if not chunks:
        return 0.0
    vals: list[float] = []
    for c in chunks:
        for key in ("rerank_score", "similarity", "dense_score", "rrf_score"):
            v = c.get(key)
            if v is not None:
                vals.append(float(v))
                break
    if not vals:
        return 0.0
    return max(0.0, min(1.0, sum(vals) / len(vals)))
