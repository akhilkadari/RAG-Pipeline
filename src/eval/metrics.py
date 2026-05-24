"""Eval data types + per-case metric computation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import (
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
)


@dataclass
class EvalCase:
    id: str
    category: str
    question: str
    expected_answer: str
    expected_sources: list[str] = field(default_factory=list)


@dataclass
class MetricBundle:
    correctness: float
    faithfulness: float
    retrieval_relevance: float
    citation_accuracy: float
    citation_coverage: float
    confidence_composite: float
    is_idk: bool
    expected_idk: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    citations: list[int]
    chunks_used: list[dict[str, Any]]
    metrics: MetricBundle
    raw_response: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.id,
            "category": self.case.category,
            "question": self.case.question,
            "expected_answer": self.case.expected_answer,
            "expected_sources": self.case.expected_sources,
            "answer": self.answer,
            "citations": self.citations,
            "chunks_used": [
                {
                    "filename": c.get("metadata", {}).get("filename"),
                    "section": c.get("metadata", {}).get("section_heading"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in self.chunks_used
            ],
            "metrics": self.metrics.as_dict(),
        }


class CaseEvaluator:
    """Computes per-case metrics using LLM-as-judge for correctness & faithfulness."""

    def __init__(self, judge: LLMClient | None = None) -> None:
        self._judge = judge or LLMClient(model=settings.judge_model)

    def evaluate(self, case: EvalCase, response: dict[str, Any]) -> MetricBundle:
        answer = response.get("answer", "")
        chunks = response.get("chunks", [])
        citation_report = response.get("citation_report", {}) or {}
        confidence = response.get("confidence", {}) or {}
        is_idk = bool(response.get("is_idk", False))
        expected_idk = case.category == "no_answer"

        # Correctness vs golden
        correctness = self._correctness(case, answer, expected_idk, is_idk)

        # Faithfulness: claims grounded in retrieved context
        faithfulness = self._faithfulness(answer, chunks)

        # Retrieval relevance: did expected_sources appear in retrieved chunk filenames?
        retrieval_relevance = _retrieval_relevance(case.expected_sources, chunks)

        return MetricBundle(
            correctness=correctness,
            faithfulness=faithfulness,
            retrieval_relevance=retrieval_relevance,
            citation_accuracy=float(citation_report.get("accuracy", 0.0)),
            citation_coverage=float(citation_report.get("coverage", 0.0)),
            confidence_composite=float(confidence.get("composite", 0.0)),
            is_idk=is_idk,
            expected_idk=expected_idk,
        )

    # ─── judge helpers ─────────────────────────────────────
    def _correctness(
        self,
        case: EvalCase,
        system_answer: str,
        expected_idk: bool,
        is_idk: bool,
    ) -> float:
        # Special handling: for no-answer cases, full credit if the system declined.
        if expected_idk:
            return 1.0 if is_idk else 0.0
        # If the system declined when an answer existed, partial credit only if it
        # surfaced relevant docs — we conservatively give 0 here.
        if is_idk:
            return 0.0

        result = self._judge.json(
            CORRECTNESS_PROMPT.format(
                question=case.question,
                golden=case.expected_answer,
                system=system_answer,
            ),
            temperature=0.0,
            max_tokens=200,
        )
        score = result.get("score")
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0

    def _faithfulness(self, answer: str, chunks: list[dict[str, Any]]) -> float:
        if not answer.strip() or not chunks:
            return 0.0
        # Build a context block (smallish) from the chunks.
        ctx_parts = []
        for i, c in enumerate(chunks, start=1):
            ctx_parts.append(f"[{i}] {(c.get('text') or '')[:1500]}")
        context = "\n\n---\n\n".join(ctx_parts)
        result = self._judge.json(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer),
            temperature=0.0,
            max_tokens=300,
        )
        score = result.get("score")
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0


def _retrieval_relevance(
    expected_sources: list[str], chunks: list[dict[str, Any]]
) -> float:
    if not expected_sources:
        # If there's no expected source, retrieval relevance is undefined; return 1.0
        # so it doesn't penalize no-answer cases.
        return 1.0
    retrieved = {
        (c.get("metadata", {}) or {}).get("filename") for c in chunks
    } - {None}
    expected = set(expected_sources)
    if not expected:
        return 1.0
    return len(expected & retrieved) / len(expected)
