"""End-to-end RAG pipeline orchestrator: retrieve -> generate -> verify -> score."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import settings
from src.generation import (
    Answer,
    CitationVerifier,
    ConfidenceScorer,
    GroundedGenerator,
)
from src.generation.citation_verifier import CitationReport
from src.generation.confidence import ConfidenceBreakdown
from src.retrieval import HybridRetriever, RetrievalResult


@dataclass
class RAGResponse:
    question: str
    answer: str
    is_idk: bool
    citations: list[int]
    chunks: list[dict[str, Any]]
    retrieval: dict[str, Any]
    citation_report: dict[str, Any]
    confidence: dict[str, Any]
    fallback: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "is_idk": self.is_idk,
            "citations": self.citations,
            "chunks": self.chunks,
            "retrieval": self.retrieval,
            "citation_report": self.citation_report,
            "confidence": self.confidence,
            "fallback": self.fallback,
            "metadata": self.metadata,
        }


class RAGPipeline:
    """High-level Q&A orchestrator."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: GroundedGenerator | None = None,
        verifier: CitationVerifier | None = None,
        scorer: ConfidenceScorer | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or GroundedGenerator()
        self.verifier = verifier or CitationVerifier()
        self.scorer = scorer or ConfidenceScorer()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )

    def ask(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        use_rerank: bool = True,
        verify: bool = True,
    ) -> RAGResponse:
        retrieval = self.retriever.retrieve(
            question, mode=mode, use_rerank=use_rerank
        )

        # ─── Below-threshold short-circuit (IDK) ───────────
        retrieval_strength = retrieval.avg_rerank_score or _avg_rrf(retrieval)
        if retrieval_strength < self.confidence_threshold:
            return self._build_idk_response(
                question, retrieval, retrieval_strength
            )

        answer: Answer = self.generator.generate(question, retrieval.final_chunks)

        if answer.is_idk:
            return self._build_idk_from_model(question, retrieval, answer, retrieval_strength)

        citation_report: CitationReport
        if verify:
            citation_report = self.verifier.verify(
                answer.answer_text, retrieval.final_chunks
            )
        else:
            citation_report = CitationReport(
                checks=[],
                sentences_with_citation=0,
                total_sentences=0,
            )

        confidence: ConfidenceBreakdown = self.scorer.score(
            question, answer.answer_text, retrieval.final_chunks, citation_report
        )

        return RAGResponse(
            question=question,
            answer=answer.answer_text,
            is_idk=False,
            citations=answer.citations,
            chunks=_summarize_chunks(retrieval.final_chunks),
            retrieval=_summarize_retrieval(retrieval),
            citation_report=citation_report.as_dict(),
            confidence=confidence.as_dict(),
            metadata={"mode": mode, "use_rerank": use_rerank},
        )

    # ─── IDK helpers ───────────────────────────────────────
    def _build_idk_response(
        self,
        question: str,
        retrieval: RetrievalResult,
        strength: float,
    ) -> RAGResponse:
        nearby_docs = _document_summary(retrieval.fused or retrieval.final_chunks)
        message = (
            "I don't have enough information in the indexed documentation to answer that "
            "with confidence. "
        )
        if nearby_docs:
            message += "Documents that might be worth checking manually: " + ", ".join(
                nearby_docs
            )

        return RAGResponse(
            question=question,
            answer=message,
            is_idk=True,
            citations=[],
            chunks=_summarize_chunks(retrieval.final_chunks),
            retrieval=_summarize_retrieval(retrieval),
            citation_report=CitationReport(
                checks=[], sentences_with_citation=0, total_sentences=0
            ).as_dict(),
            confidence={
                "retrieval": strength,
                "citation_coverage": 0.0,
                "citation_accuracy": 0.0,
                "completeness": 0.0,
                "composite": 0.0,
                "completeness_missing": [],
            },
            fallback={
                "reason": "below_confidence_threshold",
                "threshold": self.confidence_threshold,
                "retrieval_strength": strength,
                "candidate_documents": nearby_docs,
            },
        )

    def _build_idk_from_model(
        self,
        question: str,
        retrieval: RetrievalResult,
        answer: Answer,
        strength: float,
    ) -> RAGResponse:
        nearby_docs = _document_summary(retrieval.fused or retrieval.final_chunks)
        return RAGResponse(
            question=question,
            answer=answer.answer_text,
            is_idk=True,
            citations=answer.citations,
            chunks=_summarize_chunks(retrieval.final_chunks),
            retrieval=_summarize_retrieval(retrieval),
            citation_report=CitationReport(
                checks=[], sentences_with_citation=0, total_sentences=0
            ).as_dict(),
            confidence={
                "retrieval": strength,
                "citation_coverage": 0.0,
                "citation_accuracy": 0.0,
                "completeness": 0.0,
                "composite": strength * 0.3,
                "completeness_missing": [],
            },
            fallback={
                "reason": "model_declined",
                "threshold": self.confidence_threshold,
                "retrieval_strength": strength,
                "candidate_documents": nearby_docs,
            },
        )


# ─── helpers ───────────────────────────────────────────────
def _avg_rrf(retrieval: RetrievalResult) -> float:
    if not retrieval.final_chunks:
        return 0.0
    scores = [c.get("rrf_score", c.get("similarity", 0.0)) for c in retrieval.final_chunks]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _summarize_retrieval(retrieval: RetrievalResult) -> dict[str, Any]:
    return {
        "dense_count": len(retrieval.dense_hits),
        "sparse_count": len(retrieval.sparse_hits),
        "fused_count": len(retrieval.fused),
        "final_count": len(retrieval.final_chunks),
        "used_rerank": retrieval.used_rerank,
        "avg_rerank_score": retrieval.avg_rerank_score,
    }


def _summarize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        summary.append(
            {
                "index": i,
                "id": c.get("id"),
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}),
                "rerank_score": c.get("rerank_score"),
                "rrf_score": c.get("rrf_score"),
                "dense_score": c.get("dense_score"),
                "sparse_score": c.get("sparse_score"),
            }
        )
    return summary


def _document_summary(chunks: list[dict[str, Any]], limit: int = 3) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        meta = c.get("metadata", {})
        name = meta.get("filename") or meta.get("source")
        if name and name not in seen:
            seen.append(name)
        if len(seen) >= limit:
            break
    return seen
