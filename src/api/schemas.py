"""Pydantic request/response schemas for the FastAPI service."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── /v1/ask ──────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    use_rerank: bool = True
    verify: bool = True


class CitationCheckOut(BaseModel):
    claim: str
    cited_indexes: list[int]
    supported: bool
    reason: str


class CitationReportOut(BaseModel):
    checks: list[CitationCheckOut]
    supported_count: int
    total_checks: int
    coverage: float
    accuracy: float


class ConfidenceOut(BaseModel):
    retrieval: float
    citation_coverage: float
    citation_accuracy: float
    completeness: float
    composite: float
    completeness_missing: list[str] = []


class ChunkOut(BaseModel):
    index: int
    id: Optional[str] = None
    text: str
    metadata: dict[str, Any] = {}
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class FallbackOut(BaseModel):
    reason: str
    threshold: float
    retrieval_strength: float
    candidate_documents: list[str] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    is_idk: bool
    citations: list[int]
    chunks: list[ChunkOut]
    retrieval: dict[str, Any]
    citation_report: CitationReportOut
    confidence: ConfidenceOut
    fallback: Optional[FallbackOut] = None
    metadata: dict[str, Any] = {}


# ─── /v1/documents ────────────────────────────────────────
class DocumentRecord(BaseModel):
    filename: str
    source: Optional[str] = None
    chunk_count: int
    chunking_strategies: list[str] = []


class ListDocumentsResponse(BaseModel):
    count: int
    documents: list[DocumentRecord]


# ─── /v1/ingest ───────────────────────────────────────────
class IngestResponse(BaseModel):
    saved_files: list[str]
    indexing: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    chroma_chunks: int
    bm25_chunks: int
