"""FastAPI app exposing the RAG pipeline."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

# Silence ChromaDB's broken telemetry calls (known incompatibility with newer
# posthog versions in chromadb==0.5.5). These warnings are cosmetic only.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.indexing import BM25Index, ChromaVectorStore, IndexingPipeline
from src.loaders import UnsupportedFileTypeError, get_loader
from src.pipeline import RAGPipeline

from .schemas import (
    AskRequest,
    AskResponse,
    DocumentRecord,
    HealthResponse,
    IngestResponse,
    ListDocumentsResponse,
)

app = FastAPI(
    title="Helix RAG API",
    description=(
        "Retrieval-Augmented Generation service with hybrid (dense + sparse) "
        "retrieval, citation verification, and confidence scoring over internal "
        "documentation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── shared singletons ────────────────────────────────────
# Lazily constructed because they may need API keys / disk state.
_state: dict[str, Any] = {}


def get_pipeline() -> RAGPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        pipeline = RAGPipeline()
        _state["pipeline"] = pipeline
    return pipeline


def get_indexing_pipeline() -> IndexingPipeline:
    indexer = _state.get("indexer")
    if indexer is None:
        indexer = IndexingPipeline()
        _state["indexer"] = indexer
    return indexer


# ─── routes ───────────────────────────────────────────────
@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    chroma = ChromaVectorStore()
    bm25 = BM25Index()
    bm25.load()
    return HealthResponse(
        status="ok",
        chroma_chunks=chroma.count(),
        bm25_chunks=bm25.count(),
    )


@app.post("/v1/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    response = pipeline.ask(
        payload.question,
        mode=payload.mode,
        use_rerank=payload.use_rerank,
        verify=payload.verify,
    )
    return AskResponse(**response.to_dict())


@app.get("/v1/documents", response_model=ListDocumentsResponse)
def list_documents() -> ListDocumentsResponse:
    chroma = ChromaVectorStore()
    chunks = chroma.all_chunks()
    by_doc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"chunk_count": 0, "strategies": set(), "source": None}
    )
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        name = meta.get("filename") or meta.get("source") or "unknown"
        rec = by_doc[name]
        rec["chunk_count"] += 1
        if meta.get("chunking_strategy"):
            rec["strategies"].add(meta["chunking_strategy"])
        rec["source"] = meta.get("source")

    documents = [
        DocumentRecord(
            filename=name,
            source=rec["source"],
            chunk_count=rec["chunk_count"],
            chunking_strategies=sorted(rec["strategies"]),
        )
        for name, rec in sorted(by_doc.items())
    ]
    return ListDocumentsResponse(count=len(documents), documents=documents)


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for upload in files:
        if not upload.filename:
            continue
        target = settings.raw_dir / Path(upload.filename).name
        target.write_bytes(await upload.read())
        saved_paths.append(target)

    documents = []
    for path in saved_paths:
        try:
            loader = get_loader(path)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        try:
            loaded = loader.load(path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to load {path.name}: {type(exc).__name__}: {exc}",
            ) from exc

        # Persist normalized JSON next to the raw file.
        relative = path.relative_to(settings.raw_dir)
        out_path = settings.processed_dir / relative.with_suffix(
            relative.suffix + ".json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([d.to_dict() for d in loaded], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        documents.extend(loaded)

    indexer = get_indexing_pipeline()
    report = indexer.index(documents)

    # Refresh the QA pipeline's retriever so it sees new chunks.
    _state["pipeline"] = RAGPipeline()

    return IngestResponse(
        saved_files=[str(p.relative_to(settings.raw_dir)) for p in saved_paths],
        indexing=report.as_dict(),
    )
