"""Centralised configuration loaded from environment / .env.

All knobs (paths, model names, retrieval k, thresholds) live here so the rest
of the codebase imports `settings` and never reads env vars directly.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Override any field via environment variable."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── LLM provider ──────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    generation_model: str = Field(default="gpt-4o-mini", alias="GENERATION_MODEL")
    judge_model: str = Field(default="gpt-4o-mini", alias="JUDGE_MODEL")

    # ─── Paths ─────────────────────────────────────────────
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw", alias="RAW_DIR")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed", alias="PROCESSED_DIR")
    index_dir: Path = Field(default=PROJECT_ROOT / "data" / "index", alias="INDEX_DIR")
    chroma_collection: str = Field(default="rag_chunks", alias="CHROMA_COLLECTION")

    # ─── Retrieval ─────────────────────────────────────────
    dense_top_k: int = Field(default=10, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=10, alias="SPARSE_TOP_K")
    rrf_dense_weight: float = Field(default=0.7, alias="RRF_DENSE_WEIGHT")
    rrf_sparse_weight: float = Field(default=0.3, alias="RRF_SPARSE_WEIGHT")
    rrf_k_constant: int = Field(default=60, alias="RRF_K_CONSTANT")
    rerank_input: int = Field(default=20, alias="RERANK_INPUT")
    rerank_output: int = Field(default=5, alias="RERANK_OUTPUT")
    dedup_threshold: float = Field(default=0.95, alias="DEDUP_THRESHOLD")
    confidence_threshold: float = Field(default=0.35, alias="CONFIDENCE_THRESHOLD")

    # ─── Chunking ──────────────────────────────────────────
    default_chunker: str = Field(default="recursive", alias="DEFAULT_CHUNKER")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    # ─── API ───────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
