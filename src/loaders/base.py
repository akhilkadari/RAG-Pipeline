from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

class Document(BaseModel):
    text: str = Field(..., description="Cleaned plaintext content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source info: file path, section, page, timestamps, etc."
    )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}

class BaseLoader(ABC):
    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        raise NotImplementedError

    @staticmethod
    def _base_metadata(path: Path) -> dict[str, Any]:
        return {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lstrip(".").lower(),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }