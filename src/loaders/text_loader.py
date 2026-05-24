from __future__ import annotations

from pathlib import Path

from .base import BaseLoader, Document

class TextLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = (".txt",)

    def load(self, path: Path) -> list[Document]:
        raw = path.read_text(encoding="utf-8")
        text = self._normalize(raw)
        metadata = self._base_metadata(path)
        return [Document(text=text, metadata=metadata)]
    
    @staticmethod
    def _normalize(raw: str) -> str:
        # convert windows line endings to unix line endings
        normalized= raw.replace("\r\n", "\n").replace("\r", "\n")
        # remove trailing whitespace from every line
        lines = [line.rstrip() for line in normalized.split('\n')]
        # collapse whole thing - drop leading/trailing blank lines
        return "\n".join(lines).strip()