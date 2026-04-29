from __future__ import annotations
from pathlib import Path
from .base import BaseLoader
from .html_loader import HTMLLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader

_LOADERS: tuple[type[BaseLoader], ...] = (
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    PDFLoader,
)

class UnsupportedFileTypeError(ValueError):
    pass

def get_loader(path: Path) -> BaseLoader:
    # get the file extension and convert it to lowercase
    ext = path.suffix.lower()
    # loop through each loader class and check if the extension is supported
    for loader_cls in _LOADERS:
        # if the extension is supported, return the loader class
        if ext in loader_cls.SUPPORTED_EXTENSIONS:
            return loader_cls()
    # if no loader class is found, raise an error
    raise UnsupportedFileTypeError(f"Unsupported file type: {ext} (file: {path.name})")