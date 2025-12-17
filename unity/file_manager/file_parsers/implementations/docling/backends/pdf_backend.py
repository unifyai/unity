from __future__ import annotations

from typing import Sequence

from unity.file_manager.file_parsers.implementations.docling.backends.base_document_backend import (
    BaseDocumentBackend,
)
from unity.file_manager.file_parsers.types.formats import FileFormat


class PdfBackend(BaseDocumentBackend):
    """Docling-backed PDF parser backend."""

    name = "pdf_backend"
    supported_formats: Sequence[FileFormat] = (FileFormat.PDF,)
    allow_text_fallback_on_convert_failure = False
