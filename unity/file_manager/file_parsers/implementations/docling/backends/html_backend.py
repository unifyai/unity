from __future__ import annotations

from typing import Sequence

from unity.file_manager.file_parsers.implementations.docling.backends.base_document_backend import (
    BaseDocumentBackend,
)
from unity.file_manager.file_parsers.types.formats import FileFormat


class HtmlBackend(BaseDocumentBackend):
    """Docling-backed HTML parser backend (best-effort; falls back to text when needed)."""

    name = "html_backend"
    supported_formats: Sequence[FileFormat] = (FileFormat.HTML,)
    allow_text_fallback_on_convert_failure = True
