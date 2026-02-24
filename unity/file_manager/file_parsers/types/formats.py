from __future__ import annotations

"""
File format and MIME type enums.

- `unity.file_manager.file_parsers` must not depend on legacy code.
- File formats are part of the strict parser I/O contract and are used by both the
  FileParser and the FileManager (for routing, indexing, and return models).
"""

from enum import Enum


class FileFormat(str, Enum):
    """Canonical file formats supported by the pipeline."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    JPEG = "jpeg"
    PNG = "png"
    UNKNOWN = "unknown"


_IMAGE_FORMATS: frozenset[FileFormat] = frozenset({FileFormat.JPEG, FileFormat.PNG})


def is_image_format(fmt: FileFormat) -> bool:
    """Return True if *fmt* is a raster image handled by the vision path."""
    return fmt in _IMAGE_FORMATS


class MimeType(str, Enum):
    """Common MIME types for supported formats."""

    TEXT_PLAIN = "text/plain"
    TEXT_MARKDOWN = "text/markdown"
    TEXT_CSV = "text/csv"
    TEXT_HTML = "text/html"
    APPLICATION_JSON = "application/json"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_DOCX = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    APPLICATION_DOC = "application/msword"
    APPLICATION_XML = "application/xml"
    APPLICATION_XLS = "application/vnd.ms-excel"
    APPLICATION_XLSX = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    APPLICATION_OCTET_STREAM = "application/octet-stream"


_EXT_TO_FORMAT: dict[str, FileFormat] = {
    ".pdf": FileFormat.PDF,
    ".docx": FileFormat.DOCX,
    ".doc": FileFormat.DOC,
    ".xlsx": FileFormat.XLSX,
    ".xls": FileFormat.XLSX,
    ".csv": FileFormat.CSV,
    ".txt": FileFormat.TXT,
    ".md": FileFormat.TXT,
    ".html": FileFormat.HTML,
    ".htm": FileFormat.HTML,
    ".xml": FileFormat.XML,
    ".json": FileFormat.JSON,
    ".jpg": FileFormat.JPEG,
    ".jpeg": FileFormat.JPEG,
    ".png": FileFormat.PNG,
}

_EXT_TO_MIME: dict[str, MimeType] = {
    ".txt": MimeType.TEXT_PLAIN,
    ".md": MimeType.TEXT_MARKDOWN,
    ".csv": MimeType.TEXT_CSV,
    ".html": MimeType.TEXT_HTML,
    ".htm": MimeType.TEXT_HTML,
    ".json": MimeType.APPLICATION_JSON,
    ".pdf": MimeType.APPLICATION_PDF,
    ".docx": MimeType.APPLICATION_DOCX,
    ".doc": MimeType.APPLICATION_DOC,
    ".xml": MimeType.APPLICATION_XML,
    ".xls": MimeType.APPLICATION_XLS,
    ".xlsx": MimeType.APPLICATION_XLSX,
    ".jpg": MimeType.IMAGE_JPEG,
    ".jpeg": MimeType.IMAGE_JPEG,
    ".png": MimeType.IMAGE_PNG,
}


def extension_to_format(ext: str) -> FileFormat:
    """Map a file extension to a FileFormat enum (unknown if not mapped)."""
    try:
        return _EXT_TO_FORMAT.get((ext or "").lower(), FileFormat.UNKNOWN)
    except Exception:
        return FileFormat.UNKNOWN


def extension_to_mime(ext: str) -> MimeType:
    """Map a file extension to a MimeType enum (octet-stream if not mapped)."""
    try:
        return _EXT_TO_MIME.get((ext or "").lower(), MimeType.APPLICATION_OCTET_STREAM)
    except Exception:
        return MimeType.APPLICATION_OCTET_STREAM
