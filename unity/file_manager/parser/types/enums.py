from __future__ import annotations

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
    UNKNOWN = "unknown"


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
    APPLICATION_OCTET_STREAM = "application/octet-stream"


_EXT_TO_FORMAT = {
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
}

_EXT_TO_MIME = {
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
}


def extension_to_format(ext: str) -> FileFormat:
    try:
        return _EXT_TO_FORMAT.get(ext.lower(), FileFormat.UNKNOWN)
    except Exception:
        return FileFormat.UNKNOWN


def extension_to_mime(ext: str) -> MimeType:
    try:
        return _EXT_TO_MIME.get(ext.lower(), MimeType.APPLICATION_OCTET_STREAM)
    except Exception:
        return MimeType.APPLICATION_OCTET_STREAM
