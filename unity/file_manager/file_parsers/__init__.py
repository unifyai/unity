"""
`unity.file_manager.file_parsers`
================================

New, modular file parsing subsystem.

Key design points
-----------------
- The parser is format-aware and backend-driven (mirrors Docling's architecture).
- The canonical internal representation is a typed ContentNode graph.
- The public boundary returned to the FileManager is `FileParseResult`.

This package is intentionally self-contained so the FileManager can treat the
parser as a black box with a stable I/O contract.
"""

from __future__ import annotations

from .file_parser import FileParser
from .settings import FILE_PARSER_SETTINGS, FileParserSettings
from .types.backend import BaseFileParserBackend
from .types.contracts import FileParseRequest
from .types.enums import ContentType, NodeKind
from .types.formats import FileFormat, MimeType, extension_to_format, extension_to_mime
from .types.graph import ContentGraph, ContentNode
from .types.json_types import JsonObject, JsonScalar, JsonValue
from .types.contracts import (
    ConversionHop,
    FileParseMetadata,
    FileParseResult,
    FileParseTrace,
    ParseError,
    StepStatus,
    StepTrace,
)
from .types.table import ExtractedTable

__all__ = [
    # facade
    "FileParser",
    # settings
    "FileParserSettings",
    "FILE_PARSER_SETTINGS",
    # typed contracts
    "BaseFileParserBackend",
    "FileParseRequest",
    "FileFormat",
    "MimeType",
    "extension_to_format",
    "extension_to_mime",
    "JsonScalar",
    "JsonValue",
    "JsonObject",
    "ContentGraph",
    "ContentNode",
    "NodeKind",
    "ContentType",
    "ExtractedTable",
    "ConversionHop",
    "FileParseResult",
    "FileParseTrace",
    "FileParseMetadata",
    "StepTrace",
    "StepStatus",
    "ParseError",
]
