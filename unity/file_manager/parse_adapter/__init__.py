"""
`unity.file_manager.parse_adapter`
=================================

This package is the explicit boundary between:
- the **FileParser** (which produces strictly typed parse artifacts), and
- the **FileManager** (which owns storage/ingestion schemas and I/O behavior).

Responsibilities
----------------
- Convert parser outputs (`FileParseResult`) into FileManager ingestion inputs:
  - `/Content/` rows (hierarchical navigation surface)
  - per-table `/Tables/<label>` payloads (already in `ExtractedTable.rows`)

The adapter is intentionally hosted under `unity.file_manager` so the separation
of concerns is obvious: the parser does not need to know anything about how the
FileManager stores or ingests content.
"""

from __future__ import annotations

from .adapter import FileManagerIngestPayload, adapt_parse_result_for_file_manager

__all__ = [
    "FileManagerIngestPayload",
    "adapt_parse_result_for_file_manager",
]
