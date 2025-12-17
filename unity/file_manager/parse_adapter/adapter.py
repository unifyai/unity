from __future__ import annotations

"""FileManager-side adapter for parser outputs.

The FileParser returns a parse-only `FileParseResult` containing artifacts such as:
- `graph` (ContentGraph)
- `tables` (ExtractedTable)
- `trace`/`metadata`

The FileManager, however, needs *ingestion-ready* payloads:
- `/Content/` rows (hierarchical navigation surface)
- per-table context rows under `/Tables/<label>`

This module performs that transformation and is the only place where parser
artifacts are converted into FileManager ingestion inputs.
"""

from dataclasses import dataclass
from typing import List

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.types.config import FilePipelineConfig
from unity.file_manager.types.file import FileContentRow

from .lowering.content_rows import lower_graph_to_content_rows


@dataclass(frozen=True)
class AdaptedParseOutput:
    """Ingestion-ready payloads derived from a `FileParseResult`."""

    content_rows: List[FileContentRow]
    tables: List[ExtractedTable]
    document_summary: str = ""


def adapt_parse_result_for_file_manager(
    parse_result: FileParseResult,
    *,
    config: FilePipelineConfig,
) -> AdaptedParseOutput:
    """
    Adapt parse artifacts into FileManager ingestion inputs.

    Notes
    -----
    - This function is intentionally *best-effort* and **never raises**.
      The FileManager pipeline must remain robust: ingestion should still proceed
      for other files even if lowering fails for one file.
    - The adapter does not mutate `parse_result`; it only derives FileManager-owned
      payloads from it.

    Invariants
    ----------
    - When `parse_result.status != 'success'`, the adapter returns empty rows but
      still forwards extracted `tables` (if any) for debugging/visibility.
    - `/Content/` rows are always typed `FileContentRow` objects.
    - Heavy table data should not be duplicated into `/Content/` text fields;
      raw rows belong in `/Tables/<label>`.
    """

    # Tables are already a strict, ingestion-ready typed payload from the parser boundary.
    tables = list(getattr(parse_result, "tables", []) or [])

    if getattr(parse_result, "status", "error") != "success":
        return AdaptedParseOutput(content_rows=[], tables=tables, document_summary="")

    graph = getattr(parse_result, "graph", None)
    if graph is None:
        return AdaptedParseOutput(content_rows=[], tables=tables, document_summary="")

    try:
        low = lower_graph_to_content_rows(
            graph=graph,
            file_path=str(getattr(parse_result, "logical_path", "") or ""),
            file_format=getattr(parse_result, "file_format", None),
            tables=tables,
            business_contexts=getattr(
                getattr(config, "ingest", None),
                "business_contexts",
                None,
            ),
        )
        return AdaptedParseOutput(
            content_rows=list(low.rows or []),
            tables=tables,
            document_summary=str(low.document_summary or ""),
        )
    except Exception:
        return AdaptedParseOutput(content_rows=[], tables=tables, document_summary="")
