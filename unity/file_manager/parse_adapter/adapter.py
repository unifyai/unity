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

Transport handle construction (CSV dialect sniffing, handle selection, optional
materialisation) is delegated to the shared
:func:`unity.common.pipeline.transport.build_table_handles` helper so that the
same logic is available to both the FM adapter and the DM ingest script.
"""

from dataclasses import dataclass
from typing import List

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.common.pipeline import (
    LocalArtifactStore,
    ParsedFileBundle,
    build_table_handles,
)
from unity.file_manager.types.config import FilePipelineConfig
from unity.file_manager.types.file import FileContentRow

from .lowering.content_rows import lower_graph_to_content_rows


@dataclass(frozen=True)
class FileManagerIngestPayload:
    """FileManager-owned ingest payloads derived from a `FileParseResult`."""

    content_rows: List[FileContentRow]
    tables: List[ExtractedTable]
    bundle: ParsedFileBundle
    document_summary: str = ""


def adapt_parse_result_for_file_manager(
    parse_result: FileParseResult,
    *,
    config: FilePipelineConfig,
) -> FileManagerIngestPayload:
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

    tables = list(getattr(parse_result, "tables", []) or [])
    bundle = _build_parsed_file_bundle(parse_result, config=config)

    if getattr(parse_result, "status", "error") != "success":
        return FileManagerIngestPayload(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )

    graph = getattr(parse_result, "graph", None)
    if graph is None:
        return FileManagerIngestPayload(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )

    try:
        lowered_content = lower_graph_to_content_rows(
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
        return FileManagerIngestPayload(
            content_rows=list(lowered_content.rows or []),
            tables=tables,
            bundle=bundle,
            document_summary=str(lowered_content.document_summary or ""),
        )
    except Exception:
        return FileManagerIngestPayload(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )


def _build_parsed_file_bundle(
    parse_result: FileParseResult,
    *,
    config: FilePipelineConfig,
) -> ParsedFileBundle:
    artifact_store = _build_artifact_store(config)
    table_inputs = build_table_handles(
        parse_result,
        artifact_store=artifact_store,
        artifact_format=config.transport.artifact_format,
    )
    return ParsedFileBundle(result=parse_result, table_inputs=table_inputs)


def _build_artifact_store(config: FilePipelineConfig) -> LocalArtifactStore | None:
    if getattr(config.transport, "table_input_mode", "source_reference") != (
        "materialized_artifact"
    ):
        return None
    return LocalArtifactStore(root_dir=config.transport.artifact_root_dir)
