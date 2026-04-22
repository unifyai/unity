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
from typing import TYPE_CHECKING, List

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.common.pipeline import (
    IngestPlan,
    LocalArtifactStore,
    ParsedFileBundle,
    TableMeta,
    build_table_handles,
)
from unity.file_manager.types.config import FilePipelineConfig
from unity.file_manager.types.file import FileContentRow

from .lowering.content_rows import lower_graph_to_content_rows

if TYPE_CHECKING:
    from unity.common.pipeline.artifact_store import ArtifactStore


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


# ---------------------------------------------------------------------------
# Parse -> ingest plan lowering
# ---------------------------------------------------------------------------


def lower_to_ingest_plan(
    parse_result: FileParseResult,
    *,
    run_id: str,
    config: FilePipelineConfig,
    artifact_store: "ArtifactStore",
    artifact_format: str = "jsonl",
    source_gs_uri: str = "",
) -> IngestPlan:
    """Lower a ``FileParseResult`` into a pointer-only ``IngestPlan``.

    This is the canonical parse-to-ingest boundary used by the production
    parse worker to hand work to the ingest worker without ever shipping
    heavy payloads (document graphs, full table rows) through the queue
    manifest.

    Steps
    -----
    1. Build per-table ``TableInputHandle`` objects via the shared
       :func:`unity.common.pipeline.transport.build_table_handles` helper.
       When the store is provided, inline rows are materialised into
       GCS-backed ``ObjectStoreArtifactHandle`` so the ingest worker
       streams rows out-of-band.
    2. Run ``lower_graph_to_content_rows`` against the document graph and
       materialise the resulting ``FileContentRow`` stream via
       :meth:`ArtifactStore.materialize_content_rows`.  This keeps the
       ``DocumentGraph`` (potentially tens of MB for long PDFs) out of the
       manifest entirely.
    3. Build a stripped ``parse_summary`` (``graph=None``, ``full_text=""``,
       ``tables=[]``) that still contains every field
       ``FileRecord.to_file_record_entry`` needs -- status, error,
       file_format, mime_type, summary, metadata, trace.
    4. Attach per-table ``TableMeta`` rows so the ingest worker can
       provision contexts and resolve embed columns without rehydrating
       the full ``ExtractedTable`` list.

    The function is intentionally total (never raises on lowering failure)
    -- a partially empty plan is still a valid plan; the ingest worker
    will record per-table failures in the run ledger.
    """
    logical_path = str(getattr(parse_result, "logical_path", "") or "")
    parse_status = str(getattr(parse_result, "status", "error") or "error")

    table_inputs = build_table_handles(
        parse_result,
        artifact_store=artifact_store,
        artifact_format=artifact_format,
        job_id=run_id,
        source_gs_uri=source_gs_uri,
    )
    tables_meta = _build_tables_meta(parse_result)

    content_rows_handle = None
    document_summary = ""
    if parse_status == "success":
        graph = getattr(parse_result, "graph", None)
        if graph is not None:
            try:
                lowered = lower_graph_to_content_rows(
                    graph=graph,
                    file_path=logical_path,
                    file_format=getattr(parse_result, "file_format", None),
                    tables=list(getattr(parse_result, "tables", []) or []),
                    business_contexts=getattr(
                        getattr(config, "ingest", None),
                        "business_contexts",
                        None,
                    ),
                )
                document_summary = str(lowered.document_summary or "")
                content_rows = list(lowered.rows or [])
                if content_rows:
                    content_rows_handle = artifact_store.materialize_content_rows(
                        content_rows,
                        logical_path=logical_path,
                        artifact_format=artifact_format,
                        job_id=run_id,
                    )
            except Exception:
                content_rows_handle = None
                document_summary = ""

    parse_summary = _make_parse_summary(parse_result)

    return IngestPlan(
        run_id=run_id,
        file_path=logical_path,
        parse_status="success" if parse_status == "success" else "error",
        parse_summary=parse_summary,
        document_summary=document_summary,
        content_rows_handle=content_rows_handle,
        tables_meta=tables_meta,
        table_inputs=table_inputs,
    )


def _build_tables_meta(parse_result: FileParseResult) -> List[TableMeta]:
    out: List[TableMeta] = []
    for table in list(getattr(parse_result, "tables", []) or []):
        out.append(
            TableMeta(
                table_id=str(getattr(table, "table_id", "") or ""),
                label=str(getattr(table, "label", "") or ""),
                columns=list(getattr(table, "columns", []) or []),
                row_count=getattr(table, "num_rows", None),
                sheet_name=getattr(table, "sheet_name", None),
                table_summary=getattr(table, "table_summary", None),
            ),
        )
    return out


def _make_parse_summary(pr: FileParseResult) -> FileParseResult:
    """Return a ``FileParseResult`` stripped of all heavy fields.

    Drops:
      - ``graph`` (potentially tens of MB for long documents)
      - ``full_text`` (can mirror the full body of a PDF/DOCX)
      - ``tables`` (row payloads already referenced via handles)

    Keeps: ``logical_path``, ``status``, ``error``, ``file_format``,
    ``mime_type``, ``summary``, ``metadata``, ``trace`` so the ingest
    worker can call ``FileRecord.to_file_record_entry`` unchanged.
    """
    return pr.model_copy(update={"graph": None, "full_text": "", "tables": []})
