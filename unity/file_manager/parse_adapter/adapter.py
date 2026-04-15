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

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List

from unity.file_manager.file_parsers.types.contracts import (
    FileParseResult,
    FileParseTrace,
)
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.pipeline import (
    CsvFileHandle,
    InlineRowsHandle,
    LocalArtifactStore,
    ObjectStoreArtifactHandle,
    ParsedFileBundle,
    XlsxSheetHandle,
)
from unity.file_manager.types.config import FilePipelineConfig
from unity.file_manager.types.file import FileContentRow

from .lowering.content_rows import lower_graph_to_content_rows


@dataclass(frozen=True)
class AdaptedParseOutput:
    """Ingestion-ready payloads derived from a `FileParseResult`."""

    content_rows: List[FileContentRow]
    tables: List[ExtractedTable]
    bundle: ParsedFileBundle
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

    # Tables remain semantic at the parser boundary; transport handles live in the bundle.
    tables = list(getattr(parse_result, "tables", []) or [])
    bundle = _build_parsed_file_bundle(parse_result, tables=tables, config=config)

    if getattr(parse_result, "status", "error") != "success":
        return AdaptedParseOutput(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )

    graph = getattr(parse_result, "graph", None)
    if graph is None:
        return AdaptedParseOutput(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )

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
            bundle=bundle,
            document_summary=str(low.document_summary or ""),
        )
    except Exception:
        return AdaptedParseOutput(
            content_rows=[],
            tables=tables,
            bundle=bundle,
            document_summary="",
        )


def _build_parsed_file_bundle(
    parse_result: FileParseResult,
    *,
    tables: List[ExtractedTable],
    config: FilePipelineConfig,
) -> ParsedFileBundle:
    table_inputs = {}
    fmt = getattr(parse_result, "file_format", None)
    trace = getattr(parse_result, "trace", None)
    source_local_path = _resolve_source_local_path(trace)
    storage_uri = _to_storage_uri(source_local_path)
    logical_path = str(getattr(parse_result, "logical_path", "") or "")
    artifact_store = _build_artifact_store(config)
    parse_succeeded = getattr(parse_result, "status", "error") == "success"

    for index, table in enumerate(tables, start=1):
        table_id = str(getattr(table, "table_id", f"table:{index}"))
        rows = list(getattr(table, "rows", []) or [])
        row_count = getattr(table, "num_rows", None)
        if row_count is None:
            row_count = len(rows)
        columns = list(getattr(table, "columns", []) or [])

        if rows:
            handle = InlineRowsHandle(
                rows=rows,
                columns=columns,
                row_count=row_count,
            )
        elif fmt == FileFormat.CSV and source_local_path:
            dialect = _detect_csv_dialect(Path(source_local_path))
            handle = CsvFileHandle(
                storage_uri=storage_uri,
                logical_path=str(getattr(parse_result, "logical_path", "") or ""),
                source_local_path=source_local_path,
                columns=columns,
                encoding=str(dialect["encoding"]),
                delimiter=str(dialect["delimiter"]),
                quotechar=str(dialect["quotechar"]),
                has_header=bool(dialect["has_header"]),
                row_count=row_count,
            )
        elif (
            fmt == FileFormat.XLSX
            and source_local_path
            and getattr(table, "sheet_name", None)
        ):
            handle = XlsxSheetHandle(
                storage_uri=storage_uri,
                logical_path=str(getattr(parse_result, "logical_path", "") or ""),
                source_local_path=source_local_path,
                sheet_name=str(table.sheet_name),
                columns=columns,
                has_header=True,
                row_count=row_count,
            )
        else:
            handle = InlineRowsHandle(
                rows=[],
                columns=columns,
                row_count=row_count,
            )

        if (
            artifact_store is not None
            and parse_succeeded
            and _can_materialize_handle(handle)
        ):
            handle = artifact_store.materialize_table_input(
                handle,
                logical_path=logical_path,
                table_id=table_id,
                artifact_format=config.transport.artifact_format,
            )

        table_inputs[table_id] = handle

    return ParsedFileBundle(result=parse_result, table_inputs=table_inputs)


def _resolve_source_local_path(trace: FileParseTrace | None) -> str:
    if trace is None:
        return ""
    return str(trace.parsed_local_path or trace.source_local_path or "")


def _to_storage_uri(source_local_path: str) -> str:
    if not source_local_path:
        return ""
    return Path(source_local_path).expanduser().resolve().as_uri()


def _build_artifact_store(config: FilePipelineConfig) -> LocalArtifactStore | None:
    if getattr(config.transport, "table_input_mode", "source_reference") != (
        "materialized_artifact"
    ):
        return None
    return LocalArtifactStore(root_dir=config.transport.artifact_root_dir)


def _can_materialize_handle(
    handle: (
        InlineRowsHandle | CsvFileHandle | XlsxSheetHandle | ObjectStoreArtifactHandle
    ),
) -> bool:
    if isinstance(handle, InlineRowsHandle):
        return bool(handle.rows)
    return True


def _detect_csv_dialect(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        sample_bytes = fh.read(65536)

    sample_text = ""
    encoding = "utf-8"
    for candidate in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sample_text = sample_bytes.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue

    if not sample_text:
        sample_text = sample_bytes.decode("utf-8", errors="replace")

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        has_header = bool(csv.Sniffer().has_header(sample_text))
    except csv.Error:
        dialect = csv.get_dialect("excel")
        has_header = True

    delimiter = getattr(dialect, "delimiter", ",") or ","
    quotechar = getattr(dialect, "quotechar", '"') or '"'
    preview_rows = _preview_csv_rows(
        sample_text,
        delimiter=delimiter,
        quotechar=quotechar,
    )
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "quotechar": quotechar,
        "has_header": has_header or _looks_like_header(preview_rows),
    }


def _preview_csv_rows(
    sample_text: str,
    *,
    delimiter: str,
    quotechar: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    reader = csv.reader(
        io.StringIO(sample_text),
        delimiter=delimiter,
        quotechar=quotechar,
    )
    for row in reader:
        stripped = [str(cell).strip() for cell in row]
        if not any(stripped):
            continue
        rows.append(stripped)
        if len(rows) >= 3:
            break
    return rows


def _looks_like_header(rows: list[list[str]]) -> bool:
    if not rows:
        return True
    first_row = rows[0]
    if not first_row:
        return False
    non_empty = [cell for cell in first_row if cell]
    if len(non_empty) != len(first_row):
        return False
    if len({cell.casefold() for cell in non_empty}) != len(non_empty):
        return False
    return all(_is_header_label(cell) for cell in first_row)


def _is_header_label(cell: str) -> bool:
    text = str(cell).strip()
    if not text:
        return False
    if text[0].isdigit():
        return False
    if _is_numeric_like(text):
        return False
    return any(char.isalpha() for char in text)


def _is_numeric_like(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    normalized = text.replace(",", "").replace("_", "")
    try:
        float(normalized)
        return True
    except ValueError:
        return False
