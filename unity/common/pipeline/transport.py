"""Shared table-handle construction for the parse-to-ingest boundary.

Given a ``FileParseResult``, ``build_table_handles`` produces the correct
``TableInputHandle`` for every extracted table:

- Small tables with rows already in memory -> ``InlineRowsHandle``
- CSV files with deferred rows            -> ``CsvFileHandle``
- XLSX sheets with deferred rows          -> ``XlsxSheetHandle``

If an ``ArtifactStore`` is provided and the caller requests materialisation,
handles are additionally persisted as durable artifacts (JSONL today,
Parquet/Arrow later) and replaced with ``ObjectStoreArtifactHandle``.

Both the FM adapter and the DM ingest script call this function so the
transport logic is defined once.  A future GCS adapter only needs to
implement the same ``ArtifactStore`` protocol -- no changes here.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from .types import (
    CsvFileHandle,
    InlineRowsHandle,
    ObjectStoreArtifactHandle,
    TableInputHandle,
    XlsxSheetHandle,
)

if TYPE_CHECKING:
    from .artifact_store import ArtifactStore
    from unity.file_manager.file_parsers.types.contracts import FileParseResult
    from unity.file_manager.file_parsers.types.table import ExtractedTable


def build_table_handles(
    parse_result: "FileParseResult",
    *,
    artifact_store: Optional["ArtifactStore"] = None,
    artifact_format: str = "jsonl",
) -> Dict[str, TableInputHandle]:
    """Build transport handles for every table in *parse_result*.

    Parameters
    ----------
    parse_result:
        The semantic parse output whose ``tables`` will be examined.
    artifact_store:
        When provided **and** a handle is materialisable, the handle is
        written to the store and replaced with an
        ``ObjectStoreArtifactHandle``.  Pass ``None`` (the default) to
        keep source-reference handles.
    artifact_format:
        Format used when materialising (``"jsonl"`` today).

    Returns
    -------
    dict[str, TableInputHandle]
        Mapping from ``table_id`` to the appropriate handle.
    """
    tables: list[ExtractedTable] = list(getattr(parse_result, "tables", []) or [])
    file_format = getattr(parse_result, "file_format", None)
    trace = getattr(parse_result, "trace", None)
    source_local_path = _resolve_source_local_path(trace)
    storage_uri = _to_storage_uri(source_local_path)
    logical_path = str(getattr(parse_result, "logical_path", "") or "")
    parse_succeeded = getattr(parse_result, "status", "error") == "success"

    result: Dict[str, TableInputHandle] = {}
    for index, table in enumerate(tables, start=1):
        table_id = str(getattr(table, "table_id", f"table:{index}"))
        handle = _handle_for_table(
            table,
            file_format=file_format,
            source_local_path=source_local_path,
            storage_uri=storage_uri,
            logical_path=logical_path,
        )

        if artifact_store is not None and parse_succeeded and _can_materialize(handle):
            handle = artifact_store.materialize_table_input(
                handle,
                logical_path=logical_path,
                table_id=table_id,
                artifact_format=artifact_format,
            )

        result[table_id] = handle

    return result


# ---------------------------------------------------------------------------
# Per-table handle selection
# ---------------------------------------------------------------------------


def _handle_for_table(
    table: "ExtractedTable",
    *,
    file_format: object,
    source_local_path: str,
    storage_uri: str,
    logical_path: str,
) -> TableInputHandle:
    rows = list(getattr(table, "rows", []) or [])
    row_count = getattr(table, "num_rows", None)
    if row_count is None:
        row_count = len(rows)
    columns = list(getattr(table, "columns", []) or [])

    if rows:
        return InlineRowsHandle(
            rows=rows,
            columns=columns,
            row_count=row_count,
        )

    fmt_name = _format_name(file_format)

    if fmt_name == "csv" and source_local_path:
        stored = getattr(table, "csv_dialect", None)
        if stored is not None:
            encoding = str(stored.encoding)
            delimiter = str(stored.delimiter)
            quotechar = str(stored.quotechar)
            has_header = bool(stored.has_header)
        else:
            dialect = detect_csv_dialect(Path(source_local_path))
            encoding = str(dialect["encoding"])
            delimiter = str(dialect["delimiter"])
            quotechar = str(dialect["quotechar"])
            has_header = bool(dialect["has_header"])
        return CsvFileHandle(
            storage_uri=storage_uri,
            logical_path=logical_path,
            source_local_path=source_local_path,
            columns=columns,
            encoding=encoding,
            delimiter=delimiter,
            quotechar=quotechar,
            has_header=has_header,
            row_count=row_count,
        )

    if fmt_name == "xlsx" and source_local_path and getattr(table, "sheet_name", None):
        return XlsxSheetHandle(
            storage_uri=storage_uri,
            logical_path=logical_path,
            source_local_path=source_local_path,
            sheet_name=str(table.sheet_name),
            columns=columns,
            has_header=True,
            row_count=row_count,
        )

    return InlineRowsHandle(rows=[], columns=columns, row_count=row_count)


def _format_name(file_format: object) -> str:
    """Normalise a file format enum or string to a lowercase name."""
    if file_format is None:
        return ""
    name = getattr(file_format, "value", None) or str(file_format)
    return str(name).lower().strip()


# ---------------------------------------------------------------------------
# CSV dialect detection (moved from adapter.py for reuse)
# ---------------------------------------------------------------------------


def detect_csv_dialect(path: Path) -> dict[str, object]:
    """Sniff encoding, delimiter, quotechar and header presence."""
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_source_local_path(trace: object) -> str:
    if trace is None:
        return ""
    return str(
        getattr(trace, "parsed_local_path", None)
        or getattr(trace, "source_local_path", None)
        or "",
    )


def _to_storage_uri(source_local_path: str) -> str:
    if not source_local_path:
        return ""
    return Path(source_local_path).expanduser().resolve().as_uri()


def _can_materialize(handle: TableInputHandle) -> bool:
    if isinstance(handle, ObjectStoreArtifactHandle):
        return False
    if isinstance(handle, InlineRowsHandle):
        return bool(handle.rows)
    return True


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
