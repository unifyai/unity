from __future__ import annotations

import csv
import io
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from unity.file_manager.file_parsers.implementations.native.spreadsheet_support import (
    finalize_spreadsheet_result,
    normalize_tabular_value,
    should_inline_tabular_rows,
)
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.types.contracts import (
    FileParseRequest,
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.file_parsers.utils.tracing import traced_step

logger = logging.getLogger(__name__)


class NativeCsvBackend(BaseFileParserBackend):
    """Streaming CSV backend backed by polars lazy scans."""

    name = "native_csv_backend"
    supported_formats: Sequence[FileFormat] = (FileFormat.CSV,)

    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        return fmt in self.supported_formats

    @staticmethod
    def iter_rows(
        path: str | Path,
        *,
        delimiter: str = ",",
        quotechar: str = '"',
        has_header: bool = True,
        encoding: str = "utf8-lossy",
        columns: Sequence[str] | None = None,
        batch_size: int = 5_000,
    ) -> "Iterator[dict[str, object]]":
        """Stream rows from a CSV via Polars lazy scan.

        This is the single row-iteration entry point for CSV files.
        Callers that hold a ``CsvFileHandle`` should unpack it into
        keyword arguments and call this directly rather than
        reimplementing CSV reading.
        """
        polars_encoding = (
            "utf8-lossy" if encoding not in ("utf8", "utf8-lossy") else encoding
        )
        lf = pl.scan_csv(
            str(path),
            separator=delimiter,
            quote_char=quotechar,
            has_header=has_header,
            encoding=polars_encoding,
            infer_schema_length=500,
            try_parse_dates=True,
        )
        override = list(columns or [])
        for batch_df in lf.collect(streaming=True).iter_slices(n_rows=batch_size):
            if override and len(override) == len(batch_df.columns):
                batch_df.columns = override
            for row_dict in batch_df.iter_rows(named=True):
                yield {str(k): normalize_tabular_value(v) for k, v in row_dict.items()}

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        started = time.perf_counter()
        path = Path(ctx.source_local_path).expanduser().resolve()

        trace = FileParseTrace(
            logical_path=str(ctx.logical_path),
            backend=self.name,
            file_format=ctx.file_format,
            mime_type=ctx.mime_type,
            status=StepStatus.SUCCESS,
            source_local_path=str(path),
            parsed_local_path=str(path),
        )

        if not path.exists() or not path.is_file():
            trace.status = StepStatus.FAILED
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error=f"File not found: {path}",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
            )

        try:
            with traced_step(trace, name="detect_csv_dialect") as step:
                dialect_info = _detect_csv_dialect(path)
                step.counters["sample_bytes"] = dialect_info["sample_bytes"]

            with traced_step(trace, name="scan_csv_schema") as step:
                lazy_frame = pl.scan_csv(
                    str(path),
                    separator=dialect_info["delimiter"],
                    quote_char=dialect_info["quotechar"],
                    has_header=bool(dialect_info["has_header"]),
                    encoding="utf8-lossy",
                    infer_schema_length=500,
                    try_parse_dates=True,
                )
                schema = lazy_frame.collect_schema()
                columns = [str(name) for name in schema.names()]
                step.counters["columns"] = len(columns)

            with traced_step(trace, name="scan_csv_profile") as step:
                row_count = int(lazy_frame.select(pl.len()).collect().item())
                sample_limit = max(int(FILE_PARSER_SETTINGS.TABULAR_SAMPLE_ROWS), 0)
                sample_rows = (
                    _normalize_row_dicts(
                        lazy_frame.head(sample_limit).collect().to_dicts(),
                    )
                    if sample_limit > 0
                    else []
                )
                inline_rows = []
                if should_inline_tabular_rows(
                    row_count=row_count,
                    settings=FILE_PARSER_SETTINGS,
                ):
                    inline_rows = _normalize_row_dicts(lazy_frame.collect().to_dicts())
                step.counters["rows"] = row_count
                step.counters["sample_rows"] = len(sample_rows)
                step.counters["inline_rows"] = len(inline_rows)

            sheet_name = (path.stem or "Sheet 1").strip() or "Sheet 1"
            table = ExtractedTable(
                table_id="table:1",
                label=sheet_name,
                sheet_name=sheet_name,
                columns=columns,
                rows=inline_rows,
                sample_rows=sample_rows,
                num_rows=row_count,
                num_cols=len(columns),
            )

            result = finalize_spreadsheet_result(
                logical_path=str(ctx.logical_path),
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
                settings=FILE_PARSER_SETTINGS,
                tables=[table],
                sheet_names=[sheet_name],
            )
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return result

        except Exception as e:
            logger.exception("Native CSV parse failed: %s", e)
            trace.status = StepStatus.FAILED
            trace.warnings.append(str(e))
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error=str(e),
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
            )


def _detect_csv_dialect(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        sample_bytes = fh.read(65536)
    encoding = "utf-8"
    sample_text = ""
    for candidate in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sample_text = sample_bytes.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if not sample_text:
        sample_text = sample_bytes.decode("utf-8", errors="replace")

    sniffed = None
    has_header = True
    try:
        sniffed = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        has_header = bool(csv.Sniffer().has_header(sample_text))
    except csv.Error:
        sniffed = csv.get_dialect("excel")

    delimiter = getattr(sniffed, "delimiter", ",") or ","
    quotechar = getattr(sniffed, "quotechar", '"') or '"'
    preview_rows = _preview_csv_rows(
        sample_text,
        delimiter=delimiter,
        quotechar=quotechar,
    )
    has_header = has_header or _looks_like_header(preview_rows)
    return {
        "delimiter": delimiter,
        "quotechar": quotechar,
        "encoding": encoding,
        "has_header": has_header,
        "sample_bytes": len(sample_bytes),
    }


def _normalize_row_dicts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                str(key): normalize_tabular_value(value)
                for key, value in dict(row).items()
            },
        )
    return normalized


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
