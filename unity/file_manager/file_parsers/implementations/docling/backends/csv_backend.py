from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.implementations.docling.steps.docling_convert import (
    docling_convert,
    new_docling_converter,
)
from unity.file_manager.file_parsers.implementations.docling.steps.docling_graph import (
    build_spreadsheet_graph_from_docling,
)
from unity.file_manager.file_parsers.utils.tracing import traced_step
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.contracts import FileParseRequest
from unity.file_manager.file_parsers.types.enums import NodeKind
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.contracts import (
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.file_parsers.utils.format_policy import (
    bound_spreadsheet_full_text,
    build_spreadsheet_profile_text,
    fallback_spreadsheet_summary,
)

logger = logging.getLogger(__name__)


class CsvBackend(BaseFileParserBackend):
    """Docling-backed CSV parser backend."""

    name = "csv_backend"
    supported_formats: Sequence[FileFormat] = (FileFormat.CSV,)

    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        return fmt in self.supported_formats

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
            with traced_step(trace, name="docling_converter_init"):
                converter = new_docling_converter(settings=FILE_PARSER_SETTINGS)

            with traced_step(trace, name="docling_convert"):
                conv = docling_convert(
                    converter=converter,
                    source=str(path),
                    settings=FILE_PARSER_SETTINGS,
                )
                if conv.warnings:
                    trace.warnings.extend(list(conv.warnings))
                if conv.status == "partial_success":
                    trace.status = StepStatus.DEGRADED
                if not conv.ok or conv.document is None:
                    raise RuntimeError(
                        (
                            conv.error.message
                            if conv.error
                            else "Docling convert failed"
                        ),
                    )
                docling_doc = conv.document

            with traced_step(trace, name="build_spreadsheet_graph"):
                built = build_spreadsheet_graph_from_docling(docling_doc)

            del docling_doc, conv, converter
            gc.collect()

            # Normalize the single-sheet name to the file stem for intuitive contexts.
            sheet_name = (path.stem or "Sheet 1").strip() or "Sheet 1"
            try:
                if len(built.sheet_names) == 1 and built.sheet_names[0] == "Sheet 1":
                    for n in built.graph.nodes.values():
                        if n.kind != NodeKind.SHEET:
                            continue
                        n.title = sheet_name
                        try:
                            n.meta["sheet_name"] = sheet_name
                        except Exception:
                            pass
                        try:
                            if n.payload is not None and hasattr(
                                n.payload,
                                "sheet_name",
                            ):
                                n.payload.sheet_name = sheet_name  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if len(built.tables) == 1:
                        built.tables[0].sheet_name = sheet_name
                        built.tables[0].label = sheet_name
            except Exception as e:
                trace.warnings.append(f"sheet_name_normalization_failed: {e}")

            # Do NOT export full_text for spreadsheets (can be huge and low value)
            profile_text = build_spreadsheet_profile_text(
                logical_path=str(ctx.logical_path),
                tables=list(built.tables or []),
                # CSV is treated as a single-sheet spreadsheet; rely on table.sheet_name (normalized above).
                sheet_names=None,
            )
            full_text = bound_spreadsheet_full_text(
                profile_text=profile_text,
                settings=FILE_PARSER_SETTINGS,
            )
            summary = fallback_spreadsheet_summary(
                logical_path=str(ctx.logical_path),
                tables=list(built.tables or []),
                sheet_names=None,
            )

            # Counters
            trace.counters["nodes"] = len(built.graph.nodes)
            trace.counters["tables"] = len(built.tables)
            trace.counters["sheets"] = len(built.sheet_names)

            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="success",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                tables=built.tables,
                summary=summary,
                full_text=full_text,
                trace=trace,
                graph=built.graph,
            )

        except Exception as e:
            logger.exception("CSV parse failed: %s", e)
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
