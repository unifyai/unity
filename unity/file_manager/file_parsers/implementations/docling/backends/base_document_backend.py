from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar, Optional, Sequence

from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.implementations.docling.steps.docling_convert import (
    docling_convert,
    new_docling_converter,
)
from unity.file_manager.file_parsers.implementations.docling.steps.docling_index import (
    index_docling_structure,
)
from unity.file_manager.file_parsers.implementations.docling.steps.document_graph import (
    build_document_graph_fallback,
    build_document_graph_hybrid,
    build_document_graph_from_text,
)
from unity.file_manager.file_parsers.implementations.docling.steps.document_enrichment import (
    generate_hierarchical_summaries,
)
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.contracts import (
    FileParseRequest,
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.utils.format_policy import (
    extract_metadata_from_text_best_effort,
)
from unity.file_manager.file_parsers.utils.tracing import traced_step

logger = logging.getLogger(__name__)


class BaseDocumentBackend(BaseFileParserBackend):
    """
    Shared Docling-backed backend for document-like formats.

    This class contains the **common Docling pipeline**:
    - convert input file to a DoclingDocument
    - build a ContentGraph (hybrid with fallbacks)
    - export full_text
    - best-effort enrichment (summaries + metadata)

    Format-specific backends (PDF/HTML/XML/JSON) should subclass this and only
    override:
    - `name`
    - `supported_formats`
    - `allow_text_fallback_on_convert_failure`
    """

    name: ClassVar[str] = "base_document_backend"
    supported_formats: ClassVar[Sequence[FileFormat]] = ()

    # For formats where Docling conversion may reasonably fail for common real-world
    # files (e.g., arbitrary JSON/XML), we allow a plain-text fallback rather than
    # erroring the parse.
    allow_text_fallback_on_convert_failure: ClassVar[bool] = False

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

            docling_doc = None
            with traced_step(trace, name="docling_convert") as step:
                conv = docling_convert(converter=converter, source=str(path))
                if not conv.ok or conv.document is None:
                    if self.allow_text_fallback_on_convert_failure:
                        step.status = StepStatus.DEGRADED
                        step.warnings.append(
                            f"docling_convert_failed_fallback_text: {conv.error.message if conv.error else 'unknown'}",
                        )
                        docling_doc = None
                    else:
                        raise RuntimeError(
                            (
                                conv.error.message
                                if conv.error
                                else "Docling convert failed"
                            ),
                        )
                else:
                    docling_doc = conv.document

            # ------------------- Plain-text fallback path ------------------- #
            if docling_doc is None:
                with traced_step(trace, name="fallback_read_text") as step:
                    try:
                        raw = path.read_bytes()
                    except Exception:
                        raw = b""
                    try:
                        text = raw.decode("utf-8")
                    except Exception:
                        text = raw.decode("utf-8", errors="ignore")

                with traced_step(trace, name="build_document_graph_text_fallback"):
                    built = build_document_graph_from_text(
                        text,
                        settings=FILE_PARSER_SETTINGS,
                    )

                full_text = str(text or "")
                meta = extract_metadata_from_text_best_effort(
                    text=full_text,
                    settings=FILE_PARSER_SETTINGS,
                )
                summary = ""
                try:
                    root = built.graph.nodes.get(built.graph.root_id)
                    summary = (
                        str(getattr(root, "summary", "") or "").strip()
                        if root is not None
                        else ""
                    )
                except Exception:
                    summary = ""
                if not summary:
                    summary = (
                        (full_text or "")[:2000].strip()
                        or str(ctx.logical_path or "").strip()
                        or "Empty document"
                    )

                trace.counters["nodes"] = len(built.graph.nodes)
                trace.counters["tables"] = len(built.tables)
                trace.duration_ms = (time.perf_counter() - started) * 1000.0
                return FileParseResult(
                    logical_path=str(ctx.logical_path),
                    status="success",
                    file_format=ctx.file_format,
                    mime_type=ctx.mime_type,
                    tables=list(built.tables or []),
                    summary=summary,
                    full_text=full_text,
                    metadata=meta,
                    trace=trace,
                    graph=built.graph,
                )

            # ------------------- Docling document pipeline ------------------- #
            with traced_step(trace, name="docling_index_structure") as step:
                doc_index = index_docling_structure(docling_doc)
                try:
                    step.counters["headings"] = len(
                        list(getattr(doc_index, "heading_order", []) or []),
                    )
                except Exception as e:
                    step.warnings.append(f"headings_count_failed: {e}")

            # Merge consecutive Docling tables (best-effort) before graph extraction
            merged_tables = []
            keep_refs: set[str] | None = None
            with traced_step(trace, name="merge_consecutive_tables") as step:
                try:
                    from unity.file_manager.file_parsers.implementations.docling.utils.table_merge import (
                        merge_consecutive_table_items,
                    )

                    merged_tables = list(
                        merge_consecutive_table_items(docling_doc, doc_index) or [],
                    )
                    keep_refs = {
                        str(getattr(t, "self_ref"))
                        for t in merged_tables
                        if getattr(t, "self_ref", None) is not None
                    }
                    step.counters["tables_after_merge"] = len(merged_tables)
                except Exception as e:
                    # Non-fatal; degrade and continue with raw tables
                    step.status = StepStatus.DEGRADED
                    step.warnings.append(str(e))
                    merged_tables = []
                    keep_refs = None

            built = None
            with traced_step(trace, name="build_document_graph_hybrid") as step:
                try:
                    built = build_document_graph_hybrid(
                        docling_doc,
                        doc_index=doc_index,
                        merged_table_items=merged_tables or None,
                        settings=FILE_PARSER_SETTINGS,
                    )
                    step.counters["strategy"] = 1
                except Exception as e:
                    step.status = StepStatus.DEGRADED
                    step.warnings.append(str(e))
                    built = None

            if built is None:
                with traced_step(trace, name="build_document_graph_native") as step:
                    try:
                        built = build_document_graph_fallback(
                            docling_doc,
                            doc_index=doc_index,
                            keep_table_self_refs=keep_refs,
                            settings=FILE_PARSER_SETTINGS,
                        )
                        step.counters["strategy"] = 1
                    except Exception as e:
                        step.status = StepStatus.DEGRADED
                        step.warnings.append(str(e))
                        built = None

            with traced_step(trace, name="export_full_text") as step:
                # full_text is useful for debugging and document-level summaries.
                try:
                    full_text = str(docling_doc.export_to_text())
                except Exception as e:
                    # Non-fatal: keep outcome valid and record degradation.
                    step.status = StepStatus.DEGRADED
                    step.warnings.append(f"export_to_text failed: {e}")
                    full_text = ""

            # The Docling document holds a parallel copy of all content.
            # Release it now that graph + full_text have been extracted.
            del docling_doc, conv, converter, doc_index, merged_tables

            if built is None:
                with traced_step(
                    trace,
                    name="build_document_graph_text_fallback",
                ) as step:
                    try:
                        built = build_document_graph_from_text(
                            full_text,
                            settings=FILE_PARSER_SETTINGS,
                        )
                    except Exception as e:
                        step.status = StepStatus.FAILED
                        step.warnings.append(str(e))
                        raise

            # Enrichment (best-effort): hierarchical summaries + metadata
            if built is not None and built.graph is not None:
                with traced_step(trace, name="generate_hierarchical_summaries") as step:
                    try:
                        generate_hierarchical_summaries(
                            built.graph,
                            settings=FILE_PARSER_SETTINGS,
                        )
                    except Exception as e:
                        step.status = StepStatus.DEGRADED
                        step.warnings.append(str(e))

            meta = extract_metadata_from_text_best_effort(
                text=full_text,
                settings=FILE_PARSER_SETTINGS,
            )

            summary = ""
            try:
                if built is not None and built.graph is not None:
                    root = built.graph.nodes.get(built.graph.root_id)
                    summary = (
                        str(getattr(root, "summary", "") or "").strip()
                        if root is not None
                        else ""
                    )
            except Exception:
                summary = ""
            if not summary:
                summary = (
                    (full_text or "")[:2000].strip()
                    or str(ctx.logical_path or "").strip()
                    or "Empty document"
                )

            # Counters
            if built is not None and built.graph is not None:
                trace.counters["nodes"] = len(built.graph.nodes)
            if built is not None:
                trace.counters["tables"] = len(built.tables)

            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="success",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                tables=(built.tables if built is not None else []),
                summary=summary,
                full_text=full_text,
                metadata=meta,
                trace=trace,
                graph=(built.graph if built is not None else None),
            )

        except Exception as e:
            logger.exception("%s parse failed: %s", self.name, e)
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
