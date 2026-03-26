from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.conversion import (
    DocumentConversionManager,
    DocxToPdfConverter,
)
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
from unity.file_manager.file_parsers.utils.tracing import traced_step
from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
from unity.file_manager.file_parsers.types.contracts import FileParseRequest
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.contracts import (
    ConversionHop,
    FileParseResult,
    FileParseTrace,
    StepStatus,
)
from unity.file_manager.file_parsers.utils.format_policy import (
    extract_metadata_from_text_best_effort,
)

logger = logging.getLogger(__name__)


class MsWordBackend(BaseFileParserBackend):
    name = "ms_word_backend"
    supported_formats: Sequence[FileFormat] = (FileFormat.DOCX, FileFormat.DOC)

    def __init__(
        self,
        *,
        conversion_manager: Optional[DocumentConversionManager] = None,
        cleanup_converted_files: bool = True,
    ) -> None:
        # `.doc` is not supported by Docling directly; we convert to PDF first.
        self._conversion_manager = conversion_manager or DocumentConversionManager(
            converters=[DocxToPdfConverter()],
        )
        self._cleanup_converted_files = cleanup_converted_files

    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        return fmt in self.supported_formats

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        started = time.perf_counter()
        original_path = Path(ctx.source_local_path).expanduser().resolve()

        trace = FileParseTrace(
            logical_path=str(ctx.logical_path),
            backend=self.name,
            file_format=ctx.file_format,
            mime_type=ctx.mime_type,
            status=StepStatus.SUCCESS,
            source_local_path=str(original_path),
            parsed_local_path=str(original_path),
        )

        if not original_path.exists() or not original_path.is_file():
            trace.status = StepStatus.FAILED
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error=f"File not found: {original_path}",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
            )

        parse_path = original_path
        to_cleanup: list[Path] = []

        try:
            # Convert legacy `.doc` to `.pdf` when necessary.
            if original_path.suffix.lower() == ".doc":
                with traced_step(trace, name="convert_doc_to_pdf"):
                    res = self._conversion_manager.convert(original_path)
                    trace.conversion_chain.append(
                        ConversionHop(
                            operation="doc_to_pdf",
                            src=str(res.src),
                            dst=(str(res.dst) if res.dst is not None else None),
                            backend=str(res.backend),
                            ok=bool(res.ok),
                            message=str(res.message or ""),
                        ),
                    )
                    if not res.ok or not res.dst:
                        raise RuntimeError(f"DOC->PDF conversion failed: {res.message}")
                    parse_path = res.dst
                    trace.parsed_local_path = str(parse_path)
                    if res.backend != "reuse":
                        to_cleanup.append(parse_path)

            with traced_step(trace, name="docling_converter_init"):
                converter = new_docling_converter(settings=FILE_PARSER_SETTINGS)

            with traced_step(trace, name="docling_convert"):
                conv = docling_convert(converter=converter, source=str(parse_path))
                if not conv.ok or conv.document is None:
                    raise RuntimeError(
                        (
                            conv.error.message
                            if conv.error
                            else "Docling convert failed"
                        ),
                    )
                docling_doc = conv.document

            with traced_step(trace, name="docling_index_structure") as step:
                doc_index = index_docling_structure(docling_doc)
                try:
                    step.counters["headings"] = len(
                        list(getattr(doc_index, "heading_order", []) or []),
                    )
                except Exception as e:
                    step.warnings.append(f"headings_count_failed: {e}")

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
                try:
                    full_text = str(docling_doc.export_to_text())
                except Exception as e:
                    step.status = StepStatus.DEGRADED
                    step.warnings.append(f"export_to_text failed: {e}")
                    full_text = ""

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

                with traced_step(trace, name="extract_metadata") as step:
                    try:
                        meta = extract_metadata_from_text_best_effort(
                            text=full_text,
                            settings=FILE_PARSER_SETTINGS,
                        )
                    except Exception as e:
                        step.status = StepStatus.DEGRADED
                        step.warnings.append(str(e))
                        meta = extract_metadata_from_text_best_effort(
                            text=full_text,
                            settings=FILE_PARSER_SETTINGS,
                        )
            else:
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
                try:
                    from unity.common.token_utils import (
                        clip_text_to_token_limit_conservative,
                    )

                    summary = clip_text_to_token_limit_conservative(
                        full_text,
                        FILE_PARSER_SETTINGS.EMBEDDING_MAX_INPUT_TOKENS,
                        FILE_PARSER_SETTINGS.EMBEDDING_ENCODING,
                    )
                except Exception:
                    summary = (full_text or "")[:2000].strip()

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
            logger.exception("Word parse failed: %s", e)
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
        finally:
            if self._cleanup_converted_files and to_cleanup:
                for p in to_cleanup:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception as e:
                        # Cleanup is best-effort but should never be silent in trace
                        trace.warnings.append(
                            f"Failed to cleanup converted file: {p}: {e}",
                        )
