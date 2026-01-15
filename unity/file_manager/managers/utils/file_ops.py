from __future__ import annotations

from typing import List

from unity.file_manager.file_parsers import ContentType
from unity.file_manager.types.config import FilePipelineConfig as _FilePipelineConfig
from unity.file_manager.types.ingest import (
    BaseIngestedFile as _IngestedBase,
    IngestedPDF as _IngestedPDF,
    IngestedDocx as _IngestedDocx,
    IngestedDoc as _IngestedDoc,
    IngestedXlsx as _IngestedXlsx,
    IngestedCsv as _IngestedCsv,
    ContentRef as _ContentRef,
    TableRef as _TableRef,
    FileMetrics as _FileMetrics,
    IngestedFileUnion,
)
from unity.file_manager.file_parsers.types.formats import FileFormat as _FileFormat
from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.types.file import FileContentRow
from unity.file_manager.managers.file_manager import FileManager


def build_compact_ingest_model(
    file_manager: FileManager,
    *,
    file_path: str,
    parse_result: FileParseResult,
    config: _FilePipelineConfig,
) -> IngestedFileUnion:
    """Build a typed, reference-first ingest model without heavy fields.

    This function builds the appropriate Ingested* Pydantic model based on
    the file format, populating it with compact metadata and context references.

    Parameters
    ----------
    file_manager : FileManager
        The FileManager instance for context/helper methods.
    file_path : str
        The file path.
    parse_result : FileParseResult
        The FileParseResult from the file parser.
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    IngestedFileUnion
        A typed Pydantic model (IngestedPDF, IngestedXlsx, etc.) with compact
        reference-first data. Heavy artifacts (full_text, records) are NOT included.
    """

    def _ctype(r: FileContentRow) -> ContentType:
        return getattr(r, "content_type", None)

    # Identity via describe() API (returns FileStorageMap)
    # If file is not yet indexed, fall back to adapter-based identity
    try:
        storage = file_manager.describe(file_path=file_path)
        source_uri = storage.source_uri
    except (FileNotFoundError, ValueError):
        # File not yet indexed - get source_uri from adapter
        source_uri = file_manager._resolve_to_uri(file_path)
    display_path = file_path  # Use file_path as display_path

    # Destination naming depends on ingest mode
    dest_path = (
        file_path
        if config.ingest.mode == "per_file"
        else (config.ingest.unified_label or "Unified")
    )

    # Content reference
    content_ctx = file_manager._ctx_for_file(dest_path)
    from unity.file_manager.parse_adapter import adapt_parse_result_for_file_manager

    adapted = adapt_parse_result_for_file_manager(parse_result, config=config)
    record_count = len(list(adapted.content_rows or []))
    try:
        text_chars = len(parse_result.full_text)
    except Exception:
        text_chars = 0
    content_ref = _ContentRef(
        context=content_ctx,
        record_count=record_count,
        text_chars=text_chars,
    )

    # Tables preview
    tables_meta: List[_TableRef] = []
    try:
        tables = list(getattr(parse_result, "tables", []) or [])
        for idx, tbl in enumerate(tables, start=1):
            label = str(getattr(tbl, "label", None) or f"{idx:02d}")
            label_safe = file_manager.safe(label)
            columns = list(getattr(tbl, "columns", []) or [])[:16]
            row_count = len(getattr(tbl, "rows", []) or [])
            tables_meta.append(
                _TableRef(
                    name=label_safe,
                    context=file_manager._ctx_for_file_table(dest_path, label_safe),
                    row_count=row_count,
                    columns=columns,
                ),
            )
    except Exception:
        tables_meta = []

    # Metrics (from parse result; file_size not available from describe())
    metrics = _FileMetrics(
        file_size=None,  # File size can be obtained from adapter if needed
        processing_time=(
            (getattr(getattr(parse_result, "trace", None), "duration_ms", None) or 0.0)
            / 1000.0
            if getattr(parse_result, "trace", None) is not None
            else None
        ),
        confidence_score=(
            getattr(getattr(parse_result, "metadata", None), "confidence_score", None)
            if getattr(parse_result, "metadata", None) is not None
            else None
        ),
    )

    # Summary excerpt (trim)
    summary_excerpt = (parse_result.summary or "")[:512]

    # Determine canonical file format/mime (handle enum or string)
    ffmt = parse_result.file_format or getattr(
        getattr(parse_result, "file_format", None),
        "value",
        None,
    )
    mime = getattr(parse_result, "mime_type", None)

    if isinstance(ffmt, _FileFormat):
        fmt_key = ffmt.value.lower()
    elif isinstance(ffmt, str):
        fmt_key = ffmt.lower()
    else:
        fmt_key = str(ffmt or "").lower()

    Model = {
        _FileFormat.PDF.value: _IngestedPDF,
        _FileFormat.DOCX.value: _IngestedDocx,
        _FileFormat.DOC.value: _IngestedDoc,
        _FileFormat.XLSX.value: _IngestedXlsx,
        _FileFormat.CSV.value: _IngestedCsv,
    }.get(fmt_key, _IngestedBase)

    base_kwargs = dict(
        file_path=file_path,
        source_uri=source_uri,
        display_path=display_path,
        file_format=ffmt,
        mime_type=mime,
        status=parse_result.status,
        error=parse_result.error,
        created_at=None,  # Not available from describe(); set during ingest if needed
        modified_at=None,  # Not available from describe(); set during ingest if needed
        summary_excerpt=summary_excerpt,
        content_ref=content_ref,
        tables_ref=tables_meta,
        metrics=metrics,
    )

    if Model is _IngestedPDF:
        # Derive counts from lowered content rows and tables
        rows = list(adapted.content_rows or [])

        return Model(
            **base_kwargs,
            page_count=None,
            total_sections=sum(1 for r in rows if _ctype(r) == ContentType.SECTION)
            or None,
            image_count=sum(1 for r in rows if _ctype(r) == ContentType.IMAGE) or None,
            table_count=len(list(getattr(parse_result, "tables", []) or [])) or None,
            total_records=len(rows),
        )
    if Model in (_IngestedDocx, _IngestedDoc):
        rows = list(adapted.content_rows or [])
        return Model(
            **base_kwargs,
            total_sections=sum(1 for r in rows if _ctype(r) == ContentType.SECTION)
            or None,
            image_count=sum(1 for r in rows if _ctype(r) == ContentType.IMAGE) or None,
            table_count=len(list(getattr(parse_result, "tables", []) or [])) or None,
            total_records=len(rows),
        )
    if Model is _IngestedXlsx:
        rows = list(adapted.content_rows or [])
        sheet_names: List[str] = []
        for r in rows:
            try:
                ctype = _ctype(r)
                if ctype != ContentType.SHEET:
                    continue
                title = getattr(r, "title", None) or None
                if title and title not in sheet_names:
                    sheet_names.append(str(title))
            except Exception:
                continue
        tables = list(getattr(parse_result, "tables", []) or [])
        return Model(
            **base_kwargs,
            sheet_count=len(sheet_names) if sheet_names else (len(tables) or None),
            sheet_names=sheet_names,
            table_count=len(tables) or None,
        )
    if Model is _IngestedCsv:
        tables = list(getattr(parse_result, "tables", []) or [])
        return Model(
            **base_kwargs,
            table_count=len(tables) or 1,
        )

    return Model(**base_kwargs)
