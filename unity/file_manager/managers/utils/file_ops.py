from __future__ import annotations

from typing import Any, List

from unity.file_manager.types.config import FilePipelineConfig as _FilePipelineConfig
from unity.file_manager.types.file import ParsedFile
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
from unity.file_manager.parser.types.enums import FileFormat as _FileFormat


def build_compact_ingest_model(
    manager: Any,
    *,
    file_path: str,
    document: Any,
    parse_result: ParsedFile,
    config: _FilePipelineConfig,
) -> IngestedFileUnion:
    """Build a typed, reference-first ingest model without heavy fields.

    This function builds the appropriate Ingested* Pydantic model based on
    the file format, populating it with compact metadata and context references.

    Parameters
    ----------
    manager : FileManager
        The FileManager instance for context/helper methods.
    file_path : str
        The file path.
    document : Document
        The parsed document object.
    parse_result : ParsedFile
        The ParsedFile Pydantic model from Document.to_parse_result().
    config : FilePipelineConfig
        Pipeline configuration.

    Returns
    -------
    IngestedFileUnion
        A typed Pydantic model (IngestedPDF, IngestedXlsx, etc.) with compact
        reference-first data. Heavy artifacts (full_text, records) are NOT included.
    """

    # Identity via file_info (returns FileInfo Pydantic model)
    info = manager._file_info(identifier=file_path)
    source_uri = info.source_uri
    display_path = file_path  # Use file_path as display_path

    # Destination naming depends on ingest mode
    dest_path = (
        file_path
        if config.ingest.mode == "per_file"
        else (config.ingest.unified_label or "Unified")
    )

    # Content reference
    content_ctx = manager._ctx_for_file(dest_path)
    record_count = parse_result.total_records
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
        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        for idx, tbl in enumerate(tables, start=1):
            sheet_name = getattr(tbl, "sheet_name", None)
            section_path = getattr(tbl, "section_path", None)
            label = sheet_name or section_path or f"{idx:02d}"
            label_safe = manager.safe(str(label))
            columns = list(getattr(tbl, "columns", []) or [])[:16]
            row_count = len(getattr(tbl, "rows", []) or [])
            tables_meta.append(
                _TableRef(
                    name=label_safe,
                    context=manager._ctx_for_file_table(dest_path, label_safe),
                    row_count=row_count,
                    columns=columns,
                ),
            )
    except Exception:
        tables_meta = []

    # Metrics
    metrics = _FileMetrics(
        file_size=parse_result.file_size,
        processing_time=parse_result.processing_time,
        confidence_score=parse_result.confidence_score,
    )

    # Summary excerpt (trim)
    summary_excerpt = (parse_result.summary or "")[:512]

    # Determine canonical file format/mime (handle enum or string)
    ffmt = parse_result.file_format or getattr(
        getattr(document, "metadata", None),
        "file_format",
        None,
    )
    mime = getattr(
        getattr(document, "metadata", None),
        "mime_type",
        None,
    )

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
        created_at=parse_result.created_at,
        modified_at=parse_result.modified_at,
        summary_excerpt=summary_excerpt,
        content_ref=content_ref,
        tables_ref=tables_meta,
        metrics=metrics,
    )

    if Model is _IngestedPDF:
        return Model(
            **base_kwargs,
            page_count=getattr(
                getattr(document, "metadata", None),
                "total_pages",
                None,
            ),
            total_sections=len(getattr(document, "sections", []) or []),
            image_count=len(
                getattr(getattr(document, "metadata", None), "images", []) or [],
            ),
            table_count=len(
                getattr(getattr(document, "metadata", None), "tables", []) or [],
            ),
            total_records=record_count,
        )
    if Model in (_IngestedDocx, _IngestedDoc):
        return Model(
            **base_kwargs,
            total_sections=len(getattr(document, "sections", []) or []),
            image_count=len(
                getattr(getattr(document, "metadata", None), "images", []) or [],
            ),
            table_count=len(
                getattr(getattr(document, "metadata", None), "tables", []) or [],
            ),
            total_records=record_count,
        )
    if Model is _IngestedXlsx:
        tables = list(getattr(getattr(document, "metadata", None), "tables", []) or [])
        sheet_names = []
        try:
            for t in tables:
                nm = getattr(t, "sheet_name", None)
                if nm and nm not in sheet_names:
                    sheet_names.append(nm)
        except Exception:
            sheet_names = sheet_names
        return Model(
            **base_kwargs,
            sheet_count=len(sheet_names) if sheet_names else (len(tables) or None),
            sheet_names=sheet_names,
            table_count=len(tables) or None,
        )
    if Model is _IngestedCsv:
        tables = list(getattr(getattr(document, "metadata", None), "tables", []) or [])
        return Model(
            **base_kwargs,
            table_count=len(tables) or 1,
        )

    return Model(**base_kwargs)
