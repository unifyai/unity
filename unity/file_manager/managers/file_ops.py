from __future__ import annotations

from typing import Any, Dict, List

from unity.file_manager.types.config import FilePipelineConfig as _FilePipelineConfig
from unity.file_manager.types.parsed import (
    BaseParsedFile as _ParsedBase,
    ParsedPDF as _ParsedPDF,
    ParsedDocx as _ParsedDocx,
    ParsedDoc as _ParsedDoc,
    ParsedXlsx as _ParsedXlsx,
    ParsedCsv as _ParsedCsv,
    ContentRef as _ContentRef,
    TableRef as _TableRef,
    FileMetrics as _FileMetrics,
)
from unity.file_manager.parser.types.enums import FileFormat as _FileFormat


def build_compact_parse_model(
    manager: Any,
    *,
    file_path: str,
    document: Any,
    result: Dict[str, Any],
    config: _FilePipelineConfig,
) -> _ParsedBase:
    """Build a typed, reference-first parse model without heavy fields.

    Notes
    -----
    - Uses manager helpers (_build_identity, _safe, _ctx_for_file/_ctx_for_file_table)
    - Selects a format-specific Pydantic model (PDF/DOCX/DOC/XLSX/CSV) where possible
    - Populates common identity/status and light metrics only
    """

    # Identity
    ident = manager._build_file_identity(file_path)
    # FileIdentity is a Pydantic model; access via attributes
    try:
        source_uri = getattr(ident, "source_uri", None)
        display_path = getattr(ident, "display_path", None)
    except Exception:
        source_uri = None
        display_path = None

    # Destination naming depends on ingest mode
    dest_path = (
        file_path
        if config.ingest.mode == "per_file"
        else (config.ingest.unified_label or "Unified")
    )

    # Content reference
    content_ctx = manager._ctx_for_file(dest_path)
    record_count = int(result.get("total_records") or 0)
    try:
        text_chars = len(result.get("full_text") or "")
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
            label_safe = manager._safe(str(label))
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
        file_size=result.get("file_size"),
        processing_time=result.get("processing_time"),
        confidence_score=result.get("confidence_score"),
    )

    # Summary excerpt (trim)
    summary_excerpt = (result.get("summary") or "")[:512]

    # Determine canonical file format/mime (handle enum or string)
    ffmt = result.get("file_format") or getattr(
        getattr(document, "metadata", None),
        "file_format",
        None,
    )
    mime = result.get("mime_type") or getattr(
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
        _FileFormat.PDF.value: _ParsedPDF,
        _FileFormat.DOCX.value: _ParsedDocx,
        _FileFormat.DOC.value: _ParsedDoc,
        _FileFormat.XLSX.value: _ParsedXlsx,
        _FileFormat.CSV.value: _ParsedCsv,
    }.get(fmt_key, _ParsedBase)

    base_kwargs = dict(
        file_path=file_path,
        source_uri=source_uri,
        display_path=display_path,
        file_format=ffmt,
        mime_type=mime,
        status=result.get("status", "success"),
        error=result.get("error"),
        created_at=result.get("created_at"),
        modified_at=result.get("modified_at"),
        summary_excerpt=summary_excerpt,
        content_ref=content_ref,
        tables_ref=tables_meta,
        metrics=metrics,
    )

    if Model is _ParsedPDF:
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
    if Model in (_ParsedDocx, _ParsedDoc):
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
    if Model is _ParsedXlsx:
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
    if Model is _ParsedCsv:
        tables = list(getattr(getattr(document, "metadata", None), "tables", []) or [])
        return Model(
            **base_kwargs,
            table_count=len(tables) or 1,
        )

    return Model(**base_kwargs)
