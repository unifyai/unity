from __future__ import annotations

"""
Postconditions / invariants enforcement for `FileParseResult`.

Backends are responsible for producing high-quality `FileParseResult` objects,
but the `FileParser` facade enforces a minimal set of invariants so downstream
pipelines (FileManager ingestion) never see a "success" result missing core
retrieval fields.

This module exists purely to keep `file_parser.py` readable and to centralize
the invariant logic in one place.
"""


from unity.common.token_utils import clip_text_to_token_limit_conservative
from unity.file_manager.file_parsers.settings import (
    FILE_PARSER_SETTINGS,
    FileParserSettings,
)
from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.utils.format_policy import (
    extract_metadata_from_text_best_effort,
    fallback_spreadsheet_summary,
)


def enforce_parse_success_invariants(
    result: FileParseResult,
    *,
    settings: FileParserSettings = FILE_PARSER_SETTINGS,
) -> FileParseResult:
    """
    Enforce minimal invariants on a `FileParseResult` in-place and return it.

    Current invariants (success results only)
    ----------------------------------------
    - `metadata` must be non-null (best-effort LLM extraction with deterministic fallback)
    - `summary` must be non-empty
      - For CSV/XLSX: use a deterministic spreadsheet fallback summary
      - For other formats: clip `full_text` to an embedding-safe token budget
      - If still empty: fall back to `logical_path` / "Empty document"
    """
    if getattr(result, "status", "error") != "success":
        return result

    _ensure_metadata(result, settings=settings)
    _ensure_summary(result, settings=settings)
    return result


def _ensure_metadata(result: FileParseResult, *, settings: FileParserSettings) -> None:
    if result.metadata is not None:
        return
    # Prefer using the backend-provided text artifacts; fall back deterministically when empty.
    try:
        result.metadata = extract_metadata_from_text_best_effort(
            text=(result.full_text or result.summary or ""),
            settings=settings,
        )
    except Exception:
        result.metadata = extract_metadata_from_text_best_effort(
            text="",
            settings=settings,
        )


def _ensure_summary(result: FileParseResult, *, settings: FileParserSettings) -> None:
    if (result.summary or "").strip():
        return

    if result.file_format in (FileFormat.CSV, FileFormat.XLSX):
        result.summary = fallback_spreadsheet_summary(
            logical_path=result.logical_path,
            tables=list(result.tables or []),
            sheet_names=None,
        )
    else:
        result.summary = clip_text_to_token_limit_conservative(
            result.full_text or "",
            settings.EMBEDDING_MAX_INPUT_TOKENS,
            settings.EMBEDDING_ENCODING,
        ).strip()

    # Ensure *some* summary even for empty documents.
    if not (result.summary or "").strip():
        result.summary = str(result.logical_path or "").strip() or "Empty document"
