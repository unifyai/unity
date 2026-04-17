"""LLM enrichment for parse results.

Backends return raw parse artifacts (graph, tables, full_text) with
no LLM calls.  This module is the **single place** where all LLM
enrichment runs — called by ``FileParser._parse_and_enrich`` after
every backend completes:

- **Documents** (PDF, DOCX, TXT, HTML, etc.): full hierarchical
  summary pipeline — paragraph → section → document.  The root
  summary becomes ``result.summary``.
- **Spreadsheets** (CSV, XLSX): LLM profile summariser over the
  bounded table profile text.
- **Metadata**: LLM extraction for all formats.

After enrichment, ``result.summary`` is guaranteed to be a real string
(no longer ``SUMMARY_UNSET``; may be empty for truly empty documents)
and ``result.metadata`` is guaranteed to be populated.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, FrozenSet

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import FileParseResult

logger = logging.getLogger(__name__)

_SPREADSHEET_FORMATS: FrozenSet[str] = frozenset({"csv", "xlsx"})


def enrich_parse_result(result: "FileParseResult") -> None:
    """Run LLM enrichment on a parse result (in-place).

    File type is determined from ``result.file_format``:

    - **Spreadsheets** (CSV, XLSX) → LLM spreadsheet profile summariser.
    - **Documents** with a ``graph`` → full hierarchical summary pipeline
      (paragraph → section → document).
    - **All formats** → LLM metadata extraction.

    Timing is recorded as a ``StepTrace`` appended to ``result.trace.steps``.
    """
    if getattr(result, "status", "error") != "success":
        return

    from unity.file_manager.file_parsers.settings import FILE_PARSER_SETTINGS
    from unity.file_manager.file_parsers.types.contracts import StepStatus, StepTrace

    settings = FILE_PARSER_SETTINGS
    text = result.full_text or ""
    t0 = time.perf_counter()

    _enrich_summary(result, text=text, settings=settings)
    _enrich_metadata(result, text=text, settings=settings)
    _ensure_summary_fallback(result, settings=settings)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    step = StepTrace(
        name="llm_enrichment",
        status=StepStatus.SUCCESS,
        duration_ms=elapsed_ms,
    )
    if result.trace is not None:
        result.trace.steps.append(step)


def _is_spreadsheet_format(result: "FileParseResult") -> bool:
    """True when the result came from a tabular backend (CSV / XLSX)."""
    fmt = getattr(result, "file_format", None)
    if fmt is None:
        return False
    fmt_val = fmt.value if hasattr(fmt, "value") else str(fmt)
    return fmt_val.lower() in _SPREADSHEET_FORMATS


def _enrich_summary(result: "FileParseResult", *, text: str, settings) -> None:
    """Route to the correct summary enrichment based on file format."""
    if _is_spreadsheet_format(result):
        _enrich_spreadsheet_summary(result, text=text, settings=settings)
    elif result.graph is not None:
        _enrich_document_summary(result, settings=settings)


def _enrich_document_summary(result: "FileParseResult", *, settings) -> None:
    """Run the full hierarchical summary pipeline on the document graph.

    Generates paragraph-level summaries via LLM, composes section
    summaries from those, then composes the document summary from
    section summaries.  The root node's summary becomes
    ``result.summary``.
    """
    from unity.file_manager.file_parsers.implementations.docling.steps.document_enrichment import (
        generate_hierarchical_summaries,
    )

    graph = result.graph
    if graph is None:
        return

    try:
        generate_hierarchical_summaries(graph, settings=settings)
    except Exception:
        logger.debug("Hierarchical summary enrichment failed", exc_info=True)
        return

    root = graph.nodes.get(graph.root_id) if graph.root_id else None
    if root is not None:
        summary = str(getattr(root, "summary", "") or "").strip()
        if summary:
            result.summary = summary


def _enrich_spreadsheet_summary(
    result: "FileParseResult",
    *,
    text: str,
    settings,
) -> None:
    """Run the LLM spreadsheet profile summariser."""
    from unity.file_manager.file_parsers.utils.format_policy import (
        summarize_spreadsheet_profile_best_effort,
    )

    from unity.file_manager.file_parsers.types.contracts import SUMMARY_UNSET

    existing = result.summary or ""
    fallback = "" if existing == SUMMARY_UNSET else existing

    try:
        new_summary = summarize_spreadsheet_profile_best_effort(
            profile_text=text,
            settings=settings,
            fallback=fallback,
        )
        if new_summary:
            result.summary = new_summary
    except Exception:
        logger.debug("Spreadsheet summary enrichment failed", exc_info=True)


def _enrich_metadata(result: "FileParseResult", *, text: str, settings) -> None:
    """Extract metadata via LLM."""
    from unity.file_manager.file_parsers.utils.format_policy import (
        extract_metadata_from_text_best_effort,
    )

    try:
        result.metadata = extract_metadata_from_text_best_effort(
            text=text,
            settings=settings,
        )
    except Exception:
        logger.debug("Metadata enrichment failed", exc_info=True)


def _ensure_summary_fallback(result: "FileParseResult", *, settings) -> None:
    """Guarantee ``result.summary`` is a real string after enrichment.

    If LLM enrichment didn't produce a summary, fall back to:
    - Spreadsheets: deterministic ``fallback_spreadsheet_summary``
    - Documents: clipped ``full_text``
    - Last resort: ``logical_path`` or ``"Empty document"``
    """
    from unity.file_manager.file_parsers.types.contracts import SUMMARY_UNSET

    s = (result.summary or "").strip()
    if s and s != SUMMARY_UNSET:
        return

    from unity.common.token_utils import clip_text_to_token_limit_conservative
    from unity.file_manager.file_parsers.types.formats import FileFormat
    from unity.file_manager.file_parsers.utils.format_policy import (
        fallback_spreadsheet_summary,
    )

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

    if not (result.summary or "").strip():
        result.summary = str(result.logical_path or "").strip() or "Empty document"
