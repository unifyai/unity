from __future__ import annotations

"""
Postconditions / invariants enforcement for `FileParseResult`.

Backends are responsible for producing high-quality `FileParseResult` objects,
but the `FileParser` facade enforces a minimal set of **structural** invariants
so downstream pipelines never see malformed results.

This module is called inside ``_parse_single`` which runs in subprocess workers.
Therefore it must NEVER perform LLM calls.  All LLM-based enrichment (summaries,
metadata) belongs exclusively in ``enrich.py``.
"""


from unity.file_manager.file_parsers.settings import (
    FILE_PARSER_SETTINGS,
    FileParserSettings,
)
from unity.file_manager.file_parsers.types.contracts import FileParseResult


def enforce_parse_success_invariants(
    result: FileParseResult,
    *,
    settings: FileParserSettings = FILE_PARSER_SETTINGS,
) -> FileParseResult:
    """
    Enforce minimal structural invariants on a `FileParseResult` in-place.

    Current invariants (success results only)
    ----------------------------------------
    - ``full_text`` is always a string (never ``None``).
    - ``tables`` is always a list.

    Summary and metadata are intentionally left as-is (``None`` = unenriched).
    LLM enrichment is the sole responsibility of ``enrich.py``.
    """
    if getattr(result, "status", "error") != "success":
        return result

    if result.full_text is None:
        result.full_text = ""
    if result.tables is None:  # type: ignore[comparison-overlap]
        result.tables = []

    return result
