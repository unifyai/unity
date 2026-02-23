"""
Table catalog lowering helpers.

This module builds a retrieval-friendly "table profile" text (columns, sample rows,
business-context descriptions/rules) and optionally summarizes it with an LLM.

It is used by lowering logic to emit `/Content/` catalog rows for tabular formats
(CSV/XLSX) so that RAG agents can:
- search table/sheet summaries in `/Content/`
- then query the concrete data in `/Tables/<label>`
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from unity.common.llm_client import new_llm_client
from unity.common.token_utils import (
    clip_text_to_token_limit_conservative,
    conservative_token_estimate,
)
from unity.file_manager.file_parsers.prompts.table_prompts import (
    build_table_catalog_prompt,
)
from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.types.config import BusinessContextsConfig


def _find_business_context(
    business_contexts: Optional[BusinessContextsConfig],
    *,
    file_path: str,
    table_label: str,
) -> Tuple[List[str], List[str], Optional[str], Dict[str, str], List[str]]:
    """
    Returns:
      (global_rules, file_rules, table_description, column_descriptions, table_rules)
    """
    if business_contexts is None:
        return ([], [], None, {}, [])

    global_rules = list(getattr(business_contexts, "global_rules", []) or [])
    file_rules: List[str] = []
    table_desc: Optional[str] = None
    col_desc: Dict[str, str] = {}
    table_rules: List[str] = []

    for fc in getattr(business_contexts, "file_contexts", []) or []:
        if getattr(fc, "file_path", None) != file_path:
            continue
        file_rules = list(getattr(fc, "file_rules", []) or [])
        for tc in getattr(fc, "table_contexts", []) or []:
            if getattr(tc, "table", None) != table_label:
                continue
            table_desc = getattr(tc, "table_description", None)
            col_desc = dict(getattr(tc, "column_descriptions", {}) or {})
            table_rules = list(getattr(tc, "table_rules", []) or [])
            return (global_rules, file_rules, table_desc, col_desc, table_rules)

    return (global_rules, file_rules, table_desc, col_desc, table_rules)


def build_table_profile_text(
    table: ExtractedTable,
    *,
    file_path: str,
    business_contexts: Optional[BusinessContextsConfig],
    max_sample_rows: int = 25,
) -> str:
    global_rules, file_rules, table_desc, col_desc, table_rules = (
        _find_business_context(
            business_contexts,
            file_path=file_path,
            table_label=table.label,
        )
    )

    sample_rows = list(table.sample_rows or [])
    if max_sample_rows > 0:
        sample_rows = sample_rows[:max_sample_rows]

    lines: List[str] = []
    lines.append(f"Table Label: {table.label}")
    if table.sheet_name:
        lines.append(f"Sheet: {table.sheet_name}")
    if table_desc:
        lines.append(f"Table Description: {table_desc}")

    if global_rules:
        lines.append("Global Rules:")
        lines.extend([f"- {r}" for r in global_rules])
    if file_rules:
        lines.append("File Rules:")
        lines.extend([f"- {r}" for r in file_rules])
    if table_rules:
        lines.append("Table Rules:")
        lines.extend([f"- {r}" for r in table_rules])

    if table.columns:
        lines.append("Columns:")
        for c in table.columns:
            desc = (col_desc.get(c) or "").strip()
            if desc:
                lines.append(f"- {c}: {desc}")
            else:
                lines.append(f"- {c}")

    if sample_rows:
        lines.append("Sample Rows (JSON):")
        try:
            lines.append(json.dumps(sample_rows, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(sample_rows))

    return "\n".join(lines).strip() + "\n"


def summarize_table_profile(
    *,
    profile_text: str,
    settings: FileParserSettings,
) -> str:
    """
    Summarize a table profile into an embedding-safe summary.

    This is sync by design; the FileManager pipeline already runs file-level concurrency.
    """
    from unity.file_manager.file_parsers.utils.summary_compression import (
        generate_summary_with_compression,
    )

    prompt = build_table_catalog_prompt(
        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )

    # Clip input to stay within the summarizer's context budget
    prompt_tokens = conservative_token_estimate(prompt, settings.SUMMARY_ENCODING)
    usable = max(settings.SUMMARY_MAX_TOKENS - prompt_tokens, 256)
    clipped = clip_text_to_token_limit_conservative(
        profile_text,
        usable,
        settings.SUMMARY_ENCODING,
    )

    client = new_llm_client(
        settings.SUMMARY_MODEL,
        async_client=False,
        reasoning_effort=None,
        service_tier=None,
        origin="FileParser.generate_summary",
    )

    return generate_summary_with_compression(
        client,
        prompt,
        clipped,
        embedding_encoding=settings.EMBEDDING_ENCODING,
        max_embedding_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
