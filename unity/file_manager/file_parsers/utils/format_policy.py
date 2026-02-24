"""
Format-aware output policy for `unity.file_manager.file_parsers`.

Why this module exists
----------------------
The FileParser is *format-aware*. That does **not** just mean routing to a
different backend per file extension — it also means:

- Avoiding huge or low-value strings in parser outputs (e.g., dumping an entire
  spreadsheet into `full_text`).
- Still producing useful, retrieval-oriented outputs for **all** formats:
  `summary` and `metadata` should be populated on success for CSV/XLSX just as
  they are for PDF/DOCX/TXT.

To keep these decisions maintainable, we centralize them here so that future
changes (e.g., different budgets, different spreadsheet profile formatting)
are not scattered across multiple backends.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import DefaultDict, Iterable, List, Optional, Sequence

from unity.common.llm_client import new_llm_client
from unity.common.token_utils import (
    clip_text_to_token_limit_conservative,
    conservative_token_estimate,
    has_meaningful_text,
)
from unity.file_manager.file_parsers.prompts.metadata_prompts import (
    build_metadata_extraction_prompt,
)
from unity.file_manager.file_parsers.prompts.table_prompts import (
    build_spreadsheet_summary_prompt,
)
from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.types.contracts import FileParseMetadata
from unity.file_manager.file_parsers.types.metadata_extraction import (
    DocumentMetadataExtraction,
)
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.file_parsers.utils.summary_compression import (
    generate_summary_with_compression,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\\-]{2,}")


def _uniq(items: Iterable[str]) -> List[str]:
    """Return unique strings while preserving first-seen order."""
    out: List[str] = []
    seen: set[str] = set()
    for it in items:
        s = str(it or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _snake(s: str) -> str:
    """Normalize an arbitrary string into a conservative snake_case token."""
    t = (s or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _fallback_topics_from_text(
    text: str,
    *,
    min_count: int,
    max_count: int,
) -> List[str]:
    toks = []
    for m in _WORD_RE.finditer(text or ""):
        tok = _snake(m.group(0))
        if tok and len(tok) >= 3:
            toks.append(tok)
        if len(toks) >= (max_count * 3):
            break
    uniq = _uniq(toks)
    if len(uniq) >= min_count:
        return uniq[:max_count]
    # Pad deterministically when the source is too small.
    while len(uniq) < min_count:
        uniq.append(f"topic_{len(uniq)+1}")
    return uniq[:max_count]


def _fallback_metadata_from_text(text: str) -> FileParseMetadata:
    topics = _fallback_topics_from_text(text, min_count=3, max_count=8)
    tags = _fallback_topics_from_text(text, min_count=5, max_count=12)
    return FileParseMetadata(
        key_topics=", ".join(topics),
        named_entities="",
        content_tags=", ".join(tags),
        confidence_score=0.3,
    )


def extract_metadata_from_text_best_effort(
    *,
    text: str,
    settings: FileParserSettings,
) -> FileParseMetadata:
    """
    Best-effort metadata extraction that never returns None.

    The primary path uses an LLM constrained by a Pydantic schema. If that fails
    for any reason (model error, invalid JSON, etc.) we fall back to a small,
    deterministic extraction to ensure `FileParseResult.metadata` is populated.
    """
    if not has_meaningful_text(text):
        return _fallback_metadata_from_text(text or "")

    prompt = build_metadata_extraction_prompt(
        schema_json=DocumentMetadataExtraction.model_json_schema(),
    )
    client = new_llm_client(
        settings.SUMMARY_MODEL,
        async_client=False,
        reasoning_effort=None,
        service_tier=None,
        origin="FileParser.extract_metadata",
    )

    budgets = [
        int(settings.SUMMARY_MAX_TOKENS),
        max(int(settings.SUMMARY_MAX_TOKENS) // 2, 4000),
    ]

    for budget in budgets:
        try:
            try:
                prompt_tokens = conservative_token_estimate(
                    prompt,
                    settings.SUMMARY_ENCODING,
                )
                usable = max(int(budget) - int(prompt_tokens), 256)
            except Exception:
                usable = int(budget)

            clipped = clip_text_to_token_limit_conservative(
                text,
                usable,
                settings.SUMMARY_ENCODING,
            )
            resp = client.copy().generate(prompt + clipped).strip()
            validated = DocumentMetadataExtraction.model_validate_json(resp)

            key_topics = ", ".join([str(x) for x in list(validated.key_topics or [])])
            tags = ", ".join([str(x) for x in list(validated.content_tags or [])])

            ents: List[str] = []
            for _k, vs in (validated.named_entities or {}).items():
                try:
                    for v in list(vs or []):
                        s = str(v).strip()
                        if s:
                            ents.append(s)
                except Exception:
                    continue
            ents_str = ", ".join(sorted({e for e in ents}))

            return FileParseMetadata(
                key_topics=key_topics,
                named_entities=ents_str,
                content_tags=tags,
                confidence_score=float(validated.confidence_score),
            )
        except Exception:
            continue

    return _fallback_metadata_from_text(text)


def build_spreadsheet_profile_text(
    *,
    logical_path: str,
    tables: Sequence[ExtractedTable],
    sheet_names: Optional[Sequence[str]] = None,
    max_tables: int = 12,
    max_sample_rows: int = 5,
) -> str:
    """
    Build a bounded, retrieval-friendly textual profile for a spreadsheet.

    This is used for:
    - `FileParseResult.full_text` (bounded, no full dumps)
    - Spreadsheet-level `summary` and `metadata` generation
    """
    safe_path = str(logical_path or "").strip() or "spreadsheet"
    lines: List[str] = []
    lines.append(f"Spreadsheet: {safe_path}")

    by_sheet: DefaultDict[str, List[ExtractedTable]] = defaultdict(list)
    for t in list(tables or []):
        by_sheet[str(t.sheet_name or "Sheet 1")].append(t)

    # Prefer the provided sheet order when available.
    ordered_sheets: List[str] = []
    if sheet_names:
        ordered_sheets.extend([str(s) for s in sheet_names if str(s).strip()])
    # Add any remaining sheets deterministically.
    ordered_sheets.extend(
        [s for s in sorted(by_sheet.keys()) if s not in set(ordered_sheets)],
    )

    total_tables = sum(len(v) for v in by_sheet.values())
    lines.append(f"Sheets: {len(ordered_sheets)}; Tables: {total_tables}")

    included = 0
    for sheet in ordered_sheets:
        if included >= max_tables:
            break
        ts = by_sheet.get(sheet, [])
        # Stable: sort by label then table_id
        ts = sorted(ts, key=lambda t: (str(t.label), str(t.table_id)))
        lines.append("")
        lines.append(f"Sheet: {sheet} (tables={len(ts)})")
        for tbl in ts:
            if included >= max_tables:
                break
            included += 1
            label = str(tbl.label)
            lines.append(f"- Table: {label}")
            if tbl.num_rows is not None or tbl.num_cols is not None:
                lines.append(f"  Shape: rows={tbl.num_rows} cols={tbl.num_cols}")
            if tbl.columns:
                lines.append(f"  Columns: {', '.join([str(c) for c in tbl.columns])}")
            sample = list(tbl.sample_rows or [])
            if max_sample_rows > 0:
                sample = sample[:max_sample_rows]
            if sample:
                try:
                    sample_json = json.dumps(sample, ensure_ascii=False)
                except Exception:
                    sample_json = str(sample)
                lines.append(f"  SampleRowsJSON: {sample_json}")

    if included < total_tables:
        lines.append("")
        lines.append(
            f"[TRUNCATED] Included {included}/{total_tables} tables (max_tables={max_tables}).",
        )
    lines.append("")
    lines.append(
        "Note: This is a bounded profile (no full spreadsheet dump). Query the per-table contexts for raw rows.",
    )
    return "\n".join(lines).strip() + "\n"


def bound_spreadsheet_full_text(
    *,
    profile_text: str,
    settings: FileParserSettings,
) -> str:
    """Clip spreadsheet profile text to a conservative token budget for safe storage/logging."""
    return clip_text_to_token_limit_conservative(
        profile_text or "",
        settings.EMBEDDING_MAX_INPUT_TOKENS,
        settings.EMBEDDING_ENCODING,
    )


def fallback_spreadsheet_summary(
    *,
    logical_path: str,
    tables: Sequence[ExtractedTable],
    sheet_names: Optional[Sequence[str]] = None,
) -> str:
    sheets = [str(s) for s in (sheet_names or []) if str(s).strip()]
    if not sheets:
        sheets = sorted({str(t.sheet_name or "Sheet 1") for t in list(tables or [])})
    sheet_preview = ", ".join(sheets[:6]) + ("…" if len(sheets) > 6 else "")
    return (
        f"Spreadsheet '{str(logical_path)}' with {len(sheets)} sheet(s) and {len(list(tables or []))} table(s). "
        f"Sheets: {sheet_preview}"
    ).strip()


def summarize_spreadsheet_profile_best_effort(
    *,
    profile_text: str,
    settings: FileParserSettings,
    fallback: str,
) -> str:
    """
    Best-effort spreadsheet summary that is always embedding-safe and non-empty.
    """
    if not has_meaningful_text(profile_text):
        return (fallback or profile_text).strip()

    prompt = build_spreadsheet_summary_prompt(
        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    try:
        prompt_tokens = conservative_token_estimate(prompt, settings.SUMMARY_ENCODING)
        usable = max(int(settings.SUMMARY_MAX_TOKENS) - int(prompt_tokens), 256)
    except Exception:
        usable = int(settings.SUMMARY_MAX_TOKENS)

    clipped_profile = clip_text_to_token_limit_conservative(
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

    try:
        out = generate_summary_with_compression(
            client,
            prompt,
            clipped_profile,
            embedding_encoding=settings.EMBEDDING_ENCODING,
            max_embedding_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
        )
        out = (out or "").strip()
        return out if out else (fallback or "").strip()
    except Exception:
        return (fallback or "").strip()
