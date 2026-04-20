"""
Content lowering: ContentGraph → `/Content/` rows.

This module is the adapter between the parser-internal `ContentGraph` and the
FileManager’s storage contract:
- hierarchical, retrieval-friendly rows go into `/Content/`
- raw tabular rows go into `/Tables/<label>` (handled elsewhere)

It is intentionally format-aware:
- For PDFs/DOCX: emit document/section/paragraph/sentence rows (and table/image catalog rows when present).
- For CSV/XLSX: emit document/sheet/table catalog rows (do NOT emit full dumps or deep text hierarchy).

Content_text policy
-------------------
`/Content/` is the *navigation* surface. It should contain searchable summaries
and small pieces of text, but must avoid duplicating large payloads:

- `sheet` and `table` rows must have `content_text=None`
- raw tabular rows belong in `/Tables/<label>` contexts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from unity.common.token_utils import clip_text_to_token_limit_conservative
from unity.file_manager.file_parsers.settings import (
    FILE_PARSER_SETTINGS,
    FileParserSettings,
)
from unity.file_manager.file_parsers.types.enums import ContentType, NodeKind
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.graph import ContentGraph, ContentNode
from unity.file_manager.file_parsers.types.table import ExtractedTable
from unity.file_manager.types.config import BusinessContextsConfig
from unity.file_manager.types.file import FileContentRow

from .catalog import build_table_profile_text, summarize_table_profile


@dataclass(frozen=True)
class ContentLoweringOutput:
    """FileManager-owned `/Content/` rows plus an optional file-level summary."""

    rows: List[FileContentRow]
    document_summary: str = ""


def _stable_children(graph: ContentGraph, node: ContentNode) -> List[ContentNode]:
    """Return children in stable order (by `order` then node_id)."""
    kids = [graph.nodes[cid] for cid in (node.children_ids or []) if cid in graph.nodes]
    kids.sort(
        key=lambda n: ((n.order if n.order is not None else 1_000_000_000), n.node_id),
    )
    return kids


def _build_content_id(
    *,
    document_id: int = 0,
    section_id: Optional[int] = None,
    paragraph_id: Optional[int] = None,
    sentence_id: Optional[int] = None,
    image_id: Optional[int] = None,
    table_id: Optional[int] = None,
    sheet_id: Optional[int] = None,
) -> Dict[str, int]:
    d: Dict[str, int] = {"document": int(document_id)}
    if section_id is not None:
        d["section"] = int(section_id)
    if sheet_id is not None:
        d["sheet"] = int(sheet_id)
    if paragraph_id is not None:
        d["paragraph"] = int(paragraph_id)
    if sentence_id is not None:
        d["sentence"] = int(sentence_id)
    if image_id is not None:
        d["image"] = int(image_id)
    if table_id is not None:
        d["table"] = int(table_id)
    return d


def _row(
    *,
    content_type: ContentType,
    title: Optional[str],
    summary: Optional[str],
    content_text: Optional[str],
    content_id: Dict[str, int],
) -> FileContentRow:
    """
    Construct a `FileContentRow` payload.

    Important: this helper is intentionally dumb; any *format-aware* policy about
    which row types are allowed to carry `content_text` should be applied before
    calling this function (see `_should_emit_content_text`).
    """
    return FileContentRow(
        content_type=content_type,
        title=title,
        summary=summary,
        content_text=content_text,
        content_id=content_id,
    )


def _should_emit_content_text(
    *,
    content_type: ContentType,
    file_format: Optional[FileFormat],
) -> bool:
    """
    Return True iff `/Content/` rows of this type are allowed to carry `content_text`.

    This is the centralized policy point for preventing large/incompatible payloads
    from being written to the `/Content/` context.

    Current policy
    --------------
    - `sheet` and `table` rows MUST NOT populate `content_text` (tables/sheets can be huge;
      raw rows belong in `/Tables/<label>`).
    - Textual rows (`paragraph`, `sentence`) MAY populate `content_text`.
    """
    # Note: this rule is intentionally *not* dependent on file_format. Even in a PDF/DOCX,
    # `table` rows should be catalog-only; the raw tabular data lives in /Tables/ contexts.
    if content_type in (ContentType.SHEET, ContentType.TABLE):
        return False
    return True


_CHUNK_SIZE = 20


def _summarize_table_profiles_parallel(
    profiles: List[str],
    *,
    settings: "FileParserSettings",
) -> List[str]:
    """Summarize table profiles, using threads when there are multiple tables.

    Profiles are processed in chunks of ``_CHUNK_SIZE`` to bound peak memory:
    each chunk's profile strings are released after its LLM summaries are
    collected, preventing all profiles from being held simultaneously.
    """
    if not profiles:
        return []
    if len(profiles) == 1:
        return [summarize_table_profile(profile_text=profiles[0], settings=settings)]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Optional[str]] = [None] * len(profiles)

    for chunk_start in range(0, len(profiles), _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, len(profiles))
        chunk_profiles = profiles[chunk_start:chunk_end]

        with ThreadPoolExecutor(max_workers=min(len(chunk_profiles), 8)) as pool:
            future_to_idx = {
                pool.submit(
                    summarize_table_profile,
                    profile_text=p,
                    settings=settings,
                ): (chunk_start + i)
                for i, p in enumerate(chunk_profiles)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        # Release this chunk's profile strings — they are no longer needed
        # once the LLM summaries are collected.
        for i in range(chunk_start, chunk_end):
            profiles[i] = ""

    return [r or "" for r in results]


def lower_graph_to_content_rows(
    *,
    graph: ContentGraph,
    file_path: str,
    file_format: Optional[FileFormat],
    tables: List[ExtractedTable],
    business_contexts: Optional[BusinessContextsConfig],
    settings: FileParserSettings = FILE_PARSER_SETTINGS,
) -> ContentLoweringOutput:
    """
    Lower a ContentGraph (plus extracted tables) into `/Content/` rows.

    Design
    ------
    - `/Content/` is the semantic navigation surface.
    - For spreadsheets (XLSX/CSV): emit document + sheet + table catalog rows.
    - For PDFs/DOCX: emit document + section + paragraph + sentence (+ image/table catalog rows when present).
    """
    fmt = file_format
    rows: List[FileContentRow] = []

    # Defensive: root node should exist.
    root = graph.nodes.get(graph.root_id)
    if root is None:
        return ContentLoweringOutput(rows=[], document_summary="")

    # ------------------------------------------------------------------
    # Spreadsheet lowering: document + sheet + table catalog rows
    # ------------------------------------------------------------------
    if fmt in (FileFormat.XLSX, FileFormat.CSV):
        # Document row
        doc_title = root.title or str(file_path)
        rows.append(
            _row(
                content_type=ContentType.DOCUMENT,
                title=doc_title,
                summary="",
                content_text=None,
                content_id=_build_content_id(document_id=0),
            ),
        )

        # Sheet rows (from graph)
        sheet_nodes = [n for n in graph.nodes.values() if n.kind == NodeKind.SHEET]
        sheet_nodes.sort(
            key=lambda n: (
                n.order if n.order is not None else 1_000_000_000,
                n.node_id,
            ),
        )
        sheet_name_to_id: Dict[str, int] = {}
        for sheet_idx, sn in enumerate(sheet_nodes):
            name = (
                sn.title or sn.meta.get("sheet_name") or f"Sheet {sheet_idx+1}"
            ).strip()
            sheet_name_to_id[name] = sheet_idx
            # Lightweight sheet summary from table labels under this sheet (no LLM)
            table_labels: List[str] = []
            for child in _stable_children(graph, sn):
                if child.kind == NodeKind.TABLE:
                    table_labels.append(
                        str(
                            child.title
                            or child.meta.get("table_label")
                            or child.node_id,
                        ),
                    )
            sheet_summary = (
                f"Sheet '{name}' contains {len(table_labels)} table(s): "
                + ", ".join([str(x) for x in table_labels[:10]])
                + ("…" if len(table_labels) > 10 else "")
                if table_labels
                else f"Sheet '{name}'"
            )
            rows.append(
                _row(
                    content_type=ContentType.SHEET,
                    title=name,
                    summary=sheet_summary,
                    content_text=None,
                    content_id=_build_content_id(document_id=0, sheet_id=sheet_idx),
                ),
            )

        # Table catalog rows from extracted tables (richer than graph-only).
        #
        # Phase 1: prepare metadata and profile texts sequentially.
        # Tables with no data rows get a cheap templated summary instead of
        # an LLM call — this avoids wasting LLM budget (and memory) on the
        # many empty/trivial "tables" that Docling's flood-fill produces
        # (e.g., section headers, navigation labels, single-cell titles).
        table_prep: List[dict] = []
        llm_indices: List[int] = []

        for tbl in list(tables or []):
            sheet_name = tbl.sheet_name or "Sheet 1"
            sheet_id = sheet_name_to_id.get(sheet_name)
            # If sheet is missing from graph, synthesize a stable id from first appearance order.
            if sheet_id is None:
                sheet_id = len(sheet_name_to_id)
                sheet_name_to_id[sheet_name] = sheet_id
                rows.append(
                    _row(
                        content_type=ContentType.SHEET,
                        title=sheet_name,
                        summary=f"Sheet '{sheet_name}'",
                        content_text=None,
                        content_id=_build_content_id(document_id=0, sheet_id=sheet_id),
                    ),
                )

            # Prefer stable table_id per sheet using the ordering encoded
            # in table_id suffix when present.
            table_ix = 0
            try:
                if isinstance(tbl.table_id, str) and ":" in tbl.table_id:
                    table_ix = int(tbl.table_id.split(":", 1)[-1])
            except Exception:
                table_ix = 0

            num_rows = tbl.num_rows or 0
            num_cols = tbl.num_cols or 0

            if num_rows == 0 or num_cols == 0:
                # Empty table — use a cheap templated summary, no LLM call.
                label = tbl.label or ""
                cols_str = ", ".join((tbl.columns or [])[:5])
                if tbl.columns and len(tbl.columns) > 5:
                    cols_str += f" (+{len(tbl.columns) - 5} more)"
                summary = (
                    f"Empty table '{label}'"
                    + (f" on sheet '{sheet_name}'" if sheet_name else "")
                    + (f" with columns: {cols_str}" if cols_str else "")
                )
                table_prep.append(
                    {
                        "tbl": tbl,
                        "profile": "",
                        "summary": summary,
                        "sheet_id": sheet_id,
                        "table_ix": table_ix,
                        "needs_llm": False,
                    },
                )
            else:
                profile = build_table_profile_text(
                    tbl,
                    file_path=file_path,
                    business_contexts=business_contexts,
                    max_sample_rows=25,
                )
                llm_indices.append(len(table_prep))
                table_prep.append(
                    {
                        "tbl": tbl,
                        "profile": profile,
                        "summary": "",
                        "sheet_id": sheet_id,
                        "table_ix": table_ix,
                        "needs_llm": True,
                    },
                )

        # Phase 2: summarize non-empty profiles (LLM calls) — parallel
        # when >1 table.  Only tables that need LLM summarization are sent.
        llm_profiles = [table_prep[i]["profile"] for i in llm_indices]
        llm_summaries = _summarize_table_profiles_parallel(
            llm_profiles,
            settings=settings,
        )
        for idx, summary in zip(llm_indices, llm_summaries):
            table_prep[idx]["summary"] = summary
            table_prep[idx]["profile"] = ""  # release profile string

        # Phase 3: assemble rows in original order.
        for tp in table_prep:
            summary = clip_text_to_token_limit_conservative(
                tp["summary"],
                settings.EMBEDDING_MAX_INPUT_TOKENS,
                settings.EMBEDDING_ENCODING,
            )
            rows.append(
                _row(
                    content_type=ContentType.TABLE,
                    title=str(tp["tbl"].label),
                    summary=summary,
                    content_text=(
                        tp["profile"]
                        if _should_emit_content_text(
                            content_type=ContentType.TABLE,
                            file_format=fmt,
                        )
                        else None
                    ),
                    content_id=_build_content_id(
                        document_id=0,
                        sheet_id=tp["sheet_id"],
                        table_id=tp["table_ix"],
                    ),
                ),
            )

        return ContentLoweringOutput(rows=rows, document_summary="")

    # ------------------------------------------------------------------
    # Document lowering: document + section + paragraph + sentence (+ image/table)
    # ------------------------------------------------------------------
    section_counter = 0
    paragraph_counter = 0
    image_counter = 0
    table_counter = 0

    node_to_ids: Dict[str, Dict[str, int]] = {}

    # Document row first
    doc_title = root.title or str(file_path)
    node_to_ids[root.node_id] = _build_content_id(document_id=0)
    doc_summary = ""
    try:
        doc_summary = str(getattr(root, "summary", None) or "").strip()
    except Exception:
        doc_summary = ""
    if doc_summary:
        doc_summary = clip_text_to_token_limit_conservative(
            doc_summary,
            settings.EMBEDDING_MAX_INPUT_TOKENS,
            settings.EMBEDDING_ENCODING,
        )
    rows.append(
        _row(
            content_type=ContentType.DOCUMENT,
            title=doc_title,
            summary=doc_summary,
            content_text=None,
            content_id=node_to_ids[root.node_id],
        ),
    )

    def walk(node: ContentNode, inherited: Dict[str, int]) -> None:
        nonlocal section_counter, paragraph_counter, image_counter, table_counter

        for child in _stable_children(graph, node):
            if child.kind == NodeKind.SECTION:
                section_id = section_counter
                section_counter += 1
                cid = _build_content_id(document_id=0, section_id=section_id)
                node_to_ids[child.node_id] = cid
                rows.append(
                    _row(
                        content_type=ContentType.SECTION,
                        title=child.title,
                        summary=child.summary or (child.title or ""),
                        content_text=None,
                        content_id=cid,
                    ),
                )
                walk(child, cid)
                continue

            if child.kind == NodeKind.PARAGRAPH:
                paragraph_id = paragraph_counter
                paragraph_counter += 1
                cid = dict(inherited)
                cid.update({"paragraph": paragraph_id})
                node_to_ids[child.node_id] = cid

                text = (child.text or "").strip()
                raw_summary = (child.summary or "").strip() or text
                summary = clip_text_to_token_limit_conservative(
                    raw_summary,
                    settings.EMBEDDING_MAX_INPUT_TOKENS,
                    settings.EMBEDDING_ENCODING,
                )
                rows.append(
                    _row(
                        content_type=ContentType.PARAGRAPH,
                        title=child.title,
                        summary=summary,
                        content_text=text,
                        content_id=cid,
                    ),
                )
                walk(child, cid)
                continue

            if child.kind == NodeKind.SENTENCE:
                sid: Optional[int] = None
                try:
                    sid = int(
                        getattr(
                            getattr(child, "payload", None),
                            "sentence_index",
                            None,
                        ),
                    )
                except Exception:
                    sid = None
                if sid is None:
                    sid = 0
                cid = dict(inherited)
                cid.update({"sentence": int(sid)})
                node_to_ids[child.node_id] = cid
                text = (child.text or "").strip()
                raw_summary = (child.summary or "").strip() or text
                summary = clip_text_to_token_limit_conservative(
                    raw_summary,
                    settings.EMBEDDING_MAX_INPUT_TOKENS,
                    settings.EMBEDDING_ENCODING,
                )
                rows.append(
                    _row(
                        content_type=ContentType.SENTENCE,
                        title=None,
                        summary=summary,
                        content_text=text,
                        content_id=cid,
                    ),
                )
                continue

            if child.kind == NodeKind.IMAGE:
                image_id = image_counter
                image_counter += 1
                cid = dict(inherited)
                cid.update({"image": image_id})
                node_to_ids[child.node_id] = cid
                caption = ""
                try:
                    caption = str(
                        getattr(getattr(child, "payload", None), "caption", "") or "",
                    ).strip()
                except Exception:
                    caption = ""
                rows.append(
                    _row(
                        content_type=ContentType.IMAGE,
                        title=child.title,
                        summary=caption,
                        content_text=(caption or None),
                        content_id=cid,
                    ),
                )
                continue

            if child.kind == NodeKind.TABLE:
                table_id = table_counter
                table_counter += 1
                cid = dict(inherited)
                cid.update({"table": table_id})
                node_to_ids[child.node_id] = cid
                label = (
                    child.title or child.meta.get("table_label") or f"{table_id:02d}"
                )
                rows.append(
                    _row(
                        content_type=ContentType.TABLE,
                        title=str(label),
                        summary=str(label),
                        content_text=None,
                        content_id=cid,
                    ),
                )
                continue

            # Default: recurse without emitting a row
            walk(child, inherited)

    walk(root, node_to_ids[root.node_id])
    return ContentLoweringOutput(rows=rows, document_summary=doc_summary)
