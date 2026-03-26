"""
Graph builders from DoclingDocument → ContentGraph (+ extracted tables).

These functions are the bridge between Docling's internal item tree and Unity's
format-agnostic `ContentGraph`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from unity.file_manager.file_parsers.types.graph import (
    BBox,
    ContentGraph,
    ContentNode,
    DocumentPayload,
    ImagePayload,
    ParagraphPayload,
    Provenance,
    SectionPayload,
    SentencePayload,
    SheetPayload,
    TablePayload,
)
from unity.file_manager.file_parsers.types.table import ExtractedTable


def _bbox_from_docling(bbox_obj: Any) -> Optional[BBox]:
    if bbox_obj is None:
        return None
    d: Dict[str, Any] = {}
    try:
        if hasattr(bbox_obj, "model_dump"):
            d = bbox_obj.model_dump()
        else:
            d = {k: getattr(bbox_obj, k, None) for k in ("l", "t", "r", "b")}
    except Exception:
        d = {}

    try:
        origin_val = getattr(bbox_obj, "coord_origin", None)
        origin = getattr(origin_val, "value", None) or getattr(origin_val, "name", None)
    except Exception:
        origin = None

    try:
        l = float(d.get("l"))
        t = float(d.get("t"))
        r = float(d.get("r"))
        b = float(d.get("b"))
        return BBox(l=l, t=t, r=r, b=b, origin=origin)
    except Exception:
        return None


def _prov_from_docling(item: Any) -> Optional[Provenance]:
    try:
        prov = getattr(item, "prov", None) or []
        if not prov:
            return None
        p0 = prov[0]
        return Provenance(
            page_no=getattr(p0, "page_no", None),
            bbox=_bbox_from_docling(getattr(p0, "bbox", None)),
        )
    except Exception:
        return None


class _IdGen:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def next(self, kind: str) -> str:
        i = self._counters[kind]
        self._counters[kind] = i + 1
        return f"{kind}:{i}"


class _GraphBuilder:
    def __init__(self) -> None:
        self.ids = _IdGen()
        self.nodes: Dict[str, ContentNode] = {}
        self.root_id: Optional[str] = None

    def add_node(
        self,
        *,
        kind: str,
        parent_id: Optional[str],
        title: Optional[str] = None,
        text: Optional[str] = None,
        summary: Optional[str] = None,
        provenance: Optional[Provenance] = None,
        payload: Any = None,
        meta: Optional[Dict[str, Any]] = None,
        order: Optional[int] = None,
    ) -> str:
        node_id = self.ids.next(kind)
        node = ContentNode(
            node_id=node_id,
            kind=kind,  # type: ignore[arg-type]
            parent_id=parent_id,
            children_ids=[],
            order=order,
            title=title,
            text=text,
            summary=summary,
            provenance=provenance,
            payload=payload,
            meta=dict(meta or {}),
        )
        self.nodes[node_id] = node
        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        return node_id

    def finish(self) -> ContentGraph:
        assert self.root_id is not None, "root_id must be set"
        return ContentGraph(root_id=self.root_id, nodes=self.nodes)


def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.

    We keep this simple and robust. If higher quality segmentation is desired,
    it can be swapped for spaCy in a later step without changing graph I/O.
    """
    import re

    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


@dataclass(frozen=True)
class DocumentGraphBuildResult:
    graph: ContentGraph
    tables: List[ExtractedTable]


def build_document_graph_from_docling(
    docling_doc: Any,
    *,
    keep_table_self_refs: Optional[set[str]] = None,
) -> DocumentGraphBuildResult:
    """
    Build a document-oriented graph (PDF/DOCX) from a DoclingDocument.

    Mapping rules
    -------------
    - TitleItem -> document.title (first one wins)
    - SectionHeaderItem -> section nodes (hierarchy by `level`)
    - TextItem/ListItem/Code/etc -> paragraph nodes
    - Paragraphs -> sentence nodes (light splitter)
    - TableItem -> table nodes + ExtractedTable outputs
    - PictureItem -> image nodes
    """
    from docling_core.types.doc.labels import DocItemLabel

    b = _GraphBuilder()
    doc_id = b.add_node(kind="document", parent_id=None, payload=DocumentPayload())
    b.root_id = doc_id

    section_stack: List[Tuple[int, str, str]] = []  # (level, node_id, title)
    table_out: List[ExtractedTable] = []
    table_ix = 0
    para_ix = 0

    def current_section_parent() -> str:
        return section_stack[-1][1] if section_stack else doc_id

    for item, _level in docling_doc.iterate_items(
        with_groups=False,
        traverse_pictures=True,
    ):
        label = getattr(item, "label", None)

        # Document title
        if label == DocItemLabel.TITLE:
            if not b.nodes[doc_id].title:
                b.nodes[doc_id].title = getattr(item, "text", None) or getattr(
                    item,
                    "orig",
                    None,
                )
            continue

        # Sections
        if label == DocItemLabel.SECTION_HEADER:
            title = (getattr(item, "text", None) or "").strip()
            if not title:
                continue
            level_num = int(getattr(item, "level", 1) or 1)

            while section_stack and section_stack[-1][0] >= level_num:
                section_stack.pop()
            parent = current_section_parent()

            # Compute a human path snapshot (titles only)
            path = [t for (_lvl, _sid, t) in section_stack] + [title]

            sec_id = b.add_node(
                kind="section",
                parent_id=parent,
                title=title,
                provenance=_prov_from_docling(item),
                payload=SectionPayload(level=level_num, path=path),
            )
            section_stack.append((level_num, sec_id, title))
            continue

        # Tables
        if label == DocItemLabel.TABLE:
            try:
                ref = getattr(item, "self_ref", None)
            except Exception:
                ref = None
            ref_key = str(ref) if ref is not None else None
            if (
                keep_table_self_refs is not None
                and ref_key is not None
                and ref_key not in keep_table_self_refs
            ):
                continue
            table_ix += 1
            # DataFrame conversion is the most robust + stable representation.
            try:
                df = item.export_to_dataframe(doc=docling_doc)
                columns = [str(c) for c in list(df.columns)]
                rows = df.to_dict(orient="records")
            except Exception:
                columns = []
                rows = []

            # Sample for catalog / summary (bounded)
            sample_rows = rows[:25] if rows else []

            # Label: prefer section path; fallback to numeric index
            if section_stack:
                label_str = " / ".join([t for (_lvl, _sid, t) in section_stack])
            else:
                label_str = f"{table_ix:02d}"

            table_parent = current_section_parent()
            node_id = b.add_node(
                kind="table",
                parent_id=table_parent,
                title=label_str,
                provenance=_prov_from_docling(item),
                payload=TablePayload(
                    label=label_str,
                    columns=columns,
                    sample_rows=sample_rows,
                    num_rows=len(rows) if rows is not None else None,
                    num_cols=len(columns) if columns is not None else None,
                ),
                meta={"table_label": label_str},
            )

            table_out.append(
                ExtractedTable(
                    table_id=node_id,
                    label=label_str,
                    sheet_name=None,
                    columns=columns,
                    rows=rows,
                    sample_rows=sample_rows,
                    num_rows=len(rows) if rows is not None else None,
                    num_cols=len(columns) if columns is not None else None,
                ),
            )
            continue

        # Pictures
        if label == DocItemLabel.PICTURE:
            caption = ""
            try:
                caption = (item.caption_text(docling_doc) or "").strip()
            except Exception:
                caption = ""
            b.add_node(
                kind="image",
                parent_id=current_section_parent(),
                title=None,
                text=None,
                provenance=_prov_from_docling(item),
                payload=ImagePayload(
                    caption=caption,
                    image_ref=getattr(item, "self_ref", None),
                ),
                meta={"caption": caption} if caption else {},
            )
            continue

        # Paragraph-like items
        if label in (
            DocItemLabel.PARAGRAPH,
            DocItemLabel.TEXT,
            DocItemLabel.LIST_ITEM,
            DocItemLabel.CODE,
            DocItemLabel.REFERENCE,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
        ):
            raw = getattr(item, "text", None) or getattr(item, "orig", None) or ""
            text = str(raw).strip()
            if not text:
                continue

            para_ix += 1
            par_id = b.add_node(
                kind="paragraph",
                parent_id=current_section_parent(),
                title=None,
                text=text,
                provenance=_prov_from_docling(item),
                payload=ParagraphPayload(),
                order=para_ix,
            )

            # Sentence children (best-effort)
            sent_ix = 0
            for s in _split_sentences(text):
                sent_ix += 1
                b.add_node(
                    kind="sentence",
                    parent_id=par_id,
                    title=None,
                    text=s,
                    provenance=_prov_from_docling(item),
                    payload=SentencePayload(sentence_index=sent_ix),
                    order=sent_ix,
                )
            continue

    return DocumentGraphBuildResult(graph=b.finish(), tables=table_out)


@dataclass(frozen=True)
class SpreadsheetGraphBuildResult:
    graph: ContentGraph
    tables: List[ExtractedTable]
    sheet_names: List[str]


def build_spreadsheet_graph_from_docling(
    docling_doc: Any,
    *,
    keep_table_self_refs: Optional[set[str]] = None,
) -> SpreadsheetGraphBuildResult:
    """
    Build a spreadsheet-oriented graph (XLSX) from a DoclingDocument.

    We model spreadsheets explicitly as:
    document -> sheet -> table/image

    This keeps `/Content/` intuitive (sheet ≠ section).
    """
    from docling_core.types.doc.labels import DocItemLabel, GroupLabel
    from docling_core.types.doc.document import GroupItem

    b = _GraphBuilder()
    doc_id = b.add_node(kind="document", parent_id=None, payload=DocumentPayload())
    b.root_id = doc_id

    sheet_names: List[str] = []
    current_sheet_id: Optional[str] = None
    current_sheet_name: Optional[str] = None
    table_out: List[ExtractedTable] = []

    sheet_index = -1
    table_index_by_sheet: Dict[str, int] = defaultdict(int)

    def ensure_sheet(name: str) -> str:
        nonlocal sheet_index, current_sheet_id, current_sheet_name
        if current_sheet_id is not None and current_sheet_name == name:
            return current_sheet_id
        sheet_index += 1
        sheet_names.append(name)
        current_sheet_name = name
        current_sheet_id = b.add_node(
            kind="sheet",
            parent_id=doc_id,
            title=name,
            payload=SheetPayload(sheet_index=sheet_index, sheet_name=name),
            meta={"sheet_name": name, "sheet_index": sheet_index},
            order=sheet_index,
        )
        return current_sheet_id

    def maybe_sheet_name_from_group(g: Any) -> Optional[str]:
        name = getattr(g, "name", None)
        label = getattr(g, "label", None)
        if label == GroupLabel.SHEET:
            return str(name) if name else "Sheet"
        if (
            label == GroupLabel.SECTION
            and isinstance(name, str)
            and name.lower().startswith("sheet:")
        ):
            return name.split(":", 1)[-1].strip() or "Sheet"
        return None

    for item, _level in docling_doc.iterate_items(
        with_groups=True,
        traverse_pictures=True,
    ):
        # Detect sheet boundaries via GroupItems
        if isinstance(item, GroupItem):
            sname = maybe_sheet_name_from_group(item)
            if sname:
                ensure_sheet(sname)
            continue

        label = getattr(item, "label", None)

        if label == DocItemLabel.TABLE:
            try:
                ref = getattr(item, "self_ref", None)
            except Exception:
                ref = None
            ref_key = str(ref) if ref is not None else None
            if (
                keep_table_self_refs is not None
                and ref_key is not None
                and ref_key not in keep_table_self_refs
            ):
                continue
            # Ensure we always have a sheet context
            if current_sheet_id is None:
                ensure_sheet("Sheet 1")
            assert current_sheet_id is not None

            # Extract table rows/columns
            try:
                df = item.export_to_dataframe(doc=docling_doc)
                columns = [str(c) for c in list(df.columns)]
                rows = df.to_dict(orient="records")
                del df
            except Exception:
                columns = []
                rows = []

            sample_rows = rows[:25] if rows else []

            table_index_by_sheet[current_sheet_id] += 1
            local_ix = table_index_by_sheet[current_sheet_id]

            # Label should be stable and unique within a file.
            # Prefer sheet_name if only one table; otherwise suffix with index.
            if current_sheet_name:
                label_str = (
                    current_sheet_name
                    if local_ix == 1
                    else f"{current_sheet_name}_{local_ix:02d}"
                )
            else:
                label_str = f"{local_ix:02d}"

            node_id = b.add_node(
                kind="table",
                parent_id=current_sheet_id,
                title=label_str,
                provenance=_prov_from_docling(item),
                payload=TablePayload(
                    label=label_str,
                    columns=columns,
                    sample_rows=sample_rows,
                    num_rows=len(rows) if rows is not None else None,
                    num_cols=len(columns) if columns is not None else None,
                ),
                meta={
                    "table_label": label_str,
                    "sheet_name": current_sheet_name,
                    "table_index_in_sheet": local_ix,
                },
                order=local_ix,
            )

            table_out.append(
                ExtractedTable(
                    table_id=node_id,
                    label=label_str,
                    sheet_name=current_sheet_name,
                    columns=columns,
                    rows=rows,
                    sample_rows=sample_rows,
                    num_rows=len(rows) if rows is not None else None,
                    num_cols=len(columns) if columns is not None else None,
                ),
            )
            continue

        if label == DocItemLabel.PICTURE:
            if current_sheet_id is None:
                ensure_sheet("Sheet 1")
            assert current_sheet_id is not None
            caption = ""
            try:
                caption = (item.caption_text(docling_doc) or "").strip()
            except Exception:
                caption = ""
            b.add_node(
                kind="image",
                parent_id=current_sheet_id,
                payload=ImagePayload(
                    caption=caption,
                    image_ref=getattr(item, "self_ref", None),
                ),
                provenance=_prov_from_docling(item),
                meta=(
                    {"caption": caption, "sheet_name": current_sheet_name}
                    if caption or current_sheet_name
                    else {}
                ),
            )
            continue

    if not sheet_names:
        # Some XLSX may not expose groups; create a synthetic sheet so lowering is stable.
        ensure_sheet("Sheet 1")

    return SpreadsheetGraphBuildResult(
        graph=b.finish(),
        tables=table_out,
        sheet_names=sheet_names,
    )
