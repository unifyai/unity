from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .types.document import DocumentTable, CoordOrigin


def _normalize_section_path(path: Optional[List[str]]) -> Optional[Tuple[str, ...]]:
    if not path:
        return None
    return tuple(str(x).strip() for x in path if str(x).strip())


def extract_bbox_with_origin(bbox_obj: Any) -> Dict[str, Any]:
    """Extract a bbox dict including coord_origin from a provenance bbox object.

    Returns a dict with keys: l, t, r, b (when available) and coord_origin ("TOPLEFT"|"BOTTOMLEFT").
    """
    bbox: Dict[str, Any] = {}

    # Try model_dump first when available to get numeric fields
    if hasattr(bbox_obj, "model_dump"):
        md = bbox_obj.model_dump()
        for k in ("l", "t", "r", "b"):
            if k in md and md[k] is not None:
                try:
                    bbox[k] = float(md[k])
                except Exception:
                    pass
    else:
        for k in ("l", "t", "r", "b"):
            if hasattr(bbox_obj, k):
                try:
                    val = getattr(bbox_obj, k)
                    if val is not None:
                        bbox[k] = float(val)
                except Exception:
                    pass

    # Capture coord origin as our enum label
    origin_val = None
    if hasattr(bbox_obj, "coord_origin"):
        origin_val = getattr(bbox_obj, "coord_origin")
    # Resolve to a simple string value
    origin_str: Optional[str] = None
    if origin_val is not None:
        try:
            origin_str = getattr(origin_val, "value", None) or getattr(
                origin_val,
                "name",
                None,
            )
            if not isinstance(origin_str, str):
                origin_str = str(origin_val)
        except Exception:
            origin_str = None
    if isinstance(origin_str, str) and origin_str in (
        CoordOrigin.TOPLEFT.value,
        CoordOrigin.BOTTOMLEFT.value,
    ):
        bbox["coord_origin"] = origin_str

    return bbox


def _get_coord_origin_from_bbox(
    bbox: Optional[Dict[str, Any]],
) -> Optional[CoordOrigin]:
    if not bbox or not isinstance(bbox, dict):
        return None
    val = bbox.get("coord_origin")
    if isinstance(val, CoordOrigin):
        return val
    if isinstance(val, str):
        if val.upper() == CoordOrigin.TOPLEFT.value:
            return CoordOrigin.TOPLEFT
        if val.upper() == CoordOrigin.BOTTOMLEFT.value:
            return CoordOrigin.BOTTOMLEFT
    return None


def _get_float(d: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not d or key not in d:
        return None
    try:
        return float(d[key])
    except Exception:
        return None


def should_merge_tables(prev: DocumentTable, nxt: DocumentTable) -> bool:
    """Return True when two tables should be merged per the heuristic rules.

    Rules implemented:
    1) Same label (use element_type)
    2) Consecutive pages (nxt.page == prev.page + 1)
    3) Same section_path (must be present); if not present, do not merge
    4) Vertical order across pages using coord_origin and bbox.t
    5) Same num_cols
    """
    # 1) Same label
    if (prev.element_type or "").strip() != (nxt.element_type or "").strip():
        return False

    # 2) Consecutive pages
    if not isinstance(prev.page, int) or not isinstance(nxt.page, int):
        return False
    if nxt.page != prev.page + 1:
        return False

    # 3) Same section_path and must be present
    p_path = _normalize_section_path(prev.section_path)
    n_path = _normalize_section_path(nxt.section_path)
    if not p_path or not n_path:
        return False
    if p_path != n_path:
        return False

    # 5) Same number of columns
    if (
        prev.num_cols is None
        or nxt.num_cols is None
        or int(prev.num_cols) != int(nxt.num_cols)
    ):
        return False

    # 4) Vertical positioning using coord_origin and bbox.t
    p_origin = _get_coord_origin_from_bbox(prev.bbox)
    n_origin = _get_coord_origin_from_bbox(nxt.bbox)
    # If either origin missing, cannot safely compare
    if p_origin is None or n_origin is None:
        return False
    # If origins differ, we would need page height to convert reliably; skip (unlikely)
    if p_origin != n_origin:
        return False

    t_prev = _get_float(prev.bbox, "t")
    t_next = _get_float(nxt.bbox, "t")
    if t_prev is None or t_next is None:
        return False

    if p_origin == CoordOrigin.BOTTOMLEFT:
        return t_next >= t_prev
    else:  # TOPLEFT
        return t_next <= t_prev


def merge_table_rows(prev: DocumentTable, nxt: DocumentTable) -> DocumentTable:
    """Merge nxt into prev by appending rows and updating num_rows. Headers remain from prev."""
    prev_rows = list(prev.rows or [])
    next_rows = list(nxt.rows or [])
    prev.rows = prev_rows + next_rows if next_rows else prev_rows

    # Update num_rows conservatively based on existing counts if present, else derive
    prev_count = (
        int(prev.num_rows) if isinstance(prev.num_rows, int) else len(prev.rows or [])
    )
    next_count = int(nxt.num_rows) if isinstance(nxt.num_rows, int) else len(next_rows)
    prev.num_rows = prev_count + next_count

    # Retain prev bbox and coord_origin as is
    return prev


def merge_consecutive_tables(tables: List[DocumentTable]) -> List[DocumentTable]:
    """Merge chains of consecutive tables according to the heuristic and return a new list."""
    if not tables:
        return tables
    merged: List[DocumentTable] = []
    i = 0
    n = len(tables)
    while i < n:
        base = tables[i]
        j = i + 1
        while j < n and should_merge_tables(base, tables[j]):
            base = merge_table_rows(base, tables[j])
            j += 1
        merged.append(base)
        i = j
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Docling TableItem merge helpers (operate on Docling objects so export_to_html
# yields correct, combined HTML without client-side stitching)
# ──────────────────────────────────────────────────────────────────────────────


def header_row_count_for_table_item(table_item: Any) -> int:
    """Count leading header rows in a Docling TableItem by scanning column_header flags."""
    try:
        grid = getattr(getattr(table_item, "data", None), "grid", [])
        count = 0
        for row in grid:
            any_header = any(getattr(cell, "column_header", False) for cell in row)
            if any_header:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _normalize_doc_index_path(
    path: Optional[Tuple[str, ...] | List[str]],
) -> Optional[Tuple[str, ...]]:
    if not path:
        return None
    try:
        return tuple(str(x).strip() for x in path if str(x).strip())
    except Exception:
        return None


def get_section_path_for_item(
    table_item: Any,
    doc_index: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    if not doc_index:
        return None
    try:
        ref = getattr(table_item, "self_ref", None)
        if not ref:
            return None
        ref_to_path = doc_index.get("ref_to_path", {})
        path = _normalize_doc_index_path(ref_to_path.get(ref))
        return path
    except Exception:
        return None


def extract_bbox_origin_from_prov(prov: Any) -> Dict[str, Any]:
    """Extract bbox with origin from a Docling provenance item."""
    try:
        bbox_obj = getattr(prov, "bbox", None)
        if bbox_obj is None:
            return {}
        out = extract_bbox_with_origin(bbox_obj)
        return out
    except Exception:
        return {}


def get_page_bbox_origin(
    doc: Any,
    table_item: Any,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[CoordOrigin]]:
    """Return (page_no, bbox_dict_with_origin, origin_enum) for the first provenance of a TableItem."""
    try:
        provs = list(getattr(table_item, "prov", []) or [])
        if not provs:
            return None, None, None
        prov0 = provs[0]
        page_no = getattr(prov0, "page_no", None)
        bbox_d = extract_bbox_origin_from_prov(prov0)
        origin = None
        if isinstance(bbox_d.get("coord_origin"), str):
            val = bbox_d["coord_origin"].upper()
            if val == CoordOrigin.TOPLEFT.value:
                origin = CoordOrigin.TOPLEFT
            elif val == CoordOrigin.BOTTOMLEFT.value:
                origin = CoordOrigin.BOTTOMLEFT
        return page_no, bbox_d, origin
    except Exception:
        return None, None, None


def unify_bbox_origin(
    bbox: Dict[str, Any],
    target_origin: CoordOrigin,
    page_h: float,
) -> Dict[str, Any]:
    """Convert bbox to target origin using page height if needed.

    Assumes bbox dict has keys 't','b' and optional 'coord_origin'.
    """
    try:
        if not bbox:
            return {}
        current = bbox.get("coord_origin")
        if isinstance(current, CoordOrigin):
            cur_val = current.value
        else:
            cur_val = str(current).upper() if current is not None else None
        if cur_val == target_origin.value:
            out = dict(bbox)
            out["coord_origin"] = target_origin.value
            return out
        t = float(bbox.get("t")) if bbox.get("t") is not None else None
        b = float(bbox.get("b")) if bbox.get("b") is not None else None
        if t is None or b is None:
            # Can't convert
            return {}
        if (
            target_origin == CoordOrigin.TOPLEFT
            and cur_val == CoordOrigin.BOTTOMLEFT.value
        ):
            # bottom-left -> top-left
            t_new = page_h - b
            b_new = page_h - t
        elif (
            target_origin == CoordOrigin.BOTTOMLEFT
            and cur_val == CoordOrigin.TOPLEFT.value
        ):
            # top-left -> bottom-left
            t_new = page_h - b
            b_new = page_h - t
        else:
            # Unknown origin, return as-is with target tag
            t_new = t
            b_new = b
        out = dict(bbox)
        out["t"], out["b"] = t_new, b_new
        out["coord_origin"] = target_origin.value
        return out
    except Exception:
        return {}


def should_merge_table_items(
    prev: Any,
    nxt: Any,
    doc: Any,
    doc_index: Optional[Dict[str, Any]],
) -> bool:
    """Strictly check all heuristics for Docling TableItems before merging."""
    try:
        prev_ref = getattr(prev, "self_ref", None)
        next_ref = getattr(nxt, "self_ref", None)
        # 1) Same label
        if str(getattr(prev, "label", "")) != str(getattr(nxt, "label", "")):
            return False

        # 2) Consecutive pages
        p_page, p_bbox, p_origin = get_page_bbox_origin(doc, prev)
        n_page, n_bbox, n_origin = get_page_bbox_origin(doc, nxt)
        if not isinstance(p_page, int) or not isinstance(n_page, int):
            return False
        if n_page != p_page + 1:
            return False

        # 3) Same section_path and both present
        p_path = get_section_path_for_item(prev, doc_index)
        n_path = get_section_path_for_item(nxt, doc_index)
        if not p_path or not n_path or p_path != n_path:
            return False

        # 5) Equal num_cols
        p_cols = getattr(getattr(prev, "data", None), "num_cols", None)
        n_cols = getattr(getattr(nxt, "data", None), "num_cols", None)
        if (
            not isinstance(p_cols, int)
            or not isinstance(n_cols, int)
            or p_cols != n_cols
        ):
            return False

        # 4) Vertical order by t w.r.t. coord origin
        if p_origin is None or n_origin is None or not p_bbox or not n_bbox:
            return False

        # Normalize next bbox to prev's origin when origins differ
        _, page_h = getattr(
            getattr(doc.pages.get(n_page, None), "size", None),
            "as_tuple",
            lambda: (None, None),
        )()
        if page_h is None:
            # Try alternate access
            try:
                _, page_h = doc.pages[n_page].size.as_tuple()
            except Exception:
                page_h = None

        n_bbox_same_origin = dict(n_bbox)
        if p_origin != n_origin:
            if page_h is None:
                return False
            n_bbox_same_origin = unify_bbox_origin(
                n_bbox_same_origin,
                p_origin,
                float(page_h),
            )
            if not n_bbox_same_origin:
                return False

        t_prev = float(p_bbox.get("t")) if p_bbox.get("t") is not None else None
        t_next = (
            float(n_bbox_same_origin.get("t"))
            if n_bbox_same_origin.get("t") is not None
            else None
        )
        if t_prev is None or t_next is None:
            return False

        if p_origin == CoordOrigin.BOTTOMLEFT:
            ok = t_next >= t_prev
        else:  # TOPLEFT
            ok = t_next <= t_prev
        return ok
    except Exception:
        return False


def merge_table_items_in_place(prev: Any, nxt: Any, doc: Any) -> Any:
    """Append non-header rows from nxt into prev (Docling TableItems) and update counts.

    Uses the library's TableData.add_rows so indices and num_rows are handled by Docling.
    """
    try:
        p_data = getattr(prev, "data", None)
        n_data = getattr(nxt, "data", None)
        if p_data is None or n_data is None:
            return prev
        p_grid = getattr(p_data, "grid", [])
        n_grid = getattr(n_data, "grid", [])
        headers_to_skip = header_row_count_for_table_item(nxt)
        rows_to_append = n_grid[headers_to_skip:]
        # Extract plain-text rows compatible with TableData.add_rows; drop fully empty rows
        text_rows: List[List[str]] = []
        skipped_blank = 0
        for row in rows_to_append:
            vals: List[str] = []
            any_text = False
            for cell in row:
                try:
                    if hasattr(cell, "_get_text"):
                        txt = cell._get_text(doc=doc)
                    else:
                        txt = getattr(cell, "text", "")
                except Exception:
                    txt = getattr(cell, "text", "")
                sval = txt or ""
                if isinstance(sval, str) and sval.strip():
                    any_text = True
                vals.append(sval)
            if any_text:
                # Ensure width matches columns
                if isinstance(p_data.num_cols, int) and len(vals) != p_data.num_cols:
                    if len(vals) > p_data.num_cols:
                        vals = vals[: p_data.num_cols]
                    else:
                        vals = vals + [""] * (p_data.num_cols - len(vals))
                text_rows.append(vals)
            else:
                skipped_blank += 1

        if text_rows:
            try:
                p_data.add_rows(text_rows)
            except Exception as e:
                pass

        # Update counts
        # num_rows is updated by add_rows; log for visibility

        # Retain num_cols; extend provenance for traceability
        try:
            pre_len = len(getattr(prev, "prov", []) or [])
            add_len = len(getattr(nxt, "prov", []) or [])
            prev.prov.extend(list(getattr(nxt, "prov", []) or []))
            post_len = len(getattr(prev, "prov", []) or [])
        except Exception:
            pass
        return prev
    except Exception:
        return prev


def merge_consecutive_table_items(
    doc: Any,
    doc_index: Optional[Dict[str, Any]],
) -> List[Any]:
    """Return a reduced list of Docling TableItems after in-place merges per heuristics."""
    try:
        tables = list(getattr(doc, "tables", []) or [])
    except Exception:
        tables = []
    if not tables:
        return []

    result: List[Any] = []
    i = 0
    n = len(tables)
    while i < n:
        base = tables[i]
        j = i + 1
        while j < n and should_merge_table_items(base, tables[j], doc, doc_index):
            base = merge_table_items_in_place(base, tables[j], doc)
            j += 1
        result.append(base)
        i = j
    return result
