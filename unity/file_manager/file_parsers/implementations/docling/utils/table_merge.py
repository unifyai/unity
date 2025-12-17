from __future__ import annotations

"""Docling TableItem merge helpers.

These helpers operate on Docling objects so that `export_to_html` yields correct,
combined HTML without client-side stitching.
"""

from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class CoordOrigin(str, Enum):
    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"


def extract_bbox_with_origin(bbox_obj: object) -> Dict[str, object]:
    """Extract a bbox dict including coord_origin from a provenance bbox object.

    Returns a dict with keys: l, t, r, b (when available) and coord_origin ("TOPLEFT"|"BOTTOMLEFT").
    """
    bbox: Dict[str, object] = {}

    # Try model_dump first when available to get numeric fields
    if hasattr(bbox_obj, "model_dump"):
        try:
            md = bbox_obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            md = {}
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
    origin_val = (
        getattr(bbox_obj, "coord_origin", None)
        if hasattr(bbox_obj, "coord_origin")
        else None
    )
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


def header_row_count_for_table_item(table_item: object) -> int:
    """Count leading header rows in a Docling TableItem by scanning column_header flags."""
    try:
        grid = getattr(getattr(table_item, "data", None), "grid", [])  # type: ignore[arg-type]
        count = 0
        for row in list(grid or []):
            any_header = any(getattr(cell, "column_header", False) for cell in row)
            if any_header:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _normalize_doc_index_path(
    path: Optional[Sequence[str] | Tuple[str, ...]],
) -> Optional[Tuple[str, ...]]:
    if not path:
        return None
    try:
        return tuple(str(x).strip() for x in path if str(x).strip())
    except Exception:
        return None


def _doc_index_ref_to_path(doc_index: object) -> Dict[str, Tuple[str, ...]]:
    """Extract ref_to_path mapping from either a typed index object or a raw dict."""
    if doc_index is None:
        return {}
    try:
        # Typed model style
        m = getattr(doc_index, "ref_to_path", None)
        if isinstance(m, dict):
            out: Dict[str, Tuple[str, ...]] = {}
            for k, v in m.items():
                if k is None:
                    continue
                norm = _normalize_doc_index_path(v)  # type: ignore[arg-type]
                if norm is not None:
                    out[str(k)] = norm
            return out
    except Exception:
        pass
    try:
        # Dict style
        if isinstance(doc_index, dict):
            m = doc_index.get("ref_to_path", {})
            out = {}
            if isinstance(m, dict):
                for k, v in m.items():
                    norm = _normalize_doc_index_path(v)  # type: ignore[arg-type]
                    if norm is not None:
                        out[str(k)] = norm
            return out
    except Exception:
        pass
    return {}


def get_section_path_for_item(
    table_item: object,
    doc_index: Optional[object],
) -> Optional[Tuple[str, ...]]:
    ref = getattr(table_item, "self_ref", None)
    if not ref:
        return None
    ref_to_path = _doc_index_ref_to_path(doc_index)
    return ref_to_path.get(str(ref))


def extract_bbox_origin_from_prov(prov: object) -> Dict[str, object]:
    """Extract bbox with origin from a Docling provenance item."""
    try:
        bbox_obj = getattr(prov, "bbox", None)
        if bbox_obj is None:
            return {}
        return extract_bbox_with_origin(bbox_obj)
    except Exception:
        return {}


def get_page_bbox_origin(
    doc: object,
    table_item: object,
) -> Tuple[Optional[int], Optional[Dict[str, object]], Optional[CoordOrigin]]:
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
            val = str(bbox_d["coord_origin"]).upper()
            if val == CoordOrigin.TOPLEFT.value:
                origin = CoordOrigin.TOPLEFT
            elif val == CoordOrigin.BOTTOMLEFT.value:
                origin = CoordOrigin.BOTTOMLEFT
        return int(page_no) if isinstance(page_no, int) else None, bbox_d, origin
    except Exception:
        return None, None, None


def unify_bbox_origin(
    bbox: Dict[str, object],
    target_origin: CoordOrigin,
    page_h: float,
) -> Dict[str, object]:
    """Convert bbox to target origin using page height if needed."""
    try:
        if not bbox:
            return {}
        cur_val = bbox.get("coord_origin")
        cur = str(cur_val).upper() if cur_val is not None else None
        if cur == target_origin.value:
            out = dict(bbox)
            out["coord_origin"] = target_origin.value
            return out

        t = float(bbox.get("t")) if bbox.get("t") is not None else None
        b = float(bbox.get("b")) if bbox.get("b") is not None else None
        if t is None or b is None:
            return {}

        # bottom-left <-> top-left conversion
        t_new = page_h - b
        b_new = page_h - t

        out = dict(bbox)
        out["t"], out["b"] = t_new, b_new
        out["coord_origin"] = target_origin.value
        return out
    except Exception:
        return {}


def should_merge_table_items(
    prev: object,
    nxt: object,
    doc: object,
    doc_index: Optional[object],
) -> bool:
    """Strictly check all heuristics for Docling TableItems before merging."""
    try:
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
        page_h = None
        try:
            pages = getattr(doc, "pages", None)
            if pages is not None:
                page = pages.get(n_page) if hasattr(pages, "get") else pages[n_page]  # type: ignore[index]
                size = getattr(page, "size", None)
                if size is not None and hasattr(size, "as_tuple"):
                    _w, _h = size.as_tuple()
                    page_h = _h
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
            return t_next >= t_prev
        return t_next <= t_prev
    except Exception:
        return False


def merge_table_items_in_place(prev: object, nxt: object, doc: object) -> object:
    """Append non-header rows from nxt into prev (Docling TableItems) and update counts."""
    try:
        p_data = getattr(prev, "data", None)
        n_data = getattr(nxt, "data", None)
        if p_data is None or n_data is None:
            return prev
        n_grid = getattr(n_data, "grid", [])
        headers_to_skip = header_row_count_for_table_item(nxt)
        rows_to_append = list(n_grid or [])[headers_to_skip:]

        text_rows: List[List[str]] = []
        for row in rows_to_append:
            vals: List[str] = []
            any_text = False
            for cell in row:
                try:
                    if hasattr(cell, "_get_text"):
                        txt = cell._get_text(doc=doc)  # type: ignore[attr-defined]
                    else:
                        txt = getattr(cell, "text", "")
                except Exception:
                    txt = getattr(cell, "text", "")
                sval = txt or ""
                if isinstance(sval, str) and sval.strip():
                    any_text = True
                vals.append(str(sval))
            if any_text:
                num_cols = getattr(p_data, "num_cols", None)
                if isinstance(num_cols, int) and len(vals) != num_cols:
                    if len(vals) > num_cols:
                        vals = vals[:num_cols]
                    else:
                        vals = vals + [""] * (num_cols - len(vals))
                text_rows.append(vals)

        if text_rows and hasattr(p_data, "add_rows"):
            try:
                p_data.add_rows(text_rows)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Extend provenance for traceability
        try:
            prev_prov = getattr(prev, "prov", None)
            nxt_prov = list(getattr(nxt, "prov", []) or [])
            if prev_prov is not None and nxt_prov and hasattr(prev_prov, "extend"):
                prev_prov.extend(nxt_prov)  # type: ignore[attr-defined]
        except Exception:
            pass
        return prev
    except Exception:
        return prev


def merge_consecutive_table_items(
    doc: object,
    doc_index: Optional[object],
) -> List[object]:
    """Return a reduced list of Docling TableItems after in-place merges per heuristics."""
    try:
        tables = list(getattr(doc, "tables", []) or [])
    except Exception:
        tables = []
    if not tables:
        return []

    result: List[object] = []
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
