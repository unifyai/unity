from playwright.sync_api import Page

# ==================================================================
# Intersection over Union (IoU) Calculation
# ==================================================================
def _dedup(elements, iou_threshold=0.8):
    """
    Performs Non-Maximal Suppression to remove overlapping bounding boxes.
    """
    out = []
    # Sort elements by a confidence score if available, otherwise just process
    # For now, we assume larger elements are more important
    elements.sort(key=lambda x: x.get('width', 0) * x.get('height', 0), reverse=True)
    
    for el in elements:
        # Check if the element significantly overlaps with any element already in the output
        is_overlapping = False
        for o in out:
            if _overlap_ratio(el, o) >= iou_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            out.append(el)
    return out

def _overlap_ratio(box_a, box_b):
    """
    Return intersectionArea / min(areaA, areaB).
    1.0 means the smaller box is completely inside the other.
    """
    x_left   = max(box_a['left'],              box_b['left'])
    y_top    = max(box_a['top'],               box_b['top'])
    x_right  = min(box_a['left']+box_a['width'], box_b['left']+box_b['width'])
    y_bottom = min(box_a['top'] +box_a['height'], box_b['top'] +box_b['height'])

    inter_w = max(0, x_right  - x_left)
    inter_h = max(0, y_bottom - y_top)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0

    return inter / min(box_a['width'] * box_a['height'],
                       box_b['width'] * box_b['height'])


# ==================================================================
# Fuse Heuristic and Vision Elements
# ==================================================================
def _fuse_elements(
    vision_elements: list[dict],
    heuristic_elements: list[dict],
    page: Page,
    overlap_threshold: float = 0.5,
):
    """
    Merge heuristic + vision results into one list.

    Pair a heuristic element with the *single* vision element that covers
    at least `overlap_threshold` of the smaller box.  Prefer the vision
    label and bounding box; keep the heuristic ElementHandle for clicks.
    """

    fused: list[dict] = []

    # ────────────────────────────────────────────────────────────────
    # 0️⃣  Trivial cases – only one side has data
    # ────────────────────────────────────────────────────────────────
    if not vision_elements:
        for h in heuristic_elements:
            fused.append({**h, "source": "heuristic"})
        return _finalize_and_renumber_elements(fused, 0, 0)

    if not heuristic_elements:
        try:
            vp = page.evaluate(
                "() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY})"
            )
            vw, vh = vp["w"], vp["h"]; sx, sy = vp["sx"], vp["sy"]
        except Exception:
            vw = vh = 1; sx = sy = 0  # fall back

        for v in vision_elements:
            b = v.get("bbox") or []
            fused.append({
                "label": (v.get("label") or v.get("content", "")).strip(),
                "handle": None,
                "bbox":   b,
                "source": "vision",
                "left":   (b[0] * vw) + sx,
                "top":    (b[1] * vh) + sy,
                "width":  (b[2] - b[0]) * vw,
                "height": (b[3] - b[1]) * vh,
                "fixed":  False,
            })
        return _finalize_and_renumber_elements(fused, sx, sy)

    # ────────────────────────────────────────────────────────────────
    # 1️⃣  Full fusion
    # ────────────────────────────────────────────────────────────────
    try:
        vp = page.evaluate(
            "() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY})"
        )
        vw, vh = vp["w"], vp["h"]; sx, sy = vp["sx"], vp["sy"]
    except Exception:
        vw = vh = 1; sx = sy = 0

    # Denormalise vision boxes once
    denorm_v = []
    for v in vision_elements:
        b = v.get("bbox")
        if not b:
            continue
        denorm_v.append({
            **v,
            "left":   (b[0] * vw) + sx,
            "top":    (b[1] * vh) + sy,
            "width":  (b[2] - b[0]) * vw,
            "height": (b[3] - b[1]) * vh,
        })

    used_v = [False] * len(denorm_v)

   
    # ── Pre-Pass: Disqualify coarse "container" vision boxes ───────────
    # Identify and ignore any single vision element that contains multiple
    # distinct heuristic elements, as it's likely a less useful container.
    for j, v in enumerate(denorm_v):
        # Count how many heuristic elements this vision box overlaps with
        overlapping_heuristic_count = 0
        for h in heuristic_elements:
            if _overlap_ratio(h, v) >= overlap_threshold:
                overlapping_heuristic_count += 1
        
        # If a vision box corresponds to more than one heuristic box,
        # it's too coarse. Mark it as "used" so it's ignored in the next step.
        if overlapping_heuristic_count > 1:
            used_v[j] = True

    # ── Pass A: for every heuristic element, try to find a vision partner ──
    for h in heuristic_elements:
        best_j = -1; best_ov = -1.0
        for j, v in enumerate(denorm_v):
            if used_v[j]:
                continue
            ov = _overlap_ratio(h, v)
            if ov > best_ov:
                best_ov, best_j = ov, j

        if best_ov >= overlap_threshold:
            v = denorm_v[best_j]; used_v[best_j] = True
            fused.append({
                "label": (v.get("label") or v.get("content") or h.get("label", "")).strip(),
                "handle": h.get("handle"),           # keeps clickability
                "bbox":   v["bbox"],                 # vision bbox (normalised)
                "source": "hybrid",
                "left":   v["left"],  "top": v["top"],
                "width":  v["width"], "height": v["height"],
                "fixed":  h.get("fixed", False),
            })
        else:
            fused.append({**h, "source": "heuristic"})

    # ── Pass B: any vision elements still unused go in as-is ─────────
    for j, v in enumerate(denorm_v):
        if not used_v[j]:
            fused.append({
                **v,
                "handle": None,
                "fixed":  False,
                "source": "vision",
            })

    return _finalize_and_renumber_elements(fused, sx, sy)

def _finalize_and_renumber_elements(elements: list[dict], sx: int, sy: int) -> list[dict]:
    """Helper to apply final transformations and assign sequential IDs."""
    final_list = []
    # Sort elements by position (top-to-bottom, then left-to-right) for consistent numbering
    elements.sort(key=lambda el: (el.get('top', 0), el.get('left', 0)))
    
    for i, el in enumerate(elements):
        final_list.append({
            'id': i + 1,  # Assign a new, sequential ID
            'label': el.get('label', ''),
            'handle': el.get('handle'),
            'bbox': el.get('bbox'),
            'fixed': el.get('fixed', False),
            'left': el.get('left'),
            'top': el.get('top'),
            'width': el.get('width'),
            'height': el.get('height'),
            'vleft': el.get('left', 0) - sx,
            'vtop': el.get('top', 0) - sy,
            'source': el.get('source', 'unknown')
        })
    return final_list


# ------------------------------------------------------------------
# Stable-ID assignment between frames
# ------------------------------------------------------------------
_last_frame: list[dict] | None = None
_next_id: int = 1
_retired_ids: set[int] = set()

def _assign_stable_ids(current: list[dict]) -> list[dict]:
    global _last_frame, _next_id, _retired_ids

    if _last_frame is None:
        for el in current:
            el["id"] = _next_id
            _next_id += 1
        _last_frame = current
        return current

    # ---- look-up tables from previous frame ----
    by_handle = {el["handle"]: el for el in _last_frame if el.get("handle")}
    by_label  = [(el, el.get("label","").lower()) for el in _last_frame]

    def match(old_el, new_el):
        return _overlap_ratio(old_el, new_el) >= 0.5

    taken: set[int] = set()

    # 1️⃣  try to reuse ids
    for new_el in current:
        reused = None

        h = new_el.get("handle")
        if h and h in by_handle and match(by_handle[h], new_el):
            reused = by_handle[h]["id"]
        else:
            lbl = new_el.get("label","").lower()
            for old_el, old_lbl in by_label:
                if lbl == old_lbl and match(old_el, new_el):
                    reused = old_el["id"]; break

        if reused is not None and reused not in taken:
            new_el["id"] = reused
            taken.add(reused)
        else:
            # 2️⃣  assign a brand-new id, skipping any retired numbers
            while _next_id in taken or _next_id in _retired_ids:
                _next_id += 1
            new_el["id"] = _next_id
            taken.add(_next_id)
            _next_id += 1

    # ids that existed in previous frame but not in this one → retire them
    prev_ids = {el["id"] for el in _last_frame}
    missing  = prev_ids - taken
    _retired_ids.update(missing)

    _last_frame = current
    return current

def reset_stable_ids():
    """Forget everything we knew about old elements."""
    global _last_frame, _next_id, _retired_ids
    _last_frame = None
    _next_id = 1
    _retired_ids.clear()