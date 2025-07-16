from playwright.sync_api import Page
from typing import List


# ==================================================================
# Pixel-based click
# ==================================================================
def _click_at_bbox_center(
    page: Page,
    bbox_norm: List[float],
    debug: bool = False,
) -> None:
    """
    Sends a mouse click at the centre of *bbox_norm* and optionally draws a debug dot.

    * ``bbox_norm`` is **[x0, y0, x1, y1]** in the range *0 <= v <= 1* relative
      to the current viewport.
    * The function is deliberately lightweight and synchronous so it can be
      called from any rescue path without awaiting coroutines.
    """
    # 1. Calculate the center coordinates in viewport pixels
    vp = page.evaluate("() => ({w: innerWidth, h: innerHeight})")
    cx_px = (bbox_norm[0] + bbox_norm[2]) / 2 * vp["w"]
    cy_px = (bbox_norm[1] + bbox_norm[3]) / 2 * vp["h"]
    if debug:
        # 2. JavaScript to draw a red dot at the click coordinates ( for debugging )
        draw_dot_js = """
        (args) => {
            // Remove any old dot first
            document.getElementById('gemini-debug-dot')?.remove();

            const dot = document.createElement('div');
            dot.id = 'gemini-debug-dot';
            dot.style.position = 'fixed'; // Use 'fixed' to match viewport coordinates
            dot.style.left = `${args.x - 4}px`; // Offset to center the dot
            dot.style.top = `${args.y - 4}px`;  // Offset to center the dot
            dot.style.width = '8px';
            dot.style.height = '8px';
            dot.style.backgroundColor = 'red';
            dot.style.border = '1px solid white';
            dot.style.borderRadius = '50%';
            dot.style.zIndex = '9999999';    // Ensure it's on top of everything
            dot.style.pointerEvents = 'none'; // Make it non-interactive

            document.body.appendChild(dot);
        }
        """
        # 3. Execute the JS to draw the dot and pause briefly to see it
        page.evaluate(draw_dot_js, {"x": cx_px, "y": cy_px})
        page.wait_for_timeout(3000)  # 3-second pause to see the dot

    # 4. Perform the click at the exact same coordinates
    page.mouse.click(cx_px, cy_px)


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
    elements.sort(key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True)

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
    x_left = max(box_a["left"], box_b["left"])
    y_top = max(box_a["top"], box_b["top"])
    x_right = min(box_a["left"] + box_a["width"], box_b["left"] + box_b["width"])
    y_bottom = min(box_a["top"] + box_a["height"], box_b["top"] + box_b["height"])

    inter_w = max(0, x_right - x_left)
    inter_h = max(0, y_bottom - y_top)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    return inter / min(
        box_a["width"] * box_a["height"],
        box_b["width"] * box_b["height"],
    )


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
                "() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY})",
            )
            vw, vh = vp["w"], vp["h"]
            sx, sy = vp["sx"], vp["sy"]
        except Exception:
            vw = vh = 1
            sx = sy = 0  # fall back

        for v in vision_elements:
            b = v.get("bbox") or []
            fused.append(
                {
                    "label": (v.get("label") or v.get("content", "")).strip(),
                    "handle": None,
                    "bbox": b,
                    "source": "vision",
                    "left": (b[0] * vw) + sx,
                    "top": (b[1] * vh) + sy,
                    "width": (b[2] - b[0]) * vw,
                    "height": (b[3] - b[1]) * vh,
                    "fixed": False,
                },
            )
        return _finalize_and_renumber_elements(fused, sx, sy)

    # ────────────────────────────────────────────────────────────────
    # 1️⃣  Full fusion
    # ────────────────────────────────────────────────────────────────
    try:
        vp = page.evaluate(
            "() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY})",
        )
        vw, vh = vp["w"], vp["h"]
        sx, sy = vp["sx"], vp["sy"]
    except Exception:
        vw = vh = 1
        sx = sy = 0

    # Denormalise vision boxes once
    denorm_v = []
    for v in vision_elements:
        b = v.get("bbox")
        if not b:
            continue
        denorm_v.append(
            {
                **v,
                "left": (b[0] * vw) + sx,
                "top": (b[1] * vh) + sy,
                "width": (b[2] - b[0]) * vw,
                "height": (b[3] - b[1]) * vh,
            },
        )

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
        best_j = -1
        best_ov = -1.0
        for j, v in enumerate(denorm_v):
            if used_v[j]:
                continue
            ov = _overlap_ratio(h, v)
            if ov > best_ov:
                best_ov, best_j = ov, j

        if best_ov >= overlap_threshold:
            v = denorm_v[best_j]
            used_v[best_j] = True
            fused.append(
                {
                    "label": (
                        v.get("label") or v.get("content") or h.get("label", "")
                    ).strip(),
                    "handle": h.get("handle"),  # keeps clickability
                    "bbox": v["bbox"],  # vision bbox (normalised)
                    "source": "hybrid",
                    "left": v["left"],
                    "top": v["top"],
                    "width": v["width"],
                    "height": v["height"],
                    "fixed": h.get("fixed", False),
                },
            )
        else:
            fused.append({**h, "source": "heuristic"})

    # ── Pass B: any vision elements still unused go in as-is ─────────
    for j, v in enumerate(denorm_v):
        if not used_v[j]:
            fused.append(
                {
                    **v,
                    "handle": None,
                    "fixed": False,
                    "source": "vision",
                },
            )

    return _finalize_and_renumber_elements(fused, sx, sy)


def _finalize_and_renumber_elements(
    elements: list[dict],
    sx: int,
    sy: int,
) -> list[dict]:
    """Helper to apply final transformations and assign sequential IDs."""
    final_list = []
    # Sort elements by position (top-to-bottom, then left-to-right) for consistent numbering
    elements.sort(key=lambda el: (el.get("top", 0), el.get("left", 0)))

    for i, el in enumerate(elements):
        final_list.append(
            {
                "id": i + 1,  # Assign a new, sequential ID
                "label": el.get("label", ""),
                "handle": el.get("handle"),
                "bbox": el.get("bbox"),
                "fixed": el.get("fixed", False),
                "left": el.get("left"),
                "top": el.get("top"),
                "width": el.get("width"),
                "height": el.get("height"),
                "vleft": el.get("left", 0) - sx,
                "vtop": el.get("top", 0) - sy,
                "source": el.get("source", "unknown"),
            },
        )
    return final_list


# ------------------------------------------------------------------
# Stable-ID assignment between frames
# ------------------------------------------------------------------
_last_frame: list[dict] | None = None
_next_id: int = 1
_retired_ids: set[int] = set()


def _assign_stable_ids(current: list[dict]) -> list[dict]:
    """
    Assigns stable IDs to elements across frames using a multi-pass strategy.
    1. Match by ElementHandle (most reliable).
    2. Match by Bounding Box Overlap (for vision-only elements).
    3. Match by Label (least reliable fallback).
    """
    global _last_frame, _next_id, _retired_ids

    if _last_frame is None:
        for el in current:
            el["id"] = _next_id
            _next_id += 1
        _last_frame = current
        return current

    # ---- Create look-up tables from the previous frame ----
    # Elements that had a handle
    prev_by_handle = {el["handle"]: el for el in _last_frame if el.get("handle")}
    # Elements that were vision-only (no handle)
    prev_vision_only = [el for el in _last_frame if not el.get("handle")]
    # All elements from previous frame, for label matching
    prev_by_label = [(el, el.get("label", "").lower()) for el in _last_frame]

    # --- Keep track of which IDs and previous elements have been used ---
    taken_ids: set[int] = set()
    used_prev_vision_elements = [False] * len(prev_vision_only)

    # === PASS 1: Match by ElementHandle (highest priority) ===
    for new_el in current:
        if new_el.get("id"):  # Already matched
            continue
        h = new_el.get("handle")
        if h and h in prev_by_handle:
            reused_id = prev_by_handle[h]["id"]
            if reused_id not in taken_ids:
                new_el["id"] = reused_id
                taken_ids.add(reused_id)

    # === PASS 2: Match by BBox for Handle-less Elements (Vision-only) ===
    for new_el in current:
        if new_el.get("id") or new_el.get("handle"):
            continue

        best_match_idx = -1
        best_overlap = 0.5  # Use a threshold to avoid spurious matches

        for i, old_vision_el in enumerate(prev_vision_only):
            if used_prev_vision_elements[i]:
                continue

            overlap = _overlap_ratio(old_vision_el, new_el)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_idx = i

        if best_match_idx != -1:
            reused_id = prev_vision_only[best_match_idx]["id"]
            if reused_id not in taken_ids:
                new_el["id"] = reused_id
                taken_ids.add(reused_id)
                used_prev_vision_elements[best_match_idx] = True

    # === PASS 3: Match by Label (fallback) ===
    for new_el in current:
        if new_el.get("id"):
            continue

        lbl = new_el.get("label", "").lower()
        if not lbl:
            continue

        for old_el, old_lbl in prev_by_label:
            if lbl == old_lbl:
                reused_id = old_el["id"]
                if reused_id not in taken_ids:
                    new_el["id"] = reused_id
                    taken_ids.add(reused_id)
                    break

    # === PASS 4: Assign new IDs to any remaining unmatched elements ===
    for new_el in current:
        if not new_el.get("id"):
            while _next_id in taken_ids or _next_id in _retired_ids:
                _next_id += 1
            new_el["id"] = _next_id
            taken_ids.add(_next_id)

    # ---- Retire any IDs that are no longer present ----
    prev_ids = {el["id"] for el in _last_frame}
    missing = prev_ids - taken_ids
    _retired_ids.update(missing)

    _last_frame = current
    return current


def reset_stable_ids():
    """Forget everything we knew about old elements."""
    global _last_frame, _next_id, _retired_ids
    _last_frame = None
    _next_id = 1
    _retired_ids.clear()
