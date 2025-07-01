from playwright.sync_api import Page, ElementHandle
from typing import List, Optional

def get_probe_points(bbox_norm: List[float], vw: float, vh: float) -> List[List[float]]:
    """Calculates a list of probe points within a normalized bounding box."""
    x_min, y_min, x_max, y_max = bbox_norm

    # Convert normalized to absolute pixel coordinates
    abs_x_min, abs_x_max = x_min * vw, x_max * vw
    abs_y_min, abs_y_max = y_min * vh, y_max * vh

    center_x = (abs_x_min + abs_x_max) / 2
    center_y = (abs_y_min + abs_y_max) / 2

    # Return a list of points to probe
    return [
        [center_x, center_y],
        [abs_x_min, abs_y_min],
        [abs_x_max, abs_y_min],
        [abs_x_min, abs_y_max],
        [abs_x_max, abs_y_max],
    ]

def iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if intersection_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    iou_score = intersection_area / float(box_a_area + box_b_area - intersection_area)
    return iou_score


def match_old_element(
    old_bbox: list[float], old_label: str, new_results: list[dict], *, iou_thresh: float = 0.5
) -> dict | None:
    """Finds the element in new_results that best matches the old element's properties."""
    best_match, best_iou = None, 0.0

    # 1. Primary strategy: Find the best bounding box overlap (IoU)
    for new_element in new_results:
        overlap = iou(old_bbox, new_element["bbox"])
        if overlap > best_iou:
            best_match, best_iou = new_element, overlap
    
    if best_iou >= iou_thresh:
        return best_match

    # 2. Fallback strategy: Find a matching text label
    old_text_normalized = old_label.lower().strip()
    if old_text_normalized:
        for new_element in new_results:
            new_content_normalized = new_element.get("content", "").lower()
            if old_text_normalized in new_content_normalized:
                return new_element
    
    return None

def handle_from_bbox(page: Page, bbox_norm: List[float], expected_text: str) -> Optional[ElementHandle]:
    """
    Finds the most relevant interactive ElementHandle by probing multiple points
    within the normalized bounding box and validating against expected text content.
    """
    print("\n--- [DEBUG] Inside Multi-Point handle_from_bbox ---")
    print(f"[DEBUG] Received normalized bbox: {bbox_norm}")
    print(f"[DEBUG] Validating with expected text: '{expected_text}'")

    try:
        viewport_size = page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
        vw = viewport_size.get('width')
        vh = viewport_size.get('height')
        if not vw or not vh:
            print("[DEBUG] ERROR: Could not get viewport size.")
            return None
    except Exception as e:
        print(f"[DEBUG] ERROR: Exception while getting viewport size: {e}")
        return None

    probe_points = get_probe_points(bbox_norm, vw, vh)
    
    # NEW JAVASCRIPT LOGIC - More robust scoring system
    js_function = r"""
(args) => {
    const probePts = args.points;
    const wantTextRaw = (args.expectedText || '').trim();
    const debugLog = [];

    const norm = s => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
    const wantText = norm(wantTextRaw);
    debugLog.push(`Normalized expected text to: '${wantText}'`);

    const isSimilar = (found) => {
        if (!wantText) return false;
        const f = norm(found);
        if (!f) return false;
        if (f.includes(wantText) || wantText.includes(f)) return true;
        const levenshtein = (a, b) => {
            if (a.length === 0) return b.length; if (b.length === 0) return a.length;
            const matrix = Array(b.length + 1).fill(null).map(() => Array(a.length + 1).fill(null));
            for (let i = 0; i <= a.length; i++) { matrix[0][i] = i; }
            for (let j = 0; j <= b.length; j++) { matrix[j][0] = j; }
            for (let j = 1; j <= b.length; j++) {
                for (let i = 1; i <= a.length; i++) {
                    const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                    matrix[j][i] = Math.min(matrix[j - 1][i] + 1, matrix[j][i - 1] + 1, matrix[j - 1][i - 1] + cost);
                }
            }
            return matrix[b.length][a.length];
        };
        return levenshtein(f, wantText) <= 2;
    };

    const xs = probePts.map(p => p[0]);
    const ys = probePts.map(p => p[1]);
    const bbox = { left: Math.min(...xs), top: Math.min(...ys), right: Math.max(...xs), bottom: Math.max(...ys) };
    bbox.width = bbox.right - bbox.left;
    bbox.height = bbox.bottom - bbox.top;
    debugLog.push(`Calculated pixel bbox: {left: ${bbox.left.toFixed(1)}, top: ${bbox.top.toFixed(1)}, w: ${bbox.width.toFixed(1)}, h: ${bbox.height.toFixed(1)}}`);

    const IoU = (r1, r2) => {
        const x_a = Math.max(r1.left, r2.left);
        const y_a = Math.max(r1.top, r2.top);
        const x_b = Math.min(r1.right, r2.right);
        const y_b = Math.min(r1.bottom, r2.bottom);
        const interArea = Math.max(0, x_b - x_a) * Math.max(0, y_b - y_a);
        if (interArea === 0) return 0;
        const boxAArea = (r1.width) * (r1.height);
        const boxBArea = (r2.right - r2.left) * (r2.bottom - r2.top);
        return interArea / (boxAArea + boxBArea - interArea);
    };

    // --- LOGGING DOM CONTEXT ---
    try {
        const centerEl = document.elementFromPoint(bbox.left + bbox.width / 2, bbox.top + bbox.height / 2);
        if (centerEl && centerEl.parentElement) {
            debugLog.push(`--- DOM Context (Parent of center element) ---`);
            debugLog.push(centerEl.parentElement.outerHTML.replace(/\n\s*/g, ''));
            debugLog.push(`-------------------------------------------`);
        }
    } catch (e) {
        debugLog.push(`Could not get DOM context: ${e.message}`);
    }


    // --- NEW SCORING-BASED STRATEGY ---
    debugLog.push('--- Starting Scoring Pass ---');
    const candidates = Array.from(document.querySelectorAll('*'))
        .map(el => {
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return null;

            const iou = IoU(rect, bbox);
            if (iou < 0.1) return null; // Must have significant overlap

            const textContent = (el.innerText || '').trim();
            const textSim = isSimilar(textContent);

            let score = iou * 50; // Geometric score up to 50
            if (textSim) {
                score += 100; // Big bonus for text match
            }

            const tagName = el.tagName.toLowerCase();
            if (['a', 'button', 'input', 'summary'].includes(tagName)) {
                score += 10;
            }
            if (el.hasAttribute('onclick') || el.hasAttribute('role')) {
                score += 5;
            }
            
            return { el, score, label: textContent, iou, textSim };
        })
        .filter(c => c)
        .sort((a, b) => b.score - a.score);

    if (candidates.length > 0) {
        debugLog.push(`Found ${candidates.length} candidates. Top 3:`);
        candidates.slice(0, 3).forEach(c => {
            debugLog.push(` -> Score: ${c.score.toFixed(1)}, Label: '${c.label.substring(0, 50)}', IoU: ${c.iou.toFixed(2)}, TextMatch: ${c.textSim}, Tag: <${c.el.tagName}>`);
        });
        // Choose the best candidate if its score is reasonably high
        if (candidates[0].score > 50) {
            return { element: candidates[0].el, reason: `Best candidate from scoring pass.`, log: debugLog };
        }
        debugLog.push(`Best candidate score ${candidates[0].score.toFixed(1)} was too low.`);
    } else {
        debugLog.push('Scoring Pass Failed: No overlapping elements found.');
    }
    
    return { element: null, reason: 'All passes failed. Could not resolve element.', log: debugLog };
}
"""
    try:
        args = {"points": probe_points, "expectedText": expected_text}
        result_handle = page.evaluate_handle(js_function, args)
        result_json = result_handle.json_value()

        # Print the detailed log from the JavaScript function
        print("[DEBUG] JavaScript Execution Log:")
        if result_json and 'log' in result_json:
            for line in result_json['log']:
                print(f"  | {line}")
        else:
            print("  | No log received from JavaScript function.")


        if result_json and result_json.get('element'):
            element_handle = result_handle.get_property('element').as_element()
            if element_handle:
                outer_html = element_handle.evaluate("el => el.outerHTML")
                print(f"[DEBUG] Successfully resolved ElementHandle. Reason: {result_json.get('reason')}")
                print(f"[DEBUG] Element outerHTML: {outer_html.split('>')[0]}>")
                print("--- [DEBUG] End of handle_from_bbox ---\n")
                return element_handle
    except Exception as e:
        print(f"[DEBUG] ERROR: An exception occurred during JS evaluation: {e}")

    print(f"[DEBUG] FAILED to resolve ElementHandle. Reason: {result_json.get('reason') if result_json else 'Unknown'}")
    print("--- [DEBUG] End of handle_from_bbox (failed) ---\n")
    return None

