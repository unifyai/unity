from playwright.sync_api import Page, ElementHandle
from typing import List, Optional

def get_probe_points(bbox_norm: List[float], vw: float, vh: float) -> List[List[float]]:
    """Calculates a list of probe points within a normalized bounding box."""
    x_min, y_min, x_max, y_max = bbox_norm

    # Convert normalized to absolute pixel coordinates
    abs_x_min, abs_x_max = x_min * vw, x_max * vw  # left, right
    abs_y_min, abs_y_max = y_min * vh, y_max * vh  # top,  bottom

    center_x = (abs_x_min + abs_x_max) / 2
    center_y = (abs_y_min + abs_y_max) / 2

    # Return a list of points to probe: center, corners, and midpoints
    return [
        [center_x, center_y],           # • (Center is most important, probe first)
        [abs_x_min, abs_y_min],          # ↖︎
        [abs_x_max, abs_y_min],          # ↗︎
        [abs_x_min, abs_y_max],          # ↙︎
        [abs_x_max, abs_y_max],          # ↘︎
        [center_x, abs_y_min],           # ↑
        [center_x, abs_y_max],           # ↓
        [abs_x_min, center_y],           # ←
        [abs_x_max, center_y],           # →
    ]

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
    print(f"[DEBUG] Generated {len(probe_points)} probe points.")

    # --- UPDATED JAVASCRIPT LOGIC ---
    js_function = r"""
(args) => {
    /* ---------------  inputs  --------------- */
    const probePts     = args.points;                       // [[x,y]…]  viewport px
    const wantTextRaw  = (args.expectedText || '').trim();

    /* ---------------  helpers  --------------- */
    const norm = s => (s || '')
        .replace(/\s+/g, ' ')                // collapse whitespace / newlines
        .trim()
        .toLowerCase();
    const wantText = norm(wantTextRaw);

    const levenshtein = (a,b) => {
        const m=a.length, n=b.length;
        if(!m) return n; if(!n) return m;
        const v0 = Array.from({length:n+1},(_,i)=>i);
        const v1 = new Array(n+1);
        for(let i=0;i<m;i++){
            v1[0]=i+1;
            for(let j=0;j<n;j++){
                const cost=a[i]===b[j]?0:1;
                v1[j+1]=Math.min(v1[j]+1,v0[j+1]+1,v0[j]+cost);
            }
            v0.splice(0,n+1,...v1);
        }
        return v1[n];
    };
    const isSimilar = found => {
        const f = norm(found);
        if(!wantText)          return !!f;           // any non-empty text is ok
        if(!f)                 return false;
        if(f.includes(wantText) || wantText.includes(f)) return true;
        return levenshtein(f,wantText) <= 2;
    };

    const interactiveSel = [
        'a','button','input','select','textarea','summary',
        '[role=button]','[role=link]','[onclick]','[data-action]',
        '[style*="cursor:pointer"]'
    ].join(',');

    /* ----------  geometry ---------- */
    const xs = probePts.map(p=>p[0]), ys = probePts.map(p=>p[1]);
    const bbox = {
        left  : Math.min(...xs),
        top   : Math.min(...ys),
        right : Math.max(...xs),
        bottom: Math.max(...ys)
    };
    bbox.width  = bbox.right - bbox.left;
    bbox.height = bbox.bottom - bbox.top;

    const IoU = (r1,r2)=>{
        const l=Math.max(r1.left ,r2.left ),
              t=Math.max(r1.top  ,r2.top  ),
              r=Math.min(r1.right,r2.right),
              b=Math.min(r1.bottom,r2.bottom);
        if(r<=l||b<=t) return 0;
        const inter=(r-l)*(b-t),
              union=r1.width*r1.height + r2.width*r2.height - inter;
        return inter/union;
    };
    const centreDist = (r) => {
        const cx = (r.left+r.right)/2, cy = (r.top+r.bottom)/2;
        const bx = (bbox.left+bbox.right)/2, by = (bbox.top+bbox.bottom)/2;
        return Math.hypot(cx-bx, cy-by);            // px
    };

    /* ----------  PASS-0 : text match + best score  ---------- */
    {
        const cands = Array.from(document.querySelectorAll(interactiveSel))
            .map(el => ({
                el,
                rect : el.getBoundingClientRect(),
                label: (el.innerText || el.getAttribute('aria-label') || '').trim()
            }))
            .filter(o => isSimilar(o.label))                     // text filter
            .map(o => ({...o,
                       iou : IoU(o.rect, bbox),
                       dist: centreDist(o.rect)}))
            .filter(o => o.iou > 0 || o.dist < 80)               // quick cut
            .sort((a,b) => (b.iou - a.iou) || (a.dist - b.dist));// score

        if(cands.length)
            return {element: cands[0].el,
                    reason : `IoU ${cands[0].iou.toFixed(2)}, “${cands[0].label}”`};
    }

    /* ----------  PASS-1 : probe points (bottom-up stack) ---------- */
    for(const [px,py] of probePts){
        const stack = [...document.elementsFromPoint(px,py)].reverse(); // deepest → top
        for(const el of stack){
            const cand = el.matches(interactiveSel) ? el : el.closest(interactiveSel);
            if(!cand) continue;
            const rect = cand.getBoundingClientRect();
            if(IoU(rect,bbox) < 0.1) continue;
            const label = (cand.innerText || cand.getAttribute('aria-label') || '').trim();
            if(isSimilar(label))
                return {element:cand,
                        reason:`Probe @(${px.toFixed(1)},${py.toFixed(1)}) → “${label}”`};
        }
    }

    /* ----------  PASS-2 : first interactive with big IoU **AND** text match ---------- */
    if(wantText){
        const cand = Array.from(document.querySelectorAll(interactiveSel))
                  .find(el => IoU(el.getBoundingClientRect(),bbox) > 0.5 &&
                               isSimilar(el.innerText || el.getAttribute('aria-label') || ''));
        if(cand)
            return {element:cand,reason:'Big IoU + text match'};
    }

    /* ----------  PASS-3 : absolute fallback ---------- */
    const [cx,cy]=[(bbox.left+bbox.right)/2,(bbox.top+bbox.bottom)/2];
    const fallback = document.elementsFromPoint(cx,cy)[0];
    return {element:fallback,reason:'Top-most at centre (no text match found)'};
}
"""
    try:
        # Pass both points and expected_text as a single argument object
        args = {"points": probe_points, "expectedText": expected_text}
        result_handle = page.evaluate_handle(js_function, args)
        result_json = result_handle.json_value()
        
        print(f"[DEBUG] Raw JS result: {result_json}")

        if result_json and result_json.get('element'):
            element_handle = result_handle.get_property('element').as_element()
            if element_handle:
                outer_html = element_handle.evaluate("el => el.outerHTML")
                print(f"[DEBUG] Successfully resolved ElementHandle.")
                print(f"[DEBUG] Element outerHTML: {outer_html}")
                print("--- [DEBUG] End of handle_from_bbox ---\n")
                return element_handle
    except Exception as e:
        print(f"[DEBUG] ERROR: An exception occurred during JS evaluation: {e}")

    print("--- [DEBUG] End of handle_from_bbox (failed) ---\n")
    return None