"""
Utilities that interact directly with Playwright.
"""

import json
from math import floor
from pathlib import Path
from tempfile import mkdtemp
from playwright.sync_api import BrowserContext, Page

from .js_snippets import ELEMENT_INFO_JS

MARGIN = 500  # overscan around viewport (must match VP_PAD in JS helper)

# ---------------------------------------------------------------------------
# Anything that can receive a click or typing focus
# (legacy CLICKABLE_CSS removed – element discovery now handled by __bl.scan())


def collect_elements(page: Page) -> list[dict]:
    """Return interactive elements using the injected __bl.scan() helper.

    1.  Each Playwright frame is asked for `window.__bl.scan()` which returns
        lightweight JSON (id, tag, kind, label, selector).
    2.  For every hit we locate the real ElementHandle, query its geometry
        with the old `ELEMENT_INFO_JS`, then build the same enriched dict
        that existing code expects ( left/top/width/… , vleft/vtop, handle ).
    """

    vp = page.evaluate("() => ({w:innerWidth, h:innerHeight, sx:scrollX, sy:scrollY})")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp["w"] + MARGIN, vp["h"] + MARGIN
    sx, sy = vp["sx"], vp["sy"]

    elements: list[dict] = []

    # helper to compute frame depth (root first)
    def _depth(fr):
        d = 0
        while fr.parent_frame:
            fr = fr.parent_frame
            d += 1
        return d

    # Traverse frames starting with top-level – ensures parents precede children
    for fr in sorted(page.frames, key=_depth):
        try:
            raw = fr.evaluate(
                "() => window.__bl ? JSON.stringify(window.__bl.scan()) : '[]'",
            )
        except Exception:
            continue  # cross-origin, detached, etc.

        try:
            hits = json.loads(raw)
        except Exception:
            hits = []

        for hit in hits:
            sel = hit.get("selector")
            if not sel:
                continue
            try:
                h = fr.query_selector(sel)
            except Exception:
                continue
            if h is None:
                continue

            # get geometry & misc details via legacy JS snippet
            try:
                info = fr.evaluate(ELEMENT_INFO_JS, h)
            except Exception:
                continue
            if not info:
                continue

            vl = info["left"] - sx
            vt = info["top"] - sy
            w, hgt = info["width"], info["height"]

            # augment info with fields previously added downstream
            info["vleft"] = vl
            info["vtop"] = vt
            info["hover"] = info.get("hover", False)
            info["handle"] = h
            info["label"] = hit.get("label", info.get("label", ""))
            info["selector"] = sel
            info["kind"] = hit.get("kind", "other")
            info["tag"] = hit.get("tag", info.get("tag", ""))
            info["id"] = hit.get("id")

            elements.append(info)

    return elements


def build_boxes(elements: list[dict]) -> list[dict]:
    """Convert element info → box geometry for overlay JS."""
    boxes = []
    for idx, e in enumerate(elements, 1):
        boxes.append(
            dict(
                i=idx,
                fixed=e["fixed"],
                px=floor(e["vleft"]),
                py=floor(e["vtop"]),
                x=floor(e["left"]),
                y=floor(e["top"]),
                w=floor(e["width"]),
                h=floor(e["height"]),
            ),
        )
    return boxes


def paint_overlay(page: Page, boxes: list[dict], use_vision: bool = False) -> None:
    """Draw *our* bounding-box overlay for the supplied *boxes* list.

    Historically this helper was a no-op because the JS injected by
    ``js_helper_template.js`` scanned the DOM on its own and rendered an
    overlay for any elements it deemed interactive.  With the introduction
    of OmniParser-based vision results we now have many more potentially
    interactive regions that are *not* discoverable via regular DOM
    heuristics.  To surface those we render an additional lightweight
    canvas overlay on top of the page.

    The existing DOM scan overlay ( ``window.__bl.enableOverlay`` ) remains
    useful, so we keep it – our vision overlay lives in its own canvas with
    the fixed id ``vision-overlay`` to avoid clashes.
    """

    # 1) Make sure the original helper overlay (heuristic DOM scan) is ON
    if not use_vision:
        try:
            page.evaluate(
                "() => window.__bl && window.__bl.enableOverlay && window.__bl.enableOverlay()",
            )
        except Exception:
            # The page might be mid-navigation or in a sandboxed iframe – ignore.
            pass
        return

    # 2) Draw/refresh the **vision** overlay with the Python-supplied boxes
    try:
        page.evaluate(
            """
            (boxes) => {
                // Obtain (or create) a dedicated overlay canvas
                let canvas = document.getElementById('vision-overlay');
                if (!canvas) {
                    canvas = document.createElement('canvas');
                    canvas.id = 'vision-overlay';
                    canvas.style.position = 'fixed';
                    canvas.style.inset = '0';
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                    canvas.style.pointerEvents = 'none';
                    canvas.style.zIndex = '2147483647'; // stay on top
                    document.documentElement.appendChild(canvas);
                }

                // Resize the backing buffer to keep things crisp on HiDPI
                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.getBoundingClientRect();
                if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
                    canvas.width  = rect.width  * dpr;
                    canvas.height = rect.height * dpr;
                }

                const ctx = canvas.getContext('2d');
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // reset for css pixels
                ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);

                if (!Array.isArray(boxes)) return;

                // Simple cycling colour palette (matches helper overlay)
                const COLORS = ['#ff0000', '#00bfff', '#ffa500', '#7cfc00', '#ff1493'];

                boxes.forEach((b, idx) => {
                    const c = COLORS[idx % COLORS.length];
                    ctx.fillStyle   = c + '22';  // translucent fill
                    ctx.strokeStyle = c;
                    ctx.lineWidth   = 2;

                    ctx.fillRect(b.px, b.py, b.w, b.h);
                    ctx.strokeRect(b.px, b.py, b.w, b.h);

                    // Draw the index label similar to the helper overlay
                    ctx.font = 'bold 14px sans-serif';
                    ctx.lineWidth = 4; ctx.strokeStyle = '#000';
                    ctx.strokeText(String(b.i), b.px + 4, b.py + 14);
                    ctx.fillStyle = '#fff';
                    ctx.fillText(String(b.i), b.px + 4, b.py + 14);
                });
            }
            """,
            boxes,
        )
    except Exception:
        # Page might be in the middle of navigation – skip silently.
        pass


def launch_persistent(pw, headless: bool = False) -> BrowserContext:
    """Create a persistent context so new pages open as real tabs."""
    tmp_profile = Path(mkdtemp(prefix="pw_profile_"))
    ctx = pw.chromium.launch_persistent_context(
        tmp_profile,
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            # "--use-fake-ui-for-media-stream",
            # "--use-fake-device-for-media-stream",
        ],
        # permissions=["microphone", "camera"],
    )
    # ── mask navigator.webdriver in every new page ──────────────────
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});",
    )
    return ctx


# ---------------------------------------------------------------------------
# CAPTCHA detection helpers
# ---------------------------------------------------------------------------


def detect_captcha(page: Page):
    """Detect common CAPTCHA widgets on the given Playwright *page*.

    Returns a payload dict like `{"type": "recaptcha_v2", "sitekey": "..."}`
    or None when no CAPTCHA is recognised.  The function uses only DOM inspection
    and does not rely on network traffic, making it safe for cross-origin frames.
    """

    # 1) Google reCAPTCHA v2 (checkbox / invisible)
    frame = page.query_selector(
        'iframe[src*="google.com/recaptcha" i], iframe[src*="recaptcha.net" i]',
    )
    if frame:
        # Do not trigger if already solved
        try:
            solved = page.evaluate(
                "() => { try { return window.grecaptcha && window.grecaptcha.getResponse && window.grecaptcha.getResponse().length > 0 } catch(e){ return false } }",
            )
            if solved:
                frame = None  # mark as solved; skip to next detection
        except Exception:
            pass

    if frame:
        # Try data-sitekey on parent div or iframe src param
        sitekey_el = page.query_selector("[data-sitekey]")
        sitekey = sitekey_el.get_attribute("data-sitekey") if sitekey_el else None

        # Determine if widget is invisible (size=invisible or explicit JS)
        invisible = False
        try:
            if sitekey_el and sitekey_el.get_attribute("data-size") == "invisible":
                invisible = True
            else:
                src = frame.get_attribute("src") or ""
                if "invisible" in src:
                    invisible = True
        except Exception:
            pass
        if not sitekey:
            src = frame.get_attribute("src") or ""
            if "k=" in src:
                sitekey = src.split("k=")[1].split("&")[0]
        if sitekey:
            return {"type": "recaptcha_v2", "sitekey": sitekey, "invisible": invisible}

    # 2) hCaptcha
    frame = page.query_selector('iframe[src*="hcaptcha.com" i]')
    if frame:
        invisible = False
        try:
            solved = page.evaluate(
                """
                () => {
                    // 1) hidden textarea value filled?
                    const ta = document.querySelector('textarea[name="h-captcha-response"]');
                    if (ta && ta.value && ta.value.trim().length > 0) return true;
                    // 2) hcaptcha.getResponse()
                    try {
                        return window.hcaptcha && window.hcaptcha.getResponse && window.hcaptcha.getResponse().length > 0;
                    } catch(e) {}
                    return false;
                }
                """,
            )
            if solved:
                frame = None
            else:
                # detect invis size only when unsolved
                size = frame.get_attribute("data-size") or ""
                if size == "invisible":
                    invisible = True
        except Exception:
            pass

    if frame:
        sitekey = frame.get_attribute("data-sitekey") or None
        if not sitekey:
            src = frame.get_attribute("src") or ""
            if "sitekey=" in src:
                sitekey = src.split("sitekey=")[1].split("&")[0]
        if sitekey:
            return {"type": "hcaptcha", "sitekey": sitekey, "invisible": invisible}

    # 3) Fallback: image-based captcha (heuristic)
    img = page.query_selector('img[alt*="captcha" i], img[src*="captcha" i]')
    if img:
        try:
            unanswered = page.evaluate(
                """
                (node) => {
                    const nearestForm = (el) => {
                        while (el && el.tagName.toLowerCase() !== 'form') {
                            el = el.parentElement;
                        }
                        return el;
                    };

                    const qInput = (root) => root && root.querySelector && root.querySelector('input[type="text"]:not([disabled])');

                    let inp = qInput(nearestForm(node));
                    if (!inp) inp = qInput(node.parentElement || document.body);

                    if (!inp) return true; // cannot locate input – treat as unsolved
                    return (inp.value || '').trim() === '';
                }
                """,
                img,
            )
        except Exception:
            unanswered = True

        if unanswered:
            return {"type": "image", "handle": img}

    return None
