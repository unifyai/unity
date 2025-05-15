"""
Utilities that interact directly with Playwright.
"""

from math import floor
from pathlib import Path
from tempfile import mkdtemp

from controller.playwright.js_snippets import ELEMENT_INFO_JS, UPDATE_OVERLAY_JS
from playwright.sync_api import BrowserContext, Page

MARGIN = 100  # overscan around viewport

# ---------------------------------------------------------------------------
# Anything that can receive a click or typing focus
CLICKABLE_CSS = """
button:not([disabled]):visible,
input:not([disabled]):not([type=hidden]):visible,
textarea:not([disabled]):visible,
[role=button]:visible,
[role=link]:visible,
[role=textbox]:visible,
[role=combobox]:visible,
[role=searchbox]:visible,
[role=option]:visible,
[role=listitem]:visible,
a[href]:visible,
[onclick]:visible,
*[tabindex]:not([tabindex="-1"]):visible,
[role=switch]:visible,
[role=checkbox]:visible,
[role=radio]:visible,
input[type=submit]:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=image]:not([disabled]):visible,
select:not([disabled]):visible,
label[for]:visible
"""


def collect_elements(page: Page) -> list[dict]:
    vp = page.evaluate("() => ({w:innerWidth, h:innerHeight, sx:scrollX, sy:scrollY})")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp["w"] + MARGIN, vp["h"] + MARGIN
    sx, sy = vp["sx"], vp["sy"]

    elements = []

    for frame in page.frames:
        for h in frame.locator(CLICKABLE_CSS).element_handles():
            try:
                info = frame.evaluate(ELEMENT_INFO_JS, h)
            except Exception:  # cross‑origin or stale element etc.
                continue
            if not info:
                continue

            vl = info["left"] - sx
            vt = info["top"] - sy
            w, hgt = info["width"], info["height"]

            if (vl + w) < vL or vl > vR or (vt + hgt) < vT or vt > vB:
                continue

            info["vleft"] = vl
            info["vtop"] = vt
            info["hover"] = info.get("hover", False)
            info["handle"] = h
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


def paint_overlay(page: Page, boxes: list[dict]) -> None:
    page.evaluate(UPDATE_OVERLAY_JS, boxes)


def launch_persistent(pw) -> BrowserContext:
    """Create a persistent context so new pages open as real tabs."""
    tmp_profile = Path(mkdtemp(prefix="pw_profile_"))
    ctx = pw.chromium.launch_persistent_context(
        tmp_profile,
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
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
    frame = page.query_selector('iframe[src*="google.com/recaptcha" i], iframe[src*="recaptcha.net" i]')
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
        sitekey_el = page.query_selector('[data-sitekey]')
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
        try:
            solved = page.evaluate(
                "() => { try { return window.hcaptcha && window.hcaptcha.getResponse && window.hcaptcha.getResponse().length > 0 } catch(e){ return false } }",
            )
            if solved:
                frame = None
        except Exception:
            pass

    if frame:
        sitekey = frame.get_attribute("data-sitekey") or None
        if not sitekey:
            src = frame.get_attribute("src") or ""
            if "sitekey=" in src:
                sitekey = src.split("sitekey=")[1].split("&")[0]
        if sitekey:
            return {"type": "hcaptcha", "sitekey": sitekey}

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
