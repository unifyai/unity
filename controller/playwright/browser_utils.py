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
a[href]:visible,
[onclick]:visible,
*[tabindex]:not([tabindex="-1"]):visible
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
