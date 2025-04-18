"""
Utilities that interact directly with Playwright.
"""

from math import floor
from pathlib import Path
from tempfile import mkdtemp

from js_snippets import ELEMENT_INFO_JS, UPDATE_OVERLAY_JS
from playwright.sync_api import BrowserContext, Page

MARGIN = 100  # overscan around viewport

CLICKABLE_CSS = """
button:not([disabled]):visible,
input[type=button]:not([disabled]):visible,
input[type=submit]:not([disabled]):visible,
input[type=reset]:not([disabled]):visible,
[role=button]:not([disabled]):visible,
a[href]:visible,
[onclick]:visible
"""


def collect_elements(page: Page) -> list[dict]:
    """Return info (+ handle) for all clickable elements near viewport."""
    vp = page.evaluate("() => ({w:innerWidth, h:innerHeight})")
    vL, vT = -MARGIN, -MARGIN
    vR, vB = vp["w"] + MARGIN, vp["h"] + MARGIN

    elements = []
    for handle in page.locator(CLICKABLE_CSS).element_handles():
        info = page.evaluate(ELEMENT_INFO_JS, handle)
        if not info:
            continue
        vl, vt, w, h = (
            info["vleft"],
            info["vtop"],
            info["width"],
            info["height"],
        )
        if (vl + w) < vL or vl > vR or (vt + h) < vT or vt > vB:
            continue
        info["handle"] = handle
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
    ctx = pw.chromium.launch_persistent_context(tmp_profile, headless=False)
    return ctx
