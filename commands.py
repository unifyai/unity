"""
Command‑parsing and dispatch logic.  All browser‑side actions live here.
"""

import re

from browser_utils import build_boxes, collect_elements, paint_overlay
from js_snippets import AUTO_SCROLL_JS, HANDLE_SCROLL_JS
from playwright.sync_api import BrowserContext, Page

SCROLL_DURATION = 400  # ms
AUTO_SCROLL_SPEED = 100 / 400  # px / ms  ≈ 250 px / s


class CommandRunner:
    def __init__(self, ctx: BrowserContext, log_fn):
        self.ctx = ctx
        self.active: Page = ctx.pages[0]
        self.log = log_fn

    # ---------- high‑level API (called by GUI) ----------------------------
    def new_tab(self):
        self.active = self.ctx.new_page()
        self.log("New tab opened")

    def close_tab(self, title_substr: str | None):
        if title_substr:
            tgt = next(
                (
                    pg
                    for pg in self.ctx.pages
                    if title_substr.lower() in (pg.title() or "").lower()
                ),
                None,
            )
        else:
            tgt = self.active

        if not tgt:
            self.log("No tab matches")
            return
        tgt.close()
        self.log("Tab closed")
        if self.ctx.pages:
            self.active = self.ctx.pages[0]

    def switch_tab(self, title_substr: str):
        tgt = next(
            (
                pg
                for pg in self.ctx.pages
                if title_substr.lower() in (pg.title() or "").lower()
            ),
            None,
        )
        if tgt:
            self.active = tgt
            tgt.bring_to_front()
            self.log(f"Switched to tab: {tgt.title()}")
        else:
            self.log("No tab matches")

    # ---------- search -----------------------------------------------------
    def search(self, query: str):
        """
        Type `query` into the first visible search/text input and press Enter.
        Falls back to raw keyboard typing if no eligible input is found.
        """
        try:
            box = (
                self.active.locator('input[type="search"], input[type="text"]')
                .filter(has_text="", has_not="[disabled]")
                .first
            )
            box.wait_for(state="visible", timeout=1000)
            box.fill(query)
            box.press("Enter")
            self.log(f"Searched for {query!r}")
        except Exception:
            # coarse fallback – focus the page and type
            self.active.keyboard.type(query)
            self.active.keyboard.press("Enter")
            self.log(f"Searched (fallback) for {query!r}")

    # ---------- command string dispatcher ---------------------------------
    def run(self, raw: str):
        cmd = raw.strip().lower()
        if not cmd:
            return
        # smooth scroll ----------------------------------------------------
        m = re.fullmatch(r"scroll\s+(up|down)\s+(\d+)", cmd)
        if m:
            delta = (-1 if m.group(1) == "up" else 1) * int(m.group(2))
            self.active.evaluate(
                HANDLE_SCROLL_JS,
                {"delta": delta, "duration": SCROLL_DURATION},
            )
            return
        # auto scroll ------------------------------------------------------
        if cmd in {"start scroll up", "start scroll down"}:
            self.active.evaluate(
                AUTO_SCROLL_JS,
                {"dir": "up" if "up" in cmd else "down", "speed": AUTO_SCROLL_SPEED},
            )
            return
        if cmd == "stop scroll":
            self.active.evaluate(AUTO_SCROLL_JS, {"dir": "stop", "speed": 0})
            return

        # search -----------------------------------------------------------
        m = re.fullmatch(r"search\s+(.+)", cmd)
        if m:
            self.search(m.group(1))
            return

        # tab ops ----------------------------------------------------------
        if cmd == "new tab":
            self.new_tab()
            return
        m = re.fullmatch(r"close\s+tab(?:\s+(.+))?", cmd)
        if m:
            self.close_tab(m.group(1))
            return
        m = re.fullmatch(r"switch\s+to\s+tab\s+(.+)", cmd)
        if m:
            self.switch_tab(m.group(1))
            return
        self.log("Unrecognised command")

    # ---------- helper for GUI refresh ------------------------------------
    def refresh_overlay(self):
        elements = collect_elements(self.active)
        paint_overlay(self.active, build_boxes(elements))
        return elements
