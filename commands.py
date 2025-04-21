"""
Command‑parsing and dispatch logic.  All browser‑side actions live here.
"""

import re
from constants import *
import urllib.parse

from browser_utils import build_boxes, collect_elements, paint_overlay
from js_snippets import HANDLE_SCROLL_JS, AUTO_SCROLL_JS
from actions import ActionHistory, BrowserState
from playwright.sync_api import BrowserContext, Page

SCROLL_DURATION = 400  # ms
AUTO_SCROLL_SPEED = 100 / 400  # px / ms  ≈ 250 px / s


class CommandRunner:
    def __init__(self, ctx: BrowserContext, log_fn):
        self.ctx = ctx
        self.log = log_fn
        self.active: Page = ctx.pages[0]
        self.state = BrowserState(url=self.active.url, title=self.active.title())
        self.hist = ActionHistory()

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
    def search(self, query: str) -> None:
        """
        Navigate the active tab to Google search results for `query`.
        """
        q = urllib.parse.quote_plus(query.strip())
        url = f"https://www.google.com/search?q={q}"
        # 30‑sec timeout, wait for full load
        self.active.goto(url, timeout=15000, wait_until="load")

    # ---------- command string dispatcher ---------------------------------
    def run(self, raw: str):
        cmd = raw.strip()
        low = cmd.lower()
        if not cmd:
            return
        self.hist.add(cmd)
        # open url ------------------------------------------------------
        m = re.fullmatch(rf"{CMD_OPEN_URL}\\s+(.+)", cmd)
        if m:
            url = m.group(1).strip()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            self.active.goto(url, timeout=15000, wait_until="load")
            self.state.url = url
            self.state.title = self.active.title() or url
            return
        # smooth scroll ----------------------------------------------------
        m = re.fullmatch(rf"{CMD_SCROLL_UP}|{CMD_SCROLL_DOWN}\s+(\d+)", cmd)
        if m:
            delta = (-1 if m.group(1) == "up" else 1) * int(m.group(2))
            self.active.evaluate(
                HANDLE_SCROLL_JS,
                {"delta": delta, "duration": SCROLL_DURATION},
            )
            self.state.scroll_y += delta
            try:
                # 1) try smooth JS scroll (works on ordinary pages)
                self.active.evaluate(
                    HANDLE_SCROLL_JS,
                    {"delta": delta, "duration": SCROLL_DURATION},
                )
            except Exception:
                pass

            # 2) issue a real wheel event – keeps working on cookie dialogs,
            #    modals, nested iframes, etc.
            try:
                self.active.mouse.wheel(0, delta)
            except Exception:
                pass

            # update cached scroll position from the page
            try:
                self.state.scroll_y = self.active.evaluate("Math.round(scrollY)")
            except Exception:
                self.state.scroll_y += delta
            return
        # auto scroll ------------------------------------------------------
        if cmd in {CMD_START_SCROLL_UP, CMD_START_SCROLL_DOWN}:
            self.active.evaluate(
                AUTO_SCROLL_JS,
                {"dir": "up" if "up" in cmd else "down", "speed": AUTO_SCROLL_SPEED},
            )
            self.state.auto_scroll = "up" if "up" in cmd else "down"
            return
        if cmd == CMD_STOP_SCROLLING:
            self.active.evaluate(AUTO_SCROLL_JS, {"dir": "stop", "speed": 0})
            self.state.auto_scroll = None
            return

        # continue scroll (no‑op – lets auto‑scroll keep running)  ───── NEW
        if cmd == CMD_CONT_SCROLLING:
            self.hist.add(cmd)
            return

        # click out (remove focus) ------------------------------------------
        if cmd == CMD_CLICK_OUT:
            # ToDo: get thsi working!
            self.hist.add(cmd)

            js = """
            () => {
              // create invisible focusable element far off‑screen
              const dummy = Object.assign(document.createElement("button"), {
                type: "button",
                style:
                  "position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;",
              });
              document.body.appendChild(dummy);
              dummy.focus({preventScroll:true});
              // remove after event loop tick so focus sticks
              setTimeout(() => dummy.remove(), 0);
              return document.activeElement === dummy;
            }
            """

            try:
                self.active.evaluate(js)
            except Exception:
                pass

            self.state.in_textbox = False
            return

        # ───────── keyboard shortcuts ─────────────────────────────────
        keymap = {
            CMD_PRESS_BACKSPACE: ("Backspace",),
            CMD_PRESS_DELETE: ("Delete",),
            CMD_CURSOR_LEFT: ("ArrowLeft",),
            CMD_CURSOR_RIGHT: ("ArrowRight",),
            CMD_CURSOR_UP: ("ArrowUp",),
            CMD_CURSOR_DOWN: ("ArrowDown",),
            CMD_SELECT_ALL: ("Control+a",),  # Cmd/⌘ on mac handled by browser
            CMD_MOVE_LINE_START: ("Control+ArrowLeft",),
            CMD_MOVE_LINE_END: ("Control+ArrowRight",),
            CMD_MOVE_WORD_LEFT: ("Alt+ArrowLeft",),
            CMD_MOVE_WORD_RIGHT: ("Alt+ArrowRight",),
        }
        if cmd in keymap:
            self.hist.add(cmd)
            for combo in keymap[cmd]:
                self.active.keyboard.press(combo)
            return

        if cmd == CMD_HOLD_SHIFT:
            self.hist.add(cmd)
            self.active.keyboard.down("Shift")
            return

        if cmd == CMD_RELEASE_SHIFT:
            self.hist.add(cmd)
            self.active.keyboard.up("Shift")
            return

        # press enter -------------------------------------------------------
        if cmd == CMD_PRESS_ENTER:
            self.hist.add(CMD_PRESS_ENTER)
            self.active.keyboard.press("Enter")
            return

        # enter text -------------------------------------------------------
        m = re.fullmatch(r"enter text\s+(.+)", cmd, re.DOTALL)
        if m:
            raw = m.group(1)
            # interpret common escapes (\n, \t, \b, etc.)
            try:
                text = bytes(raw, "utf-8").decode("unicode_escape")
            except Exception:
                text = raw
            self.hist.add(f"enter text {text[:30]}…")

            parts = text.split("\n")
            for i, chunk in enumerate(parts):
                if chunk:  # type visible chars
                    self.active.keyboard.type(chunk, delay=20)
                if i < len(parts) - 1:  # newline → Enter
                    self.active.keyboard.press("Enter", delay=20)
            return

        # search -----------------------------------------------------------
        m = re.fullmatch(r"search\s+(.+)", cmd)
        if m:
            self.search(m.group(1))
            return

        # tab ops ----------------------------------------------------------
        if cmd == CMD_NEW_TAB:
            self.new_tab()
            return
        m = re.fullmatch(r"close(?:\s+this)?\s+tab(?:\s+(.+))?", low)
        if m:
            self.close_tab(m.group(1))
            return
        m = re.fullmatch(r"select\s+to\s+tab\s+(.+)", cmd)
        if m:
            self.switch_tab(m.group(1))
            return
        self.log("Unrecognised command")

    # ---------- helper for GUI refresh ------------------------------------
    def refresh_overlay(self):
        elements = collect_elements(self.active)
        paint_overlay(self.active, build_boxes(elements))
        return elements
