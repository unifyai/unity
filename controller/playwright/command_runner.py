"""
Command‑parsing and dispatch logic.  All browser‑side actions live here.
"""

import re
from controller.commands import *
import urllib.parse

from controller.playwright.browser_utils import (
    build_boxes,
    collect_elements,
    paint_overlay,
)
from controller.playwright.js_snippets import HANDLE_SCROLL_JS, AUTO_SCROLL_JS
from controller.states import ActionHistory, BrowserState
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
        # ── Dialog & popup tracking (NEW) ────────────────────────────
        self._dialog = None  # playwright Dialog instance when open
        self._popups: list[Page] = []

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

    def select_tab(self, title_substr: str):
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
            self.log(f"Selected tab: {tgt.title()}")
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
        # ───────────────────── Dialog primitives ─────────────────────
        if cmd == CMD_ACCEPT_DIALOG:
            if self._dialog:
                try:
                    self._dialog.accept()
                except Exception as exc:
                    self.log(f"accept dialog failed: {exc}")
                finally:
                    self._clear_dialog_state()
            else:
                self.log("No dialog open")
            return

        if cmd == CMD_DISMISS_DIALOG:
            if self._dialog:
                try:
                    self._dialog.dismiss()
                except Exception as exc:
                    self.log(f"dismiss dialog failed: {exc}")
                finally:
                    self._clear_dialog_state()
            else:
                self.log("No dialog open")
            return

        type_prefix = CMD_TYPE_DIALOG.rstrip("*").rstrip()  # "type_dialog"
        if low.startswith(type_prefix):
            text = cmd[len(type_prefix) :].strip()
            if self._dialog and getattr(self._dialog, "type", None) == "prompt":
                try:
                    self._dialog.accept(text)
                except Exception as exc:
                    self.log(f"type dialog accept failed: {exc}")
                finally:
                    self._clear_dialog_state()
            else:
                self.log("No prompt dialog open")
            return

        # ───────────────────── Popup primitives ──────────────────────
        if cmd == CMD_CLOSE_POPUP:
            target = (
                self.active
                if self.active in self._popups
                else (self._popups[-1] if self._popups else None)
            )
            if target:
                try:
                    target.close()
                except Exception as exc:
                    self.log(f"close popup failed: {exc}")
                self._update_popups_state()
                if self.ctx.pages:
                    self.active = self.ctx.pages[0]
            else:
                self.log("No popup to close")
            return

        select_prefix = CMD_SELECT_POPUP.rstrip("*").rstrip()  # "select_popup"
        if low.startswith(select_prefix):
            needle = cmd[len(select_prefix) :].strip().lower()
            hit = next(
                (pg for pg in self._popups if needle in (pg.title() or "").lower()),
                None,
            )
            if hit:
                self.active = hit
                try:
                    hit.bring_to_front()
                except Exception:
                    pass
            else:
                self.log("No popup matches")
            return

        # open url ------------------------------------------------------
        open_prefix = CMD_OPEN_URL.rstrip("*").rstrip()  # "open_url"
        if low.startswith(open_prefix):
            url = cmd[len(open_prefix) :].strip()  # text after 1st space
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            self.active.goto(url, timeout=15000, wait_until="load")
            self.state.url = url
            self.state.title = self.active.title() or url
            return
        # ------------------------------------------------------------------
        #  smooth scroll   "scroll up 120" / "scroll down 300"
        # ------------------------------------------------------------------
        m = re.fullmatch(r"scroll_(up|down)\s+(\d+)$", low)
        if m:
            direction = m.group(1)
            pixels = int(m.group(2))
            delta = (-1 if direction == "up" else 1) * pixels

            # ---- 1.  JS attempt ------------------------------------------
            old_sy = self.active.evaluate("scrollY")
            self.active.evaluate(
                HANDLE_SCROLL_JS,
                {"delta": delta, "duration": SCROLL_DURATION},
            )

            # ---- 2.  If JS failed, send a native wheel -------------------
            new_sy = self.active.evaluate("scrollY")
            if new_sy == old_sy:
                self.active.bring_to_front()  # some sites need focus
                try:
                    self.active.mouse.wheel(0, delta)
                except Exception:
                    pass
                new_sy = self.active.evaluate("scrollY")

            # ---- 3.  Update cached state (only when it really changed) ---
            if new_sy != old_sy:
                self.state.scroll_y = new_sy
            else:
                self.log("Scroll ignored by page")

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

        # ------------------------------------------------------------------
        #  click out  – force the caret out of ANY text‑box on the page
        # ------------------------------------------------------------------
        if cmd == CMD_CLICK_OUT:
            self.hist.add(cmd)

            js_is_inbox = """
            () => {
              const el = document.activeElement;
              if (!el) return false;
              const tag  = el.tagName.toLowerCase();
              const role = el.getAttribute('role');
              return ['input','textarea'].includes(tag) ||
                     ['textbox','combobox','searchbox'].includes(role);
            }
            """

            def still_in_box() -> bool:
                try:
                    return self.active.evaluate(js_is_inbox)
                except Exception:  # cross‑origin frame focused
                    return True

            # ── 1.  try blur() directly on the element -------------------
            try:
                self.active.evaluate(
                    """() => {
                    const a = document.activeElement;
                    if (a && a.blur) a.blur(); }""",
                )
            except Exception:
                pass
            if not still_in_box():
                self.state.in_textbox = False
                return

            # ── 2.  focus the BODY element (add tabindex if needed) ------
            try:
                self.active.evaluate(
                    """() => {
                    if (!document.body.hasAttribute('tabindex'))
                        document.body.setAttribute('tabindex','-1');
                    document.body.focus({preventScroll:true});
                }""",
                )
            except Exception:
                pass
            if not still_in_box():
                self.state.in_textbox = False
                return

            # ── 3.  create + focus an off‑screen dummy button ------------
            try:
                self.active.evaluate(
                    """() => {
                    const d = Object.assign(document.createElement('button'),{
                      type:'button',
                      style:'position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;'
                    });
                    document.body.appendChild(d);
                    d.focus({preventScroll:true});
                    setTimeout(() => d.remove(), 0);
                }""",
                )
            except Exception:
                pass
            if not still_in_box():
                self.state.in_textbox = False
                return

            # ── 4.  native click near top‑left corner --------------------
            try:
                self.active.bring_to_front()
                self.active.mouse.click(5, 5, delay=10)
            except Exception:
                pass
            if not still_in_box():
                self.state.in_textbox = False
                return

            # ── 5.  final fallback – send Escape -------------------------
            try:
                self.active.keyboard.press("Escape")
            except Exception:
                pass
            self.state.in_textbox = not still_in_box()
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
        m = re.fullmatch(r"enter_text\s+(.+)", cmd, re.DOTALL)
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

        close_prefix = CMD_CLOSE_TAB.rstrip("*").rstrip()  # "close_tab"
        if low.startswith(close_prefix):
            title = cmd[len(close_prefix) :].strip()  # may be empty
            self.close_tab(title or None)
            return

        select_prefix = CMD_SELECT_TAB.rstrip("*").rstrip()  # "select_tab"
        if low.startswith(select_prefix):
            title = cmd[len(select_prefix) :].strip()
            if title:
                self.select_tab(title)
            return

    # ---------- helper for GUI refresh ------------------------------------
    def refresh_overlay(self):
        elements = collect_elements(self.active)
        paint_overlay(self.active, build_boxes(elements))
        return elements

    # ---------- internal helpers -----------------------------------------

    def _clear_dialog_state(self):
        """Reset dialog tracking and BrowserState flags."""
        self._dialog = None
        self.state.dialog_open = False
        self.state.dialog_type = None
        self.state.dialog_msg = ""

    def _update_popups_state(self):
        """Refresh cached popup titles in BrowserState."""
        try:
            alive = [pg for pg in self._popups if not pg.is_closed()]
        except Exception:
            alive = []
        self._popups = alive
        self.state.popups = [pg.title() or "(untitled)" for pg in alive]

        # If the currently active page was a popup that has since closed,
        # fall back to the first remaining page in the context (if any)
        try:
            if self.active.is_closed():
                if self.ctx.pages:
                    self.active = self.ctx.pages[0]
        except Exception:
            # self.active might already be None or detached
            if self.ctx.pages:
                self.active = self.ctx.pages[0]
