"""
Command‑parsing and dispatch logic.  All browser‑side actions live here.
"""

import re
from unity.controller.commands import *
import urllib.parse
from playwright.sync_api import BrowserContext, Page
from base64 import b64decode

import base64
import time
from typing import Callable

from .browser_utils import (
    build_boxes,
    collect_elements,
    paint_overlay,
)
from .vision_utils import _click_at_bbox_center
from .js_snippets import HANDLE_SCROLL_JS, AUTO_SCROLL_JS
from ..states import ActionHistory, BrowserState
from playwright.sync_api import Error as PWError

SCROLL_DURATION = 400  # ms
AUTO_SCROLL_SPEED = 100 / 400  # px / ms  ≈ 250 px / s


def grab_screenshot(page: Page) -> bytes:
    """
    Capture the exact visual state of a Playwright `Page`.

    Uses the Chrome DevTools Protocol (`Page.captureScreenshot`) to grab a
    PNG of the page's painted surface—no scrolling or flicker—and returns
    the raw PNG bytes.
    """
    cdp = page.context.new_cdp_session(page)
    res = cdp.send("Page.captureScreenshot", {"fromSurface": True})
    return b64decode(res["data"])


def _safe_screenshot(page: Page, log: Callable[[str], None] | None = None) -> bytes:
    """Grab a screenshot of *page* but never raise.

    1. Tries the fast CDP-based ``grab_screenshot`` first.
    2. Falls back to Playwright's built-in ``page.screenshot`` if the CDP call
       fails (for example when the page has navigated and the previous CDP
       session was detached).
    3. If both methods fail or the page is already closed, returns an empty
       ``bytes`` object so the caller can decide how to proceed without
       crashing the worker thread.
    """

    if page is None:
        return b""

    try:
        # Fast path – CDP capture (no flicker / scroll)
        return grab_screenshot(page)
    except Exception as e:
        if log:
            log(f"_safe_screenshot: CDP capture failed – {e}")

    try:
        # Slower but more robust fallback
        return page.screenshot(type="png", full_page=False)
    except Exception as e:
        if log:
            log(f"_safe_screenshot: fallback screenshot failed – {e}")
        return b""


def _update_in_textbox_state(runner, handle, label):
    """Update BrowserState.in_textbox after a click."""
    try:
        # If not Google Docs input label, fall back to standard detection
        if "Google Docs Input" not in label:
            tag = handle.evaluate("el => el.tagName?.toLowerCase?.() || ''")
            role = handle.evaluate("el => el.getAttribute?.('role') || ''")
            runner.state.in_textbox = tag in {"input", "textarea"} or role in {
                "textbox",
                "combobox",
                "searchbox",
            }
            return

        # Google Docs aware logic
        runner.state.in_textbox = handle.evaluate(
            """
            () => {
                try {
                    const textTarget = document.querySelector('.docs-texteventtarget');
                    const editor = document.querySelector('.kix-appview-editor');
                    const sel = window.getSelection();

                    const selValid = sel && sel.rangeCount > 0 && editor?.contains(sel.focusNode);
                    const hiddenFocused = document.activeElement === textTarget;
                    const caretVisible = !!document.querySelector('.kix-cursor');

                    return selValid || hiddenFocused || caretVisible;
                } catch (e) {
                    return false;
                }
            }
        """,
        )

        handle.evaluate(
            """
            () => {
                const editor = document.querySelector('.kix-appview-editor');
                if (!editor) return;

                const sel = window.getSelection();
                const range = document.createRange();

                let node = editor;
                while (node && node.firstChild) {
                    node = node.firstChild;
                }

                if (node) {
                    range.setStart(node, node.textContent?.length || 0);
                    range.collapse(true);
                    sel.removeAllRanges();
                    sel.addRange(range);
                }
            }
        """,
        )
        runner.log(f"in_textbox (Google Docs logic): {runner.state.in_textbox}")
    except Exception as e:
        runner.state.in_textbox = False
        runner.log(f"Failed to update in_textbox: {e}")


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

    # ---------- click -----------------------------------------------------
    def click(self, element_id: int, handle):
        """
        Clicks an element using its pre-resolved handle.
        """
        if not handle or not hasattr(handle, "click"):
            self.log(
                f"Error: Invalid handle provided for click command on element ID: {element_id}",
            )
            return

        try:
            label = handle.text_content() or f"element {element_id}"
            self.log(f"Executing click on: '{label.strip()}'")
            handle.click()
            # It's good practice to wait for the page to potentially settle after a click
            self.active.wait_for_load_state("domcontentloaded", timeout=20000)
            self.log(f"Successfully clicked element {element_id}")
        except Exception as e:
            # This can happen if the page navigates or the element becomes stale
            self.log(f"An exception occurred while clicking element {element_id}: {e}")

    # ---------- command string dispatcher ---------------------------------
    def run(self, raw: str, last_elements: list = None, debug: bool = False):
        """
        Parses a raw command string, executes it while recording screenshots,
        and saves a rich record to the action history.
        """
        if last_elements is None:
            last_elements = []

        cmd = raw.strip()
        low = cmd.lower()
        if not cmd:
            return

        def execute_and_record(
            command_str: str,
            action_func: Callable,
            is_nav: bool = False,
        ):
            """A wrapper to handle screenshotting and history recording."""
            before_shot_b64 = base64.b64encode(
                _safe_screenshot(self.active, self.log),
            ).decode()

            action_func()

            # For navigation actions, wait for the page to settle before the 'after' shot
            if is_nav:
                try:
                    self.active.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
            else:
                time.sleep(0.1)

            after_shot_b64 = base64.b64encode(
                _safe_screenshot(self.active, self.log),
            ).decode()

            history_record = {
                "command": command_str,
                "timestamp": time.time(),
                "before_screenshot_b64": before_shot_b64,
                "after_screenshot_b64": after_shot_b64,
            }
            self.hist.add_record(history_record)

        # --- CLICK COMMAND ---
        if low.startswith("click "):
            try:
                parts = cmd.split()
                element_id_to_click = int(parts[1])

                # treat the token as an element *ID* (hybrid logic)
                element_to_click = next(
                    (el for el in last_elements if el["id"] == element_id_to_click),
                    None,
                )
                # fall back to "old-style" 1-based index only if that failed
                if element_to_click is None and 1 <= element_id_to_click <= len(
                    last_elements,
                ):
                    element_to_click = last_elements[element_id_to_click - 1]

                if element_to_click is None:
                    self.log(
                        f"[click] No element #{element_id_to_click} in this frame "
                        f"(max id: {max(e['id'] for e in last_elements) if last_elements else '—'})",
                    )
                    return

                handle = element_to_click.get("handle")
                label = element_to_click.get(
                    "label",
                    f"element {element_id_to_click}",
                )

                self.log(
                    f"Attempting to click: '{label}' (ID: {element_id_to_click}, Source: {element_to_click.get('source')})",
                )

                def click_action():
                    if handle:
                        # METHOD 1: Preferred, robust click via Playwright handle
                        self.log("Clicking via ElementHandle.")
                        handle.click(timeout=5000)
                        _update_in_textbox_state(self, handle, label)
                    elif element_to_click.get("bbox"):
                        # METHOD 2: Fallback for vision-only or hybrid elements without a live handle
                        self.log("Clicking via bounding box coordinates.")
                        bbox = element_to_click["bbox"]
                        _click_at_bbox_center(
                            self.active,
                            bbox,
                            debug=debug,
                        )
                    else:
                        self.log(
                            f"Click failed: Element {element_id_to_click} has no handle or bbox.",
                        )

                execute_and_record(f"click {label}", click_action, is_nav=True)

            except (ValueError, PWError, IndexError) as exc:
                self.log(
                    f"A critical error occurred during click: {exc}",
                )
            return

        # ───────────────────── Dialog primitives ─────────────────────
        if cmd == CMD_ACCEPT_DIALOG:

            def action():
                if self._dialog:
                    try:
                        self._dialog.accept()
                    except Exception as exc:
                        self.log(f"accept dialog failed: {exc}")
                    finally:
                        self._clear_dialog_state()
                else:
                    self.log("No dialog open")

            execute_and_record(cmd, action)
            return

        if cmd == CMD_DISMISS_DIALOG:

            def action():
                if self._dialog:
                    try:
                        self._dialog.dismiss()
                    except Exception as exc:
                        self.log(f"dismiss dialog failed: {exc}")
                    finally:
                        self._clear_dialog_state()
                else:
                    self.log("No dialog open")

            execute_and_record(cmd, action)
            return

        type_prefix = CMD_TYPE_DIALOG.rstrip("*").rstrip()  # "type_dialog"
        if low.startswith(type_prefix):
            text = cmd[len(type_prefix) :].strip()

            def action():
                if self._dialog and getattr(self._dialog, "type", None) == "prompt":
                    try:
                        self._dialog.accept(text)
                    except Exception as exc:
                        self.log(f"type dialog accept failed: {exc}")
                    finally:
                        self._clear_dialog_state()
                else:
                    self.log("No prompt dialog open")

            execute_and_record(cmd, action)
            return

        # ───────────────────── Popup primitives ──────────────────────
        if cmd == CMD_CLOSE_POPUP:

            def action():
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

            execute_and_record(cmd, action)
            return

        select_prefix = CMD_SELECT_POPUP.rstrip("*").rstrip()  # "select_popup"
        if low.startswith(select_prefix):
            needle = cmd[len(select_prefix) :].strip().lower()

            def action():
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

            execute_and_record(cmd, action)
            return

        # open url ------------------------------------------------------
        open_prefix = CMD_OPEN_URL.rstrip("*").rstrip()  # "open_url"
        if low.startswith(open_prefix):
            url = cmd[len(open_prefix) :].strip()  # text after 1st space
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            def action():
                self.active.goto(url, timeout=15000, wait_until="load")
                self.state.url = url
                self.state.title = self.active.title() or url

            execute_and_record(cmd, action, is_nav=True)
            return
        # ------------------------------------------------------------------
        #  smooth scroll   "scroll up 120" / "scroll down 300"
        # ------------------------------------------------------------------
        m = re.fullmatch(r"scroll_(up|down)\s+(\d+)$", low)
        if m:
            direction = m.group(1)
            pixels = int(m.group(2))
            delta = (-1 if direction == "up" else 1) * pixels

            def action():
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

            execute_and_record(cmd, action)
            return

        # auto scroll ------------------------------------------------------
        # match e.g. "start_scrolling_down 600" or plain command without speed
        m = re.fullmatch(r"start_scrolling_(up|down)(?:\s+(\d+))?", low)
        if m:
            direction = m.group(1)  # 'up' or 'down'
            # If speed (pixels per second) provided, convert to px/ms; else default
            if m.group(2):
                try:
                    px_per_s = int(m.group(2))
                except ValueError:
                    px_per_s = 250  # fallback to default
                speed = px_per_s / 1000  # JS expects pixels per millisecond
            else:
                speed = AUTO_SCROLL_SPEED
                px_per_s = int(speed * 1000)

            def action():
                self.active.evaluate(
                    AUTO_SCROLL_JS,
                    {"dir": direction, "speed": speed},
                )
                self.state.auto_scroll = direction
                self.state.scroll_speed = px_per_s

            execute_and_record(cmd, action)
            return

        if cmd == CMD_STOP_SCROLLING:

            def action():
                self.active.evaluate(AUTO_SCROLL_JS, {"dir": "stop", "speed": 0})
                self.state.auto_scroll = None

            execute_and_record(cmd, action)
            return

        # continue scroll (no‑op – lets auto‑scroll keep running)  ───── NEW
        if cmd == CMD_CONT_SCROLLING:
            execute_and_record(cmd, lambda: None)  # No-op action
            return

        # ------------------------------------------------------------------
        #  click out  – force the caret out of ANY text‑box on the page
        # ------------------------------------------------------------------
        if cmd == CMD_CLICK_OUT:

            def action():
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

            execute_and_record(cmd, action)
            return

        # ───────── keyboard shortcuts ─────────────────────────────────
        keymap = {
            CMD_PRESS_BACKSPACE: ("Backspace",),
            CMD_PRESS_DELETE: ("Delete",),
            CMD_CURSOR_LEFT: ("ArrowLeft",),
            CMD_CURSOR_RIGHT: ("ArrowRight",),
            CMD_CURSOR_UP: ("ArrowUp",),
            CMD_CURSOR_DOWN: ("ArrowDown",),
        }
        if cmd in keymap:

            def action():
                for combo in keymap[cmd]:
                    self.active.keyboard.press(combo)

            execute_and_record(cmd, action)
            return

        if cmd == CMD_HOLD_SHIFT:
            execute_and_record(cmd, lambda: self.active.keyboard.down("Shift"))
            return

        if cmd == CMD_HOLD_CTRL:
            execute_and_record(cmd, lambda: self.active.keyboard.down("Control"))
            return

        if cmd == CMD_HOLD_ALT:
            execute_and_record(cmd, lambda: self.active.keyboard.down("Alt"))
            return

        if cmd == CMD_HOLD_CMD:
            execute_and_record(cmd, lambda: self.active.keyboard.down("Meta"))
            return

        if cmd == CMD_RELEASE_SHIFT:
            execute_and_record(cmd, lambda: self.active.keyboard.up("Shift"))
            return

        if cmd == CMD_RELEASE_CTRL:
            execute_and_record(cmd, lambda: self.active.keyboard.up("Control"))
            return

        if cmd == CMD_RELEASE_ALT:
            execute_and_record(cmd, lambda: self.active.keyboard.up("Alt"))
            return

        if cmd == CMD_RELEASE_CMD:
            execute_and_record(cmd, lambda: self.active.keyboard.up("Meta"))
            return

        # press enter -------------------------------------------------------
        if cmd == CMD_PRESS_ENTER:
            execute_and_record(cmd, lambda: self.active.keyboard.press("Enter"))
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

            def action():
                parts = text.split("\n")
                for i, chunk in enumerate(parts):
                    if chunk:  # type visible chars
                        self.active.keyboard.type(chunk, delay=20)
                    if i < len(parts) - 1:  # newline → Enter
                        self.active.keyboard.press("Enter", delay=20)

            execute_and_record(f"enter text {text[:30]}…", action)
            return

        # search -----------------------------------------------------------
        m = re.fullmatch(r"search\s+(.+)", cmd)
        if m:
            execute_and_record(cmd, lambda: self.search(m.group(1)), is_nav=True)
            return

        # tab ops ----------------------------------------------------------
        if cmd == CMD_NEW_TAB:
            execute_and_record(cmd, lambda: self.new_tab())
            return

        # Close current/active tab
        if cmd == CMD_CLOSE_THIS_TAB:
            execute_and_record(cmd, lambda: self.close_tab(None))
            return

        close_prefix = CMD_CLOSE_TAB.rstrip("*").rstrip()  # "close_tab"
        if low.startswith(close_prefix):
            title = cmd[len(close_prefix) :].strip()  # may be empty
            execute_and_record(cmd, lambda: self.close_tab(title or None))
            return

        select_prefix = CMD_SELECT_TAB.rstrip("*").rstrip()  # "select_tab"
        if low.startswith(select_prefix):
            title = cmd[len(select_prefix) :].strip()
            if title:
                execute_and_record(cmd, lambda: self.select_tab(title))
            return

        # ───────────────────── Navigation (back/forward/reload) ──────── NEW
        if cmd == CMD_BACK_NAV:

            def action():
                try:
                    ret = self.active.go_back(timeout=15000, wait_until="load")
                    if ret is None:
                        self.log("No back history entry")
                    else:
                        self.state.url = self.active.url
                        self.state.title = self.active.title() or self.state.url
                        # force reload to ensure fresh state
                        try:
                            self.active.reload(timeout=15000, wait_until="load")
                        except Exception:
                            pass
                        # update navigation flags heuristically
                        try:
                            self.state.can_go_back = bool(
                                self.active.evaluate("history.length > 1"),
                            )
                        except Exception:
                            self.state.can_go_back = False
                        self.state.can_go_forward = True
                        try:
                            self.active.evaluate("window.__pw_forward_avail = true")
                        except Exception:
                            pass
                except Exception as exc:
                    self.log(f"Back navigation failed: {exc}")

            execute_and_record(cmd, action, is_nav=True)
            return

        if cmd == CMD_FORWARD_NAV:

            def action():
                try:
                    ret = self.active.go_forward(timeout=15000, wait_until="load")
                    if ret is None:
                        self.log("No forward history entry")
                    else:
                        self.state.url = self.active.url
                        self.state.title = self.active.title() or self.state.url
                        self.state.can_go_forward = False
                        try:
                            self.active.evaluate("window.__pw_forward_avail = false")
                        except Exception:
                            pass
                except Exception as exc:
                    self.log(f"Forward navigation failed: {exc}")

            execute_and_record(cmd, action, is_nav=True)
            return

        if cmd == CMD_RELOAD_PAGE:

            def action():
                try:
                    self.active.reload(timeout=15000, wait_until="load")
                    self.state.url = self.active.url
                    self.state.title = self.active.title() or self.state.url
                except Exception as exc:
                    self.log(f"Reload failed: {exc}")

            execute_and_record(cmd, action, is_nav=True)
            return

        # Generic key press: e.g. "press_key a"
        if cmd.startswith("press_key "):
            key = cmd[len("press_key ") :].strip()

            def action():
                try:
                    self.active.keyboard.press(key)
                except Exception as exc:
                    self.log(f"press_key failed: {exc}")

            execute_and_record(f"press_key {key}", action)
            return

        if cmd == CMD_SOLVE_CAPTCHA:
            # Manual trigger for CAPTCHA detection/solve
            if self.state.captcha_pending:
                # Implementation of solve_captcha method
                pass

        self.log(
            f"WARNING: Unhandled command received by CommandRunner and was dropped: {cmd!r}",
        )

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
