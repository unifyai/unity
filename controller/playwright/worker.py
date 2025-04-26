"""
BrowserWorker now starts *its own* Playwright instance inside the
background thread, so every Playwright call stays on the same thread.
"""

from __future__ import annotations

import queue
import base64
import threading
import time
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable

from controller.playwright.browser_utils import (
    build_boxes,
    collect_elements,
    launch_persistent,
    paint_overlay,
)
from controller.playwright.command_runner import CommandRunner
from controller.playwright.mirror import MirrorPage
from controller.commands import *
from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright


def _update_in_textbox_state(runner, handle):
    """Update BrowserState.in_textbox after a click."""
    try:
        tag = handle.evaluate("el => el.tagName.toLowerCase()")
        role = handle.evaluate("el => el.getAttribute('role')")
        runner.state.in_textbox = tag in {"input", "textarea"} or role in {
            "textbox",
            "combobox",
            "searchbox",
        }
    except Exception:
        runner.state.in_textbox = False


class BrowserWorker(threading.Thread):
    def __init__(
        self,
        *,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[list]",
        start_url: str,
        refresh_interval: float = 0.5,
        log: Callable[[str], None] | None = None,
    ):
        super().__init__(daemon=True)
        self.command_q = command_q
        self.update_q = update_q
        self.start_url = start_url
        self.refresh_interval = refresh_interval
        self.log = log or (lambda *_: None)
        self._stop_event = threading.Event()

        # will be initialised inside `run`
        self.runner: CommandRunner | None = None

    # ------------------------------------------------------------------ API
    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        profile_dir = Path(mkdtemp(prefix="pw_profile_"))

        with sync_playwright() as pw:
            ctx = launch_persistent(pw)  # context + first window
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(self.start_url, wait_until="domcontentloaded")

            self.runner = CommandRunner(ctx, log_fn=self.log)
            mirror = MirrorPage(pw, page)
            last_elements: list[dict] = []

            try:
                while not self._stop_event.is_set():
                    # -- 1) drain commands --------------------------------
                    while True:
                        try:
                            cmd = self.command_q.get_nowait()
                        except queue.Empty:
                            break

                        # show the raw command arriving from the GUI
                        self.log(f"CMD ➜ {cmd!r}")

                        if cmd.startswith("click button ") or cmd.startswith(
                            "click_button ",
                        ):
                            tail = (
                                cmd[len("click button ") :]
                                if cmd.startswith("click button ")
                                else cmd[len("click_button ") :]
                            ).strip()

                            # ---- A) try label substring match (old logic) ----
                            needle = tail.lower()
                            hit = next(
                                (
                                    el
                                    for el in last_elements
                                    if needle in el["label"].lower()
                                ),
                                None,
                            )

                            # ---- B) if not found, try numeric prefix ----------
                            if not hit:
                                prefix, *_ = tail.split("_", 1)
                                if prefix.isdigit():
                                    idx = int(prefix)
                                    if 1 <= idx <= len(last_elements):
                                        hit = last_elements[idx - 1]

                            # ---- execute the click if we resolved an element --
                            if hit:
                                friendly = f"click {hit['label']}"
                                self.runner.hist.add(friendly)
                                hit["handle"].click()
                                _update_in_textbox_state(self.runner, hit["handle"])
                            else:
                                self.log(f"No element matches “{tail}”")
                        elif cmd.startswith("open url "):
                            url = cmd[len("open url ") :]
                            self.runner.active.goto(
                                url,
                                timeout=30000,
                                wait_until="load",
                            )
                        elif cmd == "click out":
                            self.runner.run(cmd)
                        elif cmd.startswith("click "):
                            try:
                                idx = int(cmd.split()[1])
                                if 1 <= idx <= len(last_elements):
                                    h = last_elements[idx - 1]["handle"]
                                    label = last_elements[idx - 1]["label"]
                                    self.runner.hist.add(f"click {label}")
                                    h.click()
                                    _update_in_textbox_state(self.runner, h)
                                else:
                                    self.log("Click index out of range")
                            except (ValueError, PWError) as exc:
                                self.log(f"Click failed: {exc}")
                        elif cmd == CMD_CLOSE_THIS_TAB:
                            self.runner.close_tab()
                        elif cmd.startswith("close tab "):
                            self.runner.close_tab(cmd[len("close tab ") :])
                        else:
                            self.runner.run(cmd)

                    # -- 2) refresh overlay ------------------------------
                    try:
                        last_elements = collect_elements(self.runner.active)
                    except Exception as exc:  # navigation in‑flight
                        self.log(f"collect_elements skipped: {exc}")
                        time.sleep(0.05)  # brief pause, then continue loop
                        continue
                    boxes = build_boxes(last_elements)
                    # draw overlay both in the UI page and the headless mirror
                    for pg in (self.runner.active, mirror.page):
                        try:
                            paint_overlay(pg, boxes)
                        except PWError as e:
                            # page or context went away – bail early
                            self.log(f"overlay skipped: {e}")
                            break

                    # ── update dynamic browser‑state fields ────────────────
                    try:  # NEW
                        js = """
                            () => ({
                                url   : location.href,
                                title : document.title || "",
                                inBox : (() => {
                                    const el = document.activeElement;
                                    if (!el) return false;
                                    const tag  = el.tagName.toLowerCase();
                                    const role = el.getAttribute('role');
                                    return ['input','textarea'].includes(tag) ||
                                           ['textbox','combobox','searchbox'].includes(role);
                                })(),
                                sy : Math.round(scrollY)
                            })
                        """
                        res = self.runner.active.evaluate(js)
                        self.runner.state.url = res["url"]
                        self.runner.state.title = res["title"]
                        self.runner.state.in_textbox = res["inBox"]
                        self.runner.state.scroll_y = res["sy"]
                    except Exception:
                        # during navigation or cross‑origin frames
                        self.runner.state.in_textbox = False
                        # leave scroll_y unchanged (best effort)
                    # ──────────────────────────────────────────────────

                    # ---------- package GUI update --------------------
                    elements_lite = [
                        (i + 1, e["label"], e.get("hover", False))
                        for i, e in enumerate(last_elements)
                    ]
                    tab_titles = [
                        pg.title() or "<untitled>" for pg in self.runner.ctx.pages
                    ]

                    screenshot_bytes = mirror.screenshot()
                    screenshot = screenshot = base64.b64encode(screenshot_bytes).decode(
                        "utf-8",
                    )

                    payload = {
                        "elements": elements_lite,
                        "tabs": tab_titles,
                        "screenshot": screenshot,
                        "history": self.runner.hist.dump(),
                        "state": vars(self.runner.state),
                    }
                    while True:  # keep only the latest payload
                        try:
                            self.update_q.put_nowait(payload)
                            break
                        except queue.Full:
                            try:
                                self.update_q.get_nowait()  # discard oldest
                            except queue.Empty:
                                pass
                    time.sleep(self.refresh_interval)

            finally:
                mirror.close()
                ctx.close()
                shutil.rmtree(profile_dir, ignore_errors=True)
