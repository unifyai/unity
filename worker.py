"""
BrowserWorker now starts *its own* Playwright instance inside the
background thread, so every Playwright call stays on the same thread.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable

from browser_utils import (
    build_boxes,
    collect_elements,
    launch_persistent,
    paint_overlay,
)
from commands import CommandRunner
from mirror import MirrorPage
from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright


class BrowserWorker(threading.Thread):
    def __init__(
        self,
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
        self._stop = threading.Event()

        # will be initialised inside `run`
        self.runner: CommandRunner | None = None

    # ------------------------------------------------------------------ API
    def stop(self) -> None:
        self._stop.set()

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
                while not self._stop.is_set():
                    # -- 1) drain commands --------------------------------
                    while True:
                        try:
                            cmd = self.command_q.get_nowait()
                        except queue.Empty:
                            break

                        if cmd.startswith("click button "):
                            needle = cmd[len("click button ") :].strip().lower()
                            hit = next(
                                (
                                    el
                                    for el in last_elements
                                    if needle in el["label"].lower()
                                ),
                                None,
                            )
                            if hit:
                                hit["handle"].click()
                            else:
                                self.log(f"No visible element contains “{needle}”")
                        elif cmd.startswith("click "):
                            try:
                                idx = int(cmd.split()[1])
                                if 1 <= idx <= len(last_elements):
                                    last_elements[idx - 1]["handle"].click()
                                else:
                                    self.log("Click index out of range")
                            except (ValueError, PWError) as exc:
                                self.log(f"Click failed: {exc}")
                        else:
                            self.runner.run(cmd)

                    # -- 2) refresh overlay ------------------------------
                    last_elements = collect_elements(self.runner.active)
                    boxes = build_boxes(last_elements)
                    # draw overlay both in the UI page and the headless mirror
                    paint_overlay(self.runner.active, boxes)  # visible window
                    paint_overlay(mirror.page, boxes)
                    # ---------- package GUI update --------------------
                    elements_lite = [
                        (i + 1, e["label"], e["hover"])
                        for i, e in enumerate(last_elements)
                    ]
                    tab_titles = [
                        pg.title() or "<untitled>" for pg in self.runner.ctx.pages
                    ]

                    screenshot_bytes = mirror.screenshot()

                    payload = {
                        "elements": elements_lite,
                        "tabs": tab_titles,
                        "screenshot": screenshot_bytes,
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
                profile_dir.unlink(missing_ok=True)
