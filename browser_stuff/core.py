"""
browser/core.py
===============

Typed Playwright wrapper – v15  (2025-04-24)
-------------------------------------------
• NEW  Browser.create_new_tab(url)  → opens a fresh tab, navigates, switches focus.
• Push-notification binding (__blScanDone) moved to *context* scope – eliminates
  “Function already registered” error.
• goto() / create_new_tab() wait for full document load instead of a blind sleep.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from playwright.async_api import (
    async_playwright,
    Browser as _PWBrowser,
    BrowserContext,
    Frame,
    Page,
    Playwright,
)

from heuristics import export_for_js
from overlay import _make_js_helper


# ───────────────────────── Stealth snippet (unchanged) ─────────────────────────
_STEALTH_SNIPPET = r"""
/* (identical to earlier versions – omitted here for brevity) */
"""

# helper refresh rate must match js_helper_template.js
_HELPER_REFRESH_MS = 120


# ─────────────────────────────────── Browser ───────────────────────────────────
class Browser:
    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        user_agent: str
        | None = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        locale: str | None = "en-US",
        timezone: str | None = "Europe/Berlin",
        cookies_file: str | None = None,
    ) -> None:
        self._browser_type = browser_type
        self._ua = user_agent
        self._locale = locale
        self._tz = timezone
        self._cookies_path = Path(cookies_file) if cookies_file else None

        self._pw: Optional[Playwright] = None
        self._browser: Optional[_PWBrowser] = None
        self._context: Optional[BrowserContext] = None

        # “current” tab and list of all tabs we have touched
        self._page: Optional[Page] = None
        self._pages: List[Page] = []

        # overlay / discovery state
        self._js_helper_src = ""
        self._overlay_on = False
        self._elements: Dict[int, Dict[str, Any]] = {}
        self._map_version = -1
        self._map_ts = 0.0

        # sync helpers
        self._op_lock = asyncio.Lock()
        self._scan_event: Optional[asyncio.Event] = None
        self._auto_task: Optional[asyncio.Task[None]] = None

    # ---------------------------------------------------------------------- #
    # lifecycle                                                              #
    # ---------------------------------------------------------------------- #
    async def start(self) -> None:
        if self._pw:
            raise RuntimeError("Browser already started")

        self._pw = await async_playwright().start()
        launcher = getattr(self._pw, self._browser_type)
        self._browser = await launcher.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ],
        )

        # ── context ────────────────────────────────────────────────────────
        self._context = await self._browser.new_context(
            user_agent=self._ua,
            locale=self._locale,
            timezone_id=self._tz,
            no_viewport=True,
        )

        # cookies restore
        if self._cookies_path and self._cookies_path.exists():
            with contextlib.suppress(Exception):
                self._context.add_cookies(json.loads(self._cookies_path.read_text()))

        # init-scripts injected into **every** future page
        await self._context.add_init_script(_STEALTH_SNIPPET)
        self._js_helper_src = _make_js_helper(export_for_js())
        await self._context.add_init_script(self._js_helper_src)

        # ── push-notification binding (context-wide, only once) ────────────
        self._scan_event = asyncio.Event()

        async def _scan_done(_, __version: int) -> None:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._scan_event.set)

        await self._context.expose_binding("__blScanDone", _scan_done)

        # ── first tab ──────────────────────────────────────────────────────
        first = await self._context.new_page()
        await self._setup_page(first)
        self._page = first

        # ── future tabs ────────────────────────────────────────────────────
        self._context.on("page", lambda p: asyncio.create_task(self._setup_page(p)))

    async def close(self) -> None:
        if self._context and not self._context.is_closed():
            await self._context.close()
        if self._browser and self._browser.is_connected():
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ---------------------------------------------------------------------- #
    # page / navigation helpers                                              #
    # ---------------------------------------------------------------------- #
    async def goto(self, url: str) -> None:
        """Navigate *current* tab to *url* and wait for a full load."""
        self._require_started()
        assert self._page
        await self._page.goto(url, wait_until="load")
        await self.refresh_map()
        await self._reenable_overlay()

    async def create_new_tab(self, url: str) -> None:
        """
        Open *url* in a fresh tab, wait for it to load, switch focus, and
        refresh the element map (overlay is enabled automatically if it is
        already on in the parent tab).
        """
        self._require_started()
        assert self._context

        page = await self._context.new_page()          # triggers _setup_page via
                                                       # the context’s "page" event
        await page.goto(url, wait_until="load")

        # make this the active tab
        self._page = page

        # wait until the helper is present so refresh_map doesn’t race
        try:
            await page.wait_for_function("() => window.__blVersion !== undefined", timeout=5_000)
        except Exception:
            pass

        await self.refresh_map()
    
    async def list_tabs(self) -> List[Tuple[int, str]]:
        """
        Return a list of `(index, label)` for every browser tab we know.
        The label is the current URL (quick & synchronous).
        """
        return [(i, pg.url or f"Tab {i+1}") for i, pg in enumerate(self._pages)]

    async def activate_tab(self, index: int) -> None:
        """
        Bring the tab at *index* to the foreground and make it the active
        one for all future `click/refresh_map/...` calls.
        """
        if 0 <= index < len(self._pages):
            pg = self._pages[index]
            await pg.bring_to_front()
            self._page = pg
            await self.refresh_map()

    # ---------------------------------------------------------------------- #
    # overlay control                                                        #
    # ---------------------------------------------------------------------- #
    async def enable_overlay(self) -> None:
        self._overlay_on = True
        await self._context.add_init_script("window.__blOverlayWanted = true;")
        await self._broadcast_eval(
            "window.__blOverlayWanted = true; window.__bl?.enableOverlay();"
        )

    async def disable_overlay(self) -> None:
        self._overlay_on = False
        await self._context.add_init_script("window.__blOverlayWanted = false;")
        await self._broadcast_eval(
            "window.__blOverlayWanted = false; window.__bl?.disableOverlay();"
        )

    async def _reenable_overlay_single(self, page: Page) -> None:
        if self._overlay_on:
            await page.evaluate("window.__bl?.enableOverlay();")

    async def _reenable_overlay(self) -> None:
        if self._overlay_on:
            await self._broadcast_eval("window.__bl?.enableOverlay();")

    # ---------------------------------------------------------------------- #
    # per-page bootstrap                                                     #
    # ---------------------------------------------------------------------- #
    async def _setup_page(self, page: Page) -> None:
        """Run exactly once for every new `Page`."""
        if page in self._pages:          # idempotent
            return
        self._pages.append(page)

        # stealth patch (re-evaluate inside page world)
        with contextlib.suppress(Exception):
            await page.evaluate(_STEALTH_SNIPPET)

        # re-enable overlay after each navigation inside this tab
        page.on(
            "framenavigated",
            lambda _: asyncio.create_task(self._reenable_overlay_single(page)),
        )

        # helper script + immediate overlay if needed
        await self._broadcast_eval_page(page, self._js_helper_src)
        if self._overlay_on:
            await page.evaluate(
                "window.__blOverlayWanted = true; window.__bl?.enableOverlay();"
            )

    # ---------------------------------------------------------------------- #
    # JS broadcast helpers                                                   #
    # ---------------------------------------------------------------------- #
    async def _broadcast_eval(self, script: str) -> None:
        for pg in self._pages:
            for fr in pg.frames:
                with contextlib.suppress(Exception):
                    await fr.evaluate(script)

    async def _broadcast_eval_page(self, page: Page, script: str) -> None:
        for fr in page.frames:
            with contextlib.suppress(Exception):
                await fr.evaluate(script)

    # ---------------------------------------------------------------------- #
    # element discovery (unchanged logic)                                    #
    # ---------------------------------------------------------------------- #
    async def refresh_map(self) -> Dict[int, Dict]:
        self._require_started()
        assert self._page

        def depth(fr: Frame) -> int:
            d = 0
            while fr.parent_frame:
                fr = fr.parent_frame
                d += 1
            return d

        frames = sorted(self._page.frames, key=depth)
        combined: List[Dict] = []

        for fr in frames:
            try:
                raw = await fr.evaluate(
                    "() => window.__bl ? JSON.stringify(window.__bl.scan()) : '[]'"
                )
            except Exception:
                continue
            items: List[Dict] = json.loads(raw)

            if fr.parent_frame:
                await asyncio.gather(*(self._offset_bbox(fr, it) for it in items))
            for it in items:
                it["_frame"] = fr
            combined.extend(items)

        self._elements = {it["id"]: it for it in combined}
        self._map_version = await self._get_version()
        self._map_ts = time.monotonic()
        return self.get_elements()

    async def _offset_bbox(self, fr: Frame, it: Dict) -> None:
        if "bbox" not in it:
            return
        x = y = 0
        cur = fr
        while cur.parent_frame:
            el = await cur.frame_element()
            bb = await el.bounding_box()
            x += bb["x"]
            y += bb["y"]
            cur = cur.parent_frame
        it["bbox"]["x"] += x
        it["bbox"]["y"] += y

    def get_elements(self) -> Dict[int, Dict]:
        return {
            i: {k: v for k, v in m.items() if k != "_frame"}
            for i, m in self._elements.items()
        }

    # ---------------------------------------------------------------------- #
    # push-based wait helper                                                 #
    # ---------------------------------------------------------------------- #
    async def _wait_for_next_scan(self, timeout: float = 2.0) -> None:
        if not self._scan_event:
            return
        try:
            self._scan_event.clear()
            await asyncio.wait_for(self._scan_event.wait(), timeout)
        except asyncio.TimeoutError:
            pass

    # ---------------------------------------------------------------------- #
    # user actions (click / input / scroll)  – unchanged                     #
    # ---------------------------------------------------------------------- #
    async def click(self, element_id: int, **kw) -> None:
        async with self._op_lock:
            await self._ensure_fresh_map()
            await self._do_click(element_id, **kw)

    async def _do_click(self, element_id: int, **kw) -> None:
        self._require_started()
        assert self._page
        meta = self._elements.get(element_id)
        if not meta:
            raise KeyError(f"id {element_id} not found – refresh_map() first")

        frame: Frame = meta.get("_frame") or self._page
        sel = meta["selector"]

        await frame.wait_for_selector(sel, state="attached", timeout=5_000)
        await frame.hover(sel)
        await asyncio.sleep(0.15)
        await frame.click(sel, **kw)

        await self._wait_for_next_scan()
        await self.refresh_map()

    async def input_text(
        self, element_id: int, text: str, *, clear: bool = True, delay: float = 0.05
    ) -> None:
        async with self._op_lock:
            await self._ensure_fresh_map()
            await self._do_input_text(element_id, text, clear=clear, delay=delay)

    async def _do_input_text(
        self, element_id: int, text: str, *, clear: bool, delay: float
    ) -> None:
        self._require_started()
        assert self._page
        meta = self._elements.get(element_id)
        if not meta:
            raise KeyError(f"id {element_id} not found – refresh_map() first")

        frame: Frame = meta.get("_frame") or self._page
        sel = meta["selector"]

        await frame.wait_for_selector(sel, state="attached", timeout=5_000)
        if clear:
            await frame.fill(sel, text)
        else:
            await frame.type(sel, text, delay=delay)

        await self._wait_for_next_scan()
        await self.refresh_map()

    async def scroll_up(self, pixels: int = 400) -> None:
        await self._scroll(-abs(pixels))

    async def scroll_down(self, pixels: int = 400) -> None:
        await self._scroll(abs(pixels))

    async def _scroll(self, dy: int) -> None:
        async with self._op_lock:
            self._require_started()
            assert self._page
            await self._page.evaluate("(dy) => window.scrollBy(0, dy)", dy)
            await self._wait_for_next_scan()
            await self.refresh_map()
        # ------------------------------------------------------------------ #
    # scroll inside a specific element                                   #
    # ------------------------------------------------------------------ #
    async def scroll_element_up(self, element_id: int, pixels: int = 400) -> None:
        await self._scroll_element(element_id, -abs(pixels))

    async def scroll_element_down(self, element_id: int, pixels: int = 400) -> None:
        await self._scroll_element(element_id, abs(pixels))

    async def _scroll_element(self, element_id: int, dy: int) -> None:
        async with self._op_lock:
            await self._ensure_fresh_map()
            meta = self._elements.get(element_id)
            if not meta:
                raise KeyError(f"id {element_id} not found – refresh_map() first")

            frame: Frame = meta.get("_frame") or self._page
            sel = meta["selector"]

            await frame.evaluate(
                """([sel, dy]) => {
                       const el = document.querySelector(sel);
                       if (el) el.scrollBy(0, dy);
                   }""",
                [sel, dy],
            )

            # optional: wait for the helper to re-scan, then refresh map
            await self._wait_for_next_scan()
            await self.refresh_map()


    # ---------------------------------------------------------------------- #
    # freshness guards & helpers                                             #
    # ---------------------------------------------------------------------- #
    async def _get_version(self) -> int:
        self._require_started()
        assert self._page
        with contextlib.suppress(Exception):
            return await self._page.evaluate("() => window.__blVersion ?? 0")
        return 0

    async def _ensure_fresh_map(self, *, max_age: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._map_ts > max_age:
            await self.refresh_map()
            return
        if await self._get_version() != self._map_version:
            await self.refresh_map()

    # ---------------------------------------------------------------------- #
    # get_state / autorefresh – unchanged                                    #
    # ---------------------------------------------------------------------- #
    async def get_state(
        self, *, full_page: bool = False
    ) -> Tuple[Dict[int, Dict], bytes]:
        async with self._op_lock:
            await self._page.wait_for_load_state("load")
            await self._ensure_fresh_map()
            await self._wait_for_next_scan()

            need_toggle = False
            if not self._overlay_on:
                need_toggle = True
                await self.enable_overlay()

            png = await self._page.screenshot(full_page=full_page)

            if need_toggle:
                await self.disable_overlay()
            return self.get_elements(), png

    async def start_autorefresh(self, every: float = 1.0) -> None:
        if self._auto_task:
            return

        async def _loop() -> None:
            try:
                while True:
                    await asyncio.sleep(every)
                    async with self._op_lock:
                        await self.refresh_map()
            except asyncio.CancelledError:
                pass

        self._auto_task = asyncio.create_task(_loop())

    async def stop_autorefresh(self) -> None:
        if self._auto_task:
            self._auto_task.cancel()
            with contextlib.suppress(Exception):
                await self._auto_task
            self._auto_task = None

    # ---------------------------------------------------------------------- #
    # tiny helper                                                            #
    # ---------------------------------------------------------------------- #
    def _require_started(self) -> None:
        if not self._page:
            raise RuntimeError("Browser.start() has not been awaited yet")
