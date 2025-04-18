"""
MirrorPage – a headless Chromium page that shadows a visible Playwright page
( URL, viewport‑size, navigation, scroll position ) and can take screenshots
without disturbing the UI.
"""

from __future__ import annotations

from playwright.sync_api import Browser, BrowserContext, Page, Playwright


class MirrorPage:
    def __init__(self, pw: Playwright, visible_page: Page):
        self.visible = visible_page
        self._browser: Browser = pw.chromium.launch(headless=True)
        self._ctx: BrowserContext | None = None
        self.page: Page | None = None
        self._init_mirror()

    # ---------------------------------------------------------------- private
    def _init_mirror(self):
        # copy cookies / storage from visible context
        storage = self.visible.context.storage_state()
        vp = self.visible.evaluate("() => ({w: innerWidth, h: innerHeight})")
        self._ctx = self._browser.new_context(
            viewport={"width": vp["w"], "height": vp["h"]},
            storage_state=storage,
        )
        self.page = self._ctx.new_page()
        self.page.goto(self.visible.url)

        # keep navigation in sync
        def handle_nav(frame):
            if frame.parent_frame is None:
                self.page.goto(frame.url)

        self.visible.on("framenavigated", handle_nav)

    # ---------------------------------------------------------------- public
    def sync_scroll(self):
        pos = self.visible.evaluate("() => ({x: scrollX, y: scrollY})")
        self.page.evaluate("([x,y]) => window.scrollTo(x,y)", [pos["x"], pos["y"]])

    def screenshot(self) -> bytes:
        self.sync_scroll()
        try:
            # wait up to 3 s for network to be idle if mirror is navigating
            self.page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass  # ignore—just try the screenshot anyway
        return self.page.screenshot(type="png", full_page=False)

    def close(self):
        self._ctx.close()
        self._browser.close()
