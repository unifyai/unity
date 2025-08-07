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
        self._browser: Browser = pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
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
        self._ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});",
        )
        self.page = self._ctx.new_page()
        self.page.goto(self.visible.url)

        # keep navigation in sync
        def handle_nav(frame):
            if frame.parent_frame is None:
                try:
                    url = frame.url
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                    self.page.goto(url, timeout=10000)
                except Exception:
                    pass  # ignore aborted navigations

        self.visible.on("framenavigated", handle_nav)

    # ---------------------------------------------------------------- public
    def sync_scroll(self):
        try:
            pos = self.visible.evaluate("() => ({x: scrollX, y: scrollY})")
            self.page.evaluate(
                "([x,y]) => window.scrollTo(x,y)",
                [pos["x"], pos["y"]],
            )
        except Exception:
            return

    def screenshot(self) -> bytes:
        self.sync_scroll()
        # lightweight readiness probe (optional but nice)
        try:
            state = self.page.evaluate("document.readyState")
            if state == "loading":  # still parsing / navigating
                self.page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass  # ignore & just try the screenshot
        return self.page.screenshot(type="png", full_page=False)

    def close(self):
        self._ctx.close()
        self._browser.close()
