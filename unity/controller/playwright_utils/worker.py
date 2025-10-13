"""
BrowserWorker now starts *its own* Playwright instance inside the
background thread, so every Playwright call stays on the same thread.
"""

from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable, Optional
import json
import queue
import requests
from unify.utils import http
import redis
from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright
from .browser_utils import (
    build_boxes,
    collect_elements,
    launch_persistent,
    paint_overlay,
    detect_captcha,
)
from .vision_utils import (
    _fuse_elements,
    reset_stable_ids,
    _dedup,
    _assign_stable_ids,
)
from .command_runner import CommandRunner, _safe_screenshot
from .mirror import MirrorPage
from ..commands import *
from .. import captcha_solver
from .overlay import _make_js_helper
from .heuristics import export_for_js

# Manual-solve mode: set False to disable automatic CAPTCHA sniffing
AUTO_CAPTCHA = False  # Detect only when user issues `solve_captcha`
OMNIPARSER_URL = "https://omniparser.saas.unify.ai/parse/"


class BrowserWorker(threading.Thread):
    def __init__(
        self,
        *,
        start_url: str,
        refresh_interval: float = 0.5,
        log: Callable[[str], None] | None = None,
        session_connect_url: str | None = None,
        headless: bool = False,
        mode: str = "heuristic",  # "heuristic" | "vision" | "hybrid"
        debug: bool = False,
        redis_db: int = 0,
    ):
        super().__init__(daemon=True)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._redis_db = redis_db
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe(f"browser_command_{self._redis_db}")
        self.start_url = start_url
        self.log = log or (lambda *_: None)
        self._stop_event = threading.Event()
        self.session_connect_url = session_connect_url
        self.headless = headless
        self.mode = mode.lower()
        self.use_vision = self.mode in ("vision", "hybrid")
        self.use_heuristic = self.mode in ("heuristic", "hybrid")
        self.debug = debug
        self._vision_interval = 1.0  # seconds
        self.refresh_interval = (
            self._vision_interval if self.use_vision else refresh_interval
        )
        # will be initialised inside `run`
        self.runner: CommandRunner | None = None
        # keep reference to a single CAPTCHA-solving thread (optional)
        self._captcha_thread: threading.Thread | None = None
        self._captcha_q: "queue.Queue[tuple[dict,str]]" = queue.Queue()

        # --- Non-blocking vision calls ---
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._vision_future = None
        self._last_vision_ts = 0
        self._vision_elements_cache: list[dict] = []

    # ------------------------------------------------------------------
    # OmniParser service (for vison + hybrid mode)
    # ------------------------------------------------------------------
    def _call_omniparser(
        self,
        png_bytes: bytes,
        save_annotated_image: bool = False,
    ) -> list[dict]:
        """Calls the OmniParser API and returns a list of interactive elements."""
        if not png_bytes:
            self.log("Cannot call OmniParser with empty screenshot.")
            return []
        payload = {"base64_image": base64.b64encode(png_bytes).decode("utf-8")}
        try:
            response = http.post(
                OMNIPARSER_URL,
                json=payload,
                timeout=20,
                raise_for_status=False,
            )  # Increased timeout
            response.raise_for_status()
            result = response.json()
            latency = result.get("latency", "N/A")
            if latency > 3:
                self.log(
                    (
                        f"OmniParser latency: {latency:.2f}s"
                        if isinstance(latency, (int, float))
                        else f"latency={latency}"
                    ),
                )

            # Save annotated image if available
            if save_annotated_image:
                try:
                    # Decode base64 to bytes
                    original_img_bytes = base64.b64decode(payload["base64_image"])
                    annotated_img_bytes = base64.b64decode(result["som_image_base64"])
                    # Save to file with timestamp
                    timestamp = int(time.time())
                    output_dir = "annotated_images"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path_original = os.path.join(
                        output_dir,
                        f"original_omniparser_{timestamp}.png",
                    )
                    output_path_annotated = os.path.join(
                        output_dir,
                        f"annotated_omniparser_{timestamp}.png",
                    )
                    output_parsed_content = os.path.join(
                        output_dir,
                        f"parsed_content_omniparser_{timestamp}.json",
                    )
                    # Save original image
                    with open(output_path_original, "wb") as f:
                        f.write(original_img_bytes)
                    # Save annotated image
                    with open(output_path_annotated, "wb") as f:
                        f.write(annotated_img_bytes)
                    # Save parsed content list
                    with open(output_parsed_content, "w") as f:
                        json.dump(result.get("parsed_content_list", []), f, indent=2)
                except Exception as e:
                    self.log(f"Failed to save annotated image: {e}")

            # Filter for interactive elements
            return [
                e
                for e in result.get("parsed_content_list", [])
                if e.get("interactivity", False)
            ]
        except requests.exceptions.RequestException as e:
            self.log(f"OmniParser API error: {e}")
            return []

    def _get_elements_from_heuristics(self):
        """
        Original method to discover elements using JS heuristics.
        Returns a list of elements compatible with the `last_elements` format.
        """
        try:
            return collect_elements(self.runner.active)
        except Exception as exc:
            self.log(f"Heuristic element collection failed: {exc}")
            return []

    def _populate_cache(self, vision_results: list[dict]) -> None:
        """Processes vision results and populates the elements cache."""
        if not self.runner:
            return

        try:
            # Get current viewport dimensions
            vp = self.runner.active.evaluate("() => ({w:innerWidth, h:innerHeight})")
            vw, vh = vp.get("w", 1280), vp.get("h", 720)

            # Clear the cache before repopulating
            self._vision_elements_cache = []
            for i, e in enumerate(vision_results):
                bbox_norm = e.get("bbox")
                if not bbox_norm:
                    continue

                # Denormalize to absolute page coordinates
                left = bbox_norm[0] * vw
                top = bbox_norm[1] * vh
                width = (bbox_norm[2] - bbox_norm[0]) * vw
                height = (bbox_norm[3] - bbox_norm[1]) * vh

                # The 'vleft' and 'vtop' for the overlay should be relative to the viewport.
                # The paint_overlay function expects 'px' and 'py' to be the viewport-relative
                # coordinates. We should pass the absolute 'left' and 'top' to it
                # and let it handle the scroll offset. The 'build_boxes' function does this.
                self._vision_elements_cache.append(
                    {
                        "id": i + 1,
                        "label": (e.get("label") or e.get("content", "")).strip(),
                        "bbox": bbox_norm,
                        "handle": None,  # handle is resolved just-in-time
                        "fixed": False,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        # These are relative to the document, not the viewport
                        "vleft": left,
                        "vtop": top,
                    },
                )
        except Exception as e:
            self.log(f"Failed to populate vision cache: {e}")
            self._vision_elements_cache = []  # Ensure cache is empty on failure

    # ------------------------------------------------------------------ API
    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        profile_dir = Path(mkdtemp(prefix="pw_profile_"))

        with sync_playwright() as pw:
            if self.session_connect_url:
                bb_browser = pw.chromium.connect_over_cdp(self.session_connect_url)
                ctx = bb_browser.contexts[0]
            else:
                ctx = launch_persistent(
                    pw,
                    headless=self.headless,
                )  # context + first window
            # ── Inject the discovery/overlay helper into **all** future pages ──
            js_helper_src = _make_js_helper(export_for_js())
            ctx.add_init_script(js_helper_src)

            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            # The first page already exists in the persistent context, so make
            # sure the helper is immediately available before the initial
            # navigation below.  Using evaluate() is safe because the helper
            # is an IIFE (`(()=>{...})()`) that will attach itself only once.
            try:
                page.evaluate(js_helper_src)
            except Exception:
                # about:blank may not allow evaluation; ignore and rely on the
                # navigation below, which will load the helper automatically.
                pass
            page.goto(self.start_url, wait_until="domcontentloaded")

            self.runner = CommandRunner(ctx, log_fn=self.log)

            # ────────────────── Dialog & Popup event listeners (NEW) ──────────────────

            def _on_dialog(dialog):
                # store dialog in runner and update state
                self.runner._dialog = dialog
                self.runner.state.dialog_open = True
                self.runner.state.dialog_type = dialog.type
                try:
                    self.runner.state.dialog_msg = dialog.message
                except Exception:
                    self.runner.state.dialog_msg = ""

            def _on_popup(popup_page):
                # keep track of popup windows
                self.runner._popups.append(popup_page)
                self.runner.active = popup_page  # auto-focus newest popup
                self.runner._update_popups_state()
                # listen for dialogs inside popup as well
                popup_page.on("dialog", _on_dialog)

            # attach listeners to existing context & first page
            page.on("dialog", _on_dialog)
            page.on("popup", _on_popup)

            # any new top-level page (tabs or popups) – attach handlers
            def _on_new_page(new_pg):
                new_pg.on("dialog", _on_dialog)
                new_pg.on("popup", _on_popup)

            ctx.on("page", _on_new_page)

            mirror = MirrorPage(pw, page)
            last_elements: list[dict] = []

            try:
                while not self._stop_event.is_set():
                    # -- 1) drain commands & delegate to runner --
                    cmd_processed = False
                    processed_request_id: Optional[str] = None
                    while True:
                        msg = self._pubsub.get_message()
                        if msg is None:
                            break
                        if msg["type"] != "message":
                            continue

                        try:
                            # Parse the JSON payload to get action and request_id
                            payload = json.loads(msg["data"].decode())
                            cmd_str = payload["action"]
                            request_id = payload["request_id"]
                        except (json.JSONDecodeError, KeyError, AttributeError):
                            # Fallback for older/simple command strings
                            cmd_str = msg["data"].decode()
                            request_id = None

                        self.log(f"CMD ➜ {cmd_str!r} (ID: {request_id})")

                        # Handle CAPTCHA command specially since it needs thread management
                        if cmd_str == CMD_SOLVE_CAPTCHA:
                            # Manual trigger for CAPTCHA detection/solve
                            if self.runner.state.captcha_pending:
                                self.log("CAPTCHA solving already in progress")
                            else:
                                try:
                                    cap = detect_captcha(self.runner.active)
                                except Exception:
                                    cap = None
                                if cap:
                                    self.log(f"CAPTCHA detected: {cap['type']}")
                                    self.runner.state.captcha_pending = True
                                    t = threading.Thread(
                                        target=self._solve_captcha,
                                        args=(cap,),
                                        daemon=True,
                                    )
                                    t.start()
                                    self._captcha_thread = t
                                else:
                                    self.log("No CAPTCHA widgets detected on this page")
                            cmd_processed = True
                            continue

                        # Delegate execution AND recording to the CommandRunner
                        # We pass `last_elements` for context-aware actions like click
                        self.runner.run(cmd_str, last_elements, self.debug)
                        # Store the ID of the command we just processed
                        if request_id:
                            processed_request_id = request_id
                        cmd_processed = True

                    # -- 2) ensure active page is valid & reset IDs on nav ---
                    try:
                        if self.runner.active.is_closed():
                            # update popups list and fall back to first page
                            self.runner._update_popups_state()
                            if self.runner.ctx.pages:
                                self.runner.active = self.runner.ctx.pages[0]
                            else:
                                # No pages left – break out of loop to avoid spin
                                time.sleep(0.1)
                                continue

                        # Check for URL change and reset IDs *before* element collection.
                        current_url = self.runner.active.url
                        if current_url != getattr(self, "_prev_url", None):
                            reset_stable_ids()
                        self._prev_url = current_url

                    except Exception as e:
                        self.log(f"Page state check failed, resetting IDs: {e}")
                        reset_stable_ids()  # Reset state on error as a safeguard
                        if self.runner.ctx.pages:
                            self.runner.active = self.runner.ctx.pages[0]
                        else:
                            time.sleep(0.1)
                            continue

                    now = time.time()
                    vision_results = self._vision_elements_cache
                    # A) If vision is enabled, check if it's time for a new call
                    if self.use_vision and (
                        now - self._last_vision_ts >= self._vision_interval
                    ):
                        if not self._vision_future or self._vision_future.done():
                            # If a previous vision call finished, process its results first
                            if self._vision_future and self._vision_future.done():
                                try:
                                    vision_results = self._vision_future.result()
                                    vision_results.sort(
                                        key=lambda r: (r["bbox"][1], r["bbox"][0]),
                                    )
                                    self._populate_cache(vision_results)
                                except Exception as e:
                                    self.log(f"Vision call failed: {e}")
                                    self._vision_elements_cache = (
                                        []
                                    )  # Clear cache on failure
                                self._vision_future = None

                            try:
                                # Clear the overlay before taking the screenshot
                                paint_overlay(
                                    self.runner.active,
                                    [],
                                    use_vision=self.mode in ("vision", "hybrid"),
                                    need_helper=self.mode in ("heuristic", "hybrid"),
                                )
                            except Exception as e:
                                self.log(
                                    f"Could not clear overlay before screenshot: {e}",
                                )

                            # Now, trigger the next vision call
                            self._last_vision_ts = now
                            png_bytes = _safe_screenshot(self.runner.active, self.log)
                            self._vision_future = self._executor.submit(
                                self._call_omniparser,
                                png_bytes,
                                save_annotated_image=self.debug,
                            )

                    # -- 3) refresh overlay ------------------------------
                    try:
                        # C) Decide which element list to use
                        heuristic_elements = (
                            self._get_elements_from_heuristics()
                            if self.use_heuristic
                            else []
                        )
                        match self.mode:
                            case "hybrid":
                                # 2. Fuse the elements from both sources.
                                fused_elements = _fuse_elements(
                                    vision_results,
                                    heuristic_elements,
                                    self.runner.active,
                                    overlap_threshold=0.3,
                                )

                                # 3. De-duplicate the fused list to remove overlapping boxes.
                                last_elements = _dedup(
                                    fused_elements,
                                    iou_threshold=0.8,
                                )

                            case "vision":
                                fused_elements = _fuse_elements(
                                    vision_results,
                                    [],
                                    self.runner.active,
                                )
                                last_elements = _dedup(
                                    fused_elements,
                                    iou_threshold=0.95,
                                )

                            case "heuristic":
                                last_elements = [
                                    {**h, "source": "heuristic"}
                                    for h in heuristic_elements
                                ]

                        last_elements = _assign_stable_ids(last_elements)
                        last_elements.sort(
                            key=lambda e: e.get("id") or 9999,
                        )  # Sort by the new stable ID
                    except Exception as exc:  # navigation in-flight
                        self.log(f"collect_elements skipped: {exc}")
                        time.sleep(0.05)  # brief pause, then continue loop
                        continue
                    # always refresh popups list (cleanup closed)
                    if self.runner:
                        try:
                            self.runner._update_popups_state()
                        except Exception:
                            pass
                    boxes = build_boxes(last_elements)
                    # draw overlay on in the UI page only
                    for pg in (self.runner.active,):
                        try:
                            paint_overlay(
                                pg,
                                boxes,
                                use_vision=self.mode in ("vision", "hybrid"),
                                need_helper=self.mode in ("heuristic", "hybrid"),
                            )
                        except PWError as e:
                            # page or context went away – bail early
                            self.log(f"overlay skipped: {e}")
                            break

                    # ── update dynamic browser‑state fields ────────────────
                    try:
                        js = """
                            () => {
                                const url = location.href;
                                const isGDocs = url.includes("docs.google.com");
                                const el = document.activeElement;
                                const tag = el?.tagName?.toLowerCase?.() || '';
                                const role = el?.getAttribute?.('role') || '';
                                const inStandardBox = ['input','textarea'].includes(tag) ||
                                                        ['textbox','combobox','searchbox'].includes(role);

                                let inGDocsBox = false;

                                if (isGDocs) {
                                    try {
                                    const textTarget = document.querySelector('.docs-texteventtarget');
                                    const editor = document.querySelector('.kix-appview-editor');
                                    const sel = window.getSelection();

                                    // Condition 1: editor exists, and there's a selection range inside it
                                    const selValid = sel && sel.rangeCount > 0 && editor?.contains(sel.focusNode);

                                    // Condition 2: texteventtarget is focused and visible
                                    const hiddenFocused = document.activeElement === textTarget;

                                    // Condition 3: selection range exists with caret shown
                                    const caretVisible = !!document.querySelector('.kix-cursor');

                                    inGDocsBox = (selValid || hiddenFocused || caretVisible);
                                    } catch (e) {
                                    inGDocsBox = false;
                                    }
                                }

                                return {
                                    url: url,
                                    title: document.title || '',
                                    inBox: inStandardBox || inGDocsBox,
                                    sy: Math.round(scrollY)
                                };
                            }
                        """
                        res = self.runner.active.evaluate(js)
                        self.runner.state.url = res["url"]
                        self.runner.state.title = res["title"]
                        self.runner.state.in_textbox = res["inBox"]
                        self.runner.state.scroll_y = res["sy"]

                        # nav history flags (heuristic)
                        try:
                            hist_len = self.runner.active.evaluate("history.length")
                            self.runner.state.can_go_back = hist_len > 1
                            # Forward availability: we cannot query directly; keep previous flag and reset when back used
                            # Simple heuristic: assume forward not available after normal navigation; will become true after going back
                        except Exception:
                            self.runner.state.can_go_back = False
                        # forward flag relies on Playwright property 'can_go_forward'
                        try:
                            self.runner.state.can_go_forward = bool(
                                self.runner.active.evaluate(
                                    "window.__pw_forward_avail || false",
                                ),
                            )
                        except Exception:
                            self.runner.state.can_go_forward = False
                    except Exception:
                        # during navigation or cross‑origin frames
                        self.runner.state.in_textbox = False
                        # leave scroll_y unchanged (best effort)
                    # ──────────────────────────────────────────────────

                    # -- 2.3) detect & solve CAPTCHA (NEW) -----------------
                    if not self.runner.state.captcha_pending and AUTO_CAPTCHA:
                        try:
                            cap = detect_captcha(self.runner.active)
                        except Exception:
                            cap = None
                        if cap:
                            self.log(f"CAPTCHA detected: {cap['type']}")
                            self.runner.state.captcha_pending = True
                            # start solver thread only one at a time
                            t = threading.Thread(
                                target=self._solve_captcha,
                                args=(cap,),
                                daemon=True,
                            )
                            t.start()
                            self._captcha_thread = t

                    # 2.4) process any captcha token ready -----------------
                    while not self._captcha_q.empty():
                        try:
                            payload, token = self._captcha_q.get_nowait()
                        except queue.Empty:
                            break
                        self._inject_captcha(payload, token)

                    # ---------- package GUI update --------------------
                    elements_lite = [
                        (e.get("id", i + 1), e["label"], e.get("hover", False))
                        for i, e in enumerate(last_elements)
                    ]

                    # Safely get tab titles, handling navigation states
                    tab_titles = []
                    for pg in self.runner.ctx.pages:
                        try:
                            title = pg.title() or "<untitled>"
                        except Exception:
                            # Page might be navigating or context destroyed
                            title = "<loading>"
                        tab_titles.append(title)

                    screenshot_bytes = _safe_screenshot(self.runner.active, self.log)
                    screenshot = screenshot = base64.b64encode(screenshot_bytes).decode(
                        "utf-8",
                    )

                    payload = {
                        "elements": elements_lite,
                        "tabs": tab_titles,
                        "screenshot": screenshot,
                        "history": self.runner.hist.dump(),
                        "state": vars(self.runner.state),
                        "ts": time.time(),
                    }
                    # *** Add the acknowledgement ID to the payload ***
                    if processed_request_id:
                        payload["ack_request_id"] = processed_request_id
                    try:
                        self._redis_client.publish(
                            f"browser_state_{self._redis_db}",
                            json.dumps(payload),
                        )
                    except Exception:
                        pass

                    if not cmd_processed:
                        time.sleep(self.refresh_interval)

            finally:
                # Clean up Redis connections before closing browser
                try:
                    self._pubsub.close()
                    self._redis_client.close()
                except Exception:
                    pass

                mirror.close()
                ctx.close()
                shutil.rmtree(profile_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # CAPTCHA solving helper
    # ------------------------------------------------------------------

    def _solve_captcha(self, payload: dict):
        """Background thread – calls Anti-Captcha then pushes result onto queue."""
        if not self.runner:
            return

        typ = payload.get("type")
        tok: str | None = captcha_solver.solve(payload, self.runner.active.url)
        # push result (or None) back to main thread
        self._captcha_q.put((payload, tok))

    def _inject_captcha(self, payload: dict, token: str | None):
        """Execute JS injection on Playwright thread (safe)."""
        if not self.runner:
            return
        if token is None:
            self.log("CAPTCHA solve returned None")
            self.runner.state.captcha_pending = False
            return

        page = self.runner.active
        typ = payload.get("type")
        try:
            if typ == "recaptcha_v2":
                page.evaluate(
                    "(p) => {\n"
                    "  const tk = p.tk; const inv = p.inv;\n"
                    "  const setVal = () => {\n"
                    "    const ta = document.getElementById('g-recaptcha-response') ||\n"
                    "              document.querySelector('textarea[name=\\\"g-recaptcha-response\\\"]');\n"
                    "    if (ta) { ta.style.display=''; ta.value = tk; ta.dispatchEvent(new Event('input', {bubbles:true})); }\n"
                    "  };\n"
                    "  setVal();\n"
                    "  if (inv && window.grecaptcha) {\n"
                    "     try {\n"
                    "        window.grecaptcha.execute = () => tk;\n"
                    "        window.grecaptcha.getResponse = () => tk;\n"
                    "     } catch(e){}\n"
                    "  }\n"
                    "  window.dispatchEvent(new Event('captcha-solved'));\n"
                    "}",
                    {"tk": token, "inv": payload.get("invisible", False)},
                )
            elif typ == "hcaptcha":
                page.evaluate(
                    "(p) => {\n"
                    "  const tk = p.tk; const inv = p.inv;\n"
                    "  const ta = document.querySelector('textarea[name=\\\"h-captcha-response\\\"]');\n"
                    "  if (ta) { ta.style.display=''; ta.value = tk; ta.dispatchEvent(new Event('input',{bubbles:true})); }\n"
                    "  if (inv && window.hcaptcha) {\n"
                    "     try { window.hcaptcha.getResponse = () => tk; } catch(e){}\n"
                    "  }\n"
                    "  window.dispatchEvent(new Event('captcha-solved'));\n"
                    "}",
                    {"tk": token, "inv": payload.get("invisible", False)},
                )
            elif typ == "image":
                try:
                    inp = page.query_selector('input[type="text"]:visible')
                    if inp:
                        inp.fill(token)
                except Exception:
                    pass
            self.log(f"CAPTCHA token injected ({typ})")
        except Exception as exc:
            self.log(f"Error injecting CAPTCHA token: {exc}")
        finally:
            self.runner.state.captcha_pending = False
