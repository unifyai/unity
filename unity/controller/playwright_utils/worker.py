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
from typing import Callable
import json
import queue
import requests
import redis
from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright
from .vision_utils import handle_from_bbox, iou, match_old_element
from .browser_utils import (
    build_boxes,
    collect_elements,
    launch_persistent,
    paint_overlay,
    detect_captcha,
)
from .command_runner import CommandRunner
from .mirror import MirrorPage
from ..commands import *
from .. import captcha_solver
from .overlay import _make_js_helper
from .heuristics import export_for_js

# Manual-solve mode: set False to disable automatic CAPTCHA sniffing
AUTO_CAPTCHA = False  # NEW – detect only when user issues `solve_captcha`
OMNIPARSER_URL = "https://omniparser.saas.unify.ai/parse/"

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


class BrowserWorker(threading.Thread):
    def __init__(
        self,
        *,
        start_url: str,
        refresh_interval: float = 0.5,
        log: Callable[[str], None] | None = None,
        session_connect_url: str | None = None,
        headless: bool = False,
        use_vision: bool = True
    ):
        super().__init__(daemon=True)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("browser_command")
        self.start_url = start_url
        self.refresh_interval = refresh_interval
        self.log = log or (lambda *_: None)
        self._stop_event = threading.Event()
        self.session_connect_url = session_connect_url
        self.headless = headless
        self.use_vision = use_vision
        # will be initialised inside `run`
        self.runner: CommandRunner | None = None
        # keep reference to a single CAPTCHA-solving thread (optional)
        self._captcha_thread: threading.Thread | None = None
        self._captcha_q: "queue.Queue[tuple[dict,str]]" = queue.Queue()

        # --- Non-blocking vision calls ---
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._vision_future = None
        self._last_vision_ts = 0
        self._vision_period = 2.0  # seconds
        self._vision_elements_cache: list[dict] = []

    # ------------------------------------------------------------------
    # NEW METHOD: To call the OmniParser service
    # ------------------------------------------------------------------
    def _call_omniparser(self, png_bytes: bytes, save_annotated_image: bool = True) -> list[dict]:
        """Calls the OmniParser API and returns a list of interactive elements."""
        if not png_bytes:
            self.log("Cannot call OmniParser with empty screenshot.")
            return []
        payload = {"base64_image": base64.b64encode(png_bytes).decode("utf-8")}
        try:
            self.log("Sending screenshot to OmniParser API...")
            response = requests.post(OMNIPARSER_URL, json=payload, timeout=20)  # Increased timeout
            response.raise_for_status()
            result = response.json()
            latency = result.get('latency', 'N/A')
            self.log(f"OmniParser latency: {latency:.2f}s" if isinstance(latency,(int,float)) else f"latency={latency}")
            
            # Save annotated image if available
            if save_annotated_image:
                try:
                    # Decode base64 to bytes
                    original_img_bytes = base64.b64decode(payload["base64_image"])
                    annotated_img_bytes = base64.b64decode(result['som_image_base64'])
                    # Save to file with timestamp
                    timestamp = int(time.time())
                    output_dir = "annotated_images"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path_original = os.path.join(output_dir, f"original_omniparser_{timestamp}.png")
                    output_path_annotated = os.path.join(output_dir, f"annotated_omniparser_{timestamp}.png")
                    output_parsed_content = os.path.join(output_dir, f"parsed_content_omniparser_{timestamp}.json")
                    # Save original image
                    with open(output_path_original, 'wb') as f:
                        f.write(original_img_bytes)
                    # Save annotated image
                    with open(output_path_annotated, 'wb') as f:
                        f.write(annotated_img_bytes)
                    # Save parsed content list
                    with open(output_parsed_content, 'w') as f:
                        json.dump(result.get("parsed_content_list", []), f, indent=2)
                except Exception as e:
                    self.log(f"Failed to save annotated image: {e}")
            
            # Filter for interactive elements
            return [e for e in result.get("parsed_content_list", [])]
        except requests.exceptions.RequestException as e:
            self.log(f"OmniParser API error: {e}")
            return []

    # ------------------------------------------------------------------
    # NEW METHOD: To encapsulate the original heuristic-based logic
    # ------------------------------------------------------------------
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
            # Get current viewport dimensions and scroll position
            vp = self.runner.active.evaluate("() => ({w:innerWidth, h:innerHeight, sx:scrollX, sy:scrollY})")
            vw, vh = vp.get('w', 1280), vp.get('h', 720)
            sx, sy = vp.get('sx', 0), vp.get('sy', 0)
            
            # Clear the cache before repopulating
            self._vision_elements_cache = [] 
            for i, e in enumerate(vision_results):
                bbox_norm = e.get("bbox")
                if not bbox_norm: continue
                
                # Using the corrected denormalization logic
                left = bbox_norm[0] * vw
                top = bbox_norm[1] * vh
                width = (bbox_norm[2] - bbox_norm[0]) * vw
                height = (bbox_norm[3] - bbox_norm[1]) * vh
                vleft = left - sx
                vtop = top - sy

                self._vision_elements_cache.append({
                    "id": i + 1,
                    "label": e.get("content", "").strip(),
                    "bbox": bbox_norm,
                    "handle": None, # handle is resolved just-in-time
                    "fixed": False,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "vleft": vleft,
                    "vtop": vtop,
                })
        except Exception as e:
            self.log(f"Failed to populate vision cache: {e}")
            self._vision_elements_cache = [] # Ensure cache is empty on failure

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
                    # -- 1) drain commands --------------------------------
                    while True:
                        cmd = self._pubsub.get_message()
                        if cmd is None:
                            break
                        if cmd["type"] != "message":
                            continue
                        cmd = cmd["data"]

                        # Redis delivers raw bytes – convert to str for command parsing
                        if isinstance(cmd, (bytes, bytearray)):
                            try:
                                cmd = cmd.decode()
                            except Exception:
                                # fall back to latin-1 to prevent crash, then log & skip
                                try:
                                    cmd = cmd.decode("latin-1")
                                except Exception:
                                    self.log(f"cannot decode command payload {cmd!r}")
                                    continue

                        # show the raw command arriving from the GUI
                        self.log(f"CMD ➜ {cmd!r}")

                        if cmd.lower().startswith("click"):
                            self.log(f"--- Performing Re-Identifying Vision Click for: {cmd} ---")
                            try:
                                parts = cmd.split()
                                element_id_to_click = int(parts[1])

                                # Step 1: Get the properties of the element the user clicked, from the OLD cache.
                                if not (1 <= element_id_to_click <= len(self._vision_elements_cache)):
                                    self.log(f"Cannot click: initial element ID {element_id_to_click} is out of bounds.")
                                    continue
                                
                                prev_element = self._vision_elements_cache[element_id_to_click - 1]
                                prev_bbox = prev_element["bbox"]
                                prev_label = prev_element["label"]
                                self.log(f"Attempting to re-identify element: ID {element_id_to_click}, Label '{prev_label}'")

                                # Step 2: Get a fresh snapshot from OmniParser.
                                # It's good practice to wait for the page to be idle before screenshotting.
                                self.runner.active.wait_for_load_state("networkidle", timeout=5000)
                                png_bytes = mirror.screenshot()
                                new_results = self._call_omniparser(png_bytes)

                                # Defensive tweak: Sort results for more stable IDs on static pages.
                                new_results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

                                self._populate_cache(new_results)

                                # Step 4: Find the corresponding element in the NEW results.
                                target_element = match_old_element(prev_bbox, prev_label, self._vision_elements_cache)

                                if not target_element:
                                    self.log(f"!! Click Aborted: Could not re-identify '{prev_label}' on the page.")
                                    continue

                                # Step 5: Resolve the handle for the correctly identified target and click.
                                self.log(f"Successfully re-identified as: '{target_element.get('label')}'")
                                handle = handle_from_bbox(
                                    self.runner.active, target_element["bbox"], target_element["label"]
                                )

                                if handle:
                                    self.runner.click(target_element["id"], handle)
                                    _update_in_textbox_state(
                                        self.runner, handle, target_element.get("content", "")
                                    )
                                else:
                                    self.log(f"Click failed: Could not resolve handle for re-identified element.")

                            except Exception as exc:
                                self.log(f"A critical error occurred during re-identifying vision click: {exc}")
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
                                # First try to match by helper-assigned id
                                hit_el = next(
                                    (e for e in last_elements if e.get("id") == idx),
                                    None,
                                )
                                if hit_el:
                                    h = hit_el["handle"]
                                    label = hit_el["label"]
                                elif 1 <= idx <= len(last_elements):
                                    # fallback: legacy positional index
                                    h = last_elements[idx - 1]["handle"]
                                    label = last_elements[idx - 1]["label"]
                                else:
                                    h = label = None
                                if h:
                                    self.runner.hist.add(f"click {label}")
                                    h.click()
                                    _update_in_textbox_state(self.runner, h, label)
                                else:
                                    self.log("Click index out of range")
                            except (ValueError, PWError) as exc:
                                self.log(f"Click failed: {exc}")
                        elif cmd == CMD_CLOSE_THIS_TAB:
                            self.runner.close_tab()
                        elif cmd.startswith("close tab "):
                            self.runner.close_tab(cmd[len("close tab ") :])
                        elif cmd == CMD_SOLVE_CAPTCHA:
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
                            continue  # skip further processing
                        else:
                            self.log(
                                f"DEBUG: Command {cmd!r} not handled by worker, delegating to CommandRunner.",
                            )
                            self.runner.run(cmd)

                    # -- 2) ensure active page is valid ------------------- NEW
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
                    except Exception:
                        # handle detached/None active
                        if self.runner.ctx.pages:
                            self.runner.active = self.runner.ctx.pages[0]
                        else:
                            time.sleep(0.1)
                            continue
                    
                    now = time.time()
                     # A) If vision is enabled, check if it's time for a new call
                    if self.use_vision and (now - self._last_vision_ts >= self._vision_period):
                        if not self._vision_future or self._vision_future.done():
                            # If a previous vision call finished, process its results first
                            if self._vision_future and self._vision_future.done():
                                try:
                                    vision_results = self._vision_future.result()
                                    # Process and cache the results immediately
                                    vp = self.runner.active.evaluate("() => ({w:innerWidth, h:innerHeight, sx:scrollX, sy:scrollY})")
                                    vw, vh = vp.get('w', 1280), vp.get('h', 720)
                                    sx, sy = vp.get('sx', 0), vp.get('sy', 0)
                                    
                                    # Clear the cache before populating it
                                    self._vision_elements_cache = [] 
                                    for i, e in enumerate(vision_results):
                                        bbox_norm = e.get("bbox")
                                        if not bbox_norm: continue
                                        
                                        vp = self.runner.active.evaluate("() => ({w:innerWidth, h:innerHeight, sx:scrollX, sy:scrollY})")
                                        vw, vh = vp.get('w', 1280), vp.get('h', 720)
                                        sx, sy = vp.get('sx', 0), vp.get('sy', 0)
                                        
                                        # Corrected denormalization logic
                                        left = bbox_norm[0] * vw      # Should be x_min (index 0)
                                        top = bbox_norm[1] * vh       # Should be y_min (index 1)
                                        width = (bbox_norm[2] - bbox_norm[0]) * vw  # x_max - x_min
                                        height = (bbox_norm[3] - bbox_norm[1]) * vh # y_max - y_min
                                        vleft = left - sx
                                        vtop = top - sy

                                        self._vision_elements_cache.append({
                                            "id": i + 1,
                                            "label": e.get("content", "").strip(),
                                            "bbox": bbox_norm,
                                            "handle": None,
                                            "fixed": False,
                                            "left": left,
                                            "top": top,
                                            "width": width,
                                            "height": height,
                                            "vleft": vleft,
                                            "vtop": vtop,
                                        })
                                except Exception as e:
                                    self.log(f"Vision call failed: {e}")
                                    self._vision_elements_cache = [] # Clear cache on failure
                                self._vision_future = None

                            # Now, trigger the next vision call
                            self._last_vision_ts = now
                            png_bytes = mirror.screenshot()
                            self._vision_future = self._executor.submit(self._call_omniparser, png_bytes)

                    # -- 3) refresh overlay ------------------------------
                    try:
                        # C) Decide which element list to use
                        if self.use_vision:
                            last_elements = self._vision_elements_cache
                        else:
                            last_elements = self._get_elements_from_heuristics()
                            
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
                        "ts": time.time(),
                    }
                    try:
                        self._redis_client.publish("browser_state", json.dumps(payload))
                    except Exception:
                        pass
                    time.sleep(self.refresh_interval)

            finally:
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
