"""
Tk‑based front‑end.

• All Playwright work is done in BrowserWorker (background thread)
• This file now accepts *arbitrary English* in the command bar, sends it to
  o3‑mini (OpenAI) via `agent.primitive_to_browser_action`, converts the structured
  result into the low‑level command strings understood by BrowserWorker,
  and shows everything in the log window.
"""

from __future__ import annotations

import queue
import tkinter as tk
import traceback
from tkinter import scrolledtext, ttk
from typing import Any

from agent import primitive_to_browser_action
from helpers import _slug
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


class ControlPanel(tk.Tk):
    """Main GUI window.  No Playwright calls occur on this thread."""

    REFRESH_INTERVAL_MS = 100  # how often we poll the update queue

    def __init__(
        self,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[list[tuple[int, str, bool]]]",
        text_q: queue.Queue[str],
    ):
        super().__init__()
        self.cmd_q = command_q  # GUI → worker
        self.up_q = update_q  # worker → GUI
        self.text_q = text_q  # requests -> worker
        self.elements: list[tuple[int, str, bool]] = []
        self.screenshot: bytes = b""
        self.tab_titles: list[str] = []

        self._build_widgets()
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

        self.bind(
            "<<SendTextCommand>>",
            lambda _e: self._handle_input(self._pending_text),
        )
        self._pending_text = ""

        self.after(50, self._poll_text_q)

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        self.title("Playwright helper")
        self.geometry("900x550")

        # row‑layout:
        #   0  main split (listbox + preview)
        #   1  search / open‑url bar
        #   2  command bar
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        # -------- element list ------------------------------------------
        self.listbox = tk.Listbox(self, font=("Helvetica", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=0, sticky="nse")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<Double-1>", self._on_list_click)
        self.listbox.bind("<Return>", self._on_list_click)

        # -------- right panel (log + buttons) ---------------------------
        right = tk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)

        self.log = scrolledtext.ScrolledText(right, state="disabled", height=8)
        self.log.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        btns = tk.Frame(right)
        btns.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        for i in range(2):
            btns.columnconfigure(i, weight=1)

        def make(r: int, c: int, txt: str, cmd: str) -> None:
            ttk.Button(
                btns,
                text=txt,
                command=lambda: self._handle_input(cmd),
            ).grid(row=r, column=c, sticky="ew")

        make(0, 0, "▲ Scroll 100", "scroll up 100")
        make(0, 1, "▼ Scroll 100", "scroll down 100")
        make(1, 0, "Start ▲", "start scroll up")
        make(1, 1, "Start ▼", "start scroll down")
        make(2, 0, "Stop scroll", "stop scroll")
        make(3, 0, "New tab", "new tab")
        make(3, 1, "Close tab", "close tab")

        # ---- exit button ------------------------------------------------
        ttk.Button(
            btns,
            text="Exit",
            command=self._on_exit,  # ← defined below
            style="Danger.TButton",  # optional: red‑looking style
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # -------- search / open‑url bar ---------------------------------
        self.search_var = tk.StringVar()
        self.search_mode = tk.StringVar(value="google")  # ← default
        search = tk.Frame(self)
        search.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 4))
        search.columnconfigure(1, weight=1)

        tk.Label(search, text="Search / URL:").grid(row=0, column=0, sticky="w")

        entry = tk.Entry(search, textvariable=self.search_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _e: self._send_search())

        # radio buttons for mode
        rb_google = tk.Radiobutton(
            search,
            text="Google",
            variable=self.search_mode,
            value="google",
        )
        rb_url = tk.Radiobutton(
            search,
            text="URL",
            variable=self.search_mode,
            value="url",
        )
        rb_google.grid(row=0, column=2, padx=(6, 0))
        rb_url.grid(row=0, column=3)
        rb_google.select()

        ttk.Button(search, text="Go", command=self._send_search).grid(
            row=0,
            column=4,
            padx=(6, 0),
        )

        # -------- command bar ------------------------------------------
        bar = tk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.cmd_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _e: self._send_from_entry())
        ttk.Button(bar, text="Send", command=self._send_from_entry).grid(
            row=0,
            column=2,
        )

    # ---------------------------------------------------------------- exit
    def _on_exit(self) -> None:
        """Gracefully close the GUI; `mainloop()` will return to main.py."""
        self.quit()

    # ---------------------------------------------------------------- events
    def _on_list_click(self, _e: Any) -> None:
        sel = self.listbox.curselection()
        if sel:
            idx = sel[0] + 1  # listbox is 0‑based, worker expects 1‑based
            self._queue_command(f"click {idx}")

    def _send_from_entry(self) -> None:
        text = self.cmd_var.get().strip()
        self.cmd_var.set("")
        if text:
            self._handle_input(text)

    # ---------- search / url helper ------------------------------------
    def _send_search(self) -> None:
        txt = self.search_var.get().strip()
        if not txt:
            return
        mode = self.search_mode.get()
        if mode == "url":
            cmd = f"open_url {txt}"
        else:  # google search
            cmd = f"search {txt}"
        self._handle_input(cmd)  # reuse existing path
        self.search_var.set("")  # clear after send

    # ---------------------------------------------------------------- logic
    def _handle_input(self, text: str) -> None:
        """
        Decide whether `text` is a low‑level command or English, then act.
        """
        self._log(f"> {text}")
        low = text.lower()
        if low.startswith(
            ("scroll", "start scroll", "stop scroll", "new tab", "close tab", "switch"),
        ):
            self._queue_command(text)
            return

        # free English → LLM  (catch & show full trace) ------------------
        try:
            response = primitive_to_browser_action(
                text,
                self.screenshot,
                tabs=self.tab_titles,
                buttons=[(idx, label) for idx, label, _ in self.elements],
            )
        except Exception:
            self._log_trace(traceback.format_exc())
            return

        if response is None:
            self._log("❗ Could not interpret instruction")
            return

        cmd = self._pick_low_level_cmd(response)
        if cmd:
            self._log(f"  ↳ {cmd}")
            self._queue_command(cmd)
        else:
            self._log("❗ No action selected")

    # ---------------------------------------------------------------- public API
    def send_text_command(self, text: str) -> None:
        """
        Called by primitive.init() – simply forwards to _handle_input in
        the GUI's thread‑safe manner (via event_generate).
        """
        # Ensure this runs in Tk's thread
        self.event_generate("<<SendTextCommand>>", when="tail")
        self._pending_text = text  # store temporarily

    # ---------------------------------------------------- schema → command
    def _pick_low_level_cmd(self, resp: dict) -> str | None:
        """
        Translate the pruned agent response (already dumped to a plain nested
        dict) into a single low‑level command string for the worker.

        Every `.get()` has a default, so missing keys are handled gracefully.
        """

        # ---------- tab actions ------------------------------------------
        for key, obj in resp.get("tab_actions", {}).items():
            if not obj.get("apply"):
                continue

            if key == "new_tab":
                return "new tab"

            if key.startswith("close_tab_"):
                tab = key[len("close_tab_") :]
                return f"close tab {tab.replace('_', ' ')}"

            if key.startswith("select_tab_"):
                tab = key[len("select_tab_") :]
                return f"switch to tab {tab.replace('_', ' ')}"

        # ---------- scroll actions ---------------------------------------
        sc = resp.get("scroll_actions", {})

        if sc.get("scroll_up", {}).get("apply"):
            px = sc["scroll_up"].get("pixels") or 300
            return f"scroll up {px}"

        if sc.get("scroll_down", {}).get("apply"):
            px = sc["scroll_down"].get("pixels") or 300
            return f"scroll down {px}"

        if sc.get("start_scrolling_up", {}).get("apply"):
            return "start scroll up"

        if sc.get("start_scrolling_down", {}).get("apply"):
            return "start scroll down"

        if sc.get("stop_scrolling_up", {}).get("apply") or sc.get(
            "stop_scrolling_down",
            {},
        ).get("apply"):
            return "stop scroll"

        # ---------- button actions ---------------------------------------
        slug_to_idx = {_slug(label): idx for idx, label, _ in self.elements}

        for key, obj in resp.get("button_actions", {}).items():
            if not obj.get("apply") or not key.startswith("click_button_"):
                continue

            slug_text = key[len("click_button_") :]
            if slug_text in slug_to_idx:
                return f"click {slug_to_idx[slug_text]}"  # click by index

            # Fallback: click by visible button text
            return f"click button {slug_text.replace('_', ' ')}"

        # ---------- search action ----------------------------------------
        sa = resp.get("search_action")
        if sa and sa.get("apply"):
            query = sa.get("query", "")
            return f"search {query}"

        # ---------- explicit URL navigation ------------------------------
        sua = resp.get("search_url_action")
        if sua and sua.get("apply"):
            raw = sua.get("url", "").strip()
            return f"open_url {raw}"

        # Nothing selected
        return None

    # put message onto command queue
    def _queue_command(self, cmd: str) -> None:
        try:
            self.cmd_q.put_nowait(cmd)
        except queue.Full:
            self._log("⚠ command queue full – retry shortly")

    # ---------------------------------------------------------------- queue polling
    def _poll_updates(self) -> None:
        updated = False
        while True:
            try:
                payload = self.up_q.get_nowait()
                self.elements = payload.get("elements", [])
                self.tab_titles = payload.get("tabs", [])
                img = payload.get("screenshot", b"")
                if img:
                    self.screenshot = img
                updated = True
            except queue.Empty:
                break

        if updated:
            self.listbox.delete(0, "end")
            for idx, label, hover in self.elements:
                self.listbox.insert(
                    "end",
                    f"{idx:>2}. {label}" + (" (on hover)" if hover else ""),
                )
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    def _poll_text_q(self):
        while True:
            try:
                txt = self.text_q.get_nowait()
            except queue.Empty:
                break
            self._handle_input(txt)  # reuse existing logic
        self.after(50, self._poll_text_q)

    # ---------------------------------------------------------------- logging
    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

    # ---------- pretty traceback ------------------------------------------
    def _log_trace(self, tb: str) -> None:
        """
        Insert a colourised traceback into the ScrolledText widget.
        """
        try:
            html = highlight(tb, PythonLexer(), HtmlFormatter(nowrap=True))
            # crude html→ansi strip: just remove <span … style="color:#RRGGBB">
            import html as _html
            import re

            ansi = re.sub(r"<[^>]+>", "", html)
            ansi = _html.unescape(ansi)
            self._log(ansi)
        except Exception:
            # fallback: plain
            self._log(tb)
