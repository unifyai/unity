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

from agent import primitive_to_browser_action, list_available_actions
from helpers import _slug
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


class ControlPanel(tk.Tk):
    """Main GUI window.  No Playwright calls occur on this thread."""

    REFRESH_INTERVAL_MS = 100  # how often we poll the update queue

    # ──────────────────────────────── INIT ────────────────────────────────
    def __init__(
        self,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[list[tuple[int, str, bool]]]",
        text_q: queue.Queue[str],
    ):
        super().__init__()
        self.cmd_q = command_q  # GUI → worker
        self.up_q = update_q  # worker → GUI
        self.text_q = text_q  # primitive → GUI
        self.elements: list[tuple[int, str, bool]] = []
        self.screenshot: bytes = b""
        self.tab_titles: list[str] = []
        self.history: list[dict] = []
        self.state: dict[str, Any] = {}

        self._build_widgets()

        # first refreshes
        self._refresh_state_label()
        self._refresh_actions_list()
        self._rebuild_tabs_rows()

        # timers & event bindings
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)
        self.after(50, self._poll_text_q)  # primitive queue
        self.bind(
            "<<SendTextCommand>>",
            lambda _e: self._handle_input(self._pending_text),
        )
        self._pending_text = ""

    # ──────────────────────── WIDGET CONSTRUCTION ────────────────────────
    def _build_widgets(self) -> None:
        """Build the full Tk layout (GUI thread)."""

        # ----- main window -------------------------------------------------
        self.title("Playwright helper")
        self.geometry("900x550")
        self.columnconfigure(0, weight=3)  # left notebook
        self.columnconfigure(1, weight=2)  # right panel
        self.rowconfigure(0, weight=1)

        # ===================================================================
        # LEFT NOTEBOOK  →  Elements  |  Tabs
        # ===================================================================
        left_nb = ttk.Notebook(self)
        left_nb.grid(row=0, column=0, sticky="nsew")

        # frames
        tab_elements_frame = ttk.Frame(left_nb)
        tab_tabs_frame = ttk.Frame(left_nb)
        left_nb.add(tab_elements_frame, text="Elements")
        left_nb.add(tab_tabs_frame, text="Tabs")

        # expand frames
        for f in (tab_elements_frame, tab_tabs_frame):
            f.rowconfigure(0, weight=1)
            f.columnconfigure(0, weight=1)

        # ── Elements listbox (clickables) ───────────────────────────────
        self.listbox = tk.Listbox(tab_elements_frame, font=("Helvetica", 11))
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb_el = ttk.Scrollbar(
            tab_elements_frame,
            orient="vertical",
            command=self.listbox.yview,
        )
        sb_el.grid(row=0, column=0, sticky="nse")
        self.listbox.config(yscrollcommand=sb_el.set)
        self.listbox.bind("<ButtonRelease-1>", self._on_list_click)
        self.listbox.bind("<Return>", self._on_list_click)

        # ── Tabs pane (scrollable rows with buttons) ────────────────────
        tab_canvas = tk.Canvas(tab_tabs_frame, highlightthickness=0)
        scroll_v = ttk.Scrollbar(
            tab_tabs_frame,
            orient="vertical",
            command=tab_canvas.yview,
        )
        tab_rows = ttk.Frame(tab_canvas)
        tab_canvas.create_window((0, 0), window=tab_rows, anchor="nw")
        tab_canvas.configure(yscrollcommand=scroll_v.set)
        tab_rows.bind(
            "<Configure>",
            lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all")),
        )

        tab_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_v.grid(row=0, column=1, sticky="ns")
        tab_tabs_frame.rowconfigure(0, weight=1)
        tab_tabs_frame.columnconfigure(0, weight=1)

        self._tab_rows_frame = tab_rows  # keep reference for rebuilds

        # ================================================================
        # RIGHT‑HAND PANEL →  notebook(Log/Actions) + state + buttons
        # ================================================================
        right = tk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=1)

        # Log / Actions notebook
        note = ttk.Notebook(right)
        note.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tab_log = ttk.Frame(note)
        tab_actions = ttk.Frame(note)
        note.add(tab_log, text="Log")
        note.add(tab_actions, text="Actions")

        self.log = scrolledtext.ScrolledText(tab_log, state="disabled")
        self.log.pack(fill="both", expand=True)

        self.act_box = scrolledtext.ScrolledText(tab_actions, state="disabled")
        self.act_box.pack(fill="both", expand=True)
        self._last_actions_txt = ""  # cache for anti‑jitter

        # browser‑state read‑out
        self.state_var = tk.StringVar()
        self.state_lbl = tk.Label(
            right,
            textvariable=self.state_var,
            justify="left",
            anchor="w",
            font=("Consolas", 9),
        )
        self.state_lbl.grid(row=1, column=0, sticky="nsew", padx=5)

        # control buttons
        btns = tk.Frame(right)
        btns.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
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

        ttk.Button(
            btns,
            text="Exit",
            command=self._on_exit,
            style="Danger.TButton",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    # ──────────────────────── TABS‑PANE HELPERS ────────────────────────
    def _exec_tab_cmd(self, prefix: str, title: str) -> None:
        self._log(f"> {prefix} {title}")
        self._queue_command(f"{prefix} {title}")

    def _rebuild_tabs_rows(self) -> None:
        """Re‑create the list of browser tabs with per‑row buttons."""
        for child in self._tab_rows_frame.winfo_children():
            child.destroy()

        for title in self.tab_titles:
            row = ttk.Frame(self._tab_rows_frame)
            row.pack(fill="x", pady=1, padx=2)

            ttk.Label(row, text=title, anchor="w").pack(
                side="left",
                fill="x",
                expand=True,
                padx=(0, 4),
            )

            ttk.Button(
                row,
                text="Go",
                width=4,
                command=lambda t=title: self._exec_tab_cmd("switch to tab", t),
            ).pack(side="right")

            ttk.Button(
                row,
                text="×",
                width=2,
                command=lambda t=title: self._exec_tab_cmd("close tab", t),
            ).pack(side="right", padx=(0, 2))

    # ──────────────────────── ACTIONS‑PANE HELPER ───────────────────────
    def _refresh_actions_list(self) -> None:
        """Update the Actions tab (anti‑jitter, preserves scroll)."""
        groups = list_available_actions(
            self.tab_titles,
            [(idx, label) for idx, label, _ in self.elements],
        )

        lines: list[str] = []
        for grp, names in groups.items():
            lines.append(f"[{grp}]")
            lines.extend(f"  {n}" for n in names)
            lines.append("")
        new_txt = "\n".join(lines)

        if new_txt == self._last_actions_txt:
            return
        self._last_actions_txt = new_txt

        pos = self.act_box.yview()
        self.act_box.configure(state="normal")
        self.act_box.delete("1.0", "end")
        self.act_box.insert("end", new_txt)
        self.act_box.configure(state="disabled")
        if pos[0] > 0.01:
            self.act_box.yview_moveto(pos[0])

    # ───────────────────────── BROWSER‑STATE LABEL ───────────────────────
    def _refresh_state_label(self) -> None:
        st = self.state or {}
        self.state_var.set(
            f"url:         {st.get('url', '')[:60]}\n"
            f"title:       {st.get('title', '')[:60]}\n"
            f"scroll_y:    {st.get('scroll_y', 0)}\n"
            f"auto_scroll: {st.get('auto_scroll', None)}\n"
            f"in_textbox:  {st.get('in_textbox', False)}",
        )

    # ────────────────────────── LISTBOX CLICK ───────────────────────────
    def _on_list_click(self, _e: Any) -> None:
        sel = self.listbox.curselection()
        if sel:
            pos = sel[0]
            idx = pos + 1
            label = self.elements[pos][1]
            self._log(f"> click {label}")
            self._queue_command(f"click {idx}")

    # ───────────────────────────── EXIT ─────────────────────────────────
    def _on_exit(self) -> None:
        self.quit()

    # ─────────────────────── HIGH‑LEVEL INPUT HANDLER ───────────────────
    def _handle_input(self, text: str) -> None:
        self._log(f"> {text}")
        low = text.lower()
        if low.startswith(
            (
                "click",
                "scroll",
                "start scroll",
                "stop scroll",
                "search",
                "open url",
                "new tab",
                "close tab",
                "switch",
            ),
        ):
            self._queue_command(text)
            return

        # free English → LLM
        try:
            response = primitive_to_browser_action(
                text,
                self.screenshot,
                tabs=self.tab_titles,
                buttons=[(idx, label) for idx, label, _ in self.elements],
                history=self.history,
                state=self.state,
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

    # ───────────────────────── PUBLIC PRIMITIVE API ─────────────────────
    def send_text_command(self, text: str) -> None:
        self.event_generate("<<SendTextCommand>>", when="tail")
        self._pending_text = text

    # ───────────────────────── PICK LOW‑LEVEL CMD ───────────────────────
    def _pick_low_level_cmd(self, resp: dict) -> str | None:
        # ----- tab actions -------------------------------------------------
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

        # ----- scroll actions ---------------------------------------------
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

        # ----- button actions ---------------------------------------------
        slug_to_idx = {_slug(label): idx for idx, label, _ in self.elements}
        for key, obj in resp.get("button_actions", {}).items():
            if not obj.get("apply") or not key.startswith("click_button_"):
                continue
            slug_text = key[len("click_button_") :]
            if slug_text in slug_to_idx:
                return f"click {slug_to_idx[slug_text]}"
            return f"click button {slug_text.replace('_', ' ')}"

        # ----- search / open‑url ------------------------------------------
        sa = resp.get("search_action")
        if sa and sa.get("apply"):
            return f"search {sa.get('query', '')}"
        sua = resp.get("search_url_action")
        if sua and sua.get("apply"):
            return f"open url {sua.get('url', '').strip()}"

        return None

    # ───────────────────────── COMMAND QUEUE ────────────────────────────
    def _queue_command(self, cmd: str) -> None:
        try:
            self.cmd_q.put_nowait(cmd)
        except queue.Full:
            self._log("⚠ command queue full – retry shortly")

    # ──────────────────────── UPDATE POLLING ────────────────────────────
    def _poll_updates(self) -> None:
        updated = False
        while True:
            try:
                payload = self.up_q.get_nowait()
                self.elements = payload.get("elements", [])
                self.tab_titles = payload.get("tabs", [])
                self.history = payload.get("history", self.history)
                self.state = payload.get("state", self.state)
                img = payload.get("screenshot", b"")
                if img:
                    self.screenshot = img
                updated = True
            except queue.Empty:
                break

        if updated:
            # Elements list
            self.listbox.delete(0, "end")
            for idx, label, hover in self.elements:
                self.listbox.insert(
                    "end",
                    f"{idx:>2}. {label}" + (" (on hover)" if hover else ""),
                )
            # Tabs & Actions
            self._rebuild_tabs_rows()
            self._refresh_actions_list()
            self._refresh_state_label()

        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    # ──────────────────────── PRIMITIVE TEXT POLL ───────────────────────
    def _poll_text_q(self):
        while True:
            try:
                txt = self.text_q.get_nowait()
            except queue.Empty:
                break
            self._handle_input(txt)
        self.after(50, self._poll_text_q)

    # ─────────────────────────── LOGGING ────────────────────────────────
    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

    def _log_trace(self, tb: str) -> None:
        """Insert a colourised traceback into the log."""
        try:
            html = highlight(tb, PythonLexer(), HtmlFormatter(nowrap=True))
            import html as _html, re

            ansi = _html.unescape(re.sub(r"<[^>]+>", "", html))
            self._log(ansi)
        except Exception:
            self._log(tb)
