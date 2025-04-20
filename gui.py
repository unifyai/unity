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

    # ---------- universal mouse‑wheel helper -------------------------------
    def _bind_mousewheel(self, target, canvas):  # NEW
        def wheel(ev):
            if ev.num == 4 or ev.delta > 0:
                canvas.yview_scroll(-1, "units")
            elif ev.num == 5 or ev.delta < 0:
                canvas.yview_scroll(1, "units")

        target.bind("<MouseWheel>", wheel, add=True)  # Win / macOS
        target.bind("<Button-4>", wheel, add=True)  # X11 up
        target.bind("<Button-5>", wheel, add=True)  # X11 down

    # ──────────────────────── WIDGET CONSTRUCTION ────────────────────────
    def _build_widgets(self) -> None:
        """Build the full Tk layout (GUI thread)."""

        # ----- main window -------------------------------------------------
        self.title("Playwright helper")
        self.geometry("900x550")
        self.columnconfigure(0, weight=3)  # left notebook
        self.columnconfigure(1, weight=2)  # right panel
        self.rowconfigure(0, weight=1)  # main split
        self.rowconfigure(1, weight=0)  # search / url bar
        self.rowconfigure(2, weight=0)  # command bar
        self.rowconfigure(3, weight=0)  # enter‑text bar
        self.rowconfigure(4, weight=0)  # key buttons row

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

        # ── Elements pane – scrollable buttons ──────────────────────────  # NEW
        el_canvas = tk.Canvas(tab_elements_frame, highlightthickness=0)  # NEW
        el_scroll = ttk.Scrollbar(
            tab_elements_frame,
            orient="vertical",  # NEW
            command=el_canvas.yview,
        )  # NEW
        el_rows = ttk.Frame(el_canvas)  # NEW
        el_canvas.create_window((0, 0), window=el_rows, anchor="nw")  # NEW
        el_canvas.configure(yscrollcommand=el_scroll.set)  # NEW
        el_rows.bind(  # NEW
            "<Configure>",
            lambda e: el_canvas.configure(  # NEW
                scrollregion=el_canvas.bbox("all"),
            ),  # NEW
        )  # NEW
        el_canvas.grid(row=0, column=0, sticky="nsew")  # NEW
        el_scroll.grid(row=0, column=1, sticky="ns")  # NEW
        tab_elements_frame.rowconfigure(0, weight=1)  # NEW
        tab_elements_frame.columnconfigure(0, weight=1)  # NEW

        self._elements_rows_frame = el_rows

        # ---- universal mouse‑wheel support --------------------------------
        self._bind_mousewheel(el_canvas, el_canvas)  # NEW
        self._bind_mousewheel(el_rows, el_canvas)

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

        self._tab_rows_frame = tab_rows  # keep reference for rebuilds\
        self._el_canvas = el_canvas

        # ── colour‑aware style for element buttons ────────────────────
        # Decide if the window background is “dark” or “light”
        r, g, b = [c // 256 for c in self.winfo_rgb(self.cget("bg"))]  # 0‑255
        brightness = 0.299 * r + 0.587 * g + 0.114 * b  # luminance
        dark = brightness < 128

        fg_light, fg_dark = "#000000", "#ffffff"
        bg_idle_light, bg_idle_dark = "#f0f0f0", "#3a3a3a"
        bg_active_light, bg_active_dark = "#dcdcdc", "#505050"

        style = ttk.Style()
        style.configure(  # tighter vertical padding
            "Element.TButton",
            font=("Helvetica", 11),
            anchor="w",
            relief="flat",
            padding=(2, 1),  # NEW
            foreground=fg_dark if dark else fg_light,
            background=bg_idle_dark if dark else bg_idle_light,
        )
        style.map(
            "Element.TButton",
            background=[
                ("active", bg_active_dark if dark else bg_active_light),
                ("pressed", bg_active_dark if dark else bg_active_light),
            ],
            foreground=[("pressed", fg_dark if dark else fg_light)],
        )

        # ===================================================================
        #  ROW‑1  →  Search / URL bar
        # ===================================================================
        search = tk.Frame(self)
        search.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 4))
        search.columnconfigure(1, weight=1)

        tk.Label(search, text="Search / URL:").grid(row=0, column=0, sticky="w")

        self.search_var = tk.StringVar()
        self.search_mode = tk.StringVar(value="google")
        entry_s = tk.Entry(search, textvariable=self.search_var)
        entry_s.grid(row=0, column=1, sticky="ew")
        entry_s.bind("<Return>", lambda _e: self._send_search())

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

        # ===================================================================
        #  ROW‑2  →  Command bar
        # ===================================================================
        bar = tk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry_c = tk.Entry(bar, textvariable=self.cmd_var)
        entry_c.grid(row=0, column=1, sticky="ew")
        entry_c.bind("<Return>", lambda _e: self._send_from_entry())

        # ===================================================================
        #  ROW‑3  →  Enter‑text bar
        # ===================================================================
        et = tk.Frame(self)
        et.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 6))
        et.columnconfigure(1, weight=1)

        tk.Label(et, text="Enter text:").grid(row=0, column=0, sticky="w")
        self.enter_var = tk.StringVar()
        entry_e = tk.Entry(et, textvariable=self.enter_var)
        entry_e.grid(row=0, column=1, sticky="ew")
        entry_e.bind("<Return>", lambda _e: self._send_enter_text())

        # ===================================================================
        #  ROW‑4  →  Key‑buttons bar (stack horizontally)                    # NEW
        # ===================================================================
        keyrow = tk.Frame(self)
        keyrow.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 8))

        def kbtn(txt, cmd):
            ttk.Button(
                keyrow,
                text=txt,
                width=10,
                command=lambda c=cmd: self._handle_input(c),
            ).pack(side="left", padx=1, pady=1)

        # first row
        kbtn("Enter", "press enter")
        kbtn("Backspace", "press backspace")
        kbtn("Delete", "press delete")
        kbtn("Select All", "select all")
        kbtn("Shift ⬇", "hold shift")
        kbtn("Shift ⬆", "release shift")

        # second row  (line break)
        keyrow2 = tk.Frame(self)
        keyrow2.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 8))

        def k2(txt, cmd):
            ttk.Button(
                keyrow2,
                text=txt,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            ).pack(side="left", padx=1, pady=1)

        k2("←", "cursor left")
        k2("→", "cursor right")
        k2("↑", "cursor up")
        k2("↓", "cursor down")
        k2("⌃←", "move line start")
        k2("⌃→", "move line end")
        k2("⌥←", "move word left")
        k2("⌥→", "move word right")

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
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )  # NEW

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
        cmd = f"open url {txt}" if mode == "url" else f"search {txt}"
        self._handle_input(cmd)
        self.search_var.set("")

    # ---------- enter‑text helper ------------------------------------
    def _send_enter_text(self) -> None:
        raw = self.enter_var.get()
        if not raw:
            return
        # decode user‑typed escapes → actual control chars  ( \n \t \b … )
        try:
            decoded = bytes(raw, "utf-8").decode("unicode_escape")
        except Exception:
            decoded = raw
        # send with real newline characters so the worker presses <Enter>
        self._handle_input(f"enter text {decoded}")
        self.enter_var.set("")  # clear box

    # ──────────────────────── TABS‑PANE HELPERS ────────────────────────
    def _exec_tab_cmd(self, prefix: str, title: str) -> None:
        self._log(f"> {prefix} {title}")
        self._queue_command(f"{prefix} {title}")

    def _rebuild_tabs_rows(self) -> None:
        """Re‑create the list of browser tabs with per‑row buttons."""
        for child in self._tab_rows_frame.winfo_children():
            child.destroy()

        for title in self.tab_titles:
            shown = title if len(title) <= 20 else title[:17] + "…"  # NEW
            row = ttk.Frame(self._tab_rows_frame)
            row.pack(fill="x", pady=1, padx=2)
            row = ttk.Frame(self._tab_rows_frame)
            row.pack(fill="x", pady=1, padx=2)

            ttk.Label(row, text=shown, anchor="w").pack(
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

    # ---------- element‑button helpers ---------------------------------      # NEW
    def _exec_element_click(self, idx: int, label: str) -> None:  # NEW
        self._log(f"> click {label}")  # NEW
        self._queue_command(f"click {idx}")  # NEW

    def _rebuild_elements_rows(self) -> None:  # NEW
        """Refresh the scrollable button list for page elements."""  # NEW
        for c in self._elements_rows_frame.winfo_children():  # NEW
            c.destroy()  # NEW
        for idx, label, hover in self.elements:  # NEW
            txt = f"{idx}. {label}" + ("  (hover)" if hover else "")  # NEW
            btn = ttk.Button(  # NEW
                self._elements_rows_frame,
                text=txt,
                style="Element.TButton",
                command=lambda i=idx, l=label: self._exec_element_click(i, l),
            )
            btn.pack(fill="x", padx=1, pady=0)
            self._bind_mousewheel(btn, self._el_canvas)  # NEW

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
                "enter text",
                "press enter",
                "press backspace",
                "press delete",
                "cursor left",
                "cursor right",
                "cursor up",
                "cursor down",
                "select all",
                "move line start",
                "move line end",
                "move word left",
                "move word right",
                "hold shift",
                "release shift",
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
            # Elements pane (buttons)
            self._rebuild_elements_rows()  # NEW
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
