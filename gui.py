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
from tkinter import scrolledtext, ttk, Button
from typing import Any

from agent import primitive_to_browser_action, list_available_actions
from actions import BrowserState
from action_filter import get_valid_actions
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

        # for graying out when not in textbox
        self._key_buttons = {}

        self._build_widgets()

        self._worker = None  # will be set by set_worker()

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
    def _bind_mousewheel(self, target, canvas):
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

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4)
        paned.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1, minsize=250)  # ✅ enforce min width for left
        self.columnconfigure(1, weight=2)

        # ── top‑right “X” button (absolute) ─────────────────────────────────
        close_btn = ttk.Button(
            self,
            text="×",
            width=3,
            style="Danger.TButton",
            command=self._on_exit,
        )
        # place at the top‑right corner (6 px padding)
        close_btn.place(relx=1.0, rely=0.0, x=-6, y=6, anchor="ne")

        # after all other widgets have been laid out, raise the button
        self.after_idle(close_btn.lift)  # NEW — guarantees visibility

        # ===================================================================
        # LEFT NOTEBOOK  →  Elements  |  Tabs
        # ===================================================================
        self.left_wrapper = tk.Frame(paned, width=300)
        self.left_wrapper.pack_propagate(False)  # ✅ if you're using .pack() inside
        paned.add(self.left_wrapper, minsize=200, stretch="always")

        left_nb = ttk.Notebook(self.left_wrapper)
        left_nb.pack(fill="both", expand=True)

        left_nb.columnconfigure(0, weight=1)
        left_nb.rowconfigure(0, weight=1)

        # frames
        tab_elements_frame = ttk.Frame(left_nb)
        tab_tabs_frame = ttk.Frame(left_nb)
        left_nb.add(tab_elements_frame, text="Elements")
        left_nb.add(tab_tabs_frame, text="Tabs")

        # expand frames
        for f in (tab_elements_frame, tab_tabs_frame):
            f.rowconfigure(0, weight=1)
            f.columnconfigure(0, weight=1)

        # ── Elements pane – scrollable buttons ──────────────────────────
        el_canvas = tk.Canvas(tab_elements_frame, highlightthickness=0)
        el_scroll = ttk.Scrollbar(
            tab_elements_frame,
            orient="vertical",
            command=el_canvas.yview,
        )
        el_rows = ttk.Frame(el_canvas)
        el_canvas.create_window((0, 0), window=el_rows, anchor="nw")
        el_canvas.configure(yscrollcommand=el_scroll.set)
        el_rows.bind(
            "<Configure>",
            lambda e: el_canvas.configure(
                scrollregion=el_canvas.bbox("all"),
            ),
        )
        el_canvas.grid(row=0, column=0, sticky="nsew")
        el_scroll.grid(row=0, column=1, sticky="ns")
        tab_elements_frame.rowconfigure(0, weight=1)
        tab_elements_frame.columnconfigure(0, weight=1)

        self._elements_rows_frame = el_rows

        # ---- universal mouse‑wheel support --------------------------------
        self._bind_mousewheel(el_canvas, el_canvas)
        self._bind_mousewheel(el_rows, el_canvas)

        # ── Tabs pane (scrollable rows with buttons) ────────────────────
        tab_canvas = tk.Canvas(tab_tabs_frame, highlightthickness=0)
        scroll_v = ttk.Scrollbar(
            tab_tabs_frame,
            orient="vertical",
            command=tab_canvas.yview,
        )
        tab_rows = ttk.Frame(tab_canvas)
        tab_canvas.create_window(
            (0, 0),
            window=tab_rows,
            anchor="nw",
            tags="tabframe",
            width=1,
        )

        def resize_tabs(event):
            tab_canvas.itemconfig("tabframe", width=event.width)

        tab_canvas.bind("<Configure>", resize_tabs)

        tab_canvas.configure(yscrollcommand=scroll_v.set)
        tab_rows.bind(
            "<Configure>",
            lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all")),
        )

        tab_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_v.grid(row=0, column=1, sticky="ns")
        tab_tabs_frame.rowconfigure(0, weight=1)
        tab_tabs_frame.columnconfigure(0, weight=1)

        self._tab_rows_frame = tab_rows
        tab_canvas.columnconfigure(0, weight=1)
        self._el_canvas = el_canvas
        self._el_scroll = el_scroll
        self._reset_el_scroll = False

        # ── colour‑aware style for element buttons ────────────────────
        # Decide if the window background is “dark” or “light”
        r, g, b = [c // 256 for c in self.winfo_rgb(self.cget("bg"))]  # 0‑255
        brightness = 0.299 * r + 0.587 * g + 0.114 * b  # luminance
        dark = brightness < 128

        fg_light, fg_dark = "#000000", "#ffffff"
        bg_idle_light, bg_idle_dark = "#f0f0f0", "#3a3a3a"
        bg_active_light, bg_active_dark = "#dcdcdc", "#505050"

        style = ttk.Style()

        # centre text on buttons
        style.configure("TButton", anchor="center")

        # tighter vertical padding
        style.configure(
            "Element.TButton",
            font=("Helvetica", 11),
            anchor="w",
            relief="flat",
            padding=(2, 1),
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

        # Disabled and enabled button styles
        style.map(
            "TButton",
            foreground=[
                ("disabled", "#888888"),
                ("!disabled", "#ffffff"),
            ],
            background=[
                ("disabled", "#2a2a2a"),
                ("active", "#505050"),
                ("!disabled", "#3a3a3a"),
            ],
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
        #  ROW‑3  →  Enter‑text bar
        # ===================================================================
        et = tk.Frame(self)
        et.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 6))
        et.columnconfigure(1, weight=1)

        tk.Label(et, text="Enter text:").grid(row=0, column=0, sticky="w")
        self.enter_var = tk.StringVar()
        self.enter_text_box = tk.Entry(et, textvariable=self.enter_var)
        self.enter_text_box.grid(row=0, column=1, sticky="ew")
        self.enter_text_box.bind("<Return>", lambda _e: self._send_enter_text())

        # ===================================================================
        #  ROW‑4  →  Key‑buttons bar (stack horizontally)
        # ===================================================================

        # Create frame
        self.keyrow = tk.Frame(self)
        self.keyrow.bind("<Configure>", lambda e: self._relayout_key_buttons())
        self.keyrow.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(0, 8),
        )

        # Store buttons in a list
        self._key_buttons = {}
        self._key_button_widgets = []

        key_cmds = [
            ("Enter", "press enter"),
            ("Backspace", "press backspace"),
            ("Delete", "press delete"),
            ("Select All", "select all"),
            ("Shift ⬇", "hold shift"),
            ("Shift ⬆", "release shift"),
            ("Click Out", "click out"),
        ]

        for label, cmd in key_cmds:
            b = ttk.Button(
                self.keyrow,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            self._key_buttons[cmd] = b
            self._key_button_widgets.append(b)

        # second row  (line break)
        self.keyrow2 = tk.Frame(self)
        self.keyrow2.bind("<Configure>", lambda e: self._relayout_arrow_buttons())
        self.keyrow2.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(0, 8),
        )

        self._arrow_button_widgets = []

        arrow_cmds = [
            ("←", "cursor left"),
            ("→", "cursor right"),
            ("↑", "cursor up"),
            ("↓", "cursor down"),
            ("⌃←", "move line start"),
            ("⌃→", "move line end"),
            ("⌥←", "move word left"),
            ("⌥→", "move word right"),
        ]

        for label, cmd in arrow_cmds:
            b = ttk.Button(
                self.keyrow2,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            self._key_buttons[cmd] = b
            self._arrow_button_widgets.append(b)

        # ===================================================================
        #  ROW‑6  →  LLM Command bar
        # ===================================================================
        bar = tk.Frame(self)
        bar.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(0, 8),
        )
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="LLM Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry_c = tk.Entry(bar, textvariable=self.cmd_var)
        entry_c.grid(row=0, column=1, sticky="ew")
        entry_c.bind("<Return>", lambda _e: self._send_from_entry())

        # ================================================================
        # RIGHT‑HAND PANEL →  notebook(Log/Actions) + state + buttons
        # ================================================================
        self.right_panel = tk.Frame(paned)
        paned.add(self.right_panel, stretch="always")
        right = self.right_panel
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=0)
        right.columnconfigure(0, weight=1)

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
                width=10,
                command=lambda: self._handle_input(cmd),
            ).grid(row=r, column=c, sticky="ew")

        make(0, 0, "▲ Scroll 100", "scroll up 100")
        make(0, 1, "▼ Scroll 100", "scroll down 100")
        make(1, 0, "Start ▲", "start scroll up")
        make(1, 1, "Start ▼", "start scroll down")
        make(2, 0, "Stop scroll", "stop scroll")
        make(2, 1, "Continue", "continue scroll")
        make(3, 0, "New tab", "new tab")
        make(3, 1, "Close tab", "close tab")

    # dynamic key-press button wrap
    def _relayout_key_buttons(self):
        for widget in self.keyrow.winfo_children():
            widget.grid_forget()

        width = self.keyrow.winfo_width()
        if width == 0:
            self.after(100, self._relayout_key_buttons)
            return

        # Approximate button width + padding
        min_button_px = 120
        num_cols = max(2, width // min_button_px)

        for i, b in enumerate(self._key_button_widgets):
            b.grid(row=i // num_cols, column=i % num_cols, sticky="ew", padx=1, pady=1)

        for c in range(num_cols):
            self.keyrow.columnconfigure(c, weight=1)

    def _relayout_arrow_buttons(self):
        for widget in self.keyrow2.winfo_children():
            widget.grid_forget()

        width = self.keyrow2.winfo_width()
        if width == 0:
            self.after(100, self._relayout_arrow_buttons)
            return

        min_button_px = 120
        num_cols = max(2, width // min_button_px)

        for i, b in enumerate(self._arrow_button_widgets):
            b.grid(row=i // num_cols, column=i % num_cols, sticky="ew", padx=1, pady=1)

        for c in range(num_cols):
            self.keyrow2.columnconfigure(c, weight=1)

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
            shown = title if len(title) <= 20 else title[:17] + "…"
            row = ttk.Frame(self._tab_rows_frame)
            row.grid(sticky="ew", padx=2, pady=1)
            row.columnconfigure(0, weight=1)  # title column stretches

            label = ttk.Label(row, text=shown, anchor="w")
            label.grid(row=0, column=0, sticky="ew", padx=(0, 4))

            label.bind("<Enter>", lambda e, full=title: label.configure(text=full))
            label.bind("<Leave>", lambda e, short=shown: label.configure(text=short))

            btn_font = ("Helvetica", 10, "bold")

            close_btn = Button(
                row,
                text="×",
                command=lambda t=title: self._exec_tab_cmd("close tab", t),
                padx=4,
                pady=2,  # ← bump this slightly to avoid visual clipping
                relief="flat",
                font=btn_font,
            )
            close_btn.grid(row=0, column=1, padx=(0, 2), sticky="e")

            go_btn = Button(
                row,
                text="Go",
                command=lambda t=title: self._exec_tab_cmd("switch to tab", t),
                padx=6,
                pady=2,
                relief="flat",
                font=btn_font,
            )
            go_btn.grid(row=0, column=2, sticky="e")

            # Stretch row container to fill width
            self._tab_rows_frame.columnconfigure(0, weight=1)

    def _refresh_enabled_controls(self, valid):
        for cmd, btn in self._key_buttons.items():
            btn.configure(state="normal" if cmd in valid else "disabled")
        if "enter text" in valid:
            self.enter_text_box.configure(state="normal")
        else:
            self.enter_text_box.configure(state="disabled")

    # ──────────────────────── ACTIONS‑PANE HELPER ───────────────────────
    def _refresh_actions_list(self) -> None:
        """Update the Actions tab (anti‑jitter, preserves scroll)."""
        groups = list_available_actions(
            self.tab_titles,
            [(idx, label) for idx, label, _ in self.elements],
            state=BrowserState(**self.state),
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

        valid = get_valid_actions(BrowserState(**self.state))
        self._refresh_enabled_controls(valid)

        self.act_box.configure(state="normal")
        self.act_box.delete("1.0", "end")

        for grp, names in groups.items():
            self.act_box.insert("end", f"[{grp}]\n")
            for name in names:
                is_valid = any(
                    name == v or (v.endswith("*") and name.startswith(v[:-1]))
                    for v in valid
                )
                if not is_valid:
                    self.act_box.insert("end", f"  {name}\n", "disabled")
                else:
                    self.act_box.insert("end", f"  {name}\n")
            self.act_box.insert("end", "\n")

        self.act_box.tag_config("disabled", foreground="gray")
        self.act_box.configure(state="disabled")

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

    # ---------- element‑button helpers ---------------------------------
    def _exec_element_click(self, idx: int, label: str) -> None:
        self._log(f"> click {label}")
        self._queue_command(f"click {idx}")
        self._reset_el_scroll = True

    def _rebuild_elements_rows(self) -> None:
        """Refresh the scrollable button list for page elements."""
        for c in self._elements_rows_frame.winfo_children():
            c.destroy()
        for idx, label, hover in self.elements:
            # flatten any embedded newlines to avoid tall buttons
            flat = " ".join(label.splitlines())
            txt = f"{idx}. {flat}" + ("  (hover)" if hover else "")
            btn = ttk.Button(
                self._elements_rows_frame,
                text=txt,
                style="Element.TButton",
                command=lambda i=idx, l=label: self._exec_element_click(i, l),
            )
            btn.pack(fill="x", padx=1, pady=0)
            self._bind_mousewheel(btn, self._el_canvas)
        # ---- show scrollbar only when needed ---------------------------  # NEW
        self._elements_rows_frame.update_idletasks()
        content_h = self._elements_rows_frame.winfo_reqheight()
        pane_h = self._el_canvas.winfo_height()

        # If content fits, disable scrolling and pin to top
        if content_h <= pane_h:
            self._el_scroll.grid_remove()
            self._el_canvas.configure(
                scrollregion=(0, 0, 0, pane_h),
            )  # prevent scrolling
            self._el_canvas.yview_moveto(0)  # pin to top
        else:
            self._el_scroll.grid()  # show scrollbar
            self._el_canvas.configure(scrollregion=self._el_canvas.bbox("all"))

    # ───────────────────────────── EXIT ─────────────────────────────────
    def _on_exit(self) -> None:
        # ‑‑ stop worker fast and exit ‑‑
        try:
            if self._worker:
                self._worker.stop()
                self._worker.join(timeout=0.5)
        except Exception:
            pass
        self.destroy()
        import os

        os._exit(0)

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
                "click out",
                "continue scroll",
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

    def set_worker(self, worker):
        self._worker = worker

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
        # mark commands that likely change the page content  ------------- NEW
        nav_prefixes = ("open url", "search ", "new tab", "switch to tab")
        if cmd.lower().startswith(nav_prefixes):
            self._reset_el_scroll = True
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
            self._rebuild_elements_rows()

            if self._reset_el_scroll:
                self._el_canvas.yview_moveto(0)
                self._reset_el_scroll = False

            # Tabs & Actions
            self._rebuild_tabs_rows()
            self._refresh_actions_list()
            self._refresh_state_label()
            self._refresh_enabled_controls(
                get_valid_actions(BrowserState(**self.state)),
            )

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
