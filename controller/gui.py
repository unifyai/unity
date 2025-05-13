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
from tkinter import scrolledtext, ttk, Button
from typing import Any, Callable
import threading
import itertools
import asyncio

from controller.agent import (
    text_to_browser_action,
    list_available_actions,
    ADVANCED_MODE,
)
from controller.states import BrowserState
from controller.commands import *
from controller.action_filter import get_valid_actions
from controller.helpers import _slug
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def _contrasting(color: str) -> str:
    """Return black or white depending on background luminance."""
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    # perceived luminance (ITU BT.601)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if y > 128 else "#ffffff"


class _Tooltip:
    """Tiny self‑contained tooltip; never escapes the main window."""

    PAD, DELAY = 6, 400  # px, ms

    def __init__(self, widget: tk.Widget, text: str):
        self.widget, self.text = widget, text
        self.tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    # ---------- internal --------------------------------------------------
    def _schedule(self, *_):
        # Show only when the widget is disabled
        if str(self.widget.cget("state")) != "disabled":
            return
        self._id = self.widget.after(self.DELAY, self._show)

    def _show(self):
        # Bail if the widget has been re‑enabled meanwhile
        if str(self.widget.cget("state")) != "disabled":
            return
        if self.tip:
            return
        ...

        if self.tip:  # already visible
            return

        # -- 1.  Anchor coords inside *screen* space -----------------------
        try:
            bx, by, bw, bh = self.widget.bbox("insert")  # Entry/Text
        except Exception:
            bx = by = 0
            bw = self.widget.winfo_width()
            bh = self.widget.winfo_height()

        x = self.widget.winfo_rootx() + bx + bw // 2
        y = self.widget.winfo_rooty() + by + bh + self.PAD

        # -- 2.  Clamp inside parent toplevel (so we never cover browser) --
        top = self.widget.winfo_toplevel()
        tlx, tly = top.winfo_rootx(), top.winfo_rooty()
        trw, trh = top.winfo_width(), top.winfo_height()
        scr_w, scr_h = top.winfo_screenwidth(), top.winfo_screenheight()

        # basic screen‑edge clamp
        x = max(tlx + self.PAD, min(x, tlx + trw - self.PAD))
        y = max(tly + self.PAD, min(y, tly + trh - self.PAD))

        # -- 3.  Create the floating window --------------------------------
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        bg = "#ffffe0"
        tk.Label(
            tw,
            text=self.text,
            bg=bg,
            fg=_contrasting(bg),
            relief="solid",
            borderwidth=1,
            font=("tahoma", 8),
        ).pack()

    def _hide(self, *_):
        if hasattr(self, "_id"):
            self.widget.after_cancel(self._id)
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ControlPanel(tk.Tk):
    """Main GUI window.  No Playwright calls occur on this thread."""

    REFRESH_INTERVAL_MS = 100  # how often we poll the update queue

    # ──────────────────────────────── INIT ────────────────────────────────
    def __init__(
        self,
        command_q: "queue.Queue[str]",
        update_q: "queue.Queue[dict] | None" = None,
        text_q: queue.Queue[str] | None = None,
    ):
        super().__init__()
        self.cmd_q = command_q  # GUI → worker
        self.up_q = update_q  # worker → GUI (may be None when using Redis)
        self.text_q = text_q or queue.Queue()  # primitive → GUI
        self.elements: list[tuple[int, str, bool]] = []
        self.screenshot: bytes = b""
        self.tab_titles: list[str] = []
        self.history: list[dict] = []
        self.state: dict[str, Any] = {}

        # ── async LLM helper --------------------------------------------
        self._llm_resp_q: "queue.Queue[tuple[str,Any]]" = queue.Queue()
        self._llm_busy = False
        self._llm_dots = itertools.cycle([".", "..", "..."])
        self._llm_line_idx = None

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

        # model poller
        self.after(50, self._poll_llm_resp)

        # If no update queue provided, subscribe to Redis browser_state
        if self.up_q is None:
            import redis, ast, json

            self._redis_pub = redis.Redis(host="localhost", port=6379, db=0).pubsub()
            self._redis_pub.subscribe("browser_state")

            def _get_update_from_redis():
                msg = self._redis_pub.get_message()
                if msg and msg["type"] == "message":
                    data = msg["data"]
                    try:
                        return json.loads(data)
                    except Exception:
                        try:
                            return ast.literal_eval(data.decode() if isinstance(data, bytes) else data)
                        except Exception:
                            return None
                return None

            self._pull_update = _get_update_from_redis
        else:
            # pull from queue
            def _get_update_from_queue():
                try:
                    return self.up_q.get_nowait()
                except queue.Empty:
                    return None

            self._pull_update = _get_update_from_queue

    # ─────────────────────────────────────────

    def _advance_llm_dots(self):
        if not self._llm_busy or self._llm_line_idx is None:
            return  # stop when the model reply arrives

        # update the existing line in‑place
        self.log.configure(state="normal")
        self.log.delete(self._llm_line_idx, f"{self._llm_line_idx} lineend")
        new_txt = "⏳ calling model" + next(self._llm_dots)
        self.log.insert(self._llm_line_idx, new_txt, "llm")
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)

        self.after(400, self._advance_llm_dots)

    def _log_line(self, text: str, tag: str | None = None) -> str:
        """
        Append *text* to the log and return its starting index (Tk text index).
        Optionally tag the line.
        """
        self.log.configure(state="normal")
        idx = self.log.index("end-1c linestart")
        self.log.insert("end", text + "\n", tag)
        self.log.configure(state="disabled")
        self.log.yview_moveto(1.0)
        return idx

    # ─────────────────── background LLM thread ──────────────────────
    def _start_llm_thread(self, user_text: str) -> None:
        """
        Fire a daemon thread that calls `primitive_to_browser_action`
        and drops (status, payload) tuples onto `_llm_resp_q`.
        """
        if self._llm_busy:
            self._log("⚠ LLM still working – please wait")
            return
        self._llm_busy = True

        # --- spawn animated log line ------------------------------------
        msg = "⏳ calling model" + next(self._llm_dots)
        self._llm_line_idx = self._log_line(msg, tag="llm")
        self.after(400, self._advance_llm_dots)

        # snapshot everything the model needs
        screenshot = self.screenshot
        tabs = list(self.tab_titles)
        buttons = [(i, lbl) for i, lbl, _ in self.elements]
        history = list(self.history)
        state = dict(self.state)

        def _worker():
            try:
                resp = text_to_browser_action(
                    user_text,
                    screenshot,
                    tabs=tabs,
                    buttons=buttons,
                    history=history,
                    state=state,
                )
                self._llm_resp_q.put(("ok", resp))
            except Exception as exc:
                import traceback as _tb

                self._llm_resp_q.put(("err", _tb.format_exc()))

        threading.Thread(target=_worker, daemon=True).start()

    def _poll_llm_resp(self):
        """Check the response queue every 50 ms."""
        try:
            while True:
                status, payload = self._llm_resp_q.get_nowait()

                if status == "ok":
                    if payload:
                        cmd = (
                            payload["action"]
                            if (not ADVANCED_MODE and isinstance(payload, dict))
                            else self._llm_resp_to_cmd(payload)
                        )
                        if cmd and cmd.startswith("click_button_"):
                            # keep only the numeric id in front of the slug
                            idx = cmd[len("click_button_") :].split("_", 1)[0]
                            cmd = f"click {idx}"

                        repl = f"↳ {cmd}" if cmd else "❗ No action selected"
                    else:
                        repl = "❗ Could not interpret instruction"
                else:  # "err"
                    repl = "❗ LLM error – see traceback"
                    self._log_trace(payload)

                # --- replace the animated line ---------------------------
                if self._llm_line_idx:
                    self.log.configure(state="normal")
                    self.log.delete(self._llm_line_idx, f"{self._llm_line_idx} lineend")
                    self.log.insert(self._llm_line_idx, repl + "\n")
                    self.log.configure(state="disabled")
                    self.log.yview_moveto(1.0)
                    self._llm_line_idx = None

                # if we produced a real command, queue it
                if repl.startswith("↳ "):
                    self._queue_command(repl[2:].strip())

                self._llm_busy = False

        except queue.Empty:
            pass
        self.after(50, self._poll_llm_resp)

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
        self.geometry("1000x650")

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4)
        paned.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=2)

        # ── top‑right "X" button (absolute) ─────────────────────────────────
        close_btn = ttk.Button(
            self,
            text="×",
            width=3,
            style="Danger.TButton",
            command=self._on_exit,
        )
        # place at the top‑right corner (6 px padding)
        close_btn.place(relx=1.0, rely=0.0, x=-6, y=6, anchor="ne")

        # after all other widgets have been laid out, raise the button
        self.after_idle(close_btn.lift)  # NEW — guarantees visibility

        # ===================================================================
        # LEFT NOTEBOOK  →  Elements  |  Tabs
        # ===================================================================
        self.left_wrapper = tk.Frame(paned, width=300)
        self.left_wrapper.pack_propagate(False)
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
        # Decide if the window background is "dark" or "light"
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

        # ── Disabled look (same palette used by main scroll buttons) ──
        disabled_bg = "#2a2a2a"
        active_bg = "#505050"
        idle_bg = bg_idle_dark if dark else bg_idle_light
        active_fg = fg_dark if dark else fg_light

        # Element‑list rows
        style.map(
            "Element.TButton",
            foreground=[("disabled", "#888888"), ("!disabled", active_fg)],
            background=[
                ("disabled", disabled_bg),
                ("active", active_bg),
                ("pressed", active_bg),
                ("!disabled", idle_bg),
            ],
        )

        # Per‑tab "×" and Go buttons
        style.configure(
            "TabRow.TButton",
            font=("Helvetica", 10, "bold"),
            padding=(4, 2),
            foreground=active_fg,
            background=idle_bg,
            relief="flat",
        )
        style.map(
            "TabRow.TButton",
            foreground=[("disabled", "#888888"), ("!disabled", active_fg)],
            background=[
                ("disabled", disabled_bg),
                ("active", active_bg),
                ("pressed", active_bg),
                ("!disabled", idle_bg),
            ],
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
        self.search_entry = tk.Entry(search, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew")
        self.search_entry.bind("<Return>", lambda _e: self._send_search())

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
            ("Enter", CMD_PRESS_ENTER),
            ("Backspace", CMD_PRESS_BACKSPACE),
            ("Delete", CMD_PRESS_DELETE),
            (CMD_SELECT_ALL, CMD_SELECT_ALL),
            ("Shift ⬇", CMD_HOLD_SHIFT),
            ("Shift ⬆", CMD_RELEASE_SHIFT),
            (CMD_CLICK_OUT, CMD_CLICK_OUT),
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
            ("←", CMD_CURSOR_LEFT),
            ("→", CMD_CURSOR_RIGHT),
            ("↑", CMD_CURSOR_UP),
            ("↓", CMD_CURSOR_DOWN),
            ("⌃←", CMD_MOVE_LINE_START),
            ("⌃→", CMD_MOVE_LINE_END),
            ("⌥←", CMD_MOVE_WORD_LEFT),
            ("⌥→", CMD_MOVE_WORD_RIGHT),
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
        #  ROW‑6  →  Bottom bar (Tab controls)
        # ===================================================================
        bottom_bar = tk.Frame(self)
        bottom_bar.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(0, 8),
        )

        bottom_bar.columnconfigure(0, weight=1)
        bottom_bar.columnconfigure(1, weight=1)

        # Ensure command-button registry exists before any buttons are created below.
        self._cmd_buttons: dict[str, ttk.Button] = {}

        def _mk_tab_btn(col: int, txt: str, cmd: str):
            b = ttk.Button(
                bottom_bar,
                text=txt,
                width=12,
                command=lambda: self._handle_input(cmd),
            )
            b.grid(row=0, column=col, sticky="ew", padx=2)
            self._cmd_buttons[cmd] = b

        _mk_tab_btn(0, "New Tab", CMD_NEW_TAB)
        _mk_tab_btn(1, "Close Tab", CMD_CLOSE_THIS_TAB)

        # ===================================================================
        #  ROW‑7  →  LLM Command bar (shifted down)
        # ===================================================================
        bar = tk.Frame(self)
        bar.grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(0, 8),
        )
        bar.columnconfigure(1, weight=1)
        tk.Label(bar, text="LLM Command:").grid(row=0, column=0, sticky="w")
        self.cmd_var = tk.StringVar()
        entry_c = tk.Entry(bar, textvariable=self.cmd_var)
        entry_c.grid(row=0, column=1, sticky="ew")
        entry_c.bind("<Return>", lambda _e: self._send_llm_command())

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
        note.add(tab_actions, text="Valid Actions")

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

        # ---------------------------------------------------------------
        # _cmd_buttons already initialised above
        # ---------------------------------------------------------------

        # ── SCROLL CONTROLS BLOCK (step scroll + auto-scroll toggle) ----
        scroll_block = tk.Frame(btns)
        scroll_block.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # ── Left half: Dial-pad ---------------------------------------
        keypad = tk.Frame(scroll_block)
        keypad.grid(row=0, column=0, sticky="n")

        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
        self._dtmf_buttons = []  # recreate in new parent
        for i, d in enumerate(digits):
            r, c = divmod(i, 3)
            btn = ttk.Button(
                keypad,
                text=d,
                width=4,
                command=lambda digit=d: self._on_dtmf(digit),
            )
            btn.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
            self._dtmf_buttons.append(btn)

        for col in range(3):
            keypad.columnconfigure(col, weight=1)

        # ── Right half: step scroll + toggle ---------------------------

        self.scroll_px_var = tk.StringVar(value="100")

        step = tk.Frame(scroll_block)
        step.grid(row=0, column=1, sticky="n", padx=(8, 0))
        step.rowconfigure(1, weight=1)

        def _step_pixels() -> str:
            val = self.scroll_px_var.get().strip()
            return val if val.isdigit() and int(val) > 0 else "100"

        up_btn = ttk.Button(
            step,
            text="▲",
            width=4,
            command=lambda: self._handle_input(
                CMD_SCROLL_UP.replace("*", _step_pixels()),
            ),
        )
        up_btn.grid(row=0, column=0, sticky="ew")

        px_entry = tk.Entry(step, textvariable=self.scroll_px_var, width=6, justify="center")
        px_entry.grid(row=1, column=0, sticky="ew", pady=2)

        down_btn = ttk.Button(
            step,
            text="▼",
            width=4,
            command=lambda: self._handle_input(
                CMD_SCROLL_DOWN.replace("*", _step_pixels()),
            ),
        )
        down_btn.grid(row=2, column=0, sticky="ew")

        self._step_widgets = [up_btn, down_btn, px_entry]

        # --- Three-position auto-scroll toggle -------------------------
        toggle_frame = tk.Frame(scroll_block)
        toggle_frame.grid(row=0, column=2, sticky="n", padx=(8, 0))

        scroll_block.columnconfigure(0, weight=1)  # keypad
        scroll_block.columnconfigure(1, weight=1)  # step
        scroll_block.columnconfigure(2, weight=1)  # toggle

        # IntVar: 0 = up, 1 = stop, 2 = down  (middle is default)
        self._scroll_mode = tk.IntVar(value=1)
        # Track last non-stop direction to flip on successive blank-clicks
        self._last_scroll_dir: str | None = None  # 'up' or 'down'
        # Pending slider state awaiting browser confirmation (0/1/2)
        self._scroll_pending_target: int | None = None

        self._scroll_toggle_guard = False  # re-entrancy flag
        self._manual_stop_pending = False   # wait until worker confirms

        def _on_scroll_toggle(val):
            if self._scroll_toggle_guard:
                return  # ignore programmatic updates
            try:
                mode = int(float(val))
            except Exception:
                return
            if mode == 0:
                self._queue_command(CMD_START_SCROLL_UP)
                self._last_scroll_dir = "up"
            elif mode == 2:
                self._queue_command(CMD_START_SCROLL_DOWN)
                self._last_scroll_dir = "down"
            else:
                self._queue_command(CMD_STOP_SCROLLING)
                self._manual_stop_pending = True

            # Disable toggle until browser confirms
            self.scroll_toggle.configure(state="disabled")
            self._scroll_pending_target = mode

        # horizontal slider with 3 notches
        self.scroll_toggle = tk.Scale(
            toggle_frame,
            from_=0,
            to=2,
            orient="vertical",
            length=100,
            showvalue=False,
            variable=self._scroll_mode,
            command=_on_scroll_toggle,
        )
        self.scroll_toggle.grid(row=0, column=0, sticky="ew")

        # Label markers – placed to the right of the vertical slider
        lbls = tk.Frame(toggle_frame)
        lbls.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        for row_idx, txt in enumerate(["▲", "■", "▼"]):
            tk.Label(lbls, text=txt).grid(row=row_idx, column=0, sticky="n")
            lbls.rowconfigure(row_idx, weight=1)

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

    def _send_llm_command(self) -> None:
        text = self.cmd_var.get().strip()
        self.cmd_var.set("")
        if text:
            self._handle_input(text, from_llm_box=True)

    # ---------- search / url helper ------------------------------------
    def _send_search(self) -> None:
        txt = self.search_var.get().strip()
        if not txt:
            return
        mode = self.search_mode.get()
        cmd = (
            CMD_OPEN_URL.replace("*", txt)
            if mode == "url"
            else CMD_SEARCH.replace("*", txt)
        )
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
        self._handle_input(f"enter_text {decoded}")
        self.enter_var.set("")  # clear box

    # ──────────────────────── TABS‑PANE HELPERS ────────────────────────
    def _exec_tab_cmd(self, prefix: str, title: str) -> None:
        cmd = prefix.replace("*", title)
        self._log(f"> {cmd}")
        self._queue_command(cmd)

    def _rebuild_tabs_rows(self) -> None:
        """Re‑create the list of browser tabs with per‑row buttons."""
        self._tab_row_buttons: list[tk.Button] = []

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

            go_btn = Button(
                row,
                text="Go",
                command=lambda t=title: self._exec_tab_cmd(CMD_SELECT_TAB, t),
                padx=6,
                pady=2,
                relief="flat",
                font=btn_font,
            )
            go_btn.grid(row=0, column=1, sticky="e")

            close_btn = Button(
                row,
                text="×",
                command=lambda t=title: self._exec_tab_cmd(CMD_CLOSE_TAB, t),
                padx=4,
                pady=2,  # ← bump this slightly to avoid visual clipping
                relief="flat",
                font=btn_font,
            )
            close_btn.grid(row=0, column=2, padx=(0, 2), sticky="e")

            # keep references for enable/disable
            self._tab_row_buttons.extend([close_btn, go_btn])

            # Stretch row container to fill width
            self._tab_rows_frame.columnconfigure(0, weight=1)

    def _refresh_enabled_controls(self, valid):
        REASONS = {
            CMD_ENTER_TEXT: "Only available when the page caret is in a text‑box",
            CMD_PRESS_ENTER: "Requires focus in a text‑box",
            CMD_PRESS_BACKSPACE: "Requires focus in a text‑box",
            CMD_PRESS_DELETE: "Requires focus in a text‑box",
            CMD_CURSOR_LEFT: "Requires focus in a text‑box",
            CMD_CURSOR_RIGHT: "Requires focus in a text‑box",
            CMD_CURSOR_UP: "Requires focus in a text‑box",
            CMD_CURSOR_DOWN: "Requires focus in a text‑box",
            CMD_SELECT_ALL: "Requires focus in a text‑box",
            CMD_MOVE_LINE_START: "Requires focus in a text‑box",
            CMD_MOVE_LINE_END: "Requires focus in a text‑box",
            CMD_MOVE_WORD_LEFT: "Requires focus in a text‑box",
            CMD_MOVE_WORD_RIGHT: "Requires focus in a text‑box",
            CMD_STOP_SCROLLING: "Auto‑scroll isn't running",
            CMD_CONT_SCROLLING: "Auto‑scroll isn't running",
            CMD_START_SCROLL_UP: "Already auto‑scrolling",
            CMD_START_SCROLL_DOWN: "Already auto‑scrolling",
            CMD_SCROLL_UP: "Already at the very top of the page",
        }

        def _is_ok(cmd: str) -> bool:
            for v in valid:
                if cmd == v:
                    return True
                if v.endswith("*"):
                    prefix = v[:-1]
                    if cmd.startswith(prefix) or cmd == prefix.rstrip(" _"):
                        return True
            return False

        # ---------- key‑button rows ----------------------------------
        for cmd, btn in self._key_buttons.items():
            ok = _is_ok(cmd)
            btn.configure(state="normal" if ok else "disabled")
            if not ok:
                _Tooltip(btn, REASONS.get(cmd, "Not valid in current state"))

        # ---------- scroll / tab control buttons ---------------------
        for cmd, btn in self._cmd_buttons.items():
            ok = _is_ok(cmd)
            btn.configure(state="normal" if ok else "disabled")
            if not ok:
                _Tooltip(btn, REASONS.get(cmd, "Not valid in current state"))

        # ----- Enter‑text input -------------------------------------
        ok = _is_ok(CMD_ENTER_TEXT)
        self.enter_text_box.configure(state="normal" if ok else "disabled")
        if not ok:
            _Tooltip(
                self.enter_text_box,
                "Cannot type – there's no active text‑box on the page",
            )

        # ----- Search / URL entry -------------------------------------
        ok_search = _is_ok(CMD_SEARCH) or _is_ok(CMD_OPEN_URL)
        self.search_entry.configure(state="normal" if ok_search else "disabled")
        if not ok_search:
            _Tooltip(
                self.search_entry,
                "Disabled while typing in a page text‑box",
            )

        # ----- Numbered element buttons ---------------------------------
        can_click_el = _is_ok(CMD_CLICK_BUTTON)
        for btn in getattr(self, "_element_buttons", []):
            btn.configure(state="normal" if can_click_el else "disabled")
            if not can_click_el:
                _Tooltip(btn, "Cannot click elements while typing in a text‑box")

        # ----- Per‑row "×" / Go buttons in the Tabs pane --------------
        for btn in getattr(self, "_tab_row_buttons", []):
            if btn["text"] == "×":  # close‑tab buttons
                ok = any(name.startswith("close_tab_") for name in valid)
                reason = "Tab actions disabled while typing"
            else:  # Go buttons
                ok = any(name.startswith("select_tab_") for name in valid)
                reason = "Cannot switch tabs while typing"

            btn.configure(state="normal" if ok else "disabled")
            if not ok:
                _Tooltip(btn, reason)

        # ----- Step-scroll widgets ------------------------------------ NEW
        auto = self.state.get("auto_scroll", None)
        for w in getattr(self, "_step_widgets", []):
            w.configure(state="disabled" if auto else "normal")

    # ──────────────────────── ACTIONS‑PANE HELPER ───────────────────────
    def _refresh_actions_list(self) -> None:
        """Update the Actions tab (anti‑jitter, preserves scroll)."""
        groups = list_available_actions(
            self.tab_titles,
            [(idx, label) for idx, label, _ in self.elements],
            state=BrowserState(**self.state),
        )

        # ---------- build text with ONLY currently valid actions ----------
        valid = get_valid_actions(BrowserState(**self.state))

        def _is_ok(cmd: str) -> bool:
            for v in valid:
                if cmd == v:
                    return True
                if v.endswith("*"):
                    prefix = v[:-1]
                    if cmd.startswith(prefix) or cmd == prefix.rstrip(" _"):
                        return True
            return False

        out_lines: list[str] = []
        for grp, names in groups.items():
            kept = [n for n in names if _is_ok(n)]
            if not kept:
                continue
            out_lines.append(f"[{grp}]")
            for name in kept:
                display = name if len(name) <= 45 else name[:42] + "…"
                out_lines.append(f"  {display}")
            out_lines.append("")

        new_txt = "\n".join(out_lines)

        if new_txt == self._last_actions_txt:  # anti‑jitter cache
            return
        self._last_actions_txt = new_txt

        pos = self.act_box.yview()
        self.act_box.configure(state="normal")
        self.act_box.delete("1.0", "end")
        self.act_box.insert("end", new_txt)
        self.act_box.configure(state="disabled")

        if pos[0] > 0.01:
            self.act_box.yview_moveto(pos[0])

        # update button/key enablement
        self._refresh_enabled_controls(valid)

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

        # sync the auto-scroll toggle --------------------------------
        if hasattr(self, "scroll_toggle"):
            # If awaiting confirmation, re-enable when state matches target
            if self._scroll_pending_target is not None:
                mode_to_state = {0: "up", 1: None, 2: "down"}
                expected = mode_to_state[self._scroll_pending_target]
                if self.state.get("auto_scroll") == expected:
                    self.scroll_toggle.configure(state="normal")
                    self._scroll_pending_target = None
                    if self._manual_stop_pending and expected is None:
                        self._manual_stop_pending = False

    # ---------- element‑button helpers ---------------------------------
    def _exec_element_click(self, idx: int, label: str) -> None:
        self._log(f"> click {label}")
        self._queue_command(f"click {idx}")
        self._reset_el_scroll = True

    def _rebuild_elements_rows(self) -> None:
        """Refresh the scrollable button list for page elements."""
        # clear & reset the reference list
        self._element_buttons: list[ttk.Button] = []
        for c in self._elements_rows_frame.winfo_children():
            c.destroy()
        for idx, label, hover in self.elements:
            # flatten any embedded newlines to avoid tall buttons
            flat = " ".join(label.splitlines())
            shown = flat if len(flat) <= 25 else flat[:22] + "…"
            txt = f"{idx}. {shown}" + ("  (hover)" if hover else "")
            btn = ttk.Button(
                self._elements_rows_frame,
                text=txt,
                style="Element.TButton",
                command=lambda i=idx, l=label: self._exec_element_click(i, l),
            )
            btn.pack(fill="x", padx=1, pady=0)
            self._element_buttons.append(btn)
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
    def _handle_input(self, text: str, *, from_llm_box: bool = False) -> None:
        try:
            self._log(f"> {text}")

            def _is_valid_primitive(cmd: str) -> bool:
                valid = get_valid_actions(BrowserState(**self.state))
                for pat in valid:
                    if cmd == pat:
                        return True
                    if pat.endswith("*") and cmd.startswith(pat[:-1]):
                        return True  # parameterised match
                return False

            # skip the fast‑path when the text came from the LLM box
            if (not from_llm_box) and _is_valid_primitive(text):
                self._queue_command(text)
                return

            # hand off to background thread (non‑blocking)
            self._start_llm_thread(text)
        except Exception:
            import traceback as _tb

            self._log_trace(_tb.format_exc())

    def set_worker(self, worker):
        self._worker = worker

    # ───────────────────────── PUBLIC PRIMITIVE API ─────────────────────
    def send_text_command(self, text: str) -> None:
        self.event_generate("<<SendTextCommand>>", when="tail")
        self._pending_text = text

    # ───────────────────────── PICK LOW‑LEVEL CMD ───────────────────────
    def _llm_resp_to_cmd(self, resp: dict) -> str | None:
        # ToDo: fix all of this
        # ----- tab actions -------------------------------------------------
        for key, obj in resp.get("tab_actions", {}).items():
            if not obj.get("apply"):
                continue
            if key == "new_tab":
                return CMD_NEW_TAB
            if key.startswith("close_tab"):
                slug = key[len("close_tab ") :]
                return CMD_CLOSE_TAB.replace("*", slug)
            if key.startswith("select_tab"):
                slug = key[len("select_tab") :]
                return CMD_SELECT_TAB.replace("*", slug)

        # ----- scroll actions ---------------------------------------------
        sc = resp.get("scroll_actions", {})
        if sc.get("scroll_up", {}).get("apply"):
            px = sc["scroll_up"].get("pixels") or 300
            return CMD_SCROLL_UP.replace("*", str(px))
        if sc.get("scroll_down", {}).get("apply"):
            px = sc["scroll_down"].get("pixels") or 300
            return CMD_SCROLL_DOWN.replace("*", str(px))
        if sc.get("start_scrolling_up", {}).get("apply"):
            return CMD_START_SCROLL_UP
        if sc.get("start_scrolling_down", {}).get("apply"):
            return CMD_START_SCROLL_DOWN
        if sc.get("stop_scrolling", {}).get("apply"):
            return CMD_STOP_SCROLLING
        if sc.get("continue_scrolling", {}).get("apply"):
            return CMD_CONT_SCROLLING

        # ----- button actions ---------------------------------------------
        slug_to_idx = {}
        for idx, label, _ in self.elements:
            base = _slug(label)
            slug_to_idx[base] = idx
            slug_to_idx[f"{idx}_{base}"] = idx
        for key, obj in resp.get("button_actions", {}).items():
            if not obj.get("apply") or not key.startswith("click_button_"):
                continue
            slug_text = key[len("click_button ") :]
            if slug_text in slug_to_idx:
                return CMD_CLICK_BUTTON.replace("*", slug_text)
            return CMD_CLICK_BUTTON.replace("*", slug_text.replace("_", " "))

        # ----- search / open‑url ------------------------------------------
        sa = resp.get("search")
        if sa and sa.get("apply"):
            return CMD_SEARCH.replace("*", sa.get("query", ""))
        sua = resp.get("open_url")
        if sua and sua.get("apply"):
            return CMD_OPEN_URL.replace("*", sua.get("url", "").strip())

        # ----- textbox actions ------------------------------------------ NEW
        tb = resp.get("textbox_actions", {})
        if tb:
            # 1. enter_text *
            et = tb.get("enter_text")
            if et and et.get("apply"):
                return CMD_ENTER_TEXT.replace("*", et.get("text", ""))

            # 2. single‑key / caret actions
            for cmd in (
                CMD_PRESS_ENTER,
                CMD_PRESS_BACKSPACE,
                CMD_PRESS_DELETE,
                CMD_CURSOR_LEFT,
                CMD_CURSOR_RIGHT,
                CMD_CURSOR_UP,
                CMD_CURSOR_DOWN,
                CMD_SELECT_ALL,
                CMD_MOVE_LINE_START,
                CMD_MOVE_LINE_END,
                CMD_MOVE_WORD_LEFT,
                CMD_MOVE_WORD_RIGHT,
                CMD_HOLD_SHIFT,
                CMD_RELEASE_SHIFT,
                CMD_CLICK_OUT,
            ):
                fld = tb.get(cmd)
                if fld and fld.get("apply"):
                    return cmd
        return None

    # ───────────────────────── COMMAND QUEUE ────────────────────────────
    def _queue_command(self, cmd: str) -> None:
        # mark commands that likely change the page content  ------------- NEW
        nav_prefixes = (
            CMD_OPEN_URL.rstrip("*"),
            CMD_SEARCH.rstrip("*"),
            CMD_NEW_TAB,
            CMD_SELECT_TAB.rstrip("*"),
        )
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
            payload = self._pull_update()
            if payload is None:
                break
            # ensure dict
            if not isinstance(payload, dict):
                continue
            self.elements = payload.get("elements", [])
            self.tab_titles = payload.get("tabs", [])
            self.history = payload.get("history", self.history)
            self.state = payload.get("state", self.state)
            img = payload.get("screenshot", b"")
            if img:
                self.screenshot = img
            updated = True

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
            self._log(ansi.rstrip())
        except Exception:
            self._log(tb)

    # ─────────────────────── number-key helper ──────────────────────── NEW
    def _on_dtmf(self, digit: str) -> None:
        """Handle dial-pad press – send DTMF and click numbered element when 1-9."""
        # Skip when an Entry widget currently has focus to avoid hijacking typing
        if isinstance(self.focus_get(), tk.Entry):
            return

        self._log(f"[dtmf] {digit}")

        if _dtmf_publisher is not None:
            try:
                ret = _dtmf_publisher(digit)
                if asyncio.iscoroutine(ret):
                    asyncio.create_task(ret)
            except Exception as exc:
                self._log(f"⚠ DTMF publish error: {exc}")

# Optional: external LiveKit DTMF publisher callback
_dtmf_publisher: "Callable[[int], None] | None" = None

def register_dtmf_publisher(fn):
    """Register a callback that sends DTMF for the given digit (1-9).

    The *fn* can be synchronous or async; if it returns a coroutine it will
    be scheduled on the current running loop.
    """
    global _dtmf_publisher
    _dtmf_publisher = fn
