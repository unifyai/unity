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
        self.history: list[dict] = []
        self.state: dict[str, Any] = {}

        self._build_widgets()
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

        self.bind(
            "<<SendTextCommand>>",
            lambda _e: self._handle_input(self._pending_text),
        )
        self._pending_text = ""

        self.after(50, self._poll_text_q)

    def _on_tab_go(self, _e=None):  # NEW
        sel = self.tabs_listbox.curselection()
        if not sel:
            return
        title = self.tabs_listbox.get(sel[0])
        self._log(f"> switch to tab {title}")
        self._queue_command(f"switch to tab {title}")

    def _on_tab_close(self, _e=None):  # NEW
        sel = self.tabs_listbox.curselection()
        if not sel:
            return
        title = self.tabs_listbox.get(sel[0])
        self._log(f"> close tab {title}")
        self._queue_command(f"close tab {title}")

    def _refresh_actions_list(self) -> None:  # NEW
        """Update the Actions tab with every LLM action currently possible."""
        groups = list_available_actions(
            self.tab_titles,
            [(idx, label) for idx, label, _ in self.elements],
        )
        lines: list[str] = []
        for grp, names in groups.items():
            lines.append(f"[{grp}]")
            lines.extend(f"  {n}" for n in names)
            lines.append("")
        txt = "\n".join(lines)

        self.act_box.configure(state="normal")
        self.act_box.delete("1.0", "end")
        self.act_box.insert("end", txt)
        self.act_box.configure(state="disabled")

    def _refresh_state_label(self) -> None:
        st = self.state or {}
        self.state_var.set(
            f"url:         {st.get('url', '')[:60]}\n"
            f"title:       {st.get('title', '')[:60]}\n"
            f"scroll_y:    {st.get('scroll_y', 0)}\n"
            f"auto_scroll: {st.get('auto_scroll', None)}\n"
            f"in_textbox:  {st.get('in_textbox', False)}",
        )

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        """Build the Tk layout (GUI thread)."""

        # ---- main window ------------------------------------------------------
        self.title("Playwright helper")
        self.geometry("900x550")

        # root grid: two columns (left notebook, right panel)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        # ======================================================================
        # LEFT NOTEBOOK  →  Elements  |  Tabs
        # ======================================================================
        left_nb = ttk.Notebook(self)  # NEW
        left_nb.grid(row=0, column=0, sticky="nsew")  # NEW

        # ----- Frames for each tab -------------------------------------------
        tab_elements_frame = ttk.Frame(left_nb)  # NEW
        tab_tabs_frame = ttk.Frame(left_nb)  # NEW
        left_nb.add(tab_elements_frame, text="Elements")  # NEW
        left_nb.add(tab_tabs_frame, text="Tabs")  # NEW

        # make both frames expand to fill the cell
        for f in (tab_elements_frame, tab_tabs_frame):  # NEW
            f.rowconfigure(0, weight=1)  # NEW
            f.columnconfigure(0, weight=1)

        tab_tabs_frame.rowconfigure(1, weight=1)  # NEW

        # ------------------------------------------------------------------
        # ELEMENTS LISTBOX  (inside tab_elements_frame)
        # ------------------------------------------------------------------
        self.listbox = tk.Listbox(tab_elements_frame, font=("Helvetica", 11))  # NEW
        self.listbox.grid(row=0, column=0, sticky="nsew")  # NEW
        sb_el = ttk.Scrollbar(  # NEW
            tab_elements_frame,
            orient="vertical",
            command=self.listbox.yview,
        )
        sb_el.grid(row=0, column=0, sticky="nse")  # NEW
        self.listbox.config(yscrollcommand=sb_el.set)  # NEW
        self.listbox.bind("<ButtonRelease-1>", self._on_list_click)  # NEW
        self.listbox.bind("<Return>", self._on_list_click)  # NEW

        # ------------------------------------------------------------------
        # TABS LISTBOX  (inside tab_tabs_frame) + action buttons
        # ------------------------------------------------------------------
        tab_ctrl = tk.Frame(tab_tabs_frame)  # NEW
        tab_ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 4))  # NEW
        tab_ctrl.columnconfigure(0, weight=1)  # NEW
        tab_ctrl.columnconfigure(1, weight=1)  # NEW
        ttk.Button(tab_ctrl, text="Go to", command=self._on_tab_go).grid(  # NEW
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(tab_ctrl, text="Close", command=self._on_tab_close).grid(  # NEW
            row=0,
            column=1,
            sticky="ew",
        )

        self.tabs_listbox = tk.Listbox(tab_tabs_frame, font=("Helvetica", 11))  # NEW
        self.tabs_listbox.grid(row=1, column=0, sticky="nsew")  # NEW
        sb_tab = ttk.Scrollbar(  # NEW
            tab_tabs_frame,
            orient="vertical",
            command=self.tabs_listbox.yview,
        )
        sb_tab.grid(row=1, column=0, sticky="nse")  # NEW
        self.tabs_listbox.config(yscrollcommand=sb_tab.set)  # NEW
        self.tabs_listbox.bind("<ButtonRelease-1>", self._on_tab_go)  # NEW
        self.tabs_listbox.bind("<Return>", self._on_tab_go)  # NEW

        # ======================================================================
        # RIGHT‑HAND PANEL  →  Log / Actions notebook + browser state + buttons
        # (unchanged from earlier snippets)
        # ======================================================================
        right = tk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)  # notebook (Log/Actions)
        right.rowconfigure(1, weight=0)  # browser‑state read‑out
        right.rowconfigure(2, weight=1)  # control buttons

        # -- Log / Actions notebook -------------------------------------------
        note = ttk.Notebook(right)
        note.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tab_log = ttk.Frame(note)
        tab_actions = ttk.Frame(note)
        note.add(tab_log, text="Log")
        note.add(tab_actions, text="Actions")

        # log window
        self.log = scrolledtext.ScrolledText(tab_log, state="disabled")
        self.log.pack(fill="both", expand=True)

        # available actions window
        self.act_box = scrolledtext.ScrolledText(tab_actions, state="disabled")  # NEW
        self.act_box.pack(fill="both", expand=True)  # NEW

        # -- browser‑state label ---------------------------------------------
        self.state_var = tk.StringVar()
        self.state_lbl = tk.Label(
            right,
            textvariable=self.state_var,
            justify="left",
            anchor="w",
            font=("Consolas", 9),
        )
        self.state_lbl.grid(row=1, column=0, sticky="nsew", padx=5)

        # -- control buttons --------------------------------------------------
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

    # ---------------------------------------------------------------- exit
    def _on_exit(self) -> None:
        """Gracefully close the GUI; `mainloop()` will return to main.py."""
        self.quit()

    # ---------------------------------------------------------------- events
    def _on_list_click(self, _e: Any) -> None:
        sel = self.listbox.curselection()
        if sel:
            pos = sel[0]
            idx = pos + 1  # 1‑based for the worker
            label = self.elements[pos][1]  # (idx, label, hover)

            # 1) show friendly text in the GUI log
            self._log(f"> click {label}")

            # 2) send numeric command to the worker
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
            cmd = f"open url {txt}"
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

        # free English → LLM  (catch & show full trace) ------------------
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
            return f"open url {raw}"

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
                self.history = payload.get("history", self.history)
                self.state = payload.get("state", self.state)
                self._refresh_state_label()
                img = payload.get("screenshot", b"")
                if img:
                    self.screenshot = img
                updated = True
            except queue.Empty:
                break

        if updated:
            # ---------- elements list (left / Elements tab) -------------
            self.listbox.delete(0, "end")
            for idx, label, hover in self.elements:
                self.listbox.insert(
                    "end",
                    f"{idx:>2}. {label}" + (" (on hover)" if hover else ""),
                )

            # ---------- browser‑tabs list (left / Tabs tab) -------------  # NEW
            self.tabs_listbox.delete(0, "end")  # NEW
            for title in self.tab_titles:  # NEW
                self.tabs_listbox.insert("end", title)  # NEW

            # ---------- update Actions pane -----------------------------  # NEW
            self._refresh_actions_list()  # NEW

        self._refresh_state_label()  # keep state label fresh
        self.after(self.REFRESH_INTERVAL_MS, self._poll_updates)

    def _poll_text_q(self):
        while True:
            try:
                txt = self.text_q.get_nowait()
            except queue.Empty:
                break
            self._handle_input(txt)  # reuse existing logic
        self.after(50, self._poll_text_q)
        self._refresh_state_label()  # NEW – show blank state at startup
        self._refresh_actions_list()  # NEW – populate “Actions” pane

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
