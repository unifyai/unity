# -*- coding: utf-8 -*-
"""
Tk-based front-end.

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
import logging
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pydantic import BaseModel, Field

from .agent import (
    text_to_browser_action,
    list_available_actions,
    ADVANCED_MODE,
    ask_llm,
)
from .states import BrowserState
from .commands import *
from .action_filter import get_valid_actions
from .helpers import _slug


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
        tw_label = tk.Label(
            tw,
            text=self.text,
            bg=bg,
            fg=_contrasting(bg),
            relief="solid",
            borderwidth=1,
            font=("tahoma", 8),
        )
        tw_label.pack()

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
        redis_db: int = 0,
    ):
        super().__init__()
        self.cmd_q = command_q  # GUI → worker
        self.up_q = update_q  # worker → GUI (may be None when using Redis)
        self.text_q = text_q or queue.Queue()  # primitive → GUI
        self.elements: list[tuple[int, str, bool]] = []
        self.screenshot: bytes = b""
        self.tab_titles: list[str] = []
        self.history: list[dict] = []
        self.state: BrowserState = BrowserState()
        self._redis_db = redis_db

        # ── async LLM helper --------------------------------------------
        self._llm_resp_q: "queue.Queue[tuple[str,Any]]" = queue.Queue()
        self._llm_stream_q: "queue.Queue[str]" = queue.Queue()
        self._llm_busy = False
        self._llm_dots = itertools.cycle([".", "..", "..."])
        self._llm_line_idx = None
        self._llm_mode = None

        # for graying out when not in textbox
        self._key_buttons = {}

        # Auto-scroll state management variables
        self._scroll_mode = tk.IntVar(value=1)  # 0=up, 1=stop, 2=down
        self._last_scroll_dir: str | None = None  # 'up' or 'down'
        self._scroll_pending_target: int | None = None
        self._scroll_toggle_guard = False  # re-entrancy flag
        self._manual_stop_pending = False  # wait until worker confirms

        # Layout management
        self._reset_el_scroll = False

        self._build_widgets()

        self._worker = None  # will be set by set_worker()
        self._controller = None  # will be set by set_controller()

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

        # model poller - only poll llm_resp when no controller (backward compatibility)
        if self._controller is None:
            self.after(50, self._poll_llm_resp)
        self.after(50, self._poll_llm_stream)

        # If no update queue provided, subscribe to Redis browser_state
        if self.up_q is None:
            import redis, ast, json

            self._redis_pub = redis.Redis(host="localhost", port=6379, db=0).pubsub()
            self._redis_pub.subscribe(f"browser_state_{self._redis_db}")

            def _get_update_from_redis():
                msg = self._redis_pub.get_message()
                if msg and msg["type"] == "message":
                    data = msg["data"]
                    try:
                        return json.loads(data)
                    except Exception:
                        try:
                            return ast.literal_eval(
                                data.decode() if isinstance(data, bytes) else data,
                            )
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

        # ── top-centre CAPTCHA label (hidden) ───────────────────────────── NEW
        self.captcha_lbl = tk.Label(
            self,
            text="🔒 Solving CAPTCHA…",
            fg="orange red",
            font=("Helvetica", 10, "bold"),
            bg=self.cget("bg"),
        )
        # place it off-screen initially; we'll .place(...) when active
        self.captcha_lbl.place_forget()

    # ─────────────────────────────────────────

    def _advance_llm_dots(self):
        if not self._llm_busy or self._llm_line_idx is None:
            return  # stop when the model reply arrives

        # update the existing line in‑place
        self.log.configure(state="normal")
        self.log.delete(self._llm_line_idx, f"{self._llm_line_idx} lineend")

        # Check current line content to determine if observation or action
        current_line = self.log.get(self._llm_line_idx, f"{self._llm_line_idx} lineend")
        if "observing" in current_line:
            new_txt = "🔍 observing" + next(self._llm_dots)
        else:
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

        # also mirror LLM-related lines into stream tab
        if tag == "llm" and hasattr(self, "llm_stream_box"):
            self.llm_stream_box.configure(state="normal")
            self.llm_stream_box.insert("end", text + "\n")
            self.llm_stream_box.configure(state="disabled")
            self.llm_stream_box.yview_moveto(1.0)
        return idx

    # ─────────────────── background LLM thread ──────────────────────
    def _start_llm_thread(
        self,
        user_text: str,
        *,
        is_observation: bool | None = None,
    ) -> None:
        """
        Fire a daemon thread that calls controller.act/observe if available,
        or falls back to direct calls. Only uses _llm_resp_q for backward
        compatibility when controller is not provided.
        """
        if self._llm_busy:
            self._log("⚠ LLM still working – please wait")
            return
        self._llm_busy = True

        # Disable entry & show loader icon
        self.llm_entry.configure(state="disabled")
        self.llm_loader.grid()

        # Determine observation vs action.
        if is_observation is None:
            is_observation = user_text.strip().startswith(("observe:", "?"))

        observation_text = user_text
        if is_observation:
            # Strip any legacy prefixes, if present
            if user_text.strip().startswith("observe:"):
                observation_text = user_text.strip()[8:].strip()
            elif user_text.strip().startswith("?"):
                observation_text = user_text.strip()[1:].strip()

        # --- spawn animated log line ------------------------------------
        if is_observation:
            msg = "🔍 observing" + next(self._llm_dots)
        else:
            msg = "⏳ calling model" + next(self._llm_dots)
        self._llm_line_idx = self._log_line(msg, tag="llm")
        self.after(400, self._advance_llm_dots)

        # snapshot everything the model needs
        screenshot = self.screenshot
        tabs = list(self.tab_titles)
        buttons = [(i, lbl) for i, lbl, _ in self.elements]
        history = list(self.history)
        state = self.state

        # Get expectation and multi_step_mode for act mode
        expectation = None
        multi_step_mode = False
        if not is_observation:
            if hasattr(self, "expectation_var"):
                expectation_text = self.expectation_var.get().strip()
                expectation = expectation_text if expectation_text else None
            if hasattr(self, "multi_step_var"):
                multi_step_mode = self.multi_step_var.get()

        def _worker():
            """Run the LLM call inside a thread and forward *all* log records
            emitted by this thread into the GUI-stream queue."""

            # --- queue log handler ---------------------------------
            class _QueueHandler(logging.Handler):
                def __init__(self, q):
                    super().__init__()
                    self.q = q

                def emit(self, record):
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = record.getMessage()
                    # Ensure we never block the worker – drop on full
                    try:
                        self.q.put_nowait(msg)
                    except queue.Full:
                        pass

            qh = _QueueHandler(self._llm_stream_q)
            qh.setLevel(logging.DEBUG)
            qh.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(threadName)s  %(levelname)-8s  %(message)s",
                ),
            )

            root_logger = logging.getLogger()
            root_logger.addHandler(qh)

            # Be verbose for HTTP + model calls
            logging.getLogger("urllib3").setLevel(logging.DEBUG)

            if is_observation:
                self._llm_stream_q.put(
                    f"🔍 Observation start › {observation_text[:60]}",
                )
            else:
                self._llm_stream_q.put(f"🛈 LLM call start › {user_text[:60]}")

            try:
                if is_observation:
                    # Use Controller.observe method for proper context management
                    if self._controller:
                        import asyncio

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                self._controller.observe(
                                    observation_text,
                                    response_format=str,
                                ),
                            )
                        finally:
                            loop.close()
                        self._llm_stream_q.put("🔍 Observation succeeded")

                        # Log observation result to GUI (schedule on main thread)
                        def log_observation_result():
                            # Clear animated line
                            if self._llm_line_idx is not None:
                                self.log.configure(state="normal")
                                self.log.delete(
                                    self._llm_line_idx,
                                    f"{self._llm_line_idx} lineend",
                                )
                                self.log.configure(state="disabled")
                                self._llm_line_idx = None

                            # Display the observation result
                            self.log.configure(state="normal")
                            self.log.insert(
                                "end",
                                "🔍 Observation Result:\n",
                                "observation",
                            )
                            # Format the result nicely
                            result_text = str(result)
                            # Indent multi-line results
                            if "\n" in result_text:
                                lines = result_text.split("\n")
                                formatted_result = "\n".join(
                                    f"   {line}" for line in lines
                                )
                                self.log.insert("end", formatted_result + "\n")
                            else:
                                self.log.insert("end", f"   {result_text}\n")
                            self.log.configure(state="disabled")
                            self.log.yview_moveto(1.0)

                        self.after_idle(log_observation_result)
                    else:
                        # Fallback to direct call if controller not available
                        context = {
                            "state": state,
                            "elements": buttons,
                            "tabs": tabs,
                        }
                        result = ask_llm(
                            observation_text,
                            response_format=str,
                            context=context,
                            screenshots={"current_view": screenshot},
                        )
                        self._llm_stream_q.put("🔍 Observation succeeded")
                        # Package the observation result for backward compatibility
                        self._llm_resp_q.put(("observation", result))
                else:
                    # Use Controller.act method for proper context and history management
                    if self._controller:
                        import asyncio

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                self._controller.act(
                                    user_text,
                                    expectation=expectation,
                                    multi_step_mode=multi_step_mode,
                                ),
                            )
                        finally:
                            loop.close()
                        self._llm_stream_q.put("🛈 LLM call succeeded")

                        # Log action result to GUI (schedule on main thread)
                        def log_action_result():
                            # Clear animated line
                            if self._llm_line_idx is not None:
                                self.log.configure(state="normal")
                                self.log.delete(
                                    self._llm_line_idx,
                                    f"{self._llm_line_idx} lineend",
                                )
                                self.log.configure(state="disabled")
                                self._llm_line_idx = None

                            # Format and display action result
                            # Controller.act returns str or list[str]
                            if isinstance(result, str):
                                cmds = [result] if result else []
                            elif isinstance(result, list):
                                cmds = result
                            else:
                                cmds = []

                            # Process each command in sequence (similar to legacy path)
                            for cmd in cmds:
                                if cmd and cmd.startswith("click_button_"):
                                    idx = cmd[len("click_button_") :].split("_", 1)[0]
                                    cmd = f"click {idx}"
                                line = f"↳ {cmd}" if cmd else "❗ No action selected"
                                # log and queue
                                self.log.configure(state="normal")
                                self.log.insert("end", line + "\n")
                                self.log.configure(state="disabled")
                                self.log.yview_moveto(1.0)

                        self.after_idle(log_action_result)
                    else:
                        # Fallback to direct call if controller not available
                        resp = text_to_browser_action(
                            user_text,
                            screenshot,
                            tabs=tabs,
                            buttons=buttons,
                            history=history,
                            state=state,
                            multi_step_mode=multi_step_mode,
                        )
                        self._llm_stream_q.put("🛈 LLM call succeeded")
                        # Package the response for backward compatibility
                        self._llm_resp_q.put(("ok", resp))
            except Exception as exc:
                self._llm_stream_q.put(
                    f"❌ LLM error: {exc.__class__.__name__}: {exc}",
                )
                import traceback as _tb

                if self._controller:
                    # Log error to GUI when using controller (schedule on main thread)
                    def log_error():
                        # Replace animated line with error
                        if self._llm_line_idx is not None:
                            self.log.configure(state="normal")
                            self.log.delete(
                                self._llm_line_idx,
                                f"{self._llm_line_idx} lineend",
                            )
                            self.log.insert(
                                self._llm_line_idx,
                                "❗ LLM error – see traceback\n",
                            )
                            self.log.configure(state="disabled")
                            self._llm_line_idx = None

                        # Log the full traceback
                        self._log_trace(_tb.format_exc())

                    self.after_idle(log_error)
                else:
                    # Only use _llm_resp_q for backward compatibility when no controller
                    self._llm_resp_q.put(("err", _tb.format_exc()))
            finally:
                # detach handler to avoid duplicate logs on next call
                root_logger.removeHandler(qh)

                # Reset UI state when using controller (since we don't use _llm_resp_q)
                if self._controller:

                    def reset_ui():
                        self._llm_busy = False
                        self.llm_entry.configure(state="normal")
                        self.llm_loader.grid_remove()
                        # Clear animated line
                        if self._llm_line_idx is not None:
                            self.log.configure(state="normal")
                            self.log.delete(
                                self._llm_line_idx,
                                f"{self._llm_line_idx} lineend",
                            )
                            self.log.configure(state="disabled")
                            self._llm_line_idx = None

                    # Schedule UI reset on main thread
                    self.after_idle(reset_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def _poll_llm_resp(self):
        """Check the response queue every 50 ms. Only used when controller is not provided."""
        # Only poll when controller is not provided (backward compatibility)
        if self._controller:
            return

        try:
            while True:
                status, payload = self._llm_resp_q.get_nowait()

                if status == "observation":
                    # Handle observation results
                    # Clear animated line
                    if self._llm_line_idx is not None:
                        self.log.configure(state="normal")
                        self.log.delete(
                            self._llm_line_idx,
                            f"{self._llm_line_idx} lineend",
                        )
                        self.log.configure(state="disabled")
                        self._llm_line_idx = None

                    # Display the observation result
                    self.log.configure(state="normal")
                    self.log.insert("end", "🔍 Observation Result:\n", "observation")
                    # Format the result nicely
                    result_text = str(payload)
                    # Indent multi-line results
                    if "\n" in result_text:
                        lines = result_text.split("\n")
                        formatted_result = "\n".join(f"   {line}" for line in lines)
                        self.log.insert("end", formatted_result + "\n")
                    else:
                        self.log.insert("end", f"   {result_text}\n")
                    self.log.configure(state="disabled")
                    self.log.yview_moveto(1.0)

                elif status == "ok":
                    # Determine command list: supports direct list or dict with 'action'
                    if isinstance(payload, list):
                        cmds = payload
                    elif isinstance(payload, dict) and "action" in payload:
                        cmds = payload["action"]
                    else:
                        # Legacy single-command path
                        cmd = (
                            payload["action"]
                            if (not ADVANCED_MODE and isinstance(payload, dict))
                            else self._llm_resp_to_cmd(payload)
                        )
                        cmds = [cmd] if cmd else []
                    # Clear animated line once
                    if self._llm_line_idx is not None:
                        self.log.configure(state="normal")
                        self.log.delete(
                            self._llm_line_idx,
                            f"{self._llm_line_idx} lineend",
                        )
                        self.log.configure(state="disabled")
                        self._llm_line_idx = None
                    # Process each command in sequence
                    for cmd in cmds:
                        if cmd and cmd.startswith("click_button_"):
                            idx = cmd[len("click_button_") :].split("_", 1)[0]
                            cmd = f"click {idx}"
                        line = f"↳ {cmd}" if cmd else "❗ No action selected"
                        # log and queue
                        self.log.configure(state="normal")
                        self.log.insert("end", line + "\n")
                        self.log.configure(state="disabled")
                        self.log.yview_moveto(1.0)
                        if cmd:
                            self._queue_command(cmd)
                else:  # "err"
                    self._log_trace(payload)
                    # Replace animated line with error
                    if self._llm_line_idx is not None:
                        self.log.configure(state="normal")
                        self.log.delete(
                            self._llm_line_idx,
                            f"{self._llm_line_idx} lineend",
                        )
                        self.log.insert(
                            self._llm_line_idx,
                            "❗ LLM error – see traceback\n",
                        )
                        self.log.configure(state="disabled")
                        self._llm_line_idx = None
                # Mark LLM as done and reset UI
                self._llm_busy = False
                self.llm_entry.configure(state="normal")
                self.llm_loader.grid_remove()

        except queue.Empty:
            pass
        self.after(50, self._poll_llm_resp)

    # -------------------- LLM stream poller -------------------------
    def _poll_llm_stream(self):
        try:
            while True:
                line = self._llm_stream_q.get_nowait()
                self.llm_stream_box.configure(state="normal")
                self.llm_stream_box.insert("end", line + "\n")
                self.llm_stream_box.configure(state="disabled")
                self.llm_stream_box.yview_moveto(1.0)
        except queue.Empty:
            pass
        self.after(50, self._poll_llm_stream)

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

    # ---------- LLM mode handling helpers ----------------------------------
    def _on_mode_change(self):
        """Called when Act/Observe mode is changed."""
        self._update_act_mode_widgets_visibility()

    def _update_act_mode_widgets_visibility(self):
        """Show/hide Act-mode specific widgets (expectation field and multi-step checkbox) based on current mode."""
        mode = self._llm_mode.get() if self._llm_mode else "act"
        if mode == "act":
            # Show expectation field
            if hasattr(self, "expectation_label"):
                self.expectation_label.grid()
                self.expectation_entry.grid()
            # Show multi-step checkbox
            if hasattr(self, "multi_step_checkbox"):
                self.multi_step_checkbox.grid()
        else:
            # Hide expectation field
            if hasattr(self, "expectation_label"):
                self.expectation_label.grid_remove()
                self.expectation_entry.grid_remove()
            # Hide multi-step checkbox
            if hasattr(self, "multi_step_checkbox"):
                self.multi_step_checkbox.grid_remove()

    # ──────────────────────── WIDGET CONSTRUCTION ────────────────────────
    def _build_widgets(self) -> None:
        """Build the full Tk layout (GUI thread)."""

        # ----- main window -------------------------------------------------
        self.title("Unity Browser Controller Sandbox")
        self.geometry("1400x800")  # Wider to accommodate new controls

        # Main horizontal paned window
        main_paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4)
        main_paned.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # ── top‑right "X" button (absolute) ─────────────────────────────────
        close_btn = ttk.Button(
            self,
            text="×",
            width=3,
            style="Danger.TButton",
            command=self._on_exit,
        )
        close_btn.place(relx=1.0, rely=0.0, x=-6, y=6, anchor="ne")
        self.after_idle(close_btn.lift)

        # ===================================================================
        # LEFT PANEL →  Elements & Tabs (existing)
        # ===================================================================
        self.left_wrapper = tk.Frame(main_paned, width=300)
        self.left_wrapper.pack_propagate(False)
        main_paned.add(self.left_wrapper, minsize=200, stretch="never")

        left_nb = ttk.Notebook(self.left_wrapper)
        left_nb.pack(fill="both", expand=True)

        # Elements and Tabs frames (existing implementation)
        tab_elements_frame = ttk.Frame(left_nb)
        tab_tabs_frame = ttk.Frame(left_nb)
        left_nb.add(tab_elements_frame, text="Elements")
        left_nb.add(tab_tabs_frame, text="Tabs")

        for f in (tab_elements_frame, tab_tabs_frame):
            f.rowconfigure(0, weight=1)
            f.columnconfigure(0, weight=1)

        # Elements pane setup (existing)
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
            lambda e: el_canvas.configure(scrollregion=el_canvas.bbox("all")),
        )
        el_canvas.grid(row=0, column=0, sticky="nsew")
        el_scroll.grid(row=0, column=1, sticky="ns")
        tab_elements_frame.rowconfigure(0, weight=1)
        tab_elements_frame.columnconfigure(0, weight=1)
        self._elements_rows_frame = el_rows
        self._bind_mousewheel(el_canvas, el_canvas)
        self._bind_mousewheel(el_rows, el_canvas)

        # Tabs pane setup (existing)
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

        # ===================================================================
        # CENTER PANEL →  Command Controls (NEW ORGANIZED LAYOUT)
        # ===================================================================
        center_frame = tk.Frame(main_paned, width=500)
        center_frame.pack_propagate(False)
        main_paned.add(center_frame, minsize=400, stretch="always")

        # Scrollable area for controls
        controls_canvas = tk.Canvas(center_frame, highlightthickness=0)
        controls_scroll = ttk.Scrollbar(
            center_frame,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_frame = ttk.Frame(controls_canvas)

        controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_frame.bind(
            "<Configure>",
            lambda e: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all"),
            ),
        )

        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scroll.pack(side="right", fill="y")

        self._bind_mousewheel(controls_canvas, controls_canvas)
        self._bind_mousewheel(controls_frame, controls_canvas)

        # Initialize button registry
        self._key_buttons = {}
        self._cmd_buttons = {}

        # ===================================================================
        # SECTION 1: Browser Lifecycle
        # ===================================================================
        browser_frame = ttk.LabelFrame(
            controls_frame,
            text="Browser Control",
            padding="10",
        )
        browser_frame.pack(fill="x", padx=5, pady=5)
        browser_frame.columnconfigure(0, weight=1)
        browser_frame.columnconfigure(1, weight=1)

        open_browser_btn = ttk.Button(
            browser_frame,
            text="Open Browser",
            command=lambda: self._handle_input(CMD_OPEN_BROWSER),
        )
        open_browser_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self._cmd_buttons[CMD_OPEN_BROWSER] = open_browser_btn

        close_browser_btn = ttk.Button(
            browser_frame,
            text="Close Browser",
            command=lambda: self._handle_input(CMD_CLOSE_BROWSER),
        )
        close_browser_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self._cmd_buttons[CMD_CLOSE_BROWSER] = close_browser_btn

        # ===================================================================
        # SECTION 2: Navigation & URL
        # ===================================================================
        nav_frame = ttk.LabelFrame(
            controls_frame,
            text="Navigation & URL",
            padding="10",
        )
        nav_frame.pack(fill="x", padx=5, pady=5)
        nav_frame.columnconfigure(1, weight=1)

        # URL/Search row
        tk.Label(nav_frame, text="URL/Search:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(nav_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=2)
        self.search_entry.bind("<Return>", lambda _e: self._send_search())

        # Search mode selection
        self.search_mode = tk.StringVar(value="google")
        rb_google = tk.Radiobutton(
            nav_frame,
            text="Google",
            variable=self.search_mode,
            value="google",
        )
        rb_url = tk.Radiobutton(
            nav_frame,
            text="URL",
            variable=self.search_mode,
            value="url",
        )
        rb_google.grid(row=0, column=2, padx=2)
        rb_url.grid(row=0, column=3, padx=2)

        # Navigation buttons
        nav_buttons_frame = tk.Frame(nav_frame)
        nav_buttons_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        nav_buttons_frame.columnconfigure(0, weight=1)
        nav_buttons_frame.columnconfigure(1, weight=1)
        nav_buttons_frame.columnconfigure(2, weight=1)

        back_btn = ttk.Button(
            nav_buttons_frame,
            text="← Back",
            command=lambda: self._handle_input(CMD_BACK_NAV),
        )
        back_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self._cmd_buttons[CMD_BACK_NAV] = back_btn

        forward_btn = ttk.Button(
            nav_buttons_frame,
            text="Forward →",
            command=lambda: self._handle_input(CMD_FORWARD_NAV),
        )
        forward_btn.grid(row=0, column=1, sticky="ew", padx=2)
        self._cmd_buttons[CMD_FORWARD_NAV] = forward_btn

        reload_btn = ttk.Button(
            nav_buttons_frame,
            text="🔄 Reload",
            command=lambda: self._handle_input(CMD_RELOAD_PAGE),
        )
        reload_btn.grid(row=0, column=2, sticky="ew", padx=2)
        self._cmd_buttons[CMD_RELOAD_PAGE] = reload_btn

        # ===================================================================
        # SECTION 3: Tab Management
        # ===================================================================
        tab_frame = ttk.LabelFrame(controls_frame, text="Tab Management", padding="10")
        tab_frame.pack(fill="x", padx=5, pady=5)
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.columnconfigure(1, weight=1)

        new_tab_btn = ttk.Button(
            tab_frame,
            text="+ New Tab",
            command=lambda: self._handle_input(CMD_NEW_TAB),
        )
        new_tab_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self._cmd_buttons[CMD_NEW_TAB] = new_tab_btn

        close_tab_btn = ttk.Button(
            tab_frame,
            text="✕ Close Tab",
            command=lambda: self._handle_input(CMD_CLOSE_THIS_TAB),
        )
        close_tab_btn.grid(row=0, column=1, sticky="ew", padx=2)
        self._cmd_buttons[CMD_CLOSE_THIS_TAB] = close_tab_btn

        # ===================================================================
        # SECTION 4: Text Input & Keyboard
        # ===================================================================
        text_frame = ttk.LabelFrame(
            controls_frame,
            text="Text Input & Keyboard",
            padding="10",
        )
        text_frame.pack(fill="x", padx=5, pady=5)

        # Text entry row
        text_entry_frame = tk.Frame(text_frame)
        text_entry_frame.pack(fill="x", pady=(0, 5))
        text_entry_frame.columnconfigure(1, weight=1)

        tk.Label(text_entry_frame, text="Enter text:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.enter_var = tk.StringVar()
        self.enter_text_box = tk.Entry(text_entry_frame, textvariable=self.enter_var)
        self.enter_text_box.grid(row=0, column=1, sticky="ew", padx=2)
        self.enter_text_box.bind("<Return>", lambda _e: self._send_enter_text())

        # Keyboard action buttons
        keyboard_frame = tk.Frame(text_frame)
        keyboard_frame.pack(fill="x")

        # Row 1: Basic actions
        basic_keys = [
            ("Enter", CMD_PRESS_ENTER),
            ("Backspace", CMD_PRESS_BACKSPACE),
            ("Delete", CMD_PRESS_DELETE),
            ("Click Out", CMD_CLICK_OUT),
        ]

        for i, (label, cmd) in enumerate(basic_keys):
            btn = ttk.Button(
                keyboard_frame,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            btn.grid(row=0, column=i, sticky="ew", padx=1, pady=1)
            self._key_buttons[cmd] = btn
            keyboard_frame.columnconfigure(i, weight=1)

        # Row 2: Arrow keys
        arrow_frame = tk.Frame(keyboard_frame)
        arrow_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        arrow_cmds = [
            ("←", CMD_CURSOR_LEFT),
            ("→", CMD_CURSOR_RIGHT),
            ("↑", CMD_CURSOR_UP),
            ("↓", CMD_CURSOR_DOWN),
        ]

        for i, (label, cmd) in enumerate(arrow_cmds):
            btn = ttk.Button(
                arrow_frame,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            btn.grid(row=0, column=i, sticky="ew", padx=1, pady=1)
            self._key_buttons[cmd] = btn
            arrow_frame.columnconfigure(i, weight=1)

        # Row 3: Modifier keys
        modifier_frame = tk.Frame(keyboard_frame)
        modifier_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        modifier_cmds = [
            ("Shift ⬇", CMD_HOLD_SHIFT),
            ("Shift ⬆", CMD_RELEASE_SHIFT),
            ("Ctrl ⬇", CMD_HOLD_CTRL),
            ("Ctrl ⬆", CMD_RELEASE_CTRL),
        ]

        for i, (label, cmd) in enumerate(modifier_cmds):
            btn = ttk.Button(
                modifier_frame,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            btn.grid(row=0, column=i, sticky="ew", padx=1, pady=1)
            self._key_buttons[cmd] = btn
            modifier_frame.columnconfigure(i, weight=1)

        # Row 4: Additional modifiers
        modifier_frame2 = tk.Frame(keyboard_frame)
        modifier_frame2.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        modifier_cmds2 = [
            ("Alt ⬇", CMD_HOLD_ALT),
            ("Alt ⬆", CMD_RELEASE_ALT),
            ("Cmd ⬇", CMD_HOLD_CMD),
            ("Cmd ⬆", CMD_RELEASE_CMD),
        ]

        for i, (label, cmd) in enumerate(modifier_cmds2):
            btn = ttk.Button(
                modifier_frame2,
                text=label,
                width=12,
                command=lambda c=cmd: self._handle_input(c),
            )
            btn.grid(row=0, column=i, sticky="ew", padx=1, pady=1)
            self._key_buttons[cmd] = btn
            modifier_frame2.columnconfigure(i, weight=1)

        # Custom key press
        custom_key_frame = tk.Frame(text_frame)
        custom_key_frame.pack(fill="x", pady=(5, 0))
        custom_key_frame.columnconfigure(1, weight=1)

        self.press_key_var = tk.StringVar(value="")
        press_key_btn = ttk.Button(
            custom_key_frame,
            text="Press Key:",
            width=15,
            command=lambda: self._send_press_key(),
        )
        press_key_btn.grid(row=0, column=0, sticky="w", padx=(0, 5))
        press_key_entry = tk.Entry(
            custom_key_frame,
            textvariable=self.press_key_var,
            width=10,
        )
        press_key_entry.grid(row=0, column=1, sticky="w", padx=2)
        press_key_entry.bind("<Return>", lambda _e: self._send_press_key())
        self._cmd_buttons[CMD_PRESS_KEY] = press_key_btn

        # ===================================================================
        # SECTION 5: LLM Command Interface
        # ===================================================================
        llm_frame = ttk.LabelFrame(
            controls_frame,
            text="LLM Command Interface",
            padding="10",
        )
        llm_frame.pack(fill="x", padx=5, pady=5)
        llm_frame.columnconfigure(2, weight=1)

        # Mode selection row
        mode_frame = tk.Frame(llm_frame)
        mode_frame.grid(row=0, column=0, sticky="w", padx=(0, 10))

        tk.Label(mode_frame, text="Mode:").pack(side="left")
        self._llm_mode = tk.StringVar(value="act")
        rb_act = tk.Radiobutton(
            mode_frame,
            text="Act",
            variable=self._llm_mode,
            value="act",
            command=self._on_mode_change,
        )
        rb_obs = tk.Radiobutton(
            mode_frame,
            text="Observe",
            variable=self._llm_mode,
            value="observe",
            command=self._on_mode_change,
        )
        rb_act.pack(side="left", padx=(5, 0))
        rb_obs.pack(side="left", padx=(2, 0))

        # Loader icon
        self.llm_loader = tk.Label(llm_frame, text="⏳")
        self.llm_loader.grid(row=0, column=1, padx=5)
        self.llm_loader.grid_remove()

        # Command entry
        self.cmd_var = tk.StringVar()
        self.llm_entry = tk.Entry(llm_frame, textvariable=self.cmd_var)
        self.llm_entry.grid(row=0, column=2, sticky="ew", padx=2)
        self.llm_entry.bind("<Return>", lambda _e: self._send_llm_command())

        # Expectation entry (only for Act mode)
        expectation_frame = tk.Frame(llm_frame)
        expectation_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        expectation_frame.columnconfigure(1, weight=1)

        self.expectation_label = tk.Label(expectation_frame, text="Expectation:")
        self.expectation_label.grid(row=0, column=0, sticky="w", padx=(0, 5))

        self.expectation_var = tk.StringVar()
        self.expectation_entry = tk.Entry(
            expectation_frame,
            textvariable=self.expectation_var,
        )
        self.expectation_entry.grid(row=0, column=1, sticky="ew", padx=2)
        self.expectation_entry.bind("<Return>", lambda _e: self._send_llm_command())

        # Multi-step mode checkbox (only for Act mode)
        multi_step_frame = tk.Frame(llm_frame)
        multi_step_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        self.multi_step_var = tk.BooleanVar(value=True)  # Default to checked
        self.multi_step_checkbox = tk.Checkbutton(
            multi_step_frame,
            text="Multi-step mode",
            variable=self.multi_step_var,
        )
        self.multi_step_checkbox.grid(row=0, column=0, sticky="w", padx=(0, 5))

        # Initially set visibility based on default mode
        self._update_act_mode_widgets_visibility()

        # ===================================================================
        # SECTION 6: Scrolling Controls
        # ===================================================================

        # ===================================================================
        # SECTION 6.1: Scroll Until Visible
        # ===================================================================
        scroll_until_frame = ttk.LabelFrame(
            controls_frame,
            text="Scroll Until Visible",
            padding="10",
        )
        scroll_until_frame.pack(fill="x", padx=5, pady=5)
        scroll_until_frame.columnconfigure(1, weight=1)

        # Element description row
        tk.Label(scroll_until_frame, text="Element Description:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.scroll_element_var = tk.StringVar()
        self.scroll_element_entry = tk.Entry(
            scroll_until_frame,
            textvariable=self.scroll_element_var,
        )
        self.scroll_element_entry.grid(row=0, column=1, sticky="ew", padx=2)
        self.scroll_element_entry.bind(
            "<Return>",
            lambda _e: self._send_scroll_until_visible(),
        )

        # Direction and Max retries row
        direction_retries_frame = tk.Frame(scroll_until_frame)
        direction_retries_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        direction_retries_frame.columnconfigure(2, weight=1)

        # Direction selection
        tk.Label(direction_retries_frame, text="Direction:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.scroll_direction_var = tk.StringVar(value="down")
        rb_down = tk.Radiobutton(
            direction_retries_frame,
            text="Down",
            variable=self.scroll_direction_var,
            value="down",
        )
        rb_up = tk.Radiobutton(
            direction_retries_frame,
            text="Up",
            variable=self.scroll_direction_var,
            value="up",
        )
        rb_down.grid(row=0, column=1, padx=2)
        rb_up.grid(row=0, column=2, padx=2)

        # Max retries
        tk.Label(direction_retries_frame, text="Max retries:").grid(
            row=0,
            column=3,
            sticky="w",
            padx=(10, 5),
        )
        self.scroll_max_retries_var = tk.StringVar(value="5")
        max_retries_entry = tk.Entry(
            direction_retries_frame,
            textvariable=self.scroll_max_retries_var,
            width=8,
        )
        max_retries_entry.grid(row=0, column=4, sticky="w", padx=2)
        max_retries_entry.bind("<Return>", lambda _e: self._send_scroll_until_visible())

        # Scroll until visible button
        scroll_until_btn = ttk.Button(
            scroll_until_frame,
            text="🔍 Scroll Until Visible",
            command=self._send_scroll_until_visible,
        )
        scroll_until_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        scroll_frame = ttk.LabelFrame(
            controls_frame,
            text="Scrolling Controls",
            padding="10",
        )
        scroll_frame.pack(fill="x", padx=5, pady=5)

        # Manual scroll controls
        manual_scroll_frame = tk.Frame(scroll_frame)
        manual_scroll_frame.pack(fill="x", pady=(0, 5))
        manual_scroll_frame.columnconfigure(1, weight=1)

        tk.Label(manual_scroll_frame, text="Scroll pixels:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.scroll_px_var = tk.StringVar(value="100")
        px_entry = tk.Entry(
            manual_scroll_frame,
            textvariable=self.scroll_px_var,
            width=8,
        )
        px_entry.grid(row=0, column=1, sticky="w", padx=2)

        def _step_pixels() -> str:
            val = self.scroll_px_var.get().strip()
            return val if val.isdigit() and int(val) > 0 else "100"

        scroll_up_btn = ttk.Button(
            manual_scroll_frame,
            text="▲ Scroll Up",
            command=lambda: self._handle_input(
                CMD_SCROLL_UP.replace("*", _step_pixels()),
            ),
        )
        scroll_up_btn.grid(row=0, column=2, sticky="ew", padx=2)

        scroll_down_btn = ttk.Button(
            manual_scroll_frame,
            text="▼ Scroll Down",
            command=lambda: self._handle_input(
                CMD_SCROLL_DOWN.replace("*", _step_pixels()),
            ),
        )
        scroll_down_btn.grid(row=0, column=3, sticky="ew", padx=2)

        # Auto-scroll controls
        auto_scroll_frame = tk.Frame(scroll_frame)
        auto_scroll_frame.pack(fill="x")
        auto_scroll_frame.columnconfigure(0, weight=1)
        auto_scroll_frame.columnconfigure(1, weight=1)
        auto_scroll_frame.columnconfigure(2, weight=1)

        start_up_btn = ttk.Button(
            auto_scroll_frame,
            text="⏫ Start Up",
            command=lambda: self._handle_input(CMD_START_SCROLL_UP),
        )
        start_up_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self._cmd_buttons[CMD_START_SCROLL_UP] = start_up_btn

        stop_scroll_btn = ttk.Button(
            auto_scroll_frame,
            text="⏹ Stop",
            command=lambda: self._handle_input(CMD_STOP_SCROLLING),
        )
        stop_scroll_btn.grid(row=0, column=1, sticky="ew", padx=2)
        self._cmd_buttons[CMD_STOP_SCROLLING] = stop_scroll_btn

        start_down_btn = ttk.Button(
            auto_scroll_frame,
            text="⏬ Start Down",
            command=lambda: self._handle_input(CMD_START_SCROLL_DOWN),
        )
        start_down_btn.grid(row=0, column=2, sticky="ew", padx=2)
        self._cmd_buttons[CMD_START_SCROLL_DOWN] = start_down_btn

        continue_btn = ttk.Button(
            auto_scroll_frame,
            text="▶ Continue",
            command=lambda: self._handle_input(CMD_CONT_SCROLLING),
        )
        continue_btn.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=2,
            pady=(2, 0),
        )
        self._cmd_buttons[CMD_CONT_SCROLLING] = continue_btn

        # Advanced auto-scroll toggle (3-position vertical slider)
        toggle_frame = tk.Frame(scroll_frame)
        toggle_frame.pack(fill="x", pady=(10, 0))

        tk.Label(toggle_frame, text="Auto-scroll Toggle:").pack(side="left")

        # Speed entry for auto-scroll
        self.scroll_speed_var = tk.StringVar(value="250")
        speed_entry = tk.Entry(
            toggle_frame,
            textvariable=self.scroll_speed_var,
            width=6,
            justify="center",
        )
        speed_entry.pack(side="right", padx=(4, 0))

        def _on_scroll_toggle(val):
            if self._scroll_toggle_guard:
                return  # ignore programmatic updates
            try:
                mode = int(float(val))
            except Exception:
                return

            # Helper to sanitise speed input
            def _speed_px() -> str:
                val = self.scroll_speed_var.get().strip()
                return val if val.isdigit() and int(val) > 0 else "250"

            if mode == 0:
                self._queue_command(f"{CMD_START_SCROLL_UP} {_speed_px()}")
                self._last_scroll_dir = "up"
            elif mode == 2:
                self._queue_command(f"{CMD_START_SCROLL_DOWN} {_speed_px()}")
                self._last_scroll_dir = "down"
            else:
                self._queue_command(CMD_STOP_SCROLLING)
                self._manual_stop_pending = True

            # Disable toggle until browser confirms
            self.scroll_toggle.configure(state="disabled")
            self._scroll_pending_target = mode

        # horizontal slider with 3 notches
        slider_frame = tk.Frame(toggle_frame)
        slider_frame.pack(side="left", padx=(10, 0))

        self.scroll_toggle = tk.Scale(
            slider_frame,
            from_=0,
            to=2,
            orient="vertical",
            length=100,
            showvalue=False,
            variable=self._scroll_mode,
            command=_on_scroll_toggle,
        )
        self.scroll_toggle.pack(side="left")

        # Label markers – placed to the right of the vertical slider
        lbls = tk.Frame(slider_frame)
        lbls.pack(side="right", padx=(4, 0))
        for row_idx, txt in enumerate(["▲", "■", "▼"]):
            tk.Label(lbls, text=txt).grid(row=row_idx, column=0, sticky="n")
            lbls.rowconfigure(row_idx, weight=1)

        # Add step widgets for proper disable logic
        self._step_widgets = [speed_entry, self.scroll_toggle]

        # ===================================================================
        # SECTION 7: Security & CAPTCHA
        # ===================================================================
        security_frame = ttk.LabelFrame(
            controls_frame,
            text="Security & CAPTCHA",
            padding="10",
        )
        security_frame.pack(fill="x", padx=5, pady=5)

        solve_captcha_btn = ttk.Button(
            security_frame,
            text="🔓 Solve CAPTCHA",
            command=lambda: self._handle_input(CMD_SOLVE_CAPTCHA),
        )
        solve_captcha_btn.pack(fill="x", padx=2)
        self._cmd_buttons[CMD_SOLVE_CAPTCHA] = solve_captcha_btn

        # ===================================================================
        # SECTION 8: Dialog & Popup Management
        # ===================================================================
        dialog_frame = ttk.LabelFrame(
            controls_frame,
            text="Dialog & Popup Management",
            padding="10",
        )
        dialog_frame.pack(fill="x", padx=5, pady=5)

        # Dialog response row
        dialog_response_frame = tk.Frame(dialog_frame)
        dialog_response_frame.pack(fill="x", pady=(0, 5))
        dialog_response_frame.columnconfigure(1, weight=1)

        self.dialog_msg_var = tk.StringVar()
        tk.Label(
            dialog_response_frame,
            textvariable=self.dialog_msg_var,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.dialog_input_var = tk.StringVar()
        self.dialog_entry = tk.Entry(
            dialog_response_frame,
            textvariable=self.dialog_input_var,
            width=24,
        )
        self.dialog_entry.grid(row=0, column=1, sticky="ew", padx=(6, 4))

        self.dialog_accept_btn = ttk.Button(
            dialog_response_frame,
            text="Accept",
            width=8,
            command=self._accept_dialog,
        )
        self.dialog_accept_btn.grid(row=0, column=2, padx=2)

        self.dialog_dismiss_btn = ttk.Button(
            dialog_response_frame,
            text="Dismiss",
            width=8,
            command=lambda: self._queue_command(CMD_DISMISS_DIALOG),
        )
        self.dialog_dismiss_btn.grid(row=0, column=3, padx=2)

        # Start disabled
        for b in (self.dialog_accept_btn, self.dialog_dismiss_btn):
            b.configure(state="disabled")
        self.dialog_entry.configure(state="disabled")

        # Popup controls
        popup_controls_frame = tk.Frame(dialog_frame)
        popup_controls_frame.pack(fill="x")
        popup_controls_frame.columnconfigure(0, weight=1)
        popup_controls_frame.columnconfigure(1, weight=1)

        close_popup_btn = ttk.Button(
            popup_controls_frame,
            text="Close Popup",
            command=lambda: self._handle_input(CMD_CLOSE_POPUP),
        )
        close_popup_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self._cmd_buttons[CMD_CLOSE_POPUP] = close_popup_btn

        # Add entry for selecting popup by name/identifier
        popup_select_frame = tk.Frame(dialog_frame)
        popup_select_frame.pack(fill="x", pady=(5, 0))
        popup_select_frame.columnconfigure(1, weight=1)

        tk.Label(popup_select_frame, text="Select Popup:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 5),
        )
        self.popup_select_var = tk.StringVar()
        popup_select_entry = tk.Entry(
            popup_select_frame,
            textvariable=self.popup_select_var,
            width=20,
        )
        popup_select_entry.grid(row=0, column=1, sticky="ew", padx=2)

        select_popup_btn = ttk.Button(
            popup_select_frame,
            text="Select",
            command=lambda: self._handle_input(
                f"select_popup {self.popup_select_var.get()}",
            ),
        )
        select_popup_btn.grid(row=0, column=2, padx=2)
        self._cmd_buttons[CMD_SELECT_POPUP] = select_popup_btn

        # ===================================================================
        # RIGHT PANEL →  Log, Actions, State (existing but improved)
        # ===================================================================
        self.right_panel = tk.Frame(main_paned)
        main_paned.add(self.right_panel, stretch="always")
        right = self.right_panel
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Log / Actions notebook
        note = ttk.Notebook(right)
        note.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tab_log = ttk.Frame(note)
        tab_actions = ttk.Frame(note)
        tab_llm = ttk.Frame(note)
        note.add(tab_log, text="Log")
        note.add(tab_actions, text="Valid Actions")
        note.add(tab_llm, text="LLM Stream")

        # Log setup
        self.log = scrolledtext.ScrolledText(tab_log, state="disabled")
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("observation", foreground="#00BFFF", font=("Helvetica", 10))
        self.log.tag_config("llm", foreground="#FFD700")

        # Actions setup
        self.act_box = scrolledtext.ScrolledText(tab_actions, state="disabled")
        self.act_box.pack(fill="both", expand=True)
        self._last_actions_txt = ""

        # LLM stream setup
        self.llm_stream_box = scrolledtext.ScrolledText(tab_llm, state="disabled")
        self.llm_stream_box.pack(fill="both", expand=True)

        # Browser state display
        self.state_var = tk.StringVar()
        self.state_lbl = tk.Label(
            right,
            textvariable=self.state_var,
            justify="left",
            anchor="w",
            font=("Consolas", 9),
        )
        self.state_lbl.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # ── top‑centre CAPTCHA label (hidden) ─────────────────────────────
        self.captcha_lbl = tk.Label(
            self,
            text="🔒 Solving CAPTCHA…",
            fg="orange red",
            font=("Helvetica", 10, "bold"),
            bg=self.cget("bg"),
        )
        self.captcha_lbl.place_forget()

        # Initialize widget collections for enable/disable logic
        self._arrow_button_widgets = []
        self._modifier_button_widgets = []
        self._key_button_widgets = []
        self._element_buttons = []
        self._tab_row_buttons = []

        # Set up color scheme and styling
        r, g, b = [c // 256 for c in self.winfo_rgb(self.cget("bg"))]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        dark = brightness < 128

        style = ttk.Style()
        style.configure("TButton", anchor="center")

        # Element button styling
        fg_light, fg_dark = "#000000", "#ffffff"
        bg_idle_light, bg_idle_dark = "#f0f0f0", "#3a3a3a"
        bg_active_light, bg_active_dark = "#dcdcdc", "#505050"

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

        disabled_bg = "#2a2a2a"
        active_bg = "#505050"
        idle_bg = bg_idle_dark if dark else bg_idle_light
        active_fg = fg_dark if dark else fg_light

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

        # Store references for later use
        self._el_canvas = el_canvas
        self._el_scroll = el_scroll

    # Remove the old layout functions as they're no longer needed
    def _relayout_key_buttons(self):
        pass  # No longer needed with new fixed layout

    def _relayout_arrow_buttons(self):
        pass  # No longer needed with new fixed layout

    def _send_llm_command(self) -> None:
        """Fetch text, detect selected Act/Observe radio, and forward to handler."""
        text = self.cmd_var.get().strip()
        if not text:
            return

        mode_val = self._llm_mode.get() if self._llm_mode else "act"

        # Clear entry early for snappy UX
        self.cmd_var.set("")

        # Forward to generic handler with explicit mode – no prefix hacks needed
        self._handle_input(text, from_llm_box=True, mode=mode_val)

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

    # ---------- press‑key helper ------------------------------------
    def _send_press_key(self) -> None:
        key = self.press_key_var.get().strip()
        if not key:
            self._log("⚠ Please enter a key to press")
            return
        self._handle_input(f"press_key {key}")
        self.press_key_var.set("")  # clear box

    # ---------- scroll until visible helper --------------------------
    def _send_scroll_until_visible(self) -> None:
        """Trigger the scroll until visible functionality."""
        element_desc = self.scroll_element_var.get().strip()
        if not element_desc:
            self._log("⚠ Please enter an element description")
            return

        direction = self.scroll_direction_var.get()
        max_retries_str = self.scroll_max_retries_var.get().strip()

        # Validate max_retries
        try:
            max_retries = int(max_retries_str)
            if max_retries <= 0:
                self._log("⚠ Max retries must be a positive integer")
                return
        except ValueError:
            self._log("⚠ Max retries must be a valid positive integer")
            return

        # Start the scroll until visible process in a background thread
        self._start_scroll_until_visible(element_desc, direction, max_retries)

        # Clear the element description for next use
        self.scroll_element_var.set("")

    def _start_scroll_until_visible(
        self,
        element_description: str,
        direction: str,
        max_retries: int,
    ) -> None:
        """Start scroll until visible in a background thread."""
        if self._llm_busy:
            self._log("⚠ LLM still working – please wait")
            return

        self._llm_busy = True

        # Show loader and disable entry
        self.llm_loader.grid()
        self.scroll_element_entry.configure(state="disabled")

        # Log the action
        self._log(
            f"🔍 Scrolling until '{element_description}' is visible ({direction}, max {max_retries} retries)",
        )

        def _worker():
            """Run scroll_until_visible in a thread."""
            try:
                if self._controller:
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self._scroll_until_visible_impl(
                                element_description,
                                direction,
                                max_retries,
                            ),
                        )
                        self._log(f"✓ {result}")
                    finally:
                        loop.close()
                else:
                    self._log("❌ Controller not available for scroll until visible")
            except Exception as exc:
                import traceback as _tb

                self._log(f"❌ Scroll until visible error: {exc}")
                self._log_trace(_tb.format_exc())
            finally:
                # Reset UI state
                def reset_ui():
                    self._llm_busy = False
                    self.llm_loader.grid_remove()
                    self.scroll_element_entry.configure(state="normal")

                self.after_idle(reset_ui)

        threading.Thread(target=_worker, daemon=True).start()

    async def _scroll_until_visible_impl(
        self,
        element_description: str,
        direction: str = "down",
        max_retries: int = 5,
    ) -> str:
        """
        Implementation of scroll_until_visible that replicates ActionProvider's functionality
        using controller's act and observe methods.
        """

        class ElementVisibility(BaseModel):
            is_visible: bool = Field(
                description="True if the element is visible on the screen, False otherwise.",
            )
            reason: str = Field(description="The reason for the visibility status.")

        for i in range(max_retries):
            # First, check if the element is already visible
            visibility_status = await self._controller.observe(
                f"Is '{element_description}' currently visible on the screen?",
                response_format=ElementVisibility,
            )

            if visibility_status.is_visible:
                return f"Success: Element '{element_description}' is now visible."

            # If not visible, perform the scroll action
            await self._controller.act(
                f"Scroll {direction} slightly",
                expectation=f"The page should scroll {direction}.",
            )
            await asyncio.sleep(1)

        # If the loop finishes without finding the element, return a failure message
        return f"Failure: Could not find element '{element_description}' after {max_retries} scrolls."

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
            CMD_PRESS_KEY: "Requires focus in a text-box",
            CMD_HOLD_SHIFT: "Requires focus in a text-box",
            CMD_RELEASE_SHIFT: "Requires focus in a text-box",
            CMD_HOLD_CTRL: "Requires focus in a text-box",
            CMD_HOLD_ALT: "Requires focus in a text-box",
            CMD_RELEASE_CTRL: "Requires focus in a text-box",
            CMD_RELEASE_ALT: "Requires focus in a text-box",
            CMD_HOLD_CMD: "Requires focus in a text-box",
            CMD_RELEASE_CMD: "Requires focus in a text-box",
            CMD_STOP_SCROLLING: "Auto-scroll isn't running",
            CMD_CONT_SCROLLING: "Auto-scroll isn't running",
            CMD_START_SCROLL_UP: "Already auto-scrolling",
            CMD_START_SCROLL_DOWN: "Already auto-scrolling",
            CMD_SCROLL_UP: "Already at the very top of the page",
            CMD_BACK_NAV: "No previous page in history",
            CMD_FORWARD_NAV: "No forward history entry",
            CMD_RELOAD_PAGE: "",
        }

        def _is_ok(cmd: str) -> bool:
            for v in valid:
                if cmd == v:
                    return True
                if v.endswith("*"):
                    prefix = v[:-1]
                    if cmd.startswith(prefix) or cmd == prefix.rstrip(" _"):
                        return True
                # allow numeric argument after fixed patterns (e.g. start_scrolling_down 600)
                if cmd.startswith(v + " "):
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
                "Cannot type – there's no active text-box on the page",
            )

        # ----- Search / URL entry -------------------------------------
        ok_search = _is_ok(CMD_SEARCH) or _is_ok(CMD_OPEN_URL)
        self.search_entry.configure(state="normal" if ok_search else "disabled")
        if not ok_search:
            _Tooltip(
                self.search_entry,
                "Disabled while typing in a page text-box",
            )

        # ----- Numbered element buttons ---------------------------------
        can_click_el = _is_ok(CMD_CLICK_BUTTON)
        for btn in getattr(self, "_element_buttons", []):
            btn.configure(state="normal" if can_click_el else "disabled")
            if not can_click_el:
                _Tooltip(btn, "Cannot click elements while typing in a text-box")

        # ----- Per‑row "×" / Go buttons in the Tabs pane --------------
        for btn in getattr(self, "_tab_row_buttons", []):
            if btn["text"] == "×":  # close-tab buttons
                ok = any(name.startswith("close_tab_") for name in valid)
                reason = "Tab actions disabled while typing"
            else:  # Go buttons
                ok = any(name.startswith("select_tab_") for name in valid)
                reason = "Cannot switch tabs while typing"

            btn.configure(state="normal" if ok else "disabled")
            if not ok:
                _Tooltip(btn, reason)

        # ----- Step-scroll widgets ------------------------------------ NEW
        auto = self.state.auto_scroll
        for w in getattr(self, "_step_widgets", []):
            if hasattr(w, "configure"):
                w.configure(state="disabled" if auto else "normal")

    # ──────────────────────── ACTIONS‑PANE HELPER ───────────────────────
    def _refresh_actions_list(self) -> None:
        """Update the Actions tab (anti-jitter, preserves scroll)."""
        groups = list_available_actions(
            self.tab_titles,
            [(idx, label) for idx, label, _ in self.elements],
            state=self.state,
        )

        # ---------- build text with ONLY currently valid actions ----------
        valid = get_valid_actions(self.state)

        def _is_ok(cmd: str) -> bool:
            for v in valid:
                if cmd == v:
                    return True
                if v.endswith("*"):
                    prefix = v[:-1]
                    if cmd.startswith(prefix) or cmd == prefix.rstrip(" _"):
                        return True
                # allow numeric argument after fixed patterns (e.g. start_scrolling_down 600)
                if cmd.startswith(v + " "):
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

        if new_txt == self._last_actions_txt:  # anti-jitter cache
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
        st = self.state
        self.state_var.set(
            f"url:         {st.url[:60]}\n"
            f"title:       {st.title[:60]}\n"
            f"scroll_y:    {st.scroll_y}\n"
            f"auto_scroll: {st.auto_scroll}\n"
            f"scroll_speed: {st.scroll_speed}\n"
            f"in_textbox:  {st.in_textbox}\n"
            f"captcha_pending: {st.captcha_pending}",
        )

        # sync the auto-scroll toggle --------------------------------
        if hasattr(self, "scroll_toggle"):
            # If awaiting confirmation, re-enable when state matches target
            if self._scroll_pending_target is not None:
                mode_to_state = {0: "up", 1: None, 2: "down"}
                expected = mode_to_state[self._scroll_pending_target]
                if self.state.auto_scroll == expected:
                    self.scroll_toggle.configure(state="normal")
                    self._scroll_pending_target = None
                    if self._manual_stop_pending and expected is None:
                        self._manual_stop_pending = False

            # Update toggle position to match current auto-scroll state
            current_auto = self.state.auto_scroll
            if current_auto == "up":
                target_pos = 0
            elif current_auto == "down":
                target_pos = 2
            else:
                target_pos = 1  # stopped

            # Only update if different to avoid triggering callback
            if self._scroll_mode.get() != target_pos:
                self._scroll_toggle_guard = True
                self._scroll_mode.set(target_pos)
                self._scroll_toggle_guard = False

        # Lazy-create CAPTCHA banner so we don't depend on call order
        if not hasattr(self, "captcha_lbl"):
            self.captcha_lbl = tk.Label(
                self,
                text="🔒 Solving CAPTCHA…",
                fg="orange red",
                font=("Helvetica", 10, "bold"),
                bg=self.cget("bg"),
            )
            self.captcha_lbl.place_forget()

        # Show or hide banner based on current flag
        if st.captcha_pending:
            self.captcha_lbl.place(relx=0.5, rely=0.0, y=6, anchor="n")
        else:
            self.captcha_lbl.place_forget()

    # ────────────────────── DIALOG BAR REFRESH ──────────────────────── NEW
    def _refresh_dialog_bar(self):
        st = self.state
        has_dialog = st.dialog_open
        if has_dialog:
            self.dialog_msg_var.set(st.dialog_msg or "(no message)")
            self.dialog_accept_btn.configure(state="normal")
            self.dialog_dismiss_btn.configure(state="normal")

            if st.dialog_type == "prompt":
                self.dialog_entry.configure(state="normal")
            else:
                self.dialog_entry.configure(state="disabled")
        else:
            self.dialog_msg_var.set("")
            self.dialog_accept_btn.configure(state="disabled")
            self.dialog_dismiss_btn.configure(state="disabled")
            self.dialog_entry.configure(state="disabled")

    # ────────────────────── ACCEPT DIALOG ACTION ─────────────────────── NEW
    def _accept_dialog(self):
        """Send the appropriate primitive based on current dialog type."""
        if self.state.dialog_type == "prompt":
            text = self.dialog_input_var.get()
            self._queue_command(CMD_TYPE_DIALOG.replace("*", text))
        else:
            self._queue_command(CMD_ACCEPT_DIALOG)

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
            if self._controller:
                self._controller.stop()
        except Exception:
            pass
        self.destroy()
        import os

        os._exit(0)

    # ─────────────────────── HIGH‑LEVEL INPUT HANDLER ───────────────────
    def _handle_input(
        self,
        text: str,
        *,
        from_llm_box: bool = False,
        mode: str | None = None,
    ) -> None:
        try:
            # Determine observation/action based on explicit mode or legacy prefix
            if mode is not None:
                is_observation = mode == "observe"
            else:
                is_observation = text.strip().startswith(("observe:", "?"))

            if is_observation:
                # Log observation requests with a different icon
                self._log(f"🔍 {text}")
            else:
                self._log(f"> {text}")

            def _is_valid_primitive(cmd: str) -> bool:
                valid = get_valid_actions(self.state)
                for pat in valid:
                    if cmd == pat:
                        return True
                    if pat.endswith("*") and cmd.startswith(pat[:-1]):
                        return True  # parameterised match
                    # allow numeric argument after fixed patterns (e.g. start_scrolling_down 600)
                    if cmd.startswith(pat + " "):
                        return True
                return False

            # skip the fast-path when the text came from the LLM box or in observation mode
            if (
                (not from_llm_box)
                and (not is_observation)
                and _is_valid_primitive(text)
            ):
                self._queue_command(text)
                return

            # hand off to background thread (non-blocking)
            self._start_llm_thread(text, is_observation=is_observation)
        except Exception:
            import traceback as _tb

            self._log_trace(_tb.format_exc())

    def set_worker(self, worker):
        self._worker = worker

    def set_controller(self, controller):
        self._controller = controller
        # Start/stop llm_resp polling based on controller availability
        if controller is None:
            # Start polling for backward compatibility
            self.after(50, self._poll_llm_resp)
        # If controller is provided, we don't need to poll llm_resp as controller handles everything

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
            spd = sc["start_scrolling_up"].get("speed")
            if spd:
                return f"{CMD_START_SCROLL_UP} {int(spd)}"
            return CMD_START_SCROLL_UP
        if sc.get("start_scrolling_down", {}).get("apply"):
            spd = sc["start_scrolling_down"].get("speed")
            if spd:
                return f"{CMD_START_SCROLL_DOWN} {int(spd)}"
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

        # ----- search / open-url ------------------------------------------
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

            # 2. single-key / caret actions
            for cmd in (
                CMD_PRESS_ENTER,
                CMD_PRESS_BACKSPACE,
                CMD_PRESS_DELETE,
                CMD_CURSOR_LEFT,
                CMD_CURSOR_RIGHT,
                CMD_CURSOR_UP,
                CMD_CURSOR_DOWN,
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
            state_data = payload.get("state", {})
            if state_data:
                # Convert dict to BrowserState object
                if isinstance(state_data, dict):
                    self.state = BrowserState(**state_data)
                else:
                    self.state = state_data
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
                get_valid_actions(self.state),
            )
            self._refresh_dialog_bar()

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
