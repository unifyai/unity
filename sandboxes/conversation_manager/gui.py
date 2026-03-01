from __future__ import annotations

import asyncio
import logging
import queue as _queue
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sandboxes.conversation_manager.commands import HELP_TEXT, parse_command
from sandboxes.conversation_manager.ipc_protocol import (
    MessageType,
    create_message,
    new_message_id,
    validate_message,
)
from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay
from sandboxes.conversation_manager.log_aggregator import LogAggregator
from sandboxes.conversation_manager.state_snapshot import (
    capture_snapshot,
    render_snapshot_text,
    save_snapshot,
)
from sandboxes.conversation_manager.trace_display import TraceDisplay

LG = logging.getLogger("conversation_manager_sandbox")

_WORKER_READY_TIMEOUT_S = 60.0
_COMMAND_TIMEOUT_S = 30.0
_RESTART_EXIT_CODE = 23

# -----------------------------------------------------------------------------
# Textual UI (optional dependency)
# -----------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.screen import Screen
    from textual.widgets import Button, Footer, Header, Input, Label, RichLog

    # Optional Textual widgets (best-effort; fallback if unavailable).
    try:  # pragma: no cover
        from textual.widgets import TabbedContent, TabPane, Tree, Collapsible, Static

        _TEXTUAL_ADVANCED_AVAILABLE = True
    except Exception:  # pragma: no cover
        _TEXTUAL_ADVANCED_AVAILABLE = False

    _TEXTUAL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency / runtime env
    _TEXTUAL_AVAILABLE = False
    _TEXTUAL_ADVANCED_AVAILABLE = False


def _make_rich_log(
    *,
    id: str,
    wrap: bool = True,
    highlight: bool = True,
    max_lines: int = 1500,
) -> "RichLog":
    """
    Construct a RichLog with safe performance defaults.

    Textual terminal rendering performance degrades sharply when RichLog buffers
    grow without bound. We cap `max_lines` when supported by the installed
    Textual version.
    """
    try:
        import inspect as _inspect

        kwargs: dict[str, Any] = {
            "id": id,
            "wrap": bool(wrap),
            "highlight": bool(highlight),
        }
        try:
            sig = _inspect.signature(RichLog)  # type: ignore[arg-type]
            if "max_lines" in sig.parameters:
                kwargs["max_lines"] = int(max(100, max_lines))
        except Exception:
            pass
        return RichLog(**kwargs)  # type: ignore[arg-type]
    except Exception:
        # Fallback with minimal args.
        return RichLog(id=id, wrap=wrap, highlight=highlight)  # type: ignore[call-arg]


@dataclass
class GuiRuntime:
    ui_to_worker: Any
    worker_to_ui: Any
    config: dict
    # UI-owned "args-like" namespace used by existing widgets for rendering.
    args: Any
    # Minimal state mirrored from worker `{"type":"state"}` messages.
    state: Any
    ready: bool = False
    last_worker_message_at: float = 0.0
    started_at: float = 0.0
    worker_pid: int | None = None
    pending: dict[str, float] = field(default_factory=dict)  # cmd_id -> sent_at
    last_timeout_warn_at: float = 0.0
    computer_status: dict[str, Any] = field(default_factory=dict)
    # Best-effort TTS de-dupe (UI-owned).
    last_tts_text: str = ""
    last_tts_at: float = 0.0
    # Debounced panel refresh flags (avoid re-rendering on every IPC message).
    dirty_tree: bool = False
    dirty_logs: bool = False
    dirty_trace: bool = False
    dirty_computer: bool = False
    # Avoid re-rendering unchanged small widgets.
    last_hint_text: str = ""
    last_computer_text: str = ""
    # UI-owned displays (also attached to `args` for compatibility).
    trace_display: TraceDisplay = field(default_factory=TraceDisplay)
    event_tree_display: EventTreeDisplay = field(default_factory=EventTreeDisplay)
    log_aggregator: LogAggregator = field(default_factory=LogAggregator)
    conversation_lines: list[str] = field(default_factory=list)


if _TEXTUAL_AVAILABLE:

    class AppendLine(Message):
        def __init__(self, line: str) -> None:
            self.line = line
            super().__init__()

    class RefreshPanels(Message):
        def __init__(
            self,
            *,
            tree: bool = False,
            logs: bool = False,
            trace: bool = False,
            computer: bool = False,
        ) -> None:
            self.tree = bool(tree)
            self.logs = bool(logs)
            self.trace = bool(trace)
            self.computer = bool(computer)
            super().__init__()

    class _BaseScreen(Screen):
        """Base screen shared by Menu/SMS/Email/Call screens with log + command input."""

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical(id="root"):
                with Horizontal(id="body"):
                    with Vertical(id="left"):
                        yield from self.compose_left()
                    with Vertical(id="right"):
                        yield _make_rich_log(
                            id="responses",
                            wrap=True,
                            highlight=True,
                            max_lines=1200,
                        )
                with Horizontal(id="cmd_row"):
                    yield Input(
                        placeholder="Commands: msg, sms, email, meet, say, end_meet, us",
                        id="command_input",
                    )
                    yield Button("Send", id="submit_command")
            yield Footer()

        def compose_left(self) -> ComposeResult:  # overridden
            yield Label("Not implemented")

        async def _route_raw(self, raw: str) -> None:
            raw = (raw or "").strip()
            if not raw:
                return
            app = self.app  # type: ignore[attr-defined]
            await app.route_command(raw)  # type: ignore[attr-defined]

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "command_input":
                await self._route_raw(event.value)
                event.input.value = ""

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "submit_command":
                inp = self.query_one("#command_input", Input)
                await self._route_raw(inp.value)
                inp.value = ""

    class MenuScreen(_BaseScreen):
        def compose_left(self) -> ComposeResult:
            yield Label("Actions", id="title")
            yield Horizontal(
                Button("Send SMS", id="nav_sms"),
                Button("Send Email", id="nav_email"),
                Button("Phone Call", id="nav_call"),
                Button("Quit", id="quit"),
            )

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            await super().on_button_pressed(event)
            if event.button.id == "nav_sms":
                self.app.push_screen(SMSScreen())  # type: ignore[attr-defined]
            elif event.button.id == "nav_email":
                self.app.push_screen(EmailScreen())  # type: ignore[attr-defined]
            elif event.button.id == "nav_call":
                self.app.push_screen(CallScreen())  # type: ignore[attr-defined]
            elif event.button.id == "quit":
                self.app.exit()  # type: ignore[attr-defined]

    class SMSScreen(_BaseScreen):
        def compose_left(self) -> ComposeResult:
            yield Label("Simulate incoming SMS", id="title")
            yield Input(placeholder="Message", id="sms_message")
            yield Horizontal(Button("Send", id="send_sms"), Button("Back", id="back"))

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            await super().on_button_pressed(event)
            if event.button.id == "send_sms":
                msg = self.query_one("#sms_message", Input).value.strip()
                await self._route_raw(f"sms {msg}")
            elif event.button.id == "back":
                self.app.pop_screen()  # type: ignore[attr-defined]

    class EmailScreen(_BaseScreen):
        def compose_left(self) -> ComposeResult:
            yield Label("Simulate incoming Email", id="title")
            yield Input(placeholder="Subject", id="email_subject")
            yield Input(placeholder="Body", id="email_body")
            yield Horizontal(Button("Send", id="send_email"), Button("Back", id="back"))

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            await super().on_button_pressed(event)
            if event.button.id == "send_email":
                subj = (
                    self.query_one("#email_subject", Input).value.strip()
                    or "Sandbox Email"
                )
                body = self.query_one("#email_body", Input).value.strip()
                await self._route_raw(f"email {subj} | {body}")
            elif event.button.id == "back":
                self.app.pop_screen()  # type: ignore[attr-defined]

    class CallScreen(_BaseScreen):
        def compose_left(self) -> ComposeResult:
            yield Label("Simulate voice call (Unify Meet)", id="title")
            yield Horizontal(
                Button("Start Call", id="call_start"),
                Button("End Call", id="call_end"),
                Button("Back", id="back"),
            )
            yield Input(placeholder="Utterance (Send uses: say <text>)", id="call_utt")
            yield Button("Send Utterance", id="call_say")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            await super().on_button_pressed(event)
            if event.button.id == "call_start":
                await self._route_raw("meet")
            elif event.button.id == "call_end":
                await self._route_raw("end_meet")
            elif event.button.id == "call_say":
                utt = self.query_one("#call_utt", Input).value.strip()
                await self._route_raw(f"say {utt}")
            elif event.button.id == "back":
                self.app.pop_screen()  # type: ignore[attr-defined]

    # ─────────────────────────────────────────────────────────────────────
    # Dashboard UI
    # ─────────────────────────────────────────────────────────────────────

    if _TEXTUAL_ADVANCED_AVAILABLE:

        class DashboardScreen(Screen):
            CSS = """
            #root { height: 1fr; }
            /* Keep the top controls/tabs from consuming the whole screen. */
            #top_row { height: 18; }
            #left_controls { width: 35%; padding: 1; }
            #right_tabs { width: 65%; padding: 1; }
            #conversation { height: 1fr; border: round $surface; }
            #cmd_row { height: auto; }
            #command_input { width: 1fr; }
            #logs { height: 18; }
            """

            async def _route_raw(self, raw: str) -> None:
                raw = (raw or "").strip()
                if not raw:
                    return
                app = self.app  # type: ignore[attr-defined]
                await app.route_command(raw)  # type: ignore[attr-defined]

            def compose(self) -> ComposeResult:
                yield Header(show_clock=False)
                with Vertical(id="root"):
                    with Horizontal(id="top_row"):
                        with Vertical(id="left_controls"):
                            yield Label("Event Controls", id="title")
                            yield Button("Compose SMS", id="btn_sms")
                            yield Button("Compose Email", id="btn_email")
                            yield Button("Start Call", id="btn_meet_start")
                            yield Button("End Call", id="btn_meet_end")
                            yield Button(
                                "Start Screen Share",
                                id="btn_screen_share_toggle",
                            )
                            yield Button("Quit", id="btn_quit")
                        with Vertical(id="right_tabs"):
                            with TabbedContent(id="tabs"):
                                with TabPane("Event Tree", id="tab_tree"):
                                    yield Tree("ConversationManager", id="event_tree")
                                    yield Label("", id="tree_details")
                                with TabPane("Computer", id="tab_computer"):
                                    yield Static("", id="computer_status")
                    yield _make_rich_log(
                        id="conversation",
                        wrap=True,
                        highlight=True,
                        max_lines=2000,
                    )
                    with Horizontal(id="cmd_row"):
                        yield Input(
                            placeholder="Type a command (e.g., sms Hello, trace 3, tree, /stop)",
                            id="command_input",
                        )
                        yield Button("Mic", id="btn_mic")
                        yield Button("Send", id="submit_command")
                        yield Button("Quit", id="btn_quit_cmd")
                    with Vertical(id="logs"):
                        yield Label("Logs (Collapsible)", id="logs_title")
                        yield Collapsible(
                            _make_rich_log(
                                id="logs_cm",
                                wrap=True,
                                highlight=True,
                                max_lines=1200,
                            ),
                            title="CM Logs",
                            id="coll_cm",
                            collapsed=False,
                        )
                        yield Collapsible(
                            _make_rich_log(
                                id="logs_actor",
                                wrap=True,
                                highlight=True,
                                max_lines=1200,
                            ),
                            title="Actor Logs",
                            id="coll_actor",
                            collapsed=True,
                        )
                        yield Collapsible(
                            _make_rich_log(
                                id="trace_panel",
                                wrap=True,
                                highlight=True,
                                max_lines=1200,
                            ),
                            title="Trace (CodeAct)",
                            id="coll_trace",
                            collapsed=True,
                        )
                yield Footer()

            def on_mount(self) -> None:
                app = self.app  # type: ignore[attr-defined]
                rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]

                # Preserve tree expansion state across refreshes. We rebuild the tree
                # periodically, and without this the UI collapses nodes while the user
                # is trying to inspect children.
                self._tree_expanded_paths: set[str] = set()
                self._tree_last_interaction_at: float = 0.0
                self._last_tree_root_key: str = ""

                # Prefer keyboard-driven control by focusing the command input.
                try:
                    self.query_one("#command_input", Input).focus()
                except Exception:
                    pass

                self.set_interval(1.5, self._refresh_computer_status)

                self._refresh_header_banner()

                # Hide computer tab if computer backend not available in this config.
                try:
                    cfg = getattr(rt.args, "_actor_config", None)
                    backend_mode = (
                        getattr(cfg, "computer_backend_mode", "none") if cfg else "none"
                    )
                    if backend_mode != "real":
                        tabs = self.query_one("#tabs", TabbedContent)
                        tabs.remove_pane("Computer")
                except Exception:
                    pass

                # Initial paint: populate tree/logs/trace from existing buffers.
                # (In event-driven mode we might mount after events have already arrived.)
                try:
                    self._refresh_tree()
                except Exception:
                    pass
                try:
                    self._refresh_logs()
                except Exception:
                    pass
                try:
                    self._refresh_computer_status()
                except Exception:
                    pass

            def on_unmount(self) -> None:
                # Ensure we don't leave the microphone stream open if the user
                # exits while recording.
                try:
                    rec = getattr(self, "_voice_rec", None)
                    if rec is not None:
                        # Stop in a thread; unmount is sync.
                        import threading as _threading

                        _threading.Thread(
                            target=lambda: rec.stop(),
                            daemon=True,
                        ).start()
                        self._voice_rec = None
                except Exception:
                    pass

            def _tree_node_key(self, node: Any) -> str:
                """
                Return a stable key for a Textual Tree node.

                Prefer sandbox TreeNode metadata stored in `.data` so icon/status
                changes don't break expansion persistence.
                """
                try:
                    d = getattr(node, "data", None)
                    if d is not None:
                        cid = getattr(d, "call_id", None)
                        lbl = getattr(d, "label", None)
                        if cid:
                            return f"{cid}:{lbl or ''}"
                        if lbl:
                            return str(lbl)
                except Exception:
                    pass
                try:
                    return str(getattr(node, "label", ""))
                except Exception:
                    return ""

            def _tree_path_key(self, node: Any) -> str:
                parts: list[str] = []
                cur = node
                while cur is not None:
                    parts.append(self._tree_node_key(cur))
                    cur = getattr(cur, "parent", None)
                parts.reverse()
                return " / ".join([p for p in parts if p])

            def on_tree_node_expanded(self, event: Any) -> None:
                try:
                    node = getattr(event, "node", None)
                    if node is None:
                        return
                    self._tree_expanded_paths.add(self._tree_path_key(node))
                    self._tree_last_interaction_at = time.monotonic()
                except Exception:
                    return

            def on_tree_node_collapsed(self, event: Any) -> None:
                try:
                    node = getattr(event, "node", None)
                    if node is None:
                        return
                    self._tree_expanded_paths.discard(self._tree_path_key(node))
                    self._tree_last_interaction_at = time.monotonic()
                except Exception:
                    return

            def on_refresh_panels(self, message: RefreshPanels) -> None:
                # Coalesced refreshes triggered by incoming events / trace capture.
                try:
                    if message.tree:
                        self._refresh_tree()
                except Exception:
                    pass
                try:
                    # Trace is rendered as part of _refresh_logs.
                    if message.logs or message.trace:
                        self._refresh_logs()
                except Exception:
                    pass
                try:
                    if message.computer:
                        self._refresh_computer_status()
                except Exception:
                    pass

            def on_collapsible_toggled(self, event: Any) -> None:
                # When the trace/logs collapsibles are opened, refresh immediately.
                try:
                    c = getattr(event, "collapsible", None)
                    if c is None:
                        return
                    cid = getattr(c, "id", "") or ""
                    collapsed = bool(getattr(c, "collapsed", False))
                    if not collapsed and cid in {
                        "coll_trace",
                        "coll_cm",
                        "coll_actor",
                    }:
                        self._refresh_logs()
                except Exception:
                    return

            def _refresh_header_banner(self) -> None:
                try:
                    app = self.app  # type: ignore[attr-defined]
                    rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
                    cfg = getattr(rt.args, "_actor_config", None)
                    actor_type = (
                        getattr(cfg, "actor_type", "simulated") if cfg else "simulated"
                    )
                    mgrs = (
                        getattr(cfg, "managers_mode", "simulated")
                        if cfg
                        else "simulated"
                    )
                    backend_mode = (
                        getattr(cfg, "computer_backend_mode", "none") if cfg else "none"
                    )
                    header = self.query_one(Header)
                    header.sub_title = f"actor={actor_type} | managers={mgrs} | computer={backend_mode}"
                except Exception:
                    pass

            def _refresh_tree(self) -> None:
                try:
                    # If the user is interacting with the tree, don't rebuild it right away.
                    # This avoids "flaky" auto-collapsing while expanding nodes.
                    try:
                        if (
                            time.monotonic()
                            - float(getattr(self, "_tree_last_interaction_at", 0.0))
                        ) < 1.0:
                            return
                    except Exception:
                        pass

                    app = self.app  # type: ignore[attr-defined]
                    rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
                    tree_disp = getattr(rt.args, "_event_tree_display", None)
                    if tree_disp is None:
                        return

                    widget = self.query_one("#event_tree", Tree)

                    # Get all active trees (for concurrent Actor handles)
                    all_trees = []
                    try:
                        all_trees = tree_disp.get_all_trees() or []
                    except Exception:
                        pass

                    # Fall back to legacy single-tree if get_all_trees fails
                    if not all_trees:
                        root = tree_disp.get_tree_data()
                        if root is None:
                            return
                        all_trees = [root]

                    # Build a unique key for the current set of trees
                    tree_keys = []
                    for t in all_trees:
                        key = str(
                            getattr(t, "call_id", None)
                            or getattr(t, "label", "")
                            or "",
                        )
                        hid = getattr(t, "handle_id", None)
                        if hid is not None:
                            key = f"H{hid}:{key}"
                        tree_keys.append(key)
                    combined_key = "|".join(tree_keys)

                    if combined_key and combined_key != getattr(
                        self,
                        "_last_tree_root_key",
                        "",
                    ):
                        self._tree_expanded_paths.clear()
                        self._last_tree_root_key = combined_key

                    # Set root label to show number of concurrent executions
                    if len(all_trees) > 1:
                        widget.root.label = (
                            f"📊 Event Tree ({len(all_trees)} concurrent)"
                        )
                    else:
                        widget.root.label = "📊 Event Tree"

                    # Textual's TreeNodes doesn't implement list.clear(); remove nodes instead.
                    try:
                        for child in list(widget.root.children):
                            try:
                                child.remove()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    def _add(parent, node, *, show_handle: bool = False):
                        icon = {
                            "completed": "✓",
                            "in_progress": "⏳",
                            "error": "❌",
                        }.get(
                            getattr(node, "status", "in_progress"),
                            "•",
                        )
                        node_label = getattr(node, "label", "")
                        hid = getattr(node, "handle_id", None)
                        if show_handle and hid is not None:
                            label = f"{icon} [H{hid}] {node_label}"
                        else:
                            label = f"{icon} {node_label}"
                        child = parent.add(label)
                        try:
                            child.data = node  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        # Restore expansion state.
                        try:
                            if self._tree_path_key(child) in self._tree_expanded_paths:
                                child.expand()
                        except Exception:
                            pass
                        for c in getattr(node, "children", []) or []:
                            _add(child, c, show_handle=False)

                    # Add each tree as a top-level node (show handle ID for concurrent)
                    show_handles = len(all_trees) > 1
                    for tree_root in all_trees:
                        _add(widget.root, tree_root, show_handle=show_handles)

                    widget.root.expand()
                except Exception:
                    # Never let periodic UI refresh crash the app.
                    return

            def _refresh_logs(self) -> None:
                app = self.app  # type: ignore[attr-defined]
                rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
                lg = getattr(rt.args, "_log_aggregator", None)
                if lg is None:
                    return
                try:
                    cm_log = self.query_one("#logs_cm", RichLog)
                    actor_log = self.query_one("#logs_actor", RichLog)
                except Exception:
                    return
                try:
                    cm_log.clear()  # type: ignore[attr-defined]
                    actor_log.clear()  # type: ignore[attr-defined]
                except Exception:
                    pass

                try:
                    actor_handles = lg.get_active_handles("actor")
                    group_actor = len(actor_handles) > 1
                except Exception:
                    group_actor = False

                try:
                    cm_log.write(lg.render_expanded("cm"))
                    actor_log.write(
                        lg.render_expanded(
                            "actor",
                            group_by_handle=group_actor,
                            max_message_length=0,
                        ),
                    )
                except Exception:
                    pass

                # Trace panel (optional)
                try:
                    td = getattr(rt.args, "_trace_display", None)
                    if td is not None:
                        tr = self.query_one("#trace_panel", RichLog)
                        try:
                            tr.clear()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        try:
                            n = int(getattr(td, "entry_count")())
                        except Exception:
                            n = 0
                        if n <= 0:
                            msg = "(no trace entries yet)"
                            try:
                                err = getattr(td, "last_capture_error")()
                                if err:
                                    msg += f"\n(last capture error: {err})"
                            except Exception:
                                pass
                            tr.write(msg)
                        else:
                            # Check if there are multiple handles to decide whether to group
                            group_traces = False
                            try:
                                trace_handles = td.get_active_handles()
                                group_traces = len(trace_handles) > 1
                            except Exception:
                                pass

                            # Show the full trajectory across the session so earlier turns
                            # remain visible even after the sandbox starts a new actor loop
                            # (which resets per-event turn numbering).
                            try:
                                tr.write(td.render_all(group_by_handle=group_traces))
                            except Exception:
                                tr.write(td.render_current_event())
                except Exception:
                    pass

            def _refresh_computer_status(self) -> None:
                app = self.app  # type: ignore[attr-defined]
                rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
                try:
                    w = self.query_one("#computer_status", Static)
                except Exception:
                    return
                try:
                    snap = None
                    try:
                        snap = rt.computer_status
                    except Exception:
                        snap = None
                    if not isinstance(snap, dict) or not snap:
                        # Fallback: show config-level backend mode.
                        cfg = getattr(rt.args, "_actor_config", None)
                        backend_mode = (
                            getattr(cfg, "computer_backend_mode", "none")
                            if cfg
                            else "none"
                        )
                        if backend_mode == "none":
                            out_txt = "Computer: disabled"
                        elif backend_mode == "mock":
                            out_txt = "Computer: mock backend (waiting for activity)"
                        else:
                            out_txt = "Computer: real backend (waiting for activity)"
                        try:
                            if out_txt == rt.last_computer_text:
                                return
                            rt.last_computer_text = out_txt
                        except Exception:
                            pass
                        w.update(out_txt)
                        return

                    connected = snap.get("connected", None)
                    last_error = snap.get("last_error", None)
                    url = snap.get("last_url") or "(unknown URL)"
                    actions = snap.get("actions") or []
                    last = actions[-3:] if len(actions) > 3 else actions
                    last_lines = (
                        "\n".join(
                            [
                                f"- {a.get('kind')}: {a.get('detail')}"
                                for a in last
                                if isinstance(a, dict)
                            ],
                        )
                        if last
                        else "(no activity yet)"
                    )
                    status = (
                        "connected"
                        if connected
                        else ("disconnected" if connected is False else "unknown")
                    )
                    out = f"Computer: {status}\nURL: {url}\nRecent:\n{last_lines}"
                    if last_error:
                        out += f"\nError: {last_error}"
                    try:
                        if out == rt.last_computer_text:
                            return
                        rt.last_computer_text = out
                    except Exception:
                        pass
                    w.update(out)
                except Exception:
                    w.update("Computer panel unavailable.")

            async def on_input_submitted(self, event: Input.Submitted) -> None:
                if event.input.id != "command_input":
                    return
                try:
                    await self._route_raw(event.value)
                finally:
                    event.input.value = ""

            async def on_button_pressed(self, event: Button.Pressed) -> None:
                app = self.app  # type: ignore[attr-defined]
                if event.button.id == "submit_command":
                    try:
                        inp = self.query_one("#command_input", Input)
                        await self._route_raw(inp.value)
                    finally:
                        try:
                            inp.value = ""
                        except Exception:
                            pass
                    return
                if event.button.id == "btn_quit":
                    app.exit()
                    return
                if event.button.id == "btn_quit_cmd":
                    app.exit()
                    return
                if event.button.id == "btn_sms":
                    try:
                        app.post_message(AppendLine("[ui] Compose SMS: type message, then press Enter"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        inp = self.query_one("#command_input", Input)
                        inp.value = "sms "
                        inp.focus()
                    except Exception:
                        pass
                    return
                if event.button.id == "btn_email":
                    try:
                        app.post_message(AppendLine("[ui] Compose Email: use `email <subject> | <body>`"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        inp = self.query_one("#command_input", Input)
                        inp.value = "email Subject | Body"
                        inp.focus()
                    except Exception:
                        pass
                    return
                if event.button.id == "btn_meet_start":
                    try:
                        app.post_message(AppendLine("[ui] Start Call pressed"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    await app.route_command("meet")  # type: ignore[attr-defined]
                    return
                if event.button.id == "btn_meet_end":
                    try:
                        app.post_message(AppendLine("[ui] End Call pressed"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    await app.route_command("end_meet")  # type: ignore[attr-defined]
                    return
                if event.button.id == "btn_screen_share_toggle":
                    btn = self.query_one("#btn_screen_share_toggle", Button)
                    sharing = getattr(self, "_screen_share_active", False)
                    if not sharing:
                        try:
                            app.post_message(AppendLine("[ui] Start Screen Share pressed"))  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        await app.route_command("assistant_screen_share_start")  # type: ignore[attr-defined]
                        self._screen_share_active = True
                        btn.label = "Stop Screen Share"
                    else:
                        try:
                            app.post_message(AppendLine("[ui] Stop Screen Share pressed"))  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        await app.route_command("assistant_screen_share_stop")  # type: ignore[attr-defined]
                        self._screen_share_active = False
                        btn.label = "Start Screen Share"
                    return
                if event.button.id == "btn_mic":
                    # Toggle mic recording: first click starts, second click stops.
                    if not bool(getattr(app.runtime.args, "voice", False)):  # type: ignore[attr-defined]
                        app.post_message(AppendLine("⚠️ Restart with `--voice` to enable recording."))  # type: ignore[attr-defined]
                        return
                    try:
                        from sandboxes.utils import (
                            start_voice_recording,
                            transcribe_deepgram_no_input,
                        )
                    except Exception as exc:
                        app.post_message(AppendLine(f"⚠️ Voice mode unavailable ({exc})."))  # type: ignore[attr-defined]
                        return

                    try:
                        btn = self.query_one("#btn_mic", Button)
                    except Exception:
                        btn = None  # type: ignore[assignment]

                    rec = getattr(self, "_voice_rec", None)
                    if rec is None:
                        # Start recording.
                        try:
                            self._voice_rec = await asyncio.to_thread(
                                start_voice_recording,
                            )
                            if btn is not None:
                                btn.label = "Stop"
                            app.post_message(AppendLine("[ui] Recording... click Stop to finish"))  # type: ignore[attr-defined]
                        except Exception as exc:
                            app.post_message(AppendLine(f"⚠️ Failed to start recording ({exc})."))  # type: ignore[attr-defined]
                        return

                    # Stop recording + transcribe off the UI loop.
                    self._voice_rec = None
                    if btn is not None:
                        btn.label = "Mic"

                    async def _finish(handle) -> None:
                        try:
                            audio = await asyncio.to_thread(handle.stop)
                            text = (
                                await asyncio.to_thread(
                                    transcribe_deepgram_no_input,
                                    audio,
                                )
                                or ""
                            ).strip()
                        except Exception as exc:
                            app.post_message(AppendLine(f"❌ Voice transcription failed: {exc}"))  # type: ignore[attr-defined]
                            return
                        if not text:
                            app.post_message(AppendLine("⚠️ Transcription was empty. Please try again."))  # type: ignore[attr-defined]
                            return
                        try:
                            inp = self.query_one("#command_input", Input)
                            cur = (inp.value or "").rstrip()
                            if cur:
                                sep = "" if cur.endswith(" ") else " "
                                inp.value = cur + sep + text
                            else:
                                if bool(getattr(app.runtime.state, "in_call", False)):  # type: ignore[attr-defined]
                                    inp.value = "sayv " + text
                                else:
                                    inp.value = "sms " + text
                            inp.focus()
                        except Exception:
                            pass

                    asyncio.create_task(_finish(rec))
                    return

    class ModernizedMessagingApp(App):
        CSS = """
        #root { height: 1fr; }
        #body { height: 1fr; }
        #left { width: 45%; padding: 1; }
        #right { width: 55%; padding: 1; }
        #responses { height: 1fr; border: round $surface; }
        #cmd_row { height: auto; }
        #command_input { width: 1fr; }
        """

        def __init__(self, *, ui_to_worker: Any, worker_to_ui: Any, config: dict):
            super().__init__()
            from types import SimpleNamespace as _SimpleNamespace

            from sandboxes.conversation_manager.config_manager import (
                ActorConfig as _ActorConfig,
            )

            cfg = dict(config or {})
            actor_cfg = _ActorConfig.from_json_obj(
                (
                    cfg.get("actor_config")
                    if isinstance(cfg.get("actor_config"), dict)
                    else cfg
                ),
            )
            args = _SimpleNamespace(**cfg)
            setattr(args, "voice", bool(getattr(args, "voice", False)))
            setattr(args, "_actor_config", actor_cfg)

            # UI-owned displays (used by dashboard refresh methods).
            trace_display = TraceDisplay()
            event_tree_display = EventTreeDisplay()
            log_aggregator = LogAggregator()
            setattr(args, "_trace_display", trace_display)
            setattr(args, "_event_tree_display", event_tree_display)
            setattr(args, "_log_aggregator", log_aggregator)

            state = _SimpleNamespace(
                active=False,
                in_call=False,
                pending_clarification=False,
            )

            worker_pid = None
            try:
                wp = cfg.get("worker_pid")
                if wp is not None:
                    worker_pid = int(wp)
            except Exception:
                worker_pid = None

            now = time.monotonic()
            self.runtime = GuiRuntime(
                ui_to_worker=ui_to_worker,
                worker_to_ui=worker_to_ui,
                config=cfg,
                args=args,
                state=state,
                ready=False,
                last_worker_message_at=now,
                started_at=now,
                worker_pid=worker_pid,
                trace_display=trace_display,
                event_tree_display=event_tree_display,
                log_aggregator=log_aggregator,
                computer_status={},
            )

        def compose(self) -> ComposeResult:
            yield Header()
            yield Footer()

        def on_mount(self) -> None:
            if _TEXTUAL_ADVANCED_AVAILABLE:
                self.push_screen(DashboardScreen())
            else:
                self.push_screen(MenuScreen())
            # Poll worker messages frequently (UI remains responsive even if worker blocks).
            self.set_interval(0.075, self._poll_worker_messages)
            self.set_interval(0.5, self._watchdogs)
            # Debounce expensive panel refreshes (tree/logs/trace) to keep the UI snappy
            # under high event volume.
            self.set_interval(0.2, self._flush_dirty_panels)
            # Disable input until worker announces readiness.
            self._set_input_enabled(False)
            self.post_message(AppendLine("[ui] Waiting for worker..."))

        def _flush_dirty_panels(self) -> None:
            rt = self.runtime
            if not _TEXTUAL_ADVANCED_AVAILABLE:
                return
            tree = bool(rt.dirty_tree)
            logs = bool(rt.dirty_logs)
            trace = bool(rt.dirty_trace)
            computer = bool(rt.dirty_computer)
            if not (tree or logs or trace or computer):
                return
            rt.dirty_tree = False
            rt.dirty_logs = False
            rt.dirty_trace = False
            rt.dirty_computer = False
            try:
                self.screen.post_message(
                    RefreshPanels(tree=tree, logs=logs, trace=trace, computer=computer),
                )
            except Exception:
                pass

        async def route_command(self, raw: str) -> None:
            rt = self.runtime
            trimmed = (raw or "").strip()
            if not trimmed:
                return

            # Always echo the user's command (UI-side) so the pane feels responsive.
            self.post_message(AppendLine(f"[you] {trimmed}"))

            cmd = parse_command(
                text=trimmed,
                in_call=bool(getattr(rt.state, "in_call", False)),
                active=bool(getattr(rt.state, "active", False)),
            )

            # Local validation errors (mirror CommandRouter behavior).
            if cmd.kind == "unknown":
                if cmd.error and cmd.error != "empty":
                    self.post_message(AppendLine(cmd.error))
                return

            # UI-only commands
            if cmd.kind == "help":
                self.post_message(AppendLine("\n" + HELP_TEXT + "\n"))
                return
            if cmd.kind in {"show_logs", "collapse_logs"}:
                self._handle_logs_locally(kind=cmd.kind, args=cmd.args)
                return
            if cmd.kind == "save_state":
                for line in self._save_state_from_ui(cmd.args):
                    self.post_message(AppendLine(line))
                return

            # Quit should stop both UI and worker (best-effort).
            if cmd.kind == "quit":
                try:
                    rt.ui_to_worker.put_nowait(
                        create_message(
                            MessageType.SHUTDOWN,
                            payload={},
                            id=new_message_id(),
                        ),
                    )
                except Exception:
                    pass
                self.exit()
                return

            # Voice commands stay in UI process; translate to non-voice commands for worker.
            if cmd.kind == "scenario_seed_voice":
                desc = await self._record_and_transcribe_best_effort()
                if not desc:
                    self.post_message(AppendLine("⚠️ Voice transcription was empty."))
                    return
                trimmed = f"us {desc}"
                self.post_message(AppendLine(f"▶️ {desc}"))
            elif cmd.kind == "event" and cmd.name == "sayv":
                text = (cmd.args or "").strip()
                if not text:
                    text = await self._record_and_transcribe_best_effort()
                if not text:
                    self.post_message(AppendLine("⚠️ Voice transcription was empty."))
                    return
                trimmed = f"say {text}"
                self.post_message(AppendLine(f"▶️ {text}"))

            # Immediate steering acknowledgment.
            if cmd.kind == "steering":
                self.post_message(AppendLine(f"✓ Sent: {cmd.args}"))

            # Ship to worker.
            cmd_id = new_message_id()
            msg = create_message(
                MessageType.EXECUTE_RAW,
                id=cmd_id,
                payload={
                    "raw": trimmed,
                    "in_call": bool(getattr(rt.state, "in_call", False)),
                },
            )
            try:
                rt.ui_to_worker.put_nowait(msg)
                rt.pending[cmd_id] = time.monotonic()
            except _queue.Full:
                self.post_message(
                    AppendLine("❌ UI→worker queue is full. Please retry."),
                )
            except Exception as exc:
                self.post_message(AppendLine(f"❌ Failed to send to worker: {exc}"))

        async def on_append_line(self, msg: AppendLine) -> None:
            try:
                self.runtime.conversation_lines.append(str(msg.line))
            except Exception:
                pass
            # Write into the active screen's response log.
            try:
                scr = self.screen
                log = (
                    scr.query_one("#conversation", RichLog)
                    if _TEXTUAL_ADVANCED_AVAILABLE
                    else scr.query_one("#responses", RichLog)
                )
                log.write(msg.line)
            except Exception:
                pass

        # ──────────────────────────────────────────────────────────────
        # IPC polling + handlers
        # ──────────────────────────────────────────────────────────────

        def _poll_worker_messages(self) -> None:
            rt = self.runtime
            q = rt.worker_to_ui
            processed = 0
            # Drain a bounded number of messages per tick to avoid UI starvation.
            while processed < 200:
                try:
                    msg = q.get_nowait()
                except _queue.Empty:
                    break
                except Exception:
                    break

                processed += 1
                if not isinstance(msg, dict):
                    continue
                if not validate_message(msg):
                    continue

                rt.last_worker_message_at = time.monotonic()
                t = str(msg.get("type") or "")
                payload = msg.get("payload") or {}
                mid = msg.get("id")
                if (
                    isinstance(mid, str)
                    and mid in rt.pending
                    and t
                    in {
                        MessageType.LINES,
                        MessageType.ERROR,
                    }
                ):
                    try:
                        rt.pending.pop(mid, None)
                    except Exception:
                        pass
                if t == MessageType.READY:
                    self._handle_ready()
                elif t == MessageType.LINES:
                    self._handle_lines(payload)
                elif t == MessageType.STATE:
                    self._handle_state(payload)
                elif t == MessageType.EVENT:
                    self._handle_event(payload)
                elif t == MessageType.ERROR:
                    self._handle_error(payload)
                elif t == MessageType.WORKER_EXIT:
                    self._handle_worker_exit(payload)

        def _watchdogs(self) -> None:
            """
            Periodic edge-case handling:
            - worker ready timeout
            - worker process death (if pid is provided)
            - per-command IPC timeouts
            """

            rt = self.runtime
            now = time.monotonic()

            # Ready timeout.
            if (not rt.ready) and (now - rt.started_at) > _WORKER_READY_TIMEOUT_S:
                self.post_message(
                    AppendLine(
                        "⚠️ Worker failed to start within 60s. Check logs and restart the sandbox.",
                    ),
                )
                self._set_input_enabled(False)
                # Prevent repeat.
                rt.started_at = now + 1e9

            # Worker liveness check (best-effort POSIX).
            if rt.worker_pid is not None:
                try:
                    os.kill(int(rt.worker_pid), 0)
                except Exception:
                    self.post_message(
                        AppendLine(
                            "❌ Worker process crashed unexpectedly. Please restart the sandbox.",
                        ),
                    )
                    self._set_input_enabled(False)
                    rt.worker_pid = None

            # Per-command timeout warnings.
            try:
                oldest = min(rt.pending.values()) if rt.pending else None
            except Exception:
                oldest = None
            if oldest is not None and (now - float(oldest)) > _COMMAND_TIMEOUT_S:
                if (now - rt.last_timeout_warn_at) > 5.0:
                    self.post_message(
                        AppendLine(
                            "⚠️ Worker not responding. The worker may be busy or stuck. "
                            "You can wait or restart the sandbox.",
                        ),
                    )
                    rt.last_timeout_warn_at = now

        def _handle_ready(self) -> None:
            rt = self.runtime
            if rt.ready:
                return
            rt.ready = True
            self._set_input_enabled(True)
            self.post_message(AppendLine("[ui] ✓ Worker ready"))

        def _handle_lines(self, payload: dict) -> None:
            lines = payload.get("lines") if isinstance(payload, dict) else None
            if not isinstance(lines, list):
                return
            for ln in lines:
                s = str(ln)
                if s.strip():
                    self.post_message(AppendLine(s))
            # Lines often imply logs/tree/trace changed; refresh panels best-effort.
            try:
                rt = self.runtime
                rt.dirty_logs = True
                rt.dirty_trace = True
            except Exception:
                pass

        def _handle_state(self, payload: dict) -> None:
            rt = self.runtime
            if not isinstance(payload, dict):
                return
            try:
                rt.state.active = bool(payload.get("active", False))
                rt.state.in_call = bool(payload.get("in_call", False))
                rt.state.pending_clarification = bool(
                    payload.get("pending_clarification", False),
                )
            except Exception:
                return
            self._update_input_placeholder()

        def _handle_event(self, payload: dict) -> None:
            if not isinstance(payload, dict):
                return
            channel = str(payload.get("channel") or "")
            ev = payload.get("event")
            if not isinstance(ev, dict):
                return
            try:
                self._apply_event_to_displays(channel=channel, event=ev)
            except Exception:
                return

        def _handle_error(self, payload: dict) -> None:
            msg = ""
            if isinstance(payload, dict):
                msg = str(payload.get("message") or "")
                tb = str(payload.get("traceback") or "")
            else:
                tb = ""
            self.post_message(AppendLine(f"❌ Worker Error: {msg}".rstrip()))
            if tb:
                self.post_message(AppendLine("Traceback:"))
                self.post_message(AppendLine(tb))
            self.post_message(AppendLine("Please restart the sandbox."))
            # If worker reports an error before ready, disable input.
            self._set_input_enabled(False)

        def _handle_worker_exit(self, payload: dict) -> None:
            restart = False
            try:
                if isinstance(payload, dict):
                    restart = bool(payload.get("restart", False))
            except Exception:
                restart = False

            if restart:
                self.post_message(
                    AppendLine("[worker] Restart requested (config switch)."),
                )
                self.exit(_RESTART_EXIT_CODE)
                return

            self.post_message(
                AppendLine("❌ Worker process exited. Please restart the sandbox."),
            )
            self._set_input_enabled(False)

        def _apply_event_to_displays(self, *, channel: str, event: dict) -> None:
            rt = self.runtime
            # EventBus events arrive on channel like "eventbus:ManagerMethod".
            if channel.startswith("eventbus:"):
                kind = channel.split(":", 1)[1]
                if kind == "ManagerMethod":
                    from unity.events.types.manager_method import ManagerMethodPayload

                    try:
                        mm = ManagerMethodPayload.model_validate(
                            event.get("payload") or {},
                        )
                    except Exception:
                        return
                    call_id = str(event.get("calling_id") or "")
                    try:
                        rt.event_tree_display.handle_manager_method(
                            call_id=call_id,
                            payload=mm,
                        )
                    except Exception:
                        pass
                    try:
                        direction = (mm.phase or "").strip().lower()
                        label = (mm.hierarchy_label or "").strip()
                        msg = f"{mm.manager}.{mm.method}"
                        if direction:
                            msg += f" [{direction}]"
                        if label:
                            msg += f" — {label}"
                        rt.log_aggregator.handle_structured_event(
                            category="manager",
                            message=msg,
                        )
                    except Exception:
                        pass
                    # Trace is sourced from `trace:entry` by default. EventBus
                    # ManagerMethod events do not include code/stdout for execute_code
                    # boundaries, and can create empty placeholder trace entries.
                    try:
                        rt.dirty_tree = True
                        rt.dirty_logs = True
                        rt.dirty_trace = True
                    except Exception:
                        pass
                    return
                # Ignore other EventBus event types. They are high-volume
                # (LLM/ToolLoop/Comms/Message) and make logs noisy + UI sluggish.
                return

            # Computer status snapshots (emitted by runtime process).
            if channel == "computer:status":
                try:
                    rt.computer_status = dict(event)
                except Exception:
                    rt.computer_status = {}
                try:
                    rt.dirty_computer = True
                except Exception:
                    pass
                return

            # Sandbox trace entries streamed from the worker (CodeAct execute_code wrapper).
            if channel == "trace:entry":
                try:
                    code = str(event.get("code") or "")
                    res = (
                        event.get("result")
                        if isinstance(event.get("result"), dict)
                        else {}
                    )
                    # Extract handle_id if present in the event
                    trace_hid: int | None = None
                    try:
                        hid_val = event.get("handle_id", -1)
                        if hid_val is not None:
                            trace_hid = int(hid_val)
                            if trace_hid < 0:
                                trace_hid = None
                    except Exception:
                        trace_hid = None
                    rt.trace_display.capture_execution(
                        code=code,
                        result=res,
                        handle_id=trace_hid,
                    )
                except Exception:
                    return
                try:
                    rt.dirty_trace = True
                except Exception:
                    pass
                return

            # Raw broker events (app:comms:* / app:actor:*) forwarded by the runtime process.
            if channel.startswith("broker:"):
                try:
                    self._handle_broker_event(
                        channel=channel[len("broker:") :],
                        event=event,
                    )
                except Exception:
                    return
                try:
                    rt.dirty_logs = True
                except Exception:
                    pass
                return

        def _handle_broker_event(self, *, channel: str, event: dict) -> None:
            """
            Update log buffers from broker events so the GUI log panes stay populated.

            `event` is expected to match `unity.conversation_manager.events.Event.to_dict()`.
            """

            rt = self.runtime
            name = str(event.get("event_name") or "")
            payload = (
                event.get("payload") if isinstance(event.get("payload"), dict) else {}
            )

            # Choose category based on channel prefix.
            cat = "cm"
            if str(channel).startswith("app:actor:"):
                cat = "actor"

            # Extract handle_id for concurrent tracking
            actor_hid: int | None = None
            try:
                hid_val = payload.get("handle_id", -1)
                if hid_val is not None:
                    actor_hid = int(hid_val)
                    if actor_hid < 0:
                        actor_hid = None
            except Exception:
                actor_hid = None

            # NOTE: Store full message; truncation happens at render time in log_aggregator.
            msg = name or "Event"
            try:
                # SMS events
                if name == "SMSReceived":
                    content = str(payload.get("content") or "").strip()
                    if content:
                        msg = f"SMSReceived: {content}"
                elif name == "SMSSent":
                    content = str(payload.get("content") or "").strip()
                    if content:
                        msg = f"SMSSent: {content}"
                # Unify console message events
                elif name == "UnifyMessageReceived":
                    content = str(payload.get("content") or "").strip()
                    attachments = payload.get("attachments") or []
                    if attachments:
                        msg = f"UnifyMessageReceived: {content} [+{len(attachments)} files]"
                    elif content:
                        msg = f"UnifyMessageReceived: {content}"
                elif name == "UnifyMessageSent":
                    content = str(payload.get("content") or "").strip()
                    attachments = payload.get("attachments") or []
                    if attachments:
                        msg = f"UnifyMessageSent: {content} [+{len(attachments)} files]"
                    elif content:
                        msg = f"UnifyMessageSent: {content}"
                # Email events
                elif name == "EmailSent":
                    subj = str(payload.get("subject") or "").strip()
                    if subj:
                        msg = f"EmailSent: {subj}"
                elif name == "EmailReceived":
                    subj = str(payload.get("subject") or "").strip()
                    attachments = payload.get("attachments") or []
                    if attachments:
                        msg = f"EmailReceived: {subj} [+{len(attachments)} files]"
                    elif subj:
                        msg = f"EmailReceived: {subj}"
                # Phone call state events
                elif name == "PhoneCallReceived":
                    msg = "PhoneCallReceived"
                elif name == "PhoneCallStarted":
                    msg = "PhoneCallStarted"
                elif name == "PhoneCallAnswered":
                    msg = "PhoneCallAnswered"
                elif name == "PhoneCallNotAnswered":
                    reason = str(payload.get("reason") or "").strip()
                    msg = (
                        f"PhoneCallNotAnswered: {reason}"
                        if reason
                        else "PhoneCallNotAnswered"
                    )
                elif name == "PhoneCallEnded":
                    msg = "PhoneCallEnded"
                # Unify Meet state events
                elif name == "UnifyMeetReceived":
                    msg = "UnifyMeetReceived"
                elif name == "UnifyMeetStarted":
                    msg = "UnifyMeetStarted"
                elif name == "UnifyMeetEnded":
                    msg = "UnifyMeetEnded"
                # Phone/meeting utterance events
                elif name in {
                    "InboundPhoneUtterance",
                    "OutboundPhoneUtterance",
                    "InboundUnifyMeetUtterance",
                    "OutboundUnifyMeetUtterance",
                }:
                    content = str(payload.get("content") or "").strip()
                    if content:
                        msg = f"{name}: {content}"
                # Voice interrupt
                elif name == "VoiceInterrupt":
                    msg = "VoiceInterrupt"
                # Call guidance
                elif name == "FastBrainNotification":
                    content = str(payload.get("content") or "").strip()
                    if content:
                        msg = f"FastBrainNotification: {content}"
                # Other useful events
                elif name == "UnknownContactCreated":
                    medium = str(payload.get("medium") or "").strip()
                    preview = str(payload.get("message_preview") or "").strip()
                    if preview:
                        msg = f"UnknownContactCreated ({medium}): {preview}"
                    else:
                        msg = f"UnknownContactCreated ({medium})"
                elif name == "DirectMessageEvent":
                    content = str(payload.get("content") or "").strip()
                    source = str(payload.get("source") or "").strip()
                    if content:
                        msg = f"DirectMessage [{source}]: {content}"
                elif name == "Error":
                    message = str(payload.get("message") or "").strip()
                    if message:
                        msg = f"Error: {message}"
                # Actor events
                elif name == "ActorHandleStarted":
                    q = str(payload.get("query") or "").strip()
                    if q:
                        msg = f"ActorHandleStarted: {q}"
                    # Set handle context for subsequent events
                    if actor_hid is not None:
                        try:
                            rt.event_tree_display.set_handle_context(
                                handle_id=actor_hid,
                            )
                        except Exception:
                            pass
                        try:
                            rt.trace_display.set_event_context(
                                event_id=f"handle-{actor_hid}",
                                handle_id=actor_hid,
                            )
                        except Exception:
                            pass
                        try:
                            rt.log_aggregator.set_handle_context(handle_id=actor_hid)
                        except Exception:
                            pass
                elif name == "ActorClarificationRequest":
                    q = str(payload.get("query") or "").strip()
                    if q:
                        msg = f"ActorClarificationRequest: {q}"
                elif name == "ActorClarificationResponse":
                    r = str(payload.get("response") or "").strip()
                    if r:
                        msg = f"ActorClarificationResponse: {r}"
                elif name == "ActorNotification":
                    r = str(payload.get("response") or "").strip()
                    if r:
                        msg = f"ActorNotification: {r}"
                elif name == "ActorResult":
                    r = str(payload.get("result") or "").strip()
                    if r:
                        msg = f"ActorResult: {r}"
                    # Mark handle tree as completed
                    if actor_hid is not None:
                        try:
                            rt.event_tree_display.mark_handle_completed(actor_hid)
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                rt.log_aggregator.handle_structured_event(
                    category=cat,  # type: ignore[arg-type]
                    message=msg,
                    handle_id=actor_hid,
                )
            except Exception:
                pass

            # Best-effort TTS in call mode: the user should hear assistant-side
            # outputs (guidance to the voice agent and assistant utterances).
            try:
                if name in {"OutboundPhoneUtterance", "FastBrainNotification"}:
                    content = str(payload.get("content") or "").strip()
                    if content:
                        self._maybe_tts(content)
            except Exception:
                pass

        def _maybe_tts(self, text: str) -> None:
            """
            Speak `text` in the UI process (call mode only).

            In the multi-process architecture the worker must not own audio/TTY.
            """

            rt = self.runtime
            if not bool(getattr(rt.args, "voice", False)):
                return
            if not bool(getattr(rt.state, "in_call", False)):
                return
            txt = (text or "").strip()
            if not txt:
                return

            # De-dupe repeats over short windows.
            try:
                now = time.monotonic()
                if (
                    rt.last_tts_text == txt
                    and (now - float(rt.last_tts_at or 0.0)) < 2.0
                ):
                    return
                rt.last_tts_text = txt
                rt.last_tts_at = now
            except Exception:
                pass

            try:
                # Use stdin-safe speech for Textual UI; the normal "press enter to skip"
                # listener competes with Textual's input handling and can cause typing lag.
                from sandboxes.utils import speak_no_stdin as _speak

                asyncio.create_task(asyncio.to_thread(_speak, txt))
            except Exception:
                return

        def _set_input_enabled(self, enabled: bool) -> None:
            try:
                scr = self.screen
                inp = scr.query_one("#command_input", Input)
                btn = scr.query_one("#submit_command", Button)
                inp.disabled = not enabled
                btn.disabled = not enabled
            except Exception:
                pass

        def _update_input_placeholder(self) -> None:
            rt = self.runtime
            active = bool(getattr(rt.state, "active", False))
            placeholder = (
                "/ask, /i, /pause, /resume, /stop"
                if active
                else "Type a command: sms, email, call, us, ..."
            )
            try:
                inp = self.screen.query_one("#command_input", Input)
                inp.placeholder = placeholder
            except Exception:
                pass

        def _handle_logs_locally(self, *, kind: str, args: str) -> None:
            rt = self.runtime
            lg = rt.log_aggregator
            raw = (args or "").strip().lower()
            cats: list[str]
            if raw in {"cm", "actor", "manager"}:
                cats = [raw]
            elif raw == "all":
                cats = ["cm", "actor", "manager"]
            else:
                self.post_message(
                    AppendLine(
                        "⚠️ Usage: show_logs <cm|actor|manager|all>  or  collapse_logs <cm|actor|manager|all>",
                    ),
                )
                return
            if kind == "show_logs":
                for c in cats:
                    try:
                        lg.expand(c)  # type: ignore[arg-type]
                    except Exception:
                        pass
            else:
                for c in cats:
                    try:
                        lg.collapse(c)  # type: ignore[arg-type]
                    except Exception:
                        pass
            try:
                rt.dirty_logs = True
                rt.dirty_trace = True
            except Exception:
                pass

        def _save_state_from_ui(self, args: str) -> list[str]:
            rt = self.runtime
            snapshot = capture_snapshot(
                log_aggregator=rt.log_aggregator,
                event_tree_display=rt.event_tree_display,
                trace_display=rt.trace_display,
                conversation_lines=list(rt.conversation_lines),
            )

            repo_root = Path(__file__).resolve().parents[2]
            if args and args.strip():
                json_path = repo_root / args.strip()
            else:
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                json_path = repo_root / f".sandbox_state_{timestamp}.json"

            try:
                save_snapshot(snapshot, json_path)
            except Exception as exc:
                return [f"❌ Failed to save state: {exc}"]

            try:
                text_path = json_path.with_suffix(".txt")
                text_content = render_snapshot_text(snapshot)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
            except Exception as exc:
                return [
                    f"💾 State saved to: {json_path}",
                    f"⚠️ Failed to save text version: {exc}",
                ]

            result_lines = [
                "💾 State saved:",
                f"   JSON: {json_path}",
                f"   Text: {text_path}",
                (
                    f"   Summary: {snapshot.summary.get('total_conversation_lines', 0)} conversation lines, "
                    f"{snapshot.summary['total_cm_logs']} CM logs, "
                    f"{snapshot.summary['total_actor_logs']} actor logs, "
                    f"{snapshot.summary['total_manager_logs']} manager logs, "
                    f"{snapshot.summary['total_traces']} traces, "
                    f"{snapshot.summary['total_event_trees']} trees"
                ),
            ]

            import os

            _launch_cwd = os.environ.get("UNITY_SANDBOX_LAUNCH_CWD", "").strip()
            _voice_root = Path(_launch_cwd).resolve() if _launch_cwd else repo_root
            voice_log = _voice_root / ".logs_voice_agent.txt"
            if voice_log.exists():
                try:
                    from sandboxes.conversation_manager.call_transcript import (
                        build_timeline,
                        format_timeline,
                        parse_cm_log,
                        parse_voice_log,
                    )

                    voice_data = parse_voice_log(voice_log)
                    cm_log = _voice_root / ".logs_conversation_sandbox.txt"
                    cm_data = parse_cm_log(cm_log) if cm_log.exists() else None
                    if voice_data.utterances:
                        timeline = build_timeline(voice_data, cm_data)
                        transcript_path = json_path.with_name(
                            json_path.stem + "_transcript.txt",
                        )
                        with open(transcript_path, "w") as f:
                            f.write(format_timeline(timeline, verbose=True))
                        result_lines.append(f"   Transcript: {transcript_path}")
                except Exception as exc:
                    result_lines.append(f"   ⚠️ Transcript failed: {exc}")

            return result_lines

        async def _record_and_transcribe_best_effort(self) -> str:
            """
            Record audio and transcribe in the UI process (blocking acceptable).
            """

            if not bool(getattr(self.runtime.args, "voice", False)):
                self.post_message(
                    AppendLine("⚠️ Restart with `--voice` to enable recording."),
                )
                return ""
            try:
                from sandboxes.utils import (
                    record_for_seconds,
                    transcribe_deepgram_no_input,
                )
            except Exception as exc:
                self.post_message(AppendLine(f"⚠️ Voice mode unavailable ({exc})."))
                return ""
            try:
                audio = await asyncio.to_thread(record_for_seconds, 6.0)
                text = (
                    await asyncio.to_thread(transcribe_deepgram_no_input, audio) or ""
                ).strip()
                return text
            except Exception as exc:
                self.post_message(AppendLine(f"❌ Voice transcription failed: {exc}"))
                return ""
