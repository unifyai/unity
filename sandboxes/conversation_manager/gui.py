from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from sandboxes.conversation_manager.command_router import CommandRouter
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.steering import is_active
from sandboxes.utils import steering_controls_hint

LG = logging.getLogger("conversation_manager_sandbox")

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


@dataclass
class GuiRuntime:
    cm: Any
    args: Any
    state: Any
    publisher: EventPublisher


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
                        yield Label("", id="steering_hint")
                        yield RichLog(id="responses", wrap=True, highlight=True)
                with Horizontal(id="cmd_row"):
                    yield Input(
                        placeholder="Commands: sms, email, call, say, end_call, us, /pause, /i, /ask, /stop",
                        id="command_input",
                    )
                    yield Button("Send", id="submit_command")
            yield Footer()

        def compose_left(self) -> ComposeResult:  # overridden
            yield Label("Not implemented")

        def on_mount(self) -> None:
            # Periodic update of steering hint visibility.
            self.set_interval(0.25, self._refresh_hint)

        def _refresh_hint(self) -> None:
            app = self.app  # type: ignore[attr-defined]
            rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
            hint = self.query_one("#steering_hint", Label)
            active = is_active(rt.cm, rt.state)
            hint.update(
                (
                    steering_controls_hint(
                        pending_clarification=False,
                        voice_enabled=False,
                    )
                    if active
                    else ""
                ),
            )

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
            yield Label("Simulate phone call", id="title")
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
                await self._route_raw("call")
            elif event.button.id == "call_end":
                await self._route_raw("end_call")
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
            #trace_panel { height: 14; }
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
                            yield Button("Start Call", id="btn_call_start")
                            yield Button("End Call", id="btn_call_end")
                            yield Button("Toggle Trace Panel", id="btn_toggle_trace")
                            yield Button("Quit", id="btn_quit")
                        with Vertical(id="right_tabs"):
                            yield Label("", id="steering_hint")
                            with TabbedContent(id="tabs"):
                                with TabPane("Event Tree", id="tab_tree"):
                                    yield Tree("ConversationManager", id="event_tree")
                                    yield Label("", id="tree_details")
                                with TabPane("Computer", id="tab_computer"):
                                    yield Static("", id="computer_status")
                    yield RichLog(id="conversation", wrap=True, highlight=True)
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
                            RichLog(id="logs_cm", wrap=True, highlight=True),
                            title="CM Logs",
                            id="coll_cm",
                            collapsed=False,
                        )
                        yield Collapsible(
                            RichLog(id="logs_actor", wrap=True, highlight=True),
                            title="Actor Logs",
                            id="coll_actor",
                            collapsed=True,
                        )
                        yield Collapsible(
                            RichLog(id="logs_manager", wrap=True, highlight=True),
                            title="Manager Logs",
                            id="coll_manager",
                            collapsed=True,
                        )
                    with Collapsible(
                        RichLog(id="trace_panel", wrap=True, highlight=True),
                        title="Trace (CodeAct)",
                        id="coll_trace",
                        collapsed=True,
                    ):
                        pass
                yield Footer()

            def on_mount(self) -> None:
                # Install GUI sink for sandbox subscriber.
                app = self.app  # type: ignore[attr-defined]
                rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]

                def _sink(line: str) -> None:
                    try:
                        conv = self.query_one("#conversation", RichLog)
                        conv.write(line)
                    except Exception:
                        pass
                    # New outbound events/notifications should refresh log + trace panels.
                    try:
                        self.post_message(RefreshPanels(logs=True, trace=True))
                    except Exception:
                        pass

                setattr(rt.args, "_gui_line_sink", _sink)
                # Allow non-UI code (subscriber / executor wrapper) to request a refresh.
                try:
                    setattr(
                        rt.args,
                        "_gui_refresh_request",
                        lambda **kw: self.post_message(RefreshPanels(**kw)),
                    )
                except Exception:
                    pass

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

                # Periodic refresh of hint + computer panel.
                self.set_interval(0.25, self._refresh_hint)
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
                        "coll_manager",
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

            def _refresh_hint(self) -> None:
                app = self.app  # type: ignore[attr-defined]
                rt: GuiRuntime = app.runtime  # type: ignore[attr-defined]
                hint = self.query_one("#steering_hint", Label)
                active = is_active(rt.cm, rt.state)
                hint.update(
                    (
                        steering_controls_hint(
                            pending_clarification=False,
                            voice_enabled=False,
                        )
                        if active
                        else ""
                    ),
                )

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
                    root = tree_disp.get_tree_data()
                    if root is None:
                        return
                    widget = self.query_one("#event_tree", Tree)

                    # If a new execution started, reset expansion state.
                    root_key = str(
                        getattr(root, "call_id", None)
                        or getattr(root, "label", "")
                        or "",
                    )
                    if root_key and root_key != getattr(
                        self,
                        "_last_tree_root_key",
                        "",
                    ):
                        self._tree_expanded_paths.clear()
                        self._last_tree_root_key = root_key

                    # Rebuild tree (simple and robust).
                    widget.root.label = str(root.label)
                    try:
                        widget.root.data = root  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    # Textual's TreeNodes doesn't implement list.clear(); remove nodes instead.
                    try:
                        for child in list(widget.root.children):
                            try:
                                child.remove()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    def _add(parent, node):
                        icon = {
                            "completed": "✓",
                            "in_progress": "⏳",
                            "error": "❌",
                        }.get(
                            getattr(node, "status", "in_progress"),
                            "•",
                        )
                        label = f"{icon} {getattr(node, 'label', '')}"
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
                            _add(child, c)

                    for c in getattr(root, "children", []) or []:
                        _add(widget.root, c)

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
                    mgr_log = self.query_one("#logs_manager", RichLog)
                except Exception:
                    return
                # Replace content (RichLog may not expose clear(); best-effort).
                try:
                    cm_log.clear()  # type: ignore[attr-defined]
                    actor_log.clear()  # type: ignore[attr-defined]
                    mgr_log.clear()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    cm_log.write(lg.render_expanded("cm"))
                    actor_log.write(lg.render_expanded("actor"))
                    mgr_log.write(lg.render_expanded("manager"))
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
                            # Show the full trajectory across the session so earlier turns
                            # remain visible even after the sandbox starts a new ActorHandle
                            # (which resets per-event turn numbering).
                            try:
                                tr.write(td.render_all())
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
                    activity = getattr(rt.args, "_computer_activity", None)
                    if activity is None:
                        w.update("Computer not available.")
                        return
                    snap = activity.snapshot_sync()
                    connected = snap.get("connected", None)
                    url = snap.get("last_url") or "(unknown URL)"
                    actions = snap.get("actions") or []
                    last = actions[-3:] if len(actions) > 3 else actions
                    last_lines = (
                        "\n".join(
                            [f"- {a.kind}: {a.detail}" for a in last],
                        )
                        if last
                        else "(no activity yet)"
                    )
                    status = (
                        "connected"
                        if connected
                        else ("disconnected" if connected is False else "unknown")
                    )
                    w.update(f"Computer: {status}\nURL: {url}\nRecent:\n{last_lines}")
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
                if event.button.id == "btn_call_start":
                    try:
                        app.post_message(AppendLine("[ui] Start Call pressed"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    await app.route_command("call")  # type: ignore[attr-defined]
                    return
                if event.button.id == "btn_call_end":
                    try:
                        app.post_message(AppendLine("[ui] End Call pressed"))  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    await app.route_command("end_call")  # type: ignore[attr-defined]
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
                if event.button.id == "btn_toggle_trace":
                    try:
                        coll = self.query_one("#coll_trace", Collapsible)
                        coll.collapsed = not coll.collapsed
                    except Exception:
                        pass

    class ModernizedMessagingApp(App):
        CSS = """
        #root { height: 1fr; }
        #body { height: 1fr; }
        #left { width: 45%; padding: 1; }
        #right { width: 55%; padding: 1; }
        #responses { height: 1fr; border: round $surface; }
        #steering_hint { height: auto; color: $text-muted; }
        #cmd_row { height: auto; }
        #command_input { width: 1fr; }
        """

        def __init__(self, runtime: GuiRuntime):
            super().__init__()
            self.runtime = runtime

        def compose(self) -> ComposeResult:
            yield Header()
            yield Footer()

        def on_mount(self) -> None:
            if _TEXTUAL_ADVANCED_AVAILABLE:
                self.push_screen(DashboardScreen())
            else:
                self.push_screen(MenuScreen())

        async def route_command(self, raw: str) -> None:
            rt = self.runtime
            st = rt.state
            router = CommandRouter(
                cm=rt.cm,
                args=rt.args,
                state=st,
                publisher=rt.publisher,
                chat_history=getattr(st, "chat_history", []),
                allow_voice=bool(getattr(rt.args, "voice", False)),
                allow_save_project=False,
                config_manager=getattr(rt.args, "_config_manager", None),
                trace_display=getattr(rt.args, "_trace_display", None),
                event_tree_display=getattr(rt.args, "_event_tree_display", None),
                log_aggregator=getattr(rt.args, "_log_aggregator", None),
            )
            res = await router.execute_raw(
                raw,
                prompt_text=None,
                in_call=bool(st.in_call),
            )
            for ln in res.lines:
                self.post_message(AppendLine(ln))
            if res.should_exit:
                self.exit()

        async def on_append_line(self, msg: AppendLine) -> None:
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


async def run_gui_mode(*, cm: Any, args: Any, state: Any) -> bool:
    """
    Run GUI mode (Textual) in-process.

    Returns:
        True if GUI ran, False if Textual unavailable (caller should fallback to REPL).
    """
    if not _TEXTUAL_AVAILABLE:
        return False

    publisher = EventPublisher(
        cm=cm,
        state=state,
    )
    runtime = GuiRuntime(cm=cm, args=args, state=state, publisher=publisher)
    app = ModernizedMessagingApp(runtime)
    await app.run_async()
    return True
