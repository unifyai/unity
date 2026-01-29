from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sandboxes.conversation_manager.command_router import CommandRouter
from sandboxes.conversation_manager.event_publisher import EventPublisher
from sandboxes.conversation_manager.steering import is_active
from sandboxes.utils import steering_controls_hint

# -----------------------------------------------------------------------------
# Textual UI (optional dependency)
# -----------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.screen import Screen
    from textual.widgets import Button, Footer, Header, Input, Label, RichLog

    _TEXTUAL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency / runtime env
    _TEXTUAL_AVAILABLE = False


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
                allow_voice=False,  # GUI is text-only
                allow_save_project=False,
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
            # Write into the active screen's response log (present on all screens)
            try:
                scr = self.screen
                log = scr.query_one("#responses", RichLog)
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
