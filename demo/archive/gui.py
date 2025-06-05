"""
T.B.D

For now, to run make sure textual is installed, simply do python new_start_chat.py.
To Exit the Terminal run `Ctrl + q`.
"""

# STEPS:
# 1- Get everything wroking in chat only demo
# 2- Connect to voice demo
# 3- Make sure everything is working technical-wise
# 4- Refine abstractions and show it to people to play with


import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Header, Static, Button, Input, Tabs
from textual.containers import VerticalScroll, HorizontalGroup, VerticalGroup, Vertical
from textual.reactive import reactive

import json

from dotenv import load_dotenv

load_dotenv()

from new_terminal_helper import run_in_new_terminal
from events import *

# --- event-bus bootstrap ----------------------------------------------------
import subprocess, sys


# will build this out later
class EmailMessage: ...


class Message(HorizontalGroup):
    def __init__(
        self,
        role: str,
        content: str,
        date: datetime.date = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.role = role
        self.content = content
        self.date = date

        self.styles.width = "80%"
        self.styles.max_height = 30
        self.styles.height = "auto"
        # self.styles.margin = 1
        self.styles.align_horizontal = "right" if self.role == "user" else "left"
        self.styles.padding = 1

    def compose(self) -> ComposeResult:
        if self.role == "typing":
            msg = Static("AI is typing...")
            msg.styles.padding = 1
            msg.styles.color = "black"
            msg.styles.width = "auto"
            msg.styles.max_width = 50
            msg.styles.max_height = "100%"
            msg.styles.height = "auto"
            yield msg
        else:
            msg = VerticalScroll(Static(self.content, expand=True, markup=False))
            # msg.styles.dock = "right"
            msg.styles.padding = 1
            msg.styles.color = "black"
            msg.styles.min_width = 20
            msg.styles.max_width = 50
            msg.styles.border = ("round", "blue" if self.role == "user" else "red")
            msg.border_title = "You" if self.role == "user" else "AI"
            msg.border_subtitle = str(self.date)
            msg.styles.max_height = "100%"
            msg.styles.height = "auto"

            yield msg


class MessagesView(VerticalScroll):
    def __init__(self, messages, ai_typing=False, *args, **kwargs):
        self.messages = messages
        self.ai_typing = ai_typing
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        self.styles.align_horizontal = "center"
        yield from [
            Message(role=msg["role"], content=msg["content"], date=msg["date"])
            for msg in self.messages
        ] + (
            [
                Message(
                    role="typing",
                    content="",
                ),
            ]
            if self.ai_typing
            else []
        )


class PhoneView(Vertical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.styles.width = "100%"
        self.styles.height = "100%"
        self.styles.align = ("center", "middle")

    def compose(self) -> ComposeResult:
        yield Button("Call Assistant", id="call_button")
        yield Button("End Call", id="end_call_button")


class CommsView(VerticalGroup):
    threads = reactive(
        {
            "whatsapp": {
                "messages": [],
                "ai_typing": False,
            },
            "telegram": {
                "messages": [],
                "ai_typing": False,
            },
            "sms": {
                "messages": [],
                "ai_typing": False,
            },
        },
        recompose=True,
    )

    current_thread = reactive("whatsapp", recompose=True)

    def compose(self) -> ComposeResult:
        if self.current_thread in ["whatsapp", "telegram", "sms"]:
            yield MessagesView(
                self.threads[self.current_thread]["messages"],
                self.threads[self.current_thread]["ai_typing"],
            )
            yield Input(placeholder="Enter your Message", id="message_input")
        elif self.current_thread == "email":
            yield Static("T.B.D")
        else:  # phone view
            yield PhoneView()


class ChatApp(App):
    def __init__(self, *args, **kwargs):
        self.call_proc = None
        self.event_manager_proc = None
        self.writer: asyncio.StreamWriter | None = None
        self.reader: asyncio.StreamReader | None = None
        super().__init__(*args, **kwargs)

    async def on_mount(self) -> None:
        """
        Ensure an event-manager is running and end up with a live
        (reader, writer) pair in `self.reader` / `self.writer`.

        • If something is already listening on 127.0.0.1:8080 we reuse that
        very connection.
        • Otherwise we start our own copy of `python -m event_manager`
        and keep retrying until it is ready.
        """
        try:
            # ── Try to connect right away ───────────────────────────────
            self.reader, self.writer = await asyncio.open_connection("127.0.0.1", 8080)
            self.event_manager_proc = None  # we didn’t spawn it
        except ConnectionRefusedError:
            # ── Nothing was listening: launch the daemon ourselves ─────
            self.event_manager_proc = subprocess.Popen(
                [sys.executable, "-m", "event_manager"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Give it a moment and then retry up to three times.
            for _ in range(3):
                await asyncio.sleep(0.3)
                try:
                    self.reader, self.writer = await asyncio.open_connection(
                        "127.0.0.1",
                        8080,
                    )
                    break
                except ConnectionRefusedError:
                    continue
            else:
                raise RuntimeError(
                    "Unable to connect to event-manager on 127.0.0.1:8080",
                )

            print("connected!")

        async def handle_gui_events():
            while True:
                try:
                    raw = await self.reader.readline()
                    if not raw:
                        break
                    msg = json.loads(raw.decode())
                    # handle msg, put msg in the right thread
                    if msg["type"] == "update_gui":
                        if msg["thread"] == "whatsapp":
                            content = json.loads(msg["content"])["message"]
                        else:
                            content = msg["content"]
                        self._add_message_to_thread(
                            msg["thread"],
                            role="assistant",
                            content=content,
                        )
                        # should probably send back an even that the gui was updated right?
                except Exception as e:
                    print(e)
                    self.writer.close()
                    await self.writer.wait_closed()

        # create event watcher here
        asyncio.create_task(handle_gui_events())

    def compose(self) -> ComposeResult:
        comms_view = CommsView()
        comms_view.styles.height = "95%"
        comms_view.styles.width = "100%"

        yield Header()
        yield Tabs("Whatsapp", "Telegram", "SMS", "Email", "Phone")
        yield comms_view

    async def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "message_input":
            val = event.input.value
            msg_view = self.query_one(CommsView)
            curr_thread = msg_view.current_thread
            date = datetime.now()
            self._add_message_to_thread(curr_thread, "user", val, date)

            # TODO: fix hardcoded whatsapp here
            events_map = {
                "whatsapp": WhatsappMessageRecievedEvent,
                "telegram": TelegramMessageRecievedEvent,
                "sms": SMSMessageRecievedEvent,
            }
            await self.publish_event(
                {
                    "type": "user_agent_event",
                    "to": "pending",
                    "event": events_map[curr_thread](
                        content=val,
                        timestamp=date,
                        role="User",
                    ).to_dict(),
                },
            )

            event.input.value = ""

    async def on_input_changed(self, event: Input.Changed):
        # just publish User typing event to manager proc
        ...

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle TabActivated message sent by Tabs."""
        msg_view = self.query_one(CommsView)
        if event.tab is None:
            pass
        else:
            msg_view.current_thread = event.tab.label_text.lower()

    async def on_button_pressed(self, event: Button.Pressed):
        # should give control to the user voice agent
        if event.button.id == "call_button":
            if not self.call_proc:
                self.call_proc = run_in_new_terminal(
                    "call.py",
                    "console",  # ← keep this
                )
                # not sure if we should wait until the proc fully connects and is awake

        if event.button.id == "end_call_button":
            if self.call_proc:
                self.call_proc.terminate()
                self.call_proc = None

    def _add_message_to_thread(self, thread, role, content, date=None):
        comms_view = self.query_one(CommsView)
        msg = {
            "role": role,
            "content": content,
            "date": date if date else datetime.now(),
        }
        threads = comms_view.threads
        comms_view.threads = {
            **threads,
            thread: {
                **threads[thread],
                "messages": [*threads[thread]["messages"], msg],
            },
        }
        if thread == comms_view.current_thread:
            comms_view.scroll_end(animate=False)
        msg_view = comms_view.query_one(MessagesView)

        # not sure why this isn't taking effect
        msg_view.scroll_end(animate=False)

    # could be a textual worker, buts its too fast so no worries
    async def publish_event(self, ev: dict):
        ev = json.dumps(ev) + "\n"
        self.writer.write(ev.encode())
        await self.writer.drain()


if __name__ == "__main__":
    app = ChatApp()
    app.theme = "textual-light"
    app.run()
