import os
import asyncio
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label
from textual.containers import Horizontal
from unity.conversation_manager.utils import publish_event
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    PhoneCallStarted,
    PhoneCallEnded,
)
from dotenv import load_dotenv
import sys, pathlib

load_dotenv(override=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


PROJECT_NAME = "ConversationManagerSandbox"


def _get_simulated_user_contact() -> dict:
    """Build a simulated user contact dict for sandbox events."""
    return {
        "contact_id": 1,
        "first_name": os.getenv("USER_NAME", "User"),
        "phone_number": os.getenv("USER_PHONE_NUMBER", "+15550001234"),
        "email_address": os.getenv("USER_EMAIL", "user@example.com"),
    }


def send_sms(message: str) -> None:
    # Publish an SMS received event for the user
    ev = {
        "topic": os.getenv("USER_PHONE_NUMBER"),
        "event": SMSReceived(
            contact=_get_simulated_user_contact(),
            content=message,
        ).to_dict(),
    }
    asyncio.create_task(publish_event(ev))


def send_email(message: str) -> None:
    # Publish an Email received event for the user
    ev = {
        "topic": os.getenv("USER_PHONE_NUMBER"),
        "event": EmailReceived(
            contact=_get_simulated_user_contact(),
            subject="Sandbox Email",
            body=message,
        ).to_dict(),
    }
    asyncio.create_task(publish_event(ev))


class MenuScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()

        # Actions menu, split into two rows to avoid overflow
        yield Horizontal(
            Button("Send SMS", id="sms", classes="small"),
            Button("Send Email", id="email", classes="small"),
            id="menu_row1",
        )
        yield Horizontal(
            Button("Send Call", id="call", classes="small"),
            Button("Quit", id="quit", classes="small"),
            id="menu_row2",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sms":
            self.app.push_screen(SMSScreen())
        elif event.button.id == "email":
            self.app.push_screen(EmailScreen())
        elif event.button.id == "call":
            self.app.push_screen(CallScreen())
        elif event.button.id == "quit":
            self.app.exit()


class SMSScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Message:")
        yield Input(placeholder="Enter message", id="message")
        yield Horizontal(
            Button("Send", id="send_sms"),
            Button("Back", id="back"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_sms":
            message = self.query_one("#message", Input).value
            send_sms(message)
        elif event.button.id == "back":
            self.app.pop_screen()


class EmailScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Message:")
        yield Input(placeholder="Enter message", id="message")
        yield Horizontal(
            Button("Send", id="send_email"),
            Button("Back", id="back"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_email":
            message = self.query_one("#message", Input).value
            send_email(message)
        elif event.button.id == "back":
            self.app.pop_screen()


class CallScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Simulate a phone call with the user")
        yield Horizontal(
            Button("Start Call", id="send_call"),
            Button("End Call", id="end_call"),
            Button("Back", id="back"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_call":
            # Publish a PhoneCallStarted event
            ev = {
                "topic": os.getenv("USER_PHONE_NUMBER"),
                "event": PhoneCallStarted(
                    contact=_get_simulated_user_contact(),
                ).to_dict(),
            }
            asyncio.create_task(publish_event(ev))
        elif event.button.id == "end_call":
            ev = {
                "topic": os.getenv("USER_PHONE_NUMBER"),
                "event": PhoneCallEnded(
                    contact=_get_simulated_user_contact(),
                ).to_dict(),
            }
            asyncio.create_task(publish_event(ev))
        elif event.button.id == "back":
            self.app.pop_screen()


class MessagingApp(App):
    CSS = """
Screen {
    background: black;
    color: white;
}

Button {
    background: black;
    color: white;
    border: round white;
}

Button.-active {
    background: gray;
}
"""

    def on_mount(self) -> None:
        self.push_screen(MenuScreen())


if __name__ == "__main__":
    # override project name from first CLI argument
    if len(sys.argv) > 1:
        PROJECT_NAME = sys.argv[1]
    MessagingApp(ansi_color=True).run()
