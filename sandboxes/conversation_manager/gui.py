import os
import asyncio
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label
from textual.containers import Horizontal
from unity.conversation_manager.utils import publish_event
from unity.conversation_manager.events import (
    SMSMessageRecievedEvent,
    EmailRecievedEvent,
    PhoneCallStopEvent,
    PhoneCallInitiatedEvent,
    WhatsappMessageRecievedEvent,
)
from dotenv import load_dotenv
import sys, pathlib

load_dotenv(override=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


PROJECT_NAME = "ConversationManagerSandbox"


def send_sms(message: str) -> None:
    # Publish an SMS received event for the user
    ev = {
        "topic": os.getenv("USER_PHONE_NUMBER"),
        "event": SMSMessageRecievedEvent(
            content=message,
            role="User",
        ).to_dict(),
    }
    asyncio.create_task(publish_event(ev))


def send_email(message: str) -> None:
    # Publish an Email received event for the user
    ev = {
        "topic": os.getenv("USER_PHONE_NUMBER"),
        "event": EmailRecievedEvent(
            content=message,
            role="User",
        ).to_dict(),
    }
    asyncio.create_task(publish_event(ev))


def send_whatsapp(message: str) -> None:
    # Publish a WhatsApp received event for the user
    ev = {
        "topic": os.getenv("USER_PHONE_NUMBER"),
        "event": WhatsappMessageRecievedEvent(
            content=message,
            role="User",
        ).to_dict(),
    }
    asyncio.create_task(publish_event(ev))


class MenuScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()

        # Actions menu, split into two rows to avoid overflow
        yield Horizontal(
            Button("Send SMS", id="sms", classes="small"),
            Button("Send WhatsApp", id="whatsapp", classes="small"),
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
        elif event.button.id == "whatsapp":
            self.app.push_screen(WhatsAppScreen())
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
        # Task name and description inputs
        yield Label("Task Name:")
        yield Input(placeholder="Enter task name", id="task_name")
        yield Label("Task Description:")
        yield Input(placeholder="Enter task description", id="task_description")
        yield Label("Purpose:")
        yield Input(placeholder="Enter purpose", id="purpose", value="general")
        yield Horizontal(
            Button("Call", id="send_call"),
            Button("End Call", id="end_call"),
            Button("Back", id="back"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_call":
            # Gather task details
            task_name = self.query_one("#task_name", Input).value
            task_description = self.query_one("#task_description", Input).value
            purpose = self.query_one("#purpose", Input).value
            # Publish a PhoneCallInitiatedEvent with task_context
            ev = {
                "topic": os.getenv("USER_PHONE_NUMBER"),
                "event": PhoneCallInitiatedEvent(
                    purpose=purpose,
                    task_context={"name": task_name, "description": task_description},
                ).to_dict(),
            }
            asyncio.create_task(publish_event(ev))
        elif event.button.id == "end_call":
            ev = {
                "topic": os.getenv("USER_PHONE_NUMBER"),
                "event": PhoneCallStopEvent().to_dict(),
            }
            asyncio.create_task(publish_event(ev))
        elif event.button.id == "back":
            self.app.pop_screen()


# class MeetScreen(Screen):
#     def compose(self) -> ComposeResult:
#         yield Header()
#         yield Label("Task Name:")
#         yield Input(placeholder="Enter task name", id="task_name")
#         yield Label("Task Description:")
#         yield Input(placeholder="Enter task description", id="task_description")
#         yield Label("Purpose:")
#         yield Input(placeholder="Enter purpose", id="purpose", value="general")
#         yield Label("Meet ID:")
#         yield Input(placeholder="Enter meet ID", id="meet_id")
#         yield Horizontal(
#             Button("Join Meet", id="send_meet"),
#             Button("End Meet", id="end_meet"),
#             Button("Back", id="back"),
#         )
#         yield Footer()

#     def on_button_pressed(self, event: Button.Pressed) -> None:
#         if event.button.id == "send_meet":
#             task_name = self.query_one("#task_name", Input).value
#             task_description = self.query_one("#task_description", Input).value
#             purpose = self.query_one("#purpose", Input).value
#             meet_id = self.query_one("#meet_id", Input).value
#             ev = {
#                 "topic": os.getenv("USER_PHONE_NUMBER"),
#                 "event": PhoneCallInitiatedEvent(
#                     purpose=purpose,
#                     task_context={"name": task_name, "description": task_description},
#                     meet_id=meet_id,
#                 ).to_dict(),
#             }
#             asyncio.create_task(publish_event(ev))
#         elif event.button.id == "end_meet":
#             ev = {
#                 "topic": os.getenv("USER_PHONE_NUMBER"),
#                 "event": PhoneCallStopEvent().to_dict(),
#             }
#             asyncio.create_task(publish_event(ev))
#         elif event.button.id == "back":
#             self.app.pop_screen()


class WhatsAppScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Message:")
        yield Input(placeholder="Enter message", id="message")
        yield Horizontal(
            Button("Send", id="send_whatsapp"),
            Button("Back", id="back"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send_whatsapp":
            message = self.query_one("#message", Input).value
            send_whatsapp(message)
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
