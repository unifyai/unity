"""
T.B.D

For now, to run make sure textual is installed, simply do python new_start_chat.py.
To Exit the Terminal run `Ctrl + q`.
"""



import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Header, Static, Button, Input
from textual.containers import VerticalScroll, HorizontalGroup, Container
from textual.reactive import reactive
from textual import work
from textual.worker import Worker, WorkerState

from unify import AsyncUnify

from dotenv import load_dotenv

load_dotenv()

client = AsyncUnify(endpoint="gpt-4o@openai")

class Message(HorizontalGroup):

    def __init__(self, role: str, content: str, date: datetime.date=None, *args, **kwargs):
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
            msg.styles.border = ("round", "blue" if self.role=="user" else "red")
            msg.border_title = "You" if self.role == "user" else "AI"
            msg.border_subtitle = str(self.date)
            msg.styles.max_height = "100%"
            msg.styles.height = "auto"

            yield msg

class MessagesView(VerticalScroll):
    messages = reactive([], recompose=True)
    ai_typing = reactive(False, recompose=True)

    def compose(self) -> ComposeResult:
        self.styles.align_horizontal = "center"
        yield from [Message(role=msg["role"], content=msg["content"], date=msg["date"]) for msg in self.messages] + ([Message(
            role="typing", content=""
        )] if self.ai_typing else [])

    
class ChatApp(App):
    def __init__(self, *args, **kwargs):
        self.llm_worker: Worker = None
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Header()
        yield MessagesView()
        yield Input(placeholder="Enter your Message", id="message_input")
    
    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "message_input":
            msg_view = self.query_one(MessagesView)
            print(msg_view.messages)
            msg_view.messages = msg_view.messages + [{"role": "user", "content": event.input.value, "date": datetime.now()}] 
            msg_view.scroll_end()
            print(msg_view.messages)
            event.input.value = ""
            self.llm_worker = self.llm_response()

    def on_input_changed(self, event: Input.Changed):
        if self.llm_worker:
            was_running = self.llm_worker.state == WorkerState.RUNNING
            if was_running and event.value != "":
                self.llm_worker.cancel()
                msg_view = self.query_one(MessagesView)
                msg_view.ai_typing = False
                self.llm_worker = self.llm_response()
            


    @work(exclusive=True)
    async def llm_response(self):
        msg_view = self.query_one(MessagesView)
        msg_view.ai_typing = False
        await asyncio.sleep(1.0)
        msg_view.ai_typing = True
        await asyncio.sleep(2.0)
        res = await client.generate(messages=[{"role": m["role"], "content": m["content"]} for m in msg_view.messages])
        msg_view.ai_typing = False
        msg_view.messages = msg_view.messages + [{"role": "assistant", "content": res, "date": datetime.now()}]
        msg_view.scroll_end()
        self.llm_worker = None


app = ChatApp()
app.theme = "textual-light"
app.run()