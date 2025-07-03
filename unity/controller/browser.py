from typing import Any, Type
from .controller import Controller
from unity.common.llm_helpers import (
    SteerableToolHandle,
    methods_to_tool_dict,
    start_async_tool_use_loop,
)
import unify


class Browser:
    """
    Encapsulates all browser-related capabilities, from simple actions
    to complex, multi-step operations and session recording.
    """

    def __init__(self, session_connect_url: str | None = None, headless: bool = False):
        self.controller = Controller(
            session_connect_url=session_connect_url,
            headless=headless,
        )
        if not self.controller.is_alive():
            self.controller.start()

    async def act(self, instruction: str) -> str:
        """
        Executes a single, high-level action in the browser.
        e.g., "Click the 'Login' button", "Type 'hello world' in the search bar"

        """
        return await self.controller.act(instruction)

    async def observe(self, query: str, response_format: Type = str) -> Any:
        """
        Asks a question about the current state of the browser page.
        e.g., "What is the title of the page?", "Is there a button with the text 'Submit'?"
        """
        return await self.controller.observe(query, response_format)

    async def multi_step(
        self,
        description: str,
        parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        """
        Performs a complex, multi-step browser task by breaking it down into
        a sequence of `act` and `observe` calls.
        """
        client = unify.AsyncUnify("o4-mini@openai")
        client.set_system_message(
            "You are a browser assistant. Your goal is to achieve the user's objective by using the `act` and `observe` tools to interact with the web page.",
        )

        tools = methods_to_tool_dict(
            self.act,
            self.observe,
            include_class_name=False,
        )

        return start_async_tool_use_loop(
            client,
            description,
            tools,
            loop_id="browser_multi_step",
            parent_chat_context=parent_chat_context,
        )

    # --- Placeholders for other planned methods ---

    async def reason(self, query: str) -> str:
        """
        Asks a question about the current state of the browser page.
        e.g., "What is the title of the page?", "Is there a button with the text 'Submit'?"
        """
        # TODO: Implement reasoning logic
        print("Starting browser reasoning...")

    def start_recording(
        self,
        include_video: bool = True,
        include_transcript: bool = True,
    ):
        # TODO: Implement screen recording logic
        print("Starting browser recording...")

    def stop_recording(self):
        # TODO: Implement logic to stop and save the recording
        print("Stopping browser recording...")

    def seed(self, state: Any):
        # TODO: Implement logic to reset the browser to a specific state
        print("Seeding browser state...")

    def close(self):
        """Shuts down the underlying controller."""
        self.controller.stop()
