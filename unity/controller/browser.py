from typing import Any, Type
from .controller import Controller
from unity.common.llm_helpers import (
    SteerableToolHandle,
)
from unity.planner.tool_loop_planner import ToolLoopPlanner


class Browser:
    """
    Encapsulates all browser-related capabilities, from simple actions
    to complex, multi-step operations and session recording.
    """

    def __init__(
        self,
        session_connect_url: str | None = None,
        headless: bool = False,
        mode: str = "heuristic",
    ):
        self.controller = Controller(
            session_connect_url=session_connect_url,
            headless=headless,
            mode=mode,
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

    async def multi_step(self, description: str) -> SteerableToolHandle:
        """
        Performs a complex, sequential browser task using a dedicated sub-agent.
        Use this for high-level goals like "Log into my account" or "Find the latest blog post and summarize it."
        Returns a handle to the sub-agent that will execute the task.
        """
        sub_planner = ToolLoopPlanner(
            session_connect_url=self.controller.session_connect_url,
            headless=self._headless,
        )
        active_task_handle = await sub_planner.execute(description)
        return active_task_handle

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

    def stop(self):
        """Shuts down the underlying controller."""
        if self.controller.is_alive():
            self.controller.stop()
            self.controller.join(timeout=5.0)
