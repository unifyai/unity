from typing import Any, Type

from unity.common.async_tool_loop import SteerableToolHandle
from unity.manager_registry import ManagerRegistry
from .computer_backends import (
    ComputerBackend,
    MagnitudeBackend,
    MockComputerBackend,
)


class Computer:
    """
    Encapsulates all computer use capabilities, from simple actions
    to complex, multi-step operations and session recording. This class uses
    a strategy pattern to delegate to a specific backend implementation
    based on the selected mode.

    Supports both web automation and general desktop/computer control
    via vision-based agents (Magnitude).

    Modes:
        - 'magnitude': Production backend using Magnitude agent service
        - 'mock': Lightweight mock backend for testing (no external services)
    """

    def __init__(
        self,
        mode: str = "magnitude",
        secret_manager=None,
        **kwargs,
    ):
        """
        Initializes the Computer with a specific backend strategy.

        Args:
            mode (str): The backend to use. Can be 'magnitude' or 'mock'.
            **kwargs: Arguments to pass to the backend constructor (e.g., headless, agent_server_url).
        """

        if mode == "magnitude":
            self.backend: ComputerBackend = MagnitudeBackend(**kwargs)
        elif mode == "mock":
            self.backend: ComputerBackend = MockComputerBackend(**kwargs)
        else:
            raise ValueError(
                f"Unknown computer mode: '{mode}'. Must be 'magnitude' or 'mock'.",
            )

        self._secret_manager = (
            ManagerRegistry.get_secret_manager()
            if secret_manager is None
            else secret_manager
        )

    async def act(self, instruction: str, expectation: str = "") -> str:
        """Executes a single, high-level action by delegating to the active backend."""
        instruction = self._secret_manager.from_placeholder(instruction)
        return await self.backend.act(instruction, expectation)

    async def observe(self, query: str, response_format: Type = str) -> Any:
        """Asks a question by delegating to the active backend."""
        query = self._secret_manager.from_placeholder(query)
        return await self.backend.observe(query, response_format)

    async def navigate(self, url: str) -> str:
        """Navigates to a specific URL by delegating to the active backend."""
        return await self.backend.navigate(url)

    async def get_screenshot(self) -> str:
        """Gets a screenshot by delegating to the active backend."""
        return await self.backend.get_screenshot()

    async def get_current_url(self) -> str:
        """Gets the current URL by delegating to the active backend."""
        return await self.backend.get_current_url()

    def stop(self):
        """Shuts down the underlying backend."""
        self.backend.stop()

    # --- Placeholders for other planned methods ---
    async def multi_step(self, description: str) -> SteerableToolHandle:
        """
        Performs a complex, sequential computer task using a dedicated sub-agent.
        Use this for high-level goals like "Log into my account" or "Find the latest blog post and summarize it."
        Returns a handle to the sub-agent that will execute the task.
        """
        raise NotImplementedError(
            "multi_step method is not yet implemented. Use HierarchicalActor or CodeActActor instead.",
        )

    async def reason(self, query: str) -> str:
        """
        Asks a question about the current state of the page/screen.
        e.g., "What is the title of the page?", "Is there a button with the text 'Submit'?"
        """
        # TODO: Implement reasoning logic
        print("Starting computer reasoning...")

    def start_recording(
        self,
        include_video: bool = True,
        include_transcript: bool = True,
    ):
        # TODO: Implement screen recording logic
        print("Starting screen recording...")

    def stop_recording(self):
        # TODO: Implement logic to stop and save the recording
        print("Stopping screen recording...")

    def seed(self, state: Any):
        # TODO: Implement logic to reset the computer to a specific state
        print("Seeding computer state...")
