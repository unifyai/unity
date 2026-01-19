"""
Mocking infrastructure for the guided learning sandbox.

This module provides reusable mocks for running the HierarchicalActor in a safe,
side-effect-free environment. It includes:

- SimpleMockVerificationClient: Always returns "ok" for verification checks
- MockStateManagerHandle: A simple handle that returns a canned result
- mock_state_managers(): Sets up simulated managers for testing/demos
- mock_computer_primitives(): Configures MockComputerBackend with custom values

Usage:
    from sandboxes.guided_learning_manager.mocks import (
        SimpleMockVerificationClient,
        mock_state_managers,
        mock_computer_primitives,
    )

    # Set up actor with mocks (computer_mode="mock" uses MockComputerBackend)
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    mock_computer_primitives(actor, url="https://custom-url.com")  # Optional customization
    mock_state_managers(actor)
    active_task.verification_client = SimpleMockVerificationClient()
"""

from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock

from pydantic import BaseModel

from unity.actor.hierarchical_actor import HierarchicalActor, VerificationAssessment
from unity.common.async_tool_loop import SteerableToolHandle


# ────────────────────────────────────────────────────────────────────────────
# SimpleMockVerificationClient
# ────────────────────────────────────────────────────────────────────────────


class SimpleMockVerificationClient:
    """
    Mock verification client that always returns success.

    Use this for tests and demos that don't need to control verification outcomes.
    The client mimics the interface of the real verification client but always
    returns status="ok".

    Usage:
        active_task.verification_client = SimpleMockVerificationClient()

    Attributes:
        generate: AsyncMock that returns VerificationAssessment JSON with status="ok"
    """

    def __init__(self):
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_response_format(self, model: type[BaseModel]) -> None:
        """Set the expected response format (Pydantic model)."""
        self._current_format = model

    def reset_response_format(self) -> None:
        """Reset to default VerificationAssessment format."""
        self._current_format = VerificationAssessment

    def reset_messages(self) -> None:
        """Reset message history (no-op for mock)."""

    def set_system_message(self, message: str) -> None:
        """Set the system message (no-op for mock)."""

    async def _side_effect(self, *args, **kwargs) -> str:
        """Return a successful verification assessment."""
        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


# ────────────────────────────────────────────────────────────────────────────
# MockStateManagerHandle
# ────────────────────────────────────────────────────────────────────────────


class MockStateManagerHandle(SteerableToolHandle):
    """
    A simple mock handle that returns a canned result.

    Use this when mocking state manager methods that return handles.

    Usage:
        async def mock_update(text: str, **kwargs):
            return MockStateManagerHandle(f"Updated: {text}")

        primitives.tasks.update = AsyncMock(side_effect=mock_update)
    """

    def __init__(self, result: Any = "Mock result"):
        self._result = result
        self._done = True

    async def result(self) -> Any:
        """Return the canned result."""
        return self._result

    def done(self) -> bool:
        """Always returns True (mock completes immediately)."""
        return self._done

    async def stop(self, reason: Optional[str] = None, **kwargs) -> Optional[str]:
        """No-op stop."""
        return None

    def interject(self, message: str) -> "MockStateManagerHandle":
        """No-op interject, returns self."""
        return self

    async def pause(self) -> Optional[str]:
        """No-op pause."""
        return None

    async def resume(self) -> Optional[str]:
        """No-op resume."""
        return None

    async def ask(self, question: str) -> "MockStateManagerHandle":
        """No-op ask, returns self."""
        return self

    async def next_clarification(self) -> dict:
        """No-op next_clarification (never returns in practice)."""
        # Block forever since mock never needs clarification
        import asyncio

        await asyncio.Future()
        return {}

    async def next_notification(self) -> dict:
        """No-op next_notification (never returns in practice)."""
        import asyncio

        await asyncio.Future()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """No-op answer_clarification."""


# ────────────────────────────────────────────────────────────────────────────
# mock_computer_primitives
# ────────────────────────────────────────────────────────────────────────────


def mock_computer_primitives(
    actor: "HierarchicalActor",
    *,
    url: str = "https://mock-url.com",
    screenshot: str = "",  # Empty uses MockComputerBackend's default (valid PNG)
) -> None:
    """
    Set up mock computer primitives to avoid real browser operations.

    This configures the actor's ComputerPrimitives to use MockComputerBackend
    with the specified URL and screenshot values.

    NOTE: When creating a HierarchicalActor with computer_mode="mock", the
    MockComputerBackend is used automatically. This function is useful when
    you need to customize the mock's return values.

    Args:
        actor: The HierarchicalActor instance to mock
        url: URL to return from get_current_url()
        screenshot: Screenshot data to return from get_screenshot()

    Usage:
        actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
        mock_computer_primitives(actor, url="https://custom-url.com")
    """
    from unity.actor.hierarchical_actor import HierarchicalActor
    from unity.function_manager.computer_backends import MockComputerBackend

    if not isinstance(actor, HierarchicalActor):
        raise TypeError(f"Expected HierarchicalActor, got {type(actor).__name__}")

    # Use MockComputerBackend - the official mock implementation
    # Only pass screenshot if explicitly provided, otherwise use the default (valid PNG)
    kwargs = {"url": url}
    if screenshot:
        kwargs["screenshot"] = screenshot
    actor.computer_primitives._computer = MockComputerBackend(**kwargs)


# ────────────────────────────────────────────────────────────────────────────
# mock_state_managers
# ────────────────────────────────────────────────────────────────────────────


def mock_state_managers(
    actor: "HierarchicalActor",
    *,
    use_simulated: bool = True,
    contact_description: str = "A sandbox contact database for testing.",
    knowledge_description: str = "A sandbox knowledge base for testing.",
    task_description: str = "A sandbox task scheduler for testing.",
    transcript_description: str = "A sandbox transcript manager for testing.",
) -> None:
    """
    Set up mock state managers for safe, side-effect-free execution.

    This function replaces the actor's state manager primitives with either:
    - Simulated managers (use_simulated=True): Full LLM-powered simulation
    - AsyncMock managers (use_simulated=False): Simple mocks returning canned results

    Args:
        actor: The HierarchicalActor instance to mock
        use_simulated: If True, use simulated managers; if False, use AsyncMock
        contact_description: Description for simulated ContactManager
        knowledge_description: Description for simulated KnowledgeManager
        task_description: Description for simulated TaskScheduler
        transcript_description: Description for simulated TranscriptManager

    Usage:
        actor = HierarchicalActor(...)
        mock_state_managers(actor)  # Uses simulated managers by default

        # Or with simple mocks:
        mock_state_managers(actor, use_simulated=False)
    """
    from unity.actor.hierarchical_actor import HierarchicalActor

    if not isinstance(actor, HierarchicalActor):
        raise TypeError(f"Expected HierarchicalActor, got {type(actor).__name__}")

    # Get the primitives instance from the StateManagerEnvironment
    primitives = None
    try:
        sm_env = actor.environments.get("primitives")
        if sm_env is not None:
            primitives = getattr(sm_env, "_primitives", None)
    except Exception:
        pass

    if primitives is None:
        # Fallback: try to get primitives directly from actor if exposed
        primitives = getattr(actor, "primitives", None)

    if primitives is None:
        raise RuntimeError(
            "Could not find primitives on actor. "
            "Ensure the actor was created with StateManagerEnvironment.",
        )

    if use_simulated:
        _apply_simulated_managers(
            primitives,
            contact_description=contact_description,
            knowledge_description=knowledge_description,
            task_description=task_description,
            transcript_description=transcript_description,
        )
    else:
        _apply_async_mock_managers(primitives)


def _apply_simulated_managers(
    primitives,
    *,
    contact_description: str,
    knowledge_description: str,
    task_description: str,
    transcript_description: str,
) -> None:
    """Apply simulated managers to the primitives instance."""
    # Import simulated managers lazily to avoid circular imports
    from unity.contact_manager.simulated import SimulatedContactManager
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
    from unity.task_scheduler.simulated import SimulatedTaskScheduler
    from unity.transcript_manager.simulated import SimulatedTranscriptManager
    from unity.guidance_manager.simulated import SimulatedGuidanceManager
    from unity.secret_manager.simulated import SimulatedSecretManager
    from unity.web_searcher.simulated import SimulatedWebSearcher

    # Replace managers with simulated versions
    # Note: We access the private attributes to bypass property caching
    primitives._contacts = SimulatedContactManager(description=contact_description)
    primitives._knowledge = SimulatedKnowledgeManager(description=knowledge_description)
    primitives._tasks = SimulatedTaskScheduler(description=task_description)
    primitives._transcripts = SimulatedTranscriptManager(
        description=transcript_description,
    )
    primitives._guidance = SimulatedGuidanceManager(
        description="A sandbox guidance manager for testing.",
    )
    primitives._secrets = SimulatedSecretManager(
        description="A sandbox secret manager for testing.",
    )
    primitives._web_search = SimulatedWebSearcher(
        description="A sandbox web searcher for testing.",
    )


def _apply_async_mock_managers(primitives) -> None:
    """Apply AsyncMock managers to the primitives instance."""

    async def mock_ask(text: str, **kwargs) -> MockStateManagerHandle:
        return MockStateManagerHandle(f"Mock ask result for: {text[:50]}...")

    async def mock_update(text: str, **kwargs) -> MockStateManagerHandle:
        return MockStateManagerHandle(f"Mock update result for: {text[:50]}...")

    # Create mock manager objects with ask/update methods
    contacts_mock = MagicMock()
    contacts_mock.ask = AsyncMock(side_effect=mock_ask)
    contacts_mock.update = AsyncMock(side_effect=mock_update)

    knowledge_mock = MagicMock()
    knowledge_mock.ask = AsyncMock(side_effect=mock_ask)
    knowledge_mock.update = AsyncMock(side_effect=mock_update)

    tasks_mock = MagicMock()
    tasks_mock.ask = AsyncMock(side_effect=mock_ask)
    tasks_mock.update = AsyncMock(side_effect=mock_update)
    tasks_mock.execute = AsyncMock(side_effect=mock_update)

    transcripts_mock = MagicMock()
    transcripts_mock.ask = AsyncMock(side_effect=mock_ask)

    guidance_mock = MagicMock()
    guidance_mock.ask = AsyncMock(side_effect=mock_ask)
    guidance_mock.update = AsyncMock(side_effect=mock_update)

    secrets_mock = MagicMock()
    secrets_mock.ask = AsyncMock(side_effect=mock_ask)
    secrets_mock.update = AsyncMock(side_effect=mock_update)

    web_search_mock = MagicMock()
    web_search_mock.ask = AsyncMock(side_effect=mock_ask)

    # Apply mocks
    primitives._contacts = contacts_mock
    primitives._knowledge = knowledge_mock
    primitives._tasks = tasks_mock
    primitives._transcripts = transcripts_mock
    primitives._guidance = guidance_mock
    primitives._secrets = secrets_mock
    primitives._web_search = web_search_mock
