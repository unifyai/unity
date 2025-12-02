import pytest

from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import PNG_RED_B64


@_handle_project
@pytest.mark.asyncio
async def test_annotate_triggers_summary_update(mocked_manager):
    """Verifies that a successful annotation triggers an update to the session summary."""
    manager, mocks = mocked_manager
    manager.set_session_context("Initial summary.")
    red_b64 = PNG_RED_B64.split(",", 1)[1]
    handles = manager._image_manager.add_images(
        [{"data": red_b64}],
        synchronous=True,
        return_handles=True,
    )
    detected_events = [DetectedEvent(1.0, "test", handles[0])]

    mocks["annotate"].generate.return_value = "A new event happened."
    mocks["summary"].generate.return_value = "Updated summary including the new event."
    await manager.annotate_events(detected_events, "test context")

    summary_task = manager._summary_update_task
    assert summary_task is not None, "Summary update task was not created"
    await summary_task

    mocks["summary"].generate.assert_called_once()
    summary_prompt = mocks["summary"].generate.call_args.args[0]
    assert "A new event happened." in summary_prompt
    assert "Initial summary." in summary_prompt
    async with manager._state_lock:
        assert manager._session_summary == "Updated summary including the new event."


@_handle_project
@pytest.mark.asyncio
async def test_should_persist_across_turns(mocked_manager):
    """Tests that the session summary persists between manager operations."""
    manager, _ = mocked_manager
    manager.set_session_context("Session A")

    manager.start_turn()
    await manager.push_speech("hi", 0.0, 0.1)
    await manager._detection_queue.put([])  # Mock empty detection result
    await manager.end_turn()

    async with manager._state_lock:
        assert "Session A" in manager._session_summary


@_handle_project
@pytest.mark.asyncio
async def test_should_be_clearable(mocked_manager):
    """Tests that the session summary can be reset by setting an empty context string."""
    manager, _ = mocked_manager
    manager.set_session_context("Some previous summary")
    manager.set_session_context("")  # Reset the summary
    async with manager._state_lock:
        assert manager._session_summary == ""
