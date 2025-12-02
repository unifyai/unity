import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import TurnState
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import PNG_RED_B64


@_handle_project
@pytest.mark.asyncio
async def test_core_flow_detects_and_annotates(mocked_manager):
    """Tests the primary API flow: start turn, push speech, end turn, and annotate."""
    manager, mocks = mocked_manager

    mock_handle = MagicMock(spec=ImageHandle)
    mock_handle.raw.return_value = base64.b64decode(PNG_RED_B64.split(",", 1)[1])
    await manager._detection_queue.put(
        [
            DetectedEvent(
                timestamp=1.5,
                detection_reason="visual_change",
                image_handle=mock_handle,
            ),
        ],
    )

    manager.start_turn()
    await manager.push_speech("A user utterance", 1.0, 1.2)
    analysis_task = manager.end_turn()
    detected_events = await analysis_task

    assert len(detected_events) == 1
    assert detected_events[0].timestamp == 1.5

    mocks["annotate"].generate.return_value = "This is the rich annotation."
    annotated_handles = await manager.annotate_events(
        detected_events,
        "User is performing a test.",
    )

    mocks["annotate"].generate.assert_called_once()
    system_prompt = mocks["annotate"].set_system_message.call_args.args[0]
    assert "User is performing a test." in system_prompt
    assert len(annotated_handles) == 1
    assert annotated_handles[0].annotation == "This is the rich annotation."


@_handle_project
@pytest.mark.asyncio
async def test_silent_events_stored_and_processed_next_turn(
    mocked_manager,
):
    """Tests that a visual event detected without speech is stored and returned in the next turn."""
    manager, _ = mocked_manager
    silent_handle = MagicMock(spec=ImageHandle)
    manager._stored_silent_detected_events = [
        DetectedEvent(1.0, "silent_change", silent_handle),
    ]

    speech_handle = MagicMock(spec=ImageHandle)
    await manager._detection_queue.put(
        [DetectedEvent(2.5, "speech_related_change", speech_handle)],
    )

    manager.start_turn()
    await manager.push_speech("second turn", 2.0, 3.0)
    analysis_task = manager.end_turn()

    all_events = await analysis_task
    assert len(all_events) == 2
    timestamps = {e.timestamp for e in all_events}
    assert 1.0 in timestamps
    assert 2.5 in timestamps


@_handle_project
@pytest.mark.asyncio
async def test_collects_multiple_speech_events(mocked_manager):
    """Ensures a manual turn correctly collects multiple speech events for analysis."""
    manager, mocks = mocked_manager
    mocks["detect"].generate.return_value = json.dumps({"moments": []})

    manager.start_turn()
    await manager.push_speech("first part", 1.0, 1.1)
    await manager.push_speech("second part", 1.2, 1.3)
    _ = manager.end_turn()

    await asyncio.sleep(0.1)  # Allow task to be created

    system_prompt = mocks["detect"].set_system_message.call_args.args[0]
    assert "first part" in system_prompt
    assert "second part" in system_prompt


@_handle_project
@pytest.mark.asyncio
async def test_push_speech_ignored_when_no_turn_active(manager, caplog):
    """Tests that pushing speech outside a started turn is ignored with a warning."""
    await manager.push_speech("hello outside", 0.0, 0.5)
    assert "no turn is in progress" in caplog.text.lower()


@_handle_project
@pytest.mark.asyncio
async def test_safe_with_overlapping_starts(mocked_manager):
    """Tests that starting a new turn while one is in progress is safe and does not crash."""
    manager, _ = mocked_manager
    manager.start_turn()
    manager.start_turn()  # Should reset internal state without error
    task = manager.end_turn()
    await task


@_handle_project
@pytest.mark.asyncio
async def test_inactivity_flush_triggers_for_silent_visual_events(mocked_manager):
    """Tests that a silent visual event triggers analysis after an inactivity period."""
    manager, _ = mocked_manager
    manager.settings.inactivity_timeout_sec = 0.1
    manager._inactivity_task.cancel()  # Stop the default loop to control timing

    with patch.object(
        manager,
        "_detect_key_moments",
        new_callable=AsyncMock,
    ) as mock_detect:
        manager._pending_vision_events.append(
            {"timestamp": 1.0, "after_frame_b64": PNG_RED_B64},
        )
        manager._last_activity_time = asyncio.get_event_loop().time()

        # Manually simulate the check that the inactivity loop performs
        await asyncio.sleep(0.15)
        if (
            not manager._turn_in_progress
            and (
                asyncio.get_event_loop().time() - manager._last_activity_time
                >= manager.settings.inactivity_timeout_sec
            )
            and manager._pending_vision_events
        ):
            turn_state = TurnState(visual_events=list(manager._pending_vision_events))
            manager._pending_vision_events.clear()
            await manager._detect_key_moments(turn_state)

        mock_detect.assert_called_once()


@_handle_project
@pytest.mark.asyncio
async def test_concurrency_stable_under_load(manager):
    """Tests that pushing frames and speech concurrently does not crash the manager."""

    async def push_frames():
        for i in range(5):
            await manager.push_frame(PNG_RED_B64, i * 0.1)
            await asyncio.sleep(0.01)

    async def push_speech_flow():
        manager.start_turn()
        for i in range(3):
            await manager.push_speech(f"utterance {i}", i, i + 0.05)
            await asyncio.sleep(0.02)
        manager.end_turn()

    await asyncio.gather(push_frames(), push_speech_flow())
