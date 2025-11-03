import asyncio
import pytest

from unity.screen_share_manager.screen_share_manager import (
    ScreenShareManager,
    TurnState,
)
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project

import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock, PropertyMock, patch


from unity.image_manager.image_manager import ImageHandle
from tests.test_screen_share_manager.conftest import (
    PNG_RED_B64,
    PNG_BLUE_B64,
    PNG_GREEN_B64,
)


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_handle_empty_turn():
    """Tests that detection with no events enqueues an empty list without error."""
    manager = ScreenShareManager()
    await manager.start()
    await manager._detect_key_moments(
        TurnState(speech_events=[], visual_events=[], latest_frame=None)
    )
    result = await manager._detection_queue.get()
    assert result == []
    await manager.stop()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_retry_on_llm_failure(mocked_manager):
    """Tests that the detection LLM call is retried on failure, per the decorator."""
    manager, mocks = mocked_manager
    manager.settings.llm_retry_max_tries = 3
    manager.settings.llm_retry_base_delay_sec = 0.01

    mocks["detect"].generate.side_effect = [
        Exception("LLM unavailable"),
        Exception("LLM still unavailable"),
        json.dumps({"moments": []}),
    ]

    await manager._detect_key_moments(
        TurnState(speech_events=[{"payload": {"content": "test", "start_time": 0.0}}])
    )

    assert mocks["detect"].generate.call_count == 3


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_handle_invalid_llm_json(mocked_manager):
    """Ensures the manager does not crash if the detection LLM returns malformed JSON."""
    manager, mocks = mocked_manager
    mocks["detect"].generate.return_value = "This is not valid JSON"

    # The retry decorator will raise the final exception, which we expect here.
    with pytest.raises(json.JSONDecodeError):
        await manager._detect_key_moments(
            TurnState(
                speech_events=[{"payload": {"content": "test", "start_time": 0.0}}]
            )
        )

    assert manager._detection_queue.empty()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_handle_llm_timeouts(mocked_manager):
    """Tests that the manager handles LLM timeouts gracefully without crashing."""
    manager, mocks = mocked_manager
    mocks["detect"].generate.side_effect = asyncio.TimeoutError()

    # The retry decorator will raise the final exception after retries.
    with pytest.raises(asyncio.TimeoutError):
        await manager._detect_key_moments(
            TurnState(speech_events=[{"payload": {"content": "x", "start_time": 0.0}}])
        )

    assert mocks["detect"].generate.call_count >= 1


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_consolidates_speech_with_subsequent_visual_outcome(
    mocked_manager,
):
    """
    WHY: This tests the primary directive of the detection prompt: merging a spoken
    intent with its visual result into a single key moment.
    """
    manager, mocks = mocked_manager
    mock_handle = MagicMock(spec=ImageHandle)

    # Mock the LLM to return the consolidated event
    mocks["detect"].generate.return_value = json.dumps(
        {"moments": [{"timestamp": 10.5, "reason": "user_speech"}]}
    )
    with patch.object(manager._image_manager, "add_images", return_value=[mock_handle]):
        turn_state = TurnState(
            speech_events=[
                {"payload": {"content": "clicking submit", "start_time": 10.0}}
            ],
            visual_events=[{"timestamp": 10.5, "after_frame_b64": PNG_GREEN_B64}],
        )
        await manager._detect_key_moments(turn_state)

    result = await manager._detection_queue.get()

    assert len(result) == 1
    assert result[0].timestamp == 10.5  # The timestamp of the visual outcome
    assert result[0].image_handle is mock_handle
