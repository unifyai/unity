import asyncio
import base64
import json
from unittest.mock import MagicMock

import pytest

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import ScreenShareManager, TurnState
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import PNG_RED_B64, PNG_BLUE_B64, PNG_GREEN_B64


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_concurrency_should_remain_stable_under_load(manager):
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


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_helpers_should_raise_on_invalid_b64_image_data():
    """Tests that the internal b64-to-image helper raises a ValueError for invalid data."""
    manager = ScreenShareManager()
    bad_b64 = "data:image/png;base64,not_a_valid_base64_string"
    with pytest.raises(ValueError):
        manager._b64_to_image(bad_b64)


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_retry_on_llm_failure(mocked_manager):
    """Tests that the detection LLM call is retried on failure, per the decorator."""
    manager, mocks = mocked_manager
    manager.settings.llm_retry_max_tries = 3
    manager.settings.llm_retry_base_delay_sec = 0.01

    mocks["detect"].generate.side_effect = [Exception("LLM unavailable"), Exception("LLM still unavailable"), json.dumps({"moments": []})]
    
    await manager._detect_key_moments(TurnState(speech_events=[{"payload": {"content": "test", "start_time": 0.0}}]))

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
        await manager._detect_key_moments(TurnState(speech_events=[{"payload": {"content": "test", "start_time": 0.0}}]))
    
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
        await manager._detect_key_moments(TurnState(speech_events=[{"payload": {"content": "x", "start_time": 0.0}}]))
    
    assert mocks["detect"].generate.call_count >= 1


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_should_continue_on_partial_failure(mocked_manager):
    """Tests that if one annotation fails, subsequent annotations are still attempted."""
    manager, mocks = mocked_manager
    h1 = MagicMock(spec=ImageHandle)
    h1.raw.return_value = base64.b64decode(PNG_RED_B64.split(",", 1)[1])
    h2 = MagicMock(spec=ImageHandle)
    h2.raw.return_value = base64.b64decode(PNG_BLUE_B64.split(",", 1)[1])

    events = [DetectedEvent(1.0, "r1", h1), DetectedEvent(2.0, "r2", h2)]
    # First annotation call succeeds, second raises an exception.
    mocks["annotate"].generate.side_effect = ["first annotation", Exception("LLM failure on second")]

    await manager.annotate_events(events, "ctx")

    assert h1.annotation == "first annotation"
    assert getattr(h2, "annotation", None) is None


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_should_handle_corrupted_image_handle(mocked_manager):
    """Tests that if an ImageHandle.raw() call fails, the manager logs and continues."""
    manager, mocks = mocked_manager
    bad_handle = MagicMock(spec=ImageHandle)
    bad_handle.raw.side_effect = Exception("broken-image-bytes")
    good_handle = MagicMock(spec=ImageHandle)
    good_handle.raw.return_value = base64.b64decode(PNG_GREEN_B64.split(",", 1)[1])

    events = [DetectedEvent(1.0, "bad", bad_handle), DetectedEvent(2.0, "good", good_handle)]
    mocks["annotate"].generate.return_value = "annotation-good"

    results = await manager.annotate_events(events, "ctx")

    # The good handle should be annotated, and the bad one should be skipped without crashing.
    assert len(results) == 1
    assert results[0].annotation == "annotation-good"
    mocks["annotate"].generate.assert_called_once() # Only called for the good handle