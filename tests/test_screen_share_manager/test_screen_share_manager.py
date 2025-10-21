from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from unity.image_manager.image_manager import ImageHandle, ImageManager
from unity.screen_share_manager.screen_share_manager import (
    DetectedEvent,
    ScreenShareManager,
    ScreenShareManagerSettings,
)
from unity.screen_share_manager.types import KeyEvent

# --- Asset Loading ---
# Helper to locate and load image assets for realistic visual change detection tests.
ASSETS_DIR = Path(__file__).parent / "assets"


def load_asset_b64(filename: str) -> str:
    """Loads an image from the assets folder and returns it as a base64 string."""
    try:
        content = (ASSETS_DIR / filename).read_bytes()
        return base64.b64encode(content).decode("utf-8")
    except FileNotFoundError:
        pytest.fail(
            f"Asset file not found: {filename}. Ensure it is in the 'assets' directory."
        )


# Load significant change assets
MODAL_BEFORE_B64 = load_asset_b64("modal_before.png")
MODAL_AFTER_B64 = load_asset_b64("modal_after.png")
BUTTON_BEFORE_B64 = load_asset_b64("button_active_before.png")
BUTTON_AFTER_B64 = load_asset_b64("button_active_after.png")

# Load insignificant change assets
CARET_BEFORE_B64 = load_asset_b64("blinking_caret_before.png")
CARET_AFTER_B64 = load_asset_b64("blinking_caret_after.png")
CURSOR_BEFORE_B64 = load_asset_b64("cursor_move_before.png")
CURSOR_AFTER_B64 = load_asset_b64("cursor_move_after.png")

# Simple solid color assets for non-visual tests
PNG_RED_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="
PNG_BLUE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk/wcAAgAB/epv2AAAAABJRU5ErkJggg=="
PNG_GREEN_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60eADwAAAABJRU5ErkJggg=="


class MockAsyncUnify:
    """A mock for unify.AsyncUnify to return canned responses."""

    def __init__(self, responses: Optional[List[str]] = None, test_id: str = "default"):
        self._responses = responses or []
        self._response_idx = 0
        self.system_message: Optional[str] = None
        self.call_history: List[Dict[str, Any]] = []
        self.test_id = test_id

    def set_system_message(self, message: str):
        self.system_message = message

    async def generate(self, *args, **kwargs) -> str:
        call_info = {
            "args": args,
            "kwargs": kwargs,
            "system_message": self.system_message,
        }
        self.call_history.append(call_info)
        if self._response_idx < len(self._responses):
            response = self._responses[self._response_idx]
            self._response_idx += 1
            if isinstance(response, Exception):
                raise response
            return response
        raise AssertionError(
            f"MockAsyncUnify '{self.test_id}' received more calls than expected."
        )


@pytest.fixture
def mock_image_manager(monkeypatch):
    """Fixture to provide a mocked ImageManager."""
    mock_im = MagicMock(spec=ImageManager)
    _pending_id_counter = 10**12

    def _add_images_side_effect(items, synchronous, return_handles):
        nonlocal _pending_id_counter
        handles = []
        for item in items:
            mock_handle = MagicMock(spec=ImageHandle)
            mock_handle.image_id = _pending_id_counter
            mock_handle.is_pending = True
            type(mock_handle).annotation = patch.object.PropertyMock(return_value=None)
            raw_data = item["data"]
            if isinstance(raw_data, str):
                raw_data = base64.b64decode(raw_data)
            mock_handle.raw.return_value = raw_data
            handles.append(mock_handle)
            _pending_id_counter += 1
        return handles

    async def _add_images_async(*args, **kwargs):
        return _add_images_side_effect(*args, **kwargs)

    mock_im.add_images = AsyncMock(side_effect=_add_images_side_effect)
    monkeypatch.setattr(
        "asyncio.to_thread",
        lambda func, *args, **kwargs: _add_images_async(*args, **kwargs),
    )
    return mock_im


@pytest.fixture
async def manager(mock_image_manager):
    """Fixture to create, start, and stop a ScreenShareManager instance."""
    # Use settings that mirror the defaults to test real-world behavior with assets
    settings = ScreenShareManagerSettings(
        mse_threshold=25.0,
        ssim_threshold=0.985,
        min_contour_area=100,
        debounce_delay_sec=0.05,
        vision_event_cooldown_sec=0.02,
        inactivity_timeout_sec=0.2,
        frame_queue_size=10,
        adaptive_drop_threshold=0.7,
    )
    mock_detection_client = MockAsyncUnify(test_id="detection")
    mock_analysis_client = MockAsyncUnify(test_id="analysis")
    mock_summary_client = MockAsyncUnify(test_id="summary")

    mgr = ScreenShareManager(
        settings=settings,
        image_manager=mock_image_manager,
        detection_client=mock_detection_client,
        analysis_client=mock_analysis_client,
        summary_client=mock_summary_client,
    )
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_manager_initialization_and_lifecycle(manager: ScreenShareManager):
    """Test that the manager starts and stops its background tasks."""
    assert manager._sequencer_task and not manager._sequencer_task.done()
    assert manager._inactivity_task and not manager._inactivity_task.done()
    assert len(manager._frame_workers) > 0
    await asyncio.sleep(0.01)


# --- Visual Detection System Tests (Using Real Image Assets) ---


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before_b64, after_b64",
    [
        (MODAL_BEFORE_B64, MODAL_AFTER_B64),
        (BUTTON_BEFORE_B64, BUTTON_AFTER_B64),
    ],
)
@_handle_project
async def test_detects_significant_visual_changes(
    manager: ScreenShareManager, before_b64, after_b64
):
    """
    Tests that the manager's real image processing logic correctly identifies
    a significant UI change (e.g., a modal appearing).
    """
    assert len(manager._pending_vision_events) == 0

    # Push the "before" state
    await manager.push_frame(before_b64, timestamp=1.0)
    await asyncio.sleep(0.1)  # Let processing happen
    assert len(manager._pending_vision_events) == 0
    assert manager._last_significant_frame_b64 is not None

    # Push the "after" state, which contains a significant change
    await manager.push_frame(after_b64, timestamp=2.0)
    await asyncio.sleep(0.1)

    # A visual event should have been detected
    assert len(manager._pending_vision_events) == 1
    event = manager._pending_vision_events[0]
    assert event["timestamp"] == 2.0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before_b64, after_b64",
    [
        (CARET_BEFORE_B64, CARET_AFTER_B64),
        (CURSOR_BEFORE_B64, CURSOR_AFTER_B64),
    ],
)
@_handle_project
async def test_ignores_insignificant_visual_changes(
    manager: ScreenShareManager, before_b64, after_b64
):
    """
    Tests that the manager's real image processing logic correctly ignores
    insignificant UI changes (e.g., a blinking cursor or small mouse movement).
    """
    await manager.push_frame(before_b64, timestamp=1.0)
    await asyncio.sleep(0.1)
    await manager.push_frame(after_b64, timestamp=1.1)
    await asyncio.sleep(0.1)

    # No event should be created because the change is too small to be significant
    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_vision_event_cooldown_prevents_flooding(manager: ScreenShareManager):
    """Tests that the vision_event_cooldown_sec prevents a flood of events."""
    # This test uses simple assets as it's testing timing, not complex detection.
    with patch.object(manager, "_calculate_mse", return_value=50.0), patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch("skimage.metrics.structural_similarity", return_value=0.9):

        await manager.push_frame(PNG_RED_B64, timestamp=1.0)
        await asyncio.sleep(0.05)
        assert len(manager._pending_vision_events) == 0

        await manager.push_frame(PNG_BLUE_B64, timestamp=1.1)
        await asyncio.sleep(0.05)
        assert len(manager._pending_vision_events) == 1

        await manager.push_frame(PNG_GREEN_B64, timestamp=1.11)
        await asyncio.sleep(0.05)
        assert len(manager._pending_vision_events) == 1

        await asyncio.sleep(manager.settings.vision_event_cooldown_sec)
        await manager.push_frame(PNG_RED_B64, timestamp=1.2)
        await asyncio.sleep(0.05)
        assert len(manager._pending_vision_events) == 2


# --- Turn Analysis Debouncing Tests ---


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_turn_analysis_debouncing_waits_for_delay(manager: ScreenShareManager):
    """Tests that analysis is delayed by the debounce setting."""
    manager._detection_client = MockAsyncUnify(responses=['{"moments": []}'])
    await manager.push_speech("test utterance", 1.0, 1.5)
    assert len(manager._detection_client.call_history) == 0
    await asyncio.sleep(manager.settings.debounce_delay_sec / 2)
    assert len(manager._detection_client.call_history) == 0
    await asyncio.sleep(manager.settings.debounce_delay_sec)
    assert len(manager._detection_client.call_history) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_debouncing_is_reset_by_new_events(manager: ScreenShareManager):
    """Tests that a new speech event resets the debounce timer."""
    manager._detection_client = MockAsyncUnify(responses=['{"moments": []}'])
    await manager.push_speech("first utterance", 1.0, 1.5)
    await asyncio.sleep(manager.settings.debounce_delay_sec * 0.9)
    assert len(manager._detection_client.call_history) == 0
    await manager.push_speech("second utterance", 2.0, 2.5)
    await asyncio.sleep(manager.settings.debounce_delay_sec * 0.9)
    assert len(manager._detection_client.call_history) == 0
    await asyncio.sleep(manager.settings.debounce_delay_sec)
    assert len(manager._detection_client.call_history) == 1
    assert (
        "second utterance"
        in manager._detection_client.call_history[0]["system_message"]
    )


# --- Adaptive Frame Dropping Tests ---


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_adaptive_frame_dropping_when_backlogged(manager: ScreenShareManager):
    """Tests that frames are dropped when the queue is nearing capacity."""
    backlog_size = (
        int(
            manager.settings.frame_queue_size * manager.settings.adaptive_drop_threshold
        )
        + 1
    )
    with patch.object(
        manager._frame_queue, "qsize", return_value=backlog_size
    ), patch.object(manager._frame_queue, "put") as mock_put:
        await manager.push_frame(PNG_RED_B64, 1.0)
        mock_put.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_frame_dropping_when_queue_is_full(manager: ScreenShareManager):
    """Tests that frames are dropped when the queue is completely full."""
    with patch.object(manager._frame_queue, "full", return_value=True), patch.object(
        manager._frame_queue, "put"
    ) as mock_put:
        await manager.push_frame(PNG_RED_B64, 1.0)
        mock_put.assert_not_called()


# --- Other Core Logic and Edge Case Tests ---


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_inactivity_flush_triggers_silent_detection(manager: ScreenShareManager):
    """Test that inactivity flushes pending visual events into a silent turn."""
    detection_response = '{"moments": [{"timestamp": 2.0, "reason": "visual_change"}]}'
    manager._detection_client = MockAsyncUnify(
        responses=[detection_response], test_id="inactivity_flush"
    )
    # Use mocks for image comparison to ensure a visual event is created for this timing test.
    with patch.object(manager, "_calculate_mse", return_value=50.0), patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch("skimage.metrics.structural_similarity", return_value=0.9):
        await manager.push_frame(PNG_RED_B64, timestamp=1.0)
        await manager.push_frame(PNG_BLUE_B64, timestamp=2.0)
        await asyncio.sleep(0.1)
        assert len(manager._pending_vision_events) == 1
        await asyncio.sleep(0.3)
        assert len(manager._detection_client.call_history) == 1
        assert "No user speech occurred." in manager._detection_client.system_message
        assert len(manager._pending_vision_events) == 0
        assert len(manager._stored_silent_detected_events) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_full_turn_speech_plus_visuals(manager: ScreenShareManager):
    """Test a complete turn from speech to annotation."""
    detection_response = '{"moments": [{"timestamp": 2.0, "reason": "user_speech"}]}'
    annotation_response = "A modal window appeared on the screen."
    summary_response = "Session started. A modal window appeared on the screen."
    manager._detection_client = MockAsyncUnify(
        responses=[detection_response], test_id="full_turn_detect"
    )
    manager._analysis_client = MockAsyncUnify(
        responses=[annotation_response], test_id="full_turn_analysis"
    )
    manager._summary_client = MockAsyncUnify(
        responses=[summary_response], test_id="full_turn_summary"
    )

    await manager.push_frame(MODAL_BEFORE_B64, timestamp=1.0)
    await manager.push_frame(MODAL_AFTER_B64, timestamp=2.0)
    await asyncio.sleep(0.01)

    await manager.push_speech("what is this pop-up?", 1.8, 2.5)
    analysis_task = manager.analyze_turn()
    await asyncio.sleep(0.1)

    detected_events = await analysis_task
    assert len(detected_events) == 1

    annotated_handles = await manager.annotate_events(
        detected_events, context="User is confused."
    )
    assert len(annotated_handles) == 1
    assert annotated_handles[0].annotation == annotation_response

    await asyncio.sleep(1.1)
    assert len(manager._summary_client.call_history) == 1
    assert manager._session_summary == summary_response


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_event_burst_consolidation(manager: ScreenShareManager):
    """Test that rapid visual changes are consolidated into one event."""
    manager.settings.burst_detection_threshold_sec = 0.1
    manager.settings.visual_event_sampling_threshold = 2
    detection_response = '{"moments": [{"timestamp": 1.4, "reason": "visual_change"}]}'
    manager._detection_client = MockAsyncUnify(
        responses=[detection_response], test_id="burst_detect"
    )

    with patch.object(manager, "_calculate_mse", return_value=50.0), patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch("skimage.metrics.structural_similarity", return_value=0.9):
        await manager.push_frame(PNG_RED_B64, timestamp=1.0)
        await manager.push_frame(PNG_BLUE_B64, timestamp=1.1)
        await manager.push_frame(PNG_RED_B64, timestamp=1.2)
        await manager.push_frame(PNG_BLUE_B64, timestamp=1.3)
        await manager.push_frame(PNG_RED_B64, timestamp=1.4)
        await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)

        assert len(manager._detection_client.call_history) == 1
        prompt = manager._detection_client.system_message
        assert "A rapid sequence of 5 visual changes" in prompt
        assert "ending at t=1.40s" in prompt


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_silent_events_are_combined_with_next_turn(manager: ScreenShareManager):
    """Test that stored silent events are returned alongside the next turn's events."""
    detection_responses = [
        '{"moments": [{"timestamp": 2.0, "reason": "visual_change"}]}',
        '{"moments": [{"timestamp": 4.5, "reason": "user_speech"}]}',
    ]
    manager._detection_client = MockAsyncUnify(
        responses=detection_responses, test_id="silent_plus_speech"
    )

    await manager.push_frame(MODAL_BEFORE_B64, timestamp=1.0)
    await manager.push_frame(MODAL_AFTER_B64, timestamp=2.0)
    await asyncio.sleep(0.3)
    assert len(manager._detection_client.call_history) == 1
    assert len(manager._stored_silent_detected_events) == 1

    await manager.push_frame(BUTTON_BEFORE_B64, timestamp=4.5)
    await manager.push_speech("now do this", 4.0, 4.8)
    analysis_task = manager.analyze_turn()
    await asyncio.sleep(0.1)

    detected_events = await analysis_task
    assert len(detected_events) == 2
    timestamps = sorted([e.timestamp for e in detected_events])
    assert timestamps == [2.0, 4.5]
    assert len(manager._stored_silent_detected_events) == 0


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_annotation_failure_on_one_event_does_not_stop_others(
    manager: ScreenShareManager,
):
    """Tests that if one annotation fails, others in the same turn are still processed."""
    handle1 = MagicMock(spec=ImageHandle)
    handle1.raw.return_value = b"1"
    handle1.annotation = None
    handle2 = MagicMock(spec=ImageHandle)
    handle2.raw.return_value = b"2"
    handle2.annotation = None
    detected_events = [
        DetectedEvent(timestamp=1.0, detection_reason="a", image_handle=handle1),
        DetectedEvent(timestamp=2.0, detection_reason="b", image_handle=handle2),
    ]
    responses = [Exception("LLM API Error"), "This is the second annotation."]
    manager._analysis_client = MockAsyncUnify(
        responses=responses, test_id="annotation_failure"
    )

    annotated_handles = await manager.annotate_events(detected_events, context="test")

    assert len(annotated_handles) == 1
    assert annotated_handles[0] == handle2
    assert handle2.annotation == "This is the second annotation."
    assert handle1.annotation is None
    assert len(manager._analysis_client.call_history) == 2
