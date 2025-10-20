# FILE: tests/test_screen_share_manager/test_screen_share_manager.py

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from unity.screen_share_manager.types import TurnAnalysisResponse, KeyEvent, DetectedEvent
from tests.helpers import _handle_project

# --- Constants and Asset Loading ---

# Simple, valid base64 PNG data URLs for testing
PNG_BLUE_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz/w3A5MBA/8/AAYDAL4/7d4eAAAAAElFTkSuQmCC"
PNG_RED_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8AARf4z/A8DMQABAL9M43+gS1dAAAAAAElFTkSuQmCC"

ASSETS_DIR = Path(__file__).parent / "assets"

def load_asset_image(filename: str) -> Image.Image:
    """Loads an image from the assets directory for vision tests."""
    path = ASSETS_DIR / filename
    if not path.exists():
        pytest.fail(f"Required asset for vision test not found: {path}")
    # Convert to grayscale and resize to match the manager's internal processing
    return Image.open(path).convert("L").resize((512, 288))

# --- Fixtures ---

@pytest.fixture
async def manager():
    """Provides a clean, started ScreenShareManager instance for each test."""
    ssm = ScreenShareManager()
    await ssm.start()
    yield ssm
    ssm.stop()

@pytest.fixture
def mocked_manager(manager: ScreenShareManager):
    """Provides a manager with its LLM clients mocked out."""
    with patch.object(manager, '_detection_client', new_callable=AsyncMock) as mock_detect_client, \
         patch.object(manager, '_analysis_client', new_callable=AsyncMock) as mock_analysis_client:
        yield manager, {"detect": mock_detect_client, "annotate": mock_analysis_client}

# --- High-Level API and Orchestration Tests ---

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_full_api_flow_detection_and_annotation(mocked_manager):
    """
    Tests the primary public API flow: push frames, push speech, get detected events,
    and then get annotated handles, using mocked LLM responses.
    """
    manager, mocks = mocked_manager

    # --- Stage 1: Detection ---
    # Configure the mock detection client to return a valid JSON response
    mocks["detect"].generate.return_value = json.dumps({
        "moments": [{'timestamp': 1.5, 'reason': 'visual_change'}]
    })

    await manager.push_frame(PNG_BLUE_B64, 0.5)
    await manager.push_frame(PNG_RED_B64, 1.5)
    await manager.push_speech("test utterance", 1.0, 2.0)
    
    # analyze_turn returns a task that we await
    analysis_task = manager.analyze_turn()
    detected_events = await analysis_task

    assert mocks["detect"].generate.called
    assert len(detected_events) == 1
    assert isinstance(detected_events[0], DetectedEvent)
    assert detected_events[0].timestamp == 1.5
    assert isinstance(detected_events[0].image_handle, ImageHandle)

    # --- Stage 2: Annotation ---
    # Configure the mock analysis client to return a TurnAnalysisResponse
    annotation_response = TurnAnalysisResponse(events=[
        KeyEvent(timestamp=1.5, image_annotation="This is the rich annotation.", representative_timestamp=1.5)
    ])
    mocks["annotate"].generate.return_value = annotation_response

    annotation_context = "User is performing a test."
    annotated_handles = await manager.annotate_events(detected_events, annotation_context)
    
    mocks["annotate"].generate.assert_called_once()
    assert len(annotated_handles) == 1
    assert annotated_handles[0].annotation == "This is the rich annotation."
    # Verify the handle is the same object instance
    assert annotated_handles[0] is detected_events[0].image_handle

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_vision_event_flush_on_inactivity(manager: ScreenShareManager):
    """
    Tests that a visual event without speech is detected after an inactivity timeout.
    """
    manager.INACTIVITY_TIMEOUT_SEC = 0.1
    
    with patch.object(manager, '_detect_key_moments', new_callable=AsyncMock) as mock_detect:
        # Manually inject a pending vision event into the manager's state
        async with manager._state_lock:
            manager._pending_vision_events.append({
                "timestamp": 1.0, "before_frame_b64": PNG_BLUE_B64, "after_frame_b64": PNG_RED_B64
            })
        
        # Wait for the inactivity flush loop to trigger
        await asyncio.sleep(0.2)
        
        mock_detect.assert_called_once()
        # The first argument (speech_event) should be None for a silent flush
        assert mock_detect.call_args[0][0] is None

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_failure_is_handled_gracefully(manager: ScreenShareManager):
    """
    Tests that if the expensive annotation LLM call fails, the process doesn't crash
    and returns an empty list.
    """
    # Create a dummy DetectedEvent with a real (pending) ImageHandle
    [handle] = manager._image_manager.add_images([{"data": PNG_RED_B64}], synchronous=False, return_handles=True)
    detected_events = [DetectedEvent(1.0, "test", handle, None)]
    
    with patch.object(manager, '_analysis_client', new_callable=AsyncMock) as mock_analysis_client:
        mock_analysis_client.generate.side_effect = Exception("LLM API Error")
        
        annotated_handles = await manager.annotate_events(detected_events, "test context")
        
        # Should return an empty list and log an error, not raise an exception
        assert annotated_handles == []

# --- Internal Logic Tests ---

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detect_key_moments_llm_call(mocked_manager):
    """
    Verifies that _detect_key_moments calls the detection LLM with the correct
    prompt and correctly processes the JSON response.
    """
    manager, mocks = mocked_manager
    mocks["detect"].generate.return_value = json.dumps({
        "moments": [{"timestamp": 1.5, "reason": "visual_change"}]
    })

    speech = {"payload": {"content": "hello", "start_time": 1.0, "end_time": 2.0}}
    visuals = [{"timestamp": 1.5, "after_frame_b64": PNG_RED_B64}]

    await manager._detect_key_moments(speech, visuals, None)
    
    # Check that the LLM was called
    mocks["detect"].generate.assert_called_once()
    
    # Check that the result was put on the detection queue for the consumer
    moments, frame_map = await manager._detection_queue.get()
    assert len(moments) == 1
    assert moments[0]['timestamp'] == 1.5
    assert frame_map[1.5] == PNG_RED_B64

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_sequencer_creates_pending_vision_event(manager: ScreenShareManager):
    """
    Tests the internal sequencer correctly identifies a significant change
    and adds a vision event to the pending list.
    """
    manager._last_significant_frame_b64 = PNG_BLUE_B64
    manager._last_significant_frame_pil = manager._b64_to_image(PNG_BLUE_B64)
    
    with patch.object(manager, "_calculate_mse", return_value=999), \
         patch("unity.screen_share_manager.screen_share_manager.ssim", return_value=0.1), \
         patch.object(manager, "_is_semantically_significant", return_value=True):
        
        event_data = {"payload": {"timestamp": 1.0, "frame_b64": PNG_RED_B64}}
        pil_image = manager._b64_to_image(PNG_RED_B64)
        await manager._results_queue.put((1, event_data, pil_image))

        await asyncio.sleep(0.05) # Allow sequencer to process

        async with manager._state_lock:
            assert len(manager._pending_vision_events) == 1
            assert manager._pending_vision_events[0]["timestamp"] == 1.0

# --- Visual Change Detection Tests ---

@pytest.mark.vision
@_handle_project
@pytest.mark.parametrize(
    "image_pair",
    [
        ("modal_before.png", "modal_after.png"),
        ("button_active_before.png", "button_active_after.png"),
    ],
)
def test_visual_change_detection_significant_changes(manager: ScreenShareManager, image_pair):
    """
    Tests that the vision pipeline correctly identifies REAL, significant UI changes.
    Requires image assets in `tests/test_screen_share_manager/assets/`.
    """
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    # 1. MSE Pre-filter: Should pass the threshold for a real change
    mse = manager._calculate_mse(img_before, img_after)
    assert mse > manager.MSE_THRESHOLD

    # 2. SSIM Perceptual Check: Should be below the threshold, indicating a difference
    from skimage.metrics import structural_similarity as ssim
    score = ssim(np.array(img_before), np.array(img_after))
    assert score < manager.SSIM_THRESHOLD

    # 3. Semantic Contour Analysis: Should find significant contours
    is_significant = manager._is_semantically_significant(img_before, img_after)
    assert is_significant is True

@pytest.mark.vision
@_handle_project
@pytest.mark.parametrize(
    "image_pair",
    [
        ("blinking_caret_before.png", "blinking_caret_after.png"),
        ("cursor_move_before.png", "cursor_move_after.png"),
    ],
)
def test_visual_change_detection_insignificant_changes(manager: ScreenShareManager, image_pair):
    """
    Tests that the vision pipeline correctly IGNORES insignificant visual noise like
    a blinking text cursor or mouse pointer movement.
    """
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    # The final semantic filter is the one designed to catch these, even if MSE/SSIM pass.
    is_significant = manager._is_semantically_significant(img_before, img_after)
    assert is_significant is False