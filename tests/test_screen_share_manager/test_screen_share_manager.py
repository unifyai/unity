# FILE: tests/test_screen_share_manager/test_screen_share_manager.py

import asyncio
from datetime import datetime
import json
from unittest.mock import patch, AsyncMock, MagicMock
import time
from pathlib import Path

import pytest
from PIL import Image
from unity.image_manager.types import (
    AnnotatedImageRef,
    ImageRefs,
    RawImageRef,
    AnnotatedImageRefs,
)
from unity.image_manager.utils import make_solid_png_base64
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from unity.screen_share_manager.types import TurnAnalysisResponse, KeyEvent
from tests.helpers import _handle_project

# A simple, valid base64 PNG for testing, now in the correct data URL format
PNG_BLUE_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 0, 255))}"
PNG_RED_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 0, 0))}"
PNG_GREEN_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 255, 0))}"
PNG_YELLOW_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 255, 0))}"
PNG_CYAN_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 255, 255))}"
PNG_MAGENTA_B64 = (
    f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 0, 255))}"
)
PNG_WHITE_B64 = (
    f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 255, 255))}"
)

# --- Asset Loading Helper ---
ASSETS_DIR = Path(__file__).parent / "assets"


def load_asset_image(filename: str) -> Image.Image:
    """Loads an image from the assets directory and converts it for testing."""
    path = ASSETS_DIR / filename
    if not path.exists():
        pytest.fail(f"Asset image not found: {path}")
    # Convert to grayscale and resize to match the manager's internal processing
    return Image.open(path).convert("L").resize((512, 288))


@pytest.fixture
def mock_loop():
    """Provides the running event loop for tests that need it."""
    loop = asyncio.get_event_loop()
    return loop


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_full_change_detection_pipeline_creates_event(
    mocked_screen_share_manager, mock_loop
):
    """
    Tests that a change passing all three stages (MSE, SSIM, Semantic)
    creates a 'pending_vision_event' via the new sequencer pipeline.
    """
    manager, mocks = mocked_screen_share_manager
    sequencer_task = asyncio.create_task(manager._sequencer())

    # Set initial frame state in the sequencer
    manager._last_significant_frame_b64 = PNG_BLUE_B64
    manager._last_significant_frame_pil = manager._b64_to_image(PNG_BLUE_B64)

    # Mock the slow comparison functions to return a significant change
    with patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch.object(manager, "_calculate_mse", return_value=150.0), patch(
        "unity.screen_share_manager.screen_share_manager.ssim", return_value=0.5
    ):

        # Simulate a worker decoding a frame and putting it on the results queue
        event_data = {"payload": {"timestamp": 10.0, "frame_b64": PNG_RED_B64}}
        pil_image = manager._b64_to_image(PNG_RED_B64)
        await manager._results_queue.put((1, event_data, pil_image))

        await asyncio.sleep(0.01)  # Allow sequencer to process the result

        assert len(manager._pending_vision_events) == 1
        assert manager._pending_vision_events[0]["timestamp"] == 10.0
        assert manager._last_significant_frame_b64 == PNG_RED_B64

    sequencer_task.cancel()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_semantic_filter_prevents_event_creation(
    mocked_screen_share_manager, mock_loop
):
    """
    Tests that if a change fails the semantic check, no event is created by the sequencer.
    """
    manager, mocks = mocked_screen_share_manager
    sequencer_task = asyncio.create_task(manager._sequencer())
    manager._last_significant_frame_b64 = PNG_BLUE_B64
    manager._last_significant_frame_pil = manager._b64_to_image(PNG_BLUE_B64)

    # This time, the semantic check returns False
    with patch.object(
        manager, "_is_semantically_significant", return_value=False
    ), patch.object(manager, "_calculate_mse", return_value=150.0), patch(
        "unity.screen_share_manager.screen_share_manager.ssim", return_value=0.5
    ):

        event_data = {"payload": {"timestamp": 10.0, "frame_b64": PNG_RED_B64}}
        pil_image = manager._b64_to_image(PNG_RED_B64)
        await manager._results_queue.put((1, event_data, pil_image))

        await asyncio.sleep(0.01)

        assert len(manager._pending_vision_events) == 0
        assert (
            manager._last_significant_frame_b64 == PNG_BLUE_B64
        )  # State should not change

    sequencer_task.cancel()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_initial_frame_sets_baseline_and_creates_no_event(
    mocked_screen_share_manager, mock_loop
):
    """
    Tests that the very first frame processed just sets the baseline and
    doesn't trigger a change detection event.
    """
    manager, mocks = mocked_screen_share_manager
    sequencer_task = asyncio.create_task(manager._sequencer())
    assert manager._last_significant_frame_b64 is None

    # Simulate the first-ever frame
    event_data = {"payload": {"timestamp": 1.0, "frame_b64": PNG_BLUE_B64}}
    pil_image = manager._b64_to_image(PNG_BLUE_B64)
    await manager._results_queue.put((1, event_data, pil_image))

    await asyncio.sleep(0.01)

    assert manager._last_significant_frame_b64 == PNG_BLUE_B64
    assert len(manager._pending_vision_events) == 0

    sequencer_task.cancel()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_sequencer_processes_events_in_order(mocked_screen_share_manager):
    """
    Tests the core logic of the sequencer: even if results arrive out of order,
    they are processed sequentially, preventing race conditions.
    """
    manager, mocks = mocked_screen_share_manager
    sequencer_task = asyncio.create_task(manager._sequencer())

    # Mock the change detection to ensure every frame is treated as significant.
    # This isolates the test to only the sequencer's ordering logic.
    with patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch.object(manager, "_calculate_mse", return_value=999.0), patch(
        "unity.screen_share_manager.screen_share_manager.ssim", return_value=0.1
    ):
        # 1. Manually process initial frame to set baseline state
        event1 = {"payload": {"timestamp": 1.0, "frame_b64": PNG_BLUE_B64}}
        pil1 = manager._b64_to_image(PNG_BLUE_B64)
        await manager._results_queue.put((1, event1, pil1))
        await asyncio.sleep(0.01)
        assert manager._last_significant_frame_b64 == PNG_BLUE_B64
        # Clear pending events created by the first frame to isolate the next steps
        manager._pending_vision_events.clear()

        # 2. Put results on the queue OUT of order (3, then 2)
        event3 = {"payload": {"timestamp": 3.0, "frame_b64": PNG_GREEN_B64}}
        pil3 = manager._b64_to_image(PNG_GREEN_B64)
        await manager._results_queue.put((3, event3, pil3))

        event2 = {"payload": {"timestamp": 2.0, "frame_b64": PNG_RED_B64}}
        pil2 = manager._b64_to_image(PNG_RED_B64)
        await manager._results_queue.put((2, event2, pil2))

        # Allow sequencer to process both buffered and new results
        await asyncio.sleep(0.05)

        # 3. Assertions
        assert manager._last_significant_frame_b64 == PNG_GREEN_B64
        assert len(manager._pending_vision_events) == 2
        assert manager._pending_vision_events[0]["timestamp"] == 2.0
        assert manager._pending_vision_events[0]["after_frame_b64"] == PNG_RED_B64
        assert manager._pending_vision_events[1]["timestamp"] == 3.0
        assert manager._pending_vision_events[1]["after_frame_b64"] == PNG_GREEN_B64

    sequencer_task.cancel()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_speech_event_triggers_analysis_and_publishes_result(
    mocked_screen_share_manager,
):
    """
    Verifies that a PhoneUtteranceEvent triggers analysis and publishes a
    result with visually-grounded annotated image references.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.0,
                image_annotation="The 'Application Submitted' confirmation screen is now visible.",
                representative_timestamp=15.5,
            )
        ]
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response
    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "Okay, I am ready to submit the application now.",
            "start_time": 15.0,
            "end_time": 16.5,
        }
    }
    async with manager._state_lock:
        manager._pending_vision_events.append(
            {
                "timestamp": 15.5,
                "before_frame_b64": PNG_BLUE_B64,
                "after_frame_b64": PNG_RED_B64,
            }
        )
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    output_job = await manager._output_queue.get()
    await manager._output_worker(output_job)

    # Assert that an event was published to the correct channel
    mocks["event_broker"].publish.assert_any_call(
        "app:comms:screen_analysis_result", mock_is_json_string=True
    )

    # Find the specific call and inspect its payload
    publish_call = next(
        call
        for call in mocks["event_broker"].publish.call_args_list
        if call.args[0] == "app:comms:screen_analysis_result"
    )
    payload = json.loads(publish_call.args[1])["payload"]

    # Assert the AnnotatedImageRef structure
    assert "images" in payload
    image_refs_data = payload["images"]["root"]
    assert len(image_refs_data) == 1
    annotated_ref = image_refs_data[0]
    assert annotated_ref["raw_image_ref"]["image_id"] == 42
    assert (
        annotated_ref["annotation"]
        == "The 'Application Submitted' confirmation screen is now visible."
    )


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_vision_event_is_stored_and_published_on_next_utterance(
    mocked_screen_share_manager,
):
    """
    Tests that a silent visual event is analyzed, stored, and then published
    together with the next user utterance.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    manager.INACTIVITY_TIMEOUT_SEC = 0.1

    # 1. Simulate a silent visual event being detected and analyzed.
    manager._pending_vision_events.append(
        {
            "timestamp": 25.0,
            "before_frame_b64": PNG_BLUE_B64,
            "after_frame_b64": PNG_RED_B64,
        }
    )
    manager._last_activity_time = asyncio.get_event_loop().time() - 1.0
    silent_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                image_annotation="User navigated to the 'Profile' page.",
                representative_timestamp=25.0,
            )
        ]
    )
    mocks["analysis_client"].generate.return_value = silent_event_analysis

    await manager._flush_pending_events_on_timeout()
    await asyncio.sleep(0.01)

    # Assert NO job was created for output, but events were stored.
    mocks["analysis_client"].generate.assert_called_once()
    assert manager._output_queue.qsize() == 0
    assert len(manager._stored_silent_key_events) == 1
    assert "publish" not in [
        c[0] for c in mocks["event_broker"].method_calls
    ]  # No publish calls yet

    # 2. Now, simulate a subsequent user utterance.
    manager._frame_buffer.append((30.0, PNG_GREEN_B64))

    speech_event_data = {
        "payload": {
            "content": "Okay, I see my profile.",
            "start_time": 30.0,
            "end_time": 31.0,
        }
    }
    speech_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=30.0,
                image_annotation="User confirmed seeing their profile information.",
                representative_timestamp=30.0,
            )
        ]
    )
    mocks["analysis_client"].generate.return_value = speech_event_analysis
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)

    # Assert an output job was created and process it.
    assert manager._output_queue.qsize() == 1
    output_job = await manager._output_queue.get()
    await manager._output_worker(output_job)

    # Assertions: second analysis happened, and one publish call with combined events.
    assert mocks["analysis_client"].generate.call_count == 2
    publish_call = next(
        call
        for call in mocks["event_broker"].publish.call_args_list
        if call.args[0] == "app:comms:screen_analysis_result"
    )
    payload = json.loads(publish_call.args[1])["payload"]
    key_events = payload["key_events"]
    images = payload["images"]["root"]

    # Should have two events and two corresponding images.
    assert len(key_events) == 2
    assert len(images) == 2
    assert key_events[0]["image_annotation"] == "User navigated to the 'Profile' page."
    assert (
        key_events[1]["image_annotation"]
        == "User confirmed seeing their profile information."
    )


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_llm_failure_is_handled_gracefully(mocked_screen_share_manager):
    """
    Tests that if the LLM call fails, an output job is still created,
    just without any key events or images.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    mocks["analysis_client"].generate.side_effect = Exception("API Error")
    speech_event_data = {
        "payload": {
            "content": "test",
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
        }
    }
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    assert manager._output_queue.qsize() == 1

    output_job = await manager._output_queue.get()
    await manager._output_worker(output_job)

    mocks["analysis_client"].generate.assert_called_once()
    # A result event should still be published, just with empty events/images
    publish_call = next(
        call
        for call in mocks["event_broker"].publish.call_args_list
        if call.args[0] == "app:comms:screen_analysis_result"
    )
    payload = json.loads(publish_call.args[1])["payload"]
    assert payload["key_events"] == []
    assert payload["images"]["root"] == []


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_analysis_clears_pending_vision_events(mocked_screen_share_manager):
    """
    Ensures that after a turn analysis is triggered, the list of pending
    vision events is cleared.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    manager._pending_vision_events.append(
        {"timestamp": 1.0, "before_frame_b64": "b", "after_frame_b64": "a"}
    )
    assert len(manager._pending_vision_events) == 1
    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])
    manager._trigger_turn_analysis(
        speech_event={
            "payload": {"content": "go", "contact_details": {"contact_id": 1}}
        }
    )
    await asyncio.sleep(0.01)
    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_realtime_annotation_is_published_for_each_key_event(
    mocked_screen_share_manager,
):
    """
    Tests that a real-time event is published for every key event
    identified by the LLM.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=14.5,
                image_annotation="Event A happened.",
                representative_timestamp=14.5,
            ),
            KeyEvent(
                timestamp=15.0,
                image_annotation="Event B happened.",
                representative_timestamp=14.5,
            ),
        ]
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response
    speech_event_data = {
        "payload": {"contact_details": {"contact_id": 1}, "content": "dummy"}
    }
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)

    # Filter for the screen_annotation channel calls
    annotation_calls = [
        call
        for call in mocks["event_broker"].publish.call_args_list
        if call.args[0] == "app:comms:screen_annotation"
    ]
    assert len(annotation_calls) == 2
    published_events = [
        json.loads(call.args[1])["payload"]["annotation"] for call in annotation_calls
    ]
    assert "Event A happened." in published_events
    assert "Event B happened." in published_events


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_summary_is_updated_after_turn_analysis(mocked_screen_share_manager):
    """
    Tests that the session summary is updated correctly after an analysis.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    manager._session_summary = "The session has just begun."
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                image_annotation="User navigated to billing page.",
                representative_timestamp=15.5,
            )
        ]
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response
    with patch.object(
        manager, "_summary_client", new_callable=AsyncMock
    ) as mock_summary_client:
        mock_summary_client.generate.return_value = "User navigated to billing."
        speech_event_data = {
            "payload": {
                "contact_details": {"contact_id": 1},
                "content": "go to billing",
            }
        }
        manager._trigger_turn_analysis(speech_event=speech_event_data)
        await asyncio.sleep(0.01)
        await manager._update_summary()
        mock_summary_client.generate.assert_called_once()
        assert manager._session_summary == "User navigated to billing."
        assert len(manager._unsummarized_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_image_upload_retries_on_failure(mocked_screen_share_manager):
    """
    Tests that the output worker retries uploading images on transient errors.
    """
    manager, mocks = mocked_screen_share_manager
    manager.IMAGE_UPLOAD_MAX_RETRIES = 2
    manager.IMAGE_UPLOAD_INITIAL_BACKOFF = 0.01

    # Mock ImageManager's add_images to fail once, then succeed.
    add_images_mock = AsyncMock(side_effect=[Exception("Network Error"), [42]])
    mocks["image_manager"].add_images = add_images_mock

    speech_event = {
        "payload": {
            "content": "test",
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
        }
    }
    key_events = [
        KeyEvent(
            timestamp=1.0,
            image_annotation="desc",
            representative_timestamp=1.0,
        )
    ]
    frame_map = {1.0: PNG_BLUE_B64}

    # Manually create and run the worker coroutine with to_thread mock
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Configure the mock to simulate the side effect of the real add_images
        mock_to_thread.side_effect = add_images_mock
        await manager._output_worker((speech_event, key_events, frame_map))

    assert mock_to_thread.call_count == 2
    # The final publish should still have happened after the successful retry
    mocks["event_broker"].publish.assert_any_call(
        "app:comms:screen_analysis_result", mock_is_json_string=True
    )


# ==============================================================================
# Visual Change Detection Accuracy Tests (Unchanged)
# ==============================================================================
# NOTE: These tests do not need to be modified as they test the internal vision
# pipeline, which was not altered by the requested changes.


@pytest.mark.vision
@_handle_project
@pytest.mark.parametrize(
    "image_pair",
    [
        ("modal_before.png", "modal_after.png"),
        ("button_active_before.png", "button_active_after.png"),
    ],
)
def test_visual_change_detection_significant_changes(
    mocked_screen_share_manager, image_pair
):
    """
    Tests that the vision pipeline correctly identifies REAL, significant UI changes.
    Requires image assets in `tests/test_screen_share_manager/assets/`.
    """
    manager, _ = mocked_screen_share_manager
    before_filename, after_filename = image_pair

    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    # 1. MSE Pre-filter: Should pass the threshold for a real change
    mse = manager._calculate_mse(img_before, img_after)
    assert mse > manager.MSE_THRESHOLD

    # 2. SSIM Perceptual Check: Should be below the threshold, indicating a difference
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

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
def test_visual_change_detection_insignificant_changes(
    mocked_screen_share_manager, image_pair
):
    """
    Tests that the vision pipeline correctly IGNORES insignificant visual noise.
    Requires image assets in `tests/test_screen_share_manager/assets/`.
    """
    manager, _ = mocked_screen_share_manager
    before_filename, after_filename = image_pair

    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    # The full pipeline should result in the change being flagged as insignificant.
    # We test the final semantic filter, as MSE/SSIM may or may not catch these.
    is_significant = manager._is_semantically_significant(img_before, img_after)
    assert is_significant is False
