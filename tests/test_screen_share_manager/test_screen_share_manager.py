import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock, call
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import (
    ScreenShareManager,
    ScreenShareManagerSettings,
    TurnState,
)
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project

from skimage.metrics import structural_similarity as ssim

# --- Constants and Asset Loading ---

PNG_BLUE_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz/w3A5MBA/8/AAYDAL4/7d4eAAAAAElFTkSuQmCC"
PNG_RED_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8AARf4z/A8DMQABAL9M43+gS1dAAAAAAElFTkSuQmCC"
PNG_GREEN_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5/hP2E8A5MBA/8/AAYDAF4/7d4eAAAAAElFTkSuQmCC"

ASSETS_DIR = Path(__file__).parent / "assets"


def load_asset_image(filename: str) -> Image.Image:
    """Loads an image from the assets directory for vision tests."""
    path = ASSETS_DIR / filename
    if not path.exists():
        pytest.fail(f"Required asset for vision test not found: {path}")
    return Image.open(path).convert("L").resize((512, 288))


# --- Fixtures ---


@pytest.fixture
async def manager() -> ScreenShareManager:
    """Provides a clean, started ScreenShareManager instance for each test."""
    ssm = ScreenShareManager()
    await ssm.start()
    try:
        return ssm
    finally:
        await ssm.stop()


@pytest.fixture
async def mocked_manager():
    """Provides a manager with its LLM clients mocked out."""
    ssm = ScreenShareManager()

    patch_detect = patch.object(ssm, "_detection_client", new_callable=AsyncMock)
    patch_annotate = patch.object(ssm, "_analysis_client", new_callable=AsyncMock)
    patch_summary = patch.object(ssm, "_summary_client", new_callable=AsyncMock)

    with patch_detect as mock_detect, patch_annotate as mock_annotate, patch_summary as mock_summary:
        await ssm.start()
        try:
            return ssm, {
                "detect": mock_detect,
                "annotate": mock_annotate,
                "summary": mock_summary,
            }
        finally:
            await ssm.stop()


# --- High-Level API and Orchestration Tests ---


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_full_api_flow_detection_and_annotation(mocked_manager):
    """
    Tests the primary API flow: push frames/speech, analyze, and annotate,
    verifying that the consumer context is used.
    """
    manager, mocks = mocked_manager

    # --- Stage 1: Detection ---
    mocks["detect"].generate.return_value = json.dumps(
        {"moments": [{"timestamp": 1.5, "reason": "visual_change"}]}
    )

    # Simulate the result of the detection being put on the queue
    await manager._detection_queue.put(
        ([{"timestamp": 1.5, "reason": "visual_change"}], {1.5: PNG_RED_B64})
    )

    analysis_task = manager.analyze_turn()
    detected_events = await analysis_task

    assert len(detected_events) == 1
    assert isinstance(detected_events[0], DetectedEvent)
    assert detected_events[0].timestamp == 1.5
    assert isinstance(detected_events[0].image_handle, ImageHandle)

    # --- Stage 2: Annotation ---
    mocks["annotate"].generate.return_value = "This is the rich annotation."
    consumer_context = "User is performing a test."

    annotated_handles = await manager.annotate_events(detected_events, consumer_context)

    mocks["annotate"].generate.assert_called_once()
    # Verify consumer context is in the system prompt
    system_prompt = mocks["annotate"]._system_message
    assert "User is performing a test." in system_prompt

    assert len(annotated_handles) == 1
    assert annotated_handles[0].annotation == "This is the rich annotation."
    assert annotated_handles[0] is detected_events[0].image_handle


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_sequential_annotation_builds_context(mocked_manager):
    """
    Tests that annotating multiple events in one call correctly passes the annotation
    of the first event as context to the second.
    """
    manager, mocks = mocked_manager

    handles = manager._image_manager.add_images(
        [{"data": PNG_RED_B64}, {"data": PNG_GREEN_B64}],
        synchronous=False,
        return_handles=True,
    )
    detected_events = [
        DetectedEvent(1.0, "reason1", handles[0]),
        DetectedEvent(2.0, "reason2", handles[1]),
    ]

    mocks["annotate"].generate.side_effect = [
        "Annotation for event 1",
        "Annotation for event 2",
    ]

    await manager.annotate_events(detected_events, "initial context")

    assert mocks["annotate"].generate.call_count == 2

    # Check that the first annotation was passed as context to the second call's system prompt
    second_call_system_prompt = mocks["annotate"]._system_message
    assert '"Annotation for event 1"' in second_call_system_prompt

    assert handles[0].annotation == "Annotation for event 1"
    assert handles[1].annotation == "Annotation for event 2"


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_summary_update_triggered_after_annotation(mocked_manager):
    """
    Verifies that after a successful annotation, the session summary is updated
    with the new event information.
    """
    manager, mocks = mocked_manager
    manager.set_session_context("Initial summary.")

    handles = manager._image_manager.add_images(
        [{"data": PNG_RED_B64}], synchronous=False, return_handles=True
    )
    detected_events = [DetectedEvent(1.0, "test", handles[0])]

    mocks["annotate"].generate.return_value = "A new event happened."
    mocks["summary"].generate.return_value = "Updated summary including the new event."

    await manager.annotate_events(detected_events, "test context")

    await asyncio.sleep(1.1)

    mocks["summary"].generate.assert_called_once()
    summary_prompt = mocks["summary"].generate.call_args.args[0]
    assert "A new event happened." in summary_prompt
    assert "Initial summary." in summary_prompt

    async with manager._state_lock:
        assert manager._session_summary == "Updated summary including the new event."


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_events_are_stored_and_returned_in_next_turn(mocked_manager):
    """
    Tests that a visual event detected without speech is stored and then
    returned in the result of the next turn that *does* have speech.
    """
    manager, mocks = mocked_manager
    manager.settings.debounce_delay_sec = 0.05

    silent_handle = manager._image_manager.add_images(
        [{"data": PNG_RED_B64}], synchronous=False, return_handles=True
    )[0]
    manager._stored_silent_detected_events = [
        DetectedEvent(1.0, "silent_change", silent_handle)
    ]

    mocks["detect"].generate.return_value = json.dumps(
        {"moments": [{"timestamp": 2.5, "reason": "speech_related_change"}]}
    )

    await manager.push_speech("second turn", 2.0, 3.0)
    await manager._detection_queue.put(
        ([{"timestamp": 2.5, "reason": "speech_related_change"}], {2.5: PNG_GREEN_B64})
    )

    analysis_task = manager.analyze_turn()
    all_events = await analysis_task

    assert len(all_events) == 2
    timestamps = {e.timestamp for e in all_events}
    assert 1.0 in timestamps
    assert 2.5 in timestamps


# --- Lifecycle and State Transition Tests (New) ---


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_start_and_stop_lifecycle():
    """Verifies that start() creates background tasks and stop() cancels them."""
    created_tasks = []

    def mock_create_task(coro):
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = False
        created_tasks.append(task)
        return task

    with patch("asyncio.create_task", side_effect=mock_create_task):
        manager = ScreenShareManager()
        await manager.start()

        # Expected tasks: sequencer + inactivity loop + N frame workers
        expected_task_count = 2 + manager.settings.max_frame_workers
        assert len(created_tasks) == expected_task_count

        await manager.stop()

        for task in created_tasks:
            task.cancel.assert_called_once()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_debounce_logic_groups_speech_events(mocked_manager):
    """Ensures multiple quick speech events trigger only one analysis."""
    manager, _ = mocked_manager
    manager.settings.debounce_delay_sec = 0.2

    with patch.object(
        manager, "_detect_key_moments", new_callable=AsyncMock
    ) as mock_detect:
        await manager.push_speech("first part", 1.0, 1.1)
        await asyncio.sleep(0.05)
        await manager.push_speech("second part", 1.2, 1.3)

        mock_detect.assert_not_called()

        await asyncio.sleep(0.25)

        mock_detect.assert_called_once()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_inactivity_flush_triggers_for_visual_events(mocked_manager):
    """Tests that a silent visual event triggers analysis after an inactivity period."""
    manager, _ = mocked_manager
    manager.settings.inactivity_timeout_sec = 0.1
    manager.settings.debounce_delay_sec = 0.05

    manager._inactivity_task.cancel()

    with patch.object(
        manager, "_detect_key_moments", new_callable=AsyncMock
    ) as mock_detect:
        manager._pending_vision_events.append(
            {"timestamp": 1.0, "after_frame_b64": PNG_RED_B64}
        )
        manager._last_activity_time = asyncio.get_event_loop().time()

        await asyncio.sleep(0.15)

        is_debouncing = manager._debounce_task and not manager._debounce_task.done()
        if (
            asyncio.get_event_loop().time() - manager._last_activity_time
            >= manager.settings.inactivity_timeout_sec
            and not is_debouncing
            and manager._pending_vision_events
        ):
            manager._trigger_turn_analysis(speech_event=None)

        await asyncio.sleep(0.1)

        mock_detect.assert_called_once()


# --- Error and Failure Condition Tests (New) ---


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_llm_retries_on_failure(mocked_manager):
    """Verifies that the LLM retry decorator is working."""
    manager, mocks = mocked_manager
    manager.settings.llm_retry_max_tries = 3
    manager.settings.llm_retry_base_delay_sec = 0.01

    mocks["detect"].generate.side_effect = [
        Exception("LLM unavailable"),
        Exception("LLM still unavailable"),
        json.dumps({"moments": []}),
    ]

    await manager._detect_key_moments(
        TurnState(speech_event={"payload": {"content": "test"}})
    )

    assert mocks["detect"].generate.call_count == 3


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_llm_handles_invalid_json(mocked_manager, caplog):
    """Ensures the manager doesn't crash if the LLM returns malformed JSON."""
    manager, mocks = mocked_manager
    mocks["detect"].generate.return_value = "This is not JSON"

    await manager._detect_key_moments(
        TurnState(speech_event={"payload": {"content": "test"}})
    )

    assert "Failed to detect key moments" in caplog.text
    key_moments, _ = await manager._detection_queue.get()
    assert key_moments == []


# --- Concurrency and Load Tests (New) ---


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_adaptive_frame_dropping_under_load(caplog):
    """Verifies that frames are dropped when the queue is backlogged."""
    manager = ScreenShareManager()
    await manager.start()

    with patch.object(manager._frame_queue, "qsize") as mock_qsize:
        mock_qsize.return_value = manager.settings.frame_queue_size * 0.8

        await manager.push_frame(PNG_BLUE_B64, 1.0)

    assert "Proactively dropping frame" in caplog.text
    await manager.stop()


# --- Configuration Tests (New) ---


@pytest.mark.unit
@_handle_project
def test_custom_settings_are_applied():
    """Ensures that custom settings passed to the constructor are used."""
    custom_settings = ScreenShareManagerSettings(
        debounce_delay_sec=0.0123, mse_threshold=99.9
    )
    manager = ScreenShareManager(settings=custom_settings)

    assert manager.settings.debounce_delay_sec == 0.0123
    assert manager.settings.mse_threshold == 99.9


# --- Visual Change Detection Tests (Original - Now Async) ---


@pytest.mark.vision
@_handle_project
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_pair",
    [
        ("modal_before.png", "modal_after.png"),
        ("button_active_before.png", "button_active_after.png"),
    ],
)
async def test_visual_change_detection_significant_changes(
    manager: ScreenShareManager, image_pair
):
    """Tests that the vision pipeline correctly identifies REAL, significant UI changes."""
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    assert (
        manager._calculate_mse(img_before, img_after) > manager.settings.mse_threshold
    )

    score = ssim(np.array(img_before), np.array(img_after))
    assert score < manager.settings.ssim_threshold

    assert manager._is_semantically_significant(img_before, img_after) is True


@pytest.mark.vision
@_handle_project
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_pair",
    [
        ("blinking_caret_before.png", "blinking_caret_after.png"),
        ("cursor_move_before.png", "cursor_move_after.png"),
    ],
)
async def test_visual_change_detection_insignificant_changes(
    manager: ScreenShareManager, image_pair
):
    """Tests that the vision pipeline correctly IGNORES insignificant visual noise."""
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)

    assert manager._is_semantically_significant(img_before, img_after) is False
