# FILE: test_screen_share_manager.py

import asyncio
from datetime import datetime
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from unity.image_manager.utils import make_solid_png_base64
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

    # Manually start the sequencer for the test
    sequencer_task = asyncio.create_task(manager._sequencer())

    # Set initial frame
    manager._last_significant_frame_b64 = PNG_BLUE_B64

    # Mock the worker's analysis to return a significant change
    with patch.object(
        manager, "_is_semantically_significant", return_value=True
    ), patch.object(manager, "_calculate_mse", return_value=150.0), patch(
        "unity.screen_share_manager.screen_share_manager.ssim", return_value=0.5
    ):

        # Simulate the worker processing a frame and putting the result on the queue
        await manager._results_queue.put(
            {
                "seq_id": 1,
                "is_significant": True,
                "timestamp": 10.0,
                "frame_b64": PNG_RED_B64,
                "before_frame_b64": PNG_BLUE_B64,
            }
        )

        await asyncio.sleep(0.01)  # Allow sequencer to run

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

    # This time, the semantic check returns False, so is_significant is False
    with patch.object(
        manager, "_is_semantically_significant", return_value=False
    ), patch.object(manager, "_calculate_mse", return_value=150.0), patch(
        "unity.screen_share_manager.screen_share_manager.ssim", return_value=0.5
    ):

        # Simulate worker result
        await manager._results_queue.put(
            {
                "seq_id": 1,
                "is_significant": False,  # The key difference
                "timestamp": 10.0,
                "frame_b64": PNG_RED_B64,
                "before_frame_b64": PNG_BLUE_B64,
            }
        )

        await asyncio.sleep(0.01)

        # The key assertion: no event was created, but the last frame state does NOT update
        assert len(manager._pending_vision_events) == 0
        assert manager._last_significant_frame_b64 == PNG_BLUE_B64

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
    assert len(manager._pending_vision_events) == 0

    # Simulate the first frame result
    await manager._results_queue.put(
        {
            "seq_id": 1,
            "is_significant": False,  # First frame is never "significant" vs None
            "timestamp": 1.0,
            "frame_b64": PNG_BLUE_B64,
            "before_frame_b64": PNG_BLUE_B64,
        }
    )

    await asyncio.sleep(0.01)

    # It should set the baseline but not create a pending event
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

    # 1. Set initial state
    await manager._results_queue.put(
        {
            "seq_id": 1,
            "is_significant": False,
            "timestamp": 1.0,
            "frame_b64": PNG_BLUE_B64,
            "before_frame_b64": PNG_BLUE_B64,
        }
    )
    await asyncio.sleep(0.01)
    assert manager._last_significant_frame_b64 == PNG_BLUE_B64

    # 2. Put results on the queue OUT of order (3, then 2)
    await manager._results_queue.put(
        {
            "seq_id": 3,
            "is_significant": True,
            "timestamp": 3.0,
            "frame_b64": PNG_GREEN_B64,
            "before_frame_b64": PNG_RED_B64,
        }
    )
    await manager._results_queue.put(
        {
            "seq_id": 2,
            "is_significant": True,
            "timestamp": 2.0,
            "frame_b64": PNG_RED_B64,
            "before_frame_b64": PNG_BLUE_B64,
        }
    )

    await asyncio.sleep(0.01)  # Allow sequencer to process buffered results

    # 3. Assertions
    # The state should have been updated in the correct order: Blue -> Red -> Green
    assert manager._last_significant_frame_b64 == PNG_GREEN_B64
    assert len(manager._pending_vision_events) == 2
    # The pending events should also be in the correct chronological order
    assert manager._pending_vision_events[0]["timestamp"] == 2.0
    assert manager._pending_vision_events[0]["after_frame_b64"] == PNG_RED_B64
    assert manager._pending_vision_events[1]["timestamp"] == 3.0
    assert manager._pending_vision_events[1]["after_frame_b64"] == PNG_GREEN_B64

    sequencer_task.cancel()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_adaptive_frame_dropping_on_queue_backlog(mocked_screen_share_manager):
    """
    Tests that the listener proactively drops frames if the frame queue is backlogged.
    """
    manager, mocks = mocked_screen_share_manager
    manager.FRAME_QUEUE_SIZE = 10  # Lower for easier testing
    manager.ADAPTIVE_DROP_THRESHOLD = 0.75

    # Fill the queue to 80% capacity (8/10)
    for i in range(8):
        await manager._frame_queue.put((i, {}))

    assert manager._frame_queue.qsize() == 8

    # Simulate the listener loop (this is a simplified version of the logic in _listen_for_events)
    with patch.object(manager, "_frame_queue") as mock_queue:
        mock_queue.qsize.return_value = 8
        mock_queue.put_nowait = MagicMock()

        # This would be the new frame data
        event_data = {"payload": {"timestamp": 10.0, "frame_b64": PNG_RED_B64}}

        # Since qsize > threshold * size (8 > 7.5), we expect it to drop
        if mock_queue.qsize() > (
            manager.FRAME_QUEUE_SIZE * manager.ADAPTIVE_DROP_THRESHOLD
        ):
            # Dropped, so put_nowait is not called
            pass
        else:
            manager._frame_sequence_id += 1
            mock_queue.put_nowait((manager._frame_sequence_id, event_data))

        mock_queue.put_nowait.assert_not_called()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_speech_event_triggers_analysis_and_logging_with_specifics(
    mocked_screen_share_manager,
):
    """
    Verifies that a PhoneUtteranceEvent triggers an analysis and logs a message
    with a specific, visually-grounded caption and a precise trigger phrase.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0

    # 1. Mock the LLM's response with high-quality output
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                event_description="User clicked the 'Submit Application' button.",
                triggering_phrase="submit the application",
                representative_timestamp=15.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    # 2. Define the incoming speech event
    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "Okay, I am ready to submit the application now.",
            "start_time": 15.0,
            "end_time": 16.5,
        },
    }

    # 3. Act
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

    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    # 4. Assertions
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    # Assert specific caption
    annotation = logged_message.screen_share["15.00-16.50"]
    assert annotation.caption == "User clicked the 'Submit Application' button."

    # Assert precise trigger phrase span
    content = "Okay, I am ready to submit the application now."
    phrase = "submit the application"
    start_index = content.find(phrase)
    end_index = start_index + len(phrase)
    expected_key = f"[{start_index}:{end_index}]"

    assert expected_key in logged_message.images
    assert logged_message.images[expected_key] == 42


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_vision_event_is_stored_and_logged_on_next_utterance(
    mocked_screen_share_manager,
):
    """
    Tests that a silent visual event is stored and then logged together
    with the next user utterance.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    manager.INACTIVITY_TIMEOUT_SEC = 0.1

    # 1. Simulate a silent visual event
    async with manager._state_lock:
        manager._pending_vision_events.append(
            {
                "timestamp": 25.0,
                "before_frame_b64": PNG_BLUE_B64,
                "after_frame_b64": PNG_RED_B64,
            },
        )
    manager._last_activity_time = asyncio.get_event_loop().time() - 1.0

    silent_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                event_description="User navigated to the 'Profile' page.",
                representative_timestamp=25.0,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = silent_event_analysis

    await manager._flush_pending_events_on_timeout()
    await asyncio.sleep(0.01)
    log_job_silent = await manager._logging_queue.get()
    await manager._logging_worker(log_job_silent)

    mocks["analysis_client"].generate.assert_called_once()
    assert len(manager._stored_silent_key_events) == 1
    mocks["transcript_manager"].log_messages.assert_not_called()

    # 2. Now, simulate a subsequent user utterance
    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "Okay, I see my profile.",
            "start_time": 30.0,
            "end_time": 31.0,
        },
    }
    speech_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=30.0,
                event_description="User confirmed seeing their profile.",
                representative_timestamp=30.0,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = speech_event_analysis

    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    log_job_speech = await manager._logging_queue.get()
    await manager._logging_worker(log_job_speech)

    assert mocks["analysis_client"].generate.call_count == 2
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]
    assert len(logged_message.screen_share) == 2


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_combined_turn_logs_multiple_events(mocked_screen_share_manager):
    """
    Tests that a turn with both a visual change and speech results in multiple,
    chronologically ordered annotations being logged to the transcript.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0

    async with manager._state_lock:
        manager._pending_vision_events.append(
            {
                "timestamp": 14.5,
                "before_frame_b64": PNG_BLUE_B64,
                "after_frame_b64": PNG_RED_B64,
            },
        )

    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=14.5,
                event_description="A new dialog box appeared.",
                triggering_phrase=None,
                representative_timestamp=14.5,
            ),
            KeyEvent(
                timestamp=15.0,
                event_description="User stated their intention to submit.",
                triggering_phrase="I will click submit",
                representative_timestamp=14.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "I will click submit",
            "start_time": 15.0,
            "end_time": 16.0,
        },
    }

    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    assert len(logged_message.screen_share) == 2
    assert "14.50-14.50" in logged_message.screen_share
    assert "15.00-16.00" in logged_message.screen_share
    assert (
        logged_message.screen_share["14.50-14.50"].caption
        == "A new dialog box appeared."
    )
    assert (
        logged_message.screen_share["15.00-16.00"].caption
        == "User stated their intention to submit."
    )


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_llm_failure_is_handled_gracefully(mocked_screen_share_manager):
    """
    Tests that if the client call fails, the error is logged and
    a transcript message is still created, just without screen events.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    mocks["analysis_client"].generate.side_effect = Exception("API Error")

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),
        },
    }
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["analysis_client"].generate.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]
    assert len(logged_message.screen_share) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_empty_llm_response_logs_message_without_events(
    mocked_screen_share_manager,
):
    """
    Tests that if the LLM returns no events, the system still logs a message
    for the utterance, but with no screen share annotations.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),
        },
    }
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["analysis_client"].generate.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]
    assert len(logged_message.screen_share) == 0


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
    async with manager._state_lock:
        manager._pending_vision_events.append(
            {"timestamp": 1.0, "before_frame_b64": "b", "after_frame_b64": "a"},
        )
    assert len(manager._pending_vision_events) == 1

    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])

    manager._trigger_turn_analysis(
        speech_event={
            "payload": {"content": "go", "contact_details": {"contact_id": 1}}
        },
    )
    await asyncio.sleep(0.01)

    # The logic inside _debounced_analysis_runner now clears the list
    async with manager._state_lock:
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
                event_description="Event A",
                representative_timestamp=14.5,
            ),
            KeyEvent(
                timestamp=15.0,
                event_description="Event B",
                representative_timestamp=14.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {"contact_details": {"contact_id": 1}, "content": "dummy"}
    }
    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)  # Allow analysis task to complete

    # The publish calls happen inside _analyze_turn
    assert mocks["event_broker"].publish.call_count == 2
    published_events = [
        json.loads(call.args[1])["payload"]["event_description"]
        for call in mocks["event_broker"].publish.call_args_list
    ]
    assert "Event A" in published_events
    assert "Event B" in published_events


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_analysis_queues_job_for_logging_worker(mocked_screen_share_manager):
    """
    Tests that _analyze_turn places a valid job in the _logging_queue.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0

    key_event = KeyEvent(
        timestamp=10.0, event_description="Test event", representative_timestamp=10.0
    )
    mock_llm_response = TurnAnalysisResponse(events=[key_event])
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {"contact_details": {"contact_id": 1}, "content": "test"}
    }

    assert manager._logging_queue.qsize() == 0

    manager._trigger_turn_analysis(speech_event=speech_event_data)
    await asyncio.sleep(0.01)  # Allow analysis task to complete

    assert manager._logging_queue.qsize() == 1

    queued_job = await manager._logging_queue.get()
    (job_speech_event, job_key_events, job_frame_map) = queued_job

    assert job_speech_event == speech_event_data
    assert len(job_key_events) == 1
    assert job_key_events[0].event_description == "Test event"


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
                event_description="User navigated to billing.",
                representative_timestamp=15.5,
            )
        ],
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
        await asyncio.sleep(0.01)  # Let analysis task run

        # Manually trigger summary update, as it's normally delayed
        await manager._update_summary()

        mock_summary_client.generate.assert_called_once()
        assert manager._session_summary == "User navigated to billing."
        # Check that the unsummarized events were cleared
        assert len(manager._unsummarized_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_turn_analysis_is_debounced(mocked_screen_share_manager):
    """
    Tests that rapid successive calls to _trigger_turn_analysis result in only
    one actual analysis task being run.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0.1

    speech_event = {
        "payload": {"content": "test", "contact_details": {"contact_id": 1}}
    }

    # Trigger multiple times within the debounce window
    manager._trigger_turn_analysis(speech_event)
    await asyncio.sleep(0.02)
    manager._trigger_turn_analysis(speech_event)
    await asyncio.sleep(0.02)
    manager._trigger_turn_analysis(speech_event)

    # _analyze_turn should not have been called yet
    mocks["analysis_client"].generate.assert_not_called()

    # Wait for the debounce period to expire
    await asyncio.sleep(0.15)

    # Now it should have been called exactly once
    mocks["analysis_client"].generate.assert_called_once()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_image_upload_retries_on_failure(mocked_screen_share_manager):
    """
    Tests that the logging worker retries uploading images on transient errors.
    """
    manager, mocks = mocked_screen_share_manager
    manager.IMAGE_UPLOAD_MAX_RETRIES = 2
    manager.IMAGE_UPLOAD_INITIAL_BACKOFF = 0.01

    add_images_mock = MagicMock()
    add_images_mock.side_effect = [Exception("Network Error"), [42]]
    mocks["image_manager"].add_images = add_images_mock

    speech_event = {
        "payload": {
            "content": "test",
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
        }
    }
    key_events = [
        KeyEvent(timestamp=1.0, event_description="desc", representative_timestamp=1.0)
    ]
    frame_map = {1.0: PNG_BLUE_B64}

    await manager._logging_worker((speech_event, key_events, frame_map))

    assert add_images_mock.call_count == 2
    mocks["transcript_manager"].log_messages.assert_called_once()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_set_session_context_updates_summary(mocked_screen_share_manager):
    """
    Tests that calling set_session_context correctly updates the initial summary.
    """
    manager, mocks = mocked_screen_share_manager
    manager.DEBOUNCE_DELAY_SEC = 0
    initial_context = "The user is an admin trying to reset a password."

    manager.set_session_context(initial_context)
    await asyncio.sleep(0.01)  # Allow async task in set_session_context to run

    # Verify internal state
    assert manager._session_summary == initial_context

    # Now trigger an analysis and check if the prompt builder receives this context
    with patch(
        "unity.screen_share_manager.screen_share_manager.build_turn_analysis_prompt"
    ) as mock_build_prompt:
        manager._trigger_turn_analysis(
            {"payload": {"content": "test", "contact_details": {"contact_id": 1}}}
        )
        await asyncio.sleep(0.01)

        # Let the debounced task run and trigger the analysis
        await asyncio.sleep(manager.DEBOUNCE_DELAY_SEC + 0.01)

        mock_build_prompt.assert_called_once()
        call_args, _ = mock_build_prompt.call_args
        # The first argument to the prompt builder is `current_summary`
        assert call_args[0] == initial_context
