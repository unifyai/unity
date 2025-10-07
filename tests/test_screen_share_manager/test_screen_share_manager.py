# FILE: test_screen_share_manager.py

import asyncio
from datetime import datetime
import json
from unittest.mock import patch, AsyncMock

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


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_ssim_change_detection_creates_pending_event(mocked_screen_share_manager):
    """
    Tests that a significant visual difference between frames triggers the
    creation of a 'pending_vision_event'.
    """
    manager, mocks = mocked_screen_share_manager

    # Set an initial frame
    manager._last_significant_frame_b64 = PNG_BLUE_B64

    # Handle a new, different frame
    await manager._handle_frame_event(
        {"payload": {"timestamp": 10.0, "frame_b64": PNG_RED_B64}},
    )

    assert len(manager._pending_vision_events) == 1
    event = manager._pending_vision_events[0]
    assert event["timestamp"] == 10.0
    assert event["before_frame_b64"] == PNG_BLUE_B64
    assert event["after_frame_b64"] == PNG_RED_B64


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_no_ssim_change_does_not_create_pending_event(
    mocked_screen_share_manager,
):
    """
    Tests that if frames are visually similar (above SSIM threshold),
    no pending event is created.
    """
    manager, mocks = mocked_screen_share_manager

    # Set an initial frame
    manager._last_significant_frame_b64 = PNG_BLUE_B64

    # Handle a new, identical frame
    await manager._handle_frame_event(
        {"payload": {"timestamp": 10.0, "frame_b64": PNG_BLUE_B64}},
    )

    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_initial_frame_sets_baseline_and_creates_no_event(
    mocked_screen_share_manager,
):
    """
    Tests that the very first frame processed just sets the baseline and
    doesn't trigger a change detection event.
    """
    manager, mocks = mocked_screen_share_manager

    assert manager._last_significant_frame_b64 is None
    assert len(manager._pending_vision_events) == 0

    # Handle the first-ever frame
    await manager._handle_frame_event(
        {"payload": {"timestamp": 1.0, "frame_b64": PNG_BLUE_B64}},
    )

    # It should set the baseline but not create a pending event
    assert manager._last_significant_frame_b64 == PNG_BLUE_B64
    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_speech_event_triggers_analysis_and_logging(mocked_screen_share_manager):
    """
    Verifies that a PhoneUtteranceEvent correctly triggers an LLM analysis
    and subsequently logs a rich Message to the TranscriptManager.
    """
    manager, mocks = mocked_screen_share_manager

    # 1. Mock the LLM's response
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                event_description="User clicked the 'Submit' button.",
                triggering_phrase="click this button",
                representative_timestamp=15.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    # 2. Define the incoming speech event with clear start/end times
    speech_event_data = {
        "event_name": "PhoneUtterance",
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "Okay, I will click this button now.",
            "start_time": 15.0,
            "end_time": 16.5,
        },
    }

    # 3. Act: Provide the visual event that the LLM is referencing
    visual_events = [
        {
            "timestamp": 15.5,  # Matches representative_timestamp
            "before_frame_b64": PNG_BLUE_B64,
            "after_frame_b64": PNG_RED_B64,
        }
    ]
    await manager._analyze_turn(
        speech_event=speech_event_data, visual_events=visual_events
    )

    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    # 4. Assertions
    mocks["analysis_client"].generate.assert_called_once()
    mocks["image_manager"].add_images.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    mocks["event_broker"].publish.assert_called_once()

    call_args = mocks["transcript_manager"].log_messages.call_args
    logged_message = call_args[0][0][0]

    assert logged_message.content == "Okay, I will click this button now."
    assert "15.00-16.50" in logged_message.screen_share
    annotation = logged_message.screen_share["15.00-16.50"]
    assert annotation.caption == "User clicked the 'Submit' button."
    assert annotation.image == PNG_RED_B64
    assert annotation.type == "speech"

    assert "[13:30]" in logged_message.images
    assert logged_message.images["[13:30]"] == 42


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

    # 1. Simulate a silent visual event on timeout
    manager.INACTIVITY_TIMEOUT_SEC = 0.5
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

    # Flush the silent event and MANUALLY run the worker to ensure it's processed
    await manager._flush_pending_events_on_timeout()
    log_job_silent = await manager._logging_queue.get()
    await manager._logging_worker(log_job_silent)

    # Assertions for the silent part
    mocks["analysis_client"].generate.assert_called_once()
    assert len(manager._stored_silent_key_events) == 1
    assert manager._stored_silent_key_events[0].timestamp == 25.0
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
                triggering_phrase="see my profile",
                representative_timestamp=30.0,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = speech_event_analysis

    # Handle the utterance event, providing a visual frame for context
    visual_events_speech = [
        {"timestamp": 30.0, "before_frame_b64": "b", "after_frame_b64": PNG_RED_B64}
    ]
    await manager._analyze_turn(
        speech_event=speech_event_data, visual_events=visual_events_speech
    )
    log_job_speech = await manager._logging_queue.get()
    await manager._logging_worker(log_job_speech)

    # Assertions for the combined logging
    assert mocks["analysis_client"].generate.call_count == 2
    mocks["transcript_manager"].log_messages.assert_called_once()

    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    assert len(logged_message.screen_share) == 2
    assert "25.00-25.00" in logged_message.screen_share
    assert "30.00-31.00" in logged_message.screen_share
    assert (
        logged_message.screen_share["25.00-25.00"].caption
        == "User navigated to the 'Profile' page."
    )
    assert (
        logged_message.screen_share["30.00-31.00"].caption
        == "User confirmed seeing their profile."
    )

    assert len(manager._stored_silent_key_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_combined_turn_logs_multiple_events(mocked_screen_share_manager):
    """
    Tests that a turn with both a visual change and speech results in multiple,
    chronologically ordered annotations being logged to the transcript.
    """
    manager, mocks = mocked_screen_share_manager

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

    await manager._analyze_turn(
        speech_event=speech_event_data, visual_events=manager._pending_vision_events
    )
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

    assert "[0:19]" in logged_message.images
    # The second event gets the second image_id, which is 43 due to the dynamic mock
    assert logged_message.images["[0:19]"] == 43

    assert mocks["event_broker"].publish.call_count == 2


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_llm_failure_is_handled_gracefully(mocked_screen_share_manager):
    """
    Tests that if the client call fails, the error is logged and
    a transcript message is still created, just without screen events.
    """
    manager, mocks = mocked_screen_share_manager
    mocks["analysis_client"].generate.side_effect = Exception("API Error")

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),
        },
    }
    await manager._analyze_turn(speech_event=speech_event_data, visual_events=[])
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["analysis_client"].generate.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]
    assert len(logged_message.screen_share) == 0
    assert len(logged_message.images) == 0


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
    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(
        events=[],
    )

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),
        },
    }
    await manager._analyze_turn(speech_event=speech_event_data, visual_events=[])
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["analysis_client"].generate.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]
    assert len(logged_message.screen_share) == 0
    assert len(logged_message.images) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_analysis_clears_pending_vision_events(mocked_screen_share_manager):
    """
    Ensures that after any analysis run, the list of pending vision events is cleared.
    """
    manager, mocks = mocked_screen_share_manager
    async with manager._state_lock:
        manager._pending_vision_events.append(
            {"timestamp": 1.0, "before_frame_b64": "b", "after_frame_b64": "a"},
        )
    assert len(manager._pending_vision_events) == 1

    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(
        events=[],
    )

    await manager._trigger_turn_analysis(
        speech_event={
            "payload": {
                "content": "go",
                "contact_details": {"contact_id": 1},
                "timestamp": datetime.now().isoformat(),
            },
        },
    )
    await asyncio.sleep(0.01)

    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_event_without_prior_utterance_is_stored(
    mocked_screen_share_manager,
):
    """
    Tests that if a silent event occurs but there's no prior utterance,
    the event is stored for the next turn.
    """
    manager, mocks = mocked_screen_share_manager
    manager._last_user_utterance_message_id = None

    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                event_description="Desc",
                representative_timestamp=25.0,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    visual_events = [
        {"timestamp": 25.0, "before_frame_b64": "b", "after_frame_b64": "a"}
    ]
    await manager._analyze_turn(speech_event=None, visual_events=visual_events)
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["analysis_client"].generate.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_not_called()
    assert len(manager._stored_silent_key_events) == 1
    assert manager._stored_silent_key_events[0].timestamp == 25.0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_triggering_phrase_not_found_in_content_is_handled(
    mocked_screen_share_manager,
):
    """
    Tests that if the LLM returns a triggering_phrase that doesn't exist,
    it's handled gracefully without a broken image link.
    """
    manager, mocks = mocked_screen_share_manager

    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                event_description="User clicked.",
                triggering_phrase="a phrase that does not exist",
                representative_timestamp=15.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "The actual spoken words.",
            "start_time": 15.0,
            "end_time": 16.5,
        },
    }

    visual_events = [
        {
            "timestamp": 15.5,
            "before_frame_b64": "b",
            "after_frame_b64": PNG_RED_B64,
        }
    ]
    await manager._analyze_turn(
        speech_event=speech_event_data, visual_events=visual_events
    )
    log_job = await manager._logging_queue.get()
    await manager._logging_worker(log_job)

    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    assert "15.00-16.50" in logged_message.screen_share
    assert len(logged_message.images) == 0


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

    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=14.5,
                event_description="Event A: A modal appeared.",
                representative_timestamp=14.5,
            ),
            KeyEvent(
                timestamp=15.0,
                event_description="Event B: User expressed intent.",
                representative_timestamp=14.5,
            ),
            KeyEvent(
                timestamp=15.8,
                event_description="Event C: User clicked a button.",
                representative_timestamp=15.8,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "dummy speech",
            "timestamp": datetime.now().isoformat(),
        },
    }

    await manager._analyze_turn(speech_event=speech_event_data, visual_events=[])

    assert mocks["event_broker"].publish.call_count == 3
    published_descriptions = [
        json.loads(call.args[1])["payload"]["event_description"]
        for call in mocks["event_broker"].publish.call_args_list
    ]
    expected_descriptions = [
        "Event A: A modal appeared.",
        "Event B: User expressed intent.",
        "Event C: User clicked a button.",
    ]
    assert sorted(published_descriptions) == sorted(expected_descriptions)


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_rapid_event_burst_is_sampled(mocked_screen_share_manager):
    """
    Tests that a rapid succession of visual events is sampled.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    visual_events = [
        {"timestamp": 10.0, "before_frame_b64": "b1", "after_frame_b64": PNG_RED_B64},
        {"timestamp": 10.5, "before_frame_b64": "b2", "after_frame_b64": PNG_GREEN_B64},
        {
            "timestamp": 11.0,
            "before_frame_b64": "b3",
            "after_frame_b64": PNG_YELLOW_B64,
        },
        {"timestamp": 11.5, "before_frame_b64": "b4", "after_frame_b64": PNG_CYAN_B64},
        {
            "timestamp": 12.0,
            "before_frame_b64": "b5",
            "after_frame_b64": PNG_MAGENTA_B64,
        },
    ]

    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])
    await manager._analyze_turn(speech_event=None, visual_events=visual_events)

    mocks["analysis_client"].generate.assert_called_once()
    call_args = mocks["analysis_client"].generate.call_args
    user_content = call_args.kwargs["user_message"]

    assert any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 3


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_slow_events_are_not_sampled(mocked_screen_share_manager):
    """
    Tests that if visual events are spaced further apart, they are not sampled.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    visual_events = [
        {"timestamp": 10.0, "before_frame_b64": "b1", "after_frame_b64": PNG_RED_B64},
        {"timestamp": 13.0, "before_frame_b64": "b2", "after_frame_b64": PNG_GREEN_B64},
        {
            "timestamp": 16.0,
            "before_frame_b64": "b3",
            "after_frame_b64": PNG_YELLOW_B64,
        },
        {"timestamp": 19.0, "before_frame_b64": "b4", "after_frame_b64": PNG_CYAN_B64},
    ]

    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])
    await manager._analyze_turn(speech_event=None, visual_events=visual_events)

    mocks["analysis_client"].generate.assert_called_once()
    call_args = mocks["analysis_client"].generate.call_args
    user_content = call_args.kwargs["user_message"]

    assert not any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 4


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_mixed_bursts_and_single_events_are_handled_correctly(
    mocked_screen_share_manager,
):
    """
    Tests the logic handles a complex sequence of single and burst events.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    visual_events = [
        {"timestamp": 10.0, "before_frame_b64": "b1", "after_frame_b64": PNG_RED_B64},
        {"timestamp": 13.0, "before_frame_b64": "b2", "after_frame_b64": PNG_GREEN_B64},
        {
            "timestamp": 13.5,
            "before_frame_b64": "b3",
            "after_frame_b64": PNG_YELLOW_B64,
        },
        {"timestamp": 14.0, "before_frame_b64": "b4", "after_frame_b64": PNG_CYAN_B64},
        {
            "timestamp": 14.5,
            "before_frame_b64": "b5",
            "after_frame_b64": PNG_MAGENTA_B64,
        },
        {"timestamp": 18.0, "before_frame_b64": "b6", "after_frame_b64": PNG_WHITE_B64},
    ]

    mocks["analysis_client"].generate.return_value = TurnAnalysisResponse(events=[])
    await manager._analyze_turn(speech_event=None, visual_events=visual_events)

    mocks["analysis_client"].generate.assert_called_once()
    call_args = mocks["analysis_client"].generate.call_args
    user_content = call_args.kwargs["user_message"]

    assert any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 5


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_analysis_queues_job_for_logging_worker(mocked_screen_share_manager):
    """
    Tests that _analyze_turn places a job in the _logging_queue.
    """
    manager, mocks = mocked_screen_share_manager

    key_event = KeyEvent(
        timestamp=10.0,
        event_description="Test event",
        representative_timestamp=10.0,
    )
    mock_llm_response = TurnAnalysisResponse(events=[key_event])
    mocks["analysis_client"].generate.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "content": "test",
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
        }
    }

    assert manager._logging_queue.qsize() == 0
    await manager._analyze_turn(speech_event=speech_event_data, visual_events=[])
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
    Tests that the session summary is updated correctly.
    """
    manager, mocks = mocked_screen_share_manager

    manager._session_summary = "The session has just begun."
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                event_description="User navigated to the billing page.",
                representative_timestamp=15.5,
            ),
        ],
    )
    mocks["analysis_client"].generate.return_value = mock_llm_response

    with patch(
        "unity.screen_share_manager.screen_share_manager.unify.AsyncUnify"
    ) as mock_unify:
        mock_summary_instance = AsyncMock()
        mock_summary_instance.generate.return_value = "User navigated to billing."
        mock_unify.side_effect = [mocks["analysis_client"], mock_summary_instance]
        manager._summary_client = mock_summary_instance

        speech_event_data = {
            "payload": {
                "content": "go to billing",
                "contact_details": {"contact_id": 1},
                "timestamp": datetime.now().isoformat(),
            }
        }

        await manager._analyze_turn(speech_event=speech_event_data, visual_events=[])
        await manager._update_summary()

        mock_summary_instance.generate.assert_called_once()
        call_args = mock_summary_instance.generate.call_args
        prompt = call_args[0][0]
        assert (
            "CURRENT SUMMARY:\n<summary>\nThe session has just begun.\n</summary>"
            in prompt
        )
        assert "NEW EVENTS THAT JUST OCCURRED:" in prompt
        assert "At t=15.50s: User navigated to the billing page." in prompt
        assert manager._session_summary == "User navigated to billing."
        assert len(manager._unsummarized_events) == 0
