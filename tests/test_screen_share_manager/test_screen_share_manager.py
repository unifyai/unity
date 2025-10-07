# FILE: test_screen_share_manager.py

import asyncio
from datetime import datetime
import json

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
                screenshot_b64=PNG_RED_B64,
                triggering_phrase="click this button",
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # 2. Define the incoming speech event
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

    # 3. Trigger the analysis
    await manager._analyze_turn(speech_event=speech_event_data)

    # 4. Assertions
    mocks["openai_client"].chat.completions.create.assert_called_once()
    mocks["image_manager"].add_images.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_called_once()
    mocks["event_broker"].publish.assert_called_once()  # For real-time annotation

    # Verify the structure of the logged message
    call_args = mocks["transcript_manager"].log_messages.call_args
    logged_message = call_args[0][0][0]

    assert logged_message.content == "Okay, I will click this button now."
    assert "15.50-15.50" in logged_message.screen_share
    annotation = logged_message.screen_share["15.50-15.50"]
    assert annotation.caption == "User clicked the 'Submit' button."
    assert annotation.image_b64 == PNG_RED_B64

    # "click this button" is length 17, starts at index 13. End is 13+17=30.
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

    # Mock the LLM response for the silent event
    silent_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                event_description="User navigated to the 'Profile' page.",
                screenshot_b64=PNG_RED_B64,
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = silent_event_analysis

    # Flush the silent event, which should store it
    await manager._flush_pending_events_on_timeout()
    await asyncio.sleep(0.1)  # Allow async task to run

    # Assertions for the silent part
    mocks["openai_client"].chat.completions.create.assert_called_once()
    assert len(manager._stored_silent_key_events) == 1
    assert manager._stored_silent_key_events[0].timestamp == 25.0
    mocks["transcript_manager"].log_messages.assert_not_called()  # Not logged yet

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

    # Mock the LLM response for the speech event
    speech_event_analysis = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=30.0,
                event_description="User confirmed seeing their profile.",
                screenshot_b64=PNG_RED_B64,
                triggering_phrase="see my profile",
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = speech_event_analysis

    # Handle the utterance event
    await manager._handle_utterance_event(speech_event_data)
    await asyncio.sleep(0.1)

    # Assertions for the combined logging
    assert mocks["openai_client"].chat.completions.create.call_count == 2
    mocks["transcript_manager"].log_messages.assert_called_once()

    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    # Check that both the silent and the speech events are in the screen_share dict
    assert len(logged_message.screen_share) == 2
    assert "25.00-25.00" in logged_message.screen_share
    assert "30.00-30.00" in logged_message.screen_share
    assert (
        logged_message.screen_share["25.00-25.00"].caption
        == "User navigated to the 'Profile' page."
    )
    assert (
        logged_message.screen_share["30.00-30.00"].caption
        == "User confirmed seeing their profile."
    )

    # The stored silent events should now be cleared
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
                screenshot_b64=PNG_RED_B64,
                triggering_phrase=None,
            ),
            KeyEvent(
                timestamp=15.0,
                event_description="User stated their intention to submit.",
                screenshot_b64=PNG_RED_B64,
                triggering_phrase="I will click submit",
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "I will click submit",
            "start_time": 15.0,
            "end_time": 16.0,
        },
    }

    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    assert len(logged_message.screen_share) == 2
    assert "14.50-14.50" in logged_message.screen_share
    assert "15.00-15.00" in logged_message.screen_share
    assert (
        logged_message.screen_share["14.50-14.50"].caption
        == "A new dialog box appeared."
    )
    assert (
        logged_message.screen_share["15.00-15.00"].caption
        == "User stated their intention to submit."
    )

    # "I will click submit" is length 19, starts at index 0. End is 19.
    assert "[0:19]" in logged_message.images
    assert logged_message.images["[0:19]"] == 42

    # Check that real-time annotations were published for both events
    assert mocks["event_broker"].publish.call_count == 2


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_llm_failure_is_handled_gracefully(mocked_screen_share_manager):
    """
    Tests that if the OpenAI client call fails, the error is logged and
    a transcript message is still created, just without screen events.
    """
    manager, mocks = mocked_screen_share_manager
    mocks["openai_client"].chat.completions.create.side_effect = Exception("API Error")

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),  # Added for Message creation
        },
    }
    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["openai_client"].chat.completions.create.assert_called_once()

    # MODIFIED: Assert that a message IS logged, as per product logic.
    # The message will simply be empty of screen_share annotations.
    mocks["transcript_manager"].log_messages.assert_called_once()

    # Optional: Verify the logged message has no screen annotations
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
    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[],
    )

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "test",
            "timestamp": datetime.now().isoformat(),  # Added for Message creation
        },
    }
    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["openai_client"].chat.completions.create.assert_called_once()

    # MODIFIED: Assert that a message IS logged, as per product logic.
    mocks["transcript_manager"].log_messages.assert_called_once()

    # Verify the logged message has no screen annotations
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
    manager._pending_vision_events.append(
        {"timestamp": 1.0, "before_frame_b64": "b", "after_frame_b64": "a"},
    )
    assert len(manager._pending_vision_events) == 1

    # Mock LLM to return an empty response, the simplest case
    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[],
    )

    await manager._analyze_turn(
        speech_event={
            "payload": {
                "content": "go",
                "contact_details": {"contact_id": 1},
                "timestamp": datetime.now().isoformat(),
            },
        },
    )

    # The list should be cleared regardless of the LLM output
    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_event_without_prior_utterance_is_stored(
    mocked_screen_share_manager,
):
    """
    Tests that if a silent event occurs but there's no last_user_utterance_message_id,
    the event is stored for the next turn.
    """
    manager, mocks = mocked_screen_share_manager

    # Ensure no prior message ID exists
    manager._last_user_utterance_message_id = None

    # Simulate a silent event
    manager._pending_vision_events.append(
        {"timestamp": 25.0, "before_frame_b64": "b", "after_frame_b64": "a"},
    )

    # Mock LLM response
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(timestamp=25.0, event_description="Desc", screenshot_b64="b64"),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # Analyze as a silent turn (speech_event=None)
    await manager._analyze_turn(speech_event=None)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    # The key assertion: no attempt to log, but the event is stored
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
    Tests that if the LLM returns a triggering_phrase that doesn't exist in the
    speech content, it's handled gracefully without creating a broken image link.
    """
    manager, mocks = mocked_screen_share_manager

    # LLM hallucinates a phrase
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=15.5,
                event_description="User clicked.",
                screenshot_b64=PNG_RED_B64,
                triggering_phrase="a phrase that does not exist",
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "The actual spoken words.",
            "start_time": 15.0,
            "end_time": 16.5,
        },
    }

    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["transcript_manager"].log_messages.assert_called_once()
    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    # A screen_share entry should still be created
    assert "15.50-15.50" in logged_message.screen_share
    # But the `images` dictionary should be empty because the phrase was not found
    assert len(logged_message.images) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_realtime_annotation_is_published_for_each_key_event(
    mocked_screen_share_manager,
):
    """
    Tests that a real-time event is published for every key event
    identified by the LLM, verifying the E2E flow to the event broker.
    """
    manager, mocks = mocked_screen_share_manager

    # 1. Mock the LLM to return multiple distinct events
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=14.5,
                event_description="Event A: A modal appeared.",
                screenshot_b64=PNG_RED_B64,
            ),
            KeyEvent(
                timestamp=15.0,
                event_description="Event B: User expressed intent.",
                screenshot_b64=PNG_RED_B64,
            ),
            KeyEvent(
                timestamp=15.8,
                event_description="Event C: User clicked a button.",
                screenshot_b64=PNG_GREEN_B64,
            ),
        ],
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # 2. Define a simple speech event to trigger the analysis
    speech_event_data = {
        "payload": {
            "contact_details": {"contact_id": 1},
            "content": "dummy speech",
            "timestamp": datetime.now().isoformat(),
        },
    }

    # 3. Trigger analysis
    await manager._analyze_turn(speech_event=speech_event_data)

    # 4. Assertions
    # Check that publish was called exactly 3 times
    assert mocks["event_broker"].publish.call_count == 3

    # Check the content of each published message
    published_descriptions = []
    for call_item in mocks["event_broker"].publish.call_args_list:
        # call_item is a tuple of (args, kwargs)
        channel = call_item.args[0]
        payload_str = call_item.args[1]

        # Verify the correct channel is used
        assert channel == "app:comms:screen_annotation"

        # Verify the payload structure and content
        payload = json.loads(payload_str)
        assert payload["event_name"] == "ScreenAnnotationEvent"
        assert "event_description" in payload["payload"]
        published_descriptions.append(payload["payload"]["event_description"])

    # Verify that all event descriptions were published
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
    Tests that a rapid succession of visual events (a 'burst') is sampled
    down to the first, middle, and last frames to reduce payload size.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    # Simulate 5 visual events in quick succession (0.5s apart)
    manager._pending_vision_events = [
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

    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[],
    )
    await manager._analyze_turn(speech_event=None)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    call_args = mocks["openai_client"].chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]

    # Verify the sampling note was added
    assert any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )

    # Count how many image sections were actually sent
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 3

    # Verify the timestamps of the sampled frames are correct (first, middle, last)
    assert "t=10.00s" in image_sections[0]["text"]  # First
    assert "t=11.00s" in image_sections[1]["text"]  # Middle
    assert "t=12.00s" in image_sections[2]["text"]  # Last

    # Verify the correct 'after' images were sent
    after_images = [item for item in user_content if item.get("type") == "image_url"]
    assert after_images[1]["image_url"]["url"] == PNG_RED_B64
    assert after_images[3]["image_url"]["url"] == PNG_YELLOW_B64
    assert after_images[5]["image_url"]["url"] == PNG_MAGENTA_B64


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_slow_events_are_not_sampled(mocked_screen_share_manager):
    """
    Tests that if visual events are spaced further apart than the burst
    threshold, they are all sent for analysis and not sampled.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    # Simulate 4 visual events spaced 3 seconds apart
    manager._pending_vision_events = [
        {"timestamp": 10.0, "before_frame_b64": "b1", "after_frame_b64": PNG_RED_B64},
        {"timestamp": 13.0, "before_frame_b64": "b2", "after_frame_b64": PNG_GREEN_B64},
        {
            "timestamp": 16.0,
            "before_frame_b64": "b3",
            "after_frame_b64": PNG_YELLOW_B64,
        },
        {"timestamp": 19.0, "before_frame_b64": "b4", "after_frame_b64": PNG_CYAN_B64},
    ]

    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[],
    )
    await manager._analyze_turn(speech_event=None)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    call_args = mocks["openai_client"].chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]

    # Verify the sampling note was NOT added
    assert not any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )

    # Verify that all 4 events were sent
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 4

    # Verify the timestamps of all frames are present
    assert "t=10.00s" in image_sections[0]["text"]
    assert "t=13.00s" in image_sections[1]["text"]
    assert "t=16.00s" in image_sections[2]["text"]
    assert "t=19.00s" in image_sections[3]["text"]


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_mixed_bursts_and_single_events_are_handled_correctly(
    mocked_screen_share_manager,
):
    """
    Tests the logic handles a complex sequence containing both single, distinct
    events and a rapid burst, ensuring only the burst is sampled.
    """
    manager, mocks = mocked_screen_share_manager
    manager.VISUAL_EVENT_SAMPLING_THRESHOLD = 3
    manager.BURST_DETECTION_THRESHOLD_SEC = 2.0

    # A single event, a gap, a burst of 4, a gap, and a final single event
    manager._pending_vision_events = [
        # Single Event 1
        {"timestamp": 10.0, "before_frame_b64": "b1", "after_frame_b64": PNG_RED_B64},
        # Burst of 4 events
        {
            "timestamp": 13.0,
            "before_frame_b64": "b2",
            "after_frame_b64": PNG_GREEN_B64,
        },  # Start of burst
        {
            "timestamp": 13.5,
            "before_frame_b64": "b3",
            "after_frame_b64": PNG_YELLOW_B64,
        },
        {
            "timestamp": 14.0,
            "before_frame_b64": "b4",
            "after_frame_b64": PNG_CYAN_B64,
        },  # Middle of burst
        {
            "timestamp": 14.5,
            "before_frame_b64": "b5",
            "after_frame_b64": PNG_MAGENTA_B64,
        },  # End of burst
        # Single Event 2
        {"timestamp": 18.0, "before_frame_b64": "b6", "after_frame_b64": PNG_WHITE_B64},
    ]

    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[],
    )
    await manager._analyze_turn(speech_event=None)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    call_args = mocks["openai_client"].chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]

    # Verify the sampling note WAS added for the burst
    assert any(
        "NOTE: The following frames are a sampled summary" in item.get("text", "")
        for item in user_content
    )

    # Verify the total number of frames sent
    # Expecting: Event 1 (1) + Sampled Burst (3) + Event 2 (1) = 5 total frames
    image_sections = [
        item for item in user_content if "Visual Change" in item.get("text", "")
    ]
    assert len(image_sections) == 5

    # Verify the correct timestamps are present
    assert "t=10.00s" in image_sections[0]["text"]  # Single event 1
    assert "t=13.00s" in image_sections[1]["text"]  # Burst start
    assert (
        "t=14.00s" in image_sections[2]["text"]
    )  # Burst middle (correct: index 2 of 4)
    assert "t=14.50s" in image_sections[3]["text"]  # Burst end
    assert "t=18.00s" in image_sections[4]["text"]  # Single event 2
