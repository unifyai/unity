import asyncio
import base64
from datetime import datetime
import json
from unittest.mock import call, ANY, AsyncMock

import pytest
from unity.image_manager.utils import make_solid_png_base64
from unity.screen_share_manager.types import TurnAnalysisResponse, KeyEvent
from tests.helpers import _handle_project

# A simple, valid base64 PNG for testing, now in the correct data URL format
PNG_BLUE_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 0, 255))}"
PNG_RED_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 0, 0))}"
PNG_GREEN_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 255, 0))}"


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
        {"payload": {"timestamp": 10.0, "frame_b64": PNG_RED_B64}}
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
        {"payload": {"timestamp": 10.0, "frame_b64": PNG_BLUE_B64}}
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
        {"payload": {"timestamp": 1.0, "frame_b64": PNG_BLUE_B64}}
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
            )
        ]
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
async def test_silent_vision_event_back_patches_on_timeout(mocked_screen_share_manager):
    """
    Tests that a silent visual event, after a period of inactivity,
    correctly back-patches the last known user utterance in the transcript.
    """
    manager, mocks = mocked_screen_share_manager

    manager._last_user_utterance_message_id = 123
    manager.INACTIVITY_TIMEOUT_SEC = 0.5
    manager._pending_vision_events.append(
        {
            "timestamp": 25.0,
            "before_frame_b64": PNG_BLUE_B64,
            "after_frame_b64": PNG_RED_B64,
        }
    )
    manager._last_activity_time = asyncio.get_event_loop().time() - 1.0

    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                event_description="User navigated to the 'Profile' page.",
                screenshot_b64=PNG_RED_B64,
                triggering_phrase=None,
            )
        ]
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    await manager._flush_pending_events_on_timeout()
    await asyncio.sleep(0.1)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    mocks["transcript_manager"].update_message_screen_share.assert_called_once()

    call_args = mocks["transcript_manager"].update_message_screen_share.call_args
    assert call_args.kwargs["message_id"] == 123
    new_event_data = call_args.kwargs["new_event"]
    assert "25.00-25.00" in new_event_data
    assert (
        new_event_data["25.00-25.00"].caption == "User navigated to the 'Profile' page."
    )
    assert len(manager._pending_vision_events) == 0


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
        }
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
        ]
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
    no transcript messages are created.
    """
    manager, mocks = mocked_screen_share_manager
    mocks["openai_client"].chat.completions.create.side_effect = Exception("API Error")

    speech_event_data = {"payload": {"content": "test"}}
    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_not_called()
    mocks["transcript_manager"].update_message_screen_share.assert_not_called()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_empty_llm_response_does_nothing(mocked_screen_share_manager):
    """
    Tests that if the LLM returns no events, the system does not attempt
    to log or back-patch anything.
    """
    manager, mocks = mocked_screen_share_manager
    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[]
    )

    speech_event_data = {"payload": {"content": "test"}}
    await manager._analyze_turn(speech_event=speech_event_data)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    mocks["transcript_manager"].log_messages.assert_not_called()
    mocks["transcript_manager"].update_message_screen_share.assert_not_called()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_analysis_clears_pending_vision_events(mocked_screen_share_manager):
    """
    Ensures that after any analysis run, the list of pending vision events is cleared.
    """
    manager, mocks = mocked_screen_share_manager
    manager._pending_vision_events.append(
        {"timestamp": 1.0, "before_frame_b64": "b", "after_frame_b64": "a"}
    )
    assert len(manager._pending_vision_events) == 1

    # Mock LLM to return an empty response, the simplest case
    mocks["openai_client"].chat.completions.create.return_value = TurnAnalysisResponse(
        events=[]
    )

    await manager._analyze_turn(speech_event={"payload": {"content": "go"}})

    # The list should be cleared regardless of the LLM output
    assert len(manager._pending_vision_events) == 0


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_event_without_prior_utterance_is_dropped(
    mocked_screen_share_manager,
):
    """
    Tests that if a silent event occurs but there's no last_user_utterance_message_id,
    the event is not back-patched and a warning is logged.
    """
    manager, mocks = mocked_screen_share_manager

    # Ensure no prior message ID exists
    manager._last_user_utterance_message_id = None

    # Simulate a silent event
    manager._pending_vision_events.append(
        {"timestamp": 25.0, "before_frame_b64": "b", "after_frame_b64": "a"}
    )

    # Mock LLM response
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(timestamp=25.0, event_description="Desc", screenshot_b64="b64")
        ]
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # Analyze as a silent turn (speech_event=None)
    await manager._analyze_turn(speech_event=None)

    mocks["openai_client"].chat.completions.create.assert_called_once()
    # The key assertion: no attempt should be made to update a non-existent message
    mocks["transcript_manager"].update_message_screen_share.assert_not_called()


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
            )
        ]
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
        ]
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # 2. Define a simple speech event to trigger the analysis
    speech_event_data = {"payload": {"content": "dummy speech"}}

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
