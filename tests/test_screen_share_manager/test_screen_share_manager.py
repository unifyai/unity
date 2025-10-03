import asyncio
import base64
from datetime import datetime
import json
from unittest.mock import call, ANY

import pytest
from unity.image_manager.utils import make_solid_png_base64
from unity.screen_share_manager.types import TurnAnalysisResponse, KeyEvent
from tests.helpers import _handle_project

# A simple, valid base64 PNG for testing, now in the correct data URL format
PNG_BLUE_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (0, 0, 255))}"
PNG_RED_B64 = f"data:image/png;base64,{make_solid_png_base64(10, 10, (255, 0, 0))}"


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
    # LLM was called
    mocks["openai_client"].chat.completions.create.assert_called_once()

    # Image was registered
    mocks["image_manager"].add_images.assert_called_once()

    # TranscriptManager was called to log the message
    mocks["transcript_manager"].log_messages.assert_called_once()

    # Verify the structure of the logged message
    call_args = mocks["transcript_manager"].log_messages.call_args
    logged_message = call_args[0][0][0]  # log_messages([message])

    assert logged_message.content == "Okay, I will click this button now."
    assert "15.50-15.50" in logged_message.screen_share
    annotation = logged_message.screen_share["15.50-15.50"]
    assert annotation.caption == "User clicked the 'Submit' button."
    assert annotation.image_b64 == PNG_RED_B64

    # Corrected the span from [13:28] to [13:30]
    # "click this button" starts at index 13 and has a length of 17.
    assert "[13:30]" in logged_message.images
    assert logged_message.images["[13:30]"] == 42  # The mocked image_id


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_silent_vision_event_back_patches_on_timeout(mocked_screen_share_manager):
    """
    Tests that a silent visual event, after a period of inactivity,
    correctly back-patches the last known user utterance in the transcript.
    """
    manager, mocks = mocked_screen_share_manager

    # 1. Simulate a prior user utterance to establish a message_id to update
    manager._last_user_utterance_message_id = 123

    # 2. Set the inactivity timeout to a short value for the test
    manager.INACTIVITY_TIMEOUT_SEC = 0.5

    # 3. Simulate a silent visual event
    manager._pending_vision_events.append(
        {
            "timestamp": 25.0,
            "before_frame_b64": PNG_BLUE_B64,
            "after_frame_b64": PNG_RED_B64,
        }
    )
    manager._last_activity_time = (
        asyncio.get_event_loop().time() - 1.0
    )  # Set activity time in the past

    # 4. Mock the LLM response for a vision-only analysis
    mock_llm_response = TurnAnalysisResponse(
        events=[
            KeyEvent(
                timestamp=25.0,
                event_description="User navigated to the 'Profile' page.",
                screenshot_b64=PNG_RED_B64,
                triggering_phrase=None,  # No speech
            )
        ]
    )
    mocks["openai_client"].chat.completions.create.return_value = mock_llm_response

    # 5. Run the flush/timeout check
    await manager._flush_pending_events_on_timeout()

    # Allow the async analysis task to complete
    await asyncio.sleep(0.1)

    # 6. Assertions
    # LLM was called for analysis
    mocks["openai_client"].chat.completions.create.assert_called_once()

    # TranscriptManager's update method was called
    mocks["transcript_manager"].update_message_screen_share.assert_called_once()

    # Verify the arguments passed to the update method
    call_args = mocks["transcript_manager"].update_message_screen_share.call_args
    assert call_args.kwargs["message_id"] == 123

    new_event_data = call_args.kwargs["new_event"]
    assert "25.00-25.00" in new_event_data
    annotation = new_event_data["25.00-25.00"]
    assert annotation.caption == "User navigated to the 'Profile' page."

    # Ensure the pending events buffer is cleared
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

    # 1. Simulate a pending visual event that occurred before the speech
    manager._pending_vision_events.append(
        {
            "timestamp": 14.5,
            "before_frame_b64": PNG_BLUE_B64,
            "after_frame_b64": PNG_RED_B64,
        }
    )

    # 2. Mock the LLM response to return two distinct events
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

    # 3. Define the speech event
    speech_event_data = {
        "event_name": "PhoneUtterance",
        "payload": {
            "contact_details": {"contact_id": 1},
            "timestamp": datetime.now().isoformat(),
            "content": "I will click submit",
            "start_time": 15.0,
            "end_time": 16.0,
        },
    }

    # 4. Trigger analysis
    await manager._analyze_turn(speech_event=speech_event_data)

    # 5. Assertions
    mocks["transcript_manager"].log_messages.assert_called_once()

    logged_message = mocks["transcript_manager"].log_messages.call_args[0][0][0]

    # Check that both events are in screen_share, correctly timestamped
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

    # Check that the image link for the speech is correct
    # Corrected the span from [0:17] to [0:19]
    # "I will click submit" starts at index 0 and has a length of 19.
    assert "[0:19]" in logged_message.images
    assert logged_message.images["[0:19]"] == 42
