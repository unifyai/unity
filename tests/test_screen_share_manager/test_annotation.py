import base64
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import (
    ScreenShareManager,
    TurnState,
    KeyEvent,
)
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import (
    PNG_RED_B64,
    PNG_BLUE_B64,
    PNG_GREEN_B64,
)


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_should_noop_for_empty_event_list(mocked_manager):
    """
    WHY: A simple edge case to ensure the method handles empty inputs
    without calling the LLM or raising an error.
    """
    manager, mocks = mocked_manager
    result = await manager.annotate_events([], "some context")
    assert result == []
    mocks["annotate"].generate.assert_not_called()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_prompt_combines_all_contexts(mocked_manager):
    """
    WHY: The quality of an annotation depends on rich context. This test verifies
    that the prompt correctly includes the long-term session summary, the immediate
    consumer context, and recent key events.
    """
    manager, mocks = mocked_manager

    # 1. Set up the different layers of context
    manager.set_session_context("This is the long-term session summary.")
    consumer_context = "The user is trying to log in."
    manager._recent_key_events.append(
        KeyEvent(
            timestamp=0.5,
            image_annotation="A previous key event.",
            representative_timestamp=0.5,
        )
    )

    # 2. Prepare the event to be annotated
    handle = manager._image_manager.add_images(
        [{"data": PNG_RED_B64.split(",", 1)[1]}], return_handles=True
    )[0]
    detected_event = DetectedEvent(1.0, "test", handle)

    # 3. Annotate and capture the prompt
    await manager.annotate_events([detected_event], consumer_context)

    mocks["annotate"].set_system_message.assert_called_once()
    prompt = mocks["annotate"].set_system_message.call_args.args[0]

    # 4. Assert all context layers are present
    assert "This is the long-term session summary." in prompt
    assert "The user is trying to log in." in prompt
    assert "A previous key event." in prompt


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_successful_annotation_triggers_summary_update(mocked_manager):
    """
    WHY: Annotating events is a key activity that should update the session's
    narrative summary. This test confirms that a successful annotation
    triggers the summary update mechanism.
    """
    manager, mocks = mocked_manager
    handle = manager._image_manager.add_images(
        [{"data": PNG_RED_B64.split(",", 1)[1]}], return_handles=True
    )[0]
    detected_event = DetectedEvent(1.0, "test", handle)
    mocks["annotate"].generate.return_value = "A new thing happened."

    # Patch the trigger method to confirm it gets called
    with patch.object(manager, "_trigger_summary_update") as mock_trigger:
        await manager.annotate_events([detected_event], "context")

        # Assertions
        mock_trigger.assert_called_once()
        # Also verify the event was added to the list of events to be summarized
        assert len(manager._unsummarized_events) == 1
        assert (
            manager._unsummarized_events[0].image_annotation == "A new thing happened."
        )


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_handles_empty_llm_response_gracefully(mocked_manager):
    """
    WHY: To ensure robustness against non-crashing but invalid LLM outputs.
    If the LLM returns an empty string, that event should be skipped, not
    annotated with empty text.
    """
    manager, mocks = mocked_manager
    h1, h2 = manager._image_manager.add_images(
        [
            {"data": PNG_RED_B64.split(",", 1)[1]},
            {"data": PNG_BLUE_B64.split(",", 1)[1]},
        ],
        return_handles=True,
    )
    events = [DetectedEvent(1.0, "r1", h1), DetectedEvent(2.0, "r2", h2)]

    # First call succeeds, second returns an empty string
    mocks["annotate"].generate.side_effect = ["Valid annotation.", "  "]

    annotated_handles = await manager.annotate_events(events, "context")

    # The final list should only contain the successfully annotated handle
    assert len(annotated_handles) == 1
    assert annotated_handles[0].annotation == "Valid annotation."
    # The second handle's annotation should not have been set
    assert getattr(h2, "annotation", None) is None


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotations_are_added_to_recent_key_events_for_next_turn(mocked_manager):
    """
    WHY: To test the statefulness of context between annotation calls. The
    annotations from one turn should serve as `_recent_key_events` context for the next.
    """
    manager, mocks = mocked_manager
    h1, h2 = manager._image_manager.add_images(
        [
            {"data": PNG_RED_B64.split(",", 1)[1]},
            {"data": PNG_BLUE_B64.split(",", 1)[1]},
        ],
        return_handles=True,
    )
    event_turn_1 = DetectedEvent(1.0, "turn1", h1)
    event_turn_2 = DetectedEvent(5.0, "turn2", h2)

    # --- Turn 1 ---
    mocks["annotate"].generate.return_value = "Annotation from turn 1."
    await manager.annotate_events([event_turn_1], "Context for turn 1")

    # Assert state was updated after turn 1
    assert len(manager._recent_key_events) == 1
    assert manager._recent_key_events[0].image_annotation == "Annotation from turn 1."

    # --- Turn 2 ---
    mocks["annotate"].generate.return_value = "Annotation from turn 2."
    await manager.annotate_events([event_turn_2], "Context for turn 2")

    # Assert that the prompt for turn 2 contained context from turn 1
    prompt_turn_2 = mocks["annotate"].set_system_message.call_args.args[0]
    assert "Recent Key Events" in prompt_turn_2
    assert "Annotation from turn 1." in prompt_turn_2
