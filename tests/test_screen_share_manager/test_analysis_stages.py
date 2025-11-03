import base64

import pytest

from unity.screen_share_manager.screen_share_manager import ScreenShareManager, TurnState
from unity.screen_share_manager.types import DetectedEvent
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import PNG_RED_B64, PNG_GREEN_B64


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_detection_should_handle_empty_turn():
    """Tests that detection with no events enqueues an empty list without error."""
    manager = ScreenShareManager()
    await manager.start()
    await manager._detect_key_moments(TurnState(speech_events=[], visual_events=[], latest_frame=None))
    result = await manager._detection_queue.get()
    assert result == []
    await manager.stop()


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotate_events_should_build_context_sequentially(mocked_manager):
    """Tests that annotating multiple events passes the annotation of the first as context to the second."""
    manager, mocks = mocked_manager
    red_b64, green_b64 = PNG_RED_B64.split(",", 1)[1], PNG_GREEN_B64.split(",", 1)[1]
    handles = manager._image_manager.add_images(
        [{"data": red_b64}, {"data": green_b64}], synchronous=True, return_handles=True
    )
    detected_events = [DetectedEvent(1.0, "reason1", handles[0]), DetectedEvent(2.0, "reason2", handles[1])]

    mocks["annotate"].generate.side_effect = ["Annotation for event 1", "Annotation for event 2"]
    await manager.annotate_events(detected_events, "initial context")

    assert mocks["annotate"].generate.call_count == 2
    second_call_prompt = mocks["annotate"].set_system_message.call_args_list[1].args[0]
    assert '"Annotation for event 1"' in second_call_prompt
    assert handles[0].annotation == "Annotation for event 1"
    assert handles[1].annotation == "Annotation for event 2"


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_annotation_should_noop_for_empty_event_list(mocked_manager):
    """Tests that annotate_events returns quickly and does not call the LLM if no events are provided."""
    manager, mocks = mocked_manager
    result = await manager.annotate_events([], "some context")
    assert result == []
    mocks["annotate"].generate.assert_not_called()