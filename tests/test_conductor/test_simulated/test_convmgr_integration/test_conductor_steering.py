from __future__ import annotations
import asyncio
from unittest.mock import patch, AsyncMock
import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


@pytest.mark.asyncio
@_handle_project
async def test_conductor_corrects_conversation_flow():
    """
    Verify that the Conductor can review a faulty conversation, use its
    own tools to find the correct information, and then steer the
    ConversationManager by interjecting the right answer.
    """
    cond = SimulatedConductor(
        description=(
            "The user is asking for the status of the 'Q3 budget report' task. "
            "However, the assistant has misunderstood and is incorrectly trying to create a new task."
        ),
        simulation_guidance=(
            "The simulated TaskScheduler knows about one task: "
            "name='Q3 budget report', status='in progress', due_date='next Friday'."
        ),
    )

    # A meta-prompt instructing the Conductor to supervise and correct the ongoing conversation.
    request_to_conductor = (
        "The current conversation is stuck. Please review the full transcript, "
        "identify the user's true intent, and provide them with the correct information."
    )

    handle = await cond.request(
        request_to_conductor,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # 1. Verify the Conductor's reasoning process:
    requested_tools = assistant_requested_tool_names(messages)
    executed_tools = tool_names_from_messages(messages)

    assert "ConversationManagerHandle_get_full_transcript" in requested_tools
    assert "TaskScheduler_ask" in requested_tools
    assert "ConversationManagerHandle_interject" in requested_tools

    assert "ConversationManagerHandle_get_full_transcript" in executed_tools
    assert "TaskScheduler_ask" in executed_tools
    assert "ConversationManagerHandle_interject" in executed_tools

    print(
        "\n✅ Test passed: Conductor correctly identified the faulty conversation and steered it.",
    )
    print(f"  - Requested Tools Sequence (The Conductor's 'plan'): {requested_tools}")
    print(f"  - Executed Tools: {executed_tools}")


@pytest.mark.asyncio
@_handle_project
async def test_conductor_steers_with_ask():
    """
    Verify that Conductor can use the ConversationManagerHandle
    to read the transcript and then ask a clarifying question.

    This test mocks both get_full_transcript and ask to ensure deterministic
    behavior and to avoid actual LLM calls during testing.
    """
    # 1. Create a real instance of the handle we want to mock methods on.
    from unity.conversation_manager_2.simulated import (
        SimulatedConversationManagerHandle,
    )

    mock_cm_handle = SimulatedConversationManagerHandle(
        assistant_id="simulated-assistant",
        contact_id="simulated-contact",
        description="A user needs help with a booking.",
    )

    # 2. Define the return values for the mocked tool calls.
    mock_transcript = {
        "status": "ok",
        "messages": [{"role": "user", "content": "I need help with my booking."}],
    }
    mock_user_response = "I need help with a flight booking."

    # 3. Create AsyncMocks and manually add the attributes required by the
    #    Conductor's tool registration logic.
    transcript_mock = AsyncMock(return_value=mock_transcript)
    transcript_mock.__name__ = "get_full_transcript"
    transcript_mock.__self__ = mock_cm_handle

    ask_mock = AsyncMock(return_value=mock_user_response)
    ask_mock.__name__ = "ask"
    ask_mock.__self__ = mock_cm_handle

    # 4. Patch both methods on the handle instance *before* passing it to the Conductor.
    with patch.object(
        mock_cm_handle,
        "get_full_transcript",
        transcript_mock,
    ), patch.object(mock_cm_handle, "ask", ask_mock):

        # 5. Now, create the Conductor, injecting the pre-patched handle.
        cond = SimulatedConductor(
            description="The user's booking request is ambiguous.",
            conversation_manager=mock_cm_handle,
        )

        request_to_conductor = (
            "The user's request is ambiguous. Please review the transcript "
            "and ask them to clarify what kind of booking they need help with."
        )

        handle = await cond.request(
            request_to_conductor,
            _return_reasoning_steps=True,
        )

        answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    # 6. Verify the final outcome and tool call sequence.
    assert "flight booking" in answer.lower()

    requested_tools = assistant_requested_tool_names(messages)
    assert all(
        tool in requested_tools
        for tool in [
            "ConversationManagerHandle_get_full_transcript",
            "ConversationManagerHandle_ask",
        ]
    )

    print("\n✅ Test passed: Conductor.ask correctly used the CM handle to clarify.")
    print(f"   - Requested Tools Sequence: {requested_tools}")
