# tests/test_simulated_conversation_manager.py
from __future__ import annotations

import asyncio
import pytest
import functools
import re
from pydantic import BaseModel, Field
from unity.conversation_manager_2.simulated import SimulatedConversationManagerHandle
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_cm():
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )
    # The 'ask' method on the main handle returns another handle
    answer_handle = await cm_handle.ask("Hello, who is this?")
    answer = await answer_handle.result()
    print(answer)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls should share the same conversation context
    via the stateful LLM.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )

    # 1) Ask for a unique favorite color
    h1 = await cm_handle.ask(
        "Please tell me your favorite color. Respond with only the color and nothing else.",
    )
    color = await h1.result()
    color = re.sub(r"\\W+", "", color.strip().lower())
    assert color, "Color should not be empty"

    # 2) Ask what color was suggested
    h2 = await cm_handle.ask("Great. What was the favorite color you just mentioned?")
    answer2 = (await h2.result()).lower()
    answer2 = re.sub(r"\\W+", "", answer2.strip().lower())

    assert color in answer2, "LLM should recall the previous favorite color"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Notification then ask – state propagated
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_stateful_notification_then_ask():
    """
    A .send_notification() call should influence a subsequent .ask() call.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )
    notification_content = "The user's meeting has been rescheduled to 4 PM."

    # 1) Tell the manager about a schedule change
    await cm_handle.send_notification(
        notification_content,
        source="test_system",
    )

    # 2) Ask about the meeting time – it should know about the update
    h_q = await cm_handle.ask("What time is my meeting now?")
    answer = (await h_q.result()).lower()

    assert (
        "4 pm" in answer or "4:00" in answer
    ), "Answer should reflect the notification"


# ────────────────────────────────────────────────────────────────────────────
# 4. Test description and simulation_guidance
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_description_and_guidance():
    """
    Tests that the description and simulation_guidance parameters correctly
    influence the LLM's behavior.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="pirate_assistant",
        contact_id="captain_contact",
        description="A conversation with a pirate captain who has lost his treasure.",
        simulation_guidance="You must always respond in pirate slang. Refer to the user as 'matey'.",
    )

    h1 = await cm_handle.ask("Who are you?")
    answer1 = await h1.result()
    assert (
        "pirate" in answer1.lower() or "captain" in answer1.lower()
    ), "Description should set the persona."

    h2 = await cm_handle.ask("What is the capital of France?")
    answer2 = await h2.result()
    assert "matey" in answer2.lower(), "Simulation guidance should be followed."


# ────────────────────────────────────────────────────────────────────────────
# 5. Test ask with structured output (Pydantic)
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_ask_with_structured_output():
    """
    Tests that the ask() method can correctly parse a response into a Pydantic model.
    """

    # Define the Pydantic model for the expected response
    class UserInfo(BaseModel):
        name: str = Field(description="The user's full name.")
        is_happy: bool = Field(description="Whether the user is currently happy.")
        age: int = Field(description="The user's age.")

    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
        simulation_guidance="The user's name is John Doe, he is 30 years old, and he is happy.",
    )

    # Ask the question and provide the response_format
    handle = await cm_handle.ask(
        "Please provide the user's information.",
        response_format=UserInfo,
    )

    # The result should be an instance of the Pydantic model
    result = await handle.result()

    assert isinstance(result, UserInfo), "Result should be an instance of UserInfo"
    assert result.name.lower() == "john doe"
    assert result.is_happy is True
    assert result.age == 30


# ────────────────────────────────────────────────────────────────────────────
#  Steerable handle tests
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 6.  Interject
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_cm(monkeypatch):
    """Verify that interject is called and influences the conversation state."""
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )

    # Interject with new information
    await cm_handle.interject("The user's flight has been delayed.")

    # Ask a question that should be influenced by the interjection
    h_q = await cm_handle.ask("What is the status of my flight?")
    answer = (await h_q.result()).lower()

    assert "delayed" in answer, "Answer should reflect the interjected information"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Stop
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_cm():
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )
    # The main handle's result will block until stop() is called
    result_task = asyncio.create_task(cm_handle.result())

    await asyncio.sleep(0.05)
    assert not result_task.done(), "Handle should not be done before stop()"

    cm_handle.stop(reason="Test cleanup")

    await asyncio.wait_for(result_task, timeout=1)
    assert cm_handle.done(), "Handle should report done after stop()"
    assert "stopped" in (await result_task).lower()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause → Resume round-trip
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_and_resume_simulated_cm(monkeypatch):
    """
    Verify that a SimulatedConversationManagerHandle may be paused and resumed.
    """
    counts = {"pause": 0, "resume": 0}

    original_pause = SimulatedConversationManagerHandle.pause

    @functools.wraps(original_pause)
    def wrapped_pause(self, *args, **kwargs):
        counts["pause"] += 1
        return original_pause(self, *args, **kwargs)

    monkeypatch.setattr(SimulatedConversationManagerHandle, "pause", wrapped_pause)

    original_resume = SimulatedConversationManagerHandle.resume

    @functools.wraps(original_resume)
    def wrapped_resume(self, *args, **kwargs):
        counts["resume"] += 1
        return original_resume(self, *args, **kwargs)

    monkeypatch.setattr(SimulatedConversationManagerHandle, "resume", wrapped_resume)

    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )

    pause_reply = cm_handle.pause()
    assert "paused" in pause_reply.lower()

    # In this simulation, pause is a state flag, so we just check the flag and resume
    assert cm_handle._paused is True, "Pause flag should be set"

    resume_reply = cm_handle.resume()
    assert "resumed" in resume_reply.lower()

    assert cm_handle._paused is False, "Pause flag should be unset"

    assert counts == {
        "pause": 1,
        "resume": 1,
    }, "pause/resume should each be called once"


# ────────────────────────────────────────────────────────────────────────────
# 9.  Nested ask on handle
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask_on_simulated_cm():
    """
    The `ask()` method on the handle should return its own handle, and state should
    be preserved across these interactions.
    """
    cm_handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id="test_contact",
    )

    # Start an initial ask to obtain a live handle
    h1 = await cm_handle.ask("Let's talk about the weather in London.")
    await h1.result()

    # Invoke a nested ask on the same handle
    h2 = await cm_handle.ask("What city did I just ask about?")
    nested_answer = await h2.result()
    assert (
        "london" in nested_answer.lower()
    ), "LLM should recall context from the first ask call"
