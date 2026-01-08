import asyncio

import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    contacts,
    capture_outgoing_sms,
    capture_task_started,
    capture_task_action_response,
    send_actor_clarification_request,
    send_incoming_sms,
)


@pytest.mark.asyncio
@_handle_project
async def test_start_task(event_broker, event_capture):
    """
    Test start_task: send an SMS that triggers the assistant to start a task,
    and verify that a task started event is published with the correct format.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to start a task that modifies data
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Create a new task to buy groceries tomorrow",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Verify the request has the correct format
    assert task_started.action_name == "start_task"
    assert len(task_started.query) > 0

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Never mind, please stop that task",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_ask(event_broker, event_capture):
    """
    Test asking a question about a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Ask about the task status
    await send_incoming_sms(
        event_broker,
        contact,
        "What's the status of that task you're working on?",
    )

    # Wait for ask action response
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "ask",
    )

    # Verify request has correct format
    assert handle_response.handle_id == task_started.handle_id
    assert len(handle_response.query) > 0

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_interject(event_broker, event_capture):
    """
    Test interjecting additional instructions into a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Interject with more information
    await send_incoming_sms(
        event_broker,
        contact,
        "Actually, for that task, please exclude my own contact from the list",
    )

    # Wait for interject action response
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "interject",
    )

    # Verify response
    assert handle_response.handle_id == task_started.handle_id

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_stop(event_broker, event_capture):
    """
    Test stopping a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task, I don't need it anymore",
    )

    # Wait for stop action response
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )

    # Verify request
    assert handle_response.handle_id == task_started.handle_id


@pytest.mark.asyncio
@_handle_project
async def test_task_completion_notification(event_broker, event_capture):
    """
    Test that task completion triggers an LLM response to notify the user.

    When a task completes (handle.result() returns), an ActorResult event
    is published which triggers cm.run_llm(), allowing the assistant to
    inform the user that the task is done.

    With steps=3, the task completes after 3 interactions:
    1. Watcher's result() call (1 step)
    2. First progress query (1 step)
    3. Second progress query (1 step) -> task completes
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with explicit request to notify on completion
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "List all my contacts and let me know once you're done",
    )

    # Wait for task to start (watcher's result() consumes 1 step)
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Send progress queries to consume remaining steps
    # Each ask operation consumes 1 step
    await send_incoming_sms(
        event_broker,
        contact,
        "How's that task going?",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "ask",
    )

    await send_incoming_sms(
        event_broker,
        contact,
        "Any progress yet?",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "ask",
    )

    # After 3 steps (result + 2 asks), task auto-completes
    # ActorResult triggers cm.run_llm() - LLM should notify the user
    await capture_outgoing_sms(event_capture, contact)


@pytest.mark.asyncio
@_handle_project
async def test_task_pause(event_broker, event_capture):
    """
    Test pausing a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Pause the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Pause that task for now",
    )

    # Wait for pause action response
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "pause",
    )

    # Verify response
    assert handle_response.handle_id == task_started.handle_id

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Actually just stop that task",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_resume(event_broker, event_capture):
    """
    Test resuming a paused task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Resume the task (even without pausing first, for test simplicity)
    await send_incoming_sms(
        event_broker,
        contact,
        "Resume that task please",
    )

    # Wait for resume action response
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "resume",
    )

    # Verify response
    assert handle_response.handle_id == task_started.handle_id

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_progress_query(event_broker, event_capture):
    """
    Test querying specific progress details from a running task.

    This verifies that questions requiring task-internal knowledge (e.g.,
    intermediate results, current step) route through the ask operation
    on the in-flight handle rather than being answered directly.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task that will have meaningful progress to query
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Ask a question that requires querying the in-flight task for details
    # This cannot be answered without asking the task itself
    await send_incoming_sms(
        event_broker,
        contact,
        "How many contacts have you found so far in that task?",
    )

    # Progress queries use the ask operation on the in-flight handle
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "ask",
    )

    # Verify response
    assert handle_response.handle_id == task_started.handle_id

    # Stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_task_answer_clarification(event_broker, event_capture):
    """
    Test answering a clarification request from a task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task with a clear, direct command
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me all my contacts with their names and phone numbers.",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Manually send an ActorClarificationRequest
    # (simulating what the inner task would send if it needed clarification)
    call_id = "test_clarification_123"
    await send_actor_clarification_request(
        event_broker,
        handle_id=task_started.handle_id,
        query="Should I include the assistant's name in the contact?",
        call_id=call_id,
    )

    # Give the CM time to process the clarification request
    await asyncio.sleep(1.0)

    # Answer the clarification via SMS
    await send_incoming_sms(
        event_broker,
        contact,
        "Yes, include the assistant's name in the contact",
    )

    # Wait for clarification answer response
    clarification_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "answer_clarification",
        call_id=call_id,
    )

    # Verify response has correct format
    assert clarification_response.handle_id == task_started.handle_id
    assert len(clarification_response.response) > 0


@pytest.mark.asyncio
@_handle_project
async def test_llm_asks_clarification_for_ambiguous_request(
    event_broker,
    event_capture,
):
    """
    Test that the LLM asks for clarification when given an ambiguous request.

    When a user's request is vague or could be interpreted multiple ways,
    the LLM should ask for clarification rather than immediately starting a task.
    """
    from unity.conversation_manager.events import SMSSent, ActorHandleStarted

    # Clear any events from initialization
    event_capture.clear()

    # Send an ambiguous request that could be interpreted multiple ways
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "I need help with a contact",  # Ambiguous: which contact? what kind of help?
    )

    # Wait for an SMS response (the LLM should ask for clarification)
    print("⏳ Waiting for clarification SMS (timeout: 300s)...")
    response = await event_capture.wait_for_event(SMSSent, timeout=300.0)

    # Verify the LLM asked a clarifying question
    content_lower = response.content.lower()
    clarification_indicators = ["which", "what", "how", "would you like", "prefer", "?"]
    has_clarification = any(
        indicator in content_lower for indicator in clarification_indicators
    )
    assert (
        has_clarification
    ), f"Expected LLM to ask a clarifying question, but got: {response.content}"

    print(f"✅ LLM asked for clarification: {response.content[:100]}...")

    # Verify no task was started (LLM should wait for clarification)
    task_events = event_capture.get_events(ActorHandleStarted)
    assert (
        len(task_events) == 0
    ), f"Expected no task to be started before clarification, but found {len(task_events)} task(s)"


@pytest.mark.asyncio
@_handle_project
async def test_llm_uses_clarification_response_in_task(
    event_broker,
    event_capture,
):
    """
    Test that the LLM uses the clarification response when starting a task.

    When the user provides clarification after an ambiguous request,
    the LLM should incorporate that clarification into the task it starts.
    """
    from unity.conversation_manager.events import SMSSent

    # Clear any events from initialization
    event_capture.clear()

    # Send an ambiguous request
    contact = contacts[1]
    await send_incoming_sms(
        event_broker,
        contact,
        "I need help with a contact",  # Ambiguous: which contact? what kind of help?
    )

    # Wait for the LLM to ask for clarification
    print("⏳ Waiting for clarification SMS (timeout: 300s)...")
    clarification_sms = await event_capture.wait_for_event(SMSSent, timeout=300.0)
    print(f"✅ LLM asked: {clarification_sms.content[:100]}...")

    # Provide clarification - be specific about what we want
    await send_incoming_sms(
        event_broker,
        contact,
        "Show me only contacts that have email addresses. Include their names and emails.",
    )

    # Wait for task to be started with the clarified request
    task_started = await capture_task_started(
        event_capture,
        "start_task",
    )

    # Verify the task query incorporates the clarification (email-related)
    query_lower = task_started.query.lower()
    assert (
        "email" in query_lower
    ), f"Expected task query to mention 'email' from clarification, but got: {task_started.query}"

    print(f"✅ Task started with clarified query: {task_started.query[:100]}...")

    # Clean up - stop the task
    await send_incoming_sms(
        event_broker,
        contact,
        "Stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )
