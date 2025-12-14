import asyncio

import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    contacts,
    capture_task_started,
    capture_task_action_response,
    send_conductor_clarification_request,
    send_incoming_sms,
)


@pytest.mark.asyncio
@_handle_project
async def test_start_task_readonly(test_redis_client, event_capture):
    """
    Test start_task_readonly: send an SMS that triggers the assistant to start a read-only task,
    and verify that a task started event is published with the correct format.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to start a read-only task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Verify the request has the correct format
    assert task_started.action_name == "start_task_readonly"
    assert len(task_started.query) > 0

    # Stop the task
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Actually, stop that task please",
    )
    await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_start_task(test_redis_client, event_capture):
    """
    Test start_task: send an SMS that triggers the assistant to start a task,
    and verify that a task started event is published with the correct format.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to start a task that modifies data
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
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
        test_redis_client,
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
async def test_task_ask(test_redis_client, event_capture):
    """
    Test asking a question about a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Ask about the task status
    await send_incoming_sms(
        test_redis_client,
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
        test_redis_client,
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
async def test_task_interject(test_redis_client, event_capture):
    """
    Test interjecting additional instructions into a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Interject with more information
    await send_incoming_sms(
        test_redis_client,
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
        test_redis_client,
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
async def test_task_stop(test_redis_client, event_capture):
    """
    Test stopping a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Stop the task
    await send_incoming_sms(
        test_redis_client,
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
async def test_task_pause(test_redis_client, event_capture):
    """
    Test pausing a running task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Pause the task
    await send_incoming_sms(
        test_redis_client,
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
        test_redis_client,
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
async def test_task_resume(test_redis_client, event_capture):
    """
    Test resuming a paused task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Resume the task (even without pausing first, for test simplicity)
    await send_incoming_sms(
        test_redis_client,
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
        test_redis_client,
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
async def test_task_done_check(test_redis_client, event_capture):
    """
    Test checking if a task is done.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Check if the task is done
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Is that task finished yet?",
    )

    # Status checks use the ask operation
    handle_response = await capture_task_action_response(
        event_capture,
        task_started.handle_id,
        "ask",
    )

    # Verify response
    assert handle_response.handle_id == task_started.handle_id

    # Stop the task
    await send_incoming_sms(
        test_redis_client,
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
async def test_task_answer_clarification(test_redis_client, event_capture):
    """
    Test answering a clarification request from a task.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Start a task
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "What contacts do I have in my contact manager?",
    )

    # Wait for task started
    task_started = await capture_task_started(
        event_capture,
        "start_task_readonly",
    )

    # Manually send a ConductorClarificationRequest
    # (simulating what the inner task would send if it needed clarification)
    call_id = "test_clarification_123"
    await send_conductor_clarification_request(
        test_redis_client,
        handle_id=task_started.handle_id,
        query="Should I include the assistant's name in the contact?",
        call_id=call_id,
    )

    # Give the CM time to process the clarification request
    await asyncio.sleep(1.0)

    # Answer the clarification via SMS
    await send_incoming_sms(
        test_redis_client,
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
