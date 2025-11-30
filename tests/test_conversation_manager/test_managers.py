import asyncio

import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from tests.test_conversation_manager.helpers import (
    contacts,
    capture_conductor_handle_response,
    capture_conductor_handle_started,
    send_conductor_clarification_request,
    send_incoming_sms,
)


@pytest.mark.asyncio
@_handle_project
async def test_conductor_ask(test_redis_client, event_capture):
    """
    Test conductor_ask: send an SMS that triggers the assistant to use conductor_ask,
    and verify that a ConductorRequest is published with the correct format.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Verify the request has the correct format
    assert conductor_handle_started.action_name == "conductor_ask"
    assert len(conductor_handle_started.query) > 0

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_request(test_redis_client, event_capture):
    """
    Test conductor_request: send an SMS that triggers the assistant to use conductor_request,
    and verify that a ConductorRequest is published with the correct format.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_request
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Use the conductor to create a new task to buy groceries tomorrow",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_request",
    )

    # Verify the request has the correct format
    assert conductor_handle_started.action_name == "conductor_request"
    assert len(conductor_handle_started.query) > 0

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_ask(test_redis_client, event_capture):
    """
    Test conductor_handle_ask: manually create a handle_id, then send a message
    to ask about the status and verify ConductorHandleRequest is generated.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Ask about the handle status
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Check on the status of conductor handle {conductor_handle_started.handle_id}",
    )

    # Wait for ConductorHandleRequest with action_name="ask"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_ask",
    )

    # Verify request has correct format
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_ask"
    assert len(handle_response.query) > 0

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_interject(test_redis_client, event_capture):
    """
    Test conductor_handle_interject: manually create a handle, then interject
    and verify ConductorHandleRequest is generated.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Interject with more information
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"For conductor handle {conductor_handle_started.handle_id}, add that I don't want my contact to be listed",
    )

    # Wait for ConductorHandleResponse with action_name="interject"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_interject",
    )

    # Verify response
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_interject"

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_stop(test_redis_client, event_capture):
    """
    Test conductor_handle_stop: manually create a handle, then stop it.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )

    # Wait for ConductorHandleResponse with action_name="stop"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )

    # Verify request
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_stop"


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_pause(test_redis_client, event_capture):
    """
    Test conductor_handle_pause: manually create a handle, then pause it.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Pause the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Pause conductor handle {conductor_handle_started.handle_id}",
    )

    # Wait for ConductorHandleResponse with action_name="pause"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_pause",
    )

    # Verify response
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_pause"

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_resume(test_redis_client, event_capture):
    """
    Test conductor_handle_resume: manually create a handle, pause it, then resume it.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Send an SMS that prompts the assistant to use conductor_ask
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Resume the handle (we don't need to pause first for this test)
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Resume conductor handle {conductor_handle_started.handle_id}",
    )

    # Wait for ConductorHandleResponse with action_name="resume"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_resume",
    )

    # Verify response
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_resume"

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_done(test_redis_client, event_capture):
    """
    Test conductor_handle_done: manually create a handle, then check if it's done.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Manually send a ConductorResponse to simulate having a handle
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Check if the handle is done
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Check if conductor handle {conductor_handle_started.handle_id} is done",
    )

    # Wait for ConductorHandleResponse with action_name="done"
    handle_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_done",
    )

    # Verify response
    assert handle_response.handle_id == conductor_handle_started.handle_id
    assert handle_response.action_name == "conductor_handle_done"

    # Stop the handle
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"Stop conductor handle {conductor_handle_started.handle_id}",
    )
    await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_stop",
    )


@pytest.mark.asyncio
@_handle_project
async def test_conductor_handle_answer_clarification(test_redis_client, event_capture):
    """
    Test conductor_handle_answer_clarification: manually send a ConductorClarificationRequest,
    then send an SMS to answer and verify ConductorClarificationResponse is generated.
    """
    # Clear any events from initialization
    event_capture.clear()

    # Manually send a ConductorResponse to simulate having a handle
    contact = contacts[1]
    await send_incoming_sms(
        test_redis_client,
        contact,
        "Ask the conductor what contacts I have in my contact manager",
    )

    # Wait for ConductorHandleStarted
    conductor_handle_started = await capture_conductor_handle_started(
        event_capture,
        "conductor_ask",
    )

    # Manually send a ConductorClarificationRequest
    # (simulating what the conductor would send if it needed clarification)
    call_id = "test_clarification_123"
    await send_conductor_clarification_request(
        test_redis_client,
        handle_id=conductor_handle_started.handle_id,
        query="Should I include the assistant's name in the contact?",
        call_id=call_id,
    )

    # Give the CM time to process the clarification request
    await asyncio.sleep(1.0)

    # Answer the clarification via SMS
    await send_incoming_sms(
        test_redis_client,
        contact,
        f"For the clarification on handle {conductor_handle_started.handle_id}, yes, include the assistant's name in the contact",
    )

    # Step 4: Wait for ConductorClarificationResponse
    clarification_response = await capture_conductor_handle_response(
        event_capture,
        conductor_handle_started.handle_id,
        "conductor_handle_answer_clarification",
        call_id=call_id,
    )

    # Verify response has correct format
    assert clarification_response.handle_id == conductor_handle_started.handle_id
    assert clarification_response.call_id == call_id
    assert len(clarification_response.response) > 0
