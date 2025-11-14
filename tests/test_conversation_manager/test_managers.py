# import asyncio

# import pytest
# from tests.helpers import _handle_project
# from tests.test_conversation_manager.helpers import (
#     capture_conductor_clarification_response,
#     capture_conductor_handle_request,
#     capture_conductor_request,
#     send_conductor_clarification_request,
#     send_conductor_response,
#     send_incoming_sms,
# )


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_ask(test_redis_client, event_capture):
#     """
#     Test conductor_ask: send an SMS that triggers the assistant to use conductor_ask,
#     and verify that a ConductorRequest is published with the correct format.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Send an SMS that prompts the assistant to use conductor_ask
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         "Use the conductor to find out what contacts I have in my contact manager",
#     )

#     # Wait for ConductorRequest
#     conductor_request = await capture_conductor_request(event_capture, "ask")

#     # Verify the request has the correct format
#     assert conductor_request.action_name == "ask"
#     assert len(conductor_request.query) > 0
#     assert conductor_request.parent_chat_context is not None


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_request(test_redis_client, event_capture):
#     """
#     Test conductor_request: send an SMS that triggers the assistant to use conductor_request,
#     and verify that a ConductorRequest is published with the correct format.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Send an SMS that prompts the assistant to use conductor_request
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         "Use the conductor to create a new task to buy groceries tomorrow",
#     )

#     # Wait for ConductorRequest
#     conductor_request = await capture_conductor_request(event_capture, "request")

#     # Verify the request has the correct format
#     assert conductor_request.action_name == "request"
#     assert len(conductor_request.query) > 0
#     assert conductor_request.parent_chat_context is not None


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_ask(test_redis_client, event_capture):
#     """
#     Test conductor_handle_ask: manually create a handle_id, then send a message
#     to ask about the status and verify ConductorHandleRequest is generated.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 42  # Assumed handle_id
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Research AI papers",
#     )

#     # Give CM time to process
#     await asyncio.sleep(0.5)

#     # Ask about the handle status
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"Check on the status of conductor handle {handle_id}",
#     )

#     # Wait for ConductorHandleRequest with action_name="ask"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "ask",
#     )

#     # Verify request has correct format
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "ask"
#     assert len(handle_request.query) > 0
#     assert handle_request.parent_chat_context is not None


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_interject(test_redis_client, event_capture):
#     """
#     Test conductor_handle_interject: manually create a handle, then interject
#     and verify ConductorHandleRequest is generated.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 43
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Search for Italian restaurants",
#     )

#     await asyncio.sleep(0.5)

#     # Interject with more information
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"For conductor handle {handle_id}, add that I prefer vegetarian options",
#     )

#     # Wait for ConductorHandleRequest with action_name="interject"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "interject",
#     )

#     # Verify request
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "interject"


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_stop(test_redis_client, event_capture):
#     """
#     Test conductor_handle_stop: manually create a handle, then stop it.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 44
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Create a long research report on quantum computing",
#     )

#     await asyncio.sleep(0.5)

#     # Stop the handle
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"Stop conductor handle {handle_id}",
#     )

#     # Wait for ConductorHandleRequest with action_name="stop"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "stop",
#     )

#     # Verify request
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "stop"


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_pause(test_redis_client, event_capture):
#     """
#     Test conductor_handle_pause: manually create a handle, then pause it.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 45
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Analyze customer feedback data",
#     )

#     await asyncio.sleep(0.5)

#     # Pause the handle
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"Pause conductor handle {handle_id}",
#     )

#     # Wait for ConductorHandleRequest with action_name="pause"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "pause",
#     )

#     # Verify request
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "pause"


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_resume(test_redis_client, event_capture):
#     """
#     Test conductor_handle_resume: manually create a handle, pause it, then resume it.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 46
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Compile a weekly report",
#     )

#     await asyncio.sleep(0.5)

#     # Resume the handle (we don't need to pause first for this test)
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"Resume conductor handle {handle_id}",
#     )

#     # Wait for ConductorHandleRequest with action_name="resume"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "resume",
#     )

#     # Verify request
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "resume"


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_done(test_redis_client, event_capture):
#     """
#     Test conductor_handle_done: manually create a handle, then check if it's done.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 47
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Get the current time",
#     )

#     await asyncio.sleep(0.5)

#     # Check if the handle is done
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"Check if conductor handle {handle_id} is done",
#     )

#     # Wait for ConductorHandleRequest with action_name="done"
#     handle_request = await capture_conductor_handle_request(
#         event_capture,
#         handle_id,
#         "done",
#     )

#     # Verify request
#     assert handle_request.handle_id == handle_id
#     assert handle_request.action_name == "done"


# @pytest.mark.asyncio
# @_handle_project
# async def test_conductor_handle_answer_clarification(test_redis_client, event_capture):
#     """
#     Test conductor_handle_answer_clarification: manually send a ConductorClarificationRequest,
#     then send an SMS to answer and verify ConductorClarificationResponse is generated.
#     """
#     # Clear any events from initialization
#     event_capture.clear()

#     # Manually send a ConductorResponse to simulate having a handle
#     handle_id = 48
#     await send_conductor_response(
#         test_redis_client,
#         handle_id=handle_id,
#         action_name="request",
#         query="Plan my vacation",
#     )

#     # Manually send a ConductorClarificationRequest
#     # (simulating what the conductor would send if it needed clarification)
#     call_id = "test_clarification_123"
#     await send_conductor_clarification_request(
#         test_redis_client,
#         handle_id=handle_id,
#         query="What is your budget for this vacation?",
#         call_id=call_id,
#     )

#     # Give the CM time to process the clarification request
#     await asyncio.sleep(1.0)

#     # Answer the clarification via SMS
#     contact_number = "+15555551111"
#     await send_incoming_sms(
#         test_redis_client,
#         contact_number,
#         f"For the clarification on handle {handle_id}, my budget is $5000",
#     )

#     # Step 4: Wait for ConductorClarificationResponse
#     clarification_response = await capture_conductor_clarification_response(
#         event_capture,
#         handle_id=handle_id,
#         call_id=call_id,
#     )

#     # Verify response has correct format
#     assert clarification_response.handle_id == handle_id
#     assert clarification_response.call_id == call_id
#     assert len(clarification_response.response) > 0
