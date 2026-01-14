"""
tests/test_conversation_manager/test_managers.py
=================================================

Tests for task-related conversation flows through the ConversationManager.

Uses the ConversationManager "single-step" API:
- Construct an input Event (typically SMSReceived)
- Call `await cm._step(event)`
- Assert on state changes and output events

These tests verify that the LLM processes various request types correctly.
The focus is on LLM behavior and immediate responses, not internal task
machinery (which varies between real and simulated implementations).
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
)

pytestmark = pytest.mark.eval


def _only(events, typ):
    """Filter events by type."""
    return [e for e in events if isinstance(e, typ)]


def _get_sms_response(events):
    """Get the first SMS response, if any."""
    sms_list = _only(events, SMSSent)
    return sms_list[0] if sms_list else None


# =============================================================================
# LLM Processing Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_task_request_processed(initialized_cm):
    """
    Test that task creation requests are processed by the LLM.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # LLM should have processed the request
    assert result.llm_ran, "Expected LLM to process task request"


@pytest.mark.asyncio
@_handle_project
async def test_stop_request_processed(initialized_cm):
    """
    Test that stop requests are processed by the LLM.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a task first
    await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # Then send stop request
    stop_result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Stop that task, I don't need it anymore",
        ),
    )

    assert stop_result.llm_ran, "Expected LLM to process stop request"


@pytest.mark.asyncio
@_handle_project
async def test_status_query_processed(initialized_cm):
    """
    Test that status queries are processed by the LLM.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a task
    await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # Ask about status
    status_result = await cm._step(
        SMSReceived(
            contact=contact,
            content="What's the status of that task you're working on?",
        ),
    )

    assert status_result.llm_ran, "Expected LLM to process status query"


@pytest.mark.asyncio
@_handle_project
async def test_modification_request_processed(initialized_cm):
    """
    Test that task modification requests are processed.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a task
    await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # Modify the task
    modify_result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Actually, for that task, please exclude my own contact from the list",
        ),
    )

    assert modify_result.llm_ran, "Expected LLM to process modification request"


@pytest.mark.asyncio
@_handle_project
async def test_pause_request_processed(initialized_cm):
    """
    Test that pause requests are processed by the LLM.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a task
    await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # Pause request
    pause_result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Pause that task for now",
        ),
    )

    assert pause_result.llm_ran, "Expected LLM to process pause request"


@pytest.mark.asyncio
@_handle_project
async def test_resume_request_processed(initialized_cm):
    """
    Test that resume requests are processed by the LLM.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Start a task
    await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me all my contacts with their names and phone numbers.",
        ),
    )

    # Resume request
    resume_result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Resume that task please",
        ),
    )

    assert resume_result.llm_ran, "Expected LLM to process resume request"


# =============================================================================
# Clarification Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_llm_asks_clarification_for_ambiguous_request(initialized_cm):
    """
    Test that the LLM asks for clarification when given an ambiguous request.

    When a user's request is vague, the LLM should ask for clarification
    rather than making assumptions. This produces an immediate SMS response.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Send an ambiguous request
    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="I need help with a contact",  # Ambiguous: which? what help?
        ),
    )

    # Should produce an SMS response asking for clarification
    sms = _get_sms_response(result.output_events)
    assert sms is not None, "Expected SMS response"

    content_lower = sms.content.lower()

    # Verify clarification indicators
    clarification_indicators = ["which", "what", "how", "would you like", "prefer", "?"]
    has_clarification = any(
        indicator in content_lower for indicator in clarification_indicators
    )
    assert (
        has_clarification
    ), f"Expected LLM to ask a clarifying question, but got: {sms.content}"


@pytest.mark.asyncio
@_handle_project
async def test_llm_processes_clarification_response(initialized_cm):
    """
    Test that the LLM processes a clarification response.

    After asking for clarification, when the user provides more details,
    the LLM should process those details.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Send an ambiguous request
    await cm._step(
        SMSReceived(
            contact=contact,
            content="I need help with a contact",
        ),
    )

    # Provide clarification
    result = await cm._step(
        SMSReceived(
            contact=contact,
            content="Show me only contacts that have email addresses.",
        ),
    )

    # LLM should process the clarified request
    assert result.llm_ran, "Expected LLM to process clarification"


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_multi_turn_task_conversation(initialized_cm):
    """
    Test a multi-turn conversation involving task management.

    Verifies that the LLM processes each turn appropriately.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[1]

    # Turn 1: Request a task
    result1 = await cm._step(
        SMSReceived(
            contact=contact,
            content="List all my contacts",
        ),
    )
    assert result1.llm_ran, "Expected LLM to run for task request"

    # Turn 2: Ask about status
    result2 = await cm._step(
        SMSReceived(
            contact=contact,
            content="How's that task going?",
        ),
    )
    assert result2.llm_ran, "Expected LLM to run for status query"

    # Turn 3: Stop the task
    result3 = await cm._step(
        SMSReceived(
            contact=contact,
            content="Stop that task please",
        ),
    )
    assert result3.llm_ran, "Expected LLM to run for stop request"
