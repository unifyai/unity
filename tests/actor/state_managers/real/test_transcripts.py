"""Real TranscriptManager tests for Actor.

Tests that Actor correctly calls real TranscriptManager methods.
"""

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.contact_manager.types.contact import Contact
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager(mock_verification):
    """Test that Actor calls TranscriptManager.ask for transcript queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and TranscriptManager
        cm = ManagerRegistry.get_contact_manager()
        tm = ManagerRegistry.get_transcript_manager()

        # Seed contacts
        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        # Seed message
        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        # Call actor with natural language query
        handle = await actor.act(
            "Show the most recent message that mentions the budget.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result is non-empty
        assert result and len(result) > 0

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.transcripts.ask")

        # Assert only transcripts tools were used (may also call contacts for participant lookup)
        state_manager_tools = get_state_manager_tools(handle)
        assert any("transcripts" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for transcript queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and TranscriptManager
        cm = ManagerRegistry.get_contact_manager()
        tm = ManagerRegistry.get_transcript_manager()

        # Seed contacts
        alice = Contact(
            first_name="Alice",
            surname="Smith",
            email_address="alice.smith@example.com",
        )
        bob = Contact(
            first_name="Bob",
            surname="Jones",
            email_address="bob.jones@example.com",
        )

        # Seed message
        tm.log_first_message_in_new_exchange(
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [bob],
                "timestamp": datetime.now(timezone.utc),
                "content": "Subject: Q3 Budget\nBody: Final numbers are ready for review.",
            },
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def ask_transcripts_question(question: str, response_format=None) -> str:
    """Answer questions by searching YOUR conversation transcripts/messages (including SMS).

    **Use when** the user asks about *their own communication history*:
    - what someone said, when, and where (chat/SMS/email transcript content)
    - find the most recent message mentioning a topic
    - list messages from a person in a time window (e.g., "last 24 hours")
    - find the last SMS with a contact

    **Do NOT use when**:
    - the user needs *current external facts* (use web search: `primitives.web.ask`)
    - the user is asking about contact records (use contacts: `primitives.contacts.ask`)
    - the user is updating guidance/knowledge/tasks (use the appropriate update tool)

    This is NOT a public web search function; it does not consult external sources.

    Args:
        question: The transcript-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the transcript manager as a string.
    """
    handle = await primitives.transcripts.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query
        handle = await actor.act(
            "Show the most recent message that mentions the budget. Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result is non-empty
        assert result and len(result) > 0

        # Assert memoized function was used
        assert_memoized_function_used(handle, "ask_transcripts_question")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.transcripts.ask")
