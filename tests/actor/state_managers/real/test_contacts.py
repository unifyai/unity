"""Real ContactManager tests for Actor.

Tests that Actor correctly calls real ContactManager methods and verifies
actual state mutations.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_memoized_function_used,
    assert_tool_called,
    get_state_manager_tools,
    make_hierarchical_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager(mock_verification):
    """Test that Actor calls ContactManager.ask for contact queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and seed data
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Eve",
            surname="Adams",
            email_address="eve.adams@example.com",
        )

        # Call actor with natural language query
        handle = await actor.act(
            "What is Eve Adams' email address?",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected email
        assert "eve.adams@example.com" in result.lower()

        # Assert correct tool was called
        assert_tool_called(handle, "primitives.contacts.ask")

        # Assert only contacts tools were used
        state_manager_tools = get_state_manager_tools(handle)
        assert all("contacts" in tool for tool in state_manager_tools)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for contact queries."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and seed data
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Eve",
            surname="Adams",
            email_address="eve.adams@example.com",
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def ask_contacts(question: str, response_format=None) -> str:
    """Query the contacts database (people/organizations) using the contacts manager.

    **Use when** the question is about stored contact records: emails, phone numbers,
    job titles, locations, preferences, account ownership, etc.

    **Do NOT use when**:
    - the question is about message history/transcripts (use transcripts)
    - the question is about current events/weather/news (use web)
    - the request is to mutate contacts/tasks/knowledge/guidance (use the relevant update tool)

    Args:
        question: The contact-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the contacts manager as a string.
    """
    handle = await primitives.contacts.ask(question, response_format=response_format)
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with natural language query
        handle = await actor.act(
            "What is Eve Adams' email address? Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert result contains expected email
        assert "eve.adams@example.com" in result.lower()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "ask_contacts")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager(mock_verification):
    """Test that Actor calls ContactManager.update for mutations."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and seed data
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Bob",
            surname="Test",
            email_address="bob.update@example.com",
        )

        # Call actor with update request
        handle = await actor.act(
            "Update the contact with email bob.update@example.com: set the phone to 555-777-8888. Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert update tool was called (NOT ask)
        assert_tool_called(handle, "primitives.contacts.update")

        # Verify the tool selection - should not call ask
        state_manager_tools = get_state_manager_tools(handle)
        assert "primitives.contacts.ask" not in state_manager_tools

        # Verify mutation actually occurred
        rows = cm.filter_contacts(filter="email_address == 'bob.update@example.com'")[
            "contacts"
        ]
        assert rows and rows[0].phone_number == "5557778888"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized(mock_verification):
    """Test that Actor uses memoized function for contact updates."""
    async with make_hierarchical_actor(impl="real") as actor:

        # Access real ContactManager and seed data
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Bob",
            surname="Test",
            email_address="bob.update@example.com",
        )

        # Create FunctionManager and seed memoized function
        fm = FunctionManager()
        implementation = '''
async def update_contact_phone(email: str, phone: str) -> str:
    """Mutate contact records (create/update/delete/merge) via the contacts manager.

    **Use when** the user requests to change contacts: add a person, edit fields,
    delete a contact, or merge duplicates.

    **Do NOT use when**:
    - the user is asking a read-only question about contacts (use `primitives.contacts.ask`)
    - the user is asking about message history/transcripts (use `primitives.transcripts.ask`)
    - the user needs current external facts (use `primitives.web.ask`)

    Args:
        email: The email address of the contact to update.
        phone: The new phone number to set.

    Returns:
        The result from the contacts manager update operation as a string.
    """
    handle = await primitives.contacts.update(
        f"Update the contact with email {email}: set the phone to {phone}."
    )
    result = await handle.result()
    return result
'''
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        # Call actor with update request
        handle = await actor.act(
            "Update the contact with email bob.update@example.com: set the phone to 555-777-8888. Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )

        # Wait for result
        result = await handle.result()

        # Assert memoized function was used
        assert_memoized_function_used(handle, "update_contact_phone")

        # Assert underlying primitive was called
        assert_tool_called(handle, "primitives.contacts.update")

        # Verify mutation actually occurred
        rows = cm.filter_contacts(filter="email_address == 'bob.update@example.com'")[
            "contacts"
        ]
        assert rows and rows[0].phone_number == "5557778888"
