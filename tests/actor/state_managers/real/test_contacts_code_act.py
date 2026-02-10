"""Real ContactManager routing tests for CodeActActor.

These mirror `test_contacts.py` but use CodeActActor (code-first tool loop).
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    extract_code_act_execute_function_names,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes read-only contact question → primitives.contacts.ask."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Eve",
            surname="Adams",
            email_address="eve.adams@example.com",
        )

        handle = await actor.act(
            "What is Eve Adams' email address?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "eve.adams@example.com" in str(result).lower()
        assert "primitives.contacts.ask" in calls
        assert all(c.startswith("primitives.contacts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for contact questions."""
    fm = FunctionManager()
    implementation = """
async def ask_contacts(question: str, response_format=None) -> str:
    \"\"\"Query the contacts database (people/organizations) using the contacts manager.\"\"\"
    handle = await primitives.contacts.ask(question, response_format=response_format)
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Eve",
            surname="Adams",
            email_address="eve.adams@example.com",
        )

        handle = await actor.act(
            "What is Eve Adams' email address?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert "eve.adams@example.com" in str(result).lower()
        assert_code_act_function_manager_used(handle)

        # Either via injected function call or direct primitive call is acceptable,
        # but the FunctionManager tool must have been used.
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        assert "ask_contacts" in snippets

        assert "primitives.contacts.ask" in calls
        assert all(c.startswith("primitives.contacts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager():
    """CodeAct routes contact mutation → primitives.contacts.update."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Bob",
            surname="Test",
            email_address="bob.update@example.com",
        )

        handle = await actor.act(
            "Update the contact with email bob.update@example.com: set the phone to 555-777-8888.",
            clarification_enabled=False,
        )
        await handle.result()

        assert "primitives.contacts.update" in calls
        assert "primitives.contacts.ask" not in calls

        rows = cm.filter_contacts(filter="email_address == 'bob.update@example.com'")[
            "contacts"
        ]
        assert rows and rows[0].phone_number == "5557778888"


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_memoized():
    """CodeAct uses FunctionManager (when available) for contact mutations."""
    fm = FunctionManager()
    implementation = """
async def update_contact_phone(email: str, phone: str) -> str:
    \"\"\"Update a contact's phone number via the contacts manager.\"\"\"
    handle = await primitives.contacts.update(
        f"Update the contact with email {email}: set the phone to {phone}."
    )
    return await handle.result()
"""
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        cm = ManagerRegistry.get_contact_manager()
        cm._create_contact(
            first_name="Bob",
            surname="Test",
            email_address="bob.update@example.com",
        )

        handle = await actor.act(
            "Update the contact with email bob.update@example.com: set the phone to 555-777-8888.",
            clarification_enabled=False,
        )
        await handle.result()

        assert_code_act_function_manager_used(handle)
        snippets = "\n\n".join(extract_code_act_execute_code_snippets(handle))
        fn_names = extract_code_act_execute_function_names(handle)
        assert (
            "update_contact_phone" in snippets or "update_contact_phone" in fn_names
        ), (
            f"Expected memoized function 'update_contact_phone' to be invoked. "
            f"execute_code snippets: {snippets!r}, execute_function calls: {fn_names}"
        )

        assert "primitives.contacts.update" in calls
        assert "primitives.contacts.ask" not in calls

        rows = cm.filter_contacts(filter="email_address == 'bob.update@example.com'")[
            "contacts"
        ]
        assert rows and rows[0].phone_number == "5557778888"
