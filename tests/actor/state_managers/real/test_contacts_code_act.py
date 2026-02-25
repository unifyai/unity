"""Real ContactManager routing tests for CodeActActor.

Validates that CodeActActor uses ``execute_function`` for simple single-primitive
contact operations, both with and without FunctionManager discovery tools.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)
from unity.manager_registry import ManagerRegistry


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes read-only contact question via execute_function."""
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
        assert_used_execute_function(handle)
        assert "primitives.contacts.ask" in calls
        assert all(c.startswith("primitives.contacts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_with_fm_tools():
    """CodeAct routes contact query via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
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
        assert_used_execute_function(handle)
        assert "primitives.contacts.ask" in calls
        assert all(c.startswith("primitives.contacts.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager():
    """CodeAct routes contact mutation via execute_function."""
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

        assert_used_execute_function(handle)
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
async def test_update_calls_manager_with_fm_tools():
    """CodeAct routes contact mutation via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
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

        assert_used_execute_function(handle)
        assert "primitives.contacts.update" in calls
        assert "primitives.contacts.ask" not in calls

        rows = cm.filter_contacts(filter="email_address == 'bob.update@example.com'")[
            "contacts"
        ]
        assert rows and rows[0].phone_number == "5557778888"
