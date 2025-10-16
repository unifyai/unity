import asyncio

import pytest

from tests.helpers import _handle_project
from unity.common.state_managers import (
    state_manager_exists,
    state_manager_method_exists,
)
from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EVENT_BUS


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_state_manager_exists_true_after_contact_manager_ask() -> None:
    """End-to-end: ContactManager.ask should emit ManagerMethod events that make
    the manager discoverable via distinct-groups on Events/ManagerMethod.manager.
    """

    cm = ContactManager()

    # Trigger a real manager-method interaction that logs to EventBus
    handle = await cm.ask("Please echo 'ping'.")
    await handle.result()

    # Ensure async logger has flushed to backend before grouping
    EVENT_BUS.join_published()
    await asyncio.sleep(0.05)

    assert state_manager_exists("ContactManager") is True


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_state_manager_exists_false_for_unknown_manager() -> None:
    """Unknown manager names should not be reported present in groups."""

    cm = ContactManager()
    handle = await cm.ask("A quick check to ensure events exist.")
    await handle.result()
    EVENT_BUS.join_published()
    await asyncio.sleep(0.05)

    assert state_manager_exists("DefinitelyNotAManagerName") is False


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_state_manager_method_exists_true_for_contact_manager_ask() -> None:
    """After ContactManager.ask runs, method-level grouping should include 'ask'."""

    cm = ContactManager()

    handle = await cm.ask("Please echo 'pong'.")
    await handle.result()

    EVENT_BUS.join_published()
    await asyncio.sleep(0.05)

    assert state_manager_method_exists("ContactManager", "ask") is True


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_state_manager_method_exists_false_for_unknown_method() -> None:
    """Unknown method names should not be reported for an existing manager."""

    cm = ContactManager()
    handle = await cm.ask("Ensure we have at least one event.")
    await handle.result()
    EVENT_BUS.join_published()
    await asyncio.sleep(0.05)

    assert (
        state_manager_method_exists("ContactManager", "definitely_not_a_method")
        is False
    )
