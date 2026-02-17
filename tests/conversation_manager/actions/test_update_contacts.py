"""
Contact mutation routing tests for the ConversationManager brain.

Verify that the CM brain correctly routes to ``update_contacts``
(not ``act``) for contact creation and modification, and that the
query text includes the right details for the ContactManager to service.

Uses SimulatedContactManager — we only verify that the brain routes
correctly and passes a well-formed question, not the CM's actual answer.
SimulatedContactManager.update() blocks deterministic mode, so we
monkeypatch it to return a dummy handle for routing verification.
"""

import functools

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_efficient,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _assert_contact_update_triggered(
    result,
    *,
    expected_substrings: list[str],
) -> None:
    """Assert ``update_contacts`` was called and each expected substring
    appears in the query text OR in the ``response_format`` keys.
    """
    events = filter_events_by_type(result.output_events, ActorHandleStarted)
    contact_events = [e for e in events if e.action_name == "update_contacts"]
    assert contact_events, (
        f"Expected update_contacts to be triggered, "
        f"but got action(s): {[e.action_name for e in events] or 'none'}"
    )
    evt = contact_events[0]
    query = evt.query.lower()
    rf_keys = " ".join((evt.response_format or {}).keys()).lower()
    searchable = f"{query} {rf_keys}"
    for substr in expected_substrings:
        assert substr.lower() in searchable, (
            f"Expected '{substr}' in update_contacts query or response_format keys, "
            f"got query: {query}, response_format keys: {rf_keys}"
        )


def _patch_simulated_update(cm_driver) -> None:
    """Bypass SimulatedContactManager.update() deterministic guard.

    The deterministic mode raises RuntimeError because the simulated store
    wouldn't actually be mutated. For routing tests we only care that the
    brain *chose* update_contacts — not that the update executes. So we
    redirect to the ``ask`` path which returns a working handle.
    """
    contact_mgr = cm_driver.cm.contact_manager
    if contact_mgr is None:
        return

    original_ask = contact_mgr.ask

    @functools.wraps(contact_mgr.update)
    async def _update_as_ask(text, **kwargs):
        return await original_ask(text, **kwargs)

    contact_mgr.update = _update_as_ask


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_create_contact(initialized_cm):
    """Saving a new contact should route to ``update_contacts``."""
    cm = initialized_cm
    _patch_simulated_update(cm)

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Just met Jane Doe at the conference. Can you save her email jane.d@example.com?",
        ),
    )

    _assert_contact_update_triggered(
        result,
        expected_substrings=["jane"],
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_save_service_number(initialized_cm):
    """Saving a service/support number should route to ``update_contacts``
    and describe the organisation, not the person who answered."""
    cm = initialized_cm
    _patch_simulated_update(cm)

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Save this number 8005551234 - it's the Acme billing support "
                "line. Sarah answered when I called."
            ),
        ),
    )

    _assert_contact_update_triggered(
        result,
        expected_substrings=["8005551234"],
    )

    # The query should reference the service/company, not frame Sarah as the contact
    events = filter_events_by_type(result.output_events, ActorHandleStarted)
    query = events[0].query.lower()
    assert (
        "acme" in query or "billing" in query or "support" in query
    ), f"update_contacts query should describe the service/organisation, got: {events[0].query}"

    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_update_existing_contact(initialized_cm):
    """Updating a field on an existing contact should route to
    ``update_contacts``."""
    cm = initialized_cm
    _patch_simulated_update(cm)

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Update Bob's phone number to +15551234567.",
        ),
    )

    _assert_contact_update_triggered(
        result,
        expected_substrings=["bob"],
    )
    assert_efficient(result, 3)
