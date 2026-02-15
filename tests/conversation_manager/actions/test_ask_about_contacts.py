"""
Contact read routing tests for the ConversationManager brain.

Verify that the CM brain correctly routes to ``ask_about_contacts``
(not ``act``) for read-only contact queries, and that the query text
includes the right details for the ContactManager to service.

Uses SimulatedContactManager — we only verify that the brain routes
correctly and passes a well-formed question, not the CM's actual answer.
"""

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


def _assert_contact_ask_triggered(
    result,
    *,
    expected_substrings: list[str],
) -> None:
    """Assert ``ask_about_contacts`` was called and each expected substring
    appears in the query text OR in the ``response_format`` keys.
    """
    events = filter_events_by_type(result.output_events, ActorHandleStarted)
    contact_events = [e for e in events if e.action_name == "ask_about_contacts"]
    assert contact_events, (
        f"Expected ask_about_contacts to be triggered, "
        f"but got action(s): {[e.action_name for e in events] or 'none'}"
    )
    evt = contact_events[0]
    query = evt.query.lower()
    rf_keys = " ".join((evt.response_format or {}).keys()).lower()
    searchable = f"{query} {rf_keys}"
    for substr in expected_substrings:
        assert substr.lower() in searchable, (
            f"Expected '{substr}' in ask_about_contacts query or response_format keys, "
            f"got query: {query}, response_format keys: {rf_keys}"
        )


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_contact_preference_lookup(initialized_cm):
    """Asking about a contact's preferred channel should route to
    ``ask_about_contacts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Does Sarah prefer phone or email?",
        ),
    )

    _assert_contact_ask_triggered(
        result,
        expected_substrings=["sarah"],
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_contact_search_by_location(initialized_cm):
    """Searching for contacts by location should route to
    ``ask_about_contacts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I'm heading to Berlin next week. Do we know anyone there?",
        ),
    )

    _assert_contact_ask_triggered(
        result,
        expected_substrings=["berlin"],
    )
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_contact_phone_number_lookup(initialized_cm):
    """Asking for a contact's phone number should route to
    ``ask_about_contacts``."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's Alice's phone number?",
        ),
    )

    _assert_contact_ask_triggered(
        result,
        expected_substrings=["alice"],
    )
    assert_efficient(result, 3)
