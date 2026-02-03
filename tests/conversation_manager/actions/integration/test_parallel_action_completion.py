"""
Regression tests for parallel action completion notification handling.

These tests verify that when multiple parallel actions are started:
1. When one action completes while others are still running, the CM sees the result
2. When the CM chooses to `wait` (because other actions are running), the completion
   notification is NOT lost
3. When subsequent actions complete, the CM does NOT re-execute already-completed work

Bug scenario this covers:
- User: "Create contact Bob, then search for X, then create task for Bob"
- CM starts parallel actions: (1) create Bob, (2) search for X
- (1) completes → CM sees "Bob created", but waits (2 still running)
- (2) completes → CM sees search result, but FORGETS Bob was created
- CM incorrectly tries to create Bob again

Root cause: Action completion notifications are cleared by commit() when CM chooses
to `wait`, causing the CM to lose track of completed work.
"""

import random
import string

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.cm_helpers import filter_events_by_type
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    wait_for_actor_completion,
    inject_actor_result,
    run_cm_until_wait,
)
from unity.conversation_manager.events import SMSReceived, ActorHandleStarted

pytestmark = [pytest.mark.integration, pytest.mark.eval]


def _random_name_suffix() -> str:
    """Generate a random alphabetic suffix for unique contact names.

    Returns names like 'Xyzab' that comply with the contact name pattern
    (no underscores, no digits).
    """
    # Use only letters to comply with contact name validation
    letters = string.ascii_lowercase
    suffix = "".join(random.choice(letters) for _ in range(5))
    return suffix.capitalize()


def _count_contacts_named(cm, first_name: str) -> int:
    """Count how many contacts exist with the given first name."""
    mgr = cm.cm.contact_manager
    if mgr is None:
        return 0
    # Use filter to find contacts by first name
    payload = mgr.filter_contacts(
        filter=f"first_name == '{first_name}'",
        limit=100,
    )
    contacts = payload.get("contacts") or []
    return len(contacts)


def _count_create_actions_for_name(events: list, contact_name: str) -> int:
    """Count how many ActorHandleStarted events are for creating a specific contact.

    This helps detect when the CM redundantly starts duplicate actions
    (even if the Actor prevents actual duplicates).

    Args:
        events: List of ActorHandleStarted events
        contact_name: The contact name to search for

    Returns:
        Number of "create contact" actions for the given name.
    """
    count = 0
    for evt in events:
        if not isinstance(evt, ActorHandleStarted):
            continue
        query = getattr(evt, "query", "") or ""
        # Check if this action is for creating this specific contact
        if contact_name.lower() in query.lower() and "create" in query.lower():
            count += 1
    return count


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_parallel_action_completion_preserves_first_result(
    initialized_cm_codeact,
):
    """
    Regression test: CM must not re-create a contact when parallel actions complete.

    Scenario:
    1. User asks to create a contact AND do another task (e.g., look up something)
    2. CM starts two parallel actions
    3. First action (create contact) completes → CM sees notification, but waits
    4. Second action completes → CM must remember first action already completed
    5. CM should NOT try to create the contact again

    This test fails if the completion notification is lost between steps 3 and 4.
    """
    cm = initialized_cm_codeact

    # Use a unique name to avoid interference from other tests
    # Name must comply with contact schema (no underscores or digits)
    unique_name = f"Testbob{_random_name_suffix()}"

    # Verify contact doesn't exist yet
    initial_count = _count_contacts_named(cm, unique_name)
    assert initial_count == 0, f"Contact {unique_name} should not exist yet"

    # Step 1: Send compound request that should trigger parallel actions
    # One action creates a contact, another does an unrelated lookup
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                f"I need you to do two things: "
                f"First, create a new contact named {unique_name}. "
                f"Second, look up Alice's phone number. "
                f"These are independent tasks."
            ),
        ),
    )

    # Step 2: Verify at least one actor was started (may be 1 or 2 parallel)
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, "Expected at least one ActorHandleStarted event"

    # Collect all actor events across all steps for later verification
    all_actor_events = list(actor_events)

    # Get handle IDs for all started actions
    handle_ids = [e.handle_id for e in actor_events]

    # Step 3: Wait for each action to complete and inject results back to CM
    # This simulates the real flow where actions complete asynchronously
    for i, handle_id in enumerate(handle_ids):
        try:
            # Wait for this actor to complete
            final_result = await wait_for_actor_completion(cm, handle_id, timeout=90)

            # Inject the result back to CM (simulates ActorResult event)
            await inject_actor_result(
                cm,
                handle_id=handle_id,
                result=final_result,
                success=True,
            )

            # Let CM process the result and decide what to do next
            step_events = await run_cm_until_wait(cm, max_steps=5)

            # Collect any new actor events from this step
            new_actor_events = filter_events_by_type(step_events, ActorHandleStarted)
            all_actor_events.extend(new_actor_events)

        except TimeoutError:
            # If an action times out, it may have already been processed
            pass
        except KeyError:
            # Handle may have been removed from in_flight_actions
            pass

    # Step 4: Verify contact was created exactly ONCE
    final_count = _count_contacts_named(cm, unique_name)
    assert final_count == 1, (
        f"Expected exactly 1 contact named {unique_name}, but found {final_count}. "
        f"This indicates the CM lost track of the contact creation and duplicated work."
    )

    # Step 5: Verify CM did not start redundant actions
    # Even if Actor prevents actual duplicates, starting redundant actions wastes resources
    # and indicates the CM lost track of completed work
    create_action_count = _count_create_actions_for_name(all_actor_events, unique_name)
    assert create_action_count <= 1, (
        f"CM started {create_action_count} action(s) to create contact '{unique_name}'. "
        f"Expected at most 1. This indicates the CM forgot about the first action's result "
        f"and started a redundant action (even though Actor may prevent actual duplicate)."
    )

    # Verify no errors occurred
    assert_no_errors(result)
