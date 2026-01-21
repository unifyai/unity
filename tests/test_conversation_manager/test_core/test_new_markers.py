"""
tests/test_conversation_manager/test_core/test_new_markers.py
=============================================================

Integration test verifying **NEW** markers work correctly under test fixtures.

This test catches fixture setup bugs where cm.last_snapshot is set before
prompt_now is patched, causing the **NEW** marker comparison to fail.

Unlike the unit tests in test_utils.py (which test Renderer in isolation),
this test exercises the actual fixture wiring that integration tests use.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS


@pytest.mark.asyncio
@_handle_project
async def test_new_marker_appears_for_incoming_message(initialized_cm):
    """
    Verify that messages added after CM initialization get **NEW** markers.

    This is a sanity check for the test fixture setup. The bug this catches:
    - module-scoped CM created with real time last_snapshot
    - function-scoped patching sets fixed timestamps for messages
    - last_snapshot (future) > message.timestamp (past) → no **NEW** marker
    - LLM sees no NEW messages → calls wait instead of responding

    The fix resets last_snapshot in initialized_cm using the patched prompt_now.
    """
    cm = initialized_cm
    contact = TEST_CONTACTS[0]

    # Add a message to the contact index (simulates receiving a message)
    cm.contact_index.push_message(
        contact_id=contact["contact_id"],
        sender_name=f"{contact['first_name']} {contact['surname']}",
        thread_name="unify_message",
        message_content="Test message for NEW marker verification",
        role="user",
    )

    # Render the state (this is what the LLM sees)
    rendered_state = cm.cm.prompt_renderer.render_state(
        cm.contact_index,
        cm.cm.notifications_bar,
        cm.cm.active_tasks,
        cm.cm.last_snapshot,
    )

    # The message should have the **NEW** marker
    assert "**NEW**" in rendered_state, (
        "Message should have **NEW** marker. "
        "This indicates the fixture setup is broken - last_snapshot may not be "
        "using the patched prompt_now. Check initialized_cm fixture."
    )
    assert "Test message for NEW marker verification" in rendered_state
