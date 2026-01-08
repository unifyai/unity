"""
Actor tests for ContactManager.update operations.

Tests that HierarchicalActor correctly generates plans calling `primitives.contacts.update`
for contact mutations (with optional read-before-write via `primitives.contacts.ask`).
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_actor

pytestmark = pytest.mark.eval


CONTACT_UPDATE_REQUESTS: list[str] = [
    "Create a new contact: Jane Doe, email jane.d@example.com, phone 15551234567.",
    "Update Bob Johnson's phone number to 555-222-1111.",
    "Delete the contact with email diana@themyscira.com.",
    (
        "Merge duplicate contacts named Alice into a single entry, preferring the one "
        "with email alice.smith@example.com."
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", CONTACT_UPDATE_REQUESTS)
async def test_updates_call_manager_update(
    request_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling primitives.contacts.update."""
    async with make_actor(impl="simulated") as actor:
        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        # Verify plan was generated
        assert handle.plan_source_code
        assert "async def" in handle.plan_source_code
        assert "primitives." in handle.plan_source_code

        from tests.test_actor.test_state_managers.utils import (
            assert_tool_called,
            get_state_manager_tools,
        )

        # Verify primitives.contacts.update was called
        assert_tool_called(handle, "primitives.contacts.update")

        # Allow primitives.contacts.ask as well (for read-before-write patterns)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.contacts.update" in state_manager_tools
        assert state_manager_tools <= {
            "primitives.contacts.ask",
            "primitives.contacts.update",
        }

        # Verify that verification was bypassed (no verification failures in log).
        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
