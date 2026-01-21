"""
Actor tests for ContactManager combined ask+update operations.

Tests that HierarchicalActor correctly generates plans calling both
`primitives.contacts.ask` and `primitives.contacts.update` (and no other state manager tools).

Pattern: On-the-fly planning (Actor generates plans dynamically)
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_REQUESTS: list[str] = [
    (
        "Ask for the current phone number for Bob Johnson using the contacts manager. "
        "Also update his phone number to 555-222-3333. "
        "Use only ContactManager.ask and ContactManager.update; do not use any other tools."
    ),
    (
        "Ask for the total number of contacts currently stored using the contacts manager. "
        "Also set Jane Doe's description to 'Preferred contact is email'. "
        "Use only ContactManager.ask and ContactManager.update; do not use any other tools."
    ),
    (
        "Answer this question using the contacts manager – what is Alice Smith's current email? "
        "Also update Alice Smith's phone number to +1-555-101-2020. "
        "Use only ContactManager.ask and ContactManager.update; no other tools."
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", COMBINED_REQUESTS)
async def test_combined_ask_update_expected_tools(
    request_text: str,
    mock_verification,
):
    """Verify Actor generates plans calling both primitives.contacts.ask and update."""
    async with make_hierarchical_actor(impl="simulated") as actor:

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

        # Verify both ask and update were called
        assert_tool_called(handle, "primitives.contacts.ask")
        assert_tool_called(handle, "primitives.contacts.update")

        # Check if request explicitly restricts tools
        request_explicitly_restricts = (
            "use only" in request_text.lower()
            or "no other tools" in request_text.lower()
        )

        state_manager_tools = set(get_state_manager_tools(handle))
        if request_explicitly_restricts:
            # Strict: only contacts.ask and contacts.update allowed (request explicitly restricted)
            assert state_manager_tools == {
                "primitives.contacts.ask",
                "primitives.contacts.update",
            }, f"Expected only primitives.contacts.ask and primitives.contacts.update (request explicitly restricted tools), saw: {state_manager_tools}"
        else:
            # Relaxed: at least contacts.ask and contacts.update must be called
            assert (
                "primitives.contacts.ask" in state_manager_tools
            ), f"Expected primitives.contacts.ask to be called, saw: {state_manager_tools}"
            assert (
                "primitives.contacts.update" in state_manager_tools
            ), f"Expected primitives.contacts.update to be called, saw: {state_manager_tools}"

        # Verify that verification was bypassed (no verification failures in log).
        log_text = "\n".join(handle.action_log)
        assert "verification failed" not in log_text.lower()
