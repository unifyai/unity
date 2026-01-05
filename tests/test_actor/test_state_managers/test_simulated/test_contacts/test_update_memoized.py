"""
Actor tests for ContactManager.update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.contacts.update`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
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
async def test_updates_use_memoized_function(
    request_text: str,
    mock_verification,
):
    """Verify Actor selects memoized function via semantic search for updates."""

    implementation = '''
async def update_contacts_instruction(instruction: str, response_format=None) -> str:
    """Mutate contact records (create/update/delete/merge) via the contacts manager.

    **Use when** the user requests to change contacts: add a person, edit fields,
    delete a contact, or merge duplicates.

    **How it works**: calls the contacts mutation tool:
    - `await primitives.contacts.update(instruction, response_format=response_format)`

    **Do NOT use when**:
    - the user is asking a read-only question about contacts (use `primitives.contacts.ask`)
    - the user is asking about message history/transcripts (use `primitives.transcripts.ask`)
    - the user needs current external facts (use `primitives.web.ask`)

    Args:
        instruction: The contact update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the contacts manager update operation as a string.
    """
    handle = await primitives.contacts.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
'''
    async with make_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "update_contacts_instruction")
        assert_tool_called(handle, "primitives.contacts.update")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may be overly strict (e.g., phone format mismatches).
        # The key test is that the memoized function was correctly selected and used.
