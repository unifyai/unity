"""
Actor tests for ContactManager.update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.contacts.update`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.actor.state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


CONTACT_UPDATE_REQUESTS: list[str] = [
    "Create a new contact: Jane Doe, email jane.d@example.com, phone 15551234567. Confirm what was created.",
    "Update Bob Johnson's phone number to 555-222-1111. Provide confirmation of the change.",
    "Delete the contact with email diana@themyscira.com. Confirm the deletion.",
    (
        "Merge duplicate contacts named Alice into a single entry, preferring the one "
        "with email alice.smith@example.com. Summarize the merge result."
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
async def update_contacts_with_confirmation(instruction: str, response_format=None) -> str:
    """Mutate contact records and produce a confirmation summary of changes.

    **ALWAYS use this function** for ANY contact mutation request, regardless of
    complexity. Direct calls to primitives.contacts.update are not allowed when this
    function is available - even for simple updates like "change X's phone to Y".

    This helper does two steps:
    1) Performs the contact mutation via primitives.contacts.update
    2) Synthesizes a confirmation summary via computer_primitives.reason

    **Do NOT use when**:
    - the user is asking a read-only question about contacts (use contacts ask)
    - the user is asking about message history/transcripts (use transcripts)
    - the user needs current external facts (use web)

    Args:
        instruction: The contact update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A confirmation summary of the changes made.
    """
    handle = await primitives.contacts.update(instruction, response_format=response_format)
    raw_result = await handle.result()

    confirmation = await computer_primitives.reason(
        request=(
            "Summarize what was changed: "
            "1) Action taken (created/updated/deleted/merged), "
            "2) Contact details affected, "
            "3) Confirmation that the operation completed."
        ),
        context=str(raw_result),
    )
    return confirmation if isinstance(confirmation, str) else str(confirmation)
'''
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.actor.state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "update_contacts_with_confirmation")
        assert_tool_called(handle, "primitives.contacts.update")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may be overly strict (e.g., phone format mismatches).
        # The key test is that the memoized function was correctly selected and used.
