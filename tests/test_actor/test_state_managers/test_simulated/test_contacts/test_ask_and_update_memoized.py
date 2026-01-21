"""
Actor tests for ContactManager ask+update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects separate
memoized functions via semantic search for combined requests and that both
`primitives.contacts.ask` and `primitives.contacts.update` are invoked.

Pattern: Memoized functions (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_REQUESTS: list[str] = [
    (
        "Ask for the current phone number for Bob Johnson using the contacts manager. "
        "Also update his phone number to 555-222-3333. "
    ),
    (
        "Ask for the total number of contacts currently stored using the contacts manager. "
        "Also set Jane Doe's description to 'Preferred contact is email'. "
    ),
    (
        "Answer this question using the contacts manager – what is Alice Smith's current email? "
        "Also update Alice Smith's phone number to +1-555-101-2020. "
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "combined_text",
    COMBINED_REQUESTS,
    ids=[
        "bob_phone_read_then_update",
        "count_contacts_then_update_description",
        "alice_email_then_update_phone",
    ],
)
async def test_combined_queries_use_memoized_function(
    combined_text: str,
    mock_verification,
):
    """Verify Actor selects separate memoized functions via semantic search for combined ops."""

    implementations = [
        '''
async def ask_contacts_question(question: str, response_format=None) -> str:
    """Query the contacts database (people/organizations) using the contacts manager.

    **Use when** the question is about stored contact records: emails, phone numbers,
    job titles, locations, preferences, account ownership, etc.

    **How it works**: calls:
    - `await primitives.contacts.ask(question, response_format=response_format)`

    **Do NOT use when**:
    - the question is about message history/transcripts (use transcripts)
    - the question is about current events/weather/news (use web)
    - the request is to mutate contacts/tasks/knowledge/guidance (use the relevant update tool)

    Args:
        question: The contact-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the contacts manager as a string.
    """
    handle = await primitives.contacts.ask(question, response_format=response_format)
    result = await handle.result()
    return result
''',
        '''
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
''',
    ]
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        fm.add_functions(implementations=implementations, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{combined_text} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "ask_contacts_question")
        assert_memoized_function_used(handle, "update_contacts_instruction")
        assert_tool_called(handle, "primitives.contacts.ask")
        assert_tool_called(handle, "primitives.contacts.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.contacts.ask" in state_manager_tools
        assert "primitives.contacts.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
