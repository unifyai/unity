"""
Actor tests for ContactManager ask+update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects separate
memoized functions via semantic search for combined requests and that both
`primitives.contacts.ask` and `primitives.contacts.update` are invoked.

Pattern: Memoized functions (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.actor.state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_REQUESTS: list[str] = [
    (
        "Ask for the current phone number for Bob Johnson using the contacts manager and provide context. "
        "Also update his phone number to 555-222-3333 and confirm the change. "
    ),
    (
        "Ask for the total number of contacts currently stored using the contacts manager with a summary. "
        "Also set Jane Doe's description to 'Preferred contact is email' and confirm. "
    ),
    (
        "Answer this question using the contacts manager – what is Alice Smith's current email? Include relevant details. "
        "Also update Alice Smith's phone number to +1-555-101-2020 and provide confirmation. "
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
async def ask_contacts_with_analysis(question: str, response_format=None) -> str:
    """Query contact records and produce a structured analysis with context.

    **ALWAYS use this function** for ANY contact-related read-only question, regardless
    of complexity. Direct calls to primitives.contacts.ask are not allowed when this
    function is available - even for simple lookups like "What is X's email?".

    This helper does two steps:
    1) Retrieves relevant contact information via primitives.contacts.ask
    2) Synthesizes a structured response with context via computer_primitives.reason

    **Do NOT use when**:
    - the question is about message history/transcripts (use transcripts)
    - the question is about current events/weather/news (use web)
    - the request is to mutate contacts (use contacts update)

    Args:
        question: The contact-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A structured analysis with contact information and context.
    """
    handle = await primitives.contacts.ask(question, response_format=response_format)
    raw_result = await handle.result()

    analysis = await computer_primitives.reason(
        request=(
            "Produce a structured summary with: "
            "1) Direct answer (the contact info requested), "
            "2) Relevant context (role, organization, preferences), "
            "3) Any related contacts or relationships if applicable."
        ),
        context=str(raw_result),
    )
    return analysis if isinstance(analysis, str) else str(analysis)
''',
        '''
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

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.actor.state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "ask_contacts_with_analysis")
        assert_memoized_function_used(handle, "update_contacts_with_confirmation")
        assert_tool_called(handle, "primitives.contacts.ask")
        assert_tool_called(handle, "primitives.contacts.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.contacts.ask" in state_manager_tools
        assert "primitives.contacts.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
