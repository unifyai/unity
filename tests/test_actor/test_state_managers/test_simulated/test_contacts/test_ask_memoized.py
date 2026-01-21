"""
Actor tests for ContactManager.ask via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.contacts.ask`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


CONTACT_QUESTIONS: list[str] = [
    "Which of our contacts prefers to be contacted by phone?",
    "Find the email address for the contact named Sarah (use your contacts only).",
    "List any contacts located in Berlin.",
    "Who is the primary point of contact for the Contoso account?",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", CONTACT_QUESTIONS)
async def test_questions_use_memoized_function(
    question: str,
    mock_verification,
):
    """Verify Actor selects memoized function via semantic search."""

    implementation = '''
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
'''
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()
        fm.add_functions(implementations=implementation, overwrite=True)
        actor.function_manager = fm

        handle = await actor.act(
            f"{question} Do not ask clarifying questions. Do not create any stubs. Generate the full plan. Proceed with the best interpretation of the request.",
            persist=False,
        )
        result = await handle.result()

        assert isinstance(result, str) and result.strip()

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_contacts_question")
        assert_tool_called(handle, "primitives.contacts.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
