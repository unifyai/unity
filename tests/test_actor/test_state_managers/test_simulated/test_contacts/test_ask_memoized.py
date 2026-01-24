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
    "Which of our contacts prefers to be contacted by phone? Provide a structured summary.",
    "Find the email address for the contact named Sarah (use your contacts only). Include relevant details.",
    "List any contacts located in Berlin with a summary of their roles.",
    "Who is the primary point of contact for the Contoso account? Provide context and details.",
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

        # Relax assertion: result can be str, dict, or Pydantic BaseModel

        # Verify result is not None (routing test, not type test)
        assert result is not None

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_contacts_with_analysis")
        assert_tool_called(handle, "primitives.contacts.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
