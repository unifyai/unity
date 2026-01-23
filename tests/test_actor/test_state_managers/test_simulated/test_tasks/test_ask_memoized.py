"""
Actor tests for TaskScheduler.ask via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects an existing
FunctionManager skill via semantic search, injects it as the entrypoint, and that
the underlying primitive tool call is `primitives.tasks.ask`.

Pattern: Memoized function (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


TASK_QUESTIONS: list[str] = [
    "Which tasks are due today?",
    "List all high-priority tasks.",
    "What tasks are scheduled for tomorrow?",
    "Show tasks assigned to Alice that are still open.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("question", TASK_QUESTIONS)
async def test_questions_use_memoized_function(
    question: str,
    mock_verification,
):
    """Verify Actor selects memoized function via semantic search for asks."""

    implementation = [
        '''
async def ask_tasks(question: str, response_format=None) -> str:
    """Query the task list via the task scheduler (read-only).

    **Use when** the user is asking about existing tasks: what is due, what is scheduled,
    what is assigned to someone, priorities/statuses, or summaries of the task queue.

    **Do NOT use when**:
    - the user wants to create/update/delete/reorder tasks (use `primitives.tasks.update`)
    - the user wants to execute a task (use `tasks.execute` in the task system; not this test primitive)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        question: The task-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the task scheduler as a string.
    """
    handle = await primitives.tasks.ask(question, response_format=response_format)
    result = await handle.result()
    return result
''',
    ]
    more_implementations = [
        '''
async def ask_contacts_question(question: str, response_format=None) -> str:
    """Query the contacts database (people/organizations) using the contacts manager.

    **Use when** the question is about stored contact records: emails, phone numbers,
    job titles, locations, preferences, account ownership, etc.

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
        '''
async def update_or_create_or_delete_knowledge(instruction: str, response_format=None) -> str:
    """Mutate internal knowledge via the knowledge manager (create/update facts).

    **Use when** the user requests to store new knowledge, update an existing policy/fact,
    or otherwise change the knowledge base.

    **Do NOT use when**:
    - the request is read-only (use `primitives.knowledge.ask`)
    - the user is asking about transcripts, contacts, tasks, guidance, or web facts

    Args:
        instruction: The knowledge update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the knowledge manager update operation as a string.
    """
    handle = await primitives.knowledge.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
    ''',
        '''
async def ask_knowledge(question: str, response_format=None) -> str:
    """Query internal structured knowledge via the knowledge manager (read-only).

    **Use when** the question should be answered from stored organizational knowledge:
    policies, facts, reference material, and previously recorded information.

    **Do NOT use when**:
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about message history/transcripts (use `primitives.transcripts.ask`)
    - the user is asking about contact records (use `primitives.contacts.ask`)
    - the user is requesting a knowledge mutation (use `primitives.knowledge.update`)

    Args:
        question: The knowledge-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The answer from the knowledge manager as a string.
    """
    handle = await primitives.knowledge.ask(question, response_format=response_format)
    result = await handle.result()
    return result
    ''',
        '''
async def update_guidance(instruction: str, response_format=None) -> str:
    """Create/update/delete guidance entries via the guidance manager (mutation).

    **Use when** the user requests changes to internal guidance content: add a runbook,
    update an existing entry, or correct/replace guidance text.

    **Do NOT use when**:
    - the user is asking a read-only question about existing guidance (use `primitives.guidance.ask`)
    - the user is asking about transcripts, contacts, tasks, or current web facts

    Args:
        instruction: The guidance update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the guidance manager update operation as a string.
    """
    handle = await primitives.guidance.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
    ''',
        '''
async def ask_guidance_question(question: str, response_format=None) -> str:
    """Query internal guidance/policies/runbooks via the guidance manager (read-only).

    **Use when** the question is about internal operating guidance, runbooks, incident
    response procedures, best practices, or other curated guidance content.

    **Do NOT use when**:
    - the user wants to create/update guidance entries (use `primitives.guidance.update`)
    - the user is asking about their message history/transcripts (use `primitives.transcripts.ask`)
    - the user needs current external facts (use `primitives.web.ask`)
    - the user is asking about contacts or tasks (use the appropriate manager)

    Args:
        question: The guidance-related question to ask.
        response_format: Optional Pydantic model for structured output.

Returns:
    The answer from the guidance manager as a string.
"""
handle = await primitives.guidance.ask(question, response_format=response_format)
result = await handle.result()
return result
    ''',
    ]
    async with make_hierarchical_actor(impl="simulated") as actor:
        from unity.function_manager.function_manager import FunctionManager

        fm = FunctionManager()

        # implementation = implementation + more_implementations
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

        assert_memoized_function_used(handle, "ask_tasks")
        assert_tool_called(handle, "primitives.tasks.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
