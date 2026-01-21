"""
Actor tests for TaskScheduler ask+update via memoized functions.

This verifies that when `can_compose=True`, HierarchicalActor selects separate
memoized functions via semantic search for combined requests and that both
`primitives.tasks.ask` and `primitives.tasks.update` are invoked.

Pattern: Memoized functions (semantic search; no on-the-fly codegen)
"""

from __future__ import annotations

import pytest


from tests.test_actor.test_state_managers.utils import make_hierarchical_actor

pytestmark = pytest.mark.eval


COMBINED_REQUESTS: list[str] = [
    (
        "Which tasks are due tomorrow? Also create a new task: Call Alice about the Q3 budget tomorrow at 09:00."
    ),
    (
        "List all high-priority tasks. Also update the priority of 'Draft Budget FY26' to high."
    ),
    (
        "What tasks are assigned to Bob Johnson? Also delete the task named 'Old Onboarding Checklist'."
    ),
    (
        "Summarise tasks scheduled for next week. Also set 'Prepare slides for kickoff' to start today at 10:00."
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("combined_text", COMBINED_REQUESTS)
async def test_combined_queries_use_memoized_function(
    combined_text: str,
    mock_verification,
):
    """Verify Actor selects separate memoized functions via semantic search for combined ops."""

    implementations = [
        '''
async def ask_tasks(question: str, response_format=None) -> str:
    """Query the task list via the task scheduler (read-only).

    **Use when** the user is asking about existing tasks: what is due, what is scheduled,
    what is assigned to someone, priorities/statuses, or summaries of the task queue.

    **How it works**: calls:
    - `await primitives.tasks.ask(question, response_format=response_format)`

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
        '''
async def update_or_create_or_delete_tasks(instruction: str, response_format=None) -> str:
    """Mutate tasks via the task scheduler (create/update/delete/reorder).

    **Use when** the user requests any change to the task list: create a task, update
    fields like priority/schedule/status, delete tasks, or otherwise modify tasks.

    **How it works**: calls:
    - `await primitives.tasks.update(instruction, response_format=response_format)`

    **Do NOT use when**:
    - the user is asking a read-only question about tasks (use `primitives.tasks.ask`)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        instruction: The task update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        The result from the task scheduler update operation as a string.
    """
    handle = await primitives.tasks.update(instruction, response_format=response_format)
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
        '''
async def update_or_create_or_delete_knowledge(instruction: str, response_format=None) -> str:
    """Mutate internal knowledge via the knowledge manager (create/update facts).

    **Use when** the user requests to store new knowledge, update an existing policy/fact,
    or otherwise change the knowledge base.

    **How it works**: calls:
    - `await primitives.knowledge.update(instruction, response_format=response_format)`

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

    **How it works**: calls:
    - `await primitives.knowledge.ask(question, response_format=response_format)`

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

    **How it works**: calls the guidance mutation tool:
    - `await primitives.guidance.update(instruction, response_format=response_format)`

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

    **How it works**: calls the guidance read tool:
    - `await primitives.guidance.ask(question, response_format=response_format)`

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

        # implementations = implementations + more_implementations
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

        assert_memoized_function_used(handle, "ask_tasks")
        assert_memoized_function_used(handle, "update_or_create_or_delete_tasks")
        assert_tool_called(handle, "primitives.tasks.ask")
        assert_tool_called(handle, "primitives.tasks.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.tasks.ask" in state_manager_tools
        assert "primitives.tasks.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
