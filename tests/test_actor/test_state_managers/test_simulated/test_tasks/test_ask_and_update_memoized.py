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
        "Which tasks are due tomorrow? Provide a prioritized summary. Also create a new task: Call Alice about the Q3 budget tomorrow at 09:00 and confirm."
    ),
    (
        "List all high-priority tasks with context and deadlines. Also update the priority of 'Draft Budget FY26' to high and confirm the change."
    ),
    (
        "What tasks are assigned to Bob Johnson? Include workload analysis. Also delete the task named 'Old Onboarding Checklist' and confirm."
    ),
    (
        "Summarise tasks scheduled for next week with priorities. Also set 'Prepare slides for kickoff' to start today at 10:00 and provide confirmation."
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
async def ask_tasks_with_analysis(question: str, response_format=None) -> str:
    """Query the task list and produce a structured analysis with priorities and context.

    **ALWAYS use this function** for ANY task-related read-only question, regardless of
    complexity. Direct calls to primitives.tasks.ask are not allowed when this function
    is available - even for simple lookups like "What tasks are due today?".

    This helper does two steps:
    1) Retrieves task information via primitives.tasks.ask
    2) Synthesizes a structured analysis with priorities via computer_primitives.reason

    **Do NOT use when**:
    - the user wants to create/update/delete/reorder tasks (use tasks update)
    - the user wants to execute a task (use tasks execute)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        question: The task-related question to ask.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A structured analysis of tasks with priorities and context.
    """
    handle = await primitives.tasks.ask(question, response_format=response_format)
    raw_result = await handle.result()

    analysis = await computer_primitives.reason(
        request=(
            "Produce a structured task summary with: "
            "1) Overview (total count, priority breakdown), "
            "2) Key tasks (name, priority, deadline, assignee), "
            "3) Recommendations or observations."
        ),
        context=str(raw_result),
    )
    return analysis if isinstance(analysis, str) else str(analysis)
''',
        '''
async def update_tasks_with_confirmation(instruction: str, response_format=None) -> str:
    """Mutate tasks and produce a confirmation summary of changes.

    **ALWAYS use this function** for ANY task mutation request, regardless of
    complexity. Direct calls to primitives.tasks.update are not allowed when this
    function is available - even for simple updates like "Create task X".

    This helper does two steps:
    1) Performs the task mutation via primitives.tasks.update
    2) Synthesizes a confirmation summary via computer_primitives.reason

    **Do NOT use when**:
    - the user is asking a read-only question about tasks (use tasks ask)
    - the user is asking about contacts/transcripts/guidance/web

    Args:
        instruction: The task update instruction text.
        response_format: Optional Pydantic model for structured output.

    Returns:
        A confirmation summary of the changes made.
    """
    handle = await primitives.tasks.update(instruction, response_format=response_format)
    raw_result = await handle.result()

    confirmation = await computer_primitives.reason(
        request=(
            "Summarize what was changed: "
            "1) Action taken (created/updated/deleted), "
            "2) Task details (name, priority, deadline, assignee), "
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
        from pydantic import BaseModel

        assert result and (
            isinstance(result, (str, dict)) or isinstance(result, BaseModel)
        )

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
            get_state_manager_tools,
        )

        assert_memoized_function_used(handle, "ask_tasks_with_analysis")
        assert_memoized_function_used(handle, "update_tasks_with_confirmation")
        assert_tool_called(handle, "primitives.tasks.ask")
        assert_tool_called(handle, "primitives.tasks.update")

        # Allow additional tools (e.g., verification steps)
        state_manager_tools = set(get_state_manager_tools(handle))
        assert "primitives.tasks.ask" in state_manager_tools
        assert "primitives.tasks.update" in state_manager_tools

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
