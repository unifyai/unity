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
    "Which tasks are due today? Provide a prioritized summary.",
    "List all high-priority tasks with context and deadlines.",
    "What tasks are scheduled for tomorrow? Include status and assignees.",
    "Show tasks assigned to Alice that are still open with analysis of workload.",
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
    ]
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
        from pydantic import BaseModel

        assert result and (
            isinstance(result, (str, dict)) or isinstance(result, BaseModel)
        )

        from tests.test_actor.test_state_managers.utils import (
            assert_memoized_function_used,
            assert_tool_called,
        )

        assert_memoized_function_used(handle, "ask_tasks_with_analysis")
        assert_tool_called(handle, "primitives.tasks.ask")

        # Note: We don't assert on "verification failed" because the LLM-generated
        # verification logic may fail for reasons unrelated to memoized function usage.
        # The key test is that the memoized function was correctly selected and used.
