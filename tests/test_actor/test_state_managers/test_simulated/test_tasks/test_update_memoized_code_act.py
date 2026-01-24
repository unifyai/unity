"""
CodeActActor memoized-function routing tests for TaskScheduler.update (simulated managers).

Mirrors `test_update_memoized.py` but validates CodeActActor uses FunctionManager search
and then executes an injected task-mutation helper function that calls `primitives.tasks.update(...)`.
"""

from __future__ import annotations

import pytest

from tests.test_actor.test_state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    make_code_act_actor,
)

pytestmark = pytest.mark.eval


UPDATE_QUERIES: list[str] = [
    "Create a new task: Call Alice about the Q3 budget tomorrow at 09:00.",
    "Delete the task named 'Old Onboarding Checklist'.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_code_act_updates_use_memoized_function(
    request_text: str,
):
    from unity.function_manager.function_manager import FunctionManager

    implementation = '''
async def update_or_create_or_delete_tasks(instruction: str, response_format=None) -> str:
    """Mutate tasks via the task scheduler (create/update/delete)."""
    handle = await primitives.tasks.update(instruction, response_format=response_format)
    result = await handle.result()
    return result
'''

    fm = FunctionManager()
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="simulated",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        result = await handle.result()
        # Verify result is not None (routing test, not type test)
        assert result is not None

        assert_code_act_function_manager_used(handle)

        code_snippets = "\n\n".join(
            extract_code_act_execute_code_snippets(handle),
        )
        assert "update_or_create_or_delete_tasks" in code_snippets, (
            "Expected CodeAct to execute the memoized function in Python code. "
            f"Snippets tail:\n{code_snippets[-800:]}"
        )

        assert "primitives.tasks.update" in set(calls), f"Calls seen: {calls}"
