"""
CodeActActor memoized-function routing tests for KnowledgeManager.update (simulated managers).

Validates CodeActActor uses FunctionManager search and then executes an injected knowledge-mutation
helper function that calls `primitives.knowledge.update(...)`.
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
    "Store: Office hours are 9–5 PT.",
    "Update the onboarding policy to require security training in week one.",
]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@pytest.mark.parametrize("request_text", UPDATE_QUERIES)
async def test_code_act_updates_use_memoized_function(
    request_text: str,
):
    from unity.function_manager.function_manager import FunctionManager

    implementation = '''
async def update_or_create_or_delete_knowledge(instruction: str, response_format=None) -> str:
    """Mutate knowledge via the knowledge manager (create/update facts)."""
    handle = await primitives.knowledge.update(instruction, response_format=response_format)
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
        assert isinstance(result, str) and result.strip()

        assert_code_act_function_manager_used(handle)

        code_snippets = "\n\n".join(
            extract_code_act_execute_code_snippets(handle),
        )
        assert "update_or_create_or_delete_knowledge" in code_snippets, (
            "Expected CodeAct to execute the memoized function in Python code. "
            f"Snippets tail:\n{code_snippets[-800:]}"
        )

        assert "primitives.knowledge.update" in set(calls), f"Calls seen: {calls}"
