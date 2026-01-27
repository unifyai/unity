"""
CodeActActor memoized-function routing tests for GuidanceManager.update (simulated managers).

Mirrors `test_update_memoized.py` but validates CodeActActor uses FunctionManager search
and then executes an injected guidance-mutation helper function that calls
`primitives.guidance.update(...)`.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.actor.state_managers.utils import (
    assert_code_act_function_manager_used,
    extract_code_act_execute_code_snippets,
    make_code_act_actor,
    wait_for_recorded_primitives_call,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_code_act_update_uses_memoized_function():
    from unity.function_manager.function_manager import FunctionManager

    implementation = '''
async def update_guidance(instruction: str, response_format=None) -> str:
    """Create/update/delete guidance entries via the guidance manager (mutation)."""
    handle = await primitives.guidance.update(instruction, response_format=response_format)
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
        request_text = (
            "Create a new guidance entry titled 'Runbook: DB Failover' with the content "
            "'Promote replica and update connection strings.'"
        )

        handle = await actor.act(
            f"{request_text} Do not ask clarifying questions. Do not create any stubs. Proceed with the best interpretation of the request.",
            clarification_enabled=False,
        )
        # CodeAct can sometimes keep iterating trying to "verify output" after the mutation.
        # For routing parity, we only need to confirm:
        # 1) FunctionManager search was used
        # 2) primitives.guidance.update was invoked (via the memoized function)
        # Then stop the handle early to keep tests fast and avoid long LLM loops.
        await wait_for_recorded_primitives_call(
            calls,
            "primitives.guidance.update",
            timeout=60.0,
        )
        try:
            await asyncio.wait_for(handle.stop("Routing verified"), timeout=30.0)
        except Exception:
            # Best-effort stop only; assertions below should still be valid.
            pass

        assert_code_act_function_manager_used(handle)

        code_snippets = "\n\n".join(
            extract_code_act_execute_code_snippets(handle),
        )
        assert "update_guidance" in code_snippets, (
            "Expected CodeAct to execute the memoized function in Python code. "
            f"Snippets tail:\n{code_snippets[-800:]}"
        )

        assert "primitives.guidance.update" in set(calls), f"Calls seen: {calls}"
