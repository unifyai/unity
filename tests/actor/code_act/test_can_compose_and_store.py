import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_code_act_can_compose_false_executes_best_matching_function():
    """
    When can_compose=False, CodeActActor should avoid the LLM tool loop and instead:
    - semantic search for a best-match stored function
    - execute it via FunctionManager.execute_function
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(
        return_value=[{"function_id": 123, "name": "my_task", "docstring": "do thing"}],
    )
    fm.execute_function = AsyncMock(
        return_value={"result": "OK", "error": None, "stdout": "", "stderr": ""},
    )

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        handle = await actor.act("Do the thing", can_compose=False, persist=False)
        res = await asyncio.wait_for(handle.result(), timeout=30)
        assert res == "OK"

        fm.search_functions.assert_called_once()
        assert fm.search_functions.call_args.kwargs["query"] == "Do the thing"
        fm.execute_function.assert_called_once()
        assert fm.execute_function.call_args.kwargs["function_name"] == "my_task"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_code_act_can_compose_false_errors_when_no_functions_match():
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value=[])
    fm.execute_function = AsyncMock()

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        handle = await actor.act(
            "Do something completely unique",
            can_compose=False,
            persist=False,
        )
        out = await asyncio.wait_for(handle.result(), timeout=30)
        assert "Error:" in str(out)
        assert "no matching functions" in str(out).lower()
        fm.execute_function.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_can_store_false_blocks_add_functions_tool():
    """
    When can_store=False, the FunctionManager_add_functions tool should not be available.
    We validate this by instructing the agent to call it; the loop should fail gracefully
    rather than executing the tool.
    """
    fm = MagicMock()
    fm.add_functions = MagicMock(return_value={"x": "added"})

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    try:
        handle = await actor.act(
            "Call the tool FunctionManager_add_functions with implementations='async def x():\\n    return 1'. "
            "Do not call execute_code.",
            can_store=False,
            persist=False,
            clarification_enabled=False,
        )
        out = await asyncio.wait_for(handle.result(), timeout=60)
        # The tool should be unavailable; we accept any clear failure surface.
        assert "FunctionManager_add_functions" in str(out)
        fm.add_functions.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass
