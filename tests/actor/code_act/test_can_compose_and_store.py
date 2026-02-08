import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.actor.code_act_actor import CodeActActor


# ---------------------------------------------------------------------------
# can_compose=False — symbolic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_code_act_can_compose_false_requires_function_manager():
    """
    can_compose=False without a function_manager should raise RuntimeError
    because there would be no usable tools (no execute_code, no execute_function).
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    # The ManagerRegistry provides a default FM, so override it to None.
    actor.function_manager = None
    try:
        with pytest.raises(RuntimeError, match="function_manager is required"):
            await actor.act("Do something", can_compose=False)
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# can_compose=False — eval tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_can_compose_false_executes_best_matching_function():
    """
    When can_compose=False, the LLM should discover stored functions via
    FunctionManager discovery tools and invoke them via execute_function.
    It must NOT use execute_code.
    """
    _fn_metadata = [
        {
            "function_id": 123,
            "name": "my_task",
            "docstring": "Does the thing requested by the user",
        },
    ]
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm.filter_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm.list_functions = MagicMock(return_value={"metadata": _fn_metadata})
    fm.execute_function = AsyncMock(
        return_value={"result": "OK", "error": None, "stdout": "", "stderr": ""},
    )

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Do the thing",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=60)

        # The LLM should have discovered the function and executed it.
        fm.execute_function.assert_called_once()
        assert fm.execute_function.call_args.kwargs["function_name"] == "my_task"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_can_compose_false_no_functions_match():
    """
    When can_compose=False and no stored functions match the query, the LLM
    should report the failure gracefully without calling execute_function.
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm.execute_function = AsyncMock()

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Do something completely unique",
            can_compose=False,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=60)

        # No matching function to execute.
        fm.execute_function.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# can_store=False
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Description type acceptance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_code_act_accepts_dict_description():
    """
    CodeActActor.act should accept a dict description (passed to async tool loop).
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        # We just verify the call doesn't raise TypeError and creates a handle
        # The handle will run an LLM loop, but we stop it immediately
        handle = await actor.act(
            {"role": "user", "content": "What is 2+2?"},
            persist=False,
            clarification_enabled=False,
        )
        # Verify we got a handle back (not testing the full loop completion)
        assert handle is not None
        # Stop the handle to avoid waiting for LLM
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_code_act_accepts_list_description():
    """
    CodeActActor.act should accept a list description (passed to async tool loop).
    """
    actor = CodeActActor(
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        # We just verify the call doesn't raise TypeError and creates a handle
        handle = await actor.act(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            persist=False,
            clarification_enabled=False,
        )
        # Verify we got a handle back
        assert handle is not None
        # Stop the handle to avoid waiting for LLM
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass
