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
@pytest.mark.timeout(300)
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
@pytest.mark.timeout(300)
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
        environments=[],
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
@pytest.mark.timeout(300)
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
# can_store=True — verify storing actually works
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_can_store_true_stores_function():
    """
    When can_store=True (the default), the LLM should be able to compose a
    function via execute_code, then store it via FunctionManager_add_functions
    for future reuse.

    The function must be complex enough that the LLM consistently considers
    it worth storing (non-trivial logic, validation, edge cases).
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm.add_functions = MagicMock(return_value={"normalize_address": "added"})

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Write a reusable Python function called `normalize_address` that:\n"
            "1. Takes a dict with optional keys: street, city, state, zip_code, country\n"
            "2. Strips whitespace and title-cases street/city/country\n"
            "3. Uppercases state and validates it is exactly 2 letters (or raises ValueError)\n"
            "4. Normalizes zip_code: strip to digits only, pad to 5 digits with leading zeros\n"
            "5. Returns a new dict with all normalized fields, plus a 'formatted' key\n"
            "   containing a single-line string like '123 Main St, Austin, TX 73301, US'\n"
            "6. Handle edge cases: missing keys default to empty string, None values\n\n"
            "Test it with an address that has messy whitespace and a 4-digit zip, "
            "then store it using FunctionManager_add_functions for future reuse.",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=60)

        # The LLM should have called add_functions with the implementation.
        fm.add_functions.assert_called_once()
        call_kwargs = fm.add_functions.call_args.kwargs
        impl = call_kwargs.get("implementations", "")
        assert "normalize_address" in str(
            impl,
        ), f"Expected 'normalize_address' in implementation, got: {impl}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# storage_check_on_return=True — eval tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_storage_check_on_return_stores_discovered_function():
    """
    When storage_check_on_return=True, the CodeActActor should compose and
    execute code, then run a storage-check loop that reviews the trajectory
    and stores reusable functions via FunctionManager_add_functions.
    result() blocks until both phases complete.

    The function must be complex enough that the librarian LLM consistently
    judges it as worth storing (non-trivial logic, validation, edge cases).
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm.add_functions = MagicMock(return_value={"parse_and_validate_contacts": "added"})

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Write a reusable Python function called `parse_and_validate_contacts` that:\n"
            "1. Takes a list of dicts, each with optional keys: name, email, phone, company\n"
            "2. Validates each entry: name must be non-empty string, email must contain '@',\n"
            "   phone (if present) must be digits/dashes/spaces only and at least 7 chars\n"
            "3. Returns a dict with keys:\n"
            "   - 'valid': list of cleaned entries (strip whitespace, normalize phone to digits-only)\n"
            "   - 'invalid': list of (index, entry, errors) tuples describing validation failures\n"
            "   - 'stats': dict with counts of total, valid, invalid, and entries_with_company\n"
            "4. Handle edge cases: None inputs, empty lists, entries that are not dicts\n\n"
            "Test it with a mixed list of 5 entries including at least 2 invalid ones "
            "and verify the stats are correct.",
            storage_check_on_return=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)

        # result() blocks until both the task and the storage check complete.
        assert result is not None

        fm.add_functions.assert_called()
        call_kwargs = fm.add_functions.call_args.kwargs
        impl = str(call_kwargs.get("implementations", ""))
        assert (
            "parse_and_validate_contacts" in impl
        ), f"Expected 'parse_and_validate_contacts' in stored implementation, got: {impl}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# storage_check_on_return — skip cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_storage_check_on_return_skipped_when_can_store_false():
    """
    storage_check_on_return=True with can_store=False should NOT wrap
    the handle in a _StorageCheckHandle. The result returns normally
    without any storage-check phase.
    """
    fm = MagicMock()
    fm.search_functions = MagicMock(return_value={"metadata": []})
    fm.filter_functions = MagicMock(return_value={"metadata": []})
    fm.list_functions = MagicMock(return_value={"metadata": []})
    fm.add_functions = MagicMock(return_value={})

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=30,
    )
    try:
        handle = await actor.act(
            "What is 2 + 2?",
            storage_check_on_return=True,
            can_store=False,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        assert result is not None

        # No storage check should have run.
        fm.add_functions.assert_not_called()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# storage_check_on_return — reorganization (merge + delete)
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_storage_check_on_return_merges_redundant_functions():
    """
    The storage check loop should recognise overlapping functions in the
    store and merge them: add a unified version and delete the old ones.

    Setup: the FunctionManager already contains two narrow greeting
    functions (greet_formal, greet_casual). The actor composes and
    executes a general-purpose `greet` function that subsumes both.
    The storage check should detect the overlap, store the merged
    version, and delete the now-redundant entries.

    result() blocks until both the task and storage-check phases complete.
    """
    _existing_functions = [
        {
            "function_id": 101,
            "name": "greet_formal",
            "docstring": "Return a formal greeting.",
            "implementation": (
                "def greet_formal(name):\n" '    return f"Good day, {name}."'
            ),
        },
        {
            "function_id": 102,
            "name": "greet_casual",
            "docstring": "Return a casual greeting.",
            "implementation": ("def greet_casual(name):\n" '    return f"Hey {name}!"'),
        },
    ]

    fm = MagicMock()
    # Discovery tools return the two existing overlapping functions.
    fm.search_functions = MagicMock(return_value={"metadata": _existing_functions})
    fm.filter_functions = MagicMock(return_value={"metadata": _existing_functions})
    fm.list_functions = MagicMock(
        return_value={"metadata": _existing_functions},
    )
    fm.add_functions = MagicMock(return_value={"greet": "added"})
    fm.delete_function = MagicMock(
        return_value={"greet_formal": "deleted", "greet_casual": "deleted"},
    )

    actor = CodeActActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        timeout=60,
    )
    try:
        handle = await actor.act(
            "Write a general-purpose Python function called `greet` that takes "
            "`name` and `style` ('formal' or 'casual') parameters. "
            "For formal: return f'Good day, {name}.'; "
            "for casual: return f'Hey {name}!'. "
            "Execute it with name='Alice' and style='formal' to verify.",
            storage_check_on_return=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None

        # The merged function should have been stored.
        fm.add_functions.assert_called()
        add_kwargs = fm.add_functions.call_args.kwargs
        impl = str(add_kwargs.get("implementations", ""))
        assert (
            "greet" in impl
        ), f"Expected 'greet' in stored implementation, got: {impl}"

        # The old redundant functions should have been deleted.
        fm.delete_function.assert_called()
        delete_kwargs = fm.delete_function.call_args.kwargs
        deleted_ids = delete_kwargs.get("function_id", [])
        if isinstance(deleted_ids, int):
            deleted_ids = [deleted_ids]
        assert set(deleted_ids) & {
            101,
            102,
        }, f"Expected deletion of function_ids 101 and/or 102, got: {deleted_ids}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Description type acceptance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(300)
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
@pytest.mark.timeout(300)
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
