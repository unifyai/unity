"""
Tests for _VenvFunctionProxy state mode API (.stateless(), .read_only()).

These tests verify that the proxy-based API for controlling execution state
works correctly when calling venv functions from CodeActActor sandbox.

Covers:
1. Default stateful behavior (state persists)
2. .stateless() mode (fresh environment, no inherited state)
3. .read_only() mode (reads state but doesn't persist changes)
4. Error handling (read_only without pool)
5. Fallback behavior (stateful without pool)
"""

from __future__ import annotations

import shutil

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import (
    FunctionManager,
    VenvPool,
    _VenvFunctionProxy,
)
from unity.common.context_registry import ContextRegistry


# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SET_VAR_FUNC = """
async def set_var(value):
    global my_var
    my_var = value
    return f"Set my_var to {value}"
""".strip()

GET_VAR_FUNC = """
async def get_var():
    return my_var
""".strip()

CHECK_VAR_FUNC = """
async def check_var():
    try:
        return my_var
    except NameError:
        return "NOT_DEFINED"
""".strip()

INCREMENT_FUNC = """
async def increment():
    global counter
    try:
        counter
    except NameError:
        counter = 0
    counter += 1
    return counter
""".strip()

# Minimal venv content
MINIMAL_VENV_CONTENT = """
[project]
name = "test-proxy-state-modes"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def _create_prepared_venv(fm: FunctionManager) -> int:
    """Create and prepare a minimal venv. Returns the venv_id."""
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    await fm.prepare_venv(venv_id=venv_id)
    return venv_id


def _cleanup_venv(fm: FunctionManager, venv_id: int) -> None:
    """Cleanup a venv directory after test."""
    try:
        venv_dir = fm._get_venv_dir(venv_id)
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
    except Exception:
        pass


def _create_proxy(
    fm: FunctionManager,
    func_data: dict,
    namespace: dict,
) -> _VenvFunctionProxy:
    """Create a _VenvFunctionProxy for testing."""
    return _VenvFunctionProxy(
        function_manager=fm,
        func_data=func_data,
        namespace=namespace,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Default Stateful Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateful_default_persists_state(function_manager_factory):
    """Default await func() should persist state across calls (stateful mode)."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Create namespace with venv pool (simulates CodeActActor sandbox)
        namespace = {"__venv_pool__": pool}

        # Create proxies for set and get functions
        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set a variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get the variable",
            },
            namespace,
        )

        # Call set_var (default stateful)
        result1 = await set_proxy(value=42)
        assert "42" in result1

        # Call get_var (default stateful) - should see persisted state
        result2 = await get_proxy()
        assert result2 == 42
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateful_counter_increments(function_manager_factory):
    """Counter should increment across stateful calls."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}
        increment_proxy = _create_proxy(
            fm,
            {
                "name": "increment",
                "venv_id": venv_id,
                "implementation": INCREMENT_FUNC,
                "docstring": "Increment counter",
            },
            namespace,
        )

        # Multiple stateful calls should see incrementing counter
        assert await increment_proxy() == 1
        assert await increment_proxy() == 2
        assert await increment_proxy() == 3
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 2. Stateless Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateless_provides_isolation(function_manager_factory):
    """await func.stateless() should not inherit or persist state."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}

        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set variable",
            },
            namespace,
        )
        check_proxy = _create_proxy(
            fm,
            {
                "name": "check_var",
                "venv_id": venv_id,
                "implementation": CHECK_VAR_FUNC,
                "docstring": "Check variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get variable",
            },
            namespace,
        )

        # Set state via stateful call
        await set_proxy(value=100)

        # Stateless call should NOT see the state
        result = await check_proxy.stateless()
        assert result == "NOT_DEFINED"

        # Stateful call should still see original state
        result = await get_proxy()
        assert result == 100
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateless_does_not_affect_stateful_state(
    function_manager_factory,
):
    """Stateless execution should not modify stateful session state."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}

        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get variable",
            },
            namespace,
        )

        # Set initial state to 42 (stateful)
        await set_proxy(value=42)

        # Try to change state via stateless call
        await set_proxy.stateless(value=999)

        # Stateful state should be unchanged
        result = await get_proxy()
        assert result == 42  # Still 42, not 999
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateless_counter_always_starts_at_one(
    function_manager_factory,
):
    """Stateless calls should always start fresh (counter = 1)."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}
        increment_proxy = _create_proxy(
            fm,
            {
                "name": "increment",
                "venv_id": venv_id,
                "implementation": INCREMENT_FUNC,
                "docstring": "Increment counter",
            },
            namespace,
        )

        # Stateful calls increment
        assert await increment_proxy() == 1
        assert await increment_proxy() == 2

        # Stateless call starts fresh
        assert await increment_proxy.stateless() == 1

        # Stateful state unchanged
        assert await increment_proxy() == 3
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 3. Read-Only Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_proxy_read_only_reads_state(function_manager_factory):
    """await func.read_only() should be able to read existing state."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}

        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get variable",
            },
            namespace,
        )

        # Set state via stateful call
        await set_proxy(value=100)

        # Read-only call should see the state
        result = await get_proxy.read_only()
        assert result == 100
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_read_only_does_not_persist_changes(function_manager_factory):
    """await func.read_only() should not persist changes to state."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}

        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get variable",
            },
            namespace,
        )

        # Set initial state to 100
        await set_proxy(value=100)

        # Modify state in read_only mode (should execute but not persist)
        result_modify = await set_proxy.read_only(value=999)
        assert "999" in result_modify

        # Stateful state should be unchanged
        result_check = await get_proxy()
        assert result_check == 100  # Still 100, not 999
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_read_only_counter_reads_but_doesnt_persist(
    function_manager_factory,
):
    """Read-only should see counter value but not persist increment."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}
        increment_proxy = _create_proxy(
            fm,
            {
                "name": "increment",
                "venv_id": venv_id,
                "implementation": INCREMENT_FUNC,
                "docstring": "Increment counter",
            },
            namespace,
        )

        # Stateful calls increment to 2
        assert await increment_proxy() == 1
        assert await increment_proxy() == 2

        # Read-only call increments from 2 to 3 (but doesn't persist)
        assert await increment_proxy.read_only() == 3

        # Stateful call continues from 2 (not 3)
        assert await increment_proxy() == 3
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 4. Error Handling Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_proxy_read_only_requires_pool(function_manager_factory):
    """read_only mode should raise error if no VenvPool is available."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    try:
        # No __venv_pool__ in namespace
        namespace = {}

        check_proxy = _create_proxy(
            fm,
            {
                "name": "check_var",
                "venv_id": venv_id,
                "implementation": CHECK_VAR_FUNC,
                "docstring": "Check variable",
            },
            namespace,
        )

        with pytest.raises(ValueError, match="read_only mode.*requires a VenvPool"):
            await check_proxy.read_only()
    finally:
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateful_fallback_without_pool(function_manager_factory):
    """Stateful mode should fall back to one-shot execution without pool."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    try:
        # No __venv_pool__ in namespace
        namespace = {}

        increment_proxy = _create_proxy(
            fm,
            {
                "name": "increment",
                "venv_id": venv_id,
                "implementation": INCREMENT_FUNC,
                "docstring": "Increment counter",
            },
            namespace,
        )

        # Without pool, each "stateful" call is actually one-shot (no persistence)
        assert await increment_proxy() == 1
        assert await increment_proxy() == 1  # Starts fresh each time
        assert await increment_proxy() == 1
    finally:
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_proxy_stateless_works_without_pool(function_manager_factory):
    """Stateless mode should work fine without VenvPool."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    try:
        namespace = {}

        increment_proxy = _create_proxy(
            fm,
            {
                "name": "increment",
                "venv_id": venv_id,
                "implementation": INCREMENT_FUNC,
                "docstring": "Increment counter",
            },
            namespace,
        )

        # Stateless always starts fresh - works without pool
        assert await increment_proxy.stateless() == 1
        assert await increment_proxy.stateless() == 1
        assert await increment_proxy.stateless() == 1
    finally:
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 5. Mixed Mode Workflow Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_proxy_mixed_mode_workflow(function_manager_factory):
    """Complex workflow mixing stateful, stateless, and read_only modes."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        namespace = {"__venv_pool__": pool}

        set_proxy = _create_proxy(
            fm,
            {
                "name": "set_var",
                "venv_id": venv_id,
                "implementation": SET_VAR_FUNC,
                "docstring": "Set variable",
            },
            namespace,
        )
        get_proxy = _create_proxy(
            fm,
            {
                "name": "get_var",
                "venv_id": venv_id,
                "implementation": GET_VAR_FUNC,
                "docstring": "Get variable",
            },
            namespace,
        )
        check_proxy = _create_proxy(
            fm,
            {
                "name": "check_var",
                "venv_id": venv_id,
                "implementation": CHECK_VAR_FUNC,
                "docstring": "Check variable",
            },
            namespace,
        )

        # Step 1: Set initial state (stateful)
        await set_proxy(value=10)

        # Step 2: Read state (stateful) - should see 10
        assert await get_proxy() == 10

        # Step 3: Stateless check - should NOT see state
        assert await check_proxy.stateless() == "NOT_DEFINED"

        # Step 4: Read-only modification - should see state but not persist
        await set_proxy.read_only(value=999)

        # Step 5: Verify state unchanged
        assert await get_proxy() == 10

        # Step 6: Actually modify state (stateful)
        await set_proxy(value=20)

        # Step 7: Verify new state
        assert await get_proxy() == 20

        # Step 8: Read-only should see new state
        assert await get_proxy.read_only() == 20
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)
