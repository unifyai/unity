"""
Tests for state_mode parameter in execute_function.

Covers three modes:
- stateful: State persists across calls (Jupyter-notebook style)
- read_only: Reads existing state but doesn't persist changes
- stateless: Fresh environment each time (pure function behavior)
"""

from __future__ import annotations

import shutil

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager, VenvPool
from unity.common.context_registry import ContextRegistry


# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SET_VAR_FUNC = """
def set_var(value):
    global my_var
    my_var = value
    return f"Set my_var to {value}"
""".strip()

GET_VAR_FUNC = """
def get_var():
    return my_var
""".strip()

CHECK_VAR_FUNC = """
def check_var():
    try:
        return my_var
    except NameError:
        return "NOT_DEFINED"
""".strip()

SIMPLE_FUNC = """
def simple_func():
    return "hello"
""".strip()

SET_PRIMITIVES_FUNC = """
def set_primitives():
    global int_var, float_var, str_var, bool_var, none_var
    int_var = 42
    float_var = 3.14
    str_var = "hello"
    bool_var = True
    none_var = None
    return "done"
""".strip()

GET_PRIMITIVES_FUNC = """
def get_primitives():
    return {
        "int": int_var,
        "float": float_var,
        "str": str_var,
        "bool": bool_var,
        "none": none_var,
    }
""".strip()

SET_COLLECTIONS_FUNC = """
def set_collections():
    global list_var, dict_var, nested_var
    list_var = [1, 2, 3, "four"]
    dict_var = {"a": 1, "b": 2}
    nested_var = {"items": [{"x": 1}, {"x": 2}]}
    return "done"
""".strip()

GET_COLLECTIONS_FUNC = """
def get_collections():
    return {
        "list": list_var,
        "dict": dict_var,
        "nested": nested_var,
    }
""".strip()

# Minimal venv content
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
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


@pytest.fixture
def venv_pool_factory():
    """Factory fixture that creates VenvPool instances and tracks cleanup."""
    pools = []

    def _create():
        pool = VenvPool()
        pools.append(pool)
        return pool

    yield _create

    # Cleanup is handled in tests since we need async


# ────────────────────────────────────────────────────────────────────────────
# Stateful Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_stateful_mode_persists_state(
    function_manager_factory,
    venv_pool_factory,
):
    """Stateful mode should persist variables across function calls."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions (no venv by default, we'll use target_venv_id override)
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Execute set_var in stateful mode with venv override
        result1 = await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 42},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )
        assert result1["error"] is None, f"Unexpected error: {result1['error']}"
        assert "42" in result1["result"]

        # Execute get_var in stateful mode - should see the persisted value
        result2 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )
        assert result2["error"] is None, f"Unexpected error: {result2['error']}"
        assert result2["result"] == 42
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_stateful_mode_requires_venv_pool(function_manager_factory):
    """Stateful mode should raise error if venv_pool is not provided."""
    fm = function_manager_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add function
        fm.add_functions(implementations=SIMPLE_FUNC, language="python")

        with pytest.raises(
            ValueError,
            match="state_mode='stateful' requires venv_pool",
        ):
            await fm.execute_function(
                function_name="simple_func",
                target_venv_id=venv_id,
                state_mode="stateful",
                venv_pool=None,
            )
    finally:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Read-Only Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_read_only_mode_reads_state(function_manager_factory, venv_pool_factory):
    """Read-only mode should be able to read existing state."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # First, set state using stateful mode
        result1 = await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 100},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )
        assert result1["error"] is None

        # Now read state using read_only mode - should see the value
        result2 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="read_only",
            venv_pool=pool,
        )
        assert result2["error"] is None, f"Unexpected error: {result2['error']}"
        assert result2["result"] == 100
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_read_only_mode_does_not_persist_changes(
    function_manager_factory,
    venv_pool_factory,
):
    """Read-only mode should not persist changes to state."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set initial state to 100
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 100},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Try to change state in read_only mode (should execute but not persist)
        result_modify = await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 999},
            target_venv_id=venv_id,
            state_mode="read_only",
            venv_pool=pool,
        )
        assert result_modify["error"] is None
        assert "999" in result_modify["result"]

        # Verify original state is unchanged (still 100, not 999)
        result_check = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )
        assert result_check["error"] is None
        assert result_check["result"] == 100  # Original value preserved
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_read_only_mode_requires_venv_pool(function_manager_factory):
    """Read-only mode should raise error if venv_pool is not provided."""
    fm = function_manager_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add function
        fm.add_functions(implementations=SIMPLE_FUNC, language="python")

        with pytest.raises(
            ValueError,
            match="state_mode='read_only' requires venv_pool",
        ):
            await fm.execute_function(
                function_name="simple_func",
                target_venv_id=venv_id,
                state_mode="read_only",
                venv_pool=None,
            )
    finally:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Stateless Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_stateless_mode_fresh_each_time(
    function_manager_factory,
    venv_pool_factory,
):
    """Stateless mode should not inherit any state from previous calls."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=CHECK_VAR_FUNC, language="python")

        # Set state in stateful mode
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 42},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Check in stateless mode - should NOT see the variable
        result = await fm.execute_function(
            function_name="check_var",
            target_venv_id=venv_id,
            state_mode="stateless",
            # venv_pool not required for stateless
        )
        assert result["error"] is None
        assert result["result"] == "NOT_DEFINED"
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_stateless_mode_no_venv_pool_required(function_manager_factory):
    """Stateless mode should work without venv_pool."""
    fm = function_manager_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add function
        fm.add_functions(implementations=SIMPLE_FUNC, language="python")

        # Should work without venv_pool
        result = await fm.execute_function(
            function_name="simple_func",
            target_venv_id=venv_id,
            state_mode="stateless",
            venv_pool=None,  # Explicitly None
        )
        assert result["error"] is None
        assert result["result"] == "hello"
    finally:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_stateless_mode_does_not_affect_stateful_state(
    function_manager_factory,
    venv_pool_factory,
):
    """Stateless execution should not affect stateful session state."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set initial state to 42 (stateful)
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 42},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Execute something in stateless mode that tries to change state
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 999},
            target_venv_id=venv_id,
            state_mode="stateless",
        )

        # Stateful state should be unchanged
        result = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )
        assert result["error"] is None
        assert result["result"] == 42  # Still 42, not 999
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# State Serialization Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_state_serialization_primitives(
    function_manager_factory,
    venv_pool_factory,
):
    """State serialization should handle primitive types correctly."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_PRIMITIVES_FUNC, language="python")
        fm.add_functions(implementations=GET_PRIMITIVES_FUNC, language="python")

        # Set state in stateful mode
        await fm.execute_function(
            function_name="set_primitives",
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Read state in read_only mode (tests serialization/deserialization)
        result = await fm.execute_function(
            function_name="get_primitives",
            target_venv_id=venv_id,
            state_mode="read_only",
            venv_pool=pool,
        )
        assert result["error"] is None, f"Unexpected error: {result['error']}"
        data = result["result"]
        assert data["int"] == 42
        assert data["float"] == 3.14
        assert data["str"] == "hello"
        assert data["bool"] is True
        assert data["none"] is None
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_state_serialization_collections(
    function_manager_factory,
    venv_pool_factory,
):
    """State serialization should handle collection types correctly."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_COLLECTIONS_FUNC, language="python")
        fm.add_functions(implementations=GET_COLLECTIONS_FUNC, language="python")

        # Set state in stateful mode
        await fm.execute_function(
            function_name="set_collections",
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Read state in read_only mode
        result = await fm.execute_function(
            function_name="get_collections",
            target_venv_id=venv_id,
            state_mode="read_only",
            venv_pool=pool,
        )
        assert result["error"] is None, f"Unexpected error: {result['error']}"
        data = result["result"]
        assert data["list"] == [1, 2, 3, "four"]
        assert data["dict"] == {"a": 1, "b": 2}
        assert data["nested"] == {"items": [{"x": 1}, {"x": 2}]}
    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Default Environment Tests (no venv)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_no_venv_ignores_state_mode(function_manager_factory):
    """Functions without venv should execute regardless of state_mode."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SIMPLE_FUNC, language="python")

    # All state modes should work (state_mode is ignored for no-venv functions)
    for mode in ["stateful", "read_only", "stateless"]:
        result = await fm.execute_function(
            function_name="simple_func",
            state_mode=mode,
            # No venv, no target_venv_id, no venv_pool needed
        )
        assert result["error"] is None
        assert result["result"] == "hello"
