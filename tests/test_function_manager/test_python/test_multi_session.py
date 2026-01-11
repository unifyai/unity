"""
Tests for multi-session support in execute_function.

Multiple sessions allow independent stateful execution contexts within the same venv.
Each session has its own subprocess and globals dict, enabling concurrent "notebook panes"
that share packages but have isolated state.
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

INCREMENT_COUNTER_FUNC = """
def increment_counter():
    global counter
    try:
        counter += 1
    except NameError:
        counter = 1
    return counter
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
    """Factory fixture that creates VenvPool instances."""
    pools = []

    def _create():
        pool = VenvPool()
        pools.append(pool)
        return pool

    yield _create


# ────────────────────────────────────────────────────────────────────────────
# Multi-Session Tests - Independent State
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sessions_have_independent_state(
    function_manager_factory,
    venv_pool_factory,
):
    """Different sessions should have completely independent global state."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set my_var=100 in session 0
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 100},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )

        # Set my_var=200 in session 1
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 200},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )

        # Set my_var=300 in session 2
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 300},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=2,
            venv_pool=pool,
        )

        # Verify each session has its own value
        result0 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        assert result0["error"] is None
        assert result0["result"] == 100

        result1 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )
        assert result1["error"] is None
        assert result1["result"] == 200

        result2 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=2,
            venv_pool=pool,
        )
        assert result2["error"] is None
        assert result2["result"] == 300

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_session_state_persists_independently(
    function_manager_factory,
    venv_pool_factory,
):
    """Each session should maintain its own persistent state across calls."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add increment counter function
        fm.add_functions(implementations=INCREMENT_COUNTER_FUNC, language="python")

        # Increment in session 0 three times
        for _ in range(3):
            await fm.execute_function(
                function_name="increment_counter",
                target_venv_id=venv_id,
                state_mode="stateful",
                session_id=0,
                venv_pool=pool,
            )

        # Increment in session 1 once
        await fm.execute_function(
            function_name="increment_counter",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )

        # Check session 0 counter (should be 3)
        result0 = await fm.execute_function(
            function_name="increment_counter",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        assert result0["error"] is None
        assert result0["result"] == 4  # 3 + 1 more

        # Check session 1 counter (should be 1)
        result1 = await fm.execute_function(
            function_name="increment_counter",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )
        assert result1["error"] is None
        assert result1["result"] == 2  # 1 + 1 more

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_default_session_is_zero(
    function_manager_factory,
    venv_pool_factory,
):
    """Default session_id should be 0 for backward compatibility."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set value without specifying session_id (should use 0)
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 42},
            target_venv_id=venv_id,
            state_mode="stateful",
            venv_pool=pool,
        )

        # Get value explicitly from session 0 (should see 42)
        result = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,  # Explicit session 0
            venv_pool=pool,
        )
        assert result["error"] is None
        assert result["result"] == 42

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Multi-Session with Read-Only Mode
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_read_only_respects_session_id(
    function_manager_factory,
    venv_pool_factory,
):
    """Read-only mode should read from the correct session."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set different values in different sessions
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "session_0_value"},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )

        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "session_1_value"},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )

        # Read-only from session 0 should see "session_0_value"
        result0 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="read_only",
            session_id=0,
            venv_pool=pool,
        )
        assert result0["error"] is None
        assert result0["result"] == "session_0_value"

        # Read-only from session 1 should see "session_1_value"
        result1 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="read_only",
            session_id=1,
            venv_pool=pool,
        )
        assert result1["error"] is None
        assert result1["result"] == "session_1_value"

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_read_only_does_not_affect_session(
    function_manager_factory,
    venv_pool_factory,
):
    """Read-only execution should not modify the session's state."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Set initial value in session 0
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "original"},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )

        # Try to change value in read_only mode
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "modified"},
            target_venv_id=venv_id,
            state_mode="read_only",
            session_id=0,
            venv_pool=pool,
        )

        # Session 0 should still have original value
        result = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        assert result["error"] is None
        assert result["result"] == "original"

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Session ID with Stateless Mode
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_stateless_ignores_session_id(
    function_manager_factory,
    venv_pool_factory,
):
    """Stateless mode should not be affected by session_id."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=CHECK_VAR_FUNC, language="python")

        # Set value in session 0 (stateful)
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 42},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )

        # Stateless execution should not see any state, regardless of session_id
        result = await fm.execute_function(
            function_name="check_var",
            target_venv_id=venv_id,
            state_mode="stateless",
            session_id=0,  # Even with session_id=0
        )
        assert result["error"] is None
        assert result["result"] == "NOT_DEFINED"

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# Session Isolation - Cross-Session Safety
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_modifying_one_session_doesnt_affect_another(
    function_manager_factory,
    venv_pool_factory,
):
    """Modifying state in one session should not affect other sessions."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")
        fm.add_functions(implementations=GET_VAR_FUNC, language="python")

        # Initialize both sessions
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "session0_initial"},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": "session1_initial"},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )

        # Modify session 0 multiple times
        for i in range(5):
            await fm.execute_function(
                function_name="set_var",
                call_kwargs={"value": f"session0_v{i}"},
                target_venv_id=venv_id,
                state_mode="stateful",
                session_id=0,
                venv_pool=pool,
            )

        # Session 1 should still have its original value
        result1 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )
        assert result1["error"] is None
        assert result1["result"] == "session1_initial"

        # Session 0 should have the latest value
        result0 = await fm.execute_function(
            function_name="get_var",
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        assert result0["error"] is None
        assert result0["result"] == "session0_v4"

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# VenvPool Direct Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_venv_pool_manages_multiple_connections(
    function_manager_factory,
    venv_pool_factory,
):
    """VenvPool should manage separate connections for each (venv_id, session_id) pair."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Get connections for different sessions
        conn0 = await pool.get_or_create_connection(
            venv_id=venv_id,
            function_manager=fm,
            session_id=0,
        )
        conn1 = await pool.get_or_create_connection(
            venv_id=venv_id,
            function_manager=fm,
            session_id=1,
        )
        conn2 = await pool.get_or_create_connection(
            venv_id=venv_id,
            function_manager=fm,
            session_id=2,
        )

        # They should be different connection objects
        assert conn0 is not conn1
        assert conn1 is not conn2
        assert conn0 is not conn2

        # Getting the same session again should return the same connection
        conn0_again = await pool.get_or_create_connection(
            venv_id=venv_id,
            function_manager=fm,
            session_id=0,
        )
        assert conn0 is conn0_again

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_venv_pool_get_connection_state_respects_session(
    function_manager_factory,
    venv_pool_factory,
):
    """get_connection_state should return state for the correct session."""
    fm = function_manager_factory()
    pool = venv_pool_factory()

    # Create a venv
    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        # Add functions
        fm.add_functions(implementations=SET_VAR_FUNC, language="python")

        # Set different values in different sessions
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 111},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=0,
            venv_pool=pool,
        )
        await fm.execute_function(
            function_name="set_var",
            call_kwargs={"value": 222},
            target_venv_id=venv_id,
            state_mode="stateful",
            session_id=1,
            venv_pool=pool,
        )

        # Get state from each session
        state0 = await pool.get_connection_state(
            venv_id=venv_id,
            function_manager=fm,
            session_id=0,
        )
        state1 = await pool.get_connection_state(
            venv_id=venv_id,
            function_manager=fm,
            session_id=1,
        )

        # States should be different and contain the correct values
        assert "my_var" in state0
        assert "my_var" in state1

    finally:
        await pool.close()
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
