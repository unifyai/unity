"""
Tests for VenvPool state inspection methods (used by CodeActActor's
inspect_state tool).

Covers:
1. VenvPool.list_active_sessions() - lists active venv sessions
2. VenvPool.get_all_states() - gets state from all active sessions
3. State filtering (internal names excluded)
4. Empty state handling
"""

from __future__ import annotations

import shutil

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import (
    FunctionManager,
    VenvPool,
)
from unity.common.context_registry import ContextRegistry


# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SET_VARS_FUNC = """
async def set_vars():
    global user_data, counter, _private_var
    user_data = {"name": "Alice", "age": 30}
    counter = 42
    _private_var = "should_be_filtered"
    return "done"
""".strip()

GET_VARS_FUNC = """
async def get_vars():
    return {"user_data": user_data, "counter": counter}
""".strip()

SIMPLE_FUNC = """
async def simple_func():
    return "hello"
""".strip()

# Minimal venv content
MINIMAL_VENV_CONTENT = """
[project]
name = "test-inspect-state"
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


# ────────────────────────────────────────────────────────────────────────────
# 1. VenvPool.list_active_sessions() Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_list_active_sessions_empty_initially(function_manager_factory):
    """list_active_sessions should return empty list when no connections exist."""
    pool = VenvPool()
    try:
        sessions = pool.list_active_sessions()
        assert sessions == []
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_list_active_sessions_after_execution(function_manager_factory):
    """list_active_sessions should return session info after function execution."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Execute a function to create a connection
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNC,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )

        sessions = pool.list_active_sessions()
        assert len(sessions) == 1
        assert sessions[0] == (venv_id, 0)  # Default session_id is 0
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_list_active_sessions_multiple_venvs(function_manager_factory):
    """list_active_sessions should list sessions from multiple venvs."""
    fm = function_manager_factory()
    venv_id1 = await _create_prepared_venv(fm)
    venv_id2 = fm.add_venv(
        venv=MINIMAL_VENV_CONTENT.replace("test-inspect-state", "test-venv-2"),
    )
    await fm.prepare_venv(venv_id=venv_id2)
    pool = VenvPool()

    try:
        # Execute in both venvs
        await pool.execute_in_venv(
            venv_id=venv_id1,
            implementation=SIMPLE_FUNC,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )
        await pool.execute_in_venv(
            venv_id=venv_id2,
            implementation=SIMPLE_FUNC,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )

        sessions = pool.list_active_sessions()
        assert len(sessions) == 2
        assert (venv_id1, 0) in sessions
        assert (venv_id2, 0) in sessions
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id1)
        _cleanup_venv(fm, venv_id2)


# ────────────────────────────────────────────────────────────────────────────
# 2. VenvPool.get_all_states() Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_get_all_states_empty_initially(function_manager_factory):
    """get_all_states should return empty dict when no connections exist."""
    fm = function_manager_factory()
    pool = VenvPool()

    try:
        states = await pool.get_all_states(function_manager=fm)
        assert states == {}
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_get_all_states_returns_state(function_manager_factory):
    """get_all_states should return state from active sessions."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Execute function that sets state
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SET_VARS_FUNC,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )

        states = await pool.get_all_states(function_manager=fm)
        assert len(states) == 1

        key = (venv_id, 0)
        assert key in states

        state = states[key]
        assert "user_data" in state
        assert "counter" in state
        # State is serialized with type info: {"type": "primitive", "value": 42}
        assert state["counter"]["type"] == "primitive"
        assert state["counter"]["value"] == 42
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_get_all_states_multiple_venvs(function_manager_factory):
    """get_all_states should return state from all active sessions."""
    fm = function_manager_factory()
    venv_id1 = await _create_prepared_venv(fm)
    venv_id2 = fm.add_venv(
        venv=MINIMAL_VENV_CONTENT.replace("test-inspect-state", "test-venv-2"),
    )
    await fm.prepare_venv(venv_id=venv_id2)
    pool = VenvPool()

    # Different functions for each venv to set different state
    set_x_func = """
async def set_x():
    global x_value
    x_value = 100
    return "done"
""".strip()

    set_y_func = """
async def set_y():
    global y_value
    y_value = 200
    return "done"
""".strip()

    try:
        # Set state in both venvs
        await pool.execute_in_venv(
            venv_id=venv_id1,
            implementation=set_x_func,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )
        await pool.execute_in_venv(
            venv_id=venv_id2,
            implementation=set_y_func,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )

        states = await pool.get_all_states(function_manager=fm)
        assert len(states) == 2

        # State is serialized: {"type": "primitive", "value": ...}
        assert states[(venv_id1, 0)]["x_value"]["value"] == 100
        assert states[(venv_id2, 0)]["y_value"]["value"] == 200
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id1)
        _cleanup_venv(fm, venv_id2)


# ────────────────────────────────────────────────────────────────────────────
# 3. State After Close Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_list_sessions_empty_after_close(function_manager_factory):
    """list_active_sessions should return empty after pool is closed."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Create a connection
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNC,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )
        assert len(pool.list_active_sessions()) == 1

        # Close the pool
        await pool.close()

        # Should be empty after close
        assert pool.list_active_sessions() == []
    finally:
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 4. Integration with State Modes
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_get_all_states_reflects_stateful_changes(function_manager_factory):
    """State inspection should reflect changes made via stateful execution."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    set_counter_func = """
async def set_counter(value):
    global counter
    counter = value
    return f"Set counter to {value}"
""".strip()

    try:
        # Set counter to 10
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=set_counter_func,
            call_kwargs={"value": 10},
            is_async=True,
            function_manager=fm,
        )

        # Inspect state - values are serialized: {"type": "primitive", "value": ...}
        states = await pool.get_all_states(function_manager=fm)
        assert states[(venv_id, 0)]["counter"]["value"] == 10

        # Update counter to 20
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=set_counter_func,
            call_kwargs={"value": 20},
            is_async=True,
            function_manager=fm,
        )

        # State inspection should reflect update
        states = await pool.get_all_states(function_manager=fm)
        assert states[(venv_id, 0)]["counter"]["value"] == 20
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_get_all_states_not_affected_by_stateless(function_manager_factory):
    """State inspection should not reflect changes from stateless execution."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    set_counter_func = """
async def set_counter(value):
    global counter
    counter = value
    return f"Set counter to {value}"
""".strip()

    try:
        # Set counter to 10 (stateful via pool)
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=set_counter_func,
            call_kwargs={"value": 10},
            is_async=True,
            function_manager=fm,
        )

        # Verify state - values are serialized: {"type": "primitive", "value": ...}
        states = await pool.get_all_states(function_manager=fm)
        assert states[(venv_id, 0)]["counter"]["value"] == 10

        # Try to change via stateless (bypasses pool - one-shot subprocess)
        await fm.execute_in_venv(
            venv_id=venv_id,
            implementation=set_counter_func,
            call_kwargs={"value": 999},
            is_async=True,
        )

        # State should be unchanged (stateless didn't affect it)
        states = await pool.get_all_states(function_manager=fm)
        assert states[(venv_id, 0)]["counter"]["value"] == 10
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)
