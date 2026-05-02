"""
Tests for persistent venv subprocess connections (VenvPool).

These tests verify that:
1. State persists across multiple function calls within the same venv
2. Subprocess crashes are detected and recovered from
3. Execution timeouts are handled correctly
4. Graceful shutdown cleans up resources
5. Concurrent calls are serialized correctly
"""

import asyncio
import pytest
import shutil

from unity.function_manager.function_manager import (
    FunctionManager,
    VenvPool,
)
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# Sample pyproject.toml with minimal dependencies (fast to sync)
MINIMAL_VENV_CONTENT = """
[project]
name = "test-persistent-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Test Functions
# ────────────────────────────────────────────────────────────────────────────

# Function that sets a global variable in a storage dict
SET_GLOBAL_FUNCTION = """
async def set_global(name: str, value: int) -> str:
    global _stored_values
    try:
        _stored_values
    except NameError:
        _stored_values = {}
    _stored_values[name] = value
    return f"Set {name} = {value}"
""".strip()

# Function that reads a global variable from storage
GET_GLOBAL_FUNCTION = """
async def get_global(name: str) -> int:
    global _stored_values
    try:
        return _stored_values.get(name, -1)
    except NameError:
        return -1
""".strip()

# Function that increments a counter (tests state mutation)
INCREMENT_COUNTER_FUNCTION = """
async def increment_counter() -> int:
    global _counter
    try:
        _counter
    except NameError:
        _counter = 0
    _counter += 1
    return _counter
""".strip()

# Long-running function for timeout tests
SLOW_FUNCTION = """
async def slow_function(seconds: float) -> str:
    import asyncio
    await asyncio.sleep(seconds)
    return "done"
""".strip()

# Function that crashes the process
CRASH_FUNCTION = """
async def crash_process() -> None:
    import os
    import signal
    os.kill(os.getpid(), signal.SIGKILL)
""".strip()

# Simple function for basic testing
SIMPLE_FUNCTION = """
async def simple_add(a: int, b: int) -> int:
    return a + b
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """
    Factory fixture that creates FunctionManager instances.

    Returns a callable that creates a FunctionManager. This ensures the
    FunctionManager is instantiated AFTER @_handle_project sets up the
    test-specific context, providing proper isolation for parallel tests.
    """
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

    # Cleanup all created managers
    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


class _RetiredConnection:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def shutdown(self) -> None:
        self.shutdown_called = True


def test_venv_pool_invalidation_retires_connections_without_closing_pool():
    """Invalidation removes pooled sessions and shuts them down through lifecycle hooks."""
    pool = VenvPool()
    connection = _RetiredConnection()
    pool._connections[(1, 0)] = connection  # type: ignore[assignment]

    invalidated = pool.invalidate_sessions()

    assert invalidated == 1
    assert pool._connections == {}
    assert pool._metadata == {}
    assert connection.shutdown_called is True
    assert pool._closed is False


# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def _create_prepared_venv(fm: FunctionManager) -> int:
    """
    Create and prepare a minimal venv. Call this INSIDE the test body
    (after @_handle_project has set up the per-test context).

    Returns the venv_id.
    """
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
# 1. State Persistence Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_state_persists_across_calls(function_manager_factory):
    """Variables set in one call should be accessible in subsequent calls."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # First call: set a global variable
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SET_GLOBAL_FUNCTION,
            call_kwargs={"name": "my_var", "value": 42},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None
        assert "Set my_var = 42" in result1["result"]

        # Second call: read the global variable
        result2 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=GET_GLOBAL_FUNCTION,
            call_kwargs={"name": "my_var"},
            is_async=True,
            function_manager=fm,
        )
        assert result2["error"] is None
        assert result2["result"] == 42
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_counter_increments_across_calls(function_manager_factory):
    """A counter variable should increment correctly across multiple calls."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Make multiple calls and verify counter increments
        for expected in [1, 2, 3, 4, 5]:
            result = await pool.execute_in_venv(
                venv_id=venv_id,
                implementation=INCREMENT_COUNTER_FUNCTION,
                call_kwargs={},
                is_async=True,
                function_manager=fm,
            )
            assert result["error"] is None
            assert result["result"] == expected
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_state_isolated_between_pools(function_manager_factory):
    """Different VenvPools should have isolated state."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    pool1 = VenvPool()
    pool2 = VenvPool()

    try:
        # Set variable in pool1
        result1 = await pool1.execute_in_venv(
            venv_id=venv_id,
            implementation=SET_GLOBAL_FUNCTION,
            call_kwargs={"name": "pool_var", "value": 100},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None

        # Read from pool2 - should NOT see pool1's variable
        result2 = await pool2.execute_in_venv(
            venv_id=venv_id,
            implementation=GET_GLOBAL_FUNCTION,
            call_kwargs={"name": "pool_var"},
            is_async=True,
            function_manager=fm,
        )
        assert result2["error"] is None
        assert result2["result"] == -1  # Default value, variable not found
    finally:
        await pool1.close()
        await pool2.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_function_definition_persists(function_manager_factory):
    """Functions defined in one call should be callable in subsequent calls."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Define a helper function
        define_helper = """
async def define_helper() -> str:
    global helper_func
    def helper_func(x):
        return x * 2
    return "helper defined"
""".strip()

        # Use the helper function
        use_helper = """
async def use_helper(val: int) -> int:
    return helper_func(val)
""".strip()

        # First call: define the helper
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=define_helper,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None

        # Second call: use the helper
        result2 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=use_helper,
            call_kwargs={"val": 21},
            is_async=True,
            function_manager=fm,
        )
        assert result2["error"] is None
        assert result2["result"] == 42
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 2. Subprocess Crash Recovery Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_detects_subprocess_death(function_manager_factory):
    """Pool should detect when subprocess dies and raise appropriate error."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # First call succeeds and establishes connection
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None
        assert result1["result"] == 3

        # Get the connection and kill the subprocess manually
        conn = await pool.get_or_create_connection(venv_id, fm)
        conn._process.kill()
        await conn._process.wait()

        # Next call should detect death and recover
        result2 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 10, "b": 20},
            is_async=True,
            function_manager=fm,
        )
        assert result2["error"] is None
        assert result2["result"] == 30
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_recovers_from_crash_function(function_manager_factory):
    """Pool should recover when a function crashes the subprocess."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # First call succeeds
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None

        # This call will crash the subprocess - it should fail
        # but the pool should recover for the next call
        try:
            await pool.execute_in_venv(
                venv_id=venv_id,
                implementation=CRASH_FUNCTION,
                call_kwargs={},
                is_async=True,
                function_manager=fm,
            )
        except (RuntimeError, EOFError):
            pass  # Expected - subprocess died

        # Give the process a moment to be fully terminated
        await asyncio.sleep(0.1)

        # Next call should work (new subprocess spawned)
        result3 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 100, "b": 200},
            is_async=True,
            function_manager=fm,
        )
        assert result3["error"] is None
        assert result3["result"] == 300
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_state_lost_after_crash(function_manager_factory):
    """State should be lost when subprocess crashes and is recreated."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Set state
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SET_GLOBAL_FUNCTION,
            call_kwargs={"name": "important_data", "value": 999},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None

        # Verify state exists
        result2 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=GET_GLOBAL_FUNCTION,
            call_kwargs={"name": "important_data"},
            is_async=True,
            function_manager=fm,
        )
        assert result2["result"] == 999

        # Kill the subprocess
        conn = await pool.get_or_create_connection(venv_id, fm)
        conn._process.kill()
        await conn._process.wait()

        await asyncio.sleep(0.1)

        # State should be lost (new subprocess)
        result3 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=GET_GLOBAL_FUNCTION,
            call_kwargs={"name": "important_data"},
            is_async=True,
            function_manager=fm,
        )
        assert result3["error"] is None
        assert result3["result"] == -1  # Default value, state was lost
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 3. Timeout Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_execution_timeout(function_manager_factory):
    """Execution should raise TimeoutError when timeout is exceeded."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        with pytest.raises(asyncio.TimeoutError):
            await pool.execute_in_venv(
                venv_id=venv_id,
                implementation=SLOW_FUNCTION,
                call_kwargs={"seconds": 10.0},  # Way longer than timeout
                is_async=True,
                function_manager=fm,
                timeout=0.5,  # Short timeout
            )
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_no_timeout_when_fast(function_manager_factory):
    """Fast functions should complete normally with a timeout set."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        result = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 5, "b": 10},
            is_async=True,
            function_manager=fm,
            timeout=10.0,  # Generous timeout
        )
        assert result["error"] is None
        assert result["result"] == 15
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_connection_recovers_after_timeout(function_manager_factory):
    """Pool should recover after a timeout by recreating the connection."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # First, do a successful call and set some state
        result1 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SET_GLOBAL_FUNCTION,
            call_kwargs={"name": "before_timeout", "value": 999},
            is_async=True,
            function_manager=fm,
        )
        assert result1["error"] is None

        # Cause a timeout
        try:
            await pool.execute_in_venv(
                venv_id=venv_id,
                implementation=SLOW_FUNCTION,
                call_kwargs={"seconds": 10.0},
                is_async=True,
                function_manager=fm,
                timeout=0.2,
            )
        except asyncio.TimeoutError:
            pass

        # Wait a moment for things to settle
        await asyncio.sleep(0.3)

        # After timeout, the pool should recreate the connection.
        # State will be lost, but calls should succeed.
        result2 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 100, "b": 200},
            is_async=True,
            function_manager=fm,
        )
        assert result2["error"] is None
        assert result2["result"] == 300

        # Verify state was lost (new subprocess)
        result3 = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=GET_GLOBAL_FUNCTION,
            call_kwargs={"name": "before_timeout"},
            is_async=True,
            function_manager=fm,
        )
        assert result3["error"] is None
        assert result3["result"] == -1  # State was lost
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 4. Graceful Shutdown Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_close_terminates_subprocesses(function_manager_factory):
    """Closing the pool should terminate all subprocesses."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    pool = VenvPool()

    try:
        # Create a connection
        result = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )
        assert result["error"] is None

        # Get the connection to check its process
        conn = pool._connections.get((venv_id, 0))
        assert conn is not None
        pid = conn._process.pid
        assert conn._process.returncode is None  # Still running

        # Close the pool
        await pool.close()

        # Process should be terminated
        await asyncio.sleep(0.1)
        assert conn._process.returncode is not None
    finally:
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_close_is_idempotent(function_manager_factory):
    """Calling close() multiple times should be safe."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    pool = VenvPool()

    try:
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )

        # Multiple close calls should not raise
        await pool.close()
        await pool.close()
        await pool.close()
    finally:
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_execute_after_close_fails(function_manager_factory):
    """Executing after close() should raise an error."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)

    pool = VenvPool()

    try:
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )

        await pool.close()

        with pytest.raises(RuntimeError, match="closed"):
            await pool.execute_in_venv(
                venv_id=venv_id,
                implementation=SIMPLE_FUNCTION,
                call_kwargs={"a": 3, "b": 4},
                is_async=True,
                function_manager=fm,
            )
    finally:
        _cleanup_venv(fm, venv_id)


# ────────────────────────────────────────────────────────────────────────────
# 5. Concurrent Call Serialization Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_concurrent_calls_are_serialized(function_manager_factory):
    """Multiple concurrent calls to the same venv should be serialized."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # Function that records call order
        record_order_func = """
async def record_order(call_id: int) -> int:
    import asyncio
    global _call_order
    try:
        _call_order
    except NameError:
        _call_order = []
    _call_order.append(call_id)
    await asyncio.sleep(0.05)  # Small delay to allow interleaving if not serialized
    return call_id
""".strip()

        # Function to get the recorded order
        get_order_func = """
async def get_order() -> list:
    global _call_order
    try:
        return _call_order
    except NameError:
        return []
""".strip()

        # Launch multiple concurrent calls
        tasks = [
            pool.execute_in_venv(
                venv_id=venv_id,
                implementation=record_order_func,
                call_kwargs={"call_id": i},
                is_async=True,
                function_manager=fm,
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        for r in results:
            assert r["error"] is None

        # Get the order
        order_result = await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=get_order_func,
            call_kwargs={},
            is_async=True,
            function_manager=fm,
        )
        assert order_result["error"] is None

        # The order should have all 5 calls (though order may vary due to task scheduling)
        order = order_result["result"]
        assert len(order) == 5
        assert set(order) == {0, 1, 2, 3, 4}
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_concurrent_calls_to_different_venvs(function_manager_factory):
    """Concurrent calls to different venvs should work independently."""
    fm = function_manager_factory()

    # Create two venvs
    venv_id1 = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_id2 = fm.add_venv(
        venv=MINIMAL_VENV_CONTENT.replace("test-persistent-venv", "test-venv-2"),
    )

    pool = VenvPool()

    try:
        # Prepare both venvs
        await fm.prepare_venv(venv_id=venv_id1)
        await fm.prepare_venv(venv_id=venv_id2)

        # Set different values in each venv concurrently
        r1, r2 = await asyncio.gather(
            pool.execute_in_venv(
                venv_id=venv_id1,
                implementation=SET_GLOBAL_FUNCTION,
                call_kwargs={"name": "venv_marker", "value": 111},
                is_async=True,
                function_manager=fm,
            ),
            pool.execute_in_venv(
                venv_id=venv_id2,
                implementation=SET_GLOBAL_FUNCTION,
                call_kwargs={"name": "venv_marker", "value": 222},
                is_async=True,
                function_manager=fm,
            ),
        )

        assert r1["error"] is None
        assert r2["error"] is None

        # Read back and verify isolation
        r3, r4 = await asyncio.gather(
            pool.execute_in_venv(
                venv_id=venv_id1,
                implementation=GET_GLOBAL_FUNCTION,
                call_kwargs={"name": "venv_marker"},
                is_async=True,
                function_manager=fm,
            ),
            pool.execute_in_venv(
                venv_id=venv_id2,
                implementation=GET_GLOBAL_FUNCTION,
                call_kwargs={"name": "venv_marker"},
                is_async=True,
                function_manager=fm,
            ),
        )

        assert r3["result"] == 111
        assert r4["result"] == 222
    finally:
        await pool.close()
        for vid in [venv_id1, venv_id2]:
            _cleanup_venv(fm, vid)


# ────────────────────────────────────────────────────────────────────────────
# 6. Connection Management Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_connection_reused_for_same_venv(function_manager_factory):
    """Multiple calls should reuse the same connection."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        # First call creates connection
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )

        conn1 = pool._connections.get((venv_id, 0))
        pid1 = conn1._process.pid

        # Second call should reuse
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 3, "b": 4},
            is_async=True,
            function_manager=fm,
        )

        conn2 = pool._connections.get((venv_id, 0))
        pid2 = conn2._process.pid

        # Same connection (same PID)
        assert pid1 == pid2
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)


@_handle_project
@pytest.mark.asyncio
async def test_is_alive_detects_dead_process(function_manager_factory):
    """is_alive() should return False after process is killed."""
    fm = function_manager_factory()
    venv_id = await _create_prepared_venv(fm)
    pool = VenvPool()

    try:
        await pool.execute_in_venv(
            venv_id=venv_id,
            implementation=SIMPLE_FUNCTION,
            call_kwargs={"a": 1, "b": 2},
            is_async=True,
            function_manager=fm,
        )

        conn = pool._connections.get((venv_id, 0))
        assert conn.is_alive() is True

        conn._process.kill()
        await conn._process.wait()

        assert conn.is_alive() is False
    finally:
        await pool.close()
        _cleanup_venv(fm, venv_id)
