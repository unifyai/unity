"""
Tests for shell function state modes in FunctionManager.

This test file covers the integration of ShellPool with FunctionManager,
enabling stateful execution of shell functions via execute_function().
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.shell_pool import ShellPool
from unity.common.context_registry import ContextRegistry


# ────────────────────────────────────────────────────────────────────────────
# Sample Shell Functions
# ────────────────────────────────────────────────────────────────────────────

SHELL_SET_VAR = """#!/bin/bash
# @name: set_shell_var
# @args: ()
# @description: Sets MY_SHELL_VAR to a value
MY_SHELL_VAR="stateful_value"
echo "Set MY_SHELL_VAR to $MY_SHELL_VAR"
""".strip()

SHELL_GET_VAR = """#!/bin/bash
# @name: get_shell_var
# @args: ()
# @description: Gets the current value of MY_SHELL_VAR
echo "MY_SHELL_VAR=$MY_SHELL_VAR"
""".strip()

SHELL_DEFINE_FUNC = """#!/bin/bash
# @name: define_greeter
# @args: ()
# @description: Defines a greet function
greet() {
    echo "Hello, $1!"
}
echo "Defined greet function"
""".strip()

SHELL_CALL_FUNC = """#!/bin/bash
# @name: call_greeter
# @args: ()
# @description: Calls the greet function
greet "World"
""".strip()

SHELL_INCREMENT = """#!/bin/bash
# @name: increment_counter
# @args: ()
# @description: Increments and prints a counter
COUNTER=$((COUNTER + 1))
echo "Counter is now $COUNTER"
""".strip()

SHELL_SIMPLE = """#!/bin/bash
# @name: simple_echo
# @args: ()
# @description: Simple echo command
echo "Hello from shell function"
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
def shell_pool_factory():
    """Factory fixture that creates ShellPool instances."""
    pools = []

    def _create():
        pool = ShellPool()
        pools.append(pool)
        return pool

    yield _create

    # Cleanup is handled in tests since we need async


# ────────────────────────────────────────────────────────────────────────────
# Stateless Mode Tests (Backward Compatibility)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateless_mode_basic(function_manager_factory):
    """Shell functions execute in stateless mode by default."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SHELL_SIMPLE, language="bash")

    result = await fm.execute_function(
        function_name="simple_echo",
        state_mode="stateless",
    )

    assert result["error"] is None
    assert "Hello from shell function" in result["stdout"]


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateless_mode_no_persistence(function_manager_factory):
    """Stateless mode does not persist state between calls."""
    fm = function_manager_factory()

    fm.add_functions(implementations=SHELL_SET_VAR, language="bash")
    fm.add_functions(implementations=SHELL_GET_VAR, language="bash")

    # Set variable
    await fm.execute_function(
        function_name="set_shell_var",
        state_mode="stateless",
    )

    # Get variable - should be empty (fresh process)
    result = await fm.execute_function(
        function_name="get_shell_var",
        state_mode="stateless",
    )

    assert result["error"] is None
    # Variable should NOT be set (stateless = fresh process)
    assert "stateful_value" not in result["stdout"]


# ────────────────────────────────────────────────────────────────────────────
# Stateful Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateful_mode_persists_variables(
    function_manager_factory,
    shell_pool_factory,
):
    """Shell variables persist across stateful executions."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_SET_VAR, language="bash")
        fm.add_functions(implementations=SHELL_GET_VAR, language="bash")

        # Set variable
        result1 = await fm.execute_function(
            function_name="set_shell_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result1["error"] is None

        # Get variable in same session - should be set
        result2 = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result2["error"] is None
        assert "stateful_value" in result2["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateful_mode_persists_functions(
    function_manager_factory,
    shell_pool_factory,
):
    """Shell functions defined in one call persist to later calls."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_DEFINE_FUNC, language="bash")
        fm.add_functions(implementations=SHELL_CALL_FUNC, language="bash")

        # Define the function
        result1 = await fm.execute_function(
            function_name="define_greeter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result1["error"] is None
        assert "Defined greet function" in result1["stdout"]

        # Call it
        result2 = await fm.execute_function(
            function_name="call_greeter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result2["error"] is None
        assert "Hello, World!" in result2["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateful_mode_accumulates_state(
    function_manager_factory,
    shell_pool_factory,
):
    """State accumulates across multiple stateful calls."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_INCREMENT, language="bash")

        # Initialize counter
        await pool.execute(language="bash", command="COUNTER=0")

        # Increment multiple times
        result1 = await fm.execute_function(
            function_name="increment_counter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "Counter is now 1" in result1["stdout"]

        result2 = await fm.execute_function(
            function_name="increment_counter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "Counter is now 2" in result2["stdout"]

        result3 = await fm.execute_function(
            function_name="increment_counter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "Counter is now 3" in result3["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_stateful_requires_pool(function_manager_factory):
    """Stateful mode raises error if shell_pool not provided."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SHELL_SIMPLE, language="bash")

    with pytest.raises(ValueError, match="shell_pool"):
        await fm.execute_function(
            function_name="simple_echo",
            state_mode="stateful",
            shell_pool=None,
        )


# ────────────────────────────────────────────────────────────────────────────
# Session Independence Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_different_sessions_independent(
    function_manager_factory,
    shell_pool_factory,
):
    """Different session_ids have independent state."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_GET_VAR, language="bash")

        # Set different values in different sessions via direct pool access
        await pool.execute(
            language="bash",
            command="MY_SHELL_VAR=session0_value",
            session_id=0,
        )
        await pool.execute(
            language="bash",
            command="MY_SHELL_VAR=session1_value",
            session_id=1,
        )

        # Read from each session
        result0 = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="stateful",
            shell_pool=pool,
            session_id=0,
        )
        result1 = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="stateful",
            shell_pool=pool,
            session_id=1,
        )

        assert "session0_value" in result0["stdout"]
        assert "session1_value" in result1["stdout"]
    finally:
        await pool.close()


# ────────────────────────────────────────────────────────────────────────────
# Read-Only Mode Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_read_only_sees_stateful_state(
    function_manager_factory,
    shell_pool_factory,
):
    """Read-only mode sees state from stateful session."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_GET_VAR, language="bash")

        # Set variable in stateful mode
        await pool.execute(
            language="bash",
            command="MY_SHELL_VAR=read_only_test_value",
        )

        # Read in read_only mode - should see the variable
        result = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert result["error"] is None
        assert "read_only_test_value" in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_read_only_does_not_persist_changes(
    function_manager_factory,
    shell_pool_factory,
):
    """Read-only mode does not persist state changes."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_SET_VAR, language="bash")
        fm.add_functions(implementations=SHELL_GET_VAR, language="bash")

        # Set initial value in stateful mode
        await pool.execute(
            language="bash",
            command="MY_SHELL_VAR=original_value",
        )

        # Execute set_shell_var in read_only - modifies ephemeral session
        result = await fm.execute_function(
            function_name="set_shell_var",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert result["error"] is None
        assert "stateful_value" in result["stdout"]

        # Stateful session should still have original value
        result = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "original_value" in result["stdout"]
        assert "stateful_value" not in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_read_only_sees_functions(
    function_manager_factory,
    shell_pool_factory,
):
    """Read-only mode can use functions defined in stateful session."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_CALL_FUNC, language="bash")

        # Define function in stateful mode
        await pool.execute(
            language="bash",
            command='greet() { echo "Hello, $1!"; }',
        )

        # Call function in read_only mode
        result = await fm.execute_function(
            function_name="call_greeter",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert result["error"] is None
        assert "Hello, World!" in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_read_only_requires_pool(function_manager_factory):
    """Read-only mode raises error if shell_pool not provided."""
    fm = function_manager_factory()
    fm.add_functions(implementations=SHELL_SIMPLE, language="bash")

    with pytest.raises(ValueError, match="shell_pool"):
        await fm.execute_function(
            function_name="simple_echo",
            state_mode="read_only",
            shell_pool=None,
        )


@_handle_project
@pytest.mark.asyncio
async def test_shell_read_only_independent_of_each_other(
    function_manager_factory,
    shell_pool_factory,
):
    """Multiple read_only executions don't affect each other."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        # Define a function that modifies and reads a variable
        modify_and_read = """#!/bin/bash
# @name: modify_and_read
# @args: ()
# @description: Modify a var and read it
TEMP_VAR=$((TEMP_VAR + 1))
echo "TEMP_VAR=$TEMP_VAR"
""".strip()

        fm.add_functions(implementations=modify_and_read, language="bash")

        # Set initial value
        await pool.execute(language="bash", command="TEMP_VAR=10")

        # Each read_only execution should see 11 (10 + 1)
        # because they all start from the same stateful snapshot
        result1 = await fm.execute_function(
            function_name="modify_and_read",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert "TEMP_VAR=11" in result1["stdout"]

        result2 = await fm.execute_function(
            function_name="modify_and_read",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert "TEMP_VAR=11" in result2["stdout"]

        # Stateful session should still have 10
        result = await pool.execute(language="bash", command="echo $TEMP_VAR")
        assert "10" in result.stdout
    finally:
        await pool.close()


# ────────────────────────────────────────────────────────────────────────────
# Mixed Python/Shell Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_mixed_python_and_shell_functions(
    function_manager_factory,
    shell_pool_factory,
):
    """Can have both Python and shell functions, executed appropriately."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        # Add a Python function
        fm.add_functions(
            implementations="def py_hello():\n    return 'Hello from Python'\n",
            language="python",
        )

        # Add a shell function
        fm.add_functions(implementations=SHELL_SIMPLE, language="bash")

        # Execute Python function
        py_result = await fm.execute_function(
            function_name="py_hello",
            state_mode="stateless",
        )
        assert py_result["error"] is None
        assert py_result["result"] == "Hello from Python"

        # Execute shell function
        sh_result = await fm.execute_function(
            function_name="simple_echo",
            state_mode="stateless",
        )
        assert sh_result["error"] is None
        assert "Hello from shell function" in sh_result["stdout"]
    finally:
        await pool.close()


# ────────────────────────────────────────────────────────────────────────────
# Zsh State Mode Tests
# ────────────────────────────────────────────────────────────────────────────

# Zsh-specific shell functions
ZSH_SET_VAR = """#!/bin/zsh
# @name: zsh_set_var
# @args: ()
# @description: Sets MY_ZSH_VAR to a value
MY_ZSH_VAR="zsh_stateful_value"
echo "Set MY_ZSH_VAR to $MY_ZSH_VAR"
""".strip()

ZSH_GET_VAR = """#!/bin/zsh
# @name: zsh_get_var
# @args: ()
# @description: Gets the current value of MY_ZSH_VAR
echo "MY_ZSH_VAR=$MY_ZSH_VAR"
""".strip()

ZSH_DEFINE_FUNC = """#!/bin/zsh
# @name: zsh_define_greeter
# @args: ()
# @description: Defines a greet function in zsh
greet() {
    echo "Hello from zsh, $1!"
}
echo "Defined zsh greet function"
""".strip()

ZSH_CALL_FUNC = """#!/bin/zsh
# @name: zsh_call_greeter
# @args: ()
# @description: Calls the greet function in zsh
greet "World"
""".strip()

ZSH_SIMPLE = """#!/bin/zsh
# @name: zsh_simple_echo
# @args: ()
# @description: Simple zsh echo command
echo "Hello from zsh function"
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_stateless_mode_basic(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh functions execute in stateless mode."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=ZSH_SIMPLE, language="zsh")

        result = await fm.execute_function(
            function_name="zsh_simple_echo",
            state_mode="stateless",
        )

        assert result["error"] is None
        assert "Hello from zsh function" in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_stateful_mode_persists_variables(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh variables persist across stateful executions."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=ZSH_SET_VAR, language="zsh")
        fm.add_functions(implementations=ZSH_GET_VAR, language="zsh")

        # Set variable
        result1 = await fm.execute_function(
            function_name="zsh_set_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result1["error"] is None

        # Get variable in same session - should be set
        result2 = await fm.execute_function(
            function_name="zsh_get_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result2["error"] is None
        assert "zsh_stateful_value" in result2["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_stateful_mode_persists_functions(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh functions defined in one call persist to later calls."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=ZSH_DEFINE_FUNC, language="zsh")
        fm.add_functions(implementations=ZSH_CALL_FUNC, language="zsh")

        # Define the function
        result1 = await fm.execute_function(
            function_name="zsh_define_greeter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result1["error"] is None
        assert "Defined zsh greet function" in result1["stdout"]

        # Call it
        result2 = await fm.execute_function(
            function_name="zsh_call_greeter",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert result2["error"] is None
        assert "Hello from zsh, World!" in result2["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_read_only_sees_stateful_state(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh read-only mode sees state from stateful session."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=ZSH_GET_VAR, language="zsh")

        # Set variable in stateful mode
        await pool.execute(
            language="zsh",
            command="MY_ZSH_VAR=zsh_read_only_test",
        )

        # Read in read_only mode - should see the variable
        result = await fm.execute_function(
            function_name="zsh_get_var",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert result["error"] is None
        assert "zsh_read_only_test" in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_read_only_does_not_persist_changes(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh read-only mode does not persist state changes."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=ZSH_SET_VAR, language="zsh")
        fm.add_functions(implementations=ZSH_GET_VAR, language="zsh")

        # Set initial value in stateful mode
        await pool.execute(
            language="zsh",
            command="MY_ZSH_VAR=original_zsh_value",
        )

        # Execute set_var in read_only - modifies ephemeral session
        result = await fm.execute_function(
            function_name="zsh_set_var",
            state_mode="read_only",
            shell_pool=pool,
        )
        assert result["error"] is None
        assert "zsh_stateful_value" in result["stdout"]

        # Stateful session should still have original value
        result = await fm.execute_function(
            function_name="zsh_get_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "original_zsh_value" in result["stdout"]
        assert "zsh_stateful_value" not in result["stdout"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_zsh_and_bash_independent_sessions(
    function_manager_factory,
    shell_pool_factory,
):
    """Zsh and bash have independent sessions in the same pool."""
    fm = function_manager_factory()
    pool = shell_pool_factory()

    try:
        fm.add_functions(implementations=SHELL_GET_VAR, language="bash")
        fm.add_functions(implementations=ZSH_GET_VAR, language="zsh")

        # Set different values in bash and zsh
        await pool.execute(
            language="bash",
            command="MY_SHELL_VAR=bash_value",
        )
        await pool.execute(
            language="zsh",
            command="MY_ZSH_VAR=zsh_value",
        )

        # Verify bash sees its value
        bash_result = await fm.execute_function(
            function_name="get_shell_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "bash_value" in bash_result["stdout"]

        # Verify zsh sees its value
        zsh_result = await fm.execute_function(
            function_name="zsh_get_var",
            state_mode="stateful",
            shell_pool=pool,
        )
        assert "zsh_value" in zsh_result["stdout"]
    finally:
        await pool.close()
