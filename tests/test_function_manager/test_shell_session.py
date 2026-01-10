"""
Tests for ShellSession - persistent shell subprocess management.

This test file covers the ShellSession class which provides persistent
shell sessions that maintain state across command executions.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers import _handle_project
from unity.function_manager.shell_session import ShellSession


# ────────────────────────────────────────────────────────────────────────────
# Phase 1.1: Basic Session Lifecycle
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_start_and_close():
    """Session can be started and closed cleanly."""
    session = ShellSession(language="bash")
    assert not session.is_running

    await session.start()
    assert session.is_running

    await session.close()
    assert not session.is_running


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_close_is_idempotent():
    """Closing a session multiple times is safe."""
    session = ShellSession(language="bash")
    await session.start()
    assert session.is_running

    await session.close()
    assert not session.is_running

    # Second close should not raise
    await session.close()
    assert not session.is_running


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_close_unstarted_is_safe():
    """Closing an unstarted session is safe."""
    session = ShellSession(language="bash")
    assert not session.is_running

    # Should not raise
    await session.close()
    assert not session.is_running


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_double_start_raises():
    """Starting an already-started session raises an error."""
    session = ShellSession(language="bash")
    await session.start()

    try:
        with pytest.raises(RuntimeError, match="already started"):
            await session.start()
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_context_manager():
    """Session works as an async context manager."""
    async with ShellSession(language="bash") as session:
        assert session.is_running

    # Should be closed after exiting context
    assert not session.is_running


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_with_custom_env():
    """Session can be created with custom environment variables."""
    session = ShellSession(
        language="bash",
        env={"CUSTOM_VAR": "custom_value"},
    )
    await session.start()
    try:
        assert session.is_running
        # Environment injection will be tested with execute() in Step 1.2
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_with_custom_cwd():
    """Session can be created with custom working directory."""
    session = ShellSession(
        language="bash",
        cwd="/tmp",
    )
    await session.start()
    try:
        assert session.is_running
        # Working directory will be tested with execute() in Step 1.2
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_zsh_start_and_close():
    """Zsh session can be started and closed (if zsh is available)."""
    if not os.path.exists("/bin/zsh"):
        pytest.skip("zsh not available")

    session = ShellSession(language="zsh")
    await session.start()
    try:
        assert session.is_running
    finally:
        await session.close()
    assert not session.is_running


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_sh_start_and_close():
    """POSIX sh session can be started and closed."""
    session = ShellSession(language="sh")
    await session.start()
    try:
        assert session.is_running
    finally:
        await session.close()
    assert not session.is_running


@_handle_project
def test_shell_session_unsupported_language():
    """Creating a session with unsupported language raises on start."""
    # The validation happens when getting the shell command
    session = ShellSession(language="fish")  # type: ignore
    # Error will be raised when trying to get the shell command
    with pytest.raises(ValueError, match="Unsupported shell language"):
        session._get_shell_command()


# ────────────────────────────────────────────────────────────────────────────
# Phase 1.2: Command Execution with Marker-Based Completion Detection
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_simple_command():
    """Can execute a simple command and get output."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("echo hello")
        assert result.exit_code == 0
        assert result.error is None
        assert "hello" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_exit_code_success():
    """Exit code 0 is captured for successful commands."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("true")
        assert result.exit_code == 0
        assert result.error is None
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_exit_code_failure():
    """Non-zero exit codes are captured correctly."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Use a subshell to get exit code without terminating the main shell
        result = await session.execute("(exit 42)")
        assert result.exit_code == 42
        # Note: non-zero exit code is not an error in our model
        # The session should remain usable
        assert session.is_running

        # Verify session is still usable
        result2 = await session.execute("echo still_alive")
        assert result2.exit_code == 0
        assert "still_alive" in result2.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_multiline_output():
    """Commands with multiple lines of output work correctly."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("echo line1; echo line2; echo line3")
        assert result.exit_code == 0
        assert "line1" in result.stdout
        assert "line2" in result.stdout
        assert "line3" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_command_with_quotes():
    """Commands with quotes work correctly."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute('echo "hello world"')
        assert result.exit_code == 0
        assert "hello world" in result.stdout

        result = await session.execute("echo 'single quotes'")
        assert result.exit_code == 0
        assert "single quotes" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_command_with_special_chars():
    """Commands with special characters work correctly."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Arithmetic expansion
        result = await session.execute("echo $((1 + 2))")
        assert result.exit_code == 0
        assert "3" in result.stdout

        # Command substitution
        result = await session.execute("echo $(echo nested)")
        assert result.exit_code == 0
        assert "nested" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_failing_command():
    """Commands that fail return non-zero exit code."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("ls /nonexistent_path_12345 2>&1")
        assert result.exit_code != 0
        assert result.error is None  # Command ran, just failed
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_stderr_merged():
    """Stderr output is captured (merged into stdout)."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("echo error_message >&2")
        # With merged stderr, it appears in stdout
        assert "error_message" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_execute_without_start_raises():
    """Executing without starting raises an error."""
    session = ShellSession(language="bash")
    with pytest.raises(RuntimeError, match="not started"):
        await session.execute("echo hello")


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_multiple_commands_sequential():
    """Multiple commands can be executed sequentially."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result1 = await session.execute("echo first")
        assert result1.exit_code == 0
        assert "first" in result1.stdout

        result2 = await session.execute("echo second")
        assert result2.exit_code == 0
        assert "second" in result2.stdout

        result3 = await session.execute("echo third")
        assert result3.exit_code == 0
        assert "third" in result3.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_command_timeout():
    """Commands that exceed timeout return error."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("sleep 10", timeout=0.5)
        assert result.exit_code == -1
        assert result.error is not None
        assert "timed out" in result.error.lower()
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_custom_env_accessible():
    """Custom environment variables are accessible in commands."""
    session = ShellSession(
        language="bash",
        env={"UNITY_TEST_VAR": "test_value_123"},
    )
    await session.start()
    try:
        result = await session.execute("echo $UNITY_TEST_VAR")
        assert result.exit_code == 0
        assert "test_value_123" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_custom_cwd_used():
    """Custom working directory is used for commands."""
    session = ShellSession(
        language="bash",
        cwd="/tmp",
    )
    await session.start()
    try:
        result = await session.execute("pwd")
        assert result.exit_code == 0
        assert "/tmp" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_empty_command():
    """Empty command executes without error."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        result = await session.execute("")
        assert result.exit_code == 0
        assert result.error is None
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_command_with_newlines():
    """Commands containing newlines work correctly."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Multi-line command (here-doc style)
        result = await session.execute("echo 'line1\nline2'")
        assert result.exit_code == 0
    finally:
        await session.close()


# ────────────────────────────────────────────────────────────────────────────
# Phase 1.3: State Persistence Across Commands
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_state_persists_shell_variables():
    """Shell variables persist across command executions."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Set a shell variable
        result = await session.execute("MY_VAR=hello_world")
        assert result.exit_code == 0

        # Read it back in a separate command
        result = await session.execute("echo $MY_VAR")
        assert result.exit_code == 0
        assert "hello_world" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_state_persists_env_variables():
    """Exported environment variables persist across commands."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Export an environment variable
        await session.execute("export UNITY_PERSISTENT_VAR=persistent_value")

        # Read it back
        result = await session.execute("echo $UNITY_PERSISTENT_VAR")
        assert result.exit_code == 0
        assert "persistent_value" in result.stdout

        # Verify it's actually exported (visible to subprocesses)
        result = await session.execute("bash -c 'echo $UNITY_PERSISTENT_VAR'")
        assert result.exit_code == 0
        assert "persistent_value" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_state_persists_functions():
    """Shell functions persist across command executions."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Define a function
        await session.execute('greet() { echo "Hello, $1!"; }')

        # Call it in a separate command
        result = await session.execute("greet World")
        assert result.exit_code == 0
        assert "Hello, World!" in result.stdout

        # Call it again with different argument
        result = await session.execute("greet Universe")
        assert result.exit_code == 0
        assert "Hello, Universe!" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_state_persists_aliases():
    """Aliases persist across command executions."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Enable alias expansion in non-interactive mode
        await session.execute("shopt -s expand_aliases")

        # Define an alias
        await session.execute("alias greet='echo Hello from alias'")

        # Use it in a separate command
        result = await session.execute("greet")
        assert result.exit_code == 0
        assert "Hello from alias" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_state_persists_cwd():
    """Working directory changes persist across commands."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Change directory
        await session.execute("cd /tmp")

        # Verify in separate command
        result = await session.execute("pwd")
        assert result.exit_code == 0
        assert "/tmp" in result.stdout

        # Change again
        await session.execute("cd /")

        # Verify again
        result = await session.execute("pwd")
        assert result.exit_code == 0
        # On macOS, /tmp is a symlink to /private/tmp, so just check for root
        assert result.stdout.strip() == "/"
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_source_script():
    """Can source scripts and retain their definitions."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Create a temp script
        await session.execute(
            "echo 'export SOURCED_VAR=from_script\n"
            'sourced_func() { echo "I am sourced"; }\' > /tmp/unity_test_source.sh',
        )

        # Source it
        await session.execute("source /tmp/unity_test_source.sh")

        # Verify variable is set
        result = await session.execute("echo $SOURCED_VAR")
        assert result.exit_code == 0
        assert "from_script" in result.stdout

        # Verify function is defined
        result = await session.execute("sourced_func")
        assert result.exit_code == 0
        assert "I am sourced" in result.stdout

        # Cleanup
        await session.execute("rm /tmp/unity_test_source.sh")
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_complex_state_accumulation():
    """Complex state accumulates correctly across many commands."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Set up a counter
        await session.execute("COUNTER=0")

        # Define an increment function
        await session.execute(
            "increment() { COUNTER=$((COUNTER + 1)); echo $COUNTER; }",
        )

        # Call it multiple times
        result = await session.execute("increment")
        assert "1" in result.stdout

        result = await session.execute("increment")
        assert "2" in result.stdout

        result = await session.execute("increment")
        assert "3" in result.stdout

        # Verify final value
        result = await session.execute("echo $COUNTER")
        assert "3" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_array_persistence():
    """Bash arrays persist across commands."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Create an array
        await session.execute("my_array=(one two three)")

        # Access elements
        result = await session.execute("echo ${my_array[0]}")
        assert "one" in result.stdout

        result = await session.execute("echo ${my_array[1]}")
        assert "two" in result.stdout

        # Add to array
        await session.execute("my_array+=(four)")

        result = await session.execute("echo ${my_array[3]}")
        assert "four" in result.stdout

        # Get all elements
        result = await session.execute("echo ${my_array[@]}")
        assert "one" in result.stdout
        assert "four" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_associative_array_persistence():
    """Bash associative arrays persist across commands (bash 4+ only)."""
    session = ShellSession(language="bash")
    await session.start()
    try:
        # Check bash version - associative arrays require bash 4+
        version_result = await session.execute(
            "echo ${BASH_VERSINFO[0]:-0}",
        )
        try:
            major_version = int(version_result.stdout.strip())
        except ValueError:
            major_version = 0

        if major_version < 4:
            pytest.skip(
                f"Associative arrays require bash 4+, found bash {major_version}",
            )

        # Declare an associative array
        await session.execute("declare -A my_map")
        await session.execute("my_map[key1]=value1")
        await session.execute("my_map[key2]=value2")

        # Access elements
        result = await session.execute("echo ${my_map[key1]}")
        assert "value1" in result.stdout

        result = await session.execute("echo ${my_map[key2]}")
        assert "value2" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_zsh_state_persistence():
    """Zsh state persists across commands."""
    if not os.path.exists("/bin/zsh"):
        pytest.skip("zsh not available")

    session = ShellSession(language="zsh")
    await session.start()
    try:
        # Set a variable
        await session.execute("MY_ZSH_VAR=zsh_value")

        # Read it back
        result = await session.execute("echo $MY_ZSH_VAR")
        assert result.exit_code == 0
        assert "zsh_value" in result.stdout

        # Define and call a function
        await session.execute('zsh_greet() { echo "Zsh says: $1"; }')
        result = await session.execute("zsh_greet Hello")
        assert "Zsh says: Hello" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_sh_state_persistence():
    """POSIX sh state persists across commands."""
    session = ShellSession(language="sh")
    await session.start()
    try:
        # Set a variable
        await session.execute("MY_SH_VAR=sh_value")

        # Read it back
        result = await session.execute("echo $MY_SH_VAR")
        assert result.exit_code == 0
        assert "sh_value" in result.stdout

        # Change directory
        await session.execute("cd /tmp")
        result = await session.execute("pwd")
        assert "/tmp" in result.stdout or "/private/tmp" in result.stdout
    finally:
        await session.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_session_independent_sessions_have_independent_state():
    """Different shell sessions have independent state."""
    session1 = ShellSession(language="bash")
    session2 = ShellSession(language="bash")

    await session1.start()
    await session2.start()

    try:
        # Set different values in different sessions
        await session1.execute("UNIQUE_VAR=session1_value")
        await session2.execute("UNIQUE_VAR=session2_value")

        # Verify they're independent
        result1 = await session1.execute("echo $UNIQUE_VAR")
        result2 = await session2.execute("echo $UNIQUE_VAR")

        assert "session1_value" in result1.stdout
        assert "session2_value" in result2.stdout

        # Make sure they didn't leak
        assert "session2_value" not in result1.stdout
        assert "session1_value" not in result2.stdout
    finally:
        await session1.close()
        await session2.close()
