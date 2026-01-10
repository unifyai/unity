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
