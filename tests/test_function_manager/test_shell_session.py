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
