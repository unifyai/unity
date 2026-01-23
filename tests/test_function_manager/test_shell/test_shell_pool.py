"""
Tests for ShellPool - multi-session shell management.

This test file covers the ShellPool class which manages multiple persistent
shell sessions, allowing independent stateful execution contexts.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers import _handle_project
from unity.function_manager.shell_pool import ShellPool

# ────────────────────────────────────────────────────────────────────────────
# Basic Pool Operations
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_basic_execution():
    """Pool can execute commands."""
    pool = ShellPool()
    try:
        result = await pool.execute(language="bash", command="echo hello")
        assert result.exit_code == 0
        assert result.error is None
        assert "hello" in result.stdout
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_context_manager():
    """Pool works as an async context manager."""
    async with ShellPool() as pool:
        result = await pool.execute(language="bash", command="echo context")
        assert result.exit_code == 0
        assert "context" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_close_is_idempotent():
    """Closing a pool multiple times is safe."""
    pool = ShellPool()
    await pool.execute(language="bash", command="echo test")

    await pool.close()
    await pool.close()  # Should not raise


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_close_empty_is_safe():
    """Closing an empty pool is safe."""
    pool = ShellPool()
    await pool.close()  # Should not raise


# ────────────────────────────────────────────────────────────────────────────
# State Persistence Within Sessions
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_state_persists_within_session():
    """State persists within the same session."""
    async with ShellPool() as pool:
        await pool.execute(language="bash", command="MY_VAR=pooled_value")
        result = await pool.execute(language="bash", command="echo $MY_VAR")
        assert result.exit_code == 0
        assert "pooled_value" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_function_persists_within_session():
    """Functions persist within the same session."""
    async with ShellPool() as pool:
        await pool.execute(
            language="bash",
            command='pool_greet() { echo "Hello from pool, $1!"; }',
        )
        result = await pool.execute(language="bash", command="pool_greet World")
        assert result.exit_code == 0
        assert "Hello from pool, World!" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_cwd_persists_within_session():
    """Working directory changes persist within the same session."""
    async with ShellPool() as pool:
        await pool.execute(language="bash", command="cd /tmp")
        result = await pool.execute(language="bash", command="pwd")
        assert result.exit_code == 0
        assert "/tmp" in result.stdout or "/private/tmp" in result.stdout


# ────────────────────────────────────────────────────────────────────────────
# Session Independence
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_sessions_are_independent():
    """Different session_ids have independent state."""
    async with ShellPool() as pool:
        # Set different values in different sessions
        await pool.execute(
            language="bash",
            command="MY_VAR=session0_value",
            session_id=0,
        )
        await pool.execute(
            language="bash",
            command="MY_VAR=session1_value",
            session_id=1,
        )

        # Verify they're independent
        result0 = await pool.execute(
            language="bash",
            command="echo $MY_VAR",
            session_id=0,
        )
        result1 = await pool.execute(
            language="bash",
            command="echo $MY_VAR",
            session_id=1,
        )

        assert "session0_value" in result0.stdout
        assert "session1_value" in result1.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_different_languages_independent():
    """Different languages have independent sessions."""
    if not os.path.exists("/bin/zsh"):
        pytest.skip("zsh not available")

    async with ShellPool() as pool:
        # Set in bash
        await pool.execute(language="bash", command="LANG_VAR=bash_value")
        # Set in zsh
        await pool.execute(language="zsh", command="LANG_VAR=zsh_value")

        # Verify they're independent
        bash_result = await pool.execute(language="bash", command="echo $LANG_VAR")
        zsh_result = await pool.execute(language="zsh", command="echo $LANG_VAR")

        assert "bash_value" in bash_result.stdout
        assert "zsh_value" in zsh_result.stdout


# ────────────────────────────────────────────────────────────────────────────
# Session Management
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_get_active_sessions():
    """Can list all active sessions."""
    async with ShellPool() as pool:
        assert pool.get_active_sessions() == []

        await pool.execute(language="bash", command="echo a", session_id=0)
        assert pool.get_active_sessions() == [("bash", 0)]

        await pool.execute(language="bash", command="echo b", session_id=1)
        active = pool.get_active_sessions()
        assert len(active) == 2
        assert ("bash", 0) in active
        assert ("bash", 1) in active


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_has_session():
    """Can check if a session exists."""
    async with ShellPool() as pool:
        assert not pool.has_session(language="bash", session_id=0)

        await pool.execute(language="bash", command="echo test")
        assert pool.has_session(language="bash", session_id=0)
        assert not pool.has_session(language="bash", session_id=1)


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_close_specific_session():
    """Can close a specific session while keeping others alive."""
    async with ShellPool() as pool:
        await pool.execute(language="bash", command="echo a", session_id=0)
        await pool.execute(language="bash", command="echo b", session_id=1)

        assert len(pool.get_active_sessions()) == 2

        closed = await pool.close_session(language="bash", session_id=0)
        assert closed is True

        assert len(pool.get_active_sessions()) == 1
        assert ("bash", 1) in pool.get_active_sessions()
        assert not pool.has_session(language="bash", session_id=0)


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_close_nonexistent_session():
    """Closing a nonexistent session returns False."""
    async with ShellPool() as pool:
        closed = await pool.close_session(language="bash", session_id=99)
        assert closed is False


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_get_session():
    """Can get direct access to a session."""
    async with ShellPool() as pool:
        session = await pool.get_session(language="bash", session_id=0)
        assert session.is_running

        # Using the session directly should work
        result = await session.execute("echo direct")
        assert "direct" in result.stdout

        # State should persist through pool.execute too
        await session.execute("DIRECT_VAR=direct_value")
        result = await pool.execute(language="bash", command="echo $DIRECT_VAR")
        assert "direct_value" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_session_reuse():
    """Same session is reused for same (language, session_id)."""
    async with ShellPool() as pool:
        session1 = await pool.get_session(language="bash", session_id=0)
        session2 = await pool.get_session(language="bash", session_id=0)
        assert session1 is session2


# ────────────────────────────────────────────────────────────────────────────
# Initial Configuration
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_custom_env():
    """Sessions can be created with custom environment variables."""
    async with ShellPool() as pool:
        result = await pool.execute(
            language="bash",
            command="echo $POOL_TEST_VAR",
            env={"POOL_TEST_VAR": "pool_env_value"},
        )
        assert "pool_env_value" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_custom_cwd():
    """Sessions can be created with custom working directory."""
    async with ShellPool() as pool:
        result = await pool.execute(
            language="bash",
            command="pwd",
            cwd="/tmp",
        )
        assert "/tmp" in result.stdout or "/private/tmp" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_env_ignored_for_existing_session():
    """Environment variables are ignored if session already exists."""
    async with ShellPool() as pool:
        # Create session with initial env
        await pool.execute(
            language="bash",
            command="echo $FIRST_VAR",
            env={"FIRST_VAR": "first"},
        )

        # Try to use different env - should be ignored
        result = await pool.execute(
            language="bash",
            command="echo $FIRST_VAR $SECOND_VAR",
            env={"SECOND_VAR": "second"},
        )
        # FIRST_VAR should still be there, SECOND_VAR should not
        assert "first" in result.stdout


# ────────────────────────────────────────────────────────────────────────────
# Timeout Handling
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_timeout():
    """Timeout is passed through to execute."""
    async with ShellPool() as pool:
        result = await pool.execute(
            language="bash",
            command="sleep 10",
            timeout=0.5,
        )
        assert result.exit_code == -1
        assert result.error is not None
        assert "timed out" in result.error.lower()


# ────────────────────────────────────────────────────────────────────────────
# Multi-Language Support
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_zsh():
    """Pool works with zsh sessions."""
    if not os.path.exists("/bin/zsh"):
        pytest.skip("zsh not available")

    async with ShellPool() as pool:
        await pool.execute(language="zsh", command="ZSH_VAR=zsh_pool_value")
        result = await pool.execute(language="zsh", command="echo $ZSH_VAR")
        assert "zsh_pool_value" in result.stdout


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_sh():
    """Pool works with POSIX sh sessions."""
    async with ShellPool() as pool:
        await pool.execute(language="sh", command="SH_VAR=sh_pool_value")
        result = await pool.execute(language="sh", command="echo $SH_VAR")
        assert "sh_pool_value" in result.stdout


# ────────────────────────────────────────────────────────────────────────────
# Session Metadata, State Inspection, and Limits
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_get_all_sessions_and_metadata_updates():
    pool = ShellPool()
    try:
        await pool.execute(language="bash", command="echo hi", session_id=0)
        sessions = pool.get_all_sessions()
        assert len(sessions) == 1
        s0 = sessions[0]
        assert s0["language"] == "bash"
        assert s0["session_id"] == 0
        assert "created_at" in s0 and "last_used" in s0

        # Use the session again and ensure last_used is updated (monotonic-ish).
        await pool.execute(language="bash", command="echo hi2", session_id=0)
        sessions2 = pool.get_all_sessions()
        assert len(sessions2) == 1
        assert sessions2[0]["last_used"] >= s0["last_used"]
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_get_session_state_detail_levels():
    pool = ShellPool()
    try:
        await pool.execute(language="bash", command="export FOO=bar", session_id=0)
        st = await pool.get_session_state(
            language="bash",
            session_id=0,
            detail="summary",
        )
        assert "cwd" in st
        assert "variables" in st
        assert isinstance(st["variables"], list)

        st2 = await pool.get_session_state(
            language="bash",
            session_id=0,
            detail="full",
        )
        assert "variables" in st2
        assert isinstance(st2["variables"], str)
    finally:
        await pool.close()


@_handle_project
@pytest.mark.asyncio
async def test_shell_pool_session_limit_enforced():
    pool = ShellPool(max_total_sessions=1)
    try:
        r1 = await pool.execute(language="bash", command="echo one", session_id=0)
        assert r1.error is None

        # Creating a different session should exceed the global limit.
        r2 = await pool.execute(language="bash", command="echo two", session_id=1)
        assert r2.error is not None
        assert r2.error_type == "resource_limit"
    finally:
        await pool.close()
