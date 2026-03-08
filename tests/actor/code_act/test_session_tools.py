from __future__ import annotations

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
from unity.actor.environments.state_managers import StateManagerEnvironment


@pytest.mark.asyncio
async def test_session_tools_list_inspect_and_close():
    actor = CodeActActor(environments=[StateManagerEnvironment()])

    sandbox = PythonExecutionSession(environments=actor.environments, venv_pool=actor._venv_pool, shell_pool=actor._shell_pool)  # type: ignore[attr-defined]
    sandbox.global_state["foo"] = 123
    token = _CURRENT_SANDBOX.set(sandbox)

    try:
        tools = actor.get_tools("act")

        # Create a shell session so list_sessions can see it.
        await actor._shell_pool.execute(language="bash", command="echo hi", session_id=0)  # type: ignore[attr-defined]
        actor._register_session_name(name="git_context", language="bash", venv_id=None, session_id=0)  # type: ignore[attr-defined]

        sessions = await tools["list_sessions"](detail="summary")
        assert "sessions" in sessions
        assert any(
            s.get("language") == "bash" and s.get("session_id") == 0
            for s in sessions["sessions"]
        )

        inspected_default = await tools["inspect_state"](detail="summary")
        assert inspected_default["session"]["language"] == "python"
        assert "foo" in inspected_default["state"]["variables"]

        inspected_shell = await tools["inspect_state"](
            session_name="git_context",
            detail="summary",
        )
        assert inspected_shell["session"]["language"] == "bash"
        assert "cwd" in inspected_shell["state"]

        closed = await tools["close_session"](session_name="git_context")
        assert closed["closed"] is True

        # Idempotent: close again returns not_found (no exception).
        closed2 = await tools["close_session"](session_name="git_context")
        assert closed2["closed"] is False
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()


@pytest.mark.asyncio
async def test_session_tools_close_all_sessions():
    actor = CodeActActor(environments=[StateManagerEnvironment()])

    sandbox = PythonExecutionSession(environments=actor.environments, venv_pool=actor._venv_pool, shell_pool=actor._shell_pool)  # type: ignore[attr-defined]
    token = _CURRENT_SANDBOX.set(sandbox)

    try:
        tools = actor.get_tools("act")
        await actor._shell_pool.execute(language="bash", command="export FOO=bar", session_id=0)  # type: ignore[attr-defined]
        actor._register_session_name(name="repo_nav", language="bash", venv_id=None, session_id=0)  # type: ignore[attr-defined]

        out = await tools["close_all_sessions"]()
        assert out["closed_count"] >= 1
        assert "bash" in out["details"]
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()


@pytest.mark.asyncio
async def test_list_sessions_includes_shell_env_id():
    """Shell sessions created with shell_env_id show it in list_sessions output."""
    actor = CodeActActor(environments=[StateManagerEnvironment()])

    sandbox = PythonExecutionSession(
        environments=actor.environments,
        venv_pool=actor._venv_pool,
        shell_pool=actor._shell_pool,
    )
    token = _CURRENT_SANDBOX.set(sandbox)

    try:
        tools = actor.get_tools("act")

        await actor._shell_pool.execute(
            language="bash",
            command="echo test",
            session_id=0,
        )
        actor._register_session_name(
            name="jq_session",
            language="bash",
            venv_id=None,
            shell_env_id=42,
            session_id=0,
        )

        sessions = await tools["list_sessions"](detail="summary")
        bash_sessions = [s for s in sessions["sessions"] if s.get("language") == "bash"]
        assert len(bash_sessions) >= 1
        assert bash_sessions[0].get("shell_env_id") == 42
        assert bash_sessions[0].get("session_name") == "jq_session"
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()


@pytest.mark.asyncio
async def test_session_name_resolves_with_shell_env_id():
    """Session name registered with shell_env_id resolves to the full 4-tuple key."""
    actor = CodeActActor()

    actor._register_session_name(
        name="cloud_tools",
        language="bash",
        venv_id=None,
        shell_env_id=99,
        session_id=5,
    )

    resolved = actor._resolve_session_name("cloud_tools")
    assert resolved is not None
    lang, venv_id, shell_env_id, session_id = resolved
    assert lang == "bash"
    assert venv_id is None
    assert shell_env_id == 99
    assert session_id == 5
