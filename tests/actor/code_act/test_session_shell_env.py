"""
Tests for shell_env_id support in SessionExecutor and _execute_shell_stateless.

Verifies that:
- SessionExecutor.execute() with shell_env_id injects PATH for all state modes
- _execute_shell_stateless() respects env overrides
- Shell env PATH is prepended correctly
- ExecutionResult includes shell_env_id
"""

import os
import stat
import tempfile

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.shell_pool import ShellPool
from unity.actor.execution.session import SessionExecutor, _execute_shell_stateless
from unity.actor.execution.types import ExecutionResult
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


@pytest.fixture
def function_manager_factory():
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/ShellEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/ShellEnvBinaries")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager(include_primitives=False)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def fake_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "session_test_tool")
        with open(p, "w") as f:
            f.write("#!/bin/sh\necho session_test_output\n")
        os.chmod(p, stat.S_IRWXU)
        yield p


# ────────────────────────────────────────────────────────────────────────────
# _execute_shell_stateless with env
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_shell_stateless_with_env():
    """_execute_shell_stateless passes env overrides to the subprocess."""
    result = await _execute_shell_stateless(
        language="bash",
        command='echo "MARKER=$UNITY_TEST_MARKER"',
        env={"UNITY_TEST_MARKER": "found_it"},
    )
    assert result["error"] is None
    assert "MARKER=found_it" in result["stdout"]


@pytest.mark.asyncio
async def test_execute_shell_stateless_without_env():
    """Without env, _execute_shell_stateless uses the parent environment."""
    result = await _execute_shell_stateless(
        language="bash",
        command='echo "MARKER=$UNITY_TEST_MARKER_ABSENT"',
    )
    assert result["error"] is None
    assert "MARKER=" in result["stdout"]
    assert "found_it" not in result["stdout"]


# ────────────────────────────────────────────────────────────────────────────
# SessionExecutor with shell_env_id
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_session_executor_shell_stateless_with_env(
    function_manager_factory,
    fake_tool,
):
    """SessionExecutor.execute with shell_env_id makes env tools available (stateless)."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="test-env", tool_paths=[fake_tool])

    async with ShellPool() as pool:
        executor = SessionExecutor(
            venv_pool=None,
            shell_pool=pool,
            function_manager=fm,
        )
        result = await executor.execute(
            code="session_test_tool",
            language="bash",
            state_mode="stateless",
            session_id=None,
            venv_id=None,
            shell_env_id=env_id,
        )

    assert result["error"] is None
    assert "session_test_output" in result["stdout"]
    assert result.get("shell_env_id") == env_id


@_handle_project
@pytest.mark.asyncio
async def test_session_executor_shell_stateless_without_env(
    function_manager_factory,
    fake_tool,
):
    """Without shell_env_id, the tool binary is NOT on PATH (negative test)."""
    fm = function_manager_factory()
    fm.add_shell_env(name="test-env", tool_paths=[fake_tool])

    async with ShellPool() as pool:
        executor = SessionExecutor(
            venv_pool=None,
            shell_pool=pool,
            function_manager=fm,
        )
        result = await executor.execute(
            code="session_test_tool",
            language="bash",
            state_mode="stateless",
            session_id=None,
            venv_id=None,
            shell_env_id=None,
        )

    assert result["result"] != 0 or "not found" in result.get("stderr", "").lower()


@_handle_project
@pytest.mark.asyncio
async def test_session_executor_shell_stateful_with_env(
    function_manager_factory,
    fake_tool,
):
    """SessionExecutor.execute with shell_env_id in stateful mode."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="stateful-env", tool_paths=[fake_tool])

    async with ShellPool() as pool:
        executor = SessionExecutor(
            venv_pool=None,
            shell_pool=pool,
            function_manager=fm,
        )
        result = await executor.execute(
            code="session_test_tool",
            language="bash",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
            shell_env_id=env_id,
        )

    assert result["error"] is None
    assert "session_test_output" in result["stdout"]
    assert result.get("shell_env_id") == env_id


@_handle_project
@pytest.mark.asyncio
async def test_session_executor_shell_env_path_prepended(
    function_manager_factory,
    fake_tool,
):
    """The shell env's bin/ dir appears first on PATH."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="path-env", tool_paths=[fake_tool])

    async with ShellPool() as pool:
        executor = SessionExecutor(
            venv_pool=None,
            shell_pool=pool,
            function_manager=fm,
        )
        result = await executor.execute(
            code='echo "$PATH"',
            language="bash",
            state_mode="stateless",
            session_id=None,
            venv_id=None,
            shell_env_id=env_id,
        )

    assert result["error"] is None
    bin_dir = str(fm._get_shell_env_bin_dir(env_id))
    first_entry = result["stdout"].strip().split(":")[0]
    assert first_entry == bin_dir


# ────────────────────────────────────────────────────────────────────────────
# ExecutionResult shell_env_id
# ────────────────────────────────────────────────────────────────────────────


def test_execution_result_includes_shell_env_id():
    """ExecutionResult model includes shell_env_id field."""
    er = ExecutionResult(shell_env_id=5)
    assert er.shell_env_id == 5

    content = er.to_llm_content()
    text = content[0]["text"]
    assert "shell_env_id" in text
    assert "5" in text


def test_execution_result_omits_shell_env_id_when_none():
    """ExecutionResult omits shell_env_id from LLM output when None."""
    er = ExecutionResult(language="bash")
    content = er.to_llm_content()
    text = content[0]["text"]
    assert "shell_env_id" not in text
