"""
Tests for shell environment storage and foreign key cascading in FunctionManager.

Tests the Functions/ShellEnvs and Functions/ShellEnvBinaries contexts, including:
- Creating and listing shell environments
- Associating functions with shell environments
- Cascading SET NULL on shell env deletion
"""

import os
import stat
import tempfile

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

SIMPLE_BASH_FUNCTION = """
# @name: greet
# @args: (name)
# @description: Greet someone by name
echo "Hello, $1!"
""".strip()

SIMPLE_BASH_FUNCTION_2 = """
# @name: farewell
# @args: (name)
# @description: Say goodbye to someone
echo "Goodbye, $1!"
""".strip()


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/ShellEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/ShellEnvBinaries")
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
def fake_binaries():
    """Create temporary fake binary files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for name in ("tool_a", "tool_b"):
            p = os.path.join(tmpdir, name)
            with open(p, "wb") as f:
                f.write(f"#!/bin/sh\necho {name}\n".encode())
            os.chmod(p, stat.S_IRWXU)
            paths.append(p)
        yield paths


# ────────────────────────────────────────────────────────────────────────────
# Shell Environment CRUD Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_env_crud_operations(function_manager_factory, fake_binaries):
    """Test add, get, list, and delete operations for shell envs."""
    fm = function_manager_factory()

    env_id_1 = fm.add_shell_env(name="env-one", tool_paths=[fake_binaries[0]])
    env_id_2 = fm.add_shell_env(name="env-two", tool_paths=[fake_binaries[1]])

    assert isinstance(env_id_1, int) and env_id_1 >= 0
    assert isinstance(env_id_2, int) and env_id_2 >= 0
    assert env_id_1 != env_id_2

    result = fm.get_shell_env(shell_env_id=env_id_1)
    assert result is not None
    assert result["shell_env_id"] == env_id_1
    assert result["name"] == "env-one"
    assert "tool_a" in result["tools"]

    assert fm.get_shell_env(shell_env_id=99999) is None

    envs = fm.list_shell_envs()
    assert len(envs) == 2
    env_ids = {e["shell_env_id"] for e in envs}
    assert env_id_1 in env_ids and env_id_2 in env_ids

    result = fm.delete_shell_env(shell_env_id=env_id_1)
    assert result is True
    assert fm.get_shell_env(shell_env_id=env_id_1) is None

    assert fm.delete_shell_env(shell_env_id=99999) is False


@_handle_project
@pytest.mark.asyncio
async def test_shell_env_update(function_manager_factory, fake_binaries):
    """Test updating shell env name and tools."""
    fm = function_manager_factory()

    env_id = fm.add_shell_env(name="original", tool_paths=[fake_binaries[0]])

    result = fm.update_shell_env(shell_env_id=env_id, name="renamed")
    assert result is True
    updated = fm.get_shell_env(shell_env_id=env_id)
    assert updated["name"] == "renamed"

    result = fm.update_shell_env(
        shell_env_id=env_id,
        tool_paths=[fake_binaries[1]],
    )
    assert result is True
    updated = fm.get_shell_env(shell_env_id=env_id)
    assert "tool_b" in updated["tools"]

    assert fm.update_shell_env(shell_env_id=99999, name="nope") is False


# ────────────────────────────────────────────────────────────────────────────
# Function-ShellEnv Association Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_function_shell_env_association(function_manager_factory, fake_binaries):
    """Test associating functions with shell envs and removing associations."""
    fm = function_manager_factory()

    fm.add_functions(implementations=[SIMPLE_BASH_FUNCTION], language="bash")
    functions = fm.list_functions()
    func_id = functions["greet"]["function_id"]
    assert fm.get_function_shell_env(function_id=func_id) is None

    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binaries[0]])
    result = fm.set_function_shell_env(function_id=func_id, shell_env_id=env_id)
    assert result is True

    env = fm.get_function_shell_env(function_id=func_id)
    assert env is not None
    assert env["shell_env_id"] == env_id

    result = fm.set_function_shell_env(function_id=func_id, shell_env_id=None)
    assert result is True
    assert fm.get_function_shell_env(function_id=func_id) is None

    assert fm.set_function_shell_env(function_id=99999, shell_env_id=env_id) is False


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_with_shell_env_id(function_manager_factory, fake_binaries):
    """shell_env_id linked to function via set_function_shell_env after add_functions."""
    fm = function_manager_factory()

    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binaries[0]])
    fm.add_functions(
        implementations=[SIMPLE_BASH_FUNCTION],
        language="bash",
    )

    func_id = fm.list_functions()["greet"]["function_id"]
    fm.set_function_shell_env(function_id=func_id, shell_env_id=env_id)

    env = fm.get_function_shell_env(function_id=func_id)
    assert env is not None
    assert env["shell_env_id"] == env_id


# ────────────────────────────────────────────────────────────────────────────
# Cascade Deletion Tests (SET NULL)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_shell_env_deletion_cascades_to_functions(
    function_manager_factory,
    fake_binaries,
):
    """Deleting a shell env should SET NULL on associated functions."""
    fm = function_manager_factory()

    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binaries[0]])
    fm.add_functions(
        implementations=[SIMPLE_BASH_FUNCTION, SIMPLE_BASH_FUNCTION_2],
        language="bash",
    )
    functions = fm.list_functions()

    func_id_1 = functions["greet"]["function_id"]
    func_id_2 = functions["farewell"]["function_id"]

    fm.set_function_shell_env(function_id=func_id_1, shell_env_id=env_id)
    fm.set_function_shell_env(function_id=func_id_2, shell_env_id=env_id)

    assert fm.get_function_shell_env(function_id=func_id_1) is not None
    assert fm.get_function_shell_env(function_id=func_id_2) is not None

    fm.delete_shell_env(shell_env_id=env_id)

    assert fm.get_function_shell_env(function_id=func_id_1) is None
    assert fm.get_function_shell_env(function_id=func_id_2) is None


@_handle_project
@pytest.mark.asyncio
async def test_shell_env_deletion_isolation(function_manager_factory, fake_binaries):
    """Deleting a shell env should not affect functions with other envs."""
    fm = function_manager_factory()

    env_id_1 = fm.add_shell_env(name="env-one", tool_paths=[fake_binaries[0]])
    env_id_2 = fm.add_shell_env(name="env-two", tool_paths=[fake_binaries[1]])

    fm.add_functions(
        implementations=[SIMPLE_BASH_FUNCTION, SIMPLE_BASH_FUNCTION_2],
        language="bash",
    )
    functions = fm.list_functions()

    func_id_1 = functions["greet"]["function_id"]
    func_id_2 = functions["farewell"]["function_id"]

    fm.set_function_shell_env(function_id=func_id_1, shell_env_id=env_id_1)
    fm.set_function_shell_env(function_id=func_id_2, shell_env_id=env_id_2)

    fm.delete_shell_env(shell_env_id=env_id_1)

    assert fm.get_function_shell_env(function_id=func_id_1) is None
    env = fm.get_function_shell_env(function_id=func_id_2)
    assert env is not None
    assert env["shell_env_id"] == env_id_2
