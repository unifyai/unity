"""
Regression tests: shell_env_id must be stored when passed through add_functions
for shell-language functions.

These tests verify that calling add_functions(language="bash", shell_env_id=X)
actually persists the shell_env_id on the function record. Previously,
add_functions dispatched to _add_shell_functions which did not accept or
forward shell_env_id, silently dropping it.
"""

import os
import stat
import tempfile

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

SIMPLE_BASH = """
# @name: greet
# @args: (name)
# @description: Greet someone
echo "Hello, $1!"
""".strip()

SIMPLE_ZSH = """
# @name: zsh_greet
# @args: (name)
# @description: Greet from zsh
echo "Hello, $1!"
""".strip()


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
def fake_binary():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "my_tool")
        with open(p, "wb") as f:
            f.write(b"#!/bin/sh\necho ok\n")
        os.chmod(p, stat.S_IRWXU)
        yield p


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_bash_with_shell_env_id(
    function_manager_factory,
    fake_binary,
):
    """add_functions(language='bash', shell_env_id=X) must persist shell_env_id."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binary])

    fm.add_functions(
        implementations=[SIMPLE_BASH],
        language="bash",
        shell_env_id=env_id,
    )

    functions = fm.list_functions()
    assert "greet" in functions
    assert functions["greet"]["shell_env_id"] == env_id


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_zsh_with_shell_env_id(
    function_manager_factory,
    fake_binary,
):
    """add_functions(language='zsh', shell_env_id=X) must persist shell_env_id."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binary])

    fm.add_functions(
        implementations=[SIMPLE_ZSH],
        language="zsh",
        shell_env_id=env_id,
    )

    functions = fm.list_functions()
    assert "zsh_greet" in functions
    assert functions["zsh_greet"]["shell_env_id"] == env_id


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_shell_without_shell_env_id(function_manager_factory):
    """add_functions(language='bash') without shell_env_id should store None."""
    fm = function_manager_factory()

    fm.add_functions(
        implementations=[SIMPLE_BASH],
        language="bash",
    )

    functions = fm.list_functions()
    assert "greet" in functions
    assert functions["greet"]["shell_env_id"] is None


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_shell_env_id_survives_overwrite(
    function_manager_factory,
    fake_binary,
):
    """Overwriting a shell function with shell_env_id preserves the env link."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="tools", tool_paths=[fake_binary])

    fm.add_functions(implementations=[SIMPLE_BASH], language="bash")

    updated = """
# @name: greet
# @args: (name --loud)
# @description: Greet someone loudly
echo "HELLO, $1!"
""".strip()

    fm.add_functions(
        implementations=[updated],
        language="bash",
        overwrite=True,
        shell_env_id=env_id,
    )

    functions = fm.list_functions()
    assert functions["greet"]["shell_env_id"] == env_id
