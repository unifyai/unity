"""
Tests for shell environment preparation and execution.

Tests that:
- prepare_shell_env materializes binaries on disk from DB
- is_shell_env_ready validates local cache
- Shell functions with shell_env_id get the env's bin/ on PATH
- PATH injection works for stateless, stateful, and read_only modes
- Different shell envs provide isolation
"""

import os
import shutil
import stat
import tempfile

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.shell_pool import ShellPool
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


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
    """Create a temporary fake binary that outputs its own name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "unity_test_tool")
        with open(p, "w") as f:
            f.write("#!/bin/sh\necho unity_test_tool_output\n")
        os.chmod(p, stat.S_IRWXU)
        yield p


@pytest.fixture
def fake_tool_pair():
    """Create two distinct temporary fake binaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = {}
        for name, output in [("tool_alpha", "alpha_out"), ("tool_beta", "beta_out")]:
            p = os.path.join(tmpdir, name)
            with open(p, "w") as f:
                f.write(f"#!/bin/sh\necho {output}\n")
            os.chmod(p, stat.S_IRWXU)
            tools[name] = p
        yield tools


# ────────────────────────────────────────────────────────────────────────────
# Prepare / Ready Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_prepare_shell_env(function_manager_factory, fake_tool):
    """prepare_shell_env creates the local bin dir with the tool binary."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="prep-test", tool_paths=[fake_tool])

    # Wipe the eagerly-created local cache to force a DB restore
    env_dir = fm._get_shell_env_dir(env_id)
    if env_dir.exists():
        shutil.rmtree(env_dir)

    bin_dir = await fm.prepare_shell_env(shell_env_id=env_id)

    assert bin_dir.exists()
    tool_path = bin_dir / "unity_test_tool"
    assert tool_path.exists()
    assert os.access(str(tool_path), os.X_OK)


@_handle_project
@pytest.mark.asyncio
async def test_is_shell_env_ready(function_manager_factory, fake_tool):
    """is_shell_env_ready returns True when local cache is valid, False otherwise."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="ready-test", tool_paths=[fake_tool])

    assert fm.is_shell_env_ready(shell_env_id=env_id) is True

    # Remove local dir → should be not ready
    env_dir = fm._get_shell_env_dir(env_id)
    if env_dir.exists():
        shutil.rmtree(env_dir)
    assert fm.is_shell_env_ready(shell_env_id=env_id) is False

    # Restore via prepare → ready again
    await fm.prepare_shell_env(shell_env_id=env_id)
    assert fm.is_shell_env_ready(shell_env_id=env_id) is True


@_handle_project
@pytest.mark.asyncio
async def test_prepare_shell_env_idempotent(function_manager_factory, fake_tool):
    """Calling prepare_shell_env twice returns the same path without error."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="idempotent", tool_paths=[fake_tool])

    path1 = await fm.prepare_shell_env(shell_env_id=env_id)
    path2 = await fm.prepare_shell_env(shell_env_id=env_id)
    assert path1 == path2


@_handle_project
@pytest.mark.asyncio
async def test_prepare_shell_env_not_found(function_manager_factory):
    """prepare_shell_env raises ValueError for non-existent env."""
    fm = function_manager_factory()
    with pytest.raises(ValueError, match="not found"):
        await fm.prepare_shell_env(shell_env_id=99999)


# ────────────────────────────────────────────────────────────────────────────
# Execution With Shell Env Tests
# ────────────────────────────────────────────────────────────────────────────


SCRIPT_CALLING_TOOL = """
# @name: call_tool
# @args: ()
# @description: Runs unity_test_tool
unity_test_tool
""".strip()


def _add_shell_function_with_env(fm, implementations, language, shell_env_id):
    """Add a shell function and link it to a shell env via set_function_shell_env."""
    fm.add_functions(implementations=implementations, language=language)
    funcs = fm.list_functions()
    # Link each newly-added function
    if isinstance(implementations, str):
        implementations = [implementations]
    for impl in implementations:
        for name, data in funcs.items():
            if data.get("shell_env_id") is None:
                fm.set_function_shell_env(
                    function_id=data["function_id"],
                    shell_env_id=shell_env_id,
                )


@_handle_project
@pytest.mark.asyncio
async def test_execute_shell_function_with_env_stateless(
    function_manager_factory,
    fake_tool,
):
    """A stateless shell function with shell_env_id can invoke env tools."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="exec-test", tool_paths=[fake_tool])
    _add_shell_function_with_env(fm, [SCRIPT_CALLING_TOOL], "bash", env_id)

    result = await fm.execute_function(function_name="call_tool")
    assert result["error"] is None
    assert "unity_test_tool_output" in result["stdout"]


@_handle_project
@pytest.mark.asyncio
async def test_execute_shell_function_with_env_stateful(
    function_manager_factory,
    fake_tool,
):
    """A stateful shell function with shell_env_id can invoke env tools."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="stateful-test", tool_paths=[fake_tool])
    _add_shell_function_with_env(fm, [SCRIPT_CALLING_TOOL], "bash", env_id)

    async with ShellPool() as pool:
        result = await fm.execute_function(
            function_name="call_tool",
            state_mode="stateful",
            shell_pool=pool,
        )
    assert result["error"] is None
    assert "unity_test_tool_output" in result["stdout"]


@_handle_project
@pytest.mark.asyncio
async def test_env_path_prepended(function_manager_factory, fake_tool):
    """The shell env's bin/ dir appears at the start of PATH."""
    fm = function_manager_factory()
    env_id = fm.add_shell_env(name="path-test", tool_paths=[fake_tool])

    path_script = """
# @name: show_path
# @args: ()
# @description: Print PATH
echo "$PATH"
""".strip()

    fm.add_functions(implementations=[path_script], language="bash")
    func_id = fm.list_functions()["show_path"]["function_id"]
    fm.set_function_shell_env(function_id=func_id, shell_env_id=env_id)

    result = await fm.execute_function(function_name="show_path")
    assert result["error"] is None

    bin_dir = str(fm._get_shell_env_bin_dir(env_id))
    first_path_entry = result["stdout"].strip().split(":")[0]
    assert first_path_entry == bin_dir


@_handle_project
@pytest.mark.asyncio
async def test_env_isolation(function_manager_factory, fake_tool_pair):
    """Two envs with different tools provide isolation."""
    fm = function_manager_factory()

    env_alpha = fm.add_shell_env(
        name="alpha",
        tool_paths=[fake_tool_pair["tool_alpha"]],
    )
    env_beta = fm.add_shell_env(
        name="beta",
        tool_paths=[fake_tool_pair["tool_beta"]],
    )

    script_alpha = """
# @name: run_alpha
# @args: ()
# @description: Run tool_alpha
tool_alpha
""".strip()

    script_beta = """
# @name: run_beta
# @args: ()
# @description: Run tool_beta
tool_beta
""".strip()

    fm.add_functions(implementations=[script_alpha], language="bash")
    func_a = fm.list_functions()["run_alpha"]["function_id"]
    fm.set_function_shell_env(function_id=func_a, shell_env_id=env_alpha)

    fm.add_functions(implementations=[script_beta], language="bash")
    func_b = fm.list_functions()["run_beta"]["function_id"]
    fm.set_function_shell_env(function_id=func_b, shell_env_id=env_beta)

    result_a = await fm.execute_function(function_name="run_alpha")
    assert result_a["error"] is None
    assert "alpha_out" in result_a["stdout"]

    result_b = await fm.execute_function(function_name="run_beta")
    assert result_b["error"] is None
    assert "beta_out" in result_b["stdout"]
