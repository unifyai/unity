"""Tests for shell env management tools in the storage loop.

Verifies that:
1. All shell env CRUD tools appear in ``_build_storage_tools`` output.
2. ``FunctionManager_add_functions`` wrapper accepts ``shell_env_id``.
3. The wrappers correctly delegate to the underlying FunctionManager methods.
4. Shell env tools have docstrings for LLM consumption.
5. Storage loop prompts include shell function guidance.
"""

import inspect
import os
import stat
import tempfile
from unittest.mock import MagicMock

import pytest

from unity.actor.code_act_actor import _build_storage_tools, CodeActActor
from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


class _MinimalGuidanceManager:
    """Stub GuidanceManager that satisfies _build_storage_tools."""

    def search(self, references=None, k=10):
        return []

    def filter(self, filter=None, offset=0, limit=100):
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        return {"deleted": True}


def _make_actor_with_mocks():
    fm = MagicMock()
    fm._include_primitives = False
    fm.search_functions = MagicMock(return_value=[])
    fm.filter_functions = MagicMock(return_value=[])
    fm.list_functions = MagicMock(return_value={})
    fm.add_functions = MagicMock(return_value={"test_fn": "added"})
    fm.delete_function = MagicMock(return_value=True)
    fm.add_venv = MagicMock(return_value=42)
    fm.list_venvs = MagicMock(return_value=[])
    fm.get_venv = MagicMock(return_value=None)
    fm.update_venv = MagicMock(return_value=True)
    fm.delete_venv = MagicMock(return_value=True)
    fm.set_function_venv = MagicMock(return_value=True)
    fm.get_function_venv = MagicMock(return_value=None)
    fm.add_shell_env = MagicMock(return_value=7)
    fm.list_shell_envs = MagicMock(return_value=[])
    fm.get_shell_env = MagicMock(return_value=None)
    fm.update_shell_env = MagicMock(return_value=True)
    fm.delete_shell_env = MagicMock(return_value=True)
    fm.set_function_shell_env = MagicMock(return_value=True)
    fm.get_function_shell_env = MagicMock(return_value=None)

    gm = _MinimalGuidanceManager()

    actor = MagicMock(spec=CodeActActor)
    actor.function_manager = fm
    actor.guidance_manager = gm
    return actor, fm


EXPECTED_SHELL_ENV_TOOLS = {
    "FunctionManager_add_shell_env",
    "FunctionManager_list_shell_envs",
    "FunctionManager_get_shell_env",
    "FunctionManager_update_shell_env",
    "FunctionManager_delete_shell_env",
    "FunctionManager_set_function_shell_env",
    "FunctionManager_get_function_shell_env",
}


# ────────────────────────────────────────────────────────────────────────────
# Tool presence tests
# ────────────────────────────────────────────────────────────────────────────


def test_shell_env_tools_present_in_storage_tools():
    """All shell env CRUD tools appear in _build_storage_tools output."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})
    for name in EXPECTED_SHELL_ENV_TOOLS:
        assert (
            name in tools
        ), f"Expected '{name}' in storage tools, got: {sorted(tools.keys())}"


def test_shell_env_tools_have_docstrings():
    """All shell env tools have non-empty docstrings for LLM consumption."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    for name in EXPECTED_SHELL_ENV_TOOLS:
        fn = tools[name]
        doc = fn.__doc__
        assert (
            doc and len(doc) > 20
        ), f"Tool '{name}' should have a meaningful docstring, got: {doc!r}"


# ────────────────────────────────────────────────────────────────────────────
# Delegation tests (mock-based)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_functions_wrapper_forwards_shell_env_id():
    """FunctionManager_add_functions wrapper accepts and forwards shell_env_id."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    add_fn = tools["FunctionManager_add_functions"]
    sig = inspect.signature(add_fn)
    assert "shell_env_id" in sig.parameters

    await add_fn("echo hello", language="bash", shell_env_id=7)
    fm.add_functions.assert_called_once_with(
        implementations="echo hello",
        language="bash",
        overwrite=False,
        venv_id=None,
        shell_env_id=7,
    )


@pytest.mark.asyncio
async def test_add_shell_env_wrapper_delegates():
    """FunctionManager_add_shell_env delegates to fm.add_shell_env."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    result = await tools["FunctionManager_add_shell_env"](
        tool_paths=["/usr/local/bin/jq"],
        name="data-tools",
    )
    fm.add_shell_env.assert_called_once_with(
        name="data-tools",
        tool_paths=["/usr/local/bin/jq"],
    )
    assert result == 7


@pytest.mark.asyncio
async def test_list_shell_envs_wrapper_delegates():
    """FunctionManager_list_shell_envs delegates to fm.list_shell_envs."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_list_shell_envs"]()
    fm.list_shell_envs.assert_called_once()


@pytest.mark.asyncio
async def test_get_shell_env_wrapper_delegates():
    """FunctionManager_get_shell_env delegates to fm.get_shell_env."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_get_shell_env"](shell_env_id=7)
    fm.get_shell_env.assert_called_once_with(shell_env_id=7)


@pytest.mark.asyncio
async def test_update_shell_env_wrapper_delegates():
    """FunctionManager_update_shell_env delegates to fm.update_shell_env."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_update_shell_env"](
        shell_env_id=7,
        name="updated",
        tool_paths=["/usr/bin/new"],
    )
    fm.update_shell_env.assert_called_once_with(
        shell_env_id=7,
        name="updated",
        tool_paths=["/usr/bin/new"],
    )


@pytest.mark.asyncio
async def test_delete_shell_env_wrapper_delegates():
    """FunctionManager_delete_shell_env delegates to fm.delete_shell_env."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_delete_shell_env"](shell_env_id=7)
    fm.delete_shell_env.assert_called_once_with(shell_env_id=7)


@pytest.mark.asyncio
async def test_set_function_shell_env_wrapper_delegates():
    """FunctionManager_set_function_shell_env delegates correctly."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_set_function_shell_env"](
        function_id=10,
        shell_env_id=7,
    )
    fm.set_function_shell_env.assert_called_once_with(
        function_id=10,
        shell_env_id=7,
    )


@pytest.mark.asyncio
async def test_get_function_shell_env_wrapper_delegates():
    """FunctionManager_get_function_shell_env delegates correctly."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_get_function_shell_env"](function_id=10)
    fm.get_function_shell_env.assert_called_once_with(function_id=10)


# ────────────────────────────────────────────────────────────────────────────
# Real FunctionManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def real_function_manager_factory():
    """Create a real FunctionManager for storage tool tests."""
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


def _make_actor_with_real_fm(fm):
    gm = _MinimalGuidanceManager()
    actor = MagicMock(spec=CodeActActor)
    actor.function_manager = fm
    actor.guidance_manager = gm
    return actor


@_handle_project
@pytest.mark.asyncio
async def test_add_shell_env_with_real_fm(real_function_manager_factory):
    """End-to-end: create shell env via storage tool, verify retrieval."""
    fm = real_function_manager_factory()
    actor = _make_actor_with_real_fm(fm)
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    with tempfile.TemporaryDirectory() as tmpdir:
        tool_path = os.path.join(tmpdir, "test_binary")
        with open(tool_path, "wb") as f:
            f.write(b"#!/bin/sh\necho ok\n")
        os.chmod(tool_path, stat.S_IRWXU)

        env_id = await tools["FunctionManager_add_shell_env"](
            tool_paths=[tool_path],
            name="real-test",
        )

    assert isinstance(env_id, int)
    envs = await tools["FunctionManager_list_shell_envs"]()
    assert any(e["shell_env_id"] == env_id for e in envs)

    env = await tools["FunctionManager_get_shell_env"](shell_env_id=env_id)
    assert env is not None
    assert env["name"] == "real-test"


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_with_shell_env_id_real_fm(real_function_manager_factory):
    """End-to-end: store a shell function and link to a shell env via set_function_shell_env."""
    fm = real_function_manager_factory()

    with tempfile.TemporaryDirectory() as tmpdir:
        tool_path = os.path.join(tmpdir, "my_tool")
        with open(tool_path, "wb") as f:
            f.write(b"#!/bin/sh\necho tool_output\n")
        os.chmod(tool_path, stat.S_IRWXU)

        env_id = fm.add_shell_env(name="linked-test", tool_paths=[tool_path])

    assert isinstance(env_id, int)

    bash_func = """
# @name: use_tool
# @args: ()
# @description: Use my_tool
my_tool
""".strip()

    result = fm.add_functions(
        implementations=[bash_func],
        language="bash",
    )
    assert result.get("use_tool") == "added"

    func_id = fm.list_functions()["use_tool"]["function_id"]

    # Link function to shell env explicitly
    linked = fm.set_function_shell_env(function_id=func_id, shell_env_id=env_id)
    assert linked is True

    env = fm.get_function_shell_env(function_id=func_id)
    assert env is not None
    assert env["shell_env_id"] == env_id


# ────────────────────────────────────────────────────────────────────────────
# Storage prompt content tests
# ────────────────────────────────────────────────────────────────────────────


def test_storage_prompt_mentions_shell_functions():
    """The _STORAGE_WHAT_CAN_BE_STORED prompt mentions shell scripts as storable."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "shell" in _STORAGE_WHAT_CAN_BE_STORED.lower()
    assert "language=" in _STORAGE_WHAT_CAN_BE_STORED
    assert "bash" in _STORAGE_WHAT_CAN_BE_STORED


def test_storage_prompt_shell_metadata_format():
    """The prompt shows the required shell metadata comment format."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "# @name:" in _STORAGE_WHAT_CAN_BE_STORED
    assert "# @args:" in _STORAGE_WHAT_CAN_BE_STORED
    assert "# @description:" in _STORAGE_WHAT_CAN_BE_STORED


def test_storage_prompt_shell_env_workflow():
    """The prompt describes the shell env workflow for CLI tool dependencies."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "FunctionManager_add_shell_env" in _STORAGE_WHAT_CAN_BE_STORED
    assert "FunctionManager_list_shell_envs" in _STORAGE_WHAT_CAN_BE_STORED
    assert "shell_env_id" in _STORAGE_WHAT_CAN_BE_STORED


def test_storage_prompt_shell_env_in_two_stores():
    """The _STORAGE_TWO_STORES prompt mentions shell env management."""
    from unity.actor.code_act_actor import _STORAGE_TWO_STORES

    assert "shell env" in _STORAGE_TWO_STORES.lower()
    assert "FunctionManager_add_shell_env" in _STORAGE_TWO_STORES
    assert "FunctionManager_set_function_shell_env" in _STORAGE_TWO_STORES


def test_storage_prompt_references_trajectory_paths():
    """The shell env prompt guides the LLM to extract tool paths from trajectory output."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "trajectory" in _STORAGE_WHAT_CAN_BE_STORED.lower()
    assert "tool_paths" in _STORAGE_WHAT_CAN_BE_STORED


def test_add_functions_wrapper_accepts_language_and_shell_env_id():
    """FunctionManager_add_functions accepts both language and shell_env_id params."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    add_fn = tools["FunctionManager_add_functions"]
    sig = inspect.signature(add_fn)
    assert "language" in sig.parameters
    assert "shell_env_id" in sig.parameters


def test_storage_prompt_standalone_environments():
    """The prompt encourages storing environments without corresponding functions."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "standalone environment" in _STORAGE_WHAT_CAN_BE_STORED.lower()
    assert "not an anti-pattern" in _STORAGE_WHAT_CAN_BE_STORED.lower()
    assert "first-class" in _STORAGE_WHAT_CAN_BE_STORED.lower()


def test_storage_prompt_standalone_venv_example():
    """The prompt gives a concrete Python venv example (scientific-visualization)."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "scientific-visualization" in _STORAGE_WHAT_CAN_BE_STORED
    assert "matplotlib" in _STORAGE_WHAT_CAN_BE_STORED


def test_storage_prompt_standalone_shell_env_example():
    """The prompt gives a concrete shell env example (cloud-integrations)."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "cloud-integrations" in _STORAGE_WHAT_CAN_BE_STORED
    assert "gcloud" in _STORAGE_WHAT_CAN_BE_STORED


def test_storage_prompt_descriptive_env_naming():
    """The prompt guides toward domain-oriented environment naming."""
    from unity.actor.code_act_actor import _STORAGE_WHAT_CAN_BE_STORED

    assert "descriptive" in _STORAGE_WHAT_CAN_BE_STORED.lower()
    assert "data-science" in _STORAGE_WHAT_CAN_BE_STORED
