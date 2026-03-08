"""Tests for venv management tools in the storage loop.

Verifies that:
1. All venv CRUD tools appear in ``_build_storage_tools`` output.
2. ``FunctionManager_add_functions`` wrapper accepts ``venv_id``.
3. The wrappers correctly delegate to the underlying FunctionManager methods.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from unity.actor.code_act_actor import _build_storage_tools, CodeActActor


class _MinimalGuidanceManager:
    """Stub GuidanceManager that satisfies _build_storage_tools."""

    def search(self, references=None, k=10):
        """Search for guidance entries."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries."""
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry."""
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update guidance."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete guidance."""
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

    gm = _MinimalGuidanceManager()

    actor = MagicMock(spec=CodeActActor)
    actor.function_manager = fm
    actor.guidance_manager = gm
    return actor, fm


EXPECTED_VENV_TOOLS = {
    "FunctionManager_add_venv",
    "FunctionManager_list_venvs",
    "FunctionManager_get_venv",
    "FunctionManager_update_venv",
    "FunctionManager_delete_venv",
    "FunctionManager_set_function_venv",
    "FunctionManager_get_function_venv",
}

EXPECTED_FM_TOOLS = {
    "FunctionManager_search_functions",
    "FunctionManager_filter_functions",
    "FunctionManager_list_functions",
    "FunctionManager_add_functions",
    "FunctionManager_delete_functions",
}


def test_venv_tools_present_in_storage_tools():
    """All venv CRUD tools appear in _build_storage_tools output."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})
    for name in EXPECTED_VENV_TOOLS:
        assert (
            name in tools
        ), f"Expected '{name}' in storage tools, got: {sorted(tools.keys())}"


def test_fm_tools_still_present():
    """Original FM tools are still present alongside venv tools."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})
    for name in EXPECTED_FM_TOOLS:
        assert name in tools, f"Expected '{name}' in storage tools"


def test_gm_tools_still_present():
    """GuidanceManager tools are still present alongside FM and venv tools."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})
    non_fm_tools = [k for k in tools if not k.startswith("FunctionManager_")]
    assert len(non_fm_tools) >= 3, (
        f"Expected GuidanceManager tools (search/filter/add/update/delete), "
        f"got non-FM tools: {non_fm_tools}"
    )


@pytest.mark.asyncio
async def test_add_functions_wrapper_forwards_venv_id():
    """FunctionManager_add_functions wrapper accepts and forwards venv_id."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    add_fn = tools["FunctionManager_add_functions"]

    sig = inspect.signature(add_fn)
    assert "venv_id" in sig.parameters, (
        f"Expected 'venv_id' in FunctionManager_add_functions signature, "
        f"got params: {list(sig.parameters.keys())}"
    )

    await add_fn("async def foo(): pass", venv_id=42)
    fm.add_functions.assert_called_once_with(
        implementations="async def foo(): pass",
        language="python",
        overwrite=False,
        venv_id=42,
    )


@pytest.mark.asyncio
async def test_add_venv_wrapper_delegates():
    """FunctionManager_add_venv delegates to fm.add_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    result = await tools["FunctionManager_add_venv"](venv="[project]\nname='x'")
    fm.add_venv.assert_called_once_with(venv="[project]\nname='x'")
    assert result == 42


@pytest.mark.asyncio
async def test_list_venvs_wrapper_delegates():
    """FunctionManager_list_venvs delegates to fm.list_venvs."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_list_venvs"]()
    fm.list_venvs.assert_called_once()


@pytest.mark.asyncio
async def test_set_function_venv_wrapper_delegates():
    """FunctionManager_set_function_venv delegates to fm.set_function_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_set_function_venv"](function_id=10, venv_id=42)
    fm.set_function_venv.assert_called_once_with(function_id=10, venv_id=42)


@pytest.mark.asyncio
async def test_get_function_venv_wrapper_delegates():
    """FunctionManager_get_function_venv delegates to fm.get_function_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_get_function_venv"](function_id=10)
    fm.get_function_venv.assert_called_once_with(function_id=10)


@pytest.mark.asyncio
async def test_update_venv_wrapper_delegates():
    """FunctionManager_update_venv delegates to fm.update_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_update_venv"](venv_id=5, venv="new content")
    fm.update_venv.assert_called_once_with(venv_id=5, venv="new content")


@pytest.mark.asyncio
async def test_delete_venv_wrapper_delegates():
    """FunctionManager_delete_venv delegates to fm.delete_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_delete_venv"](venv_id=5)
    fm.delete_venv.assert_called_once_with(venv_id=5)


@pytest.mark.asyncio
async def test_get_venv_wrapper_delegates():
    """FunctionManager_get_venv delegates to fm.get_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    await tools["FunctionManager_get_venv"](venv_id=5)
    fm.get_venv.assert_called_once_with(venv_id=5)


def test_venv_tools_have_docstrings():
    """All venv tools have non-empty docstrings for LLM consumption."""
    actor, _ = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    for name in EXPECTED_VENV_TOOLS:
        fn = tools[name]
        doc = fn.__doc__
        assert (
            doc and len(doc) > 20
        ), f"Tool '{name}' should have a meaningful docstring, got: {doc!r}"
