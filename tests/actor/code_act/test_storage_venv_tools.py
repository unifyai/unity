"""Tests for venv management tools in the storage loop.

Verifies that:
1. All venv CRUD tools appear in ``_build_storage_tools`` output.
2. ``FunctionManager_add_functions`` accepts ``venv_id``.
3. The tools correctly delegate to the underlying FunctionManager methods.
4. The ``FunctionManager_add_functions`` tool surfaces the third-party
   rejection ``ValueError`` when called with a real FunctionManager and
   code that imports non-stdlib packages without a ``venv_id``.
"""

from unittest.mock import MagicMock

import pytest

from tests.actor.code_act.conftest import make_fm_mock

from unity.actor.code_act_actor import _build_storage_tools, CodeActActor
from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project


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
    fm = make_fm_mock()
    fm._include_primitives = False
    fm.search_functions.return_value = []
    fm.filter_functions.return_value = []
    fm.list_functions.return_value = {}
    fm.add_functions.return_value = {"test_fn": "added"}
    fm.delete_function.return_value = True
    fm.add_venv.return_value = 42
    fm.list_venvs.return_value = []
    fm.get_venv.return_value = None
    fm.update_venv.return_value = True
    fm.delete_venv.return_value = True
    fm.set_function_venv.return_value = True
    fm.get_function_venv.return_value = None

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
    "FunctionManager_delete_function",
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


def test_add_functions_present_in_storage_tools():
    """FunctionManager_add_functions is present in storage tools."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})
    assert "FunctionManager_add_functions" in tools


def test_add_venv_delegates():
    """FunctionManager_add_venv delegates to fm.add_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    result = tools["FunctionManager_add_venv"](venv="[project]\nname='x'")
    fm.add_venv.assert_called_once_with(venv="[project]\nname='x'")
    assert result == 42


def test_list_venvs_delegates():
    """FunctionManager_list_venvs delegates to fm.list_venvs."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_list_venvs"]()
    fm.list_venvs.assert_called_once()


def test_set_function_venv_delegates():
    """FunctionManager_set_function_venv delegates to fm.set_function_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_set_function_venv"](function_id=10, venv_id=42)
    fm.set_function_venv.assert_called_once_with(function_id=10, venv_id=42)


def test_get_function_venv_delegates():
    """FunctionManager_get_function_venv delegates to fm.get_function_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_get_function_venv"](function_id=10)
    fm.get_function_venv.assert_called_once_with(function_id=10)


def test_update_venv_delegates():
    """FunctionManager_update_venv delegates to fm.update_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_update_venv"](venv_id=5, venv="new content")
    fm.update_venv.assert_called_once_with(venv_id=5, venv="new content")


def test_delete_venv_delegates():
    """FunctionManager_delete_venv delegates to fm.delete_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_delete_venv"](venv_id=5)
    fm.delete_venv.assert_called_once_with(venv_id=5)


def test_get_venv_delegates():
    """FunctionManager_get_venv delegates to fm.get_venv."""
    actor, fm = _make_actor_with_mocks()
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    tools["FunctionManager_get_venv"](venv_id=5)
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


# ---------------------------------------------------------------------------
# Symbolic: rejection surfaces through the storage tool wrapper with real FM
# ---------------------------------------------------------------------------

FUNCTION_WITH_REQUESTS = '''
async def fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON from a URL."""
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()
'''.strip()

FUNCTION_WITHOUT_THIRD_PARTY = '''
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''.strip()


@pytest.fixture
def real_function_manager_factory():
    """Create a real FunctionManager for symbolic rejection tests."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
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
    """Build a storage tool dict using a real FunctionManager."""
    gm = _MinimalGuidanceManager()
    actor = MagicMock(spec=CodeActActor)
    actor.function_manager = fm
    actor.guidance_manager = gm
    return actor


@_handle_project
def test_add_functions_tool_rejects_third_party_without_venv(
    real_function_manager_factory,
):
    """Calling the FunctionManager_add_functions storage tool with code that
    imports a third-party package and no venv_id raises ValueError.
    """
    fm = real_function_manager_factory()
    actor = _make_actor_with_real_fm(fm)
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    add_fn = tools["FunctionManager_add_functions"]

    with pytest.raises(ValueError, match="third-party packages"):
        add_fn(implementations=FUNCTION_WITH_REQUESTS)

    stored = fm.list_functions()
    assert not stored, "Function should not have been persisted after rejection"


@_handle_project
def test_add_functions_tool_accepts_third_party_with_venv(
    real_function_manager_factory,
):
    """Calling FunctionManager_add_functions with a venv_id succeeds for
    code that imports third-party packages."""
    fm = real_function_manager_factory()
    actor = _make_actor_with_real_fm(fm)
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    venv_id = tools["FunctionManager_add_venv"](
        venv=(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = ["requests>=2.28.0"]'
        ),
    )

    result = tools["FunctionManager_add_functions"](
        implementations=FUNCTION_WITH_REQUESTS,
        venv_id=venv_id,
    )
    assert result.get("fetch_json") == "added"

    stored = fm.list_functions()
    assert "fetch_json" in stored
    assert stored["fetch_json"]["venv_id"] == venv_id
    assert "requests" in stored["fetch_json"].get("third_party_imports", [])


@_handle_project
def test_add_functions_tool_no_rejection_for_stdlib_only(
    real_function_manager_factory,
):
    """No rejection when the function only uses stdlib imports."""
    fm = real_function_manager_factory()
    actor = _make_actor_with_real_fm(fm)
    tools, _ = _build_storage_tools(actor=actor, ask_tools={})

    result = tools["FunctionManager_add_functions"](
        implementations=FUNCTION_WITHOUT_THIRD_PARTY,
    )
    assert result.get("add") == "added"
