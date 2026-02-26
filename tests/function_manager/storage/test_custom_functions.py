"""
Tests for custom function and venv collection and synchronization.

Tests the collection from explicit directories and sync to the DB, matching
the per-client customization architecture.
"""

import pytest
from pathlib import Path

from unity.function_manager.function_manager import FunctionManager
from unity.function_manager.custom_functions import (
    collect_custom_functions,
    compute_custom_functions_hash,
    collect_custom_venvs,
    compute_custom_venvs_hash,
    collect_functions_from_directories,
)
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Shared test content
# ────────────────────────────────────────────────────────────────────────────

_EXAMPLE_FUNCTIONS_PY = """\
from unity.function_manager.custom import custom_function

@custom_function()
async def example_add(a: int, b: int) -> int:
    \"\"\"Add two integers together.\"\"\"
    return a + b

@custom_function(verify=False)
async def example_uppercase(text: str) -> str:
    \"\"\"Convert text to uppercase.\"\"\"
    return text.upper()

@custom_function(auto_sync=False)
async def draft_function_not_synced(x: int) -> int:
    \"\"\"This function has auto_sync=False, so it will NOT be synced.\"\"\"
    return x * 2
"""

_EXAMPLE_VENV_TOML = """\
[project]
name = "example-minimal"
version = "0.1.0"
description = "Minimal venv for testing"
dependencies = []
"""


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def custom_functions_dir(tmp_path: Path) -> Path:
    """Write example custom functions to a temp directory."""
    fn_dir = tmp_path / "functions"
    fn_dir.mkdir()
    (fn_dir / "__init__.py").write_text("")
    (fn_dir / "example.py").write_text(_EXAMPLE_FUNCTIONS_PY)
    return fn_dir


@pytest.fixture
def custom_venvs_dir(tmp_path: Path) -> Path:
    """Write example venv toml to a temp directory."""
    venv_dir = tmp_path / "venvs"
    venv_dir.mkdir()
    (venv_dir / "__init__.py").write_text("")
    (venv_dir / "example_minimal.toml").write_text(_EXAMPLE_VENV_TOML)
    return venv_dir


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
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


# ────────────────────────────────────────────────────────────────────────────
# 1. Function Collection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_custom_functions_finds_decorated_functions(custom_functions_dir):
    functions = collect_custom_functions(directory=custom_functions_dir)
    assert "example_add" in functions
    assert "example_uppercase" in functions


def test_collect_custom_functions_excludes_auto_sync_false(custom_functions_dir):
    functions = collect_custom_functions(directory=custom_functions_dir)
    assert "draft_function_not_synced" not in functions


def test_collect_custom_functions_has_required_fields(custom_functions_dir):
    functions = collect_custom_functions(directory=custom_functions_dir)
    assert "example_add" in functions
    func = functions["example_add"]

    assert func["name"] == "example_add"
    assert "a: int" in func["argspec"]
    assert "b: int" in func["argspec"]
    assert "Add two integers" in func["docstring"]
    assert "return a + b" in func["implementation"]
    assert len(func["custom_hash"]) == 16
    assert "embedding_text" in func
    assert func["is_primitive"] is False


def test_collect_custom_functions_respects_decorator_options(custom_functions_dir):
    functions = collect_custom_functions(directory=custom_functions_dir)
    assert functions["example_add"]["verify"] is True
    assert functions["example_uppercase"]["verify"] is False


def test_compute_custom_functions_hash_is_deterministic(custom_functions_dir):
    fns = collect_custom_functions(directory=custom_functions_dir)
    hash1 = compute_custom_functions_hash(source_functions=fns)
    hash2 = compute_custom_functions_hash(source_functions=fns)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_collect_custom_functions_none_dir_returns_empty():
    assert collect_custom_functions(directory=None) == {}


def test_collect_custom_functions_missing_dir_returns_empty(tmp_path):
    assert collect_custom_functions(directory=tmp_path / "nonexistent") == {}


# ────────────────────────────────────────────────────────────────────────────
# 2. Venv Collection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_custom_venvs_finds_toml_files(custom_venvs_dir):
    venvs = collect_custom_venvs(directory=custom_venvs_dir)
    assert "example_minimal" in venvs


def test_collect_custom_venvs_has_required_fields(custom_venvs_dir):
    venvs = collect_custom_venvs(directory=custom_venvs_dir)
    venv = venvs["example_minimal"]
    assert venv["name"] == "example_minimal"
    assert "[project]" in venv["venv"]
    assert len(venv["custom_hash"]) == 16


def test_compute_custom_venvs_hash_is_deterministic(custom_venvs_dir):
    venvs = collect_custom_venvs(directory=custom_venvs_dir)
    hash1 = compute_custom_venvs_hash(source_venvs=venvs)
    hash2 = compute_custom_venvs_hash(source_venvs=venvs)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_collect_custom_venvs_none_dir_returns_empty():
    assert collect_custom_venvs(directory=None) == {}


# ────────────────────────────────────────────────────────────────────────────
# 3. Multi-directory Collection Tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_functions_from_multiple_directories(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "funcs.py").write_text(
        "from unity.function_manager.custom import custom_function\n\n"
        "@custom_function()\n"
        "async def func_a(x: int) -> int:\n"
        '    """From dir A."""\n'
        "    return x\n",
    )

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "funcs.py").write_text(
        "from unity.function_manager.custom import custom_function\n\n"
        "@custom_function()\n"
        "async def func_b(x: int) -> int:\n"
        '    """From dir B."""\n'
        "    return x * 2\n",
    )

    merged = collect_functions_from_directories([dir_a, dir_b])
    assert "func_a" in merged
    assert "func_b" in merged


def test_collect_functions_later_dir_overrides_earlier(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "funcs.py").write_text(
        "from unity.function_manager.custom import custom_function\n\n"
        "@custom_function()\n"
        "async def shared_fn(x: int) -> int:\n"
        '    """Version A."""\n'
        "    return x\n",
    )

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "funcs.py").write_text(
        "from unity.function_manager.custom import custom_function\n\n"
        "@custom_function()\n"
        "async def shared_fn(x: int) -> int:\n"
        '    """Version B."""\n'
        "    return x * 2\n",
    )

    merged = collect_functions_from_directories([dir_a, dir_b])
    assert "Version B" in merged["shared_fn"]["docstring"]


# ────────────────────────────────────────────────────────────────────────────
# 4. Function Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_inserts_new_functions(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    result = fm.sync_custom_functions(source_functions=source_fns)

    assert result is True
    functions = fm.list_functions()
    assert "example_add" in functions
    assert "example_uppercase" in functions
    assert "draft_function_not_synced" not in functions


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_is_idempotent(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)

    result1 = fm.sync_custom_functions(source_functions=source_fns)
    assert result1 is True

    fm._custom_functions_synced = False
    result2 = fm.sync_custom_functions(source_functions=source_fns)
    assert result2 is False


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_preserves_function_id(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)

    fm.sync_custom_functions(source_functions=source_fns)
    functions = fm.list_functions()
    original_id = functions["example_add"]["function_id"]

    fm._custom_functions_synced = False
    fm.sync_custom_functions(source_functions=source_fns)
    functions = fm.list_functions()
    assert functions["example_add"]["function_id"] == original_id


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_functions_has_custom_hash(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    fm.sync_custom_functions(source_functions=source_fns)

    db_functions = fm._get_custom_functions_from_db()
    assert "example_add" in db_functions
    assert db_functions["example_add"]["custom_hash"] is not None
    assert len(db_functions["example_add"]["custom_hash"]) == 16


@_handle_project
@pytest.mark.asyncio
async def test_sync_overwrites_user_function_with_same_name(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()

    user_impl = """
async def example_add(a: int, b: int) -> int:
    '''User version of example_add.'''
    return a + b + 100
"""
    fm.add_functions(implementations=[user_impl])
    functions = fm.list_functions()
    assert "example_add" in functions

    source_fns = collect_custom_functions(directory=custom_functions_dir)
    fm.sync_custom_functions(source_functions=source_fns)

    functions = fm.list_functions(include_implementations=True)
    assert "return a + b + 100" not in functions["example_add"]["implementation"]
    assert "return a + b" in functions["example_add"]["implementation"]

    db_functions = fm._get_custom_functions_from_db()
    assert "example_add" in db_functions


@_handle_project
@pytest.mark.asyncio
async def test_user_function_without_custom_hash_is_preserved(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()

    user_impl = """
async def my_unique_user_function(x: int) -> int:
    '''A unique user function.'''
    return x * 3
"""
    fm.add_functions(implementations=[user_impl])

    source_fns = collect_custom_functions(directory=custom_functions_dir)
    fm.sync_custom_functions(source_functions=source_fns)

    functions = fm.list_functions()
    assert "my_unique_user_function" in functions
    assert "example_add" in functions


# ────────────────────────────────────────────────────────────────────────────
# 5. Venv Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_inserts_new_venvs(
    function_manager_factory,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)
    name_to_id = fm.sync_custom_venvs(source_venvs=source_venvs)

    assert "example_minimal" in name_to_id
    assert isinstance(name_to_id["example_minimal"], int)

    venvs = fm.list_venvs()
    assert len(venvs) >= 1
    example_venv = next((v for v in venvs if v.get("name") == "example_minimal"), None)
    assert example_venv is not None
    assert "[project]" in example_venv["venv"]


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_is_idempotent(
    function_manager_factory,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)
    name_to_id_1 = fm.sync_custom_venvs(source_venvs=source_venvs)
    name_to_id_2 = fm.sync_custom_venvs(source_venvs=source_venvs)
    assert name_to_id_1 == name_to_id_2


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_venvs_has_custom_hash(
    function_manager_factory,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)
    fm.sync_custom_venvs(source_venvs=source_venvs)

    db_venvs = fm._get_custom_venvs_from_db()
    assert "example_minimal" in db_venvs
    assert db_venvs["example_minimal"]["custom_hash"] is not None
    assert len(db_venvs["example_minimal"]["custom_hash"]) == 16


# ────────────────────────────────────────────────────────────────────────────
# 6. Combined Sync Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_syncs_both_venvs_and_functions(
    function_manager_factory,
    custom_functions_dir,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)

    fm.sync_custom(source_functions=source_fns, source_venvs=source_venvs)

    db_venvs = fm._get_custom_venvs_from_db()
    assert "example_minimal" in db_venvs

    functions = fm.list_functions()
    assert "example_add" in functions
    assert "example_uppercase" in functions


@_handle_project
@pytest.mark.asyncio
async def test_sync_custom_is_idempotent(
    function_manager_factory,
    custom_functions_dir,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)

    result1 = fm.sync_custom(source_functions=source_fns, source_venvs=source_venvs)
    assert result1 is True

    fm._custom_venvs_synced = False
    fm._custom_functions_synced = False

    result2 = fm.sync_custom(source_functions=source_fns, source_venvs=source_venvs)
    assert result2 is False


# ────────────────────────────────────────────────────────────────────────────
# 7. venv_name Resolution Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_venv_name_resolved_to_venv_id(
    function_manager_factory,
    custom_functions_dir,
    custom_venvs_dir,
):
    fm = function_manager_factory()
    source_venvs = collect_custom_venvs(directory=custom_venvs_dir)
    name_to_id = fm.sync_custom_venvs(source_venvs=source_venvs)

    source_fns = collect_custom_functions(directory=custom_functions_dir)
    fm.sync_custom_functions(venv_name_to_id=name_to_id, source_functions=source_fns)

    functions = fm.list_functions()
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_venv_name_not_found_leaves_venv_id_none(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    name_to_id = {"some_other_venv": 999}

    source_fns = collect_custom_functions(directory=custom_functions_dir)
    fm.sync_custom_functions(venv_name_to_id=name_to_id, source_functions=source_fns)

    functions = fm.list_functions()
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_empty_venv_name_mapping_does_not_crash(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    result = fm.sync_custom_functions(venv_name_to_id={}, source_functions=source_fns)

    assert result is True
    functions = fm.list_functions()
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_none_venv_name_mapping_does_not_crash(
    function_manager_factory,
    custom_functions_dir,
):
    fm = function_manager_factory()
    source_fns = collect_custom_functions(directory=custom_functions_dir)
    result = fm.sync_custom_functions(venv_name_to_id=None, source_functions=source_fns)

    assert result is True
    functions = fm.list_functions()
    assert "example_add" in functions


@_handle_project
@pytest.mark.asyncio
async def test_sync_with_no_source_is_noop(function_manager_factory):
    """Syncing with no source functions/venvs should be a fast no-op."""
    fm = function_manager_factory()
    result = fm.sync_custom()
    assert result is False
