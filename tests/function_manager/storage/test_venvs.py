"""
Tests for virtual environment storage and foreign key cascading in FunctionManager.

Tests the Functions/VirtualEnvs context, including:
- Creating and listing virtual environments
- Associating functions with virtual environments
- Cascading SET NULL on venv deletion
"""

import pytest

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# Sample pyproject.toml content for testing
SAMPLE_VENV_CONTENT = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]
""".strip()

SAMPLE_VENV_CONTENT_2 = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]
""".strip()

SIMPLE_FUNCTION = """
async def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b
""".strip()

SIMPLE_FUNCTION_2 = """
async def multiply_numbers(a: int, b: int) -> int:
    \"\"\"Multiply two numbers together.\"\"\"
    return a * b
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


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
# Virtual Environment CRUD Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_venv_crud_operations(function_manager_factory):
    """Test add, get, list, update, and delete operations for venvs."""
    fm = function_manager_factory()

    # Add venvs - should return unique int IDs
    venv_id_1 = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)

    assert isinstance(venv_id_1, int) and venv_id_1 >= 0
    assert isinstance(venv_id_2, int) and venv_id_2 >= 0
    assert venv_id_1 != venv_id_2

    # Get venv - should return content by ID
    result = fm.get_venv(venv_id=venv_id_1)
    assert result is not None
    assert result["venv_id"] == venv_id_1
    assert result["venv"] == SAMPLE_VENV_CONTENT

    # Get non-existent - should return None
    assert fm.get_venv(venv_id=99999) is None

    # List venvs - should return all
    venvs = fm.list_venvs()
    assert len(venvs) == 2
    venv_ids = {v["venv_id"] for v in venvs}
    assert venv_id_1 in venv_ids and venv_id_2 in venv_ids

    # Update venv - should change content
    result = fm.update_venv(venv_id=venv_id_1, venv=SAMPLE_VENV_CONTENT_2)
    assert result is True
    updated = fm.get_venv(venv_id=venv_id_1)
    assert updated["venv"] == SAMPLE_VENV_CONTENT_2

    # Update non-existent - should return False
    assert fm.update_venv(venv_id=99999, venv=SAMPLE_VENV_CONTENT) is False

    # Delete venv - should remove entry
    result = fm.delete_venv(venv_id=venv_id_1)
    assert result is True
    assert fm.get_venv(venv_id=venv_id_1) is None

    # Delete non-existent - should return False
    assert fm.delete_venv(venv_id=99999) is False


# ────────────────────────────────────────────────────────────────────────────
# Function-VirtualEnv Association Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_function_venv_association(function_manager_factory):
    """Test associating functions with venvs and removing associations."""
    fm = function_manager_factory()

    # Functions should have venv_id=None by default
    fm.add_functions(implementations=[SIMPLE_FUNCTION])
    functions = fm.list_functions()
    func_id = functions["add_numbers"]["function_id"]
    assert fm.get_function_venv(function_id=func_id) is None

    # Associate function with venv
    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    result = fm.set_function_venv(function_id=func_id, venv_id=venv_id)
    assert result is True

    venv = fm.get_function_venv(function_id=func_id)
    assert venv is not None
    assert venv["venv_id"] == venv_id
    assert venv["venv"] == SAMPLE_VENV_CONTENT

    # Remove association by setting to None
    result = fm.set_function_venv(function_id=func_id, venv_id=None)
    assert result is True
    assert fm.get_function_venv(function_id=func_id) is None

    # Non-existent function should return False
    assert fm.set_function_venv(function_id=99999, venv_id=venv_id) is False


# ────────────────────────────────────────────────────────────────────────────
# Cascade Deletion Tests (SET NULL)
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_venv_deletion_cascades_to_functions(function_manager_factory):
    """Deleting a venv should SET NULL on associated functions."""
    fm = function_manager_factory()

    # Create venv and associate it with multiple functions
    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(implementations=[SIMPLE_FUNCTION, SIMPLE_FUNCTION_2])
    functions = fm.list_functions()

    func_id_1 = functions["add_numbers"]["function_id"]
    func_id_2 = functions["multiply_numbers"]["function_id"]

    fm.set_function_venv(function_id=func_id_1, venv_id=venv_id)
    fm.set_function_venv(function_id=func_id_2, venv_id=venv_id)

    # Verify associations exist
    assert fm.get_function_venv(function_id=func_id_1) is not None
    assert fm.get_function_venv(function_id=func_id_2) is not None

    # Delete the venv
    fm.delete_venv(venv_id=venv_id)

    # Both functions should now have venv_id=None (cascaded SET NULL)
    assert fm.get_function_venv(function_id=func_id_1) is None
    assert fm.get_function_venv(function_id=func_id_2) is None


@_handle_project
@pytest.mark.asyncio
async def test_venv_deletion_isolation(function_manager_factory):
    """Deleting a venv should not affect functions with other venvs; deleting function should not delete venv."""
    fm = function_manager_factory()

    # Create two venvs
    venv_id_1 = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)

    # Create two functions, each with a different venv
    fm.add_functions(implementations=[SIMPLE_FUNCTION, SIMPLE_FUNCTION_2])
    functions = fm.list_functions()

    func_id_1 = functions["add_numbers"]["function_id"]
    func_id_2 = functions["multiply_numbers"]["function_id"]

    fm.set_function_venv(function_id=func_id_1, venv_id=venv_id_1)
    fm.set_function_venv(function_id=func_id_2, venv_id=venv_id_2)

    # Delete first venv - should only affect func_id_1
    fm.delete_venv(venv_id=venv_id_1)

    assert fm.get_function_venv(function_id=func_id_1) is None
    venv = fm.get_function_venv(function_id=func_id_2)
    assert venv is not None
    assert venv["venv_id"] == venv_id_2

    # Delete function - should not delete associated venv
    fm.delete_function(function_id=func_id_2)
    venv = fm.get_venv(venv_id=venv_id_2)
    assert venv is not None
    assert venv["venv"] == SAMPLE_VENV_CONTENT_2


# ────────────────────────────────────────────────────────────────────────────
# Third-Party Import Enforcement Tests
# ────────────────────────────────────────────────────────────────────────────

FUNCTION_WITH_THIRD_PARTY = """
async def fetch_data(url: str):
    \"\"\"Fetch data from a URL using requests.\"\"\"
    import requests
    return requests.get(url).json()
""".strip()

FUNCTION_WITH_STDLIB_ONLY = """
async def parse_data(raw: str) -> dict:
    \"\"\"Parse a JSON string and return info about it.\"\"\"
    import json
    import os
    data = json.loads(raw)
    data["cwd"] = os.getcwd()
    return data
""".strip()

FUNCTION_WITH_MULTIPLE_THIRD_PARTY = """
async def analyze(data: list):
    \"\"\"Analyze data with numpy and pandas.\"\"\"
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(data)
    return np.mean(df.values)
""".strip()


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_rejects_third_party_without_venv(function_manager_factory):
    """add_functions raises ValueError when third-party imports are present and no venv_id."""
    fm = function_manager_factory()

    with pytest.raises(ValueError, match="third-party packages"):
        fm.add_functions(implementations=[FUNCTION_WITH_THIRD_PARTY])


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_accepts_third_party_with_venv_id(function_manager_factory):
    """add_functions succeeds when third-party imports are present and venv_id is provided."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    result = fm.add_functions(
        implementations=[FUNCTION_WITH_THIRD_PARTY],
        venv_id=venv_id,
    )
    assert result["fetch_data"] == "added"

    functions = fm.list_functions()
    assert "fetch_data" in functions
    func_data = functions["fetch_data"]
    assert func_data.get("venv_id") == venv_id
    assert "requests" in func_data.get("third_party_imports", [])


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_stdlib_no_venv_required(function_manager_factory):
    """add_functions succeeds without venv_id when only stdlib imports are present."""
    fm = function_manager_factory()

    result = fm.add_functions(implementations=[FUNCTION_WITH_STDLIB_ONLY])
    assert result["parse_data"] == "added"


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_no_imports_no_venv_required(function_manager_factory):
    """add_functions succeeds without venv_id when no imports at all."""
    fm = function_manager_factory()

    result = fm.add_functions(implementations=[SIMPLE_FUNCTION])
    assert result["add_numbers"] == "added"


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_venv_id_sets_on_record(function_manager_factory):
    """venv_id passed to add_functions is set directly on the function record."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(
        implementations=[FUNCTION_WITH_THIRD_PARTY],
        venv_id=venv_id,
    )

    functions = fm.list_functions()
    func_id = functions["fetch_data"]["function_id"]
    venv = fm.get_function_venv(function_id=func_id)
    assert venv is not None
    assert venv["venv_id"] == venv_id


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_overwrite_with_venv_id(function_manager_factory):
    """overwrite=True with venv_id updates the function and sets the venv."""
    fm = function_manager_factory()

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(
        implementations=[FUNCTION_WITH_THIRD_PARTY],
        venv_id=venv_id,
    )

    venv_id_2 = fm.add_venv(venv=SAMPLE_VENV_CONTENT_2)
    updated_func = """
async def fetch_data(url: str, timeout: int = 30):
    \"\"\"Fetch data with timeout.\"\"\"
    import requests
    return requests.get(url, timeout=timeout).json()
""".strip()
    result = fm.add_functions(
        implementations=[updated_func],
        overwrite=True,
        venv_id=venv_id_2,
    )
    assert result["fetch_data"] == "updated"

    functions = fm.list_functions()
    func_id = functions["fetch_data"]["function_id"]
    venv = fm.get_function_venv(function_id=func_id)
    assert venv is not None
    assert venv["venv_id"] == venv_id_2


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_raise_on_error_false_returns_error(
    function_manager_factory,
):
    """With raise_on_error=False, third-party rejection is returned in the dict."""
    fm = function_manager_factory()

    result = fm.add_functions(
        implementations=[FUNCTION_WITH_THIRD_PARTY],
        raise_on_error=False,
    )
    assert "fetch_data" in result
    assert result["fetch_data"].startswith("error:")
    assert "third-party" in result["fetch_data"]


@_handle_project
@pytest.mark.asyncio
async def test_add_functions_multiple_third_party_listed(function_manager_factory):
    """Multiple third-party imports are all listed in the error and the record."""
    fm = function_manager_factory()

    with pytest.raises(ValueError, match="numpy") as exc_info:
        fm.add_functions(implementations=[FUNCTION_WITH_MULTIPLE_THIRD_PARTY])
    assert "pandas" in str(exc_info.value)

    venv_id = fm.add_venv(venv=SAMPLE_VENV_CONTENT)
    fm.add_functions(
        implementations=[FUNCTION_WITH_MULTIPLE_THIRD_PARTY],
        venv_id=venv_id,
    )
    functions = fm.list_functions()
    tp = functions["analyze"].get("third_party_imports", [])
    assert "numpy" in tp
    assert "pandas" in tp
