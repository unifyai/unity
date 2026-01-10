"""
Tests for shell function support in FunctionManager.

Coverage
========
✓ add_functions with language="sh"
✓ Shell script metadata parsing (@name, @args, @description)
✓ list_functions returns language field
✓ search_functions filters by language
✓ Shell functions ignore venv_id
"""

from __future__ import annotations

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager


# --------------------------------------------------------------------------- #
#  Sample shell scripts                                                        #
# --------------------------------------------------------------------------- #

SIMPLE_SH_SCRIPT = """#!/bin/sh
# @name: hello_world
# @args: ()
# @description: Prints hello world
echo "Hello, World!"
""".strip()

SH_SCRIPT_WITH_ARGS = """#!/bin/sh
# @name: process_file
# @args: (input_file output_file)
# @description: Copies input to output with processing
cat "$1" > "$2"
""".strip()

SH_SCRIPT_MINIMAL = """#!/bin/sh
# @name: minimal_script
echo "minimal"
""".strip()

SH_SCRIPT_NO_NAME = """#!/bin/sh
# @description: This script has no name
echo "no name"
""".strip()


# --------------------------------------------------------------------------- #
#  Test: Adding shell functions                                                #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_sh_function_success():
    """Test adding a basic /bin/sh function."""
    fm = FunctionManager()
    result = fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")
    assert result == {"hello_world": "added"}


@_handle_project
def test_add_sh_function_with_args():
    """Test adding a shell function with arguments."""
    fm = FunctionManager()
    result = fm.add_functions(implementations=SH_SCRIPT_WITH_ARGS, language="sh")
    assert result == {"process_file": "added"}

    # Verify the argspec was parsed correctly
    listing = fm.list_functions()
    assert "process_file" in listing
    assert listing["process_file"]["argspec"] == "(input_file output_file)"


@_handle_project
def test_add_sh_function_minimal_metadata():
    """Test adding a shell function with minimal metadata (only @name)."""
    fm = FunctionManager()
    result = fm.add_functions(implementations=SH_SCRIPT_MINIMAL, language="sh")
    assert result == {"minimal_script": "added"}

    listing = fm.list_functions()
    assert "minimal_script" in listing
    # Default argspec when not specified
    assert listing["minimal_script"]["argspec"] == "()"
    # Empty docstring when @description not specified
    assert listing["minimal_script"]["docstring"] == ""


@_handle_project
def test_add_sh_function_missing_name_fails():
    """Test that shell scripts without @name comment fail."""
    fm = FunctionManager()
    result = fm.add_functions(implementations=SH_SCRIPT_NO_NAME, language="sh")

    # Should have an error for the unnamed script
    assert len(result) == 1
    key = list(result.keys())[0]
    assert "error" in result[key]
    assert "@name" in result[key]


@_handle_project
def test_add_multiple_sh_functions():
    """Test adding multiple shell functions in one call."""
    fm = FunctionManager()
    result = fm.add_functions(
        implementations=[SIMPLE_SH_SCRIPT, SH_SCRIPT_WITH_ARGS],
        language="sh",
    )
    assert result == {"hello_world": "added", "process_file": "added"}

    listing = fm.list_functions()
    assert set(listing.keys()) == {"hello_world", "process_file"}


# --------------------------------------------------------------------------- #
#  Test: Language field in listings                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_list_functions_returns_language():
    """Test that list_functions returns the language field."""
    fm = FunctionManager()

    # Add a Python function
    fm.add_functions(implementations="def py_func():\n    return 1\n")

    # Add a shell function
    fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")

    listing = fm.list_functions()

    assert "py_func" in listing
    assert listing["py_func"]["language"] == "python"

    assert "hello_world" in listing
    assert listing["hello_world"]["language"] == "sh"


@_handle_project
def test_search_functions_filter_by_language():
    """Test filtering functions by language."""
    fm = FunctionManager()

    # Add mixed functions
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    fm.add_functions(implementations="def beta():\n    return 2\n")
    fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")

    # Search for shell functions only
    shell_funcs = fm.search_functions(filter="language == 'sh'")
    assert len(shell_funcs) == 1
    assert shell_funcs[0]["name"] == "hello_world"

    # Search for Python functions only
    py_funcs = fm.search_functions(filter="language == 'python'")
    assert len(py_funcs) == 2
    assert {f["name"] for f in py_funcs} == {"alpha", "beta"}


# --------------------------------------------------------------------------- #
#  Test: Shell functions and venv_id                                           #
# --------------------------------------------------------------------------- #


@_handle_project
def test_sh_function_venv_id_is_none():
    """Test that shell functions have venv_id=None (they don't use Python venvs)."""
    fm = FunctionManager()
    fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")

    listing = fm.list_functions()
    assert listing["hello_world"]["venv_id"] is None


# --------------------------------------------------------------------------- #
#  Test: Duplicate handling for shell functions                                #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_duplicate_sh_function_skips():
    """Test that adding duplicate shell functions skips by default."""
    fm = FunctionManager()

    result1 = fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")
    assert result1 == {"hello_world": "added"}

    # Try to add again
    result2 = fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")
    assert result2 == {"hello_world": "skipped: already exists"}


@_handle_project
def test_add_duplicate_sh_function_with_overwrite():
    """Test overwriting an existing shell function."""
    fm = FunctionManager()

    result1 = fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")
    assert result1 == {"hello_world": "added"}

    # Modify and overwrite
    modified_script = SIMPLE_SH_SCRIPT.replace("Hello, World!", "Hello, Universe!")
    result2 = fm.add_functions(
        implementations=modified_script,
        language="sh",
        overwrite=True,
    )
    assert result2 == {"hello_world": "updated"}

    # Verify the update
    listing = fm.list_functions(include_implementations=True)
    assert "Universe" in listing["hello_world"]["implementation"]


# --------------------------------------------------------------------------- #
#  Test: Delete shell functions                                                #
# --------------------------------------------------------------------------- #


@_handle_project
def test_delete_sh_function():
    """Test deleting a shell function."""
    fm = FunctionManager()
    fm.add_functions(implementations=SIMPLE_SH_SCRIPT, language="sh")

    listing = fm.list_functions()
    function_id = listing["hello_world"]["function_id"]

    result = fm.delete_function(function_id=function_id)
    assert result == {"hello_world": "deleted"}

    # Verify it's gone
    assert fm.list_functions() == {}
