"""
Tests for the 'verify' field in FunctionManager.
"""

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager


@_handle_project
def test_add_default_verify():
    """Test that functions have verify=True by default."""
    fm = FunctionManager()
    src = "def default_verify_func():\n    pass\n"
    fm.add_functions(implementations=src)

    listing = fm.list_functions()
    assert "default_verify_func" in listing
    assert listing["default_verify_func"]["verify"] is True


@_handle_project
def test_add_explicit_verify_true():
    """Test explicitly setting verify=True."""
    fm = FunctionManager()
    src = "def explicit_verify_true():\n    pass\n"
    fm.add_functions(implementations=src, verify={"explicit_verify_true": True})

    listing = fm.list_functions()
    assert listing["explicit_verify_true"]["verify"] is True


@_handle_project
def test_add_explicit_verify_false():
    """Test explicitly setting verify=False."""
    fm = FunctionManager()
    src = "def explicit_verify_false():\n    pass\n"
    fm.add_functions(implementations=src, verify={"explicit_verify_false": False})

    listing = fm.list_functions()
    assert listing["explicit_verify_false"]["verify"] is False


@_handle_project
def test_update_verify():
    """Test updating the verify status of a function."""
    fm = FunctionManager()
    src = "def update_verify_func():\n    pass\n"

    # Initial add with default (True)
    fm.add_functions(implementations=src)
    assert fm.list_functions()["update_verify_func"]["verify"] is True

    # Update to False
    fm.add_functions(
        implementations=src,
        verify={"update_verify_func": False},
        overwrite=True,
    )
    assert fm.list_functions()["update_verify_func"]["verify"] is False

    # Update back to True
    fm.add_functions(
        implementations=src,
        verify={"update_verify_func": True},
        overwrite=True,
    )
    assert fm.list_functions()["update_verify_func"]["verify"] is True
