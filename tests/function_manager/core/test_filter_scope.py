"""
Tests for the ``filter_scope`` constructor parameter on ``FunctionManager``.

A ``filter_scope`` is a Python boolean expression that is automatically ANDed
onto every read query (``list_functions``, ``filter_functions``,
``search_functions``, ``get_precondition``).  Write paths are unaffected.
"""

from __future__ import annotations

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager


def _FM(**kwargs) -> FunctionManager:
    """Create a FunctionManager with primitives disabled (tests focus on compositional functions)."""
    kwargs.setdefault("include_primitives", False)
    return FunctionManager(**kwargs)


# --------------------------------------------------------------------------- #
#  Shared source snippets                                                      #
# --------------------------------------------------------------------------- #

_PY_ALPHA = 'def alpha(x):\n    """double x"""\n    return x * 2\n'
_PY_BETA = 'def beta(y):\n    """square y"""\n    return y ** 2\n'

_SH_HELLO = (
    "#!/bin/sh\n"
    "# @name: hello_world\n"
    "# @args: ()\n"
    "# @description: Prints hello world\n"
    'echo "Hello, World!"'
)


# --------------------------------------------------------------------------- #
#  list_functions                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_list_functions():
    """A scoped instance's list_functions only returns matching rows."""
    fm_all = _FM()
    fm_all.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm_all.add_functions(implementations=_SH_HELLO, language="sh")

    # Unscoped – should see all three
    assert set(fm_all.list_functions().keys()) == {"alpha", "beta", "hello_world"}

    # Scoped to Python – should exclude the shell function
    fm_py = _FM(filter_scope="language == 'python'")
    listing = fm_py.list_functions()
    assert "alpha" in listing
    assert "beta" in listing
    assert "hello_world" not in listing


# --------------------------------------------------------------------------- #
#  filter_functions                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_filter_functions_no_caller_filter():
    """filter_functions with no explicit filter still applies the scope."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_SH_HELLO, language="sh")

    fm_py = _FM(filter_scope="language == 'python'")
    hits = fm_py.filter_functions()
    names = {h["name"] for h in hits}
    assert "alpha" in names
    assert "beta" in names
    assert "hello_world" not in names


@_handle_project
def test_filter_scope_composes_with_caller_filter():
    """When the caller also supplies a filter, both are ANDed together."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_SH_HELLO, language="sh")

    fm_py = _FM(filter_scope="language == 'python'")
    hits = fm_py.filter_functions(filter="'double' in docstring")
    names = {h["name"] for h in hits}
    # Only alpha has 'double' in its docstring AND is Python
    assert names == {"alpha"}


# --------------------------------------------------------------------------- #
#  search_functions                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_search_functions():
    """Semantic search on a scoped instance never returns out-of-scope rows."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_SH_HELLO, language="sh")

    fm_py = _FM(filter_scope="language == 'python'")
    hits = fm_py.search_functions(query="hello world", n=10)
    for h in hits:
        assert (
            h.get("language", "python") != "sh"
        ), f"search_functions returned out-of-scope row: {h['name']}"


# --------------------------------------------------------------------------- #
#  get_precondition                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_get_precondition():
    """A scoped instance can't see a function outside its scope via get_precondition."""
    fm = _FM()
    fm.add_functions(
        implementations=_SH_HELLO,
        language="sh",
        preconditions={"hello_world": {"needs_auth": True}},
    )

    # Unscoped can retrieve it
    assert fm.get_precondition(function_name="hello_world") is not None

    # Scoped to Python – hello_world is invisible
    fm_py = _FM(filter_scope="language == 'python'")
    assert fm_py.get_precondition(function_name="hello_world") is None


# --------------------------------------------------------------------------- #
#  Default (None) is a no-op                                                   #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_none_is_unscoped():
    """filter_scope=None (default) behaves identically to no scope."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA])
    fm.add_functions(implementations=_SH_HELLO, language="sh")

    fm_none = _FM(filter_scope=None)
    assert set(fm_none.list_functions().keys()) == {"alpha", "hello_world"}


# --------------------------------------------------------------------------- #
#  Writes are unaffected                                                       #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_does_not_affect_writes():
    """A scoped instance can still add functions outside its own scope."""
    fm_py = _FM(filter_scope="language == 'python'")
    # Add a shell function through the scoped instance
    result = fm_py.add_functions(implementations=_SH_HELLO, language="sh")
    assert result == {"hello_world": "added"}

    # The scoped instance can't see it (correct – out of scope)
    assert "hello_world" not in fm_py.list_functions()

    # An unscoped instance can see it (proves the write succeeded)
    fm_all = _FM()
    assert "hello_world" in fm_all.list_functions()


# --------------------------------------------------------------------------- #
#  Two differently-scoped instances see disjoint subsets                        #
# --------------------------------------------------------------------------- #


@_handle_project
def test_two_scoped_instances_see_different_subsets():
    """Non-overlapping scopes produce disjoint views of the same data."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_SH_HELLO, language="sh")

    fm_py = _FM(filter_scope="language == 'python'")
    fm_sh = _FM(filter_scope="language == 'sh'")

    py_names = set(fm_py.list_functions().keys())
    sh_names = set(fm_sh.list_functions().keys())

    assert py_names == {"alpha", "beta"}
    assert sh_names == {"hello_world"}
    assert py_names & sh_names == set()
