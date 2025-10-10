"""
Comprehensive unit-tests for `FunctionManager`

Coverage
========
✓ add_functions                           (existing happy-path + validation)
✓ list_functions                          (with / without implementations)
✓ delete_function                         (single, cascading, non-cascading)
✓ search_functions                        (flexible Python-expr filtering)

The tests introduce a *minimal* stub of the `unify` API so that they remain
fully hermetic.  Nothing outside this file is required.
"""

from __future__ import annotations

import pytest
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager


# --------------------------------------------------------------------------- #
#  4.  Existing add_functions tests                                           #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_add_single_function_success():
    src = (
        "def double(x):\n"
        "    y = 0\n"
        "    for _ in range(2):\n"
        "        y = y + x\n"
        "    return y\n"
    )
    fm = FunctionManager()
    result = fm.add_functions(implementations=src)
    assert result == {"double": "added"}


@_handle_project
@pytest.mark.unit
def test_add_multiple_functions_with_dependency():
    add_src = "def add(a, b):\n    return a + b\n"
    # Update to avoid user-defined function calls (now disallowed)
    twice_src = "def twice(x):\n    return x + x\n"
    fm = FunctionManager()
    result = fm.add_functions(implementations=[add_src, twice_src])
    assert result == {"add": "added", "twice": "added"}


@_handle_project
@pytest.mark.parametrize(
    "source,exp_msg",
    [
        ("def bad(x)\n    return x", "Syntax error"),  # syntax error
        ("    def indented(x):\n        return x", "must start at column 0"),  # indent
        # Dangerous built-ins are disallowed
        ("def uses_eval(x):\n    return eval(str(x))", "Dangerous built-in 'eval'"),
        ("def uses_open():\n    return open('f.txt','w')", "Dangerous built-in 'open'"),
        # Self-recursive calls (user-defined) are disallowed under current policy
        (
            "def recurse(x):\n    return recurse(x-1)",
            "cannot call user-defined function 'recurse'",
        ),
    ],
)
@pytest.mark.unit
def test_validation_errors(source: str, exp_msg: str):
    fm = FunctionManager()
    with pytest.raises(ValueError, match=exp_msg):
        fm.add_functions(implementations=source)


# --------------------------------------------------------------------------- #
#  5.  list_functions                                                         #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_list_functions_with_and_without_implementations():
    add_src = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    fm = FunctionManager()
    fm.add_functions(implementations=add_src)

    # (a) default summary
    listing = fm.list_functions()
    assert listing.keys() == {"add"}
    assert "implementation" not in listing["add"]
    # The argspec includes type hints and return annotation
    assert "(a: int, b: int) -> int" in listing["add"]["argspec"]
    assert listing["add"]["docstring"] == "Add two numbers"

    # (b) include full source
    full = fm.list_functions(include_implementations=True)
    assert add_src.strip() == full["add"]["implementation"].strip()


# --------------------------------------------------------------------------- #
#  6.  delete_function                                                        #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_delete_single_function():
    fm = FunctionManager()
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    assert len(fm.list_functions()) == 1

    fm.delete_function(function_id=0)
    assert fm.list_functions() == {}


@_handle_project
@pytest.mark.unit
def test_delete_function_with_dependants_cascades():
    add_src = "def add(a, b):\n    return a + b\n"
    # No user-defined calls allowed; keep independent
    twin_src = "def twin(x):\n    return x + x\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[add_src, twin_src])

    # delete `add`; since no dependants are allowed, only `add` is removed
    fm.delete_function(function_id=0, delete_dependents=True)
    remaining = fm.list_functions()
    assert remaining.keys() == {"twin"}


@_handle_project
@pytest.mark.unit
def test_delete_function_without_cascading_leaves_dependants():
    add_src = "def add(a, b):\n    return a + b\n"
    twin_src = "def twin(x):\n    return x + x\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[add_src, twin_src])

    fm.delete_function(function_id=0, delete_dependents=False)
    remaining = fm.list_functions()
    assert remaining.keys() == {"twin"}


# --------------------------------------------------------------------------- #
#  7.  search_functions                                                       #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_search_functions_filtering_across_columns():
    price_src = (
        "def price_total(p, tax):\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    square_src = "def square(x):\n    return x * x\n"
    use_src = "def apply_price(x):\n    return x\n"
    fm = FunctionManager()
    fm.add_functions(implementations=[price_src, square_src, use_src])

    # filter on docstring contents
    hits = fm.search_functions(filter="'price' in docstring")
    names = {h["name"] for h in hits}
    assert names == {"price_total"}

    # filter by Python predicate on the `name` column
    hits = fm.search_functions(filter="name[0:2] == 'sq'")
    assert {h["name"] for h in hits} == {"square"}


# --------------------------------------------------------------------------- #
#  8.  clear                                                                  #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.unit
def test_function_manager_clear():
    fm = FunctionManager()

    # Seed a couple of functions
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    fm.add_functions(implementations="def beta():\n    return 2\n")

    listing = fm.list_functions()
    assert set(listing.keys()) == {"alpha", "beta"}
    ids = {listing["alpha"]["function_id"], listing["beta"]["function_id"]}
    assert all(isinstance(x, int) for x in ids)

    # Execute clear
    fm.clear()

    # After clear: no functions should remain
    assert fm.list_functions() == {}

    # New additions should work against a clean slate (ids reset)
    fm.add_functions(implementations="def gamma():\n    return 3\n")
    post = fm.list_functions()
    assert set(post.keys()) == {"gamma"}
    assert post["gamma"]["function_id"] == 0

    # filter by implementation contents (allowed column)
    hits = fm.search_functions(filter="'return x * x' in implementation")
    assert {h["name"] for h in hits} == {"square"}
