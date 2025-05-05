"""
Unit tests for YourClass.add_functions
-------------------------------------

These tests cover:

✓ happy-path single & multiple functions
✓ syntax / indentation errors
✓ blocked ``import`` usage
✓ unknown references
✓ disallowed built-ins and attribute calls
"""

import pytest
from function_manager.function_manager import FunctionManager


# --------------------------------------------------------------------------- #
#  Happy-path tests                                                           #
# --------------------------------------------------------------------------- #
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


def test_add_multiple_functions_with_dependency():
    add_src = "def add(a, b):\n" "    return a + b\n"
    twice_src = "def twice(x):\n" "    return add(x, x)\n"
    fm = FunctionManager()
    result = fm.add_functions(implementations=[add_src, twice_src])
    assert result == {"add": "added", "twice": "added"}


# --------------------------------------------------------------------------- #
#  Validation / failure tests                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source,exp_msg",
    [
        # 1. Syntax error
        (
            "def bad(x)\n    return x",
            "Syntax error",
        ),
        # 2. Leading indentation
        (
            "    def indented(x):\n        return x",
            "must start at column 0",
        ),
        # 3. Import statement
        (
            "def foo(x):\n    import math\n    return x",
            "Imports are not allowed",
        ),
        # 4. Unknown referenced function
        (
            "def uses_unknown(x):\n    return unknown(x)",
            "references unknown function",
        ),
        # 5. Disallowed builtin (vars)
        (
            "def uses_vars(x):\n    return vars(x)",
            "Built-in 'vars' is not permitted",
        ),
        # 6. Attribute call (math.sin)
        (
            "def uses_math(x):\n    import math\n    return math.sin(x)",
            "Imports are not allowed",
        ),
    ],
)
def test_validation_errors(source: str, exp_msg: str):
    fm = FunctionManager()
    with pytest.raises(ValueError, match=exp_msg):
        fm.add_functions(implementations=source)
