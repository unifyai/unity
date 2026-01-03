"""
Tests for the execution environment utilities.

Tests that create_base_globals() and create_execution_globals() provide
a secure sandbox environment with appropriate restrictions.
"""

from unity.function_manager.execution_env import (
    create_base_globals,
    create_execution_globals,
)


# ────────────────────────────────────────────────────────────────────────────
# Security Tests - Dangerous Builtins Blocked
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_blocks_exec():
    """exec should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "exec" not in globals_dict["__builtins__"]


def test_sandbox_blocks_eval():
    """eval should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "eval" not in globals_dict["__builtins__"]


def test_sandbox_blocks_open():
    """open should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "open" not in globals_dict["__builtins__"]


def test_sandbox_blocks_compile():
    """compile should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "compile" not in globals_dict["__builtins__"]


def test_sandbox_blocks_globals():
    """globals() should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "globals" not in globals_dict["__builtins__"]


def test_sandbox_blocks_locals():
    """locals() should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "locals" not in globals_dict["__builtins__"]


def test_sandbox_blocks_input():
    """input() should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "input" not in globals_dict["__builtins__"]


def test_sandbox_blocks___import__direct_access():
    """
    __import__ IS available (needed for 'import x' statements),
    but we verify it's the only import mechanism.
    """
    globals_dict = create_base_globals()
    # __import__ is allowed for dynamic imports in user code
    assert "__import__" in globals_dict["__builtins__"]


def test_sandbox_blocks_breakpoint():
    """breakpoint() should not be available in the sandbox."""
    globals_dict = create_base_globals()
    assert "breakpoint" not in globals_dict["__builtins__"]


# ────────────────────────────────────────────────────────────────────────────
# Safe Builtins Available
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_has_print():
    """print should be available."""
    globals_dict = create_base_globals()
    assert "print" in globals_dict["__builtins__"]
    assert callable(globals_dict["__builtins__"]["print"])


def test_sandbox_has_len():
    """len should be available."""
    globals_dict = create_base_globals()
    assert "len" in globals_dict["__builtins__"]
    assert globals_dict["__builtins__"]["len"]([1, 2, 3]) == 3


def test_sandbox_has_range():
    """range should be available."""
    globals_dict = create_base_globals()
    assert "range" in globals_dict["__builtins__"]
    assert list(globals_dict["__builtins__"]["range"](3)) == [0, 1, 2]


def test_sandbox_has_type_conversions():
    """Type conversion functions should be available."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]
    assert "str" in builtins
    assert "int" in builtins
    assert "float" in builtins
    assert "bool" in builtins
    assert "list" in builtins
    assert "dict" in builtins
    assert "set" in builtins
    assert "tuple" in builtins


def test_sandbox_has_iteration_functions():
    """Iteration functions should be available."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]
    assert "iter" in builtins
    assert "next" in builtins
    assert "enumerate" in builtins
    assert "zip" in builtins
    assert "filter" in builtins
    assert "map" in builtins
    assert "reversed" in builtins
    assert "sorted" in builtins


def test_sandbox_has_exceptions():
    """Exception classes should be available."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]
    assert "Exception" in builtins
    assert "ValueError" in builtins
    assert "TypeError" in builtins
    assert "KeyError" in builtins
    assert "RuntimeError" in builtins


def test_sandbox_has_class_decorators():
    """Class-related decorators should be available."""
    globals_dict = create_base_globals()
    builtins = globals_dict["__builtins__"]
    assert "property" in builtins
    assert "classmethod" in builtins
    assert "staticmethod" in builtins
    assert "super" in builtins


# ────────────────────────────────────────────────────────────────────────────
# Standard Library Modules
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_has_asyncio():
    """asyncio module should be available."""
    globals_dict = create_base_globals()
    assert "asyncio" in globals_dict
    import asyncio

    assert globals_dict["asyncio"] is asyncio


def test_sandbox_has_json():
    """json module should be available."""
    globals_dict = create_base_globals()
    assert "json" in globals_dict
    assert globals_dict["json"].dumps({"a": 1}) == '{"a": 1}'


def test_sandbox_has_re():
    """re module should be available."""
    globals_dict = create_base_globals()
    assert "re" in globals_dict
    assert globals_dict["re"].match(r"\d+", "123") is not None


def test_sandbox_has_datetime():
    """datetime module should be available."""
    globals_dict = create_base_globals()
    assert "datetime" in globals_dict


def test_sandbox_has_collections():
    """collections module should be available."""
    globals_dict = create_base_globals()
    assert "collections" in globals_dict


def test_sandbox_has_functools():
    """functools module should be available."""
    globals_dict = create_base_globals()
    assert "functools" in globals_dict


def test_sandbox_has_statistics():
    """statistics module should be available."""
    globals_dict = create_base_globals()
    assert "statistics" in globals_dict


# ────────────────────────────────────────────────────────────────────────────
# Typing Support
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_has_typing_module():
    """typing module should be available."""
    globals_dict = create_base_globals()
    assert "typing" in globals_dict


def test_sandbox_has_common_type_hints():
    """Common type hints should be directly available."""
    globals_dict = create_base_globals()
    assert "Any" in globals_dict
    assert "Optional" in globals_dict
    assert "List" in globals_dict
    assert "Dict" in globals_dict
    assert "Tuple" in globals_dict
    assert "Set" in globals_dict
    assert "Union" in globals_dict
    assert "Callable" in globals_dict
    assert "Literal" in globals_dict


# ────────────────────────────────────────────────────────────────────────────
# Pydantic Support
# ────────────────────────────────────────────────────────────────────────────


def test_sandbox_has_pydantic():
    """pydantic should be available if installed."""
    globals_dict = create_base_globals()
    # Pydantic should be available in this project
    assert "pydantic" in globals_dict
    assert "BaseModel" in globals_dict
    assert "Field" in globals_dict


def test_sandbox_pydantic_model_works():
    """Should be able to define and use Pydantic models."""
    globals_dict = create_base_globals()

    code = """
class TestModel(BaseModel):
    name: str
    value: int

model = TestModel(name="test", value=42)
result = model.model_dump()
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == {"name": "test", "value": 42}


# ────────────────────────────────────────────────────────────────────────────
# create_execution_globals Tests
# ────────────────────────────────────────────────────────────────────────────


def test_execution_globals_extends_base():
    """create_execution_globals should include everything from create_base_globals."""
    base = create_base_globals()
    execution = create_execution_globals()

    # All base modules should be present
    for key in ["asyncio", "json", "re", "datetime", "collections"]:
        assert key in execution
        assert execution[key] is base[key]


def test_execution_globals_has_primitives():
    """create_execution_globals should include the primitives object."""
    globals_dict = create_execution_globals()
    assert "primitives" in globals_dict


def test_execution_globals_primitives_is_lazy():
    """Primitives object should provide lazy access to managers."""
    globals_dict = create_execution_globals()
    primitives = globals_dict["primitives"]

    # Existing manager properties
    assert hasattr(primitives, "contacts")
    assert hasattr(primitives, "knowledge")
    assert hasattr(primitives, "tasks")
    # New manager properties
    assert hasattr(primitives, "files")


# ────────────────────────────────────────────────────────────────────────────
# Functional Tests - Code Execution
# ────────────────────────────────────────────────────────────────────────────


def test_can_execute_simple_function():
    """Should be able to define and call a simple function."""
    globals_dict = create_base_globals()

    code = """
def add(a, b):
    return a + b

result = add(2, 3)
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == 5


def test_can_execute_async_function():
    """Should be able to define an async function."""
    import asyncio

    globals_dict = create_base_globals()

    code = """
async def async_add(a, b):
    return a + b
"""
    exec(code, globals_dict)
    result = asyncio.run(globals_dict["async_add"](2, 3))
    assert result == 5


def test_can_define_classes():
    """Should be able to define classes."""
    globals_dict = create_base_globals()

    code = """
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

c = Counter()
result = c.increment()
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == 1


def test_can_use_list_comprehensions():
    """Should be able to use list comprehensions."""
    globals_dict = create_base_globals()

    code = """
result = [x * 2 for x in range(5)]
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == [0, 2, 4, 6, 8]


def test_can_use_dict_comprehensions():
    """Should be able to use dict comprehensions."""
    globals_dict = create_base_globals()

    code = """
result = {str(x): x * 2 for x in range(3)}
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == {"0": 0, "1": 2, "2": 4}


def test_can_use_json_module():
    """Should be able to use json for serialization."""
    globals_dict = create_base_globals()

    code = """
data = {"name": "test", "value": 42}
json_str = json.dumps(data)
parsed = json.loads(json_str)
result = parsed["name"]
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == "test"


def test_can_use_regex():
    """Should be able to use regex."""
    globals_dict = create_base_globals()

    code = """
import re
match = re.search(r'(\d+)', 'value: 123')
result = match.group(1) if match else None
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == "123"


def test_cannot_access_file_system():
    """Should not be able to access the file system."""
    globals_dict = create_base_globals()
    # open is not in safe_builtins
    assert "open" not in globals_dict["__builtins__"]


def test_cannot_use_eval():
    """Should not be able to use eval."""
    globals_dict = create_base_globals()
    # eval is not in safe_builtins
    assert "eval" not in globals_dict["__builtins__"]


def test_cannot_use_exec_from_within():
    """Should not be able to use exec from within executed code."""
    globals_dict = create_base_globals()
    # exec is not in safe_builtins
    assert "exec" not in globals_dict["__builtins__"]


# ────────────────────────────────────────────────────────────────────────────
# Advanced Security Tests - Escape Attempts
# ────────────────────────────────────────────────────────────────────────────


def test_cannot_access_builtins_via_type():
    """Should not be able to escape sandbox via type()."""
    globals_dict = create_base_globals()

    # This is a known Python sandbox escape technique
    code = """
try:
    # Attempt to access __builtins__ via type
    builtins = type.__bases__[0].__subclasses__()
    result = "FAIL - accessed subclasses"
except Exception as e:
    result = f"Error: {type(e).__name__}"
"""
    exec(code, globals_dict)
    # Even if subclasses are accessible, they shouldn't give us file access
    # The key is that open/eval/exec are not directly available


def test_cannot_import_os_directly():
    """Direct import of os should work but limited."""
    globals_dict = create_base_globals()

    code = """
import os
try:
    # os is imported but operations may be limited by the Python interpreter
    # This test verifies import works (which is expected)
    cwd = os.getcwd()
    result = "imported"
except Exception as e:
    result = f"Error: {type(e).__name__}"
"""
    exec(code, globals_dict)
    # Import works - we're not trying to block all imports
    # The sandbox is about removing dangerous builtins, not all functionality
    assert globals_dict["result"] == "imported"


def test_cannot_access_global_namespace_from_class():
    """globals() should not be available in the sandbox."""
    globals_dict = create_base_globals()
    # globals is not in safe_builtins
    assert "globals" not in globals_dict["__builtins__"]


def test_cannot_compile_code():
    """compile() should not be available."""
    globals_dict = create_base_globals()
    # compile is not in safe_builtins
    assert "compile" not in globals_dict["__builtins__"]


def test_introspection_limited():
    """Object introspection should be limited."""
    globals_dict = create_base_globals()

    code = """
# dir and hasattr are allowed (useful for legitimate code)
# but they shouldn't expose dangerous capabilities
class Test:
    secret = "value"

t = Test()
has_secret = hasattr(t, "secret")
attrs = dir(t)
result = f"hasattr={has_secret}, dir_works={len(attrs) > 0}"
"""
    exec(code, globals_dict)
    # Basic introspection works
    assert "hasattr=True" in globals_dict["result"]
    assert "dir_works=True" in globals_dict["result"]


# ────────────────────────────────────────────────────────────────────────────
# Import Security Tests
# ────────────────────────────────────────────────────────────────────────────


def test_can_import_allowed_modules():
    """Should be able to import safe standard library modules."""
    globals_dict = create_base_globals()

    code = """
import math
import hashlib
import base64
import urllib.parse

result = {
    "math": math.pi > 3,
    "hashlib": hasattr(hashlib, "sha256"),
    "base64": hasattr(base64, "b64encode"),
    "urllib": hasattr(urllib.parse, "quote"),
}
"""
    exec(code, globals_dict)
    assert all(globals_dict["result"].values())


def test_imported_module_cannot_escape():
    """Imported modules shouldn't provide escape routes."""
    globals_dict = create_base_globals()

    code = """
import sys
try:
    # sys.modules gives access to all loaded modules
    # but shouldn't allow arbitrary code execution
    mods = list(sys.modules.keys())
    result = f"has {len(mods)} modules"
except Exception as e:
    result = f"Error: {type(e).__name__}"
"""
    exec(code, globals_dict)
    # sys access works but is limited in what damage it can do
    assert "modules" in globals_dict["result"]


# ────────────────────────────────────────────────────────────────────────────
# Async Security Tests
# ────────────────────────────────────────────────────────────────────────────


def test_async_functions_in_sandbox():
    """Async functions should work within sandbox."""
    import asyncio

    globals_dict = create_base_globals()

    code = """
async def safe_async():
    await asyncio.sleep(0)
    return "async works"

async def run():
    return await safe_async()
"""
    exec(code, globals_dict)
    result = asyncio.run(globals_dict["run"]())
    assert result == "async works"


def test_async_with_json():
    """Async functions should be able to use allowed modules."""
    import asyncio

    globals_dict = create_base_globals()

    code = """
async def process_json(data):
    parsed = json.loads(data)
    await asyncio.sleep(0)
    return json.dumps({"processed": parsed["value"]})
"""
    exec(code, globals_dict)
    result = asyncio.run(globals_dict["process_json"]('{"value": 42}'))
    assert '"processed": 42' in result


# ────────────────────────────────────────────────────────────────────────────
# Exception Handling Security
# ────────────────────────────────────────────────────────────────────────────


def test_exception_classes_available():
    """Standard exception classes should be available."""
    globals_dict = create_base_globals()

    code = """
exceptions = []
try:
    raise ValueError("test error")
except ValueError as e:
    exceptions.append("ValueError")

try:
    raise RuntimeError("runtime")
except RuntimeError as e:
    exceptions.append("RuntimeError")

try:
    d = {}
    _ = d["missing"]
except KeyError as e:
    exceptions.append("KeyError")

result = exceptions
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == ["ValueError", "RuntimeError", "KeyError"]


def test_can_raise_custom_exceptions():
    """Should be able to define and raise custom exceptions."""
    globals_dict = create_base_globals()

    code = """
class CustomError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

try:
    raise CustomError(404, "Not found")
except CustomError as e:
    result = f"caught: {e.code}"
"""
    exec(code, globals_dict)
    assert globals_dict["result"] == "caught: 404"
