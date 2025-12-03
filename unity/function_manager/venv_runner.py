#!/usr/bin/env python3
"""
Standalone runner script for executing functions in isolated virtual environments.

This script is designed to be copied into custom venvs and executed as a subprocess.
It has NO dependencies on the unity package - it's completely standalone.

Communication Protocol:
- Receives JSON on stdin: {"implementation": str, "call_kwargs": dict, "is_async": bool}
- Sends JSON on stdout: {"result": Any, "error": str|null, "stdout": str, "stderr": str}

The script:
1. Reads input from stdin
2. Executes the function implementation with provided kwargs
3. Captures stdout/stderr
4. Returns the result or error as JSON
"""

import asyncio
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout


def create_safe_globals():
    """Create a sandboxed globals dict for function execution."""
    import collections
    import datetime
    import functools
    import re
    import statistics
    import typing

    # Safe builtins - excluding dangerous functions
    safe_builtins = {}
    builtins_dict = (
        __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    )
    for k in [
        "print",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "range",
        "type",
        "object",
        "bytes",
        "frozenset",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "callable",
        "dir",
        "vars",
        "iter",
        "next",
        "filter",
        "map",
        "reversed",
        "enumerate",
        "zip",
        "any",
        "all",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "pow",
        "divmod",
        "sorted",
        "format",
        "chr",
        "ord",
        "Exception",
        "NotImplementedError",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "StopIteration",
        "AssertionError",
        "super",
        "property",
        "classmethod",
        "staticmethod",
        "__build_class__",
        "__name__",
        "__import__",
    ]:
        if k in builtins_dict:
            safe_builtins[k] = builtins_dict[k]

    globals_dict = {
        "__builtins__": safe_builtins,
        # Standard library modules
        "asyncio": asyncio,
        "re": re,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        "statistics": statistics,
        "functools": functools,
        # Typing module and common types
        "typing": typing,
        "Any": typing.Any,
        "Callable": typing.Callable,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Set": typing.Set,
        "Union": typing.Union,
        "Literal": typing.Literal,
    }

    # Try to add pydantic if available in this venv
    try:
        import pydantic
        from pydantic import BaseModel, Field

        globals_dict.update(
            {
                "pydantic": pydantic,
                "BaseModel": BaseModel,
                "Field": Field,
            },
        )
    except ImportError:
        pass

    return globals_dict


def execute_sync(implementation: str, call_kwargs: dict) -> dict:
    """Execute a synchronous function."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    try:
        globals_dict = create_safe_globals()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(implementation, globals_dict)

            # Find the function that was defined
            func_name = None
            for name, obj in globals_dict.items():
                if (
                    callable(obj)
                    and not name.startswith("_")
                    and name not in create_safe_globals()
                ):
                    func_name = name
                    break

            if func_name is None:
                raise ValueError("No function found in implementation")

            fn = globals_dict[func_name]
            result = fn(**call_kwargs)

    except Exception:
        error = traceback.format_exc()

    return {
        "result": result,
        "error": error,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
    }


async def execute_async(implementation: str, call_kwargs: dict) -> dict:
    """Execute an asynchronous function."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    try:
        globals_dict = create_safe_globals()

        exec(implementation, globals_dict)

        # Find the async function that was defined
        func_name = None
        base_globals = set(create_safe_globals().keys())
        for name, obj in globals_dict.items():
            if (
                asyncio.iscoroutinefunction(obj)
                and not name.startswith("_")
                and name not in base_globals
            ):
                func_name = name
                break

        if func_name is None:
            raise ValueError("No async function found in implementation")

        fn = globals_dict[func_name]

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = await fn(**call_kwargs)

    except Exception:
        error = traceback.format_exc()

    return {
        "result": result,
        "error": error,
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
    }


def make_json_serializable(obj):
    """Convert an object to a JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    # For other types, convert to string
    return str(obj)


def main():
    """Main entry point for the runner."""
    # Read input from stdin
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(
            json.dumps(
                {
                    "result": None,
                    "error": f"Invalid JSON input: {e}",
                    "stdout": "",
                    "stderr": "",
                },
            ),
        )
        sys.exit(1)

    implementation = input_data.get("implementation", "")
    call_kwargs = input_data.get("call_kwargs", {})
    is_async = input_data.get("is_async", False)

    # Execute the function
    if is_async:
        result = asyncio.run(execute_async(implementation, call_kwargs))
    else:
        result = execute_sync(implementation, call_kwargs)

    # Make result JSON-serializable
    result["result"] = make_json_serializable(result["result"])

    # Output result as JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()
