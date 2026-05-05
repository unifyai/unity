"""
Utilities for creating function execution environments.

Provides controlled global namespaces for executing user-defined and
compositional functions with:
- Safe built-in functions (excluding dangerous ones like eval, exec, open)
- Common standard library modules (asyncio, re, json, datetime, collections)
- Typing module and common type hints
- Pydantic support (if available)
- Access to primitives (state managers, computer use)
- Steerable handle infrastructure (for functions that return steerable handles)
"""

import asyncio
import collections
import datetime
import functools
import json
import re
import typing
from typing import Any, Dict

import unillm

try:
    import pydantic
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    pydantic = None
    BaseModel = None
    Field = None
    HAS_PYDANTIC = False


ENVIRONMENT_MODULES: frozenset[str] = frozenset(
    {
        "primitives",
        "pydantic",
        "unillm",
        "unity",
    },
)
"""Non-stdlib module names that the function execution environment provides.

Used by ``detect_third_party_imports`` to distinguish between packages that
need a venv and packages that the runtime already supplies.
"""


def create_base_globals() -> Dict[str, Any]:
    """
    Creates a dictionary of safe global functions for code execution.

    This provides a controlled environment with:
    - Safe built-in functions (excluding dangerous ones like eval, exec)
    - Common standard library modules (asyncio, re, json, datetime, collections)
    - Typing module and common type hints
    - Pydantic support (if available)

    Returns:
        A dictionary of globals allowed within the execution environment.
    """
    # Safe builtins - excluding dangerous functions
    safe_builtins = {
        k: __builtins__.get(k)
        for k in [
            "repr",
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
            "issubclass",
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
            "id",
            "hash",
            "hex",
            "oct",
            "bin",
            "bytearray",
            "memoryview",
            "complex",
            "slice",
            # Exception hierarchy
            "Exception",
            "BaseException",
            "ArithmeticError",
            "LookupError",
            "NotImplementedError",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "RuntimeError",
            "StopIteration",
            "StopAsyncIteration",
            "AssertionError",
            "NameError",
            "ImportError",
            "ModuleNotFoundError",
            "OSError",
            "IOError",
            "FileNotFoundError",
            "FileExistsError",
            "PermissionError",
            "IsADirectoryError",
            "NotADirectoryError",
            "EOFError",
            "UnicodeError",
            "UnicodeDecodeError",
            "UnicodeEncodeError",
            "ZeroDivisionError",
            "OverflowError",
            "TimeoutError",
            "ConnectionError",
            "BrokenPipeError",
            # File I/O
            "open",
            # Class-related
            "super",
            "property",
            "classmethod",
            "staticmethod",
            "__build_class__",
            "__name__",
            "__import__",
        ]
        if __builtins__.get(k) is not None
    }

    globals_dict = {
        "__builtins__": safe_builtins,
        # Standard library modules
        "asyncio": asyncio,
        "re": re,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        # Additional useful modules
        "statistics": __import__("statistics"),
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
        "functools": functools,
    }

    if HAS_PYDANTIC:
        globals_dict.update(
            {
                "pydantic": pydantic,
                "BaseModel": BaseModel,
                "Field": Field,
            },
        )

    return globals_dict


def create_execution_globals() -> Dict[str, Any]:
    """
    Creates execution globals for running stored functions.

    Extends create_base_globals() with:
    - The `primitives` object for lazy access to all primitive operations
      (state managers, computer use, etc.)
    - Steerable handle infrastructure for creating functions that return
      steerable handles (SteerableToolHandle, start_async_tool_loop,
      new_llm_client, reason)
    - The `unillm` module for advanced direct LLM usage
    - The `reason` helper for focused one-shot semantic reasoning steps

    All primitive imports and instantiations are lazy - only the primitives
    actually used by a function are loaded. This means functions that don't
    need computer use won't import web/desktop infrastructure.

    Steerable Functions
    -------------------
    Functions can return a ``SteerableToolHandle`` subclass to indicate they
    are steerable. The execution layer will detect this at runtime via
    ``isinstance(result, SteerableToolHandle)`` and wire up steering operations.

    Example steerable function::

        async def my_workflow(goal: str) -> SteerableToolHandle:
            client = new_llm_client()
            client.set_system_message("You are a helpful assistant.")
            handle = start_async_tool_loop(
                client=client,
                message=goal,
                tools={},
                loop_id="my-workflow",
            )
            return handle

    Returns:
        A dictionary of globals for function execution.
    """
    globals_dict = create_base_globals()

    # Import Primitives here to avoid circular imports at module load time
    from unity.function_manager.primitives import Primitives

    # Inject the primitives instance - all access is lazy
    globals_dict["primitives"] = Primitives()

    # Steerable handle infrastructure - allows compositional functions to
    # create and return steerable handles that the execution layer can detect
    # and wire up for steering operations (interject, pause, stop, etc.)
    from unity.common.async_tool_loop import (
        SteerableToolHandle,
        start_async_tool_loop,
    )
    from unity.common.llm_client import new_llm_client
    from unity.common.reasoning import reason

    globals_dict["SteerableToolHandle"] = SteerableToolHandle
    globals_dict["start_async_tool_loop"] = start_async_tool_loop
    globals_dict["new_llm_client"] = new_llm_client
    globals_dict["reason"] = reason
    globals_dict["unillm"] = unillm

    return globals_dict


# Backward compatibility aliases
create_sandbox_globals = create_base_globals
