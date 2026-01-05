"""
Utilities for creating function execution environments.

Provides controlled global namespaces for executing user-defined and
compositional functions with:
- Safe built-in functions (excluding dangerous ones like eval, exec, open)
- Common standard library modules (asyncio, re, json, datetime, collections)
- Typing module and common type hints
- Pydantic support (if available)
- Access to primitives (state managers, computer use)
"""

import asyncio
import collections
import datetime
import functools
import json
import re
import typing
from typing import Any, Dict

try:
    import pydantic
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    pydantic = None
    BaseModel = None
    Field = None
    HAS_PYDANTIC = False


def create_base_globals() -> Dict[str, Any]:
    """
    Creates a dictionary of safe global functions for code execution.

    This provides a controlled environment with:
    - Safe built-in functions (excluding dangerous ones like eval, exec, open)
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
            # Exception classes
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
            "OSError",
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

    Extends create_base_globals() with the `primitives` object, which
    provides lazy access to all primitive operations (state managers,
    computer use, etc.).

    All primitive imports and instantiations are lazy - only the primitives
    actually used by a function are loaded. This means functions that don't
    need computer use won't import browser/desktop infrastructure.

    Returns:
        A dictionary of globals for function execution, including `primitives`.
    """
    globals_dict = create_base_globals()

    # Import Primitives here to avoid circular imports at module load time
    from unity.function_manager.primitives import Primitives

    # Inject the primitives instance - all access is lazy
    globals_dict["primitives"] = Primitives()

    return globals_dict


# Backward compatibility aliases
create_sandbox_globals = create_base_globals
