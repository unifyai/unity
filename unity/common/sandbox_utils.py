"""
Shared utilities for creating sandboxed execution environments.
"""

import asyncio
import re
import json
import datetime
import collections
import typing
from typing import Dict, Any

try:
    import pydantic
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    pydantic = None
    BaseModel = None
    Field = None
    HAS_PYDANTIC = False


def create_sandbox_globals() -> Dict[str, Any]:
    """
    Creates a dictionary of safe, sandboxed global functions for code execution.

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
