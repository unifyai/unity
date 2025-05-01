"""
Sandbox module for safely executing generated plan code.

This module provides a restricted execution environment for running
generated plan code with limited access to Python builtins and
only whitelisted primitives from the planner module.
"""

import builtins
import types
import sys
from typing import Dict, Any, Set, Optional, List
from types import ModuleType

# Import primitives to add to safe globals
from . import primitives


class SecurityError(Exception):
    """Exception raised when sandbox detects unsafe code execution attempts."""

    pass


# Safe built-in functions that are allowed in the sandbox
SAFE_BUILTINS: Set[str] = {
    # Basic types
    "bool",
    "int",
    "float",
    "str",
    "list",
    "dict",
    "set",
    "tuple",
    # Basic operations
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    # Container operations
    "sorted",
    "reversed",
    "all",
    "any",
    "sum",
    "min",
    "max",
    # String operations
    "format",
    "repr",
    # Type checking
    "isinstance",
    "type",
    # Constants
    "True",
    "False",
    "None",
}


# Define allowed imports pattern
# Any module starting with 'planner.' is allowed


def _safe_import(
    name: str,
    globals: Dict[str, Any] = None,
    locals: Dict[str, Any] = None,
    fromlist: List[str] = (),
    level: int = 0,
) -> Optional[ModuleType]:
    """
    A restricted import function that only allows importing modules starting with 'planner.'.

    Args:
        name: The name of the module to import
        globals, locals, fromlist, level: Standard __import__ parameters

    Returns:
        The imported module if allowed, otherwise raises SecurityError

    Raises:
        SecurityError: If attempting to import a module not starting with 'planner.'
    """
    if name.startswith("planner."):
        return __import__(name, globals, locals, fromlist, level)

    raise SecurityError(f"Import of '{name}' is not permitted")


# Create a restricted builtins dictionary with only safe functions
safe_builtins: Dict[str, Any] = {
    name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)
}

# Add the safe import function
safe_builtins["__import__"] = _safe_import


# Create the SAFE_GLOBALS dictionary with restricted builtins
SAFE_GLOBALS: Dict[str, Any] = {"__builtins__": safe_builtins}


# Add all primitives to the safe globals
for name, value in primitives.__dict__.items():
    # Skip private attributes and imports
    if not name.startswith("_") and callable(value):
        SAFE_GLOBALS[name] = value


def exec_plan(src: str, filename: str = "<string>") -> ModuleType:
    """
    Execute plan code in a restricted sandbox environment.

    Args:
        src: The source code of the plan to execute
        filename: The filename to assign to the module and for compiled code, so inspect.getsource can find it

    Returns:
        A module containing the executed plan

    Raises:
        SecurityError: If the plan attempts to use forbidden names or imports
    """
    # Create a fresh module for the plan
    plan_module = types.ModuleType("plan")
    # Set the __file__ attribute to support inspect.getsource
    plan_module.__file__ = filename

    # Set up the module with safe globals
    plan_module.__dict__.update(SAFE_GLOBALS)

    try:
        # Execute the source code in the restricted environment
        # Compile the source with the correct filename for code objects
        code_obj = compile(src, filename, "exec")
        exec(code_obj, plan_module.__dict__)
    except (NameError, ImportError) as e:
        # Convert NameError or ImportError to SecurityError for forbidden names/imports
        error_msg = str(e)
        raise SecurityError(f"Security violation: {error_msg}") from e

    return plan_module
