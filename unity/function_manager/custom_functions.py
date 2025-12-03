"""
Collection and synchronization of custom functions from source code.

This module scans the `custom/` folder for Python files containing functions
decorated with @custom_function, and provides utilities for syncing them
to the Functions/Compositional context.
"""

import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .custom import CustomFunctionMetadata

logger = logging.getLogger(__name__)


def _get_custom_folder() -> Path:
    """Get the path to the custom functions folder."""
    return Path(__file__).parent / "custom"


def _compute_function_hash(
    name: str,
    argspec: str,
    docstring: str,
    implementation: str,
    venv_id: Optional[int],
    verify: bool,
    precondition: Optional[Dict[str, Any]],
) -> str:
    """
    Compute a hash for a custom function based on its metadata.

    This hash is used to detect changes in the function definition,
    allowing for efficient sync without unnecessary updates.
    """
    import json

    # Normalize the components
    components = [
        name,
        argspec,
        docstring or "",
        implementation,
        str(venv_id),
        str(verify),
        json.dumps(precondition, sort_keys=True) if precondition else "",
    ]
    combined = "\n".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _get_function_source(func: Callable) -> str:
    """Get the source code of a function."""
    try:
        source = inspect.getsource(func)
        # Dedent if needed
        import textwrap

        return textwrap.dedent(source)
    except (OSError, TypeError):
        return ""


def _get_function_argspec(func: Callable) -> str:
    """Get the argument specification of a function as a string."""
    try:
        sig = inspect.signature(func)
        return str(sig)
    except (ValueError, TypeError):
        return "()"


def _get_function_docstring(func: Callable) -> str:
    """Get the docstring of a function."""
    return inspect.getdoc(func) or ""


def _load_module_from_file(file_path: Path) -> Optional[Any]:
    """Dynamically load a Python module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"custom_functions_{file_path.stem}",
            file_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Failed to load custom function module {file_path}: {e}")
        return None


def _extract_functions_from_module(
    module: Any,
) -> List[Tuple[str, Callable, CustomFunctionMetadata]]:
    """
    Extract all functions decorated with @custom_function from a module.

    Returns a list of (name, function, metadata) tuples.
    """
    functions = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        # Check if it has our decorator metadata
        metadata = getattr(obj, "_custom_function_metadata", None)
        if metadata is None:
            continue
        if not isinstance(metadata, CustomFunctionMetadata):
            continue
        # Skip if auto_sync is disabled
        if not metadata.auto_sync:
            logger.debug(f"Skipping {name}: auto_sync=False")
            continue
        functions.append((name, obj, metadata))
    return functions


def collect_custom_functions() -> Dict[str, Dict[str, Any]]:
    """
    Scan the custom/ folder and collect all custom function metadata.

    Returns:
        Dict mapping function name to metadata dict with keys:
        - name: str
        - argspec: str
        - docstring: str
        - implementation: str
        - venv_id: Optional[int]
        - verify: bool
        - precondition: Optional[dict]
        - custom_hash: str
        - embedding_text: str
    """
    custom_folder = _get_custom_folder()
    if not custom_folder.exists():
        logger.debug("Custom folder does not exist, no custom functions to collect")
        return {}

    functions: Dict[str, Dict[str, Any]] = {}

    # Scan all .py files in the custom folder
    for py_file in custom_folder.glob("*.py"):
        if py_file.name.startswith("_"):
            continue  # Skip __init__.py and private files

        module = _load_module_from_file(py_file)
        if module is None:
            continue

        for name, func, metadata in _extract_functions_from_module(module):
            implementation = _get_function_source(func)
            argspec = _get_function_argspec(func)
            docstring = _get_function_docstring(func)

            custom_hash = _compute_function_hash(
                name=name,
                argspec=argspec,
                docstring=docstring,
                implementation=implementation,
                venv_id=metadata.venv_id,
                verify=metadata.verify,
                precondition=metadata.precondition,
            )

            # Build embedding text (similar to regular functions)
            embedding_text = (
                f"Function: {name}\n"
                f"Signature: {argspec}\n"
                f"Docstring: {docstring}"
            )

            functions[name] = {
                "name": name,
                "argspec": argspec,
                "docstring": docstring,
                "implementation": implementation,
                "venv_id": metadata.venv_id,
                "verify": metadata.verify,
                "precondition": metadata.precondition,
                "custom_hash": custom_hash,
                "embedding_text": embedding_text,
                "calls": [],  # Could be computed via AST analysis if needed
                "is_primitive": False,
                "guidance_ids": [],
            }

    logger.debug(f"Collected {len(functions)} custom functions")
    return functions


def compute_custom_functions_hash() -> str:
    """
    Compute an aggregate hash of all custom functions.

    This is used to quickly check if any custom functions have changed
    since the last sync, allowing for lazy synchronization.
    """
    functions = collect_custom_functions()
    if not functions:
        return ""

    # Sort by name for deterministic hash
    sorted_hashes = [
        functions[name]["custom_hash"] for name in sorted(functions.keys())
    ]
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
