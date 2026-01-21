"""
Collection and synchronization of custom functions and venvs from source code.

This module scans the `custom/` folder for:
- Python files in `custom/functions/` containing @custom_function decorated functions
- TOML files in `custom/venvs/` containing pyproject.toml content

Both are synced to the database with hash-based change detection.
"""

import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dependency_analysis import collect_dependencies_from_source

from .custom import CustomFunctionMetadata

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Path Helpers
# ────────────────────────────────────────────────────────────────────────────


def _get_custom_base_folder() -> Path:
    """Get the path to the custom base folder."""
    return Path(__file__).parent / "custom"


def _get_custom_functions_folder() -> Path:
    """Get the path to the custom functions folder."""
    return _get_custom_base_folder() / "functions"


def _get_custom_venvs_folder() -> Path:
    """Get the path to the custom venvs folder."""
    return _get_custom_base_folder() / "venvs"


# ────────────────────────────────────────────────────────────────────────────
# Venv Collection
# ────────────────────────────────────────────────────────────────────────────


def _compute_venv_hash(name: str, content: str) -> str:
    """Compute a hash for a custom venv based on its name and content."""
    combined = f"{name}\n{content}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def collect_custom_venvs() -> Dict[str, Dict[str, Any]]:
    """
    Scan the custom/venvs/ folder and collect all custom venv metadata.

    Returns:
        Dict mapping venv name (filename without .toml) to metadata dict with keys:
        - name: str
        - venv: str (the pyproject.toml content)
        - custom_hash: str
    """
    venvs_folder = _get_custom_venvs_folder()
    if not venvs_folder.exists():
        logger.debug("Custom venvs folder does not exist, no custom venvs to collect")
        return {}

    venvs: Dict[str, Dict[str, Any]] = {}

    # Scan all .toml files in the venvs folder
    for toml_file in venvs_folder.glob("*.toml"):
        if toml_file.name.startswith("_"):
            continue  # Skip private files

        name = toml_file.stem  # Filename without .toml
        content = toml_file.read_text().strip()

        custom_hash = _compute_venv_hash(name, content)

        venvs[name] = {
            "name": name,
            "venv": content,
            "custom_hash": custom_hash,
        }

    logger.debug(f"Collected {len(venvs)} custom venvs")
    return venvs


def compute_custom_venvs_hash() -> str:
    """
    Compute an aggregate hash of all custom venvs.

    This is used to quickly check if any custom venvs have changed
    since the last sync, allowing for lazy synchronization.
    """
    venvs = collect_custom_venvs()
    if not venvs:
        return ""

    # Sort by name for deterministic hash
    sorted_hashes = [venvs[name]["custom_hash"] for name in sorted(venvs.keys())]
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────────────────
# Function Collection
# ────────────────────────────────────────────────────────────────────────────


def _compute_function_hash(
    name: str,
    argspec: str,
    docstring: str,
    implementation: str,
    depends_on: List[str],
    venv_name: Optional[str],
    venv_id: Optional[int],
    verify: bool,
    precondition: Optional[Dict[str, Any]],
    windows_os_required: bool = False,
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
        "|".join(sorted(depends_on or [])),
        venv_name or "",
        str(venv_id) if venv_id is not None else "",
        str(verify),
        json.dumps(precondition, sort_keys=True) if precondition else "",
        str(windows_os_required),
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
    Scan the custom/functions/ folder and collect all custom function metadata.

    Returns:
        Dict mapping function name to metadata dict with keys:
        - name: str
        - argspec: str
        - docstring: str
        - implementation: str
        - venv_name: Optional[str] (for resolution during sync)
        - venv_id: Optional[int] (direct ID, if specified)
        - verify: bool
        - precondition: Optional[dict]
        - custom_hash: str
        - embedding_text: str
        - windows_os_required: bool (route to Windows VM when True)
    """
    functions_folder = _get_custom_functions_folder()
    if not functions_folder.exists():
        logger.debug(
            "Custom functions folder does not exist, no custom functions to collect",
        )
        return {}

    staged: List[Tuple[str, Callable, CustomFunctionMetadata, str, str, str]] = []

    # Scan all .py files in the functions folder
    for py_file in functions_folder.glob("*.py"):
        if py_file.name.startswith("_"):
            continue  # Skip __init__.py and private files

        module = _load_module_from_file(py_file)
        if module is None:
            continue

        for name, func, metadata in _extract_functions_from_module(module):
            implementation = _get_function_source(func)
            argspec = _get_function_argspec(func)
            docstring = _get_function_docstring(func)

            # Build embedding text (similar to regular functions)
            embedding_text = (
                f"Function: {name}\n"
                f"Signature: {argspec}\n"
                f"Docstring: {docstring}"
            )

            staged.append((name, func, metadata, argspec, docstring, implementation))

    # Second pass: compute dependency graph now that we know all custom names.
    functions: Dict[str, Dict[str, Any]] = {}
    known_names = {name for name, *_rest in staged}

    for name, _func, metadata, argspec, docstring, implementation in staged:
        deps = sorted(
            list(
                collect_dependencies_from_source(
                    implementation,
                    known_names,
                ),
            ),
        )

        custom_hash = _compute_function_hash(
            name=name,
            argspec=argspec,
            docstring=docstring,
            implementation=implementation,
            depends_on=deps,
            venv_name=metadata.venv_name,
            venv_id=metadata.venv_id,
            verify=metadata.verify,
            precondition=metadata.precondition,
            windows_os_required=metadata.windows_os_required,
        )

        # Rebuild embedding text (deterministic)
        embedding_text = (
            f"Function: {name}\n" f"Signature: {argspec}\n" f"Docstring: {docstring}"
        )

        functions[name] = {
            "name": name,
            "argspec": argspec,
            "docstring": docstring,
            "implementation": implementation,
            "venv_name": metadata.venv_name,
            "venv_id": metadata.venv_id,
            "verify": metadata.verify,
            "precondition": metadata.precondition,
            "custom_hash": custom_hash,
            "embedding_text": embedding_text,
            "depends_on": deps,
            "is_primitive": False,
            "guidance_ids": [],
            "windows_os_required": metadata.windows_os_required,
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
