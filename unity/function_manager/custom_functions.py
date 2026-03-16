"""
Collection and synchronization of custom functions and venvs from source code.

Functions and venvs are defined in per-client directories under
``unity/customization/clients/``.  The collection helpers accept explicit
directory paths so that the sync can target different source trees for
different clients (org -> user -> assistant cascade).
"""

import hashlib
import importlib.util
import inspect
import logging
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dependency_analysis import collect_dependencies_from_source

from .custom import CustomFunctionMetadata

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Venv Collection
# ────────────────────────────────────────────────────────────────────────────


def _compute_venv_hash(name: str, content: str) -> str:
    """Compute a hash for a custom venv based on its name and content."""
    combined = f"{name}\n{content}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def collect_custom_venvs(
    directory: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan a directory for ``.toml`` venv definitions.

    Args:
        directory: Folder containing ``*.toml`` files.  If *None* or
            non-existent, returns an empty dict.

    Returns:
        Dict mapping venv name (filename without .toml) to metadata dict with keys:
        - name: str
        - venv: str (the pyproject.toml content)
        - custom_hash: str
    """
    venvs_folder = directory
    if venvs_folder is None or not venvs_folder.exists():
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


def compute_custom_venvs_hash(
    source_venvs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Compute an aggregate hash of all custom venvs.

    Args:
        source_venvs: Pre-collected venvs dict.  If *None*, returns ``""``.
    """
    venvs = source_venvs if source_venvs is not None else {}
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
    """Dynamically load a Python module from a file path.

    Registers a synthetic parent package for the file's directory so that
    relative imports between sibling modules work
    (e.g. ``from .helpers import ...`` in a neighbouring file).
    """
    try:
        parent_dir = str(file_path.parent)
        package_name = f"custom_functions_{abs(hash(parent_dir)) % (10**8):08d}"

        if package_name not in sys.modules:
            pkg = types.ModuleType(package_name)
            pkg.__path__ = [parent_dir]
            pkg.__package__ = package_name
            sys.modules[package_name] = pkg

        module_name = f"custom_functions_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            file_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[module_name] = module
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


def collect_custom_functions(
    directory: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan a directory for ``@custom_function`` decorated Python functions.

    Args:
        directory: Folder containing ``*.py`` files.  If *None* or
            non-existent, returns an empty dict.

    Returns:
        Dict mapping function name to metadata dict with keys:
        - name, argspec, docstring, implementation
        - venv_name, venv_id, verify, precondition
        - custom_hash, embedding_text, depends_on
        - is_primitive (always False), guidance_ids (always [])
        - windows_os_required
    """
    functions_folder = directory
    if functions_folder is None or not functions_folder.exists():
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
    env_namespaces = frozenset({"primitives"})

    for name, _func, metadata, argspec, docstring, implementation in staged:
        deps = sorted(
            list(
                collect_dependencies_from_source(
                    implementation,
                    known_names,
                    environment_namespaces=env_namespaces,
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


def compute_custom_functions_hash(
    source_functions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Compute an aggregate hash of custom functions.

    Args:
        source_functions: Pre-collected functions dict.  If *None*, returns ``""``.
    """
    functions = source_functions if source_functions is not None else {}
    if not functions:
        return ""

    # Sort by name for deterministic hash
    sorted_hashes = [
        functions[name]["custom_hash"] for name in sorted(functions.keys())
    ]
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────────────────
# Multi-directory helpers (for the org -> user -> assistant cascade)
# ────────────────────────────────────────────────────────────────────────────


def collect_functions_from_directories(
    directories: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """Collect custom functions from multiple directories and merge.

    Later directories override earlier ones when function names collide
    (more-specific level wins).
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for d in directories:
        merged.update(collect_custom_functions(directory=d))
    return merged


def collect_venvs_from_directories(
    directories: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """Collect custom venvs from multiple directories and merge.

    Later directories override earlier ones when venv names collide.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for d in directories:
        merged.update(collect_custom_venvs(directory=d))
    return merged
