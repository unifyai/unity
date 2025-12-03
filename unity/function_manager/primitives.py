"""
Registry and introspection utilities for action primitives.

Action primitives are public methods from state managers (ContactManager.ask,
TaskScheduler.execute, etc.) that are exposed to the Actor for direct invocation
or composition within generated Python code.

Primitives are stored in the Functions context alongside user-defined functions,
distinguished by `is_primitive=True`. They contain argspecs and docstrings but
no implementation (the implementation lives in the Python class).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Registry of (module.ClassName, [method_names]) to expose as primitives.
# Each entry maps a fully-qualified class path to the list of public methods
# that should be available as primitives.
PRIMITIVE_SOURCES: List[Tuple[str, List[str]]] = [
    # State managers - public API methods
    (
        "unity.contact_manager.contact_manager.ContactManager",
        ["ask", "update"],
    ),
    (
        "unity.transcript_manager.transcript_manager.TranscriptManager",
        ["ask"],
    ),
    (
        "unity.knowledge_manager.knowledge_manager.KnowledgeManager",
        ["ask", "update", "refactor"],
    ),
    (
        "unity.task_scheduler.task_scheduler.TaskScheduler",
        ["ask", "update", "execute"],
    ),
    (
        "unity.secret_manager.secret_manager.SecretManager",
        ["ask", "update"],
    ),
    (
        "unity.guidance_manager.guidance_manager.GuidanceManager",
        ["ask", "update"],
    ),
    (
        "unity.web_searcher.web_searcher.WebSearcher",
        ["ask"],
    ),
    (
        "unity.skill_manager.skill_manager.SkillManager",
        ["ask"],
    ),
    # ActionProvider browser/reasoning methods
    (
        "unity.actor.action_provider.ActionProvider",
        ["navigate", "act", "observe", "query", "reason"],
    ),
]


def _import_class(class_path: str) -> Optional[type]:
    """
    Dynamically import a class from its fully-qualified path.

    Args:
        class_path: e.g. "unity.contact_manager.contact_manager.ContactManager"

    Returns:
        The class object, or None if import fails.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not import {class_path}: {e}")
        return None


def _get_method_metadata(
    cls: type,
    method_name: str,
    class_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract metadata (signature, docstring) from a class method.

    Handles functools.wraps by looking for __wrapped__ attribute.

    Args:
        cls: The class containing the method.
        method_name: Name of the method to introspect.
        class_name: Short class name for building qualified name.

    Returns:
        Primitive metadata dict, or None if method not found.
    """
    method = getattr(cls, method_name, None)
    if method is None:
        return None

    # Unwrap functools.wraps to get the original function's metadata
    fn = method
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    # Build qualified name: "ContactManager.ask"
    qualified_name = f"{class_name}.{method_name}"

    try:
        signature = str(inspect.signature(fn))
    except (ValueError, TypeError):
        signature = "(...)"

    docstring = inspect.getdoc(fn) or ""

    return {
        "name": qualified_name,
        "argspec": signature,
        "docstring": docstring,
        "embedding_text": (
            f"Function Name: {qualified_name}\n"
            f"Signature: {signature}\n"
            f"Docstring: {docstring}"
        ),
        "implementation": None,
        "is_primitive": True,
        "calls": [],
        "precondition": None,
        "verify": False,
        "guidance_ids": [],
        # Store class path for execution routing
        "primitive_class": cls.__module__ + "." + cls.__name__,
        "primitive_method": method_name,
    }


def collect_primitives() -> Dict[str, Dict[str, Any]]:
    """
    Introspect all registered primitives and return their metadata.

    Iterates through PRIMITIVE_SOURCES, imports each class, and extracts
    signature and docstring information for each registered method.

    Returns:
        Dict mapping qualified_name (e.g. "ContactManager.ask") to primitive
        metadata suitable for insertion into the Functions context.
    """
    primitives: Dict[str, Dict[str, Any]] = {}

    for class_path, method_names in PRIMITIVE_SOURCES:
        cls = _import_class(class_path)
        if cls is None:
            continue

        class_name = class_path.rsplit(".", 1)[1]

        for method_name in method_names:
            metadata = _get_method_metadata(cls, method_name, class_name)
            if metadata is not None:
                primitives[metadata["name"]] = metadata

    logger.debug(f"Collected {len(primitives)} primitives")
    return primitives


def compute_primitives_hash(primitives: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute a stable hash of all primitive signatures.

    Used to detect when primitives have changed (docstrings updated, methods
    added/removed) and a sync is needed.

    Args:
        primitives: Dict from collect_primitives().

    Returns:
        16-character hex hash string.
    """
    parts = []
    for name in sorted(primitives.keys()):
        p = primitives[name]
        # Include name, signature, and docstring in hash
        parts.append(f"{name}|{p['argspec']}|{p['docstring']}")

    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_primitive_callable(
    primitive_data: Dict[str, Any],
    action_provider: Optional[Any] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    For ActionProvider methods, uses the provided action_provider instance.
    For state manager methods, instantiates the manager (singletons).

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        action_provider: ActionProvider instance (required for ActionProvider primitives).

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    # Special case: ActionProvider methods use the provided instance
    if "ActionProvider" in class_path:
        if action_provider is None:
            logger.warning(
                f"Cannot resolve ActionProvider primitive without action_provider instance",
            )
            return None
        return getattr(action_provider, method_name, None)

    # State managers: instantiate (they're singletons)
    cls = _import_class(class_path)
    if cls is None:
        return None

    try:
        instance = cls()
        return getattr(instance, method_name, None)
    except Exception as e:
        logger.warning(f"Could not instantiate {class_path}: {e}")
        return None
