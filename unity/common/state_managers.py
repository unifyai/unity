from __future__ import annotations

from abc import ABC
from typing import Dict, Any, Set

import unify


class BaseStateManager(ABC):
    """
    Central marker base class for all state managers.

    This abstract base exists solely to provide a single common ancestor for
    manager interfaces such as ContactManager, TranscriptManager, KnowledgeManager,
    TaskScheduler, FileManager, FunctionManager, GuidanceManager, ImageManager,
    SecretManager, WebSearcher, and Conductor.

    Purpose
    -------
    - Enable straightforward `isinstance(obj, BaseStateManager)` checks.
    - Allow expressive and maintainable type hints (e.g., unions or generics
      bounded to `BaseStateManager`).

    The class intentionally defines no abstract methods to avoid constraining
    individual manager contracts.
    """


def _events_manager_method_context() -> str:
    """Return the fully-qualified context for ManagerMethod events.

    Mirrors the EventBus convention of deriving contexts from the active
    write (or read) context, falling back to a global context when unset.
    """
    try:
        ctxs: Dict[str, Any] = unify.get_active_context()
        base_ctx = ctxs.get("write") or ctxs.get("read")
    except Exception:
        base_ctx = None
    return f"{base_ctx}/Events/ManagerMethod" if base_ctx else "Events/ManagerMethod"


def _parse_group_values(groups: Any) -> Set[str]:
    """Normalise the return from a groups endpoint into a set of strings.

    The Unify client or raw REST may return a dict (e.g., versions → values),
    a flat list of values, or other simple shapes. This helper flattens them.
    """

    values: set[str] = set()
    if not groups:
        return values

    if isinstance(groups, dict):
        for v in groups.values():
            if isinstance(v, (list, tuple, set)):
                for e in v:
                    if e is not None:
                        values.add(str(e))
            elif v is not None:
                values.add(str(v))
        return values

    if isinstance(groups, (list, tuple, set)):
        for e in groups:
            if isinstance(e, dict):
                val = e.get("value")  # common shape for grouped outputs
                if val is not None:
                    values.add(str(val))
            elif e is not None:
                values.add(str(e))
        return values

    values.add(str(groups))
    return values


def _distinct_managers_via_unify() -> Set[str]:
    """Fetch distinct values of the `manager` field using unify.get_groups()."""
    context = _events_manager_method_context()
    try:
        groups = unify.get_groups(key="manager", context=context)
        return _parse_group_values(groups)
    except Exception:
        return set()


def state_manager_exists(state_manager: str) -> bool:
    """Return True if a given state manager name appears in ManagerMethod logs.

    Checks distinct values of the `manager` field in the per-type context
    `…/Events/ManagerMethod` using `unify.get_groups`.

    Parameters
    ----------
    state_manager : str
        Manager class name to look for, e.g. "ContactManager".
    """
    if not state_manager:
        return False

    managers = _distinct_managers_via_unify()
    return state_manager in managers


def state_manager_method_exists(state_manager: str, method: str) -> bool:
    """Return True if a given manager+method appears in ManagerMethod events.

    This queries distinct values of the ``method`` field under the
    ``…/Events/ManagerMethod`` context, filtered by ``manager == state_manager``
    using ``unify.get_groups``.

    Parameters
    ----------
    state_manager : str
        Manager class name (e.g., "ContactManager").
    method : str
        Method name (e.g., "ask" or "update").
    """
    if not state_manager or not method:
        return False

    context = _events_manager_method_context()
    try:
        groups = unify.get_groups(
            key="method",
            context=context,
            filter=f'manager == "{state_manager}"',
        )
        methods = _parse_group_values(groups)
        return method in methods
    except Exception:
        return False
