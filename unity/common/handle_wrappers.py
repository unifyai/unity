"""
Minimal standard for exposing wrapped handles.

Contract:
  - Wrappers MUST implement get_wrapped_handles() -> Iterable
    • Accepts any of: list[handle], list[(name, handle)], dict[str, handle], or a single handle

Helper:
  - HandleWrapperMixin provides `wrap_handle()` and a default `get_wrapped_handles()`
    backed by a private list.

No heuristics, no legacy attribute scanning.
"""

from __future__ import annotations

from typing import Any, Tuple, List


def _iter_from_container(container: Any, src: str):
    if container is None:
        return []
    # Dict[str, handle]
    if isinstance(container, dict):
        return [(f"{src}:{k}", v) for k, v in list(container.items())]
    # List/Tuple of handles or (name, handle)
    if isinstance(container, (list, tuple)):
        out: list[tuple[str, Any]] = []
        for item in list(container):
            if isinstance(item, tuple) and len(item) == 2:
                out.append((src, item[1]))
            else:
                out.append((src, item))
        return out
    # Single handle
    return [(src, container)]


def discover_wrapped_handles(obj: Any) -> List[Tuple[str, Any]]:
    """Return a list of (source, handle) pairs from the standard entrypoint.

    Only the standard method _get_wrapped_handles() is supported.
    """
    try:
        meth = getattr(obj, "_get_wrapped_handles", None)
        if not callable(meth):
            return []
        res = meth()
        return list(_iter_from_container(res, "_get_wrapped_handles"))
    except Exception:
        return []


class HandleWrapperMixin:
    """Minimal mixin to register wrapped handles.

    Usage:
        class MyWrapper(HandleWrapperMixin):
            def __init__(self, handle):
                self._wrap_handle(handle)
    """

    def _get_wrapped_handles(self) -> list:
        lst = getattr(self, "_wrapped_handles", None)
        return (
            list(lst)
            if isinstance(lst, (list, tuple))
            else ([] if lst is None else [lst])
        )

    # Convenience helpers for wrappers to register inner handles explicitly
    def _wrap_handle(self, handle: Any) -> None:
        lst = getattr(self, "_wrapped_handles", None)
        if isinstance(lst, list):
            lst.append(handle)
        elif lst is None:
            setattr(self, "_wrapped_handles", [handle])
        else:
            setattr(self, "_wrapped_handles", [lst, handle])
