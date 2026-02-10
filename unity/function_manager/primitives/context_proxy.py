"""Transparent proxy that auto-injects _parent_chat_context into primitive calls."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Optional


class _ManagerProxy:
    """Second-level proxy: wraps a single manager and intercepts method calls."""

    __slots__ = ("_manager", "_parent_chat_context")

    def __init__(self, manager: Any, parent_chat_context: Optional[list]) -> None:
        self._manager = manager
        self._parent_chat_context = parent_chat_context

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._manager, name)
        if not callable(attr) or self._parent_chat_context is None:
            return attr

        sig = inspect.signature(attr)
        if "_parent_chat_context" not in sig.parameters:
            return attr

        @functools.wraps(attr)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            if "_parent_chat_context" not in kwargs:
                kwargs["_parent_chat_context"] = self._parent_chat_context
            return attr(*args, **kwargs)

        return _wrapper


class ContextForwardingProxy:
    """Wraps a Primitives-like object and auto-injects ``_parent_chat_context``
    into method calls whose signature accepts it.

    This enables parent chat context propagation through ``execute_code`` without
    any source-code manipulation: the LLM's generated code calls
    ``primitives.contacts.ask(text="...")`` as usual, and the proxy transparently
    adds the context kwarg before the real method executes.
    """

    __slots__ = ("_target", "_parent_chat_context")

    def __init__(
        self,
        target: Any,
        *,
        _parent_chat_context: Optional[list] = None,
    ) -> None:
        self._target = target
        self._parent_chat_context = _parent_chat_context

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if self._parent_chat_context is None:
            return attr
        return _ManagerProxy(attr, self._parent_chat_context)
