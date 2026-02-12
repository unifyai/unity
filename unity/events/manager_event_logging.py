from __future__ import annotations

import asyncio
from typing import Any
import functools
from uuid import uuid4
import json
from secrets import token_hex

from ..events.event_bus import EVENT_BUS, Event
from ..events.types.manager_method import ManagerMethodPayload
from ..common.async_tool_loop import SteerableToolHandle
from ..common.hierarchical_logger import build_hierarchy_label

from ..common._async_tool.loop_config import TOOL_LOOP_LINEAGE

__all__ = [
    "new_call_id",
    "publish_manager_method_event",
    "wrap_handle_with_logging",
    "log_manager_call",
]

# ---------------------------------------------------------------------------
#  1.  Small convenience helpers
# ---------------------------------------------------------------------------


def new_call_id() -> str:
    """Return a fresh UUID4 as a `str` – keeps call-ID creation consistent."""
    return str(uuid4())


async def publish_manager_method_event(  # noqa: D401 – imperative name
    call_id: str,
    manager_name: str,
    method_name: str,
    *,
    source: str | None = None,
    **extra: Any,
) -> None:
    """
    Thin wrapper around :pyfunc:`EVENT_BUS.publish` for *ManagerMethod* events.

    Uses the typed ManagerMethodPayload model for schema consistency.

    Hierarchy behavior:
    - Reads the current async tool loop lineage from ``TOOL_LOOP_LINEAGE``.
    - If the caller provides ``hierarchy`` / ``hierarchy_label`` (e.g., boundary wrappers),
      those are used as-is.
    - Otherwise, appends a ``{manager}.{method}`` leaf to the current lineage and derives
      a label using the same ``->`` separator as async tool loops.
    """
    # Work on a copy so we can safely pop internal-only keys without mutating caller dict.
    extra_clean: dict[str, Any] = dict(extra)

    # Best-effort lineage read (ContextVar).
    parent_lineage: list[str] = []
    try:
        val = TOOL_LOOP_LINEAGE.get([])
        if isinstance(val, list):
            parent_lineage = list(val)
    except Exception:
        parent_lineage = []

    # Determine effective hierarchy:
    # - If caller provided explicit hierarchy (e.g., function wrappers), trust it.
    # - Otherwise append this manager.method leaf to current lineage.
    hierarchy: list[str]
    if "hierarchy" in extra_clean and isinstance(extra_clean.get("hierarchy"), list):
        try:
            hierarchy = list(extra_clean.get("hierarchy") or [])
        except Exception:
            hierarchy = []
    else:
        leaf = f"{manager_name}.{method_name}" if manager_name and method_name else ""
        hierarchy = [*parent_lineage, leaf] if leaf else list(parent_lineage)

    # Determine effective hierarchy_label:
    # 1) Use caller-provided hierarchy_label if present (wrappers, etc.)
    # 2) Fall back to a passed-through async tool loop label (e.g., from handle._log_label)
    # 3) Otherwise build from hierarchy + suffix (caller-provided or generated)
    hierarchy_label: str = ""
    try:
        if isinstance(extra_clean.get("hierarchy_label"), str) and extra_clean.get(
            "hierarchy_label",
        ):
            hierarchy_label = str(extra_clean.get("hierarchy_label"))
        elif isinstance(extra_clean.get("_log_label"), str) and extra_clean.get(
            "_log_label",
        ):
            hierarchy_label = str(extra_clean.get("_log_label"))
        else:
            suffix = extra_clean.get("suffix")
            if not isinstance(suffix, str) or not suffix:
                suffix = token_hex(2)
            hierarchy_label = build_hierarchy_label(hierarchy, suffix)
    except Exception:
        hierarchy_label = ""

    # Truncate traceback to avoid large payloads (best-effort).
    try:
        tb = extra_clean.get("traceback")
        if isinstance(tb, str) and len(tb) > 2000:
            extra_clean["traceback"] = tb[:2000]
    except Exception:
        pass

    # Internal-only keys should not be stored in payload.
    for k in ("hierarchy", "hierarchy_label", "suffix", "_log_label"):
        extra_clean.pop(k, None)

    payload = ManagerMethodPayload(
        manager=manager_name,
        method=method_name,
        source=source,
        hierarchy=hierarchy,
        hierarchy_label=hierarchy_label,
        **extra_clean,
    )
    await EVENT_BUS.publish(
        Event(
            type="ManagerMethod",
            calling_id=call_id,
            payload=payload,
        ),
    )


# Ensure values logged to EventBus use stable, schema-friendly text types.
def _coerce_text_value(value: Any) -> str:
    """Return a *string* suitable for logging.

    Behaviour:
    - If *value* is a list/tuple (e.g. [answer, steps]), return the first item.
    - If bytes, decode as UTF-8 (ignore errors).
    - If already str, return as-is.
    - Otherwise try JSON serialisation; fallback to str().
    """
    # Reasoning-steps pattern: [answer_str, messages]
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) > 0 else ""

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "ignore")
        except Exception:
            return str(value)

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
#  2.  Generic SteerableToolHandle wrapper
# ---------------------------------------------------------------------------


def wrap_handle_with_logging(
    inner: SteerableToolHandle,
    call_id: str,
    manager_name: str,
    method_name: str,
    *,
    display_label: str | None = None,
) -> SteerableToolHandle:
    """
    Return a proxy that logs every callable on the inner handle generically (pre-call action,
    post-call outgoing with sanitized return value) while preserving signatures/docs.
    """

    import functools as _functools
    import inspect as _inspect

    class _LoggedHandle:  # duck-typed proxy
        __slots__ = ("_inner",)

        def __init__(self, _h):
            self._inner = _h

        async def _publish(self, **payload):
            if display_label is not None:
                payload.setdefault("display_label", display_label)
            # Best-effort: attach tool loop label from the inner handle when available
            # so ManagerMethod events can reproduce the same label used in terminal logs.
            try:
                if "hierarchy_label" not in payload or not payload.get(
                    "hierarchy_label",
                ):
                    lbl = getattr(self._inner, "_log_label", None)
                    if isinstance(lbl, str) and lbl:
                        payload["hierarchy_label"] = lbl
            except Exception:
                pass
            # Best-effort: attach lineage list (when available) so outgoing events
            # preserve the full parent→child stack even though handle methods are
            # typically called outside the tool loop's ContextVar scope.
            try:
                if "hierarchy" not in payload or not payload.get("hierarchy"):
                    h = getattr(self._inner, "_log_hierarchy", None)
                    if isinstance(h, list) and h:
                        payload["hierarchy"] = list(h)
            except Exception:
                pass
            await publish_manager_method_event(
                call_id,
                manager_name,
                method_name,
                **payload,
            )

        def __dir__(self):
            try:
                return sorted(set(dir(self._inner)) | set(super().__dir__()))
            except Exception:
                return super().__dir__()

        @property
        def __wrapped__(self):  # type: ignore[override]
            return self._inner

        def __getattribute__(self, name: str):
            # Spoof class to look like the inner handle for reflection
            if name == "__class__":
                try:
                    inner = object.__getattribute__(self, "_inner")
                    return inner.__class__
                except Exception:
                    return object.__getattribute__(self, "__class__")

            # Always try inner first (including private/dunder attributes);
            # fall back to proxy attributes only if inner lookup fails.
            target = None
            try:
                inner = object.__getattribute__(self, "_inner")
                target = getattr(inner, name)
            except Exception:
                pass
            if target is None:
                return object.__getattribute__(self, name)

            if not callable(target):
                return target

            is_async = _inspect.iscoroutinefunction(target)
            if is_async:

                @_functools.wraps(target)
                async def _wrapped(*args, **kwargs):
                    try:
                        await self._publish(action=name)
                    except Exception:
                        pass
                    result = await target(*args, **kwargs)
                    try:
                        await self._publish(
                            phase="outgoing",
                            answer=_coerce_text_value(result),
                        )
                    except Exception:
                        pass
                    return result

            else:

                @_functools.wraps(target)
                def _wrapped(*args, **kwargs):
                    try:
                        asyncio.create_task(self._publish(action=name))
                    except Exception:
                        pass
                    result = target(*args, **kwargs)
                    try:
                        asyncio.create_task(
                            self._publish(
                                phase="outgoing",
                                answer=_coerce_text_value(result),
                            ),
                        )
                    except Exception:
                        pass
                    return result

            return _wrapped

    return _LoggedHandle(inner)


# ---------------------------------------------------------------------------
#  3.  Generic decorator for ManagerMethod logging
# ---------------------------------------------------------------------------


def log_manager_call(
    manager_name: str,
    method_name: str,
    payload_key: str,
    *,
    call_id_kw: str = "_call_id",
    display_label: str | None = None,
):
    """Decorator factory that publishes an incoming ManagerMethod event and
    wraps the returned handle so subsequent interactions are logged.

    The decorated coroutine must accept a text payload as its first positional
    argument or via a keyword named ``text``. A fresh ``call_id`` is injected
    into the method as a keyword argument named by ``call_id_kw`` so that the
    implementation can tag any sub-events (e.g. clarification requests) with
    the same identifier.

    ``display_label`` is a user-friendly phrase (e.g. "Checking Contact Book")
    that gets attached to every event in the lifecycle so the frontend can
    render it directly without maintaining its own mapping.
    """

    def _decorator(func):
        import inspect as _inspect

        _sig = _inspect.signature(func)
        _accepts_call_id = (
            call_id_kw in _sig.parameters
            or any(p.kind == p.VAR_KEYWORD for p in _sig.parameters.values())
        )

        @functools.wraps(func, updated=())
        async def _wrapper(self, *args, **kwargs):
            # Prefer the declared payload_key if present, falling back to the common "text"
            # convention or first positional arg.
            if payload_key in kwargs:
                payload_value = kwargs[payload_key]
            elif "text" in kwargs:
                payload_value = kwargs["text"]
            elif len(args) >= 1:
                payload_value = args[0]
            else:
                payload_value = ""

            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                manager_name,
                method_name,
                phase="incoming",
                display_label=display_label,
                **{payload_key: payload_value},
            )

            # Inject call_id only when the wrapped method declares it (for clarification events, etc.)
            if _accepts_call_id:
                kwargs[call_id_kw] = call_id
            handle = await func(self, *args, **kwargs)
            return wrap_handle_with_logging(
                handle, call_id, manager_name, method_name, display_label=display_label,
            )

        return _wrapper

    return _decorator
