from __future__ import annotations

from typing import Any, Callable
import functools
from uuid import uuid4
import json
from secrets import token_hex

from ..events.event_bus import EVENT_BUS, Event
from ..events.types.manager_method import ManagerMethodPayload
from ..common.async_tool_loop import SteerableToolHandle
from ..common._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX

from contextvars import ContextVar

# Caller context propagation: set by the ConversationManager (or other
# top-level orchestrators) before dispatching tool calls so that every
# ManagerMethod event published within that scope carries the caller's
# identity.  MemoryManager already filters on
#   source == "ConversationManager"
# (see register_auto_pin / _setup_explicit_call_callbacks), but the field
# was never actually populated until now.
_EVENT_SOURCE: ContextVar[str | None] = ContextVar("_EVENT_SOURCE", default=None)


__all__ = [
    "_EVENT_SOURCE",
    "new_call_id",
    "publish_manager_method_event",
    "wrap_handle_with_logging",
    "log_manager_call",
    "log_manager_result",
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
    - If the caller provides an explicit ``hierarchy`` (e.g., decorators and boundary
      wrappers), it is used as-is — segments are expected to already carry per-invocation
      suffixes (e.g., ``["CodeActActor.act(fb7a)", "execute_code(a318)"]``).
    - Otherwise, reads the current lineage from ``TOOL_LOOP_LINEAGE`` (already suffixed),
      appends a new suffixed leaf ``"{manager}.{method}({suffix})"`` and uses that.

    ``hierarchy_label`` is always derived as ``"->".join(hierarchy)`` — it is kept for
    backward compatibility but carries no independent information.
    TODO: remove hierarchy_label from payloads once frontend migrates.
    """
    if source is None:
        source = _EVENT_SOURCE.get(None)

    # Work on a copy so we can safely pop internal-only keys without mutating caller dict.
    extra_clean: dict[str, Any] = dict(extra)

    # Best-effort lineage read (ContextVar) — segments are already suffixed.
    parent_lineage: list[str] = []
    try:
        val = TOOL_LOOP_LINEAGE.get([])
        if isinstance(val, list):
            parent_lineage = list(val)
    except Exception:
        parent_lineage = []

    # Determine effective hierarchy.
    hierarchy: list[str]
    if "hierarchy" in extra_clean and isinstance(extra_clean.get("hierarchy"), list):
        try:
            hierarchy = list(extra_clean.get("hierarchy") or [])
        except Exception:
            hierarchy = []
    else:
        # Auto-compute: suffixed parent segments + new suffixed leaf.
        # Conceptually matches how decorators/boundary wrappers build hierarchy.
        suffix = extra_clean.get("suffix")
        if not isinstance(suffix, str) or not suffix:
            suffix = token_hex(2)
        leaf = (
            f"{manager_name}.{method_name}({suffix})"
            if manager_name and method_name
            else ""
        )
        hierarchy = [*parent_lineage, leaf] if leaf else list(parent_lineage)

    # hierarchy_label is trivially derived — no separate suffix/label logic.
    # TODO: remove hierarchy_label from payloads once frontend migrates.
    hierarchy_label = "->".join(str(s) for s in hierarchy) if hierarchy else ""

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


def wrap_handle_with_logging(
    inner: SteerableToolHandle,
    call_id: str,
    manager_name: str,
    method_name: str,
    *,
    display_label: str | None = None,
    hierarchy: list[str] | None = None,
) -> SteerableToolHandle:
    """Monkey-patch result() to publish the outgoing ManagerMethod event on completion."""
    _orig_result = inner.result
    _hierarchy = list(hierarchy) if hierarchy else []

    async def _result_with_outgoing():
        try:
            ans = await _orig_result()
        except Exception as exc:
            import traceback as _tb

            await publish_manager_method_event(
                call_id,
                manager_name,
                method_name,
                phase="outgoing",
                status="error",
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=_tb.format_exc()[:2000],
                display_label=display_label,
                hierarchy=_hierarchy,
            )
            raise
        await publish_manager_method_event(
            call_id,
            manager_name,
            method_name,
            phase="outgoing",
            answer=_coerce_text_value(ans),
            display_label=display_label,
            hierarchy=_hierarchy,
        )
        return ans

    inner.result = _result_with_outgoing  # type: ignore[attr-defined]
    return inner


# ---------------------------------------------------------------------------
#  3.  Generic decorator for ManagerMethod logging
# ---------------------------------------------------------------------------


def log_manager_call(
    manager_name: str,
    method_name: str,
    payload_key: str,
    *,
    call_id_kw: str = "_call_id",
    display_label: str | Callable[..., str] | None = None,
    forward_kwargs: tuple[str, ...] = (),
):
    """Decorator factory that publishes an incoming ManagerMethod event and
    wraps the returned handle so subsequent interactions are logged.

    The decorated coroutine must accept a text payload as its first positional
    argument or via a keyword named ``text``. A fresh ``call_id`` is injected
    into the method as a keyword argument named by ``call_id_kw`` so that the
    implementation can tag any sub-events (e.g. clarification requests) with
    the same identifier.

    ``display_label`` is a user-friendly phrase (e.g. "Checking contact book")
    that gets attached to every event in the lifecycle so the frontend can
    render it directly without maintaining its own mapping.  May also be a
    callable ``(kwargs) -> str`` for labels that depend on runtime arguments.
    """

    def _decorator(func):
        import inspect as _inspect

        _sig = _inspect.signature(func)
        _accepts_call_id = call_id_kw in _sig.parameters or any(
            p.kind == p.VAR_KEYWORD for p in _sig.parameters.values()
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

            resolved_label = (
                display_label(kwargs) if callable(display_label) else display_label
            )

            call_id = new_call_id()

            # Generate a single suffix, embed it into the hierarchy leaf,
            # and bridge it to _PENDING_LOOP_SUFFIX so the inner LoopConfig
            # produces a matching suffixed lineage segment.
            suffix = token_hex(2)
            parent_lineage: list[str] = []
            try:
                val = TOOL_LOOP_LINEAGE.get([])
                if isinstance(val, list):
                    parent_lineage = list(val)
            except Exception:
                pass
            hierarchy = [
                *parent_lineage,
                f"{manager_name}.{method_name}({suffix})",
            ]

            extra_fields: dict[str, Any] = {}
            for fk in forward_kwargs:
                fv = kwargs.get(fk)
                if fv is not None:
                    extra_fields[fk] = fv

            suffix_token = _PENDING_LOOP_SUFFIX.set(suffix)
            try:
                try:
                    await publish_manager_method_event(
                        call_id,
                        manager_name,
                        method_name,
                        phase="incoming",
                        display_label=resolved_label,
                        hierarchy=hierarchy,
                        **{payload_key: payload_value},
                        **extra_fields,
                    )
                except Exception:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "Failed to publish incoming ManagerMethod event "
                        "for %s.%s — proceeding with method execution",
                        manager_name,
                        method_name,
                        exc_info=True,
                    )

                # Inject call_id only when the wrapped method declares it
                if _accepts_call_id:
                    kwargs[call_id_kw] = call_id
                handle = await func(self, *args, **kwargs)
            finally:
                _PENDING_LOOP_SUFFIX.reset(suffix_token)

            return wrap_handle_with_logging(
                handle,
                call_id,
                manager_name,
                method_name,
                display_label=resolved_label,
                hierarchy=hierarchy,
            )

        return _wrapper

    return _decorator


# ---------------------------------------------------------------------------
#  4.  Decorator for methods that return plain results (not handles)
# ---------------------------------------------------------------------------


def log_manager_result(
    manager_name: str,
    method_name: str,
    payload_key: str,
    *,
    display_label: str | None = None,
):
    """Decorator factory for manager methods that return a plain result (str,
    dict, etc.) rather than a :class:`SteerableToolHandle`.

    Publishes incoming/outgoing ``ManagerMethod`` events and sets
    ``TOOL_LOOP_LINEAGE`` so that any inner tool loops or nested manager calls
    inherit the correct parent lineage.

    This is the counterpart to :func:`log_manager_call` for managers like
    ``MemoryManager`` whose public methods ``await handle.result()`` internally
    and return the final value directly.
    """

    def _decorator(func):
        @functools.wraps(func, updated=())
        async def _wrapper(self, *args, **kwargs):
            # Extract the payload value for the incoming event
            if payload_key in kwargs:
                payload_value = kwargs[payload_key]
            elif "text" in kwargs:
                payload_value = kwargs["text"]
            elif len(args) >= 1:
                payload_value = args[0]
            else:
                payload_value = ""

            call_id = new_call_id()

            # Pre-compute the suffixed hierarchy for this method.
            parent_lineage: list[str] = []
            try:
                val = TOOL_LOOP_LINEAGE.get([])
                if isinstance(val, list):
                    parent_lineage = list(val)
            except Exception:
                pass
            suffix = token_hex(2)
            hierarchy = [
                *parent_lineage,
                f"{manager_name}.{method_name}({suffix})",
            ]

            # ── 1. Publish incoming BEFORE modifying the lineage ──────────
            await publish_manager_method_event(
                call_id,
                manager_name,
                method_name,
                phase="incoming",
                display_label=display_label,
                hierarchy=hierarchy,
                **{payload_key: payload_value},
            )

            # ── 2. Set the lineage frame so inner tool loops inherit it ───
            lineage_token = TOOL_LOOP_LINEAGE.set(hierarchy)

            try:
                result = await func(self, *args, **kwargs)

                # ── 3. Publish outgoing with the SAME hierarchy ───────────
                await publish_manager_method_event(
                    call_id,
                    manager_name,
                    method_name,
                    phase="outgoing",
                    display_label=display_label,
                    answer=_coerce_text_value(result),
                    hierarchy=hierarchy,
                )
                return result

            except Exception as exc:
                import traceback as _tb

                await publish_manager_method_event(
                    call_id,
                    manager_name,
                    method_name,
                    phase="outgoing",
                    display_label=display_label,
                    status="error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    traceback=_tb.format_exc()[:2000],
                    hierarchy=hierarchy,
                )
                raise

            finally:
                TOOL_LOOP_LINEAGE.reset(lineage_token)

        return _wrapper

    return _decorator
