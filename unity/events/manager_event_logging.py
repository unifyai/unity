from __future__ import annotations

import asyncio
from typing import Any
import functools
from uuid import uuid4

from ..events.event_bus import EVENT_BUS, Event
from ..common.llm_helpers import SteerableToolHandle

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
    **payload: Any,
) -> None:
    """
    Thin wrapper around :pyfunc:`EVENT_BUS.publish` for *ManagerMethod* events.
    """
    await EVENT_BUS.publish(
        Event(
            type="ManagerMethod",
            calling_id=call_id,
            payload={
                "manager": manager_name,
                "method": method_name,
                **({"source": source} if source is not None else {}),
                **payload,
            },
        ),
    )


# ---------------------------------------------------------------------------
#  2.  Generic SteerableToolHandle wrapper
# ---------------------------------------------------------------------------


def wrap_handle_with_logging(
    inner: SteerableToolHandle,
    call_id: str,
    manager_name: str,
    method_name: str,
) -> SteerableToolHandle:
    """
    Return a SteerableToolHandle proxy which emits a **ManagerMethod** event every
    time the user interacts with the handle (pause/resume/…/result).
    """

    # --- normalize legacy handles that lack a 'cancel' kwarg on stop() -------
    try:
        import inspect as _inspect

        _sig = _inspect.signature(getattr(inner, "stop"))
        _has_cancel = any(
            p.kind
            in (
                _inspect.Parameter.KEYWORD_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.name == "cancel"
            for p in _sig.parameters.values()
        )
    except Exception:
        _has_cancel = True  # be permissive when unsure

    if not _has_cancel:

        class _StopAdapter(SteerableToolHandle):  # type: ignore[misc]
            __slots__ = ("_inner",)

            def __init__(self, _h: SteerableToolHandle) -> None:
                self._inner = _h

            # delegate public API, but adapt stop signature
            async def interject(self, message: str):
                return await self._inner.interject(message)

            def pause(self):
                return self._inner.pause()

            def resume(self):
                return self._inner.resume()

            def done(self):
                return self._inner.done()

            async def result(self):
                return await self._inner.result()

            async def ask(self, question: str, *a, **kw):
                return await self._inner.ask(question, *a, **kw)

            def stop(self, reason: str | None = None, *, cancel: bool | None = None):
                # Default cancel when omitted
                _cancel_flag: bool = True if cancel is None else bool(cancel)
                try:
                    return self._inner.stop(cancel=_cancel_flag, reason=reason)  # type: ignore[misc]
                except TypeError:
                    try:
                        return self._inner.stop(reason=reason)  # type: ignore[misc]
                    except TypeError:
                        if reason is not None:
                            return self._inner.stop(reason)  # type: ignore[misc]
                        return self._inner.stop()  # type: ignore[misc]

            def __getattr__(self, item):
                return getattr(self._inner, item)

        inner = _StopAdapter(inner)

    class _LoggedHandle(SteerableToolHandle):  # type: ignore[misc]
        __slots__ = ("_inner",)

        # ---------- lifecycle ------------------------------------------------
        def __init__(self, _h: SteerableToolHandle):
            self._inner = _h

        # ---------- private helper -------------------------------------------
        async def _publish(self, **payload):
            await publish_manager_method_event(
                call_id,
                manager_name,
                method_name,
                **payload,
            )

        # ---------- public API mirror ----------------------------------------
        async def interject(self, message: str):
            await self._publish(action="interject", content=message)
            return await self._inner.interject(message)

        def pause(self):
            asyncio.create_task(self._publish(action="pause"))
            return self._inner.pause()

        def resume(self):
            asyncio.create_task(self._publish(action="resume"))
            return self._inner.resume()

        def stop(self, reason: str | None = None, *, cancel: bool | None = None):
            asyncio.create_task(
                self._publish(action="stop", reason=reason, cancel=cancel),
            )
            # Canonical semantics: cancel defaults to True when omitted.
            _cancel_flag: bool = True if cancel is None else bool(cancel)
            # Adapt to legacy implementations that may not accept the 'cancel' kwarg.
            try:
                return self._inner.stop(cancel=_cancel_flag, reason=reason)  # type: ignore[misc]
            except TypeError:
                try:
                    # Older signature: stop(reason=...)
                    return self._inner.stop(reason=reason)  # type: ignore[misc]
                except TypeError:
                    try:
                        # Oldest signature: positional reason
                        if reason is not None:
                            return self._inner.stop(reason)  # type: ignore[misc]
                        return self._inner.stop()  # type: ignore[misc]
                    except Exception:
                        # Last resort: call without args
                        return self._inner.stop()  # type: ignore[misc]

        def done(self):
            return self._inner.done()

        async def result(self):
            answer = await self._inner.result()
            await self._publish(phase="outgoing", answer=answer)
            return answer

        async def ask(self, question: str, *a, **kw):
            await self._publish(action="ask", question=question)
            return await self._inner.ask(question, *a, **kw)

        # fallback for everything else
        def __getattr__(self, item):
            return getattr(self._inner, item)

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
):
    """Decorator factory that publishes an incoming ManagerMethod event and
    wraps the returned handle so subsequent interactions are logged.

    The decorated coroutine must accept a text payload as its first positional
    argument or via a keyword named ``text``. A fresh ``call_id`` is injected
    into the method as a keyword argument named by ``call_id_kw`` so that the
    implementation can tag any sub-events (e.g. clarification requests) with
    the same identifier.
    """

    def _decorator(func):
        @functools.wraps(func, updated=())
        async def _wrapper(self, *args, **kwargs):
            if "text" in kwargs:
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
                **{payload_key: payload_value},
            )

            # Inject call_id for the inner method (for clarification events, etc.)
            kwargs[call_id_kw] = call_id
            handle = await func(self, *args, **kwargs)
            return wrap_handle_with_logging(handle, call_id, manager_name, method_name)

        return _wrapper

    return _decorator
