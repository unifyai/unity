from __future__ import annotations
import pytest

from tests.helpers import _handle_project, capture_events
from unity.events.event_bus import EVENT_BUS
from unity.events.manager_event_logging import wrap_handle_with_logging, new_call_id
from unity.common.async_tool_loop import SteerableToolHandle


class _TupleAnswerHandle(SteerableToolHandle):  # returns [answer, steps]
    def __init__(self) -> None:
        self._done = False
        self._paused = False

    # SteerableHandle API
    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        return self

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        return "ack"

    # SteerableToolHandle API
    def stop(
        self,
        reason: str | None = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        self._done = True
        return "Stopped"

    async def pause(self):
        self._paused = True
        return "Paused"

    async def resume(self):
        self._paused = False
        return "Resumed"

    def done(self):
        return self._done

    async def result(self):
        self._done = True
        return ["OK", [{"role": "system", "content": "..."}]]

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class _PrivateAttrHandle(SteerableToolHandle):
    """Handle exposing private/dunder attributes to verify proxy forwarding."""

    def __init__(self) -> None:
        self._internal_flag = 123
        self.__opaque_flag__ = "ok"
        self._done = True

    # Minimal interface – inert stubs
    async def ask(self, question: str, *, parent_chat_context_cont: list[dict] | None = None) -> "SteerableToolHandle":  # type: ignore[override]
        return self

    async def interject(self, message: str, *, parent_chat_context_cont: list[dict] | None = None) -> None:  # type: ignore[override]
        return None

    def stop(self, reason: str | None = None):  # type: ignore[override]
        return "stopped"

    async def pause(self):  # type: ignore[override]
        return "paused"

    async def resume(self):  # type: ignore[override]
        return "resumed"

    def done(self):  # type: ignore[override]
        return self._done

    async def result(self):  # type: ignore[override]
        return ""

    async def next_clarification(self) -> dict:  # type: ignore[override]
        return {}

    async def next_notification(self) -> dict:  # type: ignore[override]
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return None


@pytest.mark.asyncio
@_handle_project
async def test_sanitizes_iterable_answer():
    inner = _TupleAnswerHandle()
    call_id = new_call_id()
    logged = wrap_handle_with_logging(inner, call_id, "UnitTestManager", "ask")

    # Capture ManagerMethod events to avoid race/I/O latency
    async with capture_events("ManagerMethod") as captured_events:
        # Invoke result() → wrapper publishes a ManagerMethod event with phase="outgoing"
        out = await logged.result()

    assert isinstance(out, list) and out[0] == "OK"

    # Ensure logs are flushed to backend before searching
    EVENT_BUS.join_published()

    # Fetch the newest ManagerMethod event for our manager/method with outgoing phase
    events = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("phase") == "outgoing"
    ]

    assert len(events) == 1
    evt = events[0]
    # The logged answer must be a string (sanitized from a list/tuple return)
    assert isinstance(evt.payload.get("answer"), str)
    assert evt.payload.get("answer") == "OK"


# ----------------------------------------------------------------------------
# Custom handle (mirrors tests/async_tool_loop/test_dynamic_custom_handle.py)
# ----------------------------------------------------------------------------


class _CustomArgsHandle(SteerableToolHandle):
    """Custom handle exposing extra kwargs on steering methods and a write-only helper."""

    def __init__(self) -> None:
        import asyncio as _asyncio  # local import for test isolation

        self._done_ev = _asyncio.Event()
        self._result_text: str = "inner-complete"
        self.interject_calls: list[dict] = []
        self.pause_calls: list[dict] = []
        self.resume_calls: list[dict] = []
        self.stop_calls: list[dict] = []
        self.ask_calls: list[dict] = []

    # Custom async method to verify logging of non-standard methods via awaited publish
    async def ping(self, *, note: str | None = None) -> str:
        return "pong"

    # Read-only ask with extra kwarg
    async def ask(
        self,
        question: str,
        *,
        style: str = "short",
    ) -> "SteerableToolHandle":
        """Ask about current state using a specified response style."""
        self.ask_calls.append({"question": question, "style": style})
        return self

    # Interject with extra kwargs (no images kw on purpose to surface wrapper behavior)
    async def interject(
        self,
        message: str,
        *,
        priority: int = 1,
        metadata: dict | None = None,
    ) -> None:
        """Interject a message with priority and metadata tags."""
        self.interject_calls.append(
            {"message": message, "priority": priority, "metadata": metadata or {}},
        )
        return None

    # Stop with a different kw (abandon) than wrapper's cancel
    def stop(
        self,
        *,
        reason: str | None = None,
        abandon: bool = False,
    ) -> str | None:
        """Stop execution with an optional reason; abandon toggles cancellation semantics."""
        self.stop_calls.append({"reason": reason, "abandon": abandon})
        self._done_ev.set()
        return "stopped"

    # Pause/Resume with required/optional kwargs
    async def pause(self, *, reason: str, log_to_backend: bool = False) -> str | None:
        """Pause processing for a specific reason; optionally log to backend."""
        self.pause_calls.append({"reason": reason, "log_to_backend": log_to_backend})
        return "paused"

    async def resume(self, *, resume_token: str | None = None) -> str | None:
        """Resume processing using an optional token."""
        self.resume_calls.append({"resume_token": resume_token})
        return "resumed"

    # Write-only helper: terminate with an "aborted" result
    def abort(self, *, reason: str | None = None) -> None:
        self._result_text = "aborted"
        self._done_ev.set()
        return None

    def done(self) -> bool:
        return self._done_ev.is_set()

    async def result(self) -> str:
        await self._done_ev.wait()
        return self._result_text

    # Abstract event APIs – inert stubs
    async def next_clarification(self) -> dict:  # pragma: no cover
        return {}

    async def next_notification(self) -> dict:  # pragma: no cover
        return {}

    async def answer_clarification(
        self,
        call_id: str,
        answer: str,
    ) -> None:  # pragma: no cover
        return None


@pytest.mark.asyncio
@_handle_project
async def test_forwards_custom_ask_kwargs():
    """ask() kwargs should pass through the logging wrapper to the inner handle."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    # exercise ask with an extra kwarg
    _ = await logged.ask("How are you?", style="long")
    assert inner.ask_calls and inner.ask_calls[-1]["style"] == "long"


@pytest.mark.asyncio
@_handle_project
async def test_preserves_write_only_passthrough():
    """Unknown attributes should be proxied via __getattr__; abort() should work."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(
        inner,
        new_call_id(),
        "UnitTestManager",
        "execute",
    )

    # Call write-only helper through the wrapper
    logged.abort(reason="test-abort")  # type: ignore[attr-defined]
    out = await logged.result()
    assert out == "aborted"


@pytest.mark.asyncio
@_handle_project
async def test_stop_invokes_inner():
    """Wrapper stop() should successfully invoke inner stop with reason."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(
        inner,
        new_call_id(),
        "UnitTestManager",
        "execute",
    )

    ret = logged.stop(reason="please-stop")
    assert ret == "stopped"
    # inner recorded call with our reason; abandon defaults to False
    assert inner.stop_calls and inner.stop_calls[-1]["reason"] == "please-stop"
    assert inner.stop_calls[-1]["abandon"] is False


@pytest.mark.asyncio
@_handle_project
async def test_interject_forwards_kwargs():
    """Interject kwargs should be forwarded; current wrapper passes only images kw (expected to fail)."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    await logged.interject("hello", priority=3, metadata={"k": "v"})  # type: ignore[call-arg]
    assert inner.interject_calls and inner.interject_calls[-1]["priority"] == 3
    assert inner.interject_calls[-1]["metadata"] == {"k": "v"}


@pytest.mark.asyncio
@_handle_project
async def test_pause_resume_forward_kwargs():
    """Pause/Resume kwargs should be forwarded; current wrapper drops them (expected to fail)."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(
        inner,
        new_call_id(),
        "UnitTestManager",
        "execute",
    )

    # pass custom kwargs to pause/resume via the wrapper
    assert await logged.pause(reason="testing", log_to_backend=True) == "paused"  # type: ignore[call-arg]
    assert await logged.resume(resume_token="abc") == "resumed"  # type: ignore[call-arg]
    assert inner.pause_calls and inner.pause_calls[-1] == {
        "reason": "testing",
        "log_to_backend": True,
    }
    assert inner.resume_calls and inner.resume_calls[-1] == {"resume_token": "abc"}


@pytest.mark.asyncio
@_handle_project
async def test_handle_reports_same_class():
    """Proxy should present the same class via __class__ spoofing for reflection."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    # __class__ should appear identical; type(logged) is the proxy but __class__ is spoofed
    assert logged.__class__ is inner.__class__
    assert logged.__class__.__name__ == inner.__class__.__name__


@pytest.mark.asyncio
@_handle_project
async def test_handle_includes_all_inner_methods():
    """All public callables on the inner handle must be present and callable on the proxy."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    inner_methods = {
        name
        for name in dir(inner)
        if not name.startswith("__") and callable(getattr(inner, name, None))
    }
    # spot-check a custom method
    assert "abort" in inner_methods

    for name in inner_methods:
        attr = getattr(logged, name, None)
        assert callable(attr), f"Proxy missing callable {name}"


@pytest.mark.asyncio
@_handle_project
async def test_preserves_doc_for_unwrapped():
    """For methods not overridden by the proxy, docstrings and signatures should match exactly."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    import inspect as _inspect

    # Methods the proxy overrides (skip these)
    overridden = {
        "interject",
        "pause",
        "resume",
        "stop",
        "done",
        "result",
        "ask",
        "serialize",
        "next_clarification",
        "next_notification",
        "answer_clarification",
    }

    # Choose a custom method that is not overridden
    name = "abort"
    assert name not in overridden

    inner_fn = getattr(inner, name)
    logged_fn = getattr(logged, name)

    assert _inspect.getdoc(logged_fn) == _inspect.getdoc(inner_fn)
    assert str(_inspect.signature(logged_fn)) == str(_inspect.signature(inner_fn))


@pytest.mark.asyncio
@_handle_project
async def test_doc_match_for_overridden():
    """Overridden methods should also mirror doc/signature (current behavior expected to fail)."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    import inspect as _inspect

    # Pick a couple of overridden methods
    for name in ("interject", "ask", "pause", "resume", "stop"):
        inner_fn = getattr(inner, name)
        logged_fn = getattr(logged, name)
        assert _inspect.getdoc(logged_fn) == _inspect.getdoc(inner_fn)
        assert str(_inspect.signature(logged_fn)) == str(_inspect.signature(inner_fn))


@pytest.mark.asyncio
@_handle_project
async def test_logs_custom_method_calls():
    """Custom async handle methods should emit ManagerMethod events with action=method name."""
    inner = _CustomArgsHandle()
    logged = wrap_handle_with_logging(
        inner,
        new_call_id(),
        "UnitTestManager",
        "execute",
    )

    async with capture_events("ManagerMethod") as captured_events:
        # Invoke a custom, non-standard ASYNC method to avoid scheduling races
        pong = await logged.ping(note="ensure-logged")  # type: ignore[attr-defined]

    assert pong == "pong"

    # Ensure all async publish tasks complete before querying the event log
    EVENT_BUS.join_published()

    # Verify that a ManagerMethod event was recorded for this custom action
    events = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "execute"
        and e.payload.get("action") == "ping"
    ]

    assert len(events) == 1


@pytest.mark.asyncio
@_handle_project
async def test_forwards_private_attributes():
    """Private (single-underscore) and dunder attributes must be forwarded to the inner handle."""
    inner = _PrivateAttrHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    # Access private attribute via proxy – should reflect inner value
    assert getattr(logged, "_internal_flag") == 123
    # Access a dunder-style attribute (non-mangled)
    assert getattr(logged, "__opaque_flag__") == "ok"
