from __future__ import annotations
import re
import pytest

from tests.helpers import _handle_project, capture_events
from unity.events.event_bus import EVENT_BUS
from unity.events.manager_event_logging import (
    wrap_handle_with_logging,
    new_call_id,
    log_manager_call,
    log_manager_result,
)
from unity.common.async_tool_loop import SteerableToolHandle
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

_SUFFIX_RE = re.compile(r"\(([0-9a-f]{4})\)$")


class _TupleAnswerHandle(SteerableToolHandle):  # returns [answer, steps]
    def __init__(self) -> None:
        self._done = False
        self._paused = False

    # SteerableToolHandle API
    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        return self

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ):
        return "ack"

    # SteerableToolHandle API
    async def stop(
        self,
        reason: str | None = None,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ):
        self._done = True

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
    async def ask(self, question: str, *, _parent_chat_context_cont: list[dict] | None = None) -> "SteerableToolHandle":  # type: ignore[override]
        return self

    async def interject(self, message: str, *, _parent_chat_context_cont: list[dict] | None = None) -> None:  # type: ignore[override]
        return None

    async def stop(self, reason: str | None = None):  # type: ignore[override]
        pass

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
    async def stop(
        self,
        *,
        reason: str | None = None,
        abandon: bool = False,
    ) -> None:
        """Stop execution with an optional reason; abandon toggles cancellation semantics."""
        self.stop_calls.append({"reason": reason, "abandon": abandon})
        self._done_ev.set()

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

    await logged.stop(reason="please-stop")
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


# ============================================================================
#  log_manager_result decorator tests
# ============================================================================
# The ``log_manager_result`` decorator is the counterpart to
# ``log_manager_call`` for manager methods that return plain results
# (str, dict, etc.) rather than SteerableToolHandle objects.  It is used
# by MemoryManager.
#
# Tests verify:
# 1. Incoming/outgoing ManagerMethod event pairs are published
# 2. ``display_label`` propagates to both events
# 3. ``TOOL_LOOP_LINEAGE`` is set for the method body and reset afterwards
# 4. Error paths publish outgoing events with status="error"
# 5. The payload_key is correctly extracted from args/kwargs


class _StubManager:
    """Minimal manager stub for testing the log_manager_result decorator."""

    @log_manager_result(
        "StubManager",
        "process",
        payload_key="text",
        display_label="Processing Data",
    )
    async def process(self, text: str) -> str:
        # Record the lineage visible inside the method body
        self._inner_lineage = list(TOOL_LOOP_LINEAGE.get([]))
        return f"processed: {text}"

    @log_manager_result(
        "StubManager",
        "fail",
        payload_key="text",
        display_label="Failing Gracefully",
    )
    async def fail(self, text: str) -> str:
        raise ValueError("intentional error")

    @log_manager_result(
        "StubManager",
        "pollute",
        payload_key="text",
        display_label="Polluting Lineage",
    )
    async def pollute(self, text: str) -> str:
        """Simulates what a real MemoryManager method does: the inner body
        modifies TOOL_LOOP_LINEAGE (as start_async_tool_loop would) and does
        NOT restore it.  The decorator must handle this gracefully."""
        current = list(TOOL_LOOP_LINEAGE.get([]))
        TOOL_LOOP_LINEAGE.set([*current, "inner_tool_loop"])
        return f"polluted: {text}"


@pytest.mark.asyncio
@_handle_project
async def test_result_publishes_incoming_and_outgoing():
    """A successful call should produce exactly one incoming and one outgoing event."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        result = await mgr.process("hello")

    EVENT_BUS.join_published()

    assert result == "processed: hello"

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
        and e.payload.get("phase") == "incoming"
    ]
    assert len(incoming) == 1, f"Expected 1 incoming event, got {len(incoming)}"
    assert incoming[0].payload.get("text") == "hello"

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert len(outgoing) == 1, f"Expected 1 outgoing event, got {len(outgoing)}"
    assert outgoing[0].payload.get("answer") == "processed: hello"
    assert outgoing[0].payload.get("status", "ok") == "ok"


@pytest.mark.asyncio
@_handle_project
async def test_result_display_label_propagates():
    """Both incoming and outgoing events should carry the display_label."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("test")

    EVENT_BUS.join_published()

    stub_events = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
    ]
    assert len(stub_events) >= 2

    for evt in stub_events:
        assert (
            evt.payload.get("display_label") == "Processing Data"
        ), f"display_label missing or wrong on {evt.payload.get('phase')} event"


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_set_during_body():
    """TOOL_LOOP_LINEAGE should include the method's leaf during execution."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod"):
        await mgr.process("lineage-test")

    assert hasattr(mgr, "_inner_lineage")
    assert any("StubManager.process" in s for s in mgr._inner_lineage)


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_reset_after_return():
    """TOOL_LOOP_LINEAGE should be restored to its original value after the call."""
    mgr = _StubManager()
    lineage_before = list(TOOL_LOOP_LINEAGE.get([]))

    async with capture_events("ManagerMethod"):
        await mgr.process("reset-test")

    lineage_after = list(TOOL_LOOP_LINEAGE.get([]))
    assert lineage_after == lineage_before


@pytest.mark.asyncio
@_handle_project
async def test_result_lineage_reset_after_error():
    """TOOL_LOOP_LINEAGE should be restored even when the method raises."""
    mgr = _StubManager()
    lineage_before = list(TOOL_LOOP_LINEAGE.get([]))

    async with capture_events("ManagerMethod"):
        with pytest.raises(ValueError, match="intentional error"):
            await mgr.fail("boom")

    lineage_after = list(TOOL_LOOP_LINEAGE.get([]))
    assert lineage_after == lineage_before


@pytest.mark.asyncio
@_handle_project
async def test_result_error_publishes_outgoing_with_error_status():
    """When the method raises, the outgoing event should carry error details."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        with pytest.raises(ValueError):
            await mgr.fail("error-test")

    EVENT_BUS.join_published()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "fail"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming, "No incoming event for failing method"
    call_id = incoming[0].calling_id

    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing event for failing method"
    assert outgoing[0].payload.get("status") == "error"
    assert "intentional error" in outgoing[0].payload.get("error", "")
    assert outgoing[0].payload.get("error_type") == "ValueError"
    assert outgoing[0].payload.get("display_label") == "Failing Gracefully"


@pytest.mark.asyncio
@_handle_project
async def test_result_inherits_parent_lineage():
    """When called under an existing lineage, the hierarchy should include the parent."""
    mgr = _StubManager()

    token = TOOL_LOOP_LINEAGE.set(["OuterManager.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            await mgr.process("nested-test")

        EVENT_BUS.join_published()

        incoming = [
            e
            for e in events
            if e.payload.get("manager") == "StubManager"
            and e.payload.get("method") == "process"
            and e.payload.get("phase") == "incoming"
        ]
        assert incoming
        hierarchy = incoming[0].payload.get("hierarchy", [])
        assert len(hierarchy) == 2
        assert hierarchy[0] == "OuterManager.act"
        assert re.match(r"StubManager\.process\([0-9a-f]{4}\)$", hierarchy[1])
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_result_payload_key_from_positional_arg():
    """The payload_key should be extracted from the first positional arg."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("positional-value")

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    assert incoming[0].payload.get("text") == "positional-value"


@pytest.mark.asyncio
@_handle_project
async def test_result_outgoing_hierarchy_not_polluted_by_inner_lineage():
    """When the method body modifies TOOL_LOOP_LINEAGE (as start_async_tool_loop
    does), the outgoing event must still carry the correct hierarchy — not the
    polluted value left by the inner code."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.pollute("test")

    EVENT_BUS.join_published()

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "pollute"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming, "No incoming event"
    incoming_hierarchy = incoming[0].payload.get("hierarchy", [])

    call_id = incoming[0].calling_id
    outgoing = [
        e
        for e in events
        if e.calling_id == call_id and e.payload.get("phase") == "outgoing"
    ]
    assert outgoing, "No outgoing event"
    outgoing_hierarchy = outgoing[0].payload.get("hierarchy", [])

    assert (
        incoming_hierarchy == outgoing_hierarchy
    ), f"Hierarchy mismatch: incoming={incoming_hierarchy}, outgoing={outgoing_hierarchy}"
    assert len(incoming_hierarchy) == 1
    assert re.match(r"StubManager\.pollute\([0-9a-f]{4}\)$", incoming_hierarchy[0])
    assert len(outgoing_hierarchy) == 1
    assert re.match(r"StubManager\.pollute\([0-9a-f]{4}\)$", outgoing_hierarchy[0])


@pytest.mark.asyncio
@_handle_project
async def test_result_incoming_hierarchy_not_doubled():
    """The incoming event hierarchy must not contain the leaf twice.

    Regression: if TOOL_LOOP_LINEAGE is set *before* publishing the incoming
    event, publish_manager_method_event reads the already-set lineage and
    appends the leaf a second time."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("double-check")

    incoming = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
        and e.payload.get("phase") == "incoming"
    ]
    assert incoming
    hierarchy = incoming[0].payload.get("hierarchy", [])
    assert len(hierarchy) == 1 and re.match(
        r"StubManager\.process\([0-9a-f]{4}\)$",
        hierarchy[0],
    ), f"Hierarchy wrong: {hierarchy}"


def _extract_suffix(hierarchy_label: str) -> str | None:
    """Extract the trailing 4-hex-char suffix from a hierarchy_label like 'Foo.bar(c8f5)'."""
    m = _SUFFIX_RE.search(hierarchy_label)
    return m.group(1) if m else None


# ============================================================================
#  Suffix consistency tests
# ============================================================================
# The hierarchy_label suffix (4-char hex, e.g. "(c8f5)") must be identical
# across ALL events for the same operation: incoming, outgoing, proxy actions,
# and ToolLoop events.  Before the fix, the incoming event generated its own
# random suffix independently of the loop's suffix.


class _HandleForSuffixTest(SteerableToolHandle):
    """Minimal handle for suffix consistency tests via log_manager_call."""

    def __init__(self) -> None:
        self._done = False

    async def ask(self, question: str, *, _parent_chat_context_cont=None):
        return self

    async def interject(self, message: str, *, _parent_chat_context_cont=None):
        pass

    async def stop(self, reason=None):
        self._done = True

    async def pause(self):
        pass

    async def resume(self):
        pass

    def done(self):
        return self._done

    async def result(self):
        self._done = True
        return "suffix-test-done"

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass


class _SuffixCallManager:
    """Stub manager that uses log_manager_call and returns a handle."""

    @log_manager_call(
        "SuffixCallMgr",
        "run",
        payload_key="request",
        display_label="Running Suffix Test",
    )
    async def run(self, request: str, **kwargs) -> _HandleForSuffixTest:
        return _HandleForSuffixTest()


@pytest.mark.asyncio
@_handle_project
async def test_suffix_consistent_across_incoming_outgoing():
    """log_manager_call: the hierarchy_label suffix on the incoming event must
    match the suffix on subsequent proxy events (outgoing, action, etc.)."""
    mgr = _SuffixCallManager()

    async with capture_events("ManagerMethod") as events:
        handle = await mgr.run("suffix-check")
        await handle.result()

    EVENT_BUS.join_published()

    mgr_events = [
        e
        for e in events
        if e.payload.get("manager") == "SuffixCallMgr"
        and e.payload.get("method") == "run"
    ]
    assert len(mgr_events) >= 2, f"Expected >= 2 events, got {len(mgr_events)}"

    incoming = [e for e in mgr_events if e.payload.get("phase") == "incoming"]
    assert incoming, "No incoming event found"
    incoming_suffix = _extract_suffix(incoming[0].payload.get("hierarchy_label", ""))
    assert (
        incoming_suffix
    ), f"Could not extract suffix from incoming: {incoming[0].payload.get('hierarchy_label')}"

    non_incoming = [e for e in mgr_events if e.payload.get("phase") != "incoming"]
    assert non_incoming, "No non-incoming events found"

    for evt in non_incoming:
        evt_suffix = _extract_suffix(evt.payload.get("hierarchy_label", ""))
        assert evt_suffix == incoming_suffix, (
            f"Suffix mismatch: incoming=({incoming_suffix}), "
            f"phase={evt.payload.get('phase')}/action={evt.payload.get('action')} "
            f"has ({evt_suffix})"
        )


@pytest.mark.asyncio
@_handle_project
async def test_result_suffix_consistent_across_incoming_outgoing():
    """log_manager_result: the hierarchy_label suffix on the incoming event must
    match the suffix on the outgoing event."""
    mgr = _StubManager()

    async with capture_events("ManagerMethod") as events:
        await mgr.process("suffix-result-check")

    EVENT_BUS.join_published()

    stub_events = [
        e
        for e in events
        if e.payload.get("manager") == "StubManager"
        and e.payload.get("method") == "process"
    ]
    assert len(stub_events) >= 2

    incoming = [e for e in stub_events if e.payload.get("phase") == "incoming"]
    outgoing = [e for e in stub_events if e.payload.get("phase") == "outgoing"]
    assert incoming and outgoing

    incoming_suffix = _extract_suffix(incoming[0].payload.get("hierarchy_label", ""))
    outgoing_suffix = _extract_suffix(outgoing[0].payload.get("hierarchy_label", ""))
    assert (
        incoming_suffix
    ), f"No suffix on incoming: {incoming[0].payload.get('hierarchy_label')}"
    assert (
        outgoing_suffix
    ), f"No suffix on outgoing: {outgoing[0].payload.get('hierarchy_label')}"
    assert (
        incoming_suffix == outgoing_suffix
    ), f"Suffix mismatch: incoming=({incoming_suffix}), outgoing=({outgoing_suffix})"


# ============================================================================
#  Handle repr leak tests
# ============================================================================
# The _LoggedHandle proxy publishes an outgoing event for every wrapped method.
# When ask() returns a sub-handle (SteerableToolHandle), the answer field must
# be None — not a garbage repr string like
# "<unity.common.async_tool_loop.AsyncToolLoopHandle object at 0x...>".
# The frontend interprets any non-trivial answer as displayable content, so a
# leaked repr overwrites the real answer on the node.


@pytest.mark.asyncio
@_handle_project
async def test_ask_does_not_publish_handle_repr_as_answer():
    """When ask() returns a sub-handle, the outgoing event's answer field must
    be None — not the handle's __repr__ string."""
    inner = _TupleAnswerHandle()
    call_id = new_call_id()
    logged = wrap_handle_with_logging(inner, call_id, "UnitTestManager", "ask")

    async with capture_events("ManagerMethod") as captured_events:
        sub_handle = await logged.ask("test question")

    EVENT_BUS.join_published()

    assert hasattr(sub_handle, "done"), "ask() should return a handle-like object"

    outgoing = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("phase") == "outgoing"
    ]

    assert len(outgoing) == 1, f"Expected 1 outgoing event, got {len(outgoing)}"
    answer = outgoing[0].payload.get("answer")
    assert answer is None, (
        f"Expected answer=None for a handle return, got {answer!r}. "
        "Handle reprs must not leak into the answer field."
    )


# ============================================================================
#  Steering content forwarding tests
# ============================================================================
# User-facing steering methods (interject, ask, stop) should forward their
# first positional argument to the ManagerMethod action event so the frontend
# can display the text inline (e.g. next to the "interjected" label).
# Each method writes to the semantically correct payload field:
#   ask      -> question
#   interject -> instructions
#   stop     -> instructions


@pytest.mark.asyncio
@_handle_project
async def test_interject_action_event_carries_message():
    """The action event for interject() must include the message in the
    'instructions' payload field."""
    inner = _TupleAnswerHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    async with capture_events("ManagerMethod") as captured_events:
        await logged.interject("new requirements from the boss")

    EVENT_BUS.join_published()

    action_events = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("action") == "interject"
    ]

    assert len(action_events) == 1, f"Expected 1 action event, got {len(action_events)}"
    assert (
        action_events[0].payload.get("instructions") == "new requirements from the boss"
    ), (
        f"Expected instructions='new requirements from the boss', "
        f"got {action_events[0].payload.get('instructions')!r}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_ask_action_event_carries_question():
    """The action event for ask() must include the question in the
    'question' payload field."""
    inner = _TupleAnswerHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    async with capture_events("ManagerMethod") as captured_events:
        await logged.ask("what is the current status?")

    EVENT_BUS.join_published()

    action_events = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("action") == "ask"
    ]

    assert len(action_events) == 1, f"Expected 1 action event, got {len(action_events)}"
    assert action_events[0].payload.get("question") == "what is the current status?", (
        f"Expected question='what is the current status?', "
        f"got {action_events[0].payload.get('question')!r}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_stop_action_event_carries_reason():
    """The action event for stop() must include the reason in the
    'instructions' payload field."""
    inner = _TupleAnswerHandle()
    logged = wrap_handle_with_logging(inner, new_call_id(), "UnitTestManager", "ask")

    async with capture_events("ManagerMethod") as captured_events:
        await logged.stop(reason="user changed their mind")

    EVENT_BUS.join_published()

    action_events = [
        e
        for e in captured_events
        if e.payload.get("manager") == "UnitTestManager"
        and e.payload.get("method") == "ask"
        and e.payload.get("action") == "stop"
    ]

    assert len(action_events) == 1, f"Expected 1 action event, got {len(action_events)}"
    assert action_events[0].payload.get("instructions") == "user changed their mind", (
        f"Expected instructions='user changed their mind', "
        f"got {action_events[0].payload.get('instructions')!r}"
    )
