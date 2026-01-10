"""
Tests for `SteerableToolPane` and its integration with `HierarchicalActor`.

- **Symbolic / unit-style tests**: Use lightweight fake `SteerableToolHandle` implementations
  to validate pane bookkeeping, watcher cleanup, event indexing, and steering semantics.

- **Integration tests**:
  - **CI-safe integrations**: Run a real `HierarchicalActor` with mocked primitives returning
    steerable handles (no real Unify backend needed).
  - **Real-manager integrations**: Exercise Actor → Pane → real state managers. These are
    marked with `@pytest.mark.requires_real_unify` and `@pytest.mark.eval` and are expected
    to run only in environments configured for real Unify access.

Key invariants covered:
- Handles are registered and indexed deterministically
- Clarifications/notifications are observed and surfaced via the pane
- Steering (targeted + broadcast) respects filters and is safe for completed handles
- Pane events can be captured into verification work items
"""

from __future__ import annotations

import asyncio
import contextlib
import textwrap
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from tests.test_async_tool_loop.async_helpers import _wait_for_condition
from tests.helpers import _handle_project
from unity.actor.environments import StateManagerEnvironment
from unity.actor.hierarchical_actor import HierarchicalActor, HierarchicalActorHandle
from unity.actor.steerable_tool_pane import BroadcastFilter, SteerableToolPane
from unity.common.async_tool_loop import SteerableToolHandle
from unity.function_manager.primitives import Primitives
from unity.transcript_manager.types.message import Message


class _MockHandle:
    """Minimal steerable-handle stub for pane unit tests."""

    def __init__(self) -> None:
        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notif_q: asyncio.Queue[dict] = asyncio.Queue()
        self.interject_calls: list[dict] = []
        self.answer_calls: list[tuple[str, str]] = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls: list[str | None] = []

    async def next_clarification(self) -> dict:
        return await self._clar_q.get()

    async def next_notification(self) -> dict:
        return await self._notif_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        self.answer_calls.append((call_id, answer))
        return None

    async def pause(self) -> str | None:
        self.pause_calls += 1
        return "paused"

    async def resume(self) -> str | None:
        self.resume_calls += 1
        return "resumed"

    async def stop(
        self,
        reason: str | None = None,
        *,
        parent_chat_context_cont=None,
    ) -> str | None:  # noqa: ARG002
        self.stop_calls.append(reason)
        return "stopped"

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont=None,
        images=None,
    ):  # noqa: ARG002
        self.interject_calls.append({"message": message})
        return "ok"

    # The remaining `SteerableToolHandle` methods aren't needed for Phase 1 tests.


class _BlockingHandle(_MockHandle):
    """A handle whose event streams never yield (useful for watcher cleanup tests)."""

    async def next_clarification(self) -> dict:  # pragma: no cover
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def next_notification(self) -> dict:  # pragma: no cover
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _get_status(handles: list[dict], handle_id: str) -> str:
    for h in handles:
        if h["handle_id"] == handle_id:
            return str(h["status"])
    raise AssertionError(f"handle_id not found: {handle_id}")


class _ClarifyingHandle(SteerableToolHandle):
    """A handle that emits a clarification and blocks `result()` until answered."""

    def __init__(self):
        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notif_q: asyncio.Queue[dict] = asyncio.Queue()
        self._answered = asyncio.Event()
        self._answer: str | None = None
        self._emitted = False

    async def ask(self, question: str, **kwargs):  # noqa: ARG002
        return self

    def interject(self, message: str, **kwargs):  # noqa: ARG002
        return None

    def stop(self, reason: str | None = None, **kwargs):  # noqa: ARG002
        return None

    async def pause(self) -> str | None:
        return "paused"

    async def resume(self) -> str | None:
        return "resumed"

    def done(self) -> bool:
        return self._answered.is_set()

    async def next_clarification(self) -> dict:
        return await self._clar_q.get()

    async def next_notification(self) -> dict:
        return await self._notif_q.get()

    async def answer_clarification(
        self,
        call_id: str,
        answer: str,
    ) -> None:  # noqa: ARG002
        self._answer = answer
        self._answered.set()

    async def result(self) -> str:
        if not self._emitted:
            self._emitted = True
            await self._clar_q.put(
                {
                    "type": "clarification",
                    "call_id": "C1",
                    "tool_name": "primitives.contacts.ask",
                    "question": "Which contact should I use?",
                },
            )
        await self._answered.wait()
        return f"ok:{self._answer}"


class _GateHandle(SteerableToolHandle):
    """A handle that blocks `result()` until released, and records interjections."""

    def __init__(self, *, result_value: str) -> None:
        self._clar_q: asyncio.Queue[dict] = asyncio.Queue()
        self._notif_q: asyncio.Queue[dict] = asyncio.Queue()
        self._released = asyncio.Event()
        self._result_value = result_value
        self.interject_calls: list[str] = []

    async def ask(self, question: str, **kwargs):  # noqa: ARG002
        return self

    def interject(self, message: str, **kwargs):  # noqa: ARG002
        self.interject_calls.append(message)
        return None

    def stop(self, reason: str | None = None, **kwargs):  # noqa: ARG002
        self._released.set()
        return None

    async def pause(self) -> str | None:
        return "paused"

    async def resume(self) -> str | None:
        return "resumed"

    def done(self) -> bool:
        return self._released.is_set()

    async def next_clarification(self) -> dict:
        return await self._clar_q.get()

    async def next_notification(self) -> dict:
        return await self._notif_q.get()

    async def answer_clarification(
        self,
        call_id: str,
        answer: str,
    ) -> None:  # noqa: ARG002
        return None

    def release(self) -> None:
        self._released.set()

    async def result(self) -> str:
        await self._released.wait()
        return self._result_value


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_registers_handle_and_emits_events() -> None:
    """Registers a handle and verifies clarification/notification events are logged and indexed."""
    pane = SteerableToolPane(run_id="run_test")
    handle = _MockHandle()

    await pane.register_handle(
        handle=handle,  # type: ignore[arg-type]
        handle_id="H1",
        parent_handle_id=None,
        origin_tool="primitives.contacts.ask",
        origin_step=12,
        environment_namespace="primitives",
        capabilities=["interjectable"],
        call_stack=None,
    )

    # Registration should emit an event and enqueue a wakeup token.
    async def _has_one_event() -> bool:
        return len(pane._events_log) >= 1  # type: ignore[attr-defined]

    await _wait_for_condition(_has_one_event, poll=0.01, timeout=5)
    ev0 = pane._events_log[0]  # type: ignore[attr-defined]
    assert ev0["type"] == "handle_registered"
    assert ev0["handle_id"] == "H1"

    # Push a clarification and verify it becomes pending + logged.
    await handle._clar_q.put(
        {
            "type": "clarification",
            "call_id": "C1",
            "tool_name": "primitives.contacts.ask",
            "question": "Which one?",
        },
    )

    async def _has_clarification_event() -> bool:
        return any(e.get("type") == "clarification" for e in pane._events_log)  # type: ignore[attr-defined]

    await _wait_for_condition(_has_clarification_event, poll=0.01, timeout=5)

    pending = await pane.get_pending_clarifications()
    assert len(pending) == 1
    assert pending[0]["payload"]["call_id"] == "C1"

    # Push a notification and verify it is logged.
    await handle._notif_q.put(
        {
            "type": "notification",
            "call_id": "N1",
            "tool_name": "primitives.contacts.ask",
            "message": "Searching...",
        },
    )

    async def _has_searching_notification() -> bool:
        return any(
            e.get("type") == "notification"
            and (e.get("payload") or {}).get("message") == "Searching..."
            for e in pane._events_log  # type: ignore[attr-defined]
        )

    await _wait_for_condition(_has_searching_notification, poll=0.01, timeout=5)

    # Cleanup should cancel watchers and become idempotent.
    await pane.cleanup()
    await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_steering_targeted_and_broadcast_and_answer_clarification() -> None:
    """Exercises targeted/broadcast steering APIs and `answer_clarification` pending-index behavior."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()
    h2 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=1,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h2,  # type: ignore[arg-type]
            handle_id="H2",
            parent_handle_id=None,
            origin_tool="primitives.transcripts.ask",
            origin_step=2,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_two_events() -> bool:
            return len(pane._events_log) >= 2  # type: ignore[attr-defined]

        await _wait_for_condition(_has_two_events, poll=0.01, timeout=5)

        # Broadcast interject only to contacts.* handle.
        out = await pane.broadcast_interject(
            "hello",
            origin_tool_prefixes=["primitives.contacts"],
        )
        assert out["targets"] == ["H1"]
        assert len(h1.interject_calls) == 1
        assert len(h2.interject_calls) == 0

        # Produce a clarification for H1, then answer it.
        await h1._clar_q.put(
            {
                "type": "clarification",
                "call_id": "C1",
                "tool_name": "primitives.contacts.ask",
                "question": "Which one?",
            },
        )

        async def _has_one_pending_clarification() -> bool:
            pending = await pane.get_pending_clarifications()
            return len(pending) == 1

        await _wait_for_condition(_has_one_pending_clarification, poll=0.01, timeout=5)

        await pane.answer_clarification("H1", "C1", "David Smith")
        assert h1.answer_calls == [("C1", "David Smith")]
        pending2 = await pane.get_pending_clarifications()
        assert pending2 == []

        # Pause/resume/stop are safe and return underlying results.
        assert await pane.pause("H1") == "paused"
        assert await pane.resume("H1") == "resumed"
        assert await pane.stop("H1", "done") == "stopped"

        # After stop, targeted interject is a safe no-op (but recorded).
        await pane.interject("H1", "later")
        assert len(h1.interject_calls) == 1  # unchanged
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_tracks_status_transitions() -> None:
    """Verify pane tracks running/paused/waiting_for_clarification/running/stopped."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=10,
            environment_namespace="primitives",
            capabilities=["interjectable", "pausable"],
            call_stack=None,
        )

        async def _has_one_event() -> bool:
            return len(pane._events_log) >= 1  # type: ignore[attr-defined]

        await _wait_for_condition(_has_one_event, poll=0.01, timeout=5)

        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "running"

        await pane.pause("H1")
        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "paused"

        await pane.resume("H1")
        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "running"

        # Emit a clarification and wait until pane indexes it.
        await h1._clar_q.put(
            {
                "type": "clarification",
                "call_id": "C1",
                "tool_name": "primitives.contacts.ask",
                "question": "Which?",
            },
        )

        async def _status_waiting_for_clarification() -> bool:
            return (
                _get_status(await pane.list_handles(), "H1")
                == "waiting_for_clarification"
            )

        await _wait_for_condition(
            _status_waiting_for_clarification,
            poll=0.01,
            timeout=5,
        )

        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "waiting_for_clarification"

        await pane.answer_clarification("H1", "C1", "answer")
        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "running"

        await pane.stop("H1", "done")
        handles = await pane.list_handles()
        assert _get_status(handles, "H1") == "stopped"
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_cancels_watchers_on_cleanup_and_ignores_late_registration() -> None:
    """Cancels per-handle watcher tasks on cleanup and ignores late registrations."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _BlockingHandle()
    h2 = _BlockingHandle()

    await pane.register_handle(
        handle=h1,  # type: ignore[arg-type]
        handle_id="H1",
        parent_handle_id=None,
        origin_tool="primitives.contacts.ask",
        origin_step=1,
        environment_namespace="primitives",
        capabilities=["interjectable"],
        call_stack=None,
    )
    await pane.register_handle(
        handle=h2,  # type: ignore[arg-type]
        handle_id="H2",
        parent_handle_id=None,
        origin_tool="primitives.tasks.execute",
        origin_step=2,
        environment_namespace="primitives",
        capabilities=["interjectable"],
        call_stack=None,
    )

    assert "H1" in pane._watcher_tasks  # type: ignore[attr-defined]
    assert "H2" in pane._watcher_tasks  # type: ignore[attr-defined]
    assert not pane._watcher_tasks["H1"].done()  # type: ignore[attr-defined]
    assert not pane._watcher_tasks["H2"].done()  # type: ignore[attr-defined]

    await pane.cleanup()

    assert pane._cleanup_started is True  # type: ignore[attr-defined]
    assert pane._watcher_tasks == {}  # type: ignore[attr-defined]

    # Late registrations are ignored.
    await pane.register_handle(
        handle=_MockHandle(),  # type: ignore[arg-type]
        handle_id="H3",
        parent_handle_id=None,
        origin_tool="primitives.knowledge.ask",
        origin_step=3,
        environment_namespace="primitives",
        capabilities=["interjectable"],
        call_stack=None,
    )
    assert "H3" not in pane._registry  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_broadcast_filters_by_origin_tool_prefix() -> None:
    """Broadcast interject respects `origin_tool_prefixes` filtering."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()
    h2 = _MockHandle()
    h3 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=1,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h2,  # type: ignore[arg-type]
            handle_id="H2",
            parent_handle_id=None,
            origin_tool="primitives.contacts.update",
            origin_step=2,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h3,  # type: ignore[arg-type]
            handle_id="H3",
            parent_handle_id=None,
            origin_tool="primitives.tasks.execute",
            origin_step=3,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_three_events() -> bool:
            return len(pane._events_log) >= 3  # type: ignore[attr-defined]

        await _wait_for_condition(_has_three_events, poll=0.01, timeout=5)

        out = await pane.broadcast_interject(
            "test message",
            origin_tool_prefixes=["primitives.contacts"],
        )
        assert set(out["targets"]) == {"H1", "H2"}
        assert out["count"] == 2
        assert len(h1.interject_calls) == 1
        assert len(h2.interject_calls) == 1
        assert len(h3.interject_calls) == 0
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_broadcast_filters_by_status() -> None:
    """Broadcast interject respects status filtering (e.g., only `running`)."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()
    h2 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=1,
            environment_namespace="primitives",
            capabilities=["interjectable", "pausable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h2,  # type: ignore[arg-type]
            handle_id="H2",
            parent_handle_id=None,
            origin_tool="primitives.tasks.execute",
            origin_step=2,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_two_events() -> bool:
            return len(pane._events_log) >= 2  # type: ignore[attr-defined]

        await _wait_for_condition(_has_two_events, poll=0.01, timeout=5)

        await pane.pause("H1")

        out = await pane.broadcast_interject(
            "test",
            filter=BroadcastFilter(statuses=["running"]),
        )
        assert out["targets"] == ["H2"]
        assert len(h1.interject_calls) == 0
        assert len(h2.interject_calls) == 1
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_broadcast_filters_by_capabilities() -> None:
    """Broadcast interject respects capability filtering (e.g., only `pausable`)."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()
    h2 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=1,
            environment_namespace="primitives",
            capabilities=["interjectable", "pausable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h2,  # type: ignore[arg-type]
            handle_id="H2",
            parent_handle_id=None,
            origin_tool="primitives.tasks.execute",
            origin_step=2,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_two_events() -> bool:
            return len(pane._events_log) >= 2  # type: ignore[attr-defined]

        await _wait_for_condition(_has_two_events, poll=0.01, timeout=5)

        out = await pane.broadcast_interject(
            "test",
            filter=BroadcastFilter(capabilities=["pausable"]),
        )
        assert out["targets"] == ["H1"]
        assert len(h1.interject_calls) == 1
        assert len(h2.interject_calls) == 0
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_broadcast_filters_by_step_range() -> None:
    """Broadcast interject respects origin step boundaries (created-after filtering)."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()
    h2 = _MockHandle()
    h3 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=5,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h2,  # type: ignore[arg-type]
            handle_id="H2",
            parent_handle_id=None,
            origin_tool="primitives.tasks.execute",
            origin_step=10,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )
        await pane.register_handle(
            handle=h3,  # type: ignore[arg-type]
            handle_id="H3",
            parent_handle_id=None,
            origin_tool="primitives.knowledge.ask",
            origin_step=15,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_three_events() -> bool:
            return len(pane._events_log) >= 3  # type: ignore[attr-defined]

        await _wait_for_condition(_has_three_events, poll=0.01, timeout=5)

        out = await pane.broadcast_interject(
            "test",
            filter=BroadcastFilter(created_after_step=7),
        )
        assert set(out["targets"]) == {"H2", "H3"}
        assert len(h1.interject_calls) == 0
        assert len(h2.interject_calls) == 1
        assert len(h3.interject_calls) == 1
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_steering_after_completion_is_noop_and_logged() -> None:
    """Steering a completed handle is a safe no-op and records `steering_applied` with status `no-op`."""
    pane = SteerableToolPane(run_id="run_test")
    h1 = _MockHandle()

    try:
        await pane.register_handle(
            handle=h1,  # type: ignore[arg-type]
            handle_id="H1",
            parent_handle_id=None,
            origin_tool="primitives.contacts.ask",
            origin_step=1,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

        async def _has_one_event() -> bool:
            return len(pane._events_log) >= 1  # type: ignore[attr-defined]

        await _wait_for_condition(_has_one_event, poll=0.01, timeout=5)

        # Mark as completed via internal cleanup helper (best-effort completion detection).
        await pane._cleanup_handle("H1", emit_completed=True)  # type: ignore[attr-defined]

        assert await pane.interject("H1", "too late") is None
        assert await pane.pause("H1") is None
        assert await pane.resume("H1") is None
        assert await pane.stop("H1", "done") is None

        assert h1.interject_calls == []
        assert h1.pause_calls == 0
        assert h1.resume_calls == 0
        assert h1.stop_calls == []

        noop_events = [
            e
            for e in pane._events_log  # type: ignore[attr-defined]
            if e["type"] == "steering_applied" and e["payload"].get("status") == "no-op"
        ]
        assert len(noop_events) >= 4
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_emits_overflow_marker_and_drops_notifications_after_cap() -> None:
    """Notification overflow emits a one-time `pane_overflow` marker and drops subsequent notifications."""
    pane = SteerableToolPane(run_id="run_test")
    pane.MAX_EVENTS = 10

    # Fill to cap-1 with non-notification events so the marker behavior is triggered only
    # on the next notification.
    for i in range(9):
        await pane._emit_event(  # type: ignore[attr-defined]
            {
                "type": "completed",
                "handle_id": f"H{i}",
                "origin": {
                    "origin_tool": "test",
                    "origin_step": i,
                    "environment_namespace": "test",
                },
                "payload": {},
            },
        )

    # Next notification should reserve final slot for pane_overflow.
    await pane._emit_event(  # type: ignore[attr-defined]
        {
            "type": "notification",
            "handle_id": "H9",
            "origin": {
                "origin_tool": "test",
                "origin_step": 9,
                "environment_namespace": "test",
            },
            "payload": {"message": "notif 9"},
        },
    )

    assert len(pane._events_log) == 10  # type: ignore[attr-defined]
    assert pane._events_log[-1]["type"] == "pane_overflow"  # type: ignore[attr-defined]
    assert pane._overflow_occurred is True  # type: ignore[attr-defined]

    # Subsequent notifications should be dropped (None return) and not extend log.
    out = await pane._emit_event(  # type: ignore[attr-defined]
        {
            "type": "notification",
            "handle_id": "H10",
            "origin": {
                "origin_tool": "test",
                "origin_step": 10,
                "environment_namespace": "test",
            },
            "payload": {"message": "notif 10"},
        },
    )
    assert out is None
    assert len(pane._events_log) == 10  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pane_fails_fast_on_critical_event_overflow() -> None:
    """Critical events fail-fast with RuntimeError if the durable log is at capacity."""
    pane = SteerableToolPane(run_id="run_test")
    pane.MAX_EVENTS = 3

    # Fill to cap with non-critical events.
    await pane._emit_event(  # type: ignore[attr-defined]
        {
            "type": "completed",
            "handle_id": "H0",
            "origin": {
                "origin_tool": "test",
                "origin_step": 0,
                "environment_namespace": "test",
            },
            "payload": {},
        },
    )
    await pane._emit_event(  # type: ignore[attr-defined]
        {
            "type": "completed",
            "handle_id": "H1",
            "origin": {
                "origin_tool": "test",
                "origin_step": 1,
                "environment_namespace": "test",
            },
            "payload": {},
        },
    )
    await pane._emit_event(  # type: ignore[attr-defined]
        {
            "type": "completed",
            "handle_id": "H2",
            "origin": {
                "origin_tool": "test",
                "origin_step": 2,
                "environment_namespace": "test",
            },
            "payload": {},
        },
    )
    assert len(pane._events_log) == 3  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="overflow"):
        await pane._emit_event(  # type: ignore[attr-defined]
            {
                "type": "clarification",
                "handle_id": "H_crit",
                "origin": {
                    "origin_tool": "test",
                    "origin_step": 100,
                    "environment_namespace": "test",
                },
                "payload": {"call_id": "C1", "question": "test"},
            },
        )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_concurrent_event_emission_is_safe() -> None:
    """Concurrent `_emit_event` calls do not corrupt the durable event log."""
    pane = SteerableToolPane(run_id="run_test")

    async def emit_many(prefix: str, count: int) -> None:
        for i in range(count):
            await pane._emit_event(  # type: ignore[attr-defined]
                {
                    "type": "notification",
                    "handle_id": f"{prefix}_{i}",
                    "origin": {
                        "origin_tool": "test",
                        "origin_step": i,
                        "environment_namespace": "test",
                    },
                    "payload": {"message": f"{prefix} {i}"},
                },
            )

    try:
        await asyncio.gather(
            asyncio.create_task(emit_many("A", 50)),
            asyncio.create_task(emit_many("B", 50)),
            asyncio.create_task(emit_many("C", 50)),
        )
        assert len(pane._events_log) == 150  # type: ignore[attr-defined]
    finally:
        await pane.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_concurrent_registration_is_safe() -> None:
    """Concurrent handle registrations do not corrupt registry or watcher task map."""
    pane = SteerableToolPane(run_id="run_test")

    async def register_one(i: int) -> None:
        await pane.register_handle(
            handle=_BlockingHandle(),  # type: ignore[arg-type]
            handle_id=f"H{i}",
            parent_handle_id=None,
            origin_tool=f"primitives.test.{i}",
            origin_step=i,
            environment_namespace="primitives",
            capabilities=["interjectable"],
            call_stack=None,
        )

    try:
        await asyncio.gather(*[asyncio.create_task(register_one(i)) for i in range(20)])
        assert len(pane._registry) == 20  # type: ignore[attr-defined]
        assert len(pane._watcher_tasks) == 20  # type: ignore[attr-defined]
        for t in pane._watcher_tasks.values():  # type: ignore[attr-defined]
            assert not t.done()
    finally:
        await pane.cleanup()


# ──────────────────────────────────────────────────────────────────────────────
# Integrations (Actor + Pane)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_clarification_bubbles_up_and_is_answered() -> None:
    """Integration: clarification from nested manager bubbles to actor, is answered, and run completes."""
    primitives = Primitives()
    _ = primitives.contacts
    primitives.contacts.ask = AsyncMock(return_value=_ClarifyingHandle())

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=MagicMock(search_functions=MagicMock(return_value=[])),
        environments=[StateManagerEnvironment(primitives)],
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    task = HierarchicalActorHandle(
        actor=actor,
        goal="Test pane supervisor clarification flow",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
        persist=False,
    )

    try:
        # Cancel auto-started task and inject deterministic plan.
        if task._execution_task:
            task._execution_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task._execution_task

        task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                async def main_plan():
                    h = await primitives.contacts.ask("find contact")
                    res = await h.result()
                    return res
                """,
            ),
            task,
        )
        task._execution_task = asyncio.create_task(task._initialize_and_run())

        question = await asyncio.wait_for(up_q.get(), timeout=10)
        assert "Which contact" in question

        await down_q.put("David Smith")
        final = await asyncio.wait_for(task.result(), timeout=20)
        assert "ok:David Smith" in str(final)

        handles = await task.pane.list_handles()
        assert any(h["origin_tool"] == "primitives.contacts.ask" for h in handles)
        assert task.pane.run_id == str(task.run_id)
        assert await task.pane.get_pending_clarifications() == []
    finally:
        if not task.done():
            await task.stop(final_result="done")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_broadcast_interject_reaches_concurrent_handles() -> None:
    """Integration: broadcast interject hits multiple in-flight handles."""
    primitives = Primitives()
    _ = primitives.contacts
    _ = primitives.tasks

    h1 = _GateHandle(result_value="ok:contacts.ask")
    h2 = _GateHandle(result_value="ok:contacts.update")
    h3 = _GateHandle(result_value="ok:tasks.execute")

    primitives.contacts.ask = AsyncMock(return_value=h1)
    primitives.contacts.update = AsyncMock(return_value=h2)
    primitives.tasks.execute = AsyncMock(return_value=h3)

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=MagicMock(search_functions=MagicMock(return_value=[])),
        environments=[StateManagerEnvironment(primitives)],
    )

    task = HierarchicalActorHandle(
        actor=actor,
        goal="Test pane broadcast interject filtering",
        persist=False,
    )

    try:
        if task._execution_task:
            task._execution_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task._execution_task

        task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                import asyncio

                async def main_plan():
                    h1 = await primitives.contacts.ask("a")
                    h2 = await primitives.contacts.update("b")
                    h3 = await primitives.tasks.execute("c")
                    return await asyncio.gather(h1.result(), h2.result(), h3.result())
                """,
            ),
            task,
        )
        task._execution_task = asyncio.create_task(task._initialize_and_run())

        async def _has_three_handles() -> bool:
            return len(await task.pane.list_handles()) >= 3

        await _wait_for_condition(_has_three_handles, poll=0.01, timeout=10)

        result = await task.pane.broadcast_interject(
            "User clarified: his name is David Smith",
            origin_tool_prefixes=["primitives.contacts"],
        )

        assert result["count"] == 2
        assert len(h1.interject_calls) == 1
        assert len(h2.interject_calls) == 1
        assert len(h3.interject_calls) == 0

        # Unblock handles so the plan can complete cleanly.
        h1.release()
        h2.release()
        h3.release()

        final = await asyncio.wait_for(task.result(), timeout=30)
        assert "ok:contacts.ask" in str(final)
    finally:
        if not task.done():
            await task.stop(final_result="done")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_stop_cancels_watchers_without_deadlock() -> None:
    """Integration: stopping Actor handle cleanly cancels pane watchers."""
    primitives = Primitives()
    _ = primitives.contacts

    h1 = _GateHandle(result_value="ok")
    primitives.contacts.ask = AsyncMock(return_value=h1)

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=MagicMock(search_functions=MagicMock(return_value=[])),
        environments=[StateManagerEnvironment(primitives)],
    )

    task = HierarchicalActorHandle(
        actor=actor,
        goal="Test stop cleanup",
        persist=False,
    )

    try:
        if task._execution_task:
            task._execution_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task._execution_task

        task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                async def main_plan():
                    h = await primitives.contacts.ask("x")
                    return await h.result()
                """,
            ),
            task,
        )
        task._execution_task = asyncio.create_task(task._initialize_and_run())

        async def _has_one_handle() -> bool:
            return len(await task.pane.list_handles()) >= 1

        await _wait_for_condition(_has_one_handle, poll=0.01, timeout=10)
        assert len(task.pane._watcher_tasks) >= 1  # type: ignore[attr-defined]

        await task.stop(final_result="done")
        assert task.pane._cleanup_started is True  # type: ignore[attr-defined]
        assert task.pane._watcher_tasks == {}  # type: ignore[attr-defined]
    finally:
        if not task.done():
            await task.stop(final_result="done")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_verification_captures_pane_events() -> None:
    """Integration: verification work items capture pane events."""
    primitives = Primitives()
    _ = primitives.contacts

    h1 = _GateHandle(result_value="ok")
    primitives.contacts.ask = AsyncMock(return_value=h1)

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=MagicMock(search_functions=MagicMock(return_value=[])),
        environments=[StateManagerEnvironment(primitives)],
    )

    task = HierarchicalActorHandle(
        actor=actor,
        goal="Test verification pane capture",
        persist=False,
    )

    captured: list = []

    try:
        if task._execution_task:
            task._execution_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task._execution_task

        # Intercept verification spawning so this stays deterministic (no LLM).
        def _capture_only(item):  # type: ignore[no-untyped-def]
            captured.append(item)

        task._spawn_async_verification = _capture_only  # type: ignore[method-assign]

        task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                @verify
                async def search_contacts():
                    h = await primitives.contacts.ask("find contact")
                    return await h.result()

                async def main_plan():
                    return await search_contacts()
                """,
            ),
            task,
        )
        task._execution_task = asyncio.create_task(task._initialize_and_run())

        async def _has_one_handle() -> bool:
            return len(await task.pane.list_handles()) >= 1

        await _wait_for_condition(_has_one_handle, poll=0.01, timeout=10)
        h1.release()

        final = await asyncio.wait_for(task.result(), timeout=20)
        assert "ok" in str(final)

        assert captured, "Expected at least one captured VerificationWorkItem"
        item = captured[0]
        assert getattr(item, "pane_events", None) is not None
        assert any(e.get("type") == "handle_registered" for e in item.pane_events)
    finally:
        if not task.done():
            await task.stop(final_result="done")


# ──────────────────────────────────────────────────────────────────────────────
# Real manager integrations (requires Unify)
# ──────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def actor_with_mocked_browser():
    """Create a real HierarchicalActor with ONLY StateManagerEnvironment (no browser env)."""
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    # Mock specific browser methods that the test expects
    if actor.computer_primitives is not None:
        actor.computer_primitives.navigate = AsyncMock(return_value=None)
        actor.computer_primitives.observe = AsyncMock(return_value={})
        actor.computer_primitives.act = AsyncMock(return_value="mock action complete")

    try:
        yield actor
    finally:
        with contextlib.suppress(Exception):
            await actor.close()


# ---------------------------------------------------------------------------
# Scenario 1: Cross-manager join (ask): Knowledge → Contacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_pane_registers_handles_for_cross_manager_join():
    """
    Ask a question that requires joining:
    - ContactManager custom column (employer)
    - KnowledgeManager table (Companies)

    Validates:
    - Pane registers at least one in-flight handle from primitives tools
    - Pane captures handle_registered events
    - No pending clarifications at completion
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    cm = primitives.contacts
    km = primitives.knowledge

    seed_first = "SteveTest"
    seed_last = "Joinson"
    seed_company = "BigCorpPane"

    # Put the join key in bio/rolling_summary so the manager's first-step semantic search can find it.
    cm._create_contact(
        first_name=seed_first,
        surname=seed_last,
        bio=f"{seed_first} {seed_last} works at {seed_company}.",
        rolling_summary=f"Employer: {seed_company}",
    )
    km._create_table(name="Companies", columns={"name": "str", "employees": "int"})
    km._add_rows(table="Companies", rows=[{"name": seed_company, "employees": 1200}])

    h = HierarchicalActorHandle(actor=actor, goal="canned", persist=False)
    # Cancel auto-started execution BEFORE injecting canned plan
    if h._execution_task:
        h._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await h._execution_task

    CANNED_PLAN = """
async def main_plan():
    # Deterministic plan: use primitives-only tools (no browser)
    c_handle = await primitives.contacts.ask("What company does SteveTest Joinson work for?")
    company = await c_handle.result()
    k_handle = await primitives.knowledge.ask(f"How many employees does {company} have?")
    answer = await k_handle.result()
    return str(answer)
"""
    h.plan_source_code = actor._sanitize_code(CANNED_PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    try:
        result = await asyncio.wait_for(h.result(), timeout=180.0)
        assert "1200" in str(result).replace(",", "")

        pane = h.pane
        handles = await pane.list_handles()
        pending = await pane.get_pending_clarifications()
        events = pane.get_recent_events(n=200)

        assert len(handles) > 0, "Pane should have registered at least one handle"
        assert len(pending) == 0, "No pending clarifications expected at completion"
        assert any(
            e.get("type") == "handle_registered" for e in events
        ), "Expected at least one handle_registered event"
    finally:
        with contextlib.suppress(Exception):
            await h.stop("test cleanup")
        with contextlib.suppress(Exception):
            await actor.close()


# ---------------------------------------------------------------------------
# Scenario 2: Cross-manager mutation + verification capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_verification_captures_pane_events_for_contact_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Trigger a contact mutation via a deterministic plan.

    Validates:
    - ContactManager state is mutated
    - VerificationWorkItem.pane_events is populated (at least once)
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    cm = primitives.contacts
    _ = (
        primitives.knowledge
    )  # ensure KM contexts exist for parity (real stack behavior)

    seed_email = "jane_pane_test@example.com"
    cm._create_contact(first_name="JaneTest", surname="Doe", email_address=seed_email)

    seen_items: list[object] = []
    original_spawn = HierarchicalActorHandle._spawn_async_verification

    def _spy_spawn(self: HierarchicalActorHandle, item):  # type: ignore[no-untyped-def]
        seen_items.append(item)
        return original_spawn(self, item)

    monkeypatch.setattr(
        HierarchicalActorHandle,
        "_spawn_async_verification",
        _spy_spawn,
        raising=True,
    )

    h = HierarchicalActorHandle(actor=actor, goal="canned", persist=False)
    if h._execution_task:
        h._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await h._execution_task

    CANNED_PLAN = """
async def main_plan():
    # Deterministic plan: update via primitives, then confirm via ask
    u = await primitives.contacts.update("Add JaneTest Doe's phone number +15559998877.")
    await u.result()
    q = await primitives.contacts.ask("What is JaneTest Doe's phone number?")
    return await q.result()
"""
    h.plan_source_code = actor._sanitize_code(CANNED_PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    try:
        _ = await asyncio.wait_for(h.result(), timeout=180.0)

        updated = cm.filter_contacts(filter=f"email_address == '{seed_email}'")[
            "contacts"
        ]
        assert updated, "JaneTest Doe should exist after update"
        assert getattr(updated[0], "phone_number", None) == "+15559998877"

        pane_event_sizes = [
            len(getattr(it, "pane_events", []) or []) for it in seen_items
        ]
        assert seen_items, "Expected at least one verification work item to be spawned"
        assert any(
            sz > 0 for sz in pane_event_sizes
        ), "Expected at least one VerificationWorkItem with non-empty pane_events"
    finally:
        with contextlib.suppress(Exception):
            await h.stop("test cleanup")
        with contextlib.suppress(Exception):
            await actor.close()


# ---------------------------------------------------------------------------
# Scenario 3: Concurrent handles + broadcast interject (Contacts + Transcripts)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_concurrent_handles_broadcast_interject_fans_out():
    """
    Ask for two independent answers by keeping two handles open before awaiting results,
    then broadcast an interjection to all in-flight handles.

    Validates:
    - Pane registers multiple in-flight handles
    - broadcast_interject reaches >=2 handles (fan-out)
    - steering_applied events are recorded
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    cm = primitives.contacts
    tm = primitives.transcripts

    seed_first = "EveTest"
    seed_last = "Paneson"
    seed_email = "eve_test_pane@example.com"

    created = cm._create_contact(
        first_name=seed_first,
        surname=seed_last,
        email_address=seed_email,
        bio=f"{seed_first} {seed_last} (email: {seed_email})",
        rolling_summary=f"{seed_first} {seed_last} — email {seed_email}",
    )
    eve_id = int(created["details"]["contact_id"])

    tm.log_messages(
        Message(
            medium="sms_message",
            sender_id=eve_id,
            receiver_ids=[eve_id],
            timestamp=datetime.now(UTC),
            content="Office hours policy: 9-5 PT.",
            exchange_id=123,
        ),
    )
    tm.join_published()

    async def _eve_searchable() -> bool:
        try:
            res = cm._search_contacts(
                references={"bio": f"{seed_first} {seed_last}"},
                k=5,
            )
            contacts = res.get("contacts") or []
            return any(
                getattr(c, "email_address", None) == seed_email for c in contacts
            )
        except Exception:
            return False

    async def _office_hours_searchable() -> bool:
        try:
            res = tm._search_messages(
                references={"content": "Office hours 9-5 PT"},
                k=5,
            )
            msgs = res.get("messages") or []
            return any("9-5" in str(getattr(m, "content", "")) for m in msgs)
        except Exception:
            return False

    await asyncio.wait_for(
        _wait_for_condition(_eve_searchable, poll=0.1, timeout=30.0),
        timeout=40.0,
    )
    await asyncio.wait_for(
        _wait_for_condition(_office_hours_searchable, poll=0.1, timeout=30.0),
        timeout=40.0,
    )

    h = HierarchicalActorHandle(actor=actor, goal="canned", persist=False)
    if h._execution_task:
        h._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await h._execution_task

    CANNED_PLAN = """
import asyncio

async def main_plan():
    # Create both handles before awaiting results.
    contact_handle = await primitives.contacts.ask("What is the email address for EveTest Paneson?")
    transcript_handle = await primitives.transcripts.ask("What are the office hours?")

    # Deterministic pause point for the test to broadcast interjection.
    await runtime.checkpoint("handles_ready")

    email_result, office_hours_result = await asyncio.gather(
        contact_handle.result(),
        transcript_handle.result(),
    )
    return f"{email_result} | {office_hours_result}"
"""
    h.plan_source_code = actor._sanitize_code(CANNED_PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    try:

        async def _checkpoint_reached() -> bool:
            return bool(h.runtime._checkpoint_event.is_set())

        await asyncio.wait_for(
            _wait_for_condition(_checkpoint_reached, poll=0.05, timeout=60.0),
            timeout=70.0,
        )

        async def _two_handles_registered() -> bool:
            return len(await h.pane.list_handles()) >= 2

        await asyncio.wait_for(
            _wait_for_condition(_two_handles_registered, poll=0.05, timeout=60.0),
            timeout=70.0,
        )

        pane = h.pane
        handles = await pane.list_handles()

        broadcast = await pane.broadcast_interject(
            "FYI: be concise.",
            filter=BroadcastFilter(
                origin_tool_prefixes=[
                    "primitives.contacts.",
                    "primitives.transcripts.",
                ],
                capabilities=["interjectable"],
            ),
        )

        result = await asyncio.wait_for(h.result(), timeout=180.0)

        events = pane.get_recent_events(n=500)
        steering_events = [e for e in events if e.get("type") == "steering_applied"]

        assert seed_email in str(result).lower()
        assert (
            "9-5" in str(result)
            or "9–5" in str(result)
            or "office hours" in str(result).lower()
        )
        assert len(handles) >= 2, "Expected >=2 handles registered in the pane"
        assert (
            int(broadcast.get("count") or 0) >= 2
        ), f"Expected fan-out, got: {broadcast}"
        assert len(steering_events) >= 2, "Expected >=2 steering_applied events"
    finally:
        with contextlib.suppress(Exception):
            await h.stop("test cleanup")
        with contextlib.suppress(Exception):
            await actor.close()


# ---------------------------------------------------------------------------
# Interjection routing to in-flight handles via SteerableToolPane
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_actor_interject_broadcast_routes_to_inflight_handles():
    """
    Validate that HierarchicalActorHandle.interject() can broadcast-route a user interjection
    to in-flight manager handles registered in the SteerableToolPane.
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    cm = primitives.contacts
    tm = primitives.transcripts

    seed_first = "RouteTest"
    seed_last = "Broadcast"
    seed_email = "route_test_broadcast@example.com"

    created = cm._create_contact(
        first_name=seed_first,
        surname=seed_last,
        email_address=seed_email,
        bio=f"{seed_first} {seed_last} (email: {seed_email})",
        rolling_summary=f"{seed_first} {seed_last} — email {seed_email}",
    )
    cid = int(created["details"]["contact_id"])
    tm.log_messages(
        Message(
            medium="sms_message",
            sender_id=cid,
            receiver_ids=[cid],
            timestamp=datetime.now(UTC),
            content="Office hours policy: 9-5 PT.",
            exchange_id=321,
        ),
    )
    tm.join_published()

    h = HierarchicalActorHandle(actor=actor, goal="canned", persist=False)
    if h._execution_task:
        h._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await h._execution_task

    CANNED_PLAN = f"""
import asyncio

async def main_plan():
    # Create both handles before awaiting results (ensures 2 in-flight handles).
    contact_handle = await primitives.contacts.ask("What is the email address for {seed_first} {seed_last}?")
    transcript_handle = await primitives.transcripts.ask("What are the office hours?")

    await runtime.checkpoint("handles_ready")

    email_result, office_hours_result = await asyncio.gather(
        contact_handle.result(),
        transcript_handle.result(),
    )
    return f"{{email_result}} | {{office_hours_result}}"
"""
    h.plan_source_code = actor._sanitize_code(CANNED_PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    try:

        async def _handles_ready() -> bool:
            if not h.runtime._checkpoint_event.is_set():
                return False
            return len(await h.pane.list_handles()) >= 2

        await asyncio.wait_for(
            _wait_for_condition(_handles_ready, poll=0.05, timeout=60.0),
            timeout=70.0,
        )

        msg = (
            "Please broadcast this instruction to all in-flight handles: "
            "be concise and do not ask clarifying questions."
        )
        status = await asyncio.wait_for(h.interject(msg), timeout=120.0)
        assert isinstance(status, str) and status

        async def _broadcast_applied() -> bool:
            events = h.pane.get_recent_events(n=500)
            steering = [
                e
                for e in events
                if e.get("type") == "steering_applied"
                and (e.get("payload") or {}).get("method") == "interject"
                and (e.get("payload") or {}).get("status") == "ok"
            ]
            return len(steering) >= 2

        await asyncio.wait_for(
            _wait_for_condition(_broadcast_applied, poll=0.05, timeout=60.0),
            timeout=70.0,
        )

        h.runtime._release_from_checkpoint()

        result = await asyncio.wait_for(h.result(), timeout=180.0)
        assert seed_email in str(result).lower()
        assert (
            "9-5" in str(result)
            or "9–5" in str(result)
            or "office hours" in str(result).lower()
        )
    finally:
        with contextlib.suppress(Exception):
            await h.stop("test cleanup")
        with contextlib.suppress(Exception):
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_actor_interject_targeted_routes_to_specific_handle():
    """
    Validate targeted routing: if the user provides a specific handle_id, the interjection
    should be routed to that handle.
    """
    primitives = Primitives()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        environments=[StateManagerEnvironment(primitives)],
    )

    cm = primitives.contacts
    tm = primitives.transcripts

    seed_first = "RouteTest"
    seed_last = "Targeted"
    seed_email = "route_test_targeted@example.com"
    created = cm._create_contact(
        first_name=seed_first,
        surname=seed_last,
        email_address=seed_email,
        bio=f"{seed_first} {seed_last} (email: {seed_email})",
        rolling_summary=f"{seed_first} {seed_last} — email {seed_email}",
    )
    cid = int(created["details"]["contact_id"])
    tm.log_messages(
        Message(
            medium="sms_message",
            sender_id=cid,
            receiver_ids=[cid],
            timestamp=datetime.now(UTC),
            content="Office hours policy: 9-5 PT.",
            exchange_id=654,
        ),
    )
    tm.join_published()

    h = HierarchicalActorHandle(actor=actor, goal="canned", persist=False)
    if h._execution_task:
        h._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await h._execution_task

    CANNED_PLAN = f"""
import asyncio

async def main_plan():
    contact_handle = await primitives.contacts.ask("What is the email address for {seed_first} {seed_last}?")
    transcript_handle = await primitives.transcripts.ask("What are the office hours?")
    await runtime.checkpoint("handles_ready")
    email_result, office_hours_result = await asyncio.gather(
        contact_handle.result(),
        transcript_handle.result(),
    )
    return f"{{email_result}} | {{office_hours_result}}"
"""
    h.plan_source_code = actor._sanitize_code(CANNED_PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    try:

        async def _have_ids() -> bool:
            if not h.runtime._checkpoint_event.is_set():
                return False
            return len(await h.pane.list_handles()) >= 2

        await asyncio.wait_for(
            _wait_for_condition(_have_ids, poll=0.05, timeout=60.0),
            timeout=70.0,
        )
        hs = await h.pane.list_handles()
        contact_h = next(
            (x for x in hs if "primitives.contacts" in (x.get("origin_tool") or "")),
            None,
        )
        assert contact_h is not None, f"Expected a contacts handle in pane, got: {hs}"
        hid = str(contact_h["handle_id"])

        msg = (
            f"Route this interjection ONLY to handle_id {hid}: "
            "please be concise in your response."
        )
        status = await asyncio.wait_for(h.interject(msg), timeout=120.0)
        assert isinstance(status, str) and status

        async def _targeted_applied() -> bool:
            events = h.pane.get_recent_events(n=500)
            steering = [
                e
                for e in events
                if e.get("type") == "steering_applied"
                and e.get("handle_id") == hid
                and (e.get("payload") or {}).get("method") == "interject"
            ]
            return len(steering) >= 1

        await asyncio.wait_for(
            _wait_for_condition(_targeted_applied, poll=0.05, timeout=60.0),
            timeout=70.0,
        )

        h.runtime._release_from_checkpoint()
        result = await asyncio.wait_for(h.result(), timeout=180.0)
        assert seed_email in str(result).lower()
    finally:
        with contextlib.suppress(Exception):
            await h.stop("test cleanup")
        with contextlib.suppress(Exception):
            await actor.close()
